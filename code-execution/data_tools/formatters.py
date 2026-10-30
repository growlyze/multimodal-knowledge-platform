"""Number and table formatting: turn numeric outputs into prose.

Generic helpers only - no fixed report schema lives here:

  * format_number / format_percent / format_money / format_plain /
    format_signed / format_ratio render one number as text.
  * format_table_text renders a DataFrame as a readable table, truncated to a
    budget with the true total stated in the header.

Every formatter is total: None/NaN/inf never raise and never leak a raw
"nan" into prose - the caller gets a documented fallback instead.
"""

import math

import pandas as pd

from .logging_decorator import logged

# Decimals per currency code (M4). Small-value currencies need more decimals:
# a 0.00003 BTC amount renders as 0.00 under a 2-decimal format - a report of
# zeroes that reads as a zero result.
CURRENCY_DECIMALS = {"USD": 2, "USDT": 2, "USDC": 2, "EUR": 2, "GBP": 2,
                     "BTC": 8, "ETH": 6}


def format_number(value, default=0.0):
    """Coerce to float, mapping None/NaN/inf to a JSON- and text-safe number."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def format_signed(value):
    """A signed amount with no unit: 2 decimals, more only when 2 would show 0."""
    f = format_number(value)
    sign = "-" if f < 0 else "+"
    a = abs(f)
    if a == 0 or a >= 0.01:
        return f"{sign}{a:,.2f}"
    return sign + ("%.10f" % a).rstrip("0").rstrip(".")


def format_money(value, currency="USD"):
    """Signed amount with thousands separators: 1234.5 -> '+$1,234.50'.

    The USD branch keeps the dollar format. A named non-USD currency gets its
    own decimals and a trailing code (+0.00003000 BTC). An UNKNOWN currency
    (None/empty) gets no symbol at all - a "$" on a unit nobody established
    reads as a fact that was never stated.
    """
    f = format_number(value)
    sign = "-" if f < 0 else "+"
    if currency == "USD":
        return f"{sign}${abs(f):,.2f}"
    if not currency:
        return format_signed(f)
    dp = CURRENCY_DECIMALS.get(currency, 8)
    return f"{sign}{abs(f):,.{dp}f} {currency}"


def format_percent(value):
    """Percent to one decimal: 12.34 -> '12.3%'. Input is already a percent."""
    return f"{format_number(value):.1f}%"


def format_ratio(value):
    """A 2-decimal ratio, or N/A when there is no finite value to show.

    Infinity is not a ratio; a glyph in a numeric column reads as a result.
    N/A says the one true thing.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if not math.isfinite(f) else f"{f:.2f}"


def format_plain(value):
    """0.25 -> '0.25', 15.0 -> '15': no float residue, no padded zeros."""
    return ("%.10f" % float(value)).rstrip("0").rstrip(".")


def _cell(value):
    """One value as it prints in a table cell."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return format_plain(value) if math.isfinite(value) else "n/a"
    return str(value)


@logged
def format_table_text(frame, max_rows=20, title="Rows"):
    """Render a DataFrame as a readable table, truncated to max_rows rows.

    The header always states the true total, so a truncated view still tells
    the reader how many rows there really were. Numbers print without float
    residue; everything else prints as str().
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "No rows."

    total = len(frame)
    shown = frame.head(max(1, int(max_rows))).rename(columns=str)
    header = f"=== {title} ({total} total"
    header += f", showing first {len(shown)}) ===" if total > len(shown) else ") ==="
    columns = [str(c) for c in shown.columns]
    widths = {c: max([len(c)] + [len(v) for v in shown[c].map(_cell)]) for c in columns}
    lines = [header, "  ".join(c.ljust(widths[c]) for c in columns)]
    for _, row in shown.iterrows():
        lines.append("  ".join(_cell(row[c]).ljust(widths[c]) for c in columns))
    return "\n".join(lines)
