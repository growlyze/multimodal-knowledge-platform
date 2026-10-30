"""Data loading and reshaping for tabular data.

Three tools:
    load_dataset() - pull a stored dataset through the Haystack data API
    load_file()    - read a CSV/Excel/gzip file from the sandbox filesystem
    resample()     - change the time grid of a datetime-indexed frame

Only the standard library, pandas and numpy are used here. The sandbox has no
general internet access, so HTTP calls go through urllib with a short timeout
and every failure degrades to an empty frame carrying a note on
df.attrs["load_issues"] - this module never raises on bad data.
"""

import bz2
import csv
import datetime
import gzip
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

from .logging_decorator import logged, logger

DEFAULT_API_URL = "http://haystack-api:8000"
API_TIMEOUT = 10
# /admin/data/table builds a whole page (up to 250K rows, ~30 MB of JSON)
# before sending its first byte, so a page can be merely slow rather than
# dead - TABLE_PAGE_TIMEOUT is sized for that, well inside the sandbox's
# overall per-command execution cap.
TABLE_PAGE_TIMEOUT = 90
PAGE_ATTEMPTS = 2           # one retry: a single slow page must not cost the rest

_TIME_CANDIDATES = [
    "time", "datetime", "date time", "date_time", "timestamp", "date",
    # German-style exports head the column "Datum"; it used to go
    # unrecognised, so the whole file loaded with no time index.
    "datum",
    "index",
]
# A separate date column and clock-time column (common export layout). Picking
# "time" alone parsed "18:00" with TODAY's date - a full-looking, wrong index.
_DATE_ONLY_NAMES = ("date", "datum", "day")
_CLOCK_ONLY_NAMES = ("time", "zeit", "hour", "clock")
_CLOCK_RE = r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
# d.m.Y / m/d/Y / d-m-Y at the start of a value. Year-first (ISO) never
# matches, which matters: dateutil's dayfirst=True would read 2026-03-04 as
# 3 April, so day-first is only ever applied to values this pattern owns.
_DMY_RE = r"^\s*(\d{1,2})([./-])(\d{1,2})\2(\d{2,4})(?:$|[\sT,])"
# Excel serial day numbers: 10000 is 1927-05-18, 2958465 is 9999-12-31. As
# UNIX seconds the same numbers are the first month of 1970 - never real
# data - so the two readings cannot collide.
_EXCEL_SERIAL_RANGE = (1e4, 2958466)
_EXCEL_ORIGIN = "1899-12-30"

# pandas 3 removed the old single-letter offset aliases; generated code still
# emits them, so translate rather than blow up.
_FREQ_ALIASES = {"H": "h", "T": "min", "S": "s", "L": "ms", "U": "us", "N": "ns", "M": "ME"}


def _normalize_freq(timeframe):
    """Translate legacy pandas offset aliases ("1H", "15T") to modern ones."""
    m = re.fullmatch(r"\s*(\d*)\s*([A-Za-z]+)\s*", str(timeframe))
    if not m:
        return timeframe
    count, unit = m.group(1), m.group(2)
    return f"{count}{_FREQ_ALIASES.get(unit, unit)}"


def _record_issue(df, message):
    """Append a human-readable data-quality note to df.attrs["load_issues"].

    Never raises. This is the one place a caller (or the summary, later) can
    learn what load_file() had to guess, drop or invent instead of silently
    receiving a frame that looks clean. logger.warning() already exists
    alongside most of these call sites, but that goes to server-side stdlib
    logging the model never sees - this is the channel that reaches it, via
    df.attrs (which pandas propagates through rename/drop/copy/assign, so it
    survives the rest of the pipeline; see _stamp_source, which only adds its
    own keys and must never clobber this one).
    """
    df.attrs.setdefault("load_issues", []).append(message)


def _find_time_column(df):
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in _TIME_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _numeric_epoch_unit(values):
    """Best-effort unit for a column of bare epoch numbers, by magnitude.

    A time column is sometimes UNIX seconds, sometimes milliseconds, with no
    marker either way. pandas has no way to guess this on its own:
    pd.to_datetime's default unit for a bare integer is NANOSECONDS since
    epoch, which silently reads a 2022-01-01 UNIX-SECONDS value as
    1970-01-01 00:00:01.64... - a fully "successful" parse (valid index,
    right row count, no warning) that is wrong by 52 years. Magnitude is the
    only signal available; these bands hold for any real timestamp for
    centuries either side of "now".
    """
    finite = values[np.isfinite(values)]
    if not len(finite):
        return "ns"
    magnitude = float(np.median(np.abs(finite)))
    if magnitude < 1e11:        # seconds: valid through year ~5138
        return "s"
    if magnitude < 1e14:        # milliseconds
        return "ms"
    if magnitude < 1e17:        # microseconds
        return "us"
    return "ns"                 # nanoseconds - pandas' own default, made explicit


_OFFSET_RE = re.compile(r"([+-]\d{2}):?(\d{2})\s*$")


def _file_zone(raw):
    """The fixed UTC offset every timestamp in the column states, or None.

    When a text timestamp column carries the SAME numeric offset on every
    row (e.g. +02:00 throughout), the file's own wall clock is that fixed
    offset; pandas parses such a column to UTC, which loses it. Mixed
    offsets (a daylight-saving pair, Z beside offsets, a mix with naive
    values, numbers) return None - the file states no single zone.
    """
    if pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_datetime64_any_dtype(raw):
        return None
    text = raw.dropna().astype(str)
    if not len(text):
        return None
    text = text.iloc[::max(1, len(text) // 5000)]
    found = text.str.extract(_OFFSET_RE)
    if found[0].isna().any():
        return None
    offsets = set(found[0] + found[1])
    if len(offsets) != 1:
        return None
    off = offsets.pop()
    sign = 1 if off[0] == "+" else -1
    return datetime.timezone(
        sign * datetime.timedelta(hours=int(off[1:3]), minutes=int(off[3:5])))


def _implausible_year_warning(df, parsed, time_col):
    """Flag a parsed timestamp column that lands outside any plausible range.

    A second, independent safety net beyond _numeric_epoch_unit: if some other
    unit surprise slips through, an obviously-wrong date range (a
    magnitude-detection bug, a corrupt export, a unit this heuristic did not
    anticipate) must be visible rather than silently accepted the way the
    1970 bug was. Recorded on load_issues too: the logger alone is
    server-side, so a mis-parse must never become a fact.
    """
    years = parsed.dropna().dt.year
    if len(years) and (years.min() < 1990 or years.max() > 2100):
        logger.warning(
            "time column %r parsed to an implausible range (%s to %s) - "
            "check the timestamp unit/format", time_col, parsed.min(), parsed.max())
        _record_issue(
            df, f"time column {time_col!r} parsed to an implausible date range "
            f"({parsed.min()} to {parsed.max()}); the timestamps were almost "
            "certainly not read correctly, so no date or time from this file "
            "can be trusted")


def _combine_split_date_time(df):
    """Merge a date-only column and a clock-only column into one "datetime".

    Returns df unchanged unless BOTH exist and every non-blank value of the
    clock column is a bare HH:MM[:SS] - a "Time" column holding full
    timestamps next to a "Date" column is left for _find_time_column.
    """
    lowered = {str(c).strip().lower(): c for c in df.columns}
    date_col = next((lowered[k] for k in _DATE_ONLY_NAMES if k in lowered), None)
    clock_col = next((lowered[k] for k in _CLOCK_ONLY_NAMES if k in lowered), None)
    if date_col is None or clock_col is None or date_col == clock_col:
        return df
    clock = df[clock_col].astype(str).str.strip()
    filled = df[clock_col].notna() & (clock != "")
    if not filled.any() or not clock[filled].str.fullmatch(_CLOCK_RE).all():
        return df
    dates = df[date_col]
    if pd.api.types.is_datetime64_any_dtype(dates):
        # Excel hands a date-only cell back as a midnight datetime; str() of
        # it would put a second clock time in front of ours.
        dates = dates.dt.strftime("%Y-%m-%d")
    out = df.drop(columns=[date_col, clock_col])
    out.insert(0, "datetime", dates.astype(str).str.strip() + " " + clock)
    return out


def _day_first_choice(df, values, time_col):
    """Decide day-first vs month-first for d/m/Y-shaped date strings.

    Returns True (day-first), False (month-first) or None (values are not
    d/m/Y-shaped; leave pandas' own parsing alone). pandas' default is
    month-first, which read 03.08.2026 (3 August) as 8 March - silently.
    Certain when either field exceeds 12 somewhere in the column; otherwise
    ambiguous, and the assumption is recorded on load_issues so it is never
    stated as fact.
    """
    parts = values.astype(str).str.extract(_DMY_RE)
    if not len(parts) or parts[0].isna().any():
        return None
    first, second = parts[0].astype(int), parts[2].astype(int)
    if (first > 12).any() and not (second > 12).any():
        return True
    if (second > 12).any() and not (first > 12).any():
        return False
    if (first > 12).any() and (second > 12).any():
        return None             # inconsistent file; let the NaT counter report it
    if (first == second).all():
        return False            # both readings give the same dates
    sample = values.iloc[0]
    if parts[1].iloc[0] == ".":
        # Dotted dates are the European convention; month-first dotted dates
        # are vanishingly rare in data exports.
        _record_issue(
            df, f"time column {time_col!r}: dates like {sample!r} fit both "
            "day-first and month-first; read as day-first (day.month.year, the "
            "usual meaning of dot-separated dates) - if the file is month-first, "
            "every date is wrong")
        return True
    # Slash/dash dates: prefer the reading that keeps the file compact in time
    # - a 3-day file read the wrong way round spans months.
    spans = {}
    for dayfirst in (False, True):
        parsed = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=dayfirst)
        spans[dayfirst] = parsed.max() - parsed.min()
    choice = spans[True] < spans[False]
    _record_issue(
        df, f"time column {time_col!r}: dates like {sample!r} fit both day-first "
        f"and month-first; read as {'day-first' if choice else 'month-first'} "
        "because that reading gives the shorter, more continuous date range - "
        "if that is wrong, every date is wrong")
    return choice


def _to_datetime_index(df):
    """Give the frame a sorted DatetimeIndex, from a time column if needed."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    df = _combine_split_date_time(df)
    time_col = _find_time_column(df)
    if time_col is None:
        logger.warning("no recognisable time column in %s; leaving index as-is",
                       list(df.columns)[:8])
        _record_issue(df, "no recognisable time column found; index left as-is")
        return df

    raw = df[time_col]
    non_blank = raw.notna() & (raw.astype(str).str.strip() != "")
    numeric = pd.to_numeric(raw, errors="coerce")
    finite = numeric[non_blank]
    if pd.api.types.is_datetime64_any_dtype(raw):
        # Already datetimes (read_excel on a real date cell). to_numeric turned
        # them into nanosecond integers, which the epoch branch below would
        # stamp UTC - a zone the file never stated.
        parsed = raw
    elif (non_blank.any() and finite.notna().all()
            and _EXCEL_SERIAL_RANGE[0] <= float(finite.median()) < _EXCEL_SERIAL_RANGE[1]):
        # Excel serial day numbers (46224.9166 = 2026-07-21 22:00). Read as
        # UNIX seconds they all land on 1970-01-01 - silently. Rounded to the
        # second: the stored fraction of a day is only accurate to ~1e-8,
        # which would otherwise leave 22:00:00.0000003.
        try:
            # The median can sit in the Excel range while one outlier does not
            # ([0, 10000, 2**63] - found by fuzzing): pandas then holds the
            # column as uint64 and the origin offset overflows, raising past
            # errors="coerce" just like the epoch branch below.
            parsed = pd.to_datetime(numeric, unit="D", origin=_EXCEL_ORIGIN,
                                    errors="coerce").dt.round("s")
        except (OverflowError, OutOfBoundsDatetime, ValueError) as exc:
            logger.warning("time column %r: Excel-range median but a value no "
                           "date fits (%s) - treating as unparseable", time_col, exc)
            _record_issue(
                df, f"time column {time_col!r} looks like Excel serial dates but "
                "holds a value no date fits and could not be parsed")
            return df
        _record_issue(
            df, f"time column {time_col!r} holds Excel serial day numbers "
            f"(e.g. {raw[non_blank].iloc[0]}); converted to dates - Excel stores "
            "no time zone, so the times are the file's own wall clock")
    elif non_blank.any() and finite.notna().all():
        # Every non-blank value is a bare number: this is an epoch timestamp,
        # not a calendar string. Route it through an explicit, magnitude-
        # derived unit instead of the generic string parser, which defaults
        # to nanoseconds for a bare number (see _numeric_epoch_unit).
        unit = _numeric_epoch_unit(numeric[non_blank].to_numpy(float))
        try:
            parsed = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
        except (OverflowError, OutOfBoundsDatetime, ValueError) as exc:
            # A number so large it cannot be represented as a Timestamp under
            # any of the units above raises straight out of the unit-based
            # converter instead of respecting errors="coerce" - the exact
            # class of "implausible epoch" this project already got burned by
            # once (the nanoseconds-by-default bug this branch exists to fix).
            logger.warning(
                "time column %r: magnitude implausible for any timestamp unit "
                "(detected %r, %s) - treating as unparseable", time_col, unit, exc)
            _record_issue(
                df, f"time column {time_col!r} had a magnitude no timestamp unit "
                "fits and could not be parsed")
            return df
    else:
        text = raw[non_blank].astype(str).str.strip()
        if len(text) and text.str.fullmatch(_CLOCK_RE).all():
            # pandas fills in TODAY's date for a bare clock time, producing an
            # index that looks complete and is wrong on every row.
            _record_issue(
                df, f"time column {time_col!r} holds clock times only and the "
                "file has no date column; the dates shown are today's date and "
                "do not come from the file")
        dayfirst = _day_first_choice(df, raw[non_blank], time_col)
        try:
            parsed = pd.to_datetime(raw, errors="coerce", format="mixed",
                                    dayfirst=bool(dayfirst))
        except ValueError as exc:
            # A column mixing a tz-aware string (trailing Z/offset) with naive
            # ones raises here instead of returning NaT for the bad rows -
            # contradicting load_file()'s own "this never raises" contract.
            # pandas' own error names the fix; take it rather than crash.
            logger.warning(
                "time column %r mixes timezone-aware and naive timestamps "
                "(%s); parsing with utc=True instead of raising", time_col, exc)
            parsed = pd.to_datetime(raw, errors="coerce", format="mixed", utc=True,
                                    dayfirst=bool(dayfirst))

    if parsed.isna().all():
        logger.warning("time column %r could not be parsed as datetime", time_col)
        _record_issue(df, f"time column {time_col!r} could not be parsed as datetime")
        return df
    if parsed.isna().any():
        logger.warning(
            "time column %r: %d of %d timestamp(s) could not be parsed and "
            "were dropped", time_col, int(parsed.isna().sum()), len(parsed))
        _record_issue(
            df, f"time column {time_col!r}: {int(parsed.isna().sum())} of "
            f"{len(parsed)} timestamp(s) could not be parsed and were dropped")
    _implausible_year_warning(df, parsed, time_col)

    out = df.drop(columns=[time_col]).copy()
    out.index = pd.DatetimeIndex(parsed)
    zone = _file_zone(raw)
    if zone:
        # Recorded, not applied: the index stays as parsed. describe_file()
        # shows times in this zone - summer/winter offsets together parse to
        # UTC, and the file's own wall clock is what its rows say.
        out.attrs["file_tz"] = zone
    out.index.name = "Time"
    out = out[out.index.notna()]
    # Both of these are silent repairs, and both change which row is "the next
    # row" for anything that walks by row position. Recorded BEFORE sort_index
    # for the ordering one, because afterwards there is nothing left to see.
    if not out.index.is_monotonic_increasing:
        # describe_file() reports file order; after sort_index it is gone.
        out.attrs["file_order_sorted"] = False
        _record_issue(
            out, "rows were not in chronological order and have been sorted; "
            "any column already computed against the original row order "
            "(a pre-computed column in the file) no longer lines up")
    duplicates = int(out.index.duplicated().sum())
    if duplicates:
        # Anything that walks row-by-row by position runs over these happily
        # and produces results whose timestamps cannot say WHICH of the
        # duplicate rows they meant.
        _record_issue(
            out, f"{duplicates} row(s) repeat a timestamp that another row "
            "already has, so those rows cannot be told apart by time")
    return out.sort_index()


def _stamp_source(df, path="", query=""):
    """Record where a frame came from on df.attrs["source"].

    A result carried no trace of the dataset behind it, so two datasets
    analysed one after another were indistinguishable from their outputs
    alone - there was no way to tell a current result from a stale one.
    rows/start/end/sha1 are what distinguish two files that share a schema
    and look alike in a preview.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    try:
        fingerprint = pd.util.hash_pandas_object(df, index=True).values
        digest = hashlib.sha1(fingerprint.tobytes()).hexdigest()[:12]
        index = df.index
        dated = isinstance(index, pd.DatetimeIndex) and len(index)
        df.attrs["source"] = {
            "path": str(path), "name": os.path.basename(str(path)) or str(query),
            "query": str(query), "rows": int(len(df)), "sha1": digest,
            "start": str(index[0]) if dated else "",
            "end": str(index[-1]) if dated else "",
        }
    except Exception as exc:        # provenance must never cost the data
        logger.warning("could not stamp provenance for %r (%s)", path, exc)
    return df


# ── stored datasets (Haystack data API) ────────────────────────────────

def _api_get(url, timeout=API_TIMEOUT):
    """GET a JSON document. Returns None on any failure - never raises."""
    try:
        request = urllib.request.Request(url)
        key = os.environ.get("HAYSTACK_API_KEY", "")
        if key:
            request.add_header("Authorization", "Bearer " + key)
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("data API unreachable (%s): %s: %s", url, type(exc).__name__, exc)
        return None


def _rows_to_frame(payload):
    """Pull a list-of-records out of the shapes the API might return."""
    if payload is None:
        return None
    if isinstance(payload, list):
        return pd.DataFrame(payload) if payload and isinstance(payload[0], dict) else None
    if isinstance(payload, dict):
        for key in ("rows", "data", "records", "results"):
            value = payload.get(key)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.DataFrame(value)
    return None


def _get_page(url):
    """One table page, retried once: (payload or None)."""
    for _ in range(PAGE_ATTEMPTS):
        page = _api_get(url, timeout=TABLE_PAGE_TIMEOUT)
        if page is not None:
            return page
    return None


def _fetch_table(base, table):
    """Every row of one stored table, page by page. Returns (df or None, shortfall).

    /admin/data/table serves at most 250K rows per call and says `truncated:
    true` when more exist. Reading the first page only and ignoring the flag
    gives a fraction of the rows with nothing said. `shortfall` is a sentence
    for load_issues when fewer rows came back than the catalog holds - a
    failed later page, or a count that disagrees - never silently.
    """
    url = f"{base}/admin/data/table/{urllib.parse.quote(str(table))}"
    first = _get_page(url)
    df = _rows_to_frame(first)
    if df is None and first is None:
        # Found in the catalog but unreadable right now. Said as such: falling
        # through to "no stored dataset found" told the user the upload did
        # not exist.
        return None, (f"stored table {table!r} exists but could not be read "
                      "(the data service did not answer) - no rows were loaded; "
                      "try again")
    if df is None or not isinstance(first, dict):
        return df, None
    frames, page, total = [df], first, first.get("total_rows")
    while page.get("truncated"):
        got = sum(len(f) for f in frames)
        page = _get_page(f"{url}?offset={got}")
        more = _rows_to_frame(page)
        if more is None or not len(more) or not isinstance(page, dict):
            break
        frames.append(more)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else df
    shortfall = None
    if isinstance(total, int) and len(df) < total:
        shortfall = (f"only {len(df)} of {total} rows of stored table {table!r} could be "
                     "loaded - results cover part of the data, not all of it")
        logger.warning(shortfall)
    return df, shortfall


def _exact_hit(hits, query):
    """The one search hit whose file name, file stem or table name IS the query."""
    q = str(query or "").strip().lower()
    if not q:
        return None
    found = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        fname = str(h.get("filename") or "").strip().lower()
        names = {fname, os.path.splitext(fname)[0],
                 str(h.get("table_name") or h.get("table") or "").strip().lower()}
        if q in names - {""}:
            found.append(h)
    return found[0] if len(found) == 1 else None


@logged
def load_dataset(name, api_url=None):
    """Load a stored dataset by name through the Haystack data API.

    Args:
        name: dataset name or search term (e.g. "sales_2025", "sensor dump").
        api_url: API base URL. Defaults to $HAYSTACK_API_URL, then
            http://haystack-api:8000.

    Behavior:
        1. GET {base}/admin/data/search?q={name}
        2. Load the full table for the match (an exact name match wins over
           plain search hits), page by page
        3. Set the time column as a DatetimeIndex when the table has one

    When $HAYSTACK_API_KEY is set it is sent as an
    "Authorization: Bearer ..." header. Returns an empty DataFrame with a
    note on df.attrs["load_issues"] when the dataset cannot be found or
    read - this never raises.
    """
    base = (api_url or os.environ.get("HAYSTACK_API_URL") or DEFAULT_API_URL).rstrip("/")
    search_url = f"{base}/admin/data/search?q={urllib.parse.quote(str(name))}"

    df, shortfall = None, None
    hits = _api_get(search_url)
    if isinstance(hits, dict):
        hits = hits.get("results") or hits.get("datasets") or hits.get("tables") or []
    match = hits[0] if isinstance(hits, list) and hits else None

    # An EXACT name - the file name, its stem, or the table name - is not a
    # search: an ambiguous search ("data" matching several tables) must not
    # dethrone the one table the caller named in full.
    exact = _exact_hit(hits, name) if isinstance(hits, list) else None
    if exact is not None:
        match, hits = exact, [exact]

    # Other matches are not silently discarded: still resolve to the first
    # hit (a search commonly matches several tables), but say what else
    # matched so no choice is made behind the caller's back.
    other_names = []
    if isinstance(hits, list) and len(hits) > 1:
        def _hit_name(h):
            if isinstance(h, dict):
                return h.get("table_name") or h.get("table") or h.get("name") or "?"
            return str(h)
        other_names = [_hit_name(h) for h in hits if h is not match]

    if isinstance(match, dict):
        df = _rows_to_frame(match)
        if df is None:
            table = match.get("table_name") or match.get("table") or match.get("name")
            if table:
                df, shortfall = _fetch_table(base, table)
    elif isinstance(match, str):
        df, shortfall = _fetch_table(base, match)

    if df is None:
        logger.warning("no dataset found for %r (API %s empty)", name, base)
        # Was logger.warning-only, server-side and invisible to the model -
        # indistinguishable from any other empty result. Recorded so a
        # genuinely missing dataset gets "I couldn't find it" instead of
        # more guessing.
        empty = pd.DataFrame()
        if shortfall:
            _record_issue(empty, shortfall)
        else:
            _record_issue(empty, f"no stored dataset found for {name!r}")
        return empty

    result = _to_datetime_index(df)
    result = _stamp_source(result, query=name)
    if other_names:
        _record_issue(
            result,
            f"query {name!r} matched multiple stored datasets - used "
            f"{match.get('table_name') or match.get('table') or match.get('name') if isinstance(match, dict) else match!r}, "
            f"others available: {', '.join(other_names)}",
        )
    if shortfall:
        _record_issue(result, shortfall)
    return result


# ── file loading ───────────────────────────────────────────────────────

_MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024   # sandbox tmpfs is 1g; stay under it


def _decompress_to_temp(filepath):
    """Decompress a .gz/.bz2 to a temp file. Returns (temp_path, error).

    A compressed data file is exactly one file, so it loads like the file
    inside it: "data.csv.gz" is "data.csv". The decompressed copy goes to the
    sandbox temp dir (the upload mount is read-only) and the caller deletes it
    as soon as the frame is in memory. Expansion is capped as bytes arrive, so
    a compression bomb cannot fill tmpfs before anyone notices.
    """
    gz = filepath.lower().endswith(".gz")
    opener = gzip.open if gz else bz2.open
    suffix = os.path.splitext(filepath[:-3] if gz else filepath[:-4])[1] or ".csv"
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="dt_load_")
    written = 0
    try:
        with os.fdopen(fd, "wb") as out, opener(filepath, "rb") as src:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_DECOMPRESSED_BYTES:
                    raise ValueError(
                        "decompressed size exceeds the %d MB limit"
                        % (_MAX_DECOMPRESSED_BYTES // (1024 * 1024)))
                out.write(chunk)
        if not written:
            raise ValueError("decompressed to zero bytes")
        return tmp, None
    except Exception as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None, "%s: %s" % (type(exc).__name__, exc)


@logged
def load_file(filepath):
    """Load a CSV/Excel file from the sandbox filesystem.

    Args:
        filepath: path to the file (usually /mnt/uploads/<name>.csv). A
            .gz/.bz2-compressed one works too: "data.csv.gz" loads exactly
            like "data.csv".

    Returns:
        DataFrame with a DatetimeIndex when the file has a recognisable time
        column. Empty DataFrame if the file is missing or unreadable - this
        never raises.

    Auto-detects CSV vs Excel by extension and parses the time column into a
    DatetimeIndex, understanding the common export layouts (split date/time
    columns, epoch seconds/milliseconds, Excel serial days, day-first dates,
    decimal commas, leading comment lines). Every guess, drop or repair is
    recorded on df.attrs["load_issues"]; the file's own raw header is kept in
    df.attrs["raw_columns"].
    """
    if not filepath or not os.path.exists(filepath):
        logger.warning("load_file: %r does not exist", filepath)
        # Returning a clean-looking empty frame with zero load_issues would be
        # indistinguishable from "the file genuinely had no rows". Route
        # through the same channel so a missing file degrades visibly instead
        # of silently, same contract as everything else in this module.
        empty = pd.DataFrame()
        _record_issue(empty, f"file not found: {filepath}")
        return empty

    ext = os.path.splitext(filepath)[1].lower()
    work, tmp = filepath, None
    if ext in (".gz", ".bz2"):
        # The compression wrapper is not the type: "data.csv.gz" is a CSV.
        work, err = _decompress_to_temp(filepath)
        if err is not None:
            logger.warning("load_file: could not decompress %s (%s)", filepath, err)
            empty = pd.DataFrame()
            _record_issue(empty, f"compressed file could not be decompressed: {err}")
            return empty
        tmp = work
        ext = os.path.splitext(filepath[:-3] if ext == ".gz" else filepath[:-4])[1].lower()

    layout = {"skip": 0, "sep": None, "decimal": ".", "header": None, "comments": []}
    try:
        if ext in (".xlsx", ".xls", ".xlsm"):
            raw = pd.read_excel(work)
        elif ext == ".parquet":
            raw = pd.read_parquet(work)
        else:  # .csv, .txt, .tsv and anything else worth a try
            layout = _sniff_text_layout(work)
            raw = pd.read_csv(work, sep=layout["sep"], engine="python",
                              skiprows=layout["skip"], decimal=layout["decimal"])
    except Exception as exc:  # an unreadable file must not kill the sandbox
        logger.warning("load_file: could not read %s (%s: %s)",
                       filepath, type(exc).__name__, exc)
        # Same hole as the missing-file branch above: a read failure must not
        # look identical to a clean empty file.
        empty = pd.DataFrame()
        _record_issue(empty, f"file could not be read: {type(exc).__name__}: {exc}")
        return empty
    finally:
        # The decompressed copy is scaffolding - it never outlives this call.
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    if raw is None or len(raw) == 0:
        # A header-only file read fine and has no rows - not a parse failure.
        empty = pd.DataFrame(columns=list(raw.columns) if raw is not None else [])
        _record_issue(empty, "file had no rows after loading")
        if layout["header"]:
            empty.attrs["raw_columns"] = layout["header"]
        return empty

    result = _stamp_source(_to_datetime_index(raw), filepath)
    # "columns in the file" means the raw header, time column included - after
    # _to_datetime_index the time column is the index, so the only faithful
    # copy is taken here. pandas also renames repeated header names
    # ("value" -> "value.1"); the csv module does not.
    result.attrs["raw_columns"] = layout["header"] or [str(c) for c in raw.columns]
    if layout["comments"]:
        result.attrs["file_comments"] = layout["comments"]
    return result


def _sniff_text_layout(filepath, sample_lines=50):
    """Leading comment lines, delimiter, decimal mark and raw header of a text table.

    pandas' own sniffer (sep=None) took a "# Exported by ..." line as the
    header, so every real column became NaN; and a ';'-separated file with
    decimal commas (24150,5) read every number as NaN. Both were silent.
    This decides all three from the first lines, explicitly.
    """
    with open(filepath, encoding="utf-8-sig", errors="replace") as f:
        lines = [line for line in (f.readline() for _ in range(sample_lines)) if line]
    skip, comments = 0, []
    while skip < len(lines) and (not lines[skip].strip()
                                 or lines[skip].lstrip().startswith(("#", "//"))):
        if lines[skip].strip():
            comments.append(lines[skip].strip())
        skip += 1
    layout = {"skip": skip, "sep": None, "decimal": ".", "header": None, "comments": comments}
    body = [line.rstrip("\r\n") for line in lines[skip:] if line.strip()]
    if not body:
        return layout
    head = body[0]
    sep = max(",;\t|", key=head.count)
    if not head.count(sep):
        return layout                       # whitespace or single column: let pandas sniff
    layout["sep"] = sep
    layout["header"] = [h.strip() for h in next(csv.reader([head], delimiter=sep))]
    if sep != ",":
        fields = [f.strip() for line in body[1:] for f in line.split(sep)]
        comma_dec = sum(bool(re.fullmatch(r"[+-]?\d+,\d+", f)) for f in fields)
        dot_dec = sum(bool(re.fullmatch(r"[+-]?\d*\.\d+", f)) for f in fields)
        if comma_dec and not dot_dec:
            layout["decimal"] = ","
    return layout


@logged
def resample(df, timeframe="1min", agg=None):
    """Resample a datetime-indexed DataFrame onto a different time grid.

    Args:
        df: DataFrame with a DatetimeIndex.
        timeframe: target grid ("5min", "15min", "1h", "4h", "1D", ...).
            Legacy pandas aliases like "1H" or "15T" are translated
            automatically.
        agg: optional per-column aggregation mapping (the pandas agg dict,
            e.g. {"value": "sum"}). By default open/high/low/close-shaped
            columns keep their first/max/min/last meaning, Volume sums, other
            numeric columns average, and anything else takes its first value.

    Bad input is returned unchanged - this never raises.
    """
    if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("resample: DatetimeIndex required, got %s",
                       type(getattr(df, "index", None)).__name__)
        return df

    if agg is None:
        how = {}
        for col in df.columns:
            if col in ("Open", "High", "Low", "Close", "Volume"):
                how[col] = {"Open": "first", "High": "max", "Low": "min",
                            "Close": "last", "Volume": "sum"}[col]
            elif pd.api.types.is_numeric_dtype(df[col]):
                how[col] = "mean"
            else:
                how[col] = "first"
    else:
        how = dict(agg)

    try:
        out = df.resample(_normalize_freq(timeframe)).agg(how)
    except (ValueError, TypeError) as exc:
        logger.warning("resample: bad timeframe %r (%s); returning input", timeframe, exc)
        return df
    return out.dropna(how="all")
