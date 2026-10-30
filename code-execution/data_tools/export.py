"""Write DataFrames to a styled .xlsx workbook.

What the workbook adds over `frame.to_csv(path)` is what a CSV cannot hold -
numbers that are numbers in Excel, dates Excel can sort and filter, a frozen
header row and readable column widths - plus several frames in one file as
named sheets. The rows and columns themselves are exactly what the frames
hold: a user who compares a CSV from one run with a workbook from another is
comparing like with like.

openpyxl is imported inside export_xlsx, not at module level: it is pinned in
the sandbox image (code-execution/requirements.txt), but importing data_tools
must not fail anywhere it happens to be missing - every other function here
works without it.
"""

import datetime as _dt
import math
import os

import numpy as np
import pandas as pd

from .logging_decorator import logged

# Numbers are stored as the exact float; these formats are display only, so a
# SUM in Excel equals the data's own total to the cent.
_INT_FORMAT = "#,##0"
_FLOAT_FORMAT = "#,##0.00####"   # 2 decimals, up to 6: a 0.00005 step displays without rounding
_TIME_FORMAT = "yyyy-mm-dd hh:mm:ss"


def _cell_value(value):
    """One pandas/numpy value as something openpyxl stores natively.

    * NaN/None/NaT/inf -> None (an empty cell). to_csv writes NaN as an empty
      field; openpyxl would otherwise write a float nan that Excel flags as a
      corrupt number, and inf has no Excel representation at all.
    * numpy scalars and float subclasses -> plain int/float: a value Excel
      receives as text sorts as text and sums to 0.
    * tz-aware Timestamps -> tz-naive wall-clock datetimes. openpyxl raises
      TypeError on any tz-aware datetime; the caller names the zone in the
      column header, so nothing the CSV showed is lost.
    """
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, _dt.datetime)):
        if pd.isna(value):
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            # tz_localize(None) keeps the local wall clock (09:30 stays 09:30);
            # tz_convert(None) would shift it to UTC and every time in the
            # sheet would disagree with the CSV the user has.
            ts = ts.tz_localize(None)
        return ts.to_pydatetime()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, pd.Timedelta):
        return None if pd.isna(value) else str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # lists/dicts/arrays: pd.isna is elementwise or undefined
    if isinstance(value, (str, np.str_)):
        return str(value)
    # Anything else (tuples, dicts) as its text - the same thing to_csv would
    # have written for it.
    return str(value)


def _column_tz(series):
    """The timezone name of a datetime column, or None when it has none."""
    tz = getattr(getattr(series, "dt", None), "tz", None) \
        if pd.api.types.is_datetime64_any_dtype(series) else None
    if tz is None:
        # Object columns of tz-aware Timestamps (a concatenated frame can end
        # up this way) still need their zone named.
        first = next((v for v in series if isinstance(v, pd.Timestamp)), None)
        tz = first.tzinfo if first is not None else None
    return None if tz is None else str(tz)


def _default_number_format(col, series):
    """Display format for a column: integers, floats, or nothing (text)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return _TIME_FORMAT
    if pd.api.types.is_integer_dtype(series):
        return _INT_FORMAT
    if pd.api.types.is_float_dtype(series):
        return _FLOAT_FORMAT
    return None


def _write_frame(ws, frame, column_formats=None):
    """Write a DataFrame as header + rows, with formats per column."""
    column_formats = column_formats or {}
    headers = []
    for col in frame.columns:
        tz = _column_tz(frame[col])
        # The zone goes in the header because the cell cannot carry it (Excel
        # datetimes have no timezone). Without it, "09:25" is ambiguous
        # between the data's own clock and the reader's.
        headers.append(f"{col} ({tz})" if tz else str(col))
    ws.append(headers)

    for row in frame.itertuples(index=False, name=None):
        ws.append([_cell_value(v) for v in row])

    for j, col in enumerate(frame.columns, start=1):
        fmt = column_formats.get(col) or _default_number_format(col, frame[col])
        for (cell,) in ws.iter_rows(min_row=2, min_col=j, max_col=j):
            if isinstance(cell.value, _dt.datetime):
                # openpyxl's default date format drops the seconds; second-
                # level timestamps would then show duplicate times.
                cell.number_format = _TIME_FORMAT
            elif fmt and isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                cell.number_format = fmt
            elif isinstance(cell.value, str) and cell.value.startswith("="):
                # openpyxl turns any string starting with "=" into a live
                # formula. A label or note must open as the text it is, not
                # execute in the reader's Excel (formula injection).
                cell.data_type = "s"
        # Wide enough for a full timestamp / header; a 9-wide default column
        # shows a datetime as "########".
        width = max(len(headers[j - 1]), 19 if fmt is None else 12) + 2
        ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = width
    ws.freeze_panes = "A2"


@logged
def export_xlsx(sheets, path="/tmp/export.xlsx", column_formats=None):
    """Write one or more DataFrames to a styled .xlsx workbook.

    Args:
        sheets: a dict {sheet_name: DataFrame}, or a single DataFrame (one
            sheet named "Data").
        path: where to write, must end in .xlsx. To hand the file to the
            user, write under /home/user/outputs/ and pass the returned path
            to display_file with inline=true.
        column_formats: optional {column_name: excel_number_format} overrides
            applied to every sheet. Numbers are stored as numbers, timestamps
            as Excel datetimes in the frame's own wall-clock time with the
            timezone named in the column header.

    Returns:
        The path written.
    """
    import openpyxl  # see module docstring: optional at import time

    path = str(path)
    if not path.lower().endswith(".xlsx"):
        # An xlsx written under a .csv/.xls name opens as garbage or is
        # refused by Excel - fail here, with the fix, instead.
        raise ValueError(f"export_xlsx: path must end in .xlsx, got {path!r}")
    if isinstance(sheets, pd.DataFrame):
        sheets = {"Data": sheets}
    if not isinstance(sheets, dict) or not sheets:
        raise TypeError("export_xlsx: sheets must be a DataFrame or a non-empty "
                        "{sheet_name: DataFrame} dict")
    for name, frame in sheets.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"export_xlsx: sheet {name!r} must be a DataFrame, "
                            f"got {type(frame).__name__}")

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, frame in sheets.items():
        _write_frame(wb.create_sheet(str(name)[:31]), frame, column_formats)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    wb.save(path)
    return path
