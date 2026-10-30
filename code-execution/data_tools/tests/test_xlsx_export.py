"""Tests for data_tools.export.export_xlsx (generic styled XLSX export).

The CSV a user writes with `frame.to_csv(path, index=False)` is the
reference: expected values come from that CSV, never from export_xlsx itself
- an oracle built from the new function's own output could not notice a row
or a column it dropped.
"""

import csv
import io
import math

import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl")

from data_tools import export_xlsx


def _frame():
    return pd.DataFrame({
        "time": pd.to_datetime(["2026-03-02 09:30", "2026-03-02 10:15", "2026-03-02 11:00"]),
        "value": [1.5, -2.25, 40500.125],
        "count": [3, 7, 11],
        "label": ["alpha", "beta", "=HYPERLINK(\"x\")"],
    })


def _csv_rows(frame):
    text = frame.to_csv(index=False)
    return list(csv.reader(io.StringIO(text)))


def _sheet_rows(ws):
    return [list(r) for r in ws.iter_rows(values_only=True)]


def _assert_cell_matches_csv(cell, text, col):
    """One workbook cell against the CSV's text for the same row/column."""
    if text == "":
        # to_csv writes NaN/None as an empty field; the cell must be empty too,
        # not the string "nan" (which Excel shows as text and SUM skips).
        assert cell is None, (col, cell)
        return
    if col == "time":
        # The CSV carries the naive datetime; the cell carries the same
        # wall-clock time.
        ts = pd.Timestamp(text)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        assert cell == ts.to_pydatetime(), (col, cell, text)
        return
    try:
        expected = float(text)
    except ValueError:
        assert cell == text, (col, cell, text)
        return
    # A number written as a string is the failure this export exists to avoid:
    # it sorts as text and every Excel formula over it returns 0.
    assert isinstance(cell, (int, float)) and not isinstance(cell, bool), (col, cell)
    if col in _EXACT_COLUMNS:
        assert cell == expected, (col, cell, text)
    else:
        # openpyxl serialises every float with "%.16g" (compat.safe_string), so
        # a 17-significant-digit float comes back one ULP off. Excel itself
        # shows 15 digits, so this is invisible to the user; short decimals
        # stay exact and are checked exactly above.
        assert math.isclose(cell, expected, rel_tol=1e-15, abs_tol=0.0), (col, cell, text)


# Short decimals that "%.16g" reproduces exactly - any difference here is a bug.
_EXACT_COLUMNS = frozenset({"value", "count"})


def test_sheet_equals_the_csv_export(tmp_path):
    frame = _frame()
    out = export_xlsx({"Data": frame}, path=str(tmp_path / "out.xlsx"))
    assert out == str(tmp_path / "out.xlsx")

    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["Data"]
    rows = _sheet_rows(wb["Data"])
    expected = _csv_rows(frame)

    assert rows[0] == list(frame.columns)
    assert len(rows) == len(expected) == 4
    for got, want in zip(rows[1:], expected[1:]):
        for col, cell, text in zip(frame.columns, got, want):
            _assert_cell_matches_csv(cell, text, col)


def test_multiple_sheets_and_a_single_frame(tmp_path):
    frame = _frame()
    out = export_xlsx({"First": frame, "Second": frame.head(2)},
                      path=str(tmp_path / "multi.xlsx"))
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["First", "Second"]
    assert len(_sheet_rows(wb["Second"])) == 3

    out = export_xlsx(frame, path=str(tmp_path / "single.xlsx"))
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["Data"]


def test_numbers_are_numbers_and_times_carry_a_number_format(tmp_path):
    out = export_xlsx({"Data": _frame()}, path=str(tmp_path / "f.xlsx"))
    ws = openpyxl.load_workbook(out)["Data"]
    value = ws.cell(row=2, column=2)
    count = ws.cell(row=2, column=3)
    entry = ws.cell(row=2, column=1)
    assert value.data_type == "n" and value.number_format == "#,##0.00####"
    assert count.data_type == "n" and count.number_format == "#,##0"
    assert entry.is_date and "yyyy" in entry.number_format
    assert ws.freeze_panes == "A2"


def test_column_formats_override_the_defaults(tmp_path):
    out = export_xlsx({"Data": _frame()}, path=str(tmp_path / "fmt.xlsx"),
                      column_formats={"value": '"$"#,##0.00'})
    ws = openpyxl.load_workbook(out)["Data"]
    assert "$" in ws.cell(row=2, column=2).number_format


def test_nan_and_inf_write_as_empty_cells(tmp_path):
    frame = pd.DataFrame({"value": [1.0, float("nan"), float("inf")],
                          "text": ["a", None, "c"]})
    out = export_xlsx({"Data": frame}, path=str(tmp_path / "nan.xlsx"))
    rows = _sheet_rows(openpyxl.load_workbook(out)["Data"])
    assert rows[1][0] == 1.0
    assert rows[2][0] is None and rows[3][0] is None, rows
    assert rows[2][1] is None


def test_tz_aware_timestamps_are_written_as_local_wall_clock(tmp_path):
    """openpyxl raises TypeError on a tz-aware datetime ("Excel does not
    support timezones"). The export must strip the zone - and keep the
    wall-clock time the CSV shows, not silently shift every value to UTC."""
    frame = pd.DataFrame({
        "time": pd.to_datetime(["2026-03-02 09:30", "2026-03-02 10:15"])
        .tz_localize("America/New_York"),
        "value": [1.0, 2.0],
    })
    out = export_xlsx({"Data": frame}, path=str(tmp_path / "tz.xlsx"))
    rows = _sheet_rows(openpyxl.load_workbook(out)["Data"])
    assert rows[0][0] == "time (America/New_York)"
    assert rows[1][0].hour == 9 and rows[1][0].minute == 30
    for col, cell, text in zip(frame.columns, rows[1], _csv_rows(frame)[1]):
        _assert_cell_matches_csv(cell, text, col)


def test_formula_like_text_stays_text(tmp_path):
    frame = pd.DataFrame({"note": ["=HYPERLINK(\"x\")", "ok"],
                          "value": [1, 2]})
    out = export_xlsx({"Data": frame}, path=str(tmp_path / "s.xlsx"))
    ws = openpyxl.load_workbook(out)["Data"]
    # A string starting with "=" must not become a live formula in the user's Excel.
    cell = ws.cell(row=2, column=1)
    assert cell.data_type == "s" and cell.value == "=HYPERLINK(\"x\")"


def test_zero_row_frame_writes_the_header(tmp_path):
    frame = pd.DataFrame(columns=["a", "b", "c"])
    out = export_xlsx({"Data": frame}, path=str(tmp_path / "empty.xlsx"))
    rows = _sheet_rows(openpyxl.load_workbook(out)["Data"])
    # A header-only sheet, not a missing sheet and not a crash: "no rows" is
    # a real result the user must be able to open and see.
    assert rows == [["a", "b", "c"]]


def test_rejects_a_non_xlsx_path(tmp_path):
    # Writing an xlsx under a .csv name hands the user a "CSV" Excel refuses to open.
    with pytest.raises(ValueError, match=".xlsx"):
        export_xlsx({"Data": _frame()}, path=str(tmp_path / "x.csv"))


def test_rejects_bad_sheet_input(tmp_path):
    with pytest.raises(TypeError):
        export_xlsx({}, path=str(tmp_path / "a.xlsx"))
    with pytest.raises(TypeError):
        export_xlsx({"Data": "not a frame"}, path=str(tmp_path / "b.xlsx"))
    with pytest.raises(TypeError):
        export_xlsx(None, path=str(tmp_path / "c.xlsx"))
