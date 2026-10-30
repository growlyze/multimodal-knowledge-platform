"""describe_file() / format_file_summary() on neutral tabular fixtures.

Same methodology as the original corpus tests, re-skinned from market data to
generic tables generated in-test:

  * no silent mis-parse - every file is either read correctly or says what
    went wrong in load_issues;
  * describe_file() agrees with truth computed independently with plain
    pandas, never with describe_file() itself;
  * every defect lands in `problems` as a plain sentence.

Gap classification is generic by design: a short gap (< 24h) vs a long gap
(>= 24h). No wall-clock schedule is assumed.
"""

import pandas as pd
import pytest

from data_tools import describe_file, format_file_summary, load_file

from .conftest import make_series


def _write_csv(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _series_with_gaps(tmp_path):
    """Hourly rows for 3 days with a 2h hole one day and a 2-day hole next."""
    idx = (pd.date_range("2024-03-01 09:00", "2024-03-01 12:00", freq="h")
           .append(pd.date_range("2024-03-01 14:00", "2024-03-01 17:00", freq="h"))
           .append(pd.date_range("2024-03-04 09:00", "2024-03-04 12:00", freq="h")))
    df = pd.DataFrame({"value": range(len(idx))}, index=idx)
    path = tmp_path / "gaps.csv"
    df.reset_index(names="time").to_csv(path, index=False)
    return str(path), df


# ---------------------------------------------------------------- basics

def test_clean_series_agrees_with_plain_pandas(tmp_path):
    df = make_series(rows=48, freq="h").reset_index(names="time")
    path = tmp_path / "clean.csv"
    df.to_csv(path, index=False)

    info = describe_file(str(path))
    assert info["rows"] == 48
    assert info["raw_columns"] == ["time", "value", "count"]
    assert info["n_columns"] == 3
    assert info["kind"] == "series"
    assert info["first_ts"] == pd.Timestamp(df["time"].iloc[0]).isoformat()
    assert info["last_ts"] == pd.Timestamp(df["time"].iloc[-1]).isoformat()
    assert info["interval_seconds"] == 3600
    assert info["interval"] == "1h"
    assert info["gaps"] == []
    assert info["gap_counts_by_kind"] == {}
    assert info["duplicates"] == 0
    assert info["sorted"] is True
    assert info["problems"] == []


def test_shape_and_column_dtypes_are_described(tmp_path):
    path = _write_csv(tmp_path, "mixed.csv",
                      "time,value,flag\n"
                      "2024-01-01 00:00,1.5,on\n"
                      "2024-01-01 01:00,2.5,off\n"
                      "2024-01-01 02:00,3.5,on\n")
    info = describe_file(path)
    by_name = {c["name"]: c for c in info["columns"]}
    assert by_name["value"]["type"] == "numbers"
    assert by_name["value"]["min"] == pytest.approx(1.5)
    assert by_name["value"]["max"] == pytest.approx(3.5)
    assert by_name["value"]["sum"] == pytest.approx(7.5)
    assert by_name["value"]["missing"] == 0
    assert by_name["flag"]["type"] == "text"
    assert by_name["flag"]["distinct"] == 2
    assert by_name["flag"]["top"] == {"on": 2, "off": 1}


def test_describe_file_accepts_a_dataframe():
    df = make_series(rows=30, freq="10min")
    info = describe_file(df)
    assert info["rows"] == 30
    assert info["kind"] == "series"
    assert format_file_summary(info)


# ---------------------------------------------------------------- gaps

def test_short_and_long_gaps_are_classified_by_duration(tmp_path):
    path, df = _series_with_gaps(tmp_path)
    info = describe_file(path)

    # computed independently: the two spacings above the 1h median
    diffs = pd.DatetimeIndex(df.index).to_series().diff().dropna()
    assert (diffs > diffs.median()).sum() == 2

    kinds = sorted(g["kind"] for g in info["gaps"])
    assert kinds == ["long_gap", "short_gap"], info["gaps"]
    assert info["gap_counts_by_kind"] == {"short_gap": 1, "long_gap": 1}

    short = next(g for g in info["gaps"] if g["kind"] == "short_gap")
    long_ = next(g for g in info["gaps"] if g["kind"] == "long_gap")
    assert short["minutes"] == pytest.approx(120)     # 12:00 -> 14:00
    assert short["missing_minutes"] == pytest.approx(60)   # minus the 1h median
    assert long_["minutes"] >= 24 * 60                # 03-01 17:00 -> 03-04 09:00
    assert short["start"] < short["end"] <= long_["start"]


def test_gap_problem_names_counts_where_and_how_long(tmp_path):
    path, _ = _series_with_gaps(tmp_path)
    info = describe_file(path)
    sentences = [p for p in info["problems"] if "gap" in p]
    assert sentences and "2 gaps" in sentences[0]
    assert "short gap" in sentences[0] and "long gap" in sentences[0]

    text = format_file_summary(info)
    assert "Gaps (spacing above the 1h median): 2" in text
    assert "short_gap 1" in text and "long_gap 1" in text


def test_clean_file_reports_zero_gaps_not_nothing(tmp_path):
    df = make_series(rows=10, freq="h").reset_index(names="time")
    path = tmp_path / "clean2.csv"
    df.to_csv(path, index=False)
    text = format_file_summary(describe_file(str(path)))
    # F2: an absence is only stated as the outcome of a check that ran.
    assert "median): 0 - no gaps" in text
    assert "Problems found: 0" in text


# ------------------------------------------------------- corrupt rows

def test_corrupt_rows_are_counted_not_hidden(tmp_path):
    path = _write_csv(tmp_path, "corrupt.csv",
                      "time,value,start_date\n"
                      "2024-01-01 00:00,1,2024-01-01\n"
                      "2024-01-01 01:00,oops,2024-01-02\n"
                      "2024-01-01 02:00,3.5,bad-date\n"
                      "2024-01-01 03:00,4.5,2024-01-04\n")
    info = describe_file(path)
    by_name = {c["name"]: c for c in info["columns"]}
    assert by_name["value"]["type"] == "numbers"
    assert by_name["value"]["unparseable"] == 1
    assert by_name["start_date"]["type"] == "dates"
    assert by_name["start_date"]["unparseable"] == 1

    joined = "\n".join(info["problems"])
    assert "could not be read as numbers" in joined
    assert "could not be read as dates" in joined
    text = format_file_summary(info)
    assert "| 1 corrupt" in text


def test_missing_values_are_reported_per_column(tmp_path):
    path = _write_csv(tmp_path, "missing.csv",
                      "time,value,count\n"
                      "2024-01-01 00:00,1,\n"
                      "2024-01-01 01:00,,2\n"
                      "2024-01-01 02:00,3,\n")
    info = describe_file(path)
    by_name = {c["name"]: c for c in info["columns"]}
    assert by_name["value"]["missing"] == 1
    assert by_name["count"]["missing"] == 2
    joined = "\n".join(info["problems"])
    assert "1 missing value(s) in 'value'" in joined
    assert "2 missing value(s) in 'count'" in joined


def test_duplicate_timestamps_are_flagged(tmp_path):
    path = _write_csv(tmp_path, "dupes.csv",
                      "time,value\n"
                      "2024-01-01 00:00,1\n"
                      "2024-01-01 01:00,2\n"
                      "2024-01-01 01:00,3\n")
    info = describe_file(path)
    assert info["duplicates"] == 1
    assert any("duplicate timestamp" in p for p in info["problems"])


def test_unsorted_rows_are_flagged(tmp_path):
    path = _write_csv(tmp_path, "unsorted.csv",
                      "time,value\n"
                      "2024-01-01 02:00,3\n"
                      "2024-01-01 00:00,1\n"
                      "2024-01-01 01:00,2\n")
    info = describe_file(path)
    assert info["sorted"] is False
    assert any("not in time order" in p for p in info["problems"])


# --------------------------------------------------------------- tables

def test_table_without_time_column_states_nothing_time_based(tmp_path):
    path = _write_csv(tmp_path, "table.csv",
                      "amount,category\n"
                      "10.5,alpha\n"
                      "20.0,beta\n"
                      "30.25,alpha\n")
    info = describe_file(path)
    assert info["kind"] == "table"
    assert info["first_ts"] is None and info["gaps"] == []
    assert info["duplicates"] == 0
    assert info["problems"] == []
    text = format_file_summary(info)
    assert "no time column" in text
    assert "Gaps: 0 gaps checked" in text


def test_empty_file(tmp_path):
    path = _write_csv(tmp_path, "empty.csv", "")
    info = describe_file(path)
    assert info["rows"] == 0
    assert info["first_ts"] is None and info["gaps"] == []
    # a zero-byte file is not a readable table - says so instead of describing
    assert info["readable"] is False
    assert any("could not be read" in p for p in info["problems"])


def test_header_only_file(tmp_path):
    path = _write_csv(tmp_path, "header.csv", "time,value\n")
    info = describe_file(path)
    assert info["rows"] == 0
    assert info["raw_columns"] == ["time", "value"]
    assert info["readable"] is True
    assert any("no data rows" in p for p in info["problems"])


# ------------------------------------------------------------- timezone

def test_timezone_is_stated_or_said_to_be_absent(tmp_path):
    naive = _write_csv(tmp_path, "naive.csv",
                       "time,value\n2024-01-01 00:00,1\n2024-01-01 01:00,2\n")
    info = describe_file(naive)
    assert "none stated" in info["tz"]

    offset = _write_csv(tmp_path, "offset.csv",
                        "time,value\n"
                        "2024-01-01T00:00:00+02:00,1\n"
                        "2024-01-01T01:00:00+02:00,2\n")
    info = describe_file(offset)
    assert "02:00" in info["tz"]
    # times are shown in the file's own wall clock, not UTC
    assert info["first_ts"].startswith("2024-01-01T00:00:00")


def test_gz_file_is_reported_by_its_uploaded_name(tmp_path):
    import gzip
    df = make_series(rows=20).reset_index(names="time")
    plain = tmp_path / "data.csv"
    df.to_csv(plain, index=False)
    gz = tmp_path / "data.csv.gz"
    gz.write_bytes(gzip.compress(plain.read_bytes()))
    info = describe_file(str(gz))
    assert info["file"] == "data.csv.gz"
    assert info["rows"] == 20
    assert not info["load_issues"], info["load_issues"]
    assert format_file_summary(info)


# ------------------------------------------------ no silent mis-parse

def test_excel_serial_dates_are_converted_not_read_as_1970(tmp_path):
    path = _write_csv(tmp_path, "serial.csv",
                      "date,value\n46224.0,1\n46225.0,2\n")
    info = describe_file(path)
    assert info["first_ts"].startswith("2026-07-21"), info["first_ts"]
    assert info["last_ts"].startswith("2026-07-22"), info["last_ts"]


def test_day_first_dates_read_day_first(tmp_path):
    path = _write_csv(tmp_path, "dayfirst.csv",
                      "date,value\n13.08.2026,1\n14.08.2026,2\n")
    info = describe_file(path)
    assert info["first_ts"].startswith("2026-08-13"), info["first_ts"]


def test_split_date_time_columns_are_combined(tmp_path):
    path = _write_csv(tmp_path, "split.csv",
                      "Date,Time,value\n2024-01-03,09:30,1\n2024-01-03,09:31,2\n")
    info = describe_file(path)
    assert info["first_ts"].startswith("2024-01-03T09:30"), info["first_ts"]
    assert info["interval"] == "1min"


def test_decimal_comma_file_is_not_all_missing(tmp_path):
    path = _write_csv(tmp_path, "euro.csv",
                      "Datum;value\n2026-08-03;24150,5\n2026-08-04;24160,25\n")
    info = describe_file(path)
    by_name = {c["name"]: c for c in info["columns"]}
    assert by_name["value"]["sum"] == pytest.approx(48310.75)


def test_every_misparse_is_visible(tmp_path):
    """The one-sided contract: load_file either reads the file right, or the
    summary says it did not - never both wrong and silent."""
    path = _write_csv(tmp_path, "badtime.csv",
                      "time,value\nnot-a-date,1\nalso-bad,2\n")
    info = describe_file(path)
    assert info["readable"] is False
    assert info["problems"], info
    text = format_file_summary(info)
    assert "COULD NOT BE READ CORRECTLY" in text


# ------------------------------------------------------------- rendering

def test_summary_prints_every_check_with_its_result():
    df = make_series(rows=5, freq="h")
    text = format_file_summary(describe_file(df))
    for line in ("File:", "Rows:", "Columns:", "Time range:", "Row interval:",
                 "median): 0", "Duplicate timestamps: 0",
                 "Problems found: 0"):
        assert line in text, line
