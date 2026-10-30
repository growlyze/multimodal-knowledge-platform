"""Tests for data_tools.loader: file loading, timestamp parsing, stored datasets.

The contract under test: no silent mis-parse. load_file() never raises, and
whenever it had to guess, drop or repair something, df.attrs["load_issues"]
says so.
"""

import gzip
import logging
import os

import numpy as np
import pandas as pd
import pytest

from data_tools import loader as loader_mod
from data_tools.loader import load_dataset, load_file, resample

from .conftest import make_series


def _csv_with_time(path, times, columns=("value",)):
    rows = ["time," + ",".join(columns)]
    for i, t in enumerate(times):
        rows.append(str(t) + ",".join(",%s" % (i + 1) for _ in columns))
    path.write_text("\n".join(rows) + "\n")
    return str(path)


# ---------------------------------------------------------------- load_file

def test_load_file_csv(tmp_path):
    path = tmp_path / "data.csv"
    source = make_series(rows=50)
    source.reset_index(names="time").to_csv(path, index=False)

    loaded = load_file(str(path))

    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert len(loaded) == 50
    assert {"value", "count"} <= set(loaded.columns)
    assert loaded["value"].iloc[0] == pytest.approx(source["value"].iloc[0])
    assert loaded.attrs["raw_columns"] == ["time", "value", "count"]


def test_load_file_excel(tmp_path):
    path = tmp_path / "data.xlsx"
    source = make_series(rows=30)
    source.reset_index(names="time").to_excel(path, index=False)

    loaded = load_file(str(path))

    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert len(loaded) == 30
    assert loaded["value"].iloc[-1] == pytest.approx(source["value"].iloc[-1])


def test_load_file_missing_returns_empty_not_exception():
    result = load_file("/definitely/not/here.csv")
    assert isinstance(result, pd.DataFrame) and result.empty
    assert any("file not found" in i for i in result.attrs.get("load_issues", []))


def test_load_file_unreadable_bytes_degrade_visibly(tmp_path):
    bad = tmp_path / "junk.xlsx"
    bad.write_bytes(b"\x00\x01\x02 not really a workbook \xff\xfe")
    result = load_file(str(bad))
    assert isinstance(result, pd.DataFrame) and result.empty
    assert any("could not be read" in i for i in result.attrs.get("load_issues", [])), \
        result.attrs.get("load_issues")


def test_load_file_unix_epoch_seconds(tmp_path):
    """A bare epoch-seconds time column must not silently land in 1970.

    pd.to_datetime's default unit for a bare integer is NANOSECONDS since
    epoch, so a real 2022 UNIX-seconds timestamp used to parse "successfully"
    into 1970-01-01 00:00:01.64...
    """
    path = tmp_path / "epoch_s.csv"
    path.write_text(
        "time,value\n"
        "1640995800,1\n"     # 2022-01-01 00:10:00 UTC
        "1640995860,2\n"     # +60s
    )
    df = load_file(str(path))
    assert len(df) == 2
    assert df.index[0].year == 2022, f"landed at {df.index[0]!r} instead of 2022"
    assert (df.index[1] - df.index[0]) == pd.Timedelta(seconds=60)


def test_load_file_unix_epoch_milliseconds(tmp_path):
    path = tmp_path / "epoch_ms.csv"
    path.write_text(
        "time,value\n"
        "1640995800000,1\n"
        "1640995860000,2\n"
    )
    df = load_file(str(path))
    assert len(df) == 2
    assert df.index[0].year == 2022, f"landed at {df.index[0]!r} instead of 2022"
    assert (df.index[1] - df.index[0]) == pd.Timedelta(seconds=60)


def test_load_file_unix_epoch_nanoseconds(tmp_path):
    path = tmp_path / "epoch_ns.csv"
    path.write_text(
        "time,value\n"
        "1640995800000000000,1\n"
        "1640995860000000000,2\n"
    )
    df = load_file(str(path))
    assert len(df) == 2
    assert df.index[0].year == 2022, f"landed at {df.index[0]!r} instead of 2022"
    assert (df.index[1] - df.index[0]) == pd.Timedelta(seconds=60)


def test_load_file_mixed_epoch_and_string_timestamps_drops_with_a_caveat(tmp_path):
    """A time column where SOME rows are bare epoch numbers and others are
    date strings routes through the generic string parser, where the epoch row
    reads as unparseable. That must degrade with a stated caveat, not raise
    and not silently misinterpret the epoch number."""
    path = tmp_path / "mixed_epoch_and_string.csv"
    path.write_text(
        "time,value\n"
        "1640995800,1\n"                # epoch seconds
        "2022-01-01 09:31:00,2\n"  # ISO string
    )
    df = load_file(str(path))          # must not raise
    assert len(df) == 1, f"expected the unparseable epoch row dropped, got:\n{df}"
    assert df.index[0] == pd.Timestamp("2022-01-01 09:31:00")
    assert any("could not be parsed" in m or "dropped" in m
               for m in df.attrs.get("load_issues", [])), df.attrs.get("load_issues")


def test_load_file_mixed_tz_offsets_all_aware_does_not_raise(tmp_path):
    path = tmp_path / "mixed_offsets.csv"
    path.write_text(
        "time,value\n"
        "2022-01-01T09:30:00+05:00,1\n"
        "2022-01-01T09:31:00+00:00,2\n"
    )
    df = load_file(str(path))
    assert len(df) == 2
    assert df.index.tz is not None
    assert list(df.index) == sorted(df.index)


def test_load_file_mixed_timezone_timestamps_does_not_raise(tmp_path):
    """load_file's own docstring: 'this never raises'. A column mixing a
    tz-aware and a naive timestamp string used to break that promise with an
    uncaught ValueError straight out of pandas."""
    path = tmp_path / "mixed_tz.csv"
    path.write_text(
        "time,value\n"
        "2022-01-01 09:30:00,1\n"
        "2022-01-01T09:31:00Z,2\n"
        "2022-01-01 09:32:00,3\n"
    )
    df = load_file(str(path))          # must not raise
    assert len(df) == 3
    assert isinstance(df.index, pd.DatetimeIndex)


def test_load_file_partial_timestamp_failure_is_logged(tmp_path, caplog):
    """A row with an unparseable timestamp is dropped -- and must say so."""
    path = tmp_path / "partial_bad.csv"
    path.write_text(
        "time,value\n"
        "2022-01-01 09:30,1\n"
        "not-a-date,2\n"
        "2022-01-01 09:32,3\n"
    )
    with caplog.at_level(logging.WARNING, logger="data_tools"):
        df = load_file(str(path))
    assert len(df) == 2
    assert any("could not be parsed" in r.message or "dropped" in r.message
               for r in caplog.records), [r.message for r in caplog.records]


def test_load_file_implausible_epoch_unit_is_flagged(tmp_path, caplog):
    """A timestamp column that parses to a wildly implausible date range
    (e.g. an epoch unit guessed wrong) must warn, not pass silently."""
    path = tmp_path / "weird_unit.csv"
    # Interpreted as seconds this would land in the far future (year ~53075).
    path.write_text(
        "time,value\n"
        "1640995800000000000000,1\n"
        "1640995800000000000060,2\n"
    )
    with caplog.at_level(logging.WARNING, logger="data_tools"):
        load_file(str(path))
    assert any("implausible" in r.message for r in caplog.records), \
        [r.message for r in caplog.records]


def test_load_file_excel_serial_dates_are_converted(tmp_path):
    path = tmp_path / "serial.csv"
    path.write_text(
        "date,value\n"
        "46224.0,1\n"
        "46225.5,2\n"
    )
    df = load_file(str(path))
    assert df.index[0] == pd.Timestamp("2026-07-21")
    assert df.index[1] == pd.Timestamp("2026-07-22 12:00")
    assert any("Excel serial" in i for i in df.attrs.get("load_issues", []))


def test_load_file_excel_range_median_with_huge_outlier_does_not_raise(tmp_path):
    """The median can sit in the Excel range while one outlier does not -
    pandas then overflows past errors='coerce'; must degrade, not raise."""
    path = tmp_path / "serial_outlier.csv"
    path.write_text("date,value\n0,1\n10000,2\n9223372036854775807,3\n")
    df = load_file(str(path))          # must not raise
    assert isinstance(df, pd.DataFrame)


def test_load_file_unambiguous_day_first_dates(tmp_path):
    path = tmp_path / "dayfirst.csv"
    path.write_text(
        "date,value\n"
        "13.08.2026,1\n"
        "14.08.2026,2\n"
    )
    df = load_file(str(path))
    assert df.index[0] == pd.Timestamp("2026-08-13")


def test_load_file_ambiguous_dotted_dates_parse_day_first_and_say_so(tmp_path):
    path = tmp_path / "dotted.csv"
    path.write_text(
        "date,value\n"
        "03.08.2026,1\n"
        "04.08.2026,2\n"
    )
    df = load_file(str(path))
    assert df.index[0] == pd.Timestamp("2026-08-03")
    assert any("day-first" in i for i in df.attrs.get("load_issues", [])), df.attrs


def test_load_file_iso_dates_are_not_day_first_swapped(tmp_path):
    path = tmp_path / "iso.csv"
    path.write_text(
        "date,value\n"
        "2026-03-04,1\n"
        "2026-03-05,2\n"
    )
    df = load_file(str(path))
    assert df.index[0] == pd.Timestamp("2026-03-04")


def test_load_file_split_date_and_time_columns_are_combined(tmp_path):
    path = tmp_path / "split.csv"
    path.write_text(
        "Date,Time,value\n"
        "2022-01-03,09:30,1\n"
        "2022-01-03,09:31,2\n"
    )
    df = load_file(str(path))
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp("2022-01-03 09:30")


def test_load_file_time_only_column_without_dates_is_flagged(tmp_path):
    path = tmp_path / "clocks.csv"
    path.write_text("time,value\n09:30,1\n09:31,2\n")
    df = load_file(str(path))
    assert any("clock times only" in i for i in df.attrs.get("load_issues", [])), df.attrs


def test_load_file_semicolon_decimal_comma_and_datum_header(tmp_path):
    path = tmp_path / "euro.csv"
    path.write_text(
        "Datum;value\n"
        "2026-08-03;24150,5\n"
        "2026-08-04;24160,25\n"
    )
    df = load_file(str(path))
    assert df["value"].tolist() == [24150.5, 24160.25]
    assert df.index[0] == pd.Timestamp("2026-08-03")


def test_load_file_leading_comment_line_is_skipped(tmp_path):
    path = tmp_path / "commented.csv"
    path.write_text(
        "# exported by a program\n"
        "time,value\n"
        "2022-01-03 09:30,1\n"
        "2022-01-03 09:31,2\n"
    )
    df = load_file(str(path))
    assert list(df.columns) == ["value"]
    assert df.attrs["raw_columns"] == ["time", "value"]
    assert df.attrs["file_comments"] == ["# exported by a program"]


def test_load_file_records_provenance_on_attrs(tmp_path):
    """A result must be traceable to the file that produced it."""
    path = tmp_path / "prov.csv"
    make_series(rows=20).reset_index(names="time").to_csv(path, index=False)

    df = load_file(str(path))
    source = df.attrs.get("source")
    assert source, df.attrs
    assert source["path"] == str(path)
    assert source["rows"] == 20
    assert source["start"] and source["end"]
    assert len(source["sha1"]) == 12


def test_load_file_fixed_offset_states_the_file_wall_clock(tmp_path):
    path = tmp_path / "offset.csv"
    path.write_text(
        "time,value\n"
        "2022-01-01T09:30:00+02:00,1\n"
        "2022-01-01T09:31:00+02:00,2\n"
    )
    df = load_file(str(path))
    assert df.attrs.get("file_tz") is not None
    assert "02:00" in str(df.attrs["file_tz"])


# ------------------------------------------------------------- compressed

def _stage_gz(tmp_path, name="data.csv"):
    """Write the same rows as plain and gzipped CSV. Returns both paths."""
    frame = make_series(rows=20).reset_index(names="time")
    plain = tmp_path / name
    frame.to_csv(plain, index=False)
    gz = tmp_path / (name + ".gz")
    gz.write_bytes(gzip.compress(plain.read_bytes()))
    return str(plain), str(gz)


def test_csv_gz_loads_identically_to_csv(tmp_path):
    plain, gz = _stage_gz(tmp_path)
    direct, wrapped = load_file(plain), load_file(gz)
    assert not wrapped.attrs.get("load_issues"), wrapped.attrs.get("load_issues")
    assert list(wrapped.columns) == list(direct.columns)
    assert wrapped.shape == direct.shape
    assert wrapped.equals(direct), (wrapped.head(), direct.head())
    # the raw header is what the file says, captured before the load reshapes
    assert wrapped.attrs["raw_columns"] == direct.attrs["raw_columns"]


def test_excel_gz_loads(tmp_path):
    frame = make_series(rows=20).reset_index(names="time")
    plain = tmp_path / "data.xlsx"
    frame.to_excel(plain, index=False)
    gz = tmp_path / "data.xlsx.gz"
    gz.write_bytes(gzip.compress(plain.read_bytes()))

    wrapped = load_file(str(gz))
    assert not wrapped.attrs.get("load_issues"), wrapped.attrs.get("load_issues")
    assert wrapped.shape == load_file(str(plain)).shape


def test_corrupt_gz_degrades_visibly_not_silently(tmp_path):
    bad = tmp_path / "broken.csv.gz"
    bad.write_bytes(b"\x1f\x8b\x08this is not a gzip stream")
    result = load_file(str(bad))
    assert result.empty
    issues = result.attrs.get("load_issues", [])
    assert any("could not be decompressed" in i for i in issues), issues


def test_gz_that_decompresses_to_nothing_is_an_issue(tmp_path):
    empty = tmp_path / "empty.csv.gz"
    empty.write_bytes(gzip.compress(b""))
    result = load_file(str(empty))
    assert result.empty
    issues = result.attrs.get("load_issues", [])
    assert any("could not be decompressed" in i for i in issues), issues


def test_decompressed_temp_file_is_removed_before_returning(tmp_path, monkeypatch):
    _, gz = _stage_gz(tmp_path)
    made = []
    real_mkstemp = loader_mod.tempfile.mkstemp

    def recording(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        made.append(path)
        return fd, path

    monkeypatch.setattr(loader_mod.tempfile, "mkstemp", recording)
    load_file(gz)
    assert made, "expected load_file to decompress via a temp file"
    assert all(not os.path.exists(p) for p in made), made

    # the failure path cleans up too
    made.clear()
    bad = tmp_path / "broken.csv.gz"
    bad.write_bytes(b"\x1f\x8b\x08junk")
    load_file(str(bad))
    assert all(not os.path.exists(p) for p in made), made


# ---------------------------------------------------------------- resample

def test_resample_default_aggregation():
    index = pd.date_range("2022-01-03 09:30", periods=6, freq="5min")
    df = pd.DataFrame({
        "Open": [10, 11, 12, 13, 14, 15],
        "High": [15, 12, 18, 14, 16, 20],
        "Low": [9, 8, 11, 12, 13, 14],
        "Close": [11, 12, 13, 14, 15, 16],
        "Volume": [100, 200, 300, 400, 500, 600],
    }, index=index, dtype=float)

    out = resample(df, "15min")

    assert len(out) == 2
    first = out.iloc[0]
    assert first["Open"] == 10        # first
    assert first["High"] == 18        # max of 15, 12, 18
    assert first["Low"] == 8          # min of 9, 8, 11
    assert first["Close"] == 13       # last
    assert first["Volume"] == 600     # sum of 100, 200, 300
    second = out.iloc[1]
    assert (second["Open"], second["High"], second["Low"], second["Close"]) == (13, 20, 12, 16)
    assert second["Volume"] == 1500


def test_resample_custom_agg_and_mean_default():
    index = pd.date_range("2022-01-03 09:30", periods=6, freq="5min")
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                       "count": [1, 1, 1, 1, 1, 1]}, index=index)
    out = resample(df, "15min")
    assert out["value"].iloc[0] == pytest.approx(2.0)     # mean of 1, 2, 3
    assert out["count"].iloc[0] == pytest.approx(1.0)

    summed = resample(df, "15min", agg={"value": "sum", "count": "sum"})
    assert summed["value"].iloc[0] == pytest.approx(6.0)
    assert summed["count"].sum() == 6


def test_resample_accepts_legacy_pandas_aliases():
    """Generated code still emits '15T' and '1H'; both must keep working."""
    df = make_series(rows=60, freq="5min")
    pd.testing.assert_frame_equal(resample(df, "15T"), resample(df, "15min"))
    pd.testing.assert_frame_equal(resample(df, "1H"), resample(df, "1h"))


def test_resample_bad_input_does_not_raise():
    assert resample(pd.DataFrame(), "15min").empty
    df = make_series(rows=10)
    pd.testing.assert_frame_equal(resample(df, "not-a-timeframe"), df)
    assert resample("not a frame", "15min") == "not a frame"


# ----------------------------------------------------------- load_dataset

def test_load_dataset_gracefully_empty_when_api_unreachable():
    """No API answer must give an empty frame with a note, not a crash."""
    result = load_dataset("no-such-dataset-xyz", api_url="http://127.0.0.1:9")
    assert isinstance(result, pd.DataFrame) and result.empty


def test_load_dataset_records_issue_when_nothing_found():
    """'no dataset found' must show up on df.attrs['load_issues'], so a
    genuinely missing dataset gets "I couldn't find it" instead of guessing."""
    result = load_dataset("no-such-dataset-xyz", api_url="http://127.0.0.1:9")
    issues = result.attrs.get("load_issues", [])
    assert any("no-such-dataset-xyz" in i and "no" in i.lower() for i in issues), issues


def test_load_dataset_records_issue_listing_multiple_api_matches(monkeypatch):
    """Silently taking hits[0] risks resolving to the wrong dataset with
    nothing telling the caller a choice was made for them. Still resolves to
    the first hit (never refuse/raise here), but records what else matched."""
    hits = [
        {"table_name": "table_a", "rows": [{"time": "2022-01-03", "value": 1},
                                           {"time": "2022-01-04", "value": 2}]},
        {"table_name": "table_b"},
        {"table_name": "table_c"},
    ]
    monkeypatch.setattr(loader_mod, "_api_get",
                         lambda url, timeout=None: {"results": hits} if "search" in url else None)

    df = load_dataset("data", api_url="http://fake")
    assert not df.empty, "must still resolve to the first match, not refuse"
    issues = df.attrs.get("load_issues", [])
    assert any("table_b" in i and "table_c" in i for i in issues), issues


def _paged_api(pages, total, calls, fail_first=(), timeouts=None):
    """Fake /admin/data: one search hit, then the table served page by page.

    `fail_first`: offsets whose first request times out (returns None) and
    whose retry succeeds. `timeouts`: a list that records each table call's
    timeout."""
    failed = set()

    def get(url, timeout=None):
        calls.append(url)
        if "/table/" in url and timeouts is not None:
            timeouts.append(timeout)
        if "search" in url:
            return {"results": [{"table_name": "data_big"}]}
        offset = int(url.split("offset=")[1]) if "offset=" in url else 0
        if offset in fail_first and offset not in failed:
            failed.add(offset)
            return None
        for start, rows, truncated in pages:
            if start == offset:
                return {"rows": rows, "row_count": len(rows), "total_rows": total,
                        "offset": offset, "truncated": truncated}
        return None
    return get


def _rows(start, n):
    return [{"time": str(pd.Timestamp("2022-01-03") + pd.Timedelta(hours=start + i)),
             "value": i} for i in range(n)]


def test_load_dataset_reads_every_page_of_a_truncated_table(monkeypatch):
    """A table above the endpoint's page size came back as its first page
    only, with `truncated: true` ignored - a silent partial result."""
    calls = []
    pages = [(0, _rows(0, 3), True), (3, _rows(3, 3), True), (6, _rows(6, 2), False)]
    monkeypatch.setattr(loader_mod, "_api_get", _paged_api(pages, 8, calls))
    df = load_dataset("data big", api_url="http://fake")
    assert len(df) == 8
    assert [u.split("offset=")[1] for u in calls if "offset=" in u] == ["3", "6"]
    assert not any("of 8 rows" in i for i in df.attrs.get("load_issues", []))


def test_load_dataset_says_so_when_a_later_page_fails(monkeypatch):
    pages = [(0, _rows(0, 3), True)]          # the offset=3 page never answers
    monkeypatch.setattr(loader_mod, "_api_get", _paged_api(pages, 8, []))
    df = load_dataset("data big", api_url="http://fake")
    assert len(df) == 3
    assert any("3 of 8 rows" in i for i in df.attrs.get("load_issues", [])), df.attrs


def test_load_dataset_flags_a_row_count_short_of_the_catalog(monkeypatch):
    pages = [(0, _rows(0, 3), False)]         # not truncated, but the catalog says 5
    monkeypatch.setattr(loader_mod, "_api_get", _paged_api(pages, 5, []))
    df = load_dataset("data big", api_url="http://fake")
    assert any("3 of 5 rows" in i for i in df.attrs.get("load_issues", [])), df.attrs


def test_load_dataset_retries_a_page_that_fails_once(monkeypatch):
    """One slow page must not end the read: a second request survives it."""
    calls = []
    pages = [(0, _rows(0, 3), True), (3, _rows(3, 3), True), (6, _rows(6, 2), False)]
    monkeypatch.setattr(loader_mod, "_api_get", _paged_api(pages, 8, calls, fail_first={0, 3}))
    df = load_dataset("data big", api_url="http://fake")
    assert len(df) == 8
    assert not any("of 8 rows" in i for i in df.attrs.get("load_issues", [])), df.attrs


def test_table_pages_wait_longer_than_a_search(monkeypatch):
    """API_TIMEOUT is sized for /data/search; the table endpoint builds a
    whole page before its first byte, so table calls need their own budget."""
    timeouts = []
    pages = [(0, _rows(0, 3), True), (3, _rows(3, 2), False)]
    monkeypatch.setattr(loader_mod, "_api_get", _paged_api(pages, 5, [], timeouts=timeouts))
    load_dataset("data big", api_url="http://fake")
    assert timeouts and all(t is not None and t >= 60 for t in timeouts), timeouts


def test_load_dataset_says_the_table_exists_when_it_cannot_be_read(monkeypatch):
    """The table was found but its first page never arrived: that must NOT be
    reported as "no stored dataset found"."""
    monkeypatch.setattr(loader_mod, "_api_get", _paged_api([], 8, []))
    df = load_dataset("data big", api_url="http://fake")
    issues = df.attrs.get("load_issues", [])
    assert df.empty
    assert any("data_big" in i and "could not be read" in i for i in issues), issues
    assert not any("no stored dataset found" in i for i in issues), issues


def test_load_dataset_exact_name_wins_without_an_ambiguity_note(monkeypatch):
    hits = [{"table_name": "data_x_2"}, {"table_name": "data_x"}]
    monkeypatch.setattr(loader_mod, "_api_get",
                         lambda url, timeout=None: {"results": hits} if "search" in url
                         else {"rows": _rows(0, 2), "truncated": False, "total_rows": 2})
    df = load_dataset("data_x", api_url="http://fake")
    assert len(df) == 2
    assert not any("matched multiple" in i for i in df.attrs.get("load_issues", []))


def test_load_dataset_sends_bearer_auth_when_key_set(monkeypatch):
    """$HAYSTACK_API_KEY rides as an Authorization header; absent, none."""
    seen = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"results": []}'

    def fake_urlopen(request, timeout=None):
        seen["headers"] = dict(request.header_items())
        return _FakeResponse()

    monkeypatch.setattr(loader_mod.urllib.request, "urlopen", fake_urlopen)

    monkeypatch.setenv("HAYSTACK_API_KEY", "secret-token")
    load_dataset("x", api_url="http://fake")
    assert seen["headers"].get("Authorization") == "Bearer secret-token"

    monkeypatch.delenv("HAYSTACK_API_KEY")
    load_dataset("x", api_url="http://fake")
    assert not any(k.lower() == "authorization" for k in seen["headers"])


def test_load_dataset_stamps_provenance(monkeypatch):
    monkeypatch.setattr(loader_mod, "_api_get",
                         lambda url, timeout=None: {"results": [{"table_name": "t1"}]} if "search" in url
                         else {"rows": _rows(0, 3), "truncated": False, "total_rows": 3})
    df = load_dataset("t1", api_url="http://fake")
    source = df.attrs.get("source")
    assert source and source["rows"] == 3 and source["query"] == "t1"
