"""Fuzz tests for data_tools.loader's timestamp parsing and the "never a
silent wrong number" contract the rest of the package promises.

load_file()'s docstring promises "never raises" even on a malformed file.
_to_datetime_index already has hand-fixed regression tests for specific bugs
(bare epoch integers misread as nanoseconds; mixed tz-aware/naive strings).
These tests generalize both classes with Hypothesis-generated adversarial
time columns instead of fixed examples, testing through the public
load_file() API the same way test_loader.py does.

The second group generalizes "never raises" into its two-sided form:
degrading is allowed, degrading SILENTLY is not. Whenever the result rests
on a guess or a repair, something in the returned object says so.

max_examples is deliberately modest here: these drive whole pipelines rather
than one pure function.
"""

import os
import tempfile

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from data_tools import describe_file, format_file_summary
from data_tools.loader import load_file

_TIME_VALUE = st.one_of(
    st.integers(min_value=0, max_value=10**20),   # covers s/ms/us/ns AND unrepresentable magnitudes
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=20),
    st.none(),
)


def _write_and_load(values):
    n = len(values)
    df = pd.DataFrame({"time": values, "value": list(range(n))})
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "fuzz.csv")
        df.to_csv(path, index=False)
        return load_file(path)


@given(st.lists(_TIME_VALUE, min_size=1, max_size=30))
@settings(max_examples=100, deadline=None)
def test_load_file_never_raises_on_adversarial_time_column(values):
    result = _write_and_load(values)
    assert isinstance(result, pd.DataFrame)


@given(st.lists(st.booleans(), min_size=2, max_size=20))
@settings(max_examples=100, deadline=None)
def test_load_file_never_raises_on_mixed_tz_and_naive_timestamps(tz_flags):
    """The exact bug class already fixed once: a time column mixing a
    trailing 'Z' (tz-aware) with plain ISO strings (naive) used to raise
    ValueError instead of returning NaT for the unparseable rows."""
    base = pd.Timestamp("2022-01-01")
    times = [
        (base + pd.Timedelta(hours=i)).isoformat() + ("Z" if flag else "")
        for i, flag in enumerate(tz_flags)
    ]
    result = _write_and_load(times)
    assert isinstance(result, pd.DataFrame)


@given(st.binary(max_size=2048), st.sampled_from([".csv", ".xlsx", ".csv.gz"]))
@settings(max_examples=60, deadline=None)
def test_load_file_never_raises_on_arbitrary_bytes(blob, suffix):
    """Any bytes behind any supported extension: a result or a stated failure,
    never an exception out of the sandbox."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "junk" + suffix)
        with open(path, "wb") as handle:
            handle.write(blob)
        result = load_file(path)
    assert isinstance(result, pd.DataFrame)


@given(st.lists(_TIME_VALUE, min_size=1, max_size=20))
@settings(max_examples=60, deadline=None)
def test_describe_file_never_raises_and_never_returns_unrenderable(values):
    """The summary layer sits on top of the fuzzed loader: describe_file()
    must survive every input load_file() does, and format_file_summary()
    must always produce quotable text from whatever came back."""
    df = _write_and_load(values)
    info = describe_file(df)
    assert isinstance(info, dict)
    text = format_file_summary(info)
    assert isinstance(text, str) and text.strip()


# ---------------------------------------------------------------------------
# "degrading is fine, degrading silently is not"
# ---------------------------------------------------------------------------

@given(st.lists(st.integers(min_value=0, max_value=25), min_size=2, max_size=25))
@settings(max_examples=30, deadline=None)
def test_load_file_reports_every_index_repair_it_makes(offsets):
    """Out-of-order rows get sorted and duplicate timestamps get kept, both
    silently, and both change which row is "the next row" for anything that
    walks by row position. Whenever the file as written differs from the
    frame as returned, the difference has to be on the record.
    """
    base = pd.Timestamp("2024-01-01")
    times = [base + pd.Timedelta(hours=int(o)) for o in offsets]
    rows = ["time,value"]
    for i, t in enumerate(times):
        rows.append(f"{t},{i}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "idx.csv")
        with open(path, "w") as handle:
            handle.write("\n".join(rows) + "\n")
        df = load_file(path)

    issues = df.attrs.get("load_issues", [])
    if times != sorted(times):
        assert any("chronological order" in i for i in issues), issues
    if len(set(times)) != len(times):
        assert any("repeat a timestamp" in i for i in issues), issues
    assert df.index.is_monotonic_increasing
