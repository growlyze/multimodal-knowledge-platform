"""Property-based tests (Hypothesis) for data_tools pure functions.

Unlike test_loader.py / test_describe_file.py, which pin exact outputs against
hand-built fixtures, these generate hundreds of random (but valid) inputs per
run and assert invariants that must hold for ALL of them - the general form of
bugs a fixed fixture can miss.
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from data_tools import describe_file, format_file_summary, resample
from data_tools.formatters import (
    format_money,
    format_number,
    format_percent,
    format_ratio,
    format_table_text,
)

_FINITE = st.floats(min_value=-1e9, max_value=1e9,
                    allow_nan=False, allow_infinity=False, allow_subnormal=False)


# ── formatters ────────────────────────────────────────────────────────

@given(st.one_of(_FINITE, st.floats(), st.text(max_size=10), st.none()))
@settings(max_examples=200, deadline=None)
def test_format_number_always_returns_a_finite_number(value):
    """Whatever comes in, what comes out is something a formatter can print
    and json can serialise - the silent-"nan" shape never leaks."""
    out = format_number(value)
    assert isinstance(out, float) and math.isfinite(out)


@given(_FINITE, st.sampled_from(["USD", "BTC", "EUR", None, "", "XYZ"]))
@settings(max_examples=200, deadline=None)
def test_format_money_sign_and_unit_are_consistent(value, currency):
    out = format_money(value, currency)
    assert out.startswith("+") == (value >= 0) or value == 0
    if currency == "USD":
        assert out.lstrip("+-").startswith("$")
    elif currency:
        assert out.endswith(" " + currency)
    else:
        assert "$" not in out
    assert "nan" not in out.lower()


@given(_FINITE)
@settings(max_examples=200, deadline=None)
def test_format_percent_always_one_decimal(value):
    out = format_percent(value)
    assert out.endswith("%") and len(out.split(".")[1]) == 2   # one decimal + %


@given(st.one_of(_FINITE, st.floats(), st.none()))
@settings(max_examples=200, deadline=None)
def test_format_ratio_is_finite_or_na(value):
    out = format_ratio(value)
    assert out == "N/A" or math.isfinite(float(out))


@given(st.lists(_FINITE, min_size=1, max_size=40), st.integers(min_value=1, max_value=20))
@settings(max_examples=50, deadline=None)
def test_format_table_text_always_states_the_total(values, max_rows):
    frame = pd.DataFrame({"value": values, "other": range(len(values))})
    text = format_table_text(frame, max_rows=max_rows)
    header = text.splitlines()[0]
    assert "(%d total" % len(values) in header
    body = text.splitlines()[2:]
    assert len(body) <= max_rows


# ── describe_file ─────────────────────────────────────────────────────

@st.composite
def timestamped_frames(draw, min_size=2, max_size=60):
    """A frame with a sorted DatetimeIndex of random, possibly gappy spacing."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    steps = draw(st.lists(st.integers(min_value=1, max_value=200), min_size=n - 1, max_size=n - 1))
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    idx = idx + pd.to_timedelta(np.cumsum([0] + steps), unit="min")
    values = draw(st.lists(_FINITE, min_size=n, max_size=n))
    return pd.DataFrame({"value": values}, index=idx)


@given(timestamped_frames())
@settings(max_examples=100, deadline=None)
def test_describe_file_gap_invariants_hold(df):
    info = describe_file(df)
    assert info["rows"] == len(df)
    gaps = info["gaps"]
    # kinds are from the documented vocabulary only
    assert all(g["kind"] in ("short_gap", "long_gap") for g in gaps)
    # every gap is a positive duration, classified consistently
    for g in gaps:
        assert g["minutes"] > 0
        assert (g["kind"] == "long_gap") == (g["minutes"] >= 24 * 60)
        assert g["missing_minutes"] >= 0
    # counts by kind cover every gap exactly once
    assert sum(info["gap_counts_by_kind"].values()) == len(gaps)
    # and the rendered summary is complete prose
    text = format_file_summary(info)
    assert "Problems found" in text and "Rows: %d" % len(df) in text


@given(timestamped_frames())
@settings(max_examples=100, deadline=None)
def test_describe_file_problems_are_sentences(df):
    info = describe_file(df)
    assert isinstance(info["problems"], list)
    assert all(isinstance(p, str) and p.strip() for p in info["problems"])


# ── resample ──────────────────────────────────────────────────────────

@given(st.lists(_FINITE, min_size=6, max_size=60),
       st.sampled_from(["5min", "15min", "1h"]))
@settings(max_examples=50, deadline=None)
def test_resample_sum_aggregation_preserves_the_total(values, timeframe):
    """A sum over the fine grid must equal the sum over the coarse grid -
    resampling may lose resolution but never data."""
    idx = pd.date_range("2024-01-01", periods=len(values), freq="min")
    df = pd.DataFrame({"total": values}, index=idx)
    out = resample(df, timeframe, agg={"total": "sum"})
    assert out["total"].sum() == pytest.approx(sum(values), rel=1e-9, abs=1e-6)
    assert len(out) <= len(df)


@given(st.lists(_FINITE, min_size=2, max_size=30))
@settings(max_examples=50, deadline=None)
def test_resample_never_returns_more_rows_than_it_got(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="min")
    df = pd.DataFrame({"value": values}, index=idx)
    out = resample(df, "10min")
    assert 0 < len(out) <= len(df)
