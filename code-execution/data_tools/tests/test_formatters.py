"""Tests for data_tools.formatters - number and table formatting.

The formatters turn numeric outputs into prose a reader can trust: NaN/inf
never leak a raw "nan", a currency renders with its own decimals, and a
truncated table always states its true total.
"""

import math

import numpy as np
import pandas as pd

from data_tools.formatters import (
    format_money,
    format_number,
    format_percent,
    format_plain,
    format_ratio,
    format_signed,
    format_table_text,
)


# ── format_number ─────────────────────────────────────────────────────

def test_number_passes_through_finite_floats():
    assert format_number(1234.5) == 1234.5
    assert format_number("1234.5") == 1234.5


def test_number_maps_junk_to_the_default():
    assert format_number(None) == 0.0
    assert format_number(float("nan")) == 0.0
    assert format_number(float("inf")) == 0.0
    assert format_number(float("-inf")) == 0.0
    assert format_number("not a number") == 0.0
    assert format_number(object(), default=-1.0) == -1.0


# ── format_percent ────────────────────────────────────────────────────

def test_percent_is_one_decimal():
    assert format_percent(12.34) == "12.3%"
    assert format_percent(0.0) == "0.0%"
    assert format_percent(100) == "100.0%"


def test_percent_survives_junk():
    assert format_percent(None) == "0.0%"
    assert format_percent(float("nan")) == "0.0%"


# ── format_money ──────────────────────────────────────────────────────

def test_usd_money_is_byte_identical():
    assert format_money(1234.5) == "+$1,234.50"
    assert format_money(-1234.5) == "-$1,234.50"
    assert format_money(0) == "+$0.00"


def test_btc_uses_eight_decimals_and_no_dollar_sign():
    out = format_money(0.00003, "BTC")
    assert out == "+0.00003000 BTC"
    assert "$" not in out


def test_unknown_currency_never_prints_a_dollar_sign():
    out = format_money(1234.5, None)
    assert "$" not in out
    assert out == "+1,234.50"


def test_money_tolerates_junk():
    assert format_money(None) == "+$0.00"
    assert format_money(float("inf")) == "+$0.00"


# ── format_plain / format_signed / format_ratio ───────────────────────

def test_plain_has_no_float_residue():
    assert format_plain(0.25) == "0.25"
    assert format_plain(15.0) == "15"
    assert format_plain(0.1 + 0.2) == "0.3"


def test_signed_shows_the_sign():
    assert format_signed(1234.5) == "+1,234.50"
    assert format_signed(-5) == "-5.00"
    assert format_signed(0.004) == "+0.004"
    assert format_signed(0) == "+0.00"


def test_ratio_renders_na_not_infinity():
    assert format_ratio(float("inf")) == "N/A"
    assert format_ratio(float("-inf")) == "N/A"
    assert format_ratio(float("nan")) == "N/A"
    assert format_ratio(None) == "N/A"
    assert format_ratio(1.2345) == "1.23"
    assert format_ratio(0) == "0.00"


# ── format_table_text ─────────────────────────────────────────────────

def test_table_truncates_and_states_the_true_total():
    frame = pd.DataFrame({"a": range(30), "b": range(30)})
    text = format_table_text(frame, max_rows=5)
    assert "(30 total, showing first 5)" in text
    assert len(text.splitlines()) == 2 + 5       # header + column names + rows


def test_table_without_truncation_states_plain_total():
    frame = pd.DataFrame({"a": [1, 2]})
    text = format_table_text(frame, max_rows=20)
    assert "(2 total)" in text and "showing first" not in text


def test_table_numbers_have_no_float_residue():
    frame = pd.DataFrame({"value": [0.1 + 0.2, 15.0]})
    text = format_table_text(frame)
    assert "0.3" in text
    assert "0.30000000000000004" not in text


def test_table_of_junk_or_empty():
    assert format_table_text(pd.DataFrame()) == "No rows."
    assert format_table_text(None) == "No rows."
    assert "No rows" not in format_table_text(pd.DataFrame({"a": ["x"]}))
