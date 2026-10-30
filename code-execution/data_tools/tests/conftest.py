"""Shared test fixtures.

Synthetic data only - the tests never need an external file or a network call.
This is the ONLY place a random seed appears: the tool functions themselves
contain no randomness at all.
"""

import numpy as np
import pandas as pd
import pytest


def make_series(rows=100, start="2022-01-01", freq="5min", seed=42):
    """Generate a generic synthetic time-series frame.

    Seeded, so every run produces the same rows. Two numeric columns of
    different scales - nothing market-specific, just tabular data with a
    DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=rows, freq=freq)
    value = 100 + np.cumsum(rng.standard_normal(rows))
    count = rng.integers(1, 1000, rows)
    return pd.DataFrame({"value": value, "count": count}, index=dates)


def make_table(rows=50, seed=7):
    """A plain table with no time column: numbers, text and a category."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "amount": rng.normal(100, 20, rows).round(2),
        "quantity": rng.integers(1, 100, rows),
        "category": rng.choice(["alpha", "beta", "gamma"], rows),
    })


@pytest.fixture
def series_frame():
    return make_series(rows=200)


@pytest.fixture
def table_frame():
    return make_table(rows=50)
