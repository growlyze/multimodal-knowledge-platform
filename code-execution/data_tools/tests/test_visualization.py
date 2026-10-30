"""Tests for data_tools.visualization.

Charts are checked for the things that can actually break in a headless
sandbox: a real Figure comes back, it has drawn content, and bad or empty
input still returns a Figure instead of raising.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from data_tools.visualization import plot_bar, plot_heatmap, plot_line

from .conftest import make_series


@pytest.fixture(autouse=True)
def close_figures():
    yield
    matplotlib.pyplot.close("all")


def test_headless_backend_selected():
    """The sandbox has no display; Agg must be active or every plot raises."""
    assert matplotlib.get_backend().lower() == "agg"


def test_line_chart_returns_figure_with_drawn_content():
    df = make_series(rows=50)
    fig = plot_line(df, y="value", title="My Line")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.lines, "line chart should have drawn a line"
    assert ax.get_title() == "My Line"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), df["value"].to_numpy())


def test_line_chart_plots_every_numeric_column_by_default():
    df = make_series(rows=30)
    fig = plot_line(df)
    assert len(fig.axes[0].lines) == 2        # value + count
    assert fig.axes[0].get_legend() is not None


def test_line_chart_accepts_a_column_list():
    df = make_series(rows=30)
    fig = plot_line(df, y=["value"])
    assert len(fig.axes[0].lines) == 1


def test_bar_chart_returns_figure():
    df = pd.DataFrame({"category": ["a", "b", "c"], "total": [1.0, 3.0, 2.0]})
    fig = plot_bar(df, x="category", y="total", title="Totals")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.containers, "bars should have been drawn"
    heights = [p.get_height() for p in ax.containers[0]]
    assert heights == [1.0, 3.0, 2.0]
    assert ax.get_ylabel() == "total"


def test_heatmap_draws_an_image_with_axis_labels():
    frame = pd.DataFrame({
        "alpha": ["a", "a", "b", "b"],
        "beta": [1, 2, 1, 2],
        "score": [0.4, 0.6, 0.5, 0.7],
    })
    fig = plot_heatmap(frame, "beta", "alpha", "score")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.images, "heatmap should have drawn an image"
    assert ax.images[0].get_array().shape == (2, 2)
    assert ax.get_xlabel() == "beta"
    np.testing.assert_allclose(ax.images[0].get_array(), [[0.4, 0.6], [0.5, 0.7]])


def test_empty_inputs_still_return_a_figure():
    empty = pd.DataFrame()
    assert isinstance(plot_line(empty), Figure)
    assert isinstance(plot_bar(empty), Figure)
    assert isinstance(plot_heatmap(empty, "a", "b", "c"), Figure)


def test_malformed_inputs_do_not_raise():
    junk = pd.DataFrame({"foo": ["x", "y"]})

    assert isinstance(plot_line(junk), Figure)          # no numeric columns
    assert isinstance(plot_line(junk, y="missing"), Figure)
    assert isinstance(plot_bar(junk, x="missing", y="missing"), Figure)
    assert isinstance(plot_heatmap(junk, "x", "y", "z"), Figure)  # missing cols


def test_figures_can_be_saved(tmp_path):
    """The sandbox writes charts to disk, so saving must actually work."""
    df = make_series(rows=50)
    for fig in (plot_line(df), plot_bar(df, y="value")):
        path = tmp_path / "chart.png"
        fig.savefig(path)
        assert path.exists() and path.stat().st_size > 0
        path.unlink()
