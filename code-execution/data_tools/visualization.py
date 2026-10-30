"""Generic matplotlib charts for tabular data.

Every function returns a matplotlib Figure - nothing is shown or written to
disk, so the caller decides what to do with it (fig.savefig(...) in the
sandbox, or let the notebook display it).

The Agg backend is selected before pyplot is imported because the sandbox is
headless; without it matplotlib tries to find a GUI toolkit and fails.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

from .logging_decorator import logged, missing_columns          # noqa: E402

LINE_COLOR = "#1f77b4"
UP_COLOR = "#26a69a"
BAR_COLOR = "#1f77b4"
FIGSIZE = (12, 6)


def _blank(title, message):
    """A real Figure carrying an explanatory message, so callers never get None."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, color="#888")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def _numeric_columns(frame):
    return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]


@logged
def plot_line(frame, y=None, title="Line Chart"):
    """Plot one or more numeric columns as lines over the row order/index.

    Args:
        frame: DataFrame, ideally with a DatetimeIndex.
        y: column name or list of names to plot. None plots every numeric
            column.
        title: chart title.

    Returns a matplotlib figure. Empty or unusable input returns a Figure
    carrying a message instead of raising.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return _blank(title, "No data to plot")
    columns = _numeric_columns(frame) if y is None else (
        [y] if isinstance(y, str) else list(y))
    missing = [c for c in columns if c not in frame.columns]
    if missing or not columns:
        return _blank(title, "No numeric columns to plot" if not columns
                      else f"Missing columns: {', '.join(map(str, missing))}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = frame.index
    for col in columns:
        ax.plot(x, pd.to_numeric(frame[col], errors="coerce").to_numpy(),
                color=LINE_COLOR, linewidth=1.6, label=str(col))
    if len(columns) > 1:
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_ylabel(", ".join(str(c) for c in columns) if len(columns) <= 3 else "value")
    ax.grid(alpha=0.25)
    if isinstance(x, pd.DatetimeIndex):
        fig.autofmt_xdate()
    fig.tight_layout()
    return fig


@logged
def plot_bar(frame, x=None, y=None, title="Bar Chart"):
    """Plot a bar chart of one numeric column against a categorical/x column.

    Args:
        frame: DataFrame.
        x: column whose values label the bars (None: the index).
        y: numeric column to size the bars (None: first numeric column).
        title: chart title.

    Returns a matplotlib figure. Empty or unusable input returns a Figure
    carrying a message instead of raising.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return _blank(title, "No data to plot")
    if y is None:
        numeric = _numeric_columns(frame)
        if not numeric:
            return _blank(title, "No numeric columns to plot")
        y = numeric[0]
    if y not in frame.columns or (x is not None and x not in frame.columns):
        return _blank(title, "Missing columns to plot")

    labels = frame[x] if x is not None else frame.index
    values = pd.to_numeric(frame[y], errors="coerce").to_numpy(float)
    positions = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(positions, np.nan_to_num(values), width=0.6, color=BAR_COLOR, linewidth=0)
    step = max(1, len(frame) // 15)
    ax.set_xticks(positions[::step])
    ax.set_xticklabels([str(t)[:16] for t in labels[::step]], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(str(y))
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


@logged
def plot_heatmap(frame, x_col, y_col, value_col, title="Heatmap"):
    """Plot a heatmap of a numeric column across two other columns.

    Args:
        frame: DataFrame in "long" form.
        x_col, y_col: column names to put on the two axes.
        value_col: which numeric column to colour by.
        title: chart title.

    Returns a matplotlib figure. If the data varies more than these two
    columns, each cell shows the MEAN of value_col over the others. Missing
    required columns returns a Figure carrying a message instead of raising.
    """
    needed = [x_col, y_col, value_col]
    if missing_columns(frame, needed, "plot_heatmap"):
        return _blank(title, "Data missing required columns")

    table = frame.pivot_table(index=y_col, columns=x_col,
                              values=value_col, aggfunc="mean")
    if table.empty:
        return _blank(title, "No values to plot")

    values = table.to_numpy(float)
    fig, ax = plt.subplots(figsize=(max(6, table.shape[1] * 1.2),
                                    max(5, table.shape[0] * 0.9)))
    image = ax.imshow(values, cmap="RdYlGn", aspect="auto", origin="lower")
    fig.colorbar(image, ax=ax, label=str(value_col))

    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index])
    ax.set_xlabel(str(x_col))
    ax.set_ylabel(str(y_col))
    ax.set_title(f"{title} - {value_col}")

    finite = values[np.isfinite(values)]
    midpoint = (finite.max() + finite.min()) / 2.0 if finite.size else 0.0
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = values[row, col]
            if not np.isfinite(value):
                continue
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if value > midpoint else "white")

    fig.tight_layout()
    return fig
