"""data_tools: generic data loading, inspection, export and plotting.

A small toolkit for tabular data in the code-execution sandbox:

    load_file()            read CSV/Excel/gzip files (robust timestamp parsing)
    load_dataset()         pull a stored dataset through the Haystack data API
    resample()             change the time grid of a datetime-indexed frame
    describe_file()        every fact a file summary may state, from the data
    format_file_summary()  render that as text
    export_xlsx()          write DataFrames to a styled .xlsx workbook
    plot_line/bar/heatmap  headless matplotlib charts returning Figures
    format_*()             number and table formatting helpers
    validate_code()        AST guardrail for generated code

Every tool degrades visibly instead of raising: failures land on
df.attrs["load_issues"] (or a blank Figure with an explanation).
"""

from .code_validator import validate_code
from .export import export_xlsx
from .file_summary import describe_file, format_file_summary
from .formatters import (
    CURRENCY_DECIMALS,
    format_money,
    format_number,
    format_percent,
    format_plain,
    format_ratio,
    format_signed,
    format_table_text,
)
from .loader import load_dataset, load_file, resample
from .logging_decorator import logged, logger, missing_columns
from .visualization import plot_bar, plot_heatmap, plot_line

__version__ = "2.0.0"

__all__ = [
    "CURRENCY_DECIMALS",
    "describe_file",
    "export_xlsx",
    "format_file_summary",
    "format_money",
    "format_number",
    "format_percent",
    "format_plain",
    "format_ratio",
    "format_signed",
    "format_table_text",
    "list_functions",
    "load_dataset",
    "load_file",
    "logged",
    "logger",
    "missing_columns",
    "plot_bar",
    "plot_heatmap",
    "plot_line",
    "resample",
    "validate_code",
]

_SIGNATURES = {
    "load_dataset": "load_dataset(name, api_url=None)",
    "load_file": "load_file(filepath)",
    "resample": "resample(df, timeframe='1min', agg=None)",
    "describe_file": "describe_file(path_or_df)",
    "format_file_summary": "format_file_summary(info)",
    "export_xlsx": "export_xlsx(sheets, path='/tmp/export.xlsx', column_formats=None)",
    "plot_line": "plot_line(frame, y=None, title='Line Chart')",
    "plot_bar": "plot_bar(frame, x=None, y=None, title='Bar Chart')",
    "plot_heatmap": "plot_heatmap(frame, x_col, y_col, value_col, title='Heatmap')",
    "format_number": "format_number(value, default=0.0)",
    "format_percent": "format_percent(value)",
    "format_money": "format_money(value, currency='USD')",
    "format_plain": "format_plain(value)",
    "format_signed": "format_signed(value)",
    "format_ratio": "format_ratio(value)",
    "format_table_text": "format_table_text(frame, max_rows=20, title='Rows')",
    "validate_code": "validate_code(source)",
}


def list_functions():
    """One callable-signature line per public function, for discovery."""
    return [f"{_SIGNATURES[name]} - {globals()[name].__doc__.strip().splitlines()[0]}"
            for name in _SIGNATURES]
