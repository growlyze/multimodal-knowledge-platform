"""Tests for data_tools.code_validator - AST-based rejection of manual
reimplementations.

Each test sends a source string through validate_code() and asserts either
(True, "") for valid code or (False, <substring>) for a specific rejection.
"""

import pytest

from data_tools.code_validator import validate_code


# ── helpers ───────────────────────────────────────────────────────────

def assert_rejected(source: str, *fragments: str):
    """Assert the code is rejected and the message contains every fragment."""
    ok, msg = validate_code(source)
    assert not ok, "Expected rejection but code was accepted"
    for frag in fragments:
        assert frag in msg, f"Expected {frag!r} in rejection message, got: {msg}"


def assert_accepted(source: str):
    ok, msg = validate_code(source)
    assert ok, f"Expected acceptance but code was rejected: {msg}"


# ── Rule 1: must import data_tools ────────────────────────────────────

class TestImportRequirement:

    def test_valid_import_statement(self):
        assert_accepted("""\
import data_tools
df = data_tools.load_file("data.csv")
summary = data_tools.describe_file(df)
print(summary)
""")

    def test_valid_from_import(self):
        assert_accepted("""\
from data_tools import load_file, describe_file
df = load_file("data.csv")
info = describe_file(df)
print(info)
""")

    def test_no_import_rejected(self):
        assert_rejected("""\
import pandas as pd
df = pd.read_csv("data.csv")
result = df.groupby("category").sum()
print(result)
""", "Import data_tools first")

    def test_short_inspection_allowed(self):
        """< 5 lines of simple inspection needs no import."""
        assert_accepted("""\
print(df.head())
print(df.shape)
""")

    def test_single_print_allowed(self):
        assert_accepted("print(df.describe())")

    def test_stdlib_only_allowed(self):
        """Pure stdlib code with no data analysis is fine."""
        assert_accepted("""\
import math
import json
result = math.sqrt(144)
print(json.dumps({"value": result}))
""")

    def test_numpy_without_data_tools_rejected(self):
        """Importing a non-stdlib library without data_tools is rejected."""
        assert_rejected("""\
import numpy as np
import pandas as pd
df = pd.read_csv("data.csv")
total = np.where(df['value'] > 0, df['value'], 0)
print(total.sum())
""", "Import data_tools first")


# ── Rule 2: ban shadowing library / parser names ──────────────────────

class TestShadowing:

    def test_redefine_load_file(self):
        assert_rejected("""\
from data_tools import describe_file
def load_file(path):
    return path
""", "Do not redefine", "load_file")

    def test_redefine_describe_file(self):
        assert_rejected("""\
from data_tools import load_file
def describe_file(df):
    return df.shape
""", "Do not redefine", "describe_file")

    def test_parser_alias_rejected_with_parser_message(self):
        assert_rejected("""\
from data_tools import load_file
def parse_csv(path):
    rows = []
    for line in open(path):
        rows.append(line.strip())
    return rows
""", "Do not reimplement stdlib json/csv parsing")

    def test_parse_json_alias_rejected(self):
        assert_rejected("""\
from data_tools import load_file
def parse_json(path):
    return {}
""", "Do not reimplement stdlib json/csv parsing")

    def test_helper_function_with_unrelated_name_accepted(self):
        """A function named 'prepare_data' is not a library name."""
        assert_accepted("""\
from data_tools import load_file, describe_file
def prepare_data(path):
    df = load_file(path)
    return describe_file(df)
result = prepare_data("data.csv")
""")


# ── Rule 3: ban manual row loops ──────────────────────────────────────

class TestManualRowLoops:

    def test_iterrows_rejected(self):
        assert_rejected("""\
from data_tools import load_file
df = load_file("data.csv")
for i, row in df.iterrows():
    print(row['value'])
""", "vectorized", "iterrows")

    def test_itertuples_rejected(self):
        assert_rejected("""\
from data_tools import load_file
df = load_file("data.csv")
for row in df.itertuples():
    print(row)
""", "vectorized", "itertuples")

    def test_range_len_subscript_rejected(self):
        assert_rejected("""\
from data_tools import load_file
df = load_file("data.csv")
for i in range(len(df)):
    print(df['value'][i])
""", "vectorized")

    def test_plain_range_loop_accepted(self):
        """A range loop over anything but a frame is not our business."""
        assert_accepted("""\
from data_tools import load_file
df = load_file("data.csv")
total = 0
for i in range(10):
    total += i
print(total)
""")

    def test_vectorized_code_accepted(self):
        assert_accepted("""\
from data_tools import load_file
df = load_file("data.csv")
df['double'] = df['value'] * 2
grouped = df.groupby('category')['value'].mean()
print(grouped)
""")

    def test_apply_accepted(self):
        assert_accepted("""\
from data_tools import load_file
df = load_file("data.csv")
df['label'] = df['value'].apply(lambda v: 'high' if v > 10 else 'low')
print(df.head())
""")


# ── Rule 4: ban hand-rolled stdlib parsing ────────────────────────────

class TestManualParsing:

    def test_open_and_split_lines_rejected(self):
        assert_rejected("""\
from data_tools import load_file
rows = []
for line in open("data.csv"):
    rows.append(line.split(","))
""", "Do not reimplement stdlib json/csv parsing")

    def test_split_a_plain_string_accepted(self):
        """Splitting a string literal has nothing to do with parsing a file."""
        assert_accepted("""\
from data_tools import load_file
parts = "a,b,c".split(",")
print(parts)
""")

    def test_json_by_hand_rejected(self):
        assert_rejected("""\
from data_tools import load_file
out = []
for line in open("data.json"):
    out.append(line.split(","))
""", "json.load")


# ── edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_syntax_error_rejected(self):
        ok, msg = validate_code("def foo(:")
        assert not ok
        assert "syntax error" in msg.lower()

    def test_empty_source(self):
        assert_accepted("")

    def test_comment_only(self):
        assert_accepted("# just a comment\n")

    def test_mixed_valid_and_invalid(self):
        """Valid library use plus one manual row loop => rejected."""
        assert_rejected("""\
from data_tools import load_file, describe_file
df = load_file("data.csv")
info = describe_file(df)
# but then walks the frame by hand
for i, row in df.iterrows():
    print(row)
""", "vectorized")

    def test_full_valid_pipeline(self):
        """A realistic complete pipeline using library functions."""
        assert_accepted("""\
from data_tools import (
    load_file, describe_file, format_file_summary,
    export_xlsx, plot_line, format_table_text,
)

df = load_file("data.csv")
info = describe_file(df)
print(format_file_summary(info))

df['rolling'] = df['value'].rolling(5).mean()
print(format_table_text(df.head()))
export_xlsx({"Data": df}, path="/home/user/outputs/report.xlsx")
plot_line(df).savefig("/home/user/outputs/chart.png")
""")
