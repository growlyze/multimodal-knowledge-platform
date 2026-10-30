"""AST-based code validator for LLM-generated data-analysis code.

Parses the source with the ``ast`` module and rejects it when it contains
patterns that reimplement what data_tools (or the standard library) already
provides.  The rejection message is actionable: it names the exact function
the model should call instead.

False positives (blocking valid code) are worse than false negatives (missing
a reimplementation), so every rule errs on the side of allowing.

Usage::

    from data_tools.code_validator import validate_code

    ok, msg = validate_code(source_string)
    if not ok:
        # feed *msg* back to the model as a system/tool message
        ...
"""

from __future__ import annotations

import ast
from typing import Tuple

# ── exports from data_tools that must not be shadowed ─────────────────
# Kept in sync with __init__.__all__; only the callable names matter here.
LIBRARY_FUNCTIONS = {
    "load_file", "load_dataset", "resample",
    "describe_file", "format_file_summary",
    "export_xlsx",
    "plot_line", "plot_bar", "plot_heatmap",
    "format_number", "format_percent", "format_money", "format_plain",
    "format_signed", "format_ratio", "format_table_text",
    "validate_code", "list_functions", "logged", "missing_columns",
}

# Common aliases an LLM might define that shadow library intent.
_SHADOW_ALIASES = {
    "describe", "summarize_file", "file_summary",
    "load_csv", "read_csv_data", "to_excel", "export_excel", "write_excel",
    "format_num", "format_pct", "format_money", "plot_chart", "plot_graph",
}
# Names that would reimplement standard-library / pandas parsing by hand.
_PARSER_ALIASES = {
    "parse_csv", "parse_json", "load_json", "read_json", "to_json",
    "parse_tsv", "split_csv",
}
_SHADOW_NAMES = LIBRARY_FUNCTIONS | _SHADOW_ALIASES | _PARSER_ALIASES

# Manual CSV-ish field splitting: .split(",") and friends.
_SPLIT_SEPARATORS = {",", ";", "\t", "|"}

# Standard-library modules that simple scripts legitimately import.
_STDLIB_MODULES = {
    "math", "datetime", "json", "csv", "os", "sys", "io", "re",
    "collections", "itertools", "functools", "operator", "decimal",
    "statistics", "pathlib", "copy", "pprint", "textwrap", "string",
    "time", "calendar", "random", "typing", "dataclasses", "enum",
    "abc", "contextlib", "warnings", "logging", "struct", "array",
}


def _available_functions_hint() -> str:
    return (
        "Import data_tools first. Available functions include: "
        "load_file(), load_dataset(), resample(), describe_file(), "
        "format_file_summary(), export_xlsx(), plot_line(), plot_bar(), "
        "plot_heatmap(), format_number(), format_table_text(). "
        "See list_functions() for the full catalogue."
    )


# ── helpers ───────────────────────────────────────────────────────────

def _is_simple_inspection(tree: ast.Module) -> bool:
    """True when the code is short and just prints/inspects data."""
    # Count non-empty, non-comment source lines via the AST body.
    if len(tree.body) > 8:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
    return True


def _imports_data_tools(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "data_tools" or alias.name.startswith("data_tools."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "data_tools"
                or node.module.startswith("data_tools.")
            ):
                return True
    return False


def _only_imports_stdlib(tree: ast.Module) -> bool:
    """True when every import is from stdlib (no data analysis happening)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _STDLIB_MODULES:
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in _STDLIB_MODULES:
                    return False
    return True


def _source_lines(source: str) -> int:
    """Count non-blank, non-comment source lines."""
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def _loop_iterates_range_of_len(node: ast.For) -> "str | None":
    """The name `for i in range(len(name)):` iterates over, or None."""
    it = node.iter
    if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
            and it.func.id == "range" and len(it.args) == 1):
        inner = it.args[0]
        if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                and inner.func.id == "len" and len(inner.args) == 1
                and isinstance(inner.args[0], ast.Name)):
            return inner.args[0].id
    return None


def _body_subscripts_name(node: ast.For, name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
            if child.value.id == name:
                return True
    return False


# ── per-rule checkers ─────────────────────────────────────────────────
# Each returns (False, message) on a violation, or (True, "") otherwise.

def _check_shadowed_names(tree: ast.Module) -> Tuple[bool, str]:
    """Reject function defs that shadow library or stdlib-parser names."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _PARSER_ALIASES:
                return (
                    False,
                    "Do not reimplement stdlib json/csv parsing. Use "
                    "json.load()/csv.reader()/pandas.read_csv() or "
                    "`load_file()` from data_tools."
                )
            if node.name in _SHADOW_NAMES:
                return (
                    False,
                    f"Do not redefine `{node.name}`. "
                    f"Import it from data_tools."
                )
    return True, ""


def _check_manual_row_loops(tree: ast.Module) -> Tuple[bool, str]:
    """Ban manual row-by-row iteration over a DataFrame."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("iterrows", "itertuples"):
                return (
                    False,
                    "Use pandas vectorized operations instead of manual row "
                    f"loops (.{node.func.attr}()) - e.g. df['a'] + df['b'], "
                    "df.apply(...), or df.groupby(...)."
                )
        if isinstance(node, ast.For):
            name = _loop_iterates_range_of_len(node)
            if name is not None and _body_subscripts_name(node, name):
                return (
                    False,
                    "Use pandas vectorized operations instead of manual row "
                    "loops (for i in range(len(df)) with df[...] inside) - "
                    "e.g. df['a'] + df['b'], df.apply(...), or df.groupby(...)."
                )
    return True, ""


def _check_manual_parsing(tree: ast.Module) -> Tuple[bool, str]:
    """Ban hand-rolled CSV splitting of a file being read."""
    opens_file = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open"
        for n in ast.walk(tree)
    )
    if not opens_file:
        return True, ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "split" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and arg.value in _SPLIT_SEPARATORS:
                    return (
                        False,
                        "Do not reimplement stdlib json/csv parsing - reading "
                        "a file and splitting lines by hand. Use "
                        "json.load()/csv.reader()/pandas.read_csv() or "
                        "`load_file()` from data_tools."
                    )
    return True, ""


# ── main entry point ──────────────────────────────────────────────────

def validate_code(source: str) -> Tuple[bool, str]:
    """Parse *source* and check for banned patterns.

    Returns ``(True, "")`` if code is acceptable.
    Returns ``(False, rejection_message)`` if code violates rules.
    The *rejection_message* tells the model exactly which function to use
    instead.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"Code has a syntax error: {exc}"

    lines = _source_lines(source)

    # Rule 1: must import data_tools (with exceptions).
    if not _imports_data_tools(tree):
        # Exception: very short data-inspection with no non-stdlib imports.
        if lines < 5 and _is_simple_inspection(tree) and _only_imports_stdlib(tree):
            pass  # allow
        elif _only_imports_stdlib(tree):
            pass  # allow
        else:
            return False, _available_functions_hint()

    # Rule 2: function defs shadowing library / parser names.
    ok, msg = _check_shadowed_names(tree)
    if not ok:
        return ok, msg

    # Rule 3: manual row loops instead of pandas vectorized ops.
    ok, msg = _check_manual_row_loops(tree)
    if not ok:
        return ok, msg

    # Rule 4: hand-rolled json/csv parsing.
    ok, msg = _check_manual_parsing(tree)
    if not ok:
        return ok, msg

    return True, ""
