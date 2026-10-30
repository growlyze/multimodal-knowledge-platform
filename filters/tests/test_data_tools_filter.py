"""data_tools_filter.py: when it injects the data_tools reference, and what it names."""
import ast
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_tools_filter as dtf
from data_tools_filter import DATA_TOOLS_BLOCK, OWN_MARKER, Filter

DATA_TOOLS_INIT = (Path(__file__).resolve().parents[2]
                   / "code-execution" / "data_tools" / "__init__.py")


def _run(body, user=None, filt=None):
    asyncio.run((filt or Filter()).inlet(body, user))
    return body


def _system(body):
    m = dtf._system_message(body["messages"])
    return m["content"] if m else ""


def _exported_names():
    tree = ast.parse(DATA_TOOLS_INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "__all__" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError("no __all__ in data_tools/__init__.py")


def test_block_names_only_exported_functions():
    exported = _exported_names()
    import_block = DATA_TOOLS_BLOCK.split("from data_tools import (", 1)[1].split(")\n```", 1)[0]
    names = set(re.findall(r"^\s+(\w+)(?:,|\s)", import_block, re.M))
    names |= set(re.findall(r"\b(format_\w+)\b", import_block))
    assert names, "no names parsed"
    assert names <= exported, names - exported


def test_compute_question_gets_the_block():
    body = {"messages": [{"role": "user", "content": "calculate the monthly totals"}]}
    s = _system(_run(body))
    assert OWN_MARKER in s
    assert "from data_tools import" in s


def test_plain_knowledge_question_gets_nothing():
    body = {"messages": [{"role": "user", "content": "what is a data retention policy?"}]}
    assert _system(_run(body)) == ""


def test_attached_csv_gets_block_and_file_summary_hint():
    body = {"files": [{"name": "sales_2024.csv"}],
            "messages": [{"role": "user", "content": "What's in this file?"}]}
    s = _system(_run(body))
    assert OWN_MARKER in s
    assert "sales_2024.csv" in s
    assert "format_file_summary(describe_file(path))" in s


def test_rag_filter_data_file_note_counts_as_an_attachment():
    q = ("Summarise it\n\n[System note: report.xlsx attached as data file(s). "
         "The rows are NOT in this prompt.]")
    body = {"messages": [{"role": "user", "content": q}]}
    s = _system(_run(body))
    assert "report.xlsx" in s and "describe_file" in s


def test_non_data_attachment_is_not_a_data_file():
    body = {"files": [{"name": "notes.pdf"}],
            "messages": [{"role": "user", "content": "what does it say?"}]}
    assert _system(_run(body)) == ""


def test_appends_to_existing_system_prompt_and_is_idempotent():
    body = {"messages": [{"role": "system", "content": "You are helpful."},
                         {"role": "user", "content": "plot the revenue"}]}
    _run(body)
    _run(body)
    s = _system(body)
    assert s.startswith("You are helpful.")
    assert s.count(OWN_MARKER) == 1


def test_stands_down_when_rag_filter_already_covered_the_sandbox():
    body = {"messages": [{"role": "system", "content": dtf.RAG_FILTER_MARKER + " ..."},
                         {"role": "user", "content": "compute the average"}]}
    assert OWN_MARKER not in _system(_run(body))


def test_rag_filter_marker_matches_rag_filter_hint():
    import rag_filter
    assert dtf.RAG_FILTER_MARKER in rag_filter.CODE_TOOLS_HINT.replace("\n", "")
    assert dtf.RAG_FILTER_MARKER in rag_filter.CODE_TOOLS_LIGHT.replace("\n", "")


def test_valves_and_user_valves_switch_it_off():
    f = Filter()
    f.valves.enabled = False
    body = {"messages": [{"role": "user", "content": "compute the average"}]}
    assert _system(_run(body, filt=f)) == ""
    body = {"messages": [{"role": "user", "content": "compute the average"}]}
    assert _system(_run(body, user={"valves": {"enabled": False}})) == ""


def test_file_summary_hint_can_be_disabled():
    f = Filter()
    f.valves.inject_file_summary_hint = False
    body = {"files": [{"name": "a.csv"}],
            "messages": [{"role": "user", "content": "summarise"}]}
    s = _system(_run(body, filt=f))
    assert OWN_MARKER in s and "Attached data file" not in s


def test_bad_body_passes_through():
    body = {"messages": "not a list"}
    assert _run(body) == {"messages": "not a list"}


def test_outlet_is_a_no_op():
    body = {"messages": [{"role": "assistant", "content": "x"}]}
    assert asyncio.run(Filter().outlet(body)) is body
