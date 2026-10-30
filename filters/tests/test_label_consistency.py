"""P119-P121: manual retest 2026-10-16 failures that were prompt contradictions.

6.4  META_CAPABILITY_RULES told the model saved notes ARE indexed and searchable;
     nothing ingests chat notes, so the claim is false and the model repeated it.
6.6  A model reported output from a kill_process tool that no tool list contains.

(The 6.2 label-consistency case — two filters disagreeing on the metrics labels —
left with the domain-specific tools layer it concerned; those tests are removed below.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_filter import META_CAPABILITY_RULES, CODE_TOOLS_HINT, CODE_TOOLS_LIGHT


def test_meta_rules_do_not_claim_typed_notes_are_indexed():
    # A note typed directly in chat never reaches the KB (confirmed: it's only
    # ever written to a sandbox path) - that claim must stay denied.
    assert "typed" in META_CAPABILITY_RULES.lower()
    assert "not searchable from other conversations" in META_CAPABILITY_RULES


def test_meta_rules_state_uploaded_files_are_indexed_and_searchable():
    # P123 root cause (Qdrant access log): the "vanished" uploaded note was
    # deleted twice via the /kb/ panel, not lost - ingestion works. So an
    # uploaded file IS indexed and searchable later, unless removed there.
    rules = META_CAPABILITY_RULES.lower()
    assert "uploaded file" in rules and "searchable from later conversations" in rules
    assert "removed" in rules and "admin panel" in rules
    assert "unrecoverable" not in rules and "no guarantee" not in rules


def test_hint_forbids_citing_tools_not_in_tool_list():
    for block in (CODE_TOOLS_HINT, CODE_TOOLS_LIGHT):
        assert "not in your tool list" in block


def test_tool_examples_name_only_real_tools():
    # 10-12 retest: gemini-3.8 called "run_python" and "system.list_tools" -
    # neither exists (the sandbox tool is run_command) - and the UI showed each
    # "Tool not found" as a red X. Both names came from this sentence.
    for block in (CODE_TOOLS_HINT, CODE_TOOLS_LIGHT):
        for fake in ("run_python", "execute_code"):
            assert fake not in block, fake
        assert "run_command" in block
        assert "never call a tool to list tools" in block.lower()


def test_file_ready_hint_requires_printed_rows_for_any_quoted_bar():
    # P126: the hint asked for .head()/.describe() and "the first and last index",
    # so the last timestamp got printed but never the last row - and gemini-3.8
    # filled in values beside it (1 of 3 live runs on the client's file).
    from rag_filter import FILE_READY_HINT
    assert ".tail()" in FILE_READY_HINT
    assert "copied from that row as printed in your tool output" in FILE_READY_HINT
    assert "never fill it in" in FILE_READY_HINT


def test_file_ready_hint_does_not_guess_from_the_filename():
    # 2026-10-19 retest: on a bare "what's in this file?" the model described the
    # file from its name - nothing in its tool output said so. The rule lives in
    # the hint itself, because a file-description turn never gets the tools
    # contract.
    from rag_filter import FILE_READY_HINT
    assert "do NOT guess the interval from the filename" in FILE_READY_HINT


# A test pinning a domain-specific code-language rules block was removed with
# that block.


def test_hint_delivers_downloads_through_display_file_inline():
    # download_file lives on a Tool attached to no model, so the hint must point
    # at the mechanism that exists: display_file(inline=true) renders Open
    # WebUI's in-chat file card with a Download button (verified in a browser
    # on admin and non-admin accounts, 2026-10-16).
    assert "download_file(sandbox_path" not in CODE_TOOLS_HINT
    assert "display_file" in CODE_TOOLS_HINT
    assert "inline=true" in CODE_TOOLS_HINT


def test_no_model_facing_instruction_contains_the_word_retrieval():
    # 2026-10-19 matrix: Sonnet opened 4/4 ungrounded answers with
    # "retrieval had nothing on file" / "not a retrieved source" - paraphrasing
    # our own instructions, which said "Retrieval has no definition for the
    # concept..." and listed "retrieval had no definition" as a banned example.
    # A word in the prompt is a word the model repeats.
    import re
    import rag_filter
    for name in dir(rag_filter):
        value = getattr(rag_filter, name)
        if name.isupper() and isinstance(value, str) and len(value) > 80:
            assert not re.search(r"retriev", value, re.I), name


def test_leak_guard_redacts_any_retrieval_disclosure_sentence():
    from rag_filter import Filter
    f = Filter()
    msg = {"role": "assistant", "content": (
        "No definition was found in the retrieved documentation, so this one is mine. "
        "A standard setup uses three rows: the first establishes the range.")}
    f._guard_leak_detection(msg, msg["content"])
    assert "retriev" not in msg["content"].lower()
    assert "standard setup uses three rows" in msg["content"]


def test_file_ready_hint_reads_the_raw_header_for_column_questions():
    # P149: load_file moves the timestamp column into the index, so df.columns
    # lists one column fewer than the file has. Tester's test 13: a 6-column
    # file reported as 5, a 7-column file as 6, Timestamp missing both times.
    from rag_filter import FILE_READY_HINT
    assert "nrows=0" in FILE_READY_HINT
    assert "timestamp column into the index" in FILE_READY_HINT


def test_out_of_scope_never_invents_a_source_when_the_user_asks_for_one():
    # P150: "According to my uploaded knowledge base ... cite the source
    # document" with 0 hits; gemini-3.7 answered from training data and listed
    # invented "Source References" that are not in the knowledge base.
    from rag_filter import OUT_OF_SCOPE_OK
    assert "never name, invent or cite a source document" in OUT_OF_SCOPE_OK
    assert "retriev" not in OUT_OF_SCOPE_OK.lower()


def test_sources_block_lists_inline_attached_documents():
    # P151: an attached document reaches the model inline, not via /search, so
    # the Sources block (built from index hits only) could never list it -
    # tester's multi-doc case cited session_policy inline with no Sources entry.
    from rag_filter import _with_attached_sources
    out = _with_attached_sources(
        "Sources:\n[1] a.txt\n[2] multi_doc_risk_policy.txt",
        [{"name": "multi_doc_session_policy.txt"}, {"name": "multi_doc_risk_policy.txt"}])
    assert "[3] multi_doc_session_policy.txt (attached)" in out
    assert out.count("multi_doc_risk_policy.txt") == 1
    assert _with_attached_sources("", [{"name": "x.txt"}]) == "Sources:\n[1] x.txt (attached)"
    assert _with_attached_sources("Sources:\n[1] a.txt", []) == "Sources:\n[1] a.txt"


# The label-consistency tests for the old domain-specific metrics contract were
# removed with that tools layer: the blocks they pinned no longer exist. What
# survives of the generic contract is pinned in test_routing_accuracy.py and
# rag_filter.py's own self-check (tool list, numbers-from-code, downloads).
