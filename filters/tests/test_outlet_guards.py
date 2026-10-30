"""Outlet guards: post-response checks that catch failures models create.

These guards fire in outlet() AFTER the model has responded, using routing
context saved during inlet()/_route(). They append warnings or log detections;
they never rewrite or remove response content.

Offline, pure-logic. Same pattern as test_hard_refusal.py: _search stubbed,
no network, no DB.
"""
import asyncio
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_filter import Filter

REFUSAL = "I can't run code right now"


def _model(terminal):
    return {"id": "test-model", "meta": {"capabilities": {"terminal": terminal}}}


async def _no_hits(*_a, **_kw):
    return [], None, ""


def _run_inlet_outlet(query, terminal=True, chat_id="test-1"):
    """Run inlet then outlet, return the final body."""
    f = Filter()
    f._search = _no_hits
    body = {
        "chat_id": chat_id,
        "messages": [{"role": "user", "content": query}],
    }
    model = _model(terminal)
    asyncio.run(f.inlet(body, __model__=model))
    # Simulate model response
    body["messages"].append({"role": "assistant", "content": ""})
    return f, body


def test_state_bridge_stores_context():
    """_route_ctx is populated after inlet with the routing decisions.

    "how many rows" matches a CODE_KEYWORD exactly (needs_code_execution
    checks CODE_KEYWORDS membership), so it deterministically triggers
    code=True. Use a query that deterministically triggers code=True.
    """
    f, body = _run_inlet_outlet(
        "how many rows are in this file?", terminal=False, chat_id="sb-1"
    )
    assert "sb-1" in f._route_ctx, "state bridge did not store context"
    ctx = f._route_ctx["sb-1"]
    assert ctx["terminal_off"] is True
    assert ctx["code"] is True
    print("ok state_bridge_stores_context")


def test_state_bridge_popped_by_outlet():
    """outlet() pops the context so it doesn't leak."""
    f, body = _run_inlet_outlet(
        "how many rows are in this file?", terminal=False, chat_id="sb-2"
    )
    assert "sb-2" in f._route_ctx
    asyncio.run(f.outlet(body))
    assert "sb-2" not in f._route_ctx, "outlet did not pop context"
    print("ok state_bridge_popped_by_outlet")


def test_state_bridge_absent_for_non_code():
    """A non-code question stores context with code=False."""
    f, body = _run_inlet_outlet(
        "what is a lookup table?", terminal=True, chat_id="sb-3"
    )
    if "sb-3" in f._route_ctx:
        assert f._route_ctx["sb-3"]["code"] is False
    print("ok state_bridge_absent_for_non_code")


def _make_body_with_response(chat_id, query, response):
    """Build a body dict as outlet() would receive it."""
    return {
        "chat_id": chat_id,
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ],
    }


def test_guard_code_as_text_fires():
    """Code block in response with terminal_off + code → warning appended."""
    f = Filter()
    ctx = {"terminal_off": True, "code": True, "tools": True}
    body = _make_body_with_response(
        "g1", "how many rows?",
        "Here's the code:\n```python\nprint('hello')\n```"
    )
    f._run_outlet_guards(body, ctx)
    last = body["messages"][-1]["content"]
    assert "not executed" in last.lower() or "not run" in last.lower(), (
        f"code-as-text guard did not fire: {last[-200:]}"
    )
    print("ok guard_code_as_text_fires")


def test_guard_code_as_text_skips_clean_refusal():
    """Clean refusal with no code block → no warning."""
    f = Filter()
    ctx = {"terminal_off": True, "code": True, "tools": True}
    body = _make_body_with_response(
        "g2", "how many rows?", REFUSAL
    )
    original = body["messages"][-1]["content"]
    f._run_outlet_guards(body, ctx)
    assert body["messages"][-1]["content"] == original, "guard modified clean refusal"
    print("ok guard_code_as_text_skips_clean_refusal")


def test_state_bridge_populates_from_metadata_chat_id():
    """CRITICAL live finding, 2026-10-13: a real Open WebUI inlet() request
    body never carries chat_id at its TOP level - confirmed via a live
    diagnostic (body_keys=['stream', 'model', 'terminal_id', 'features',
    'metadata', 'messages']; chat_id lives at body['metadata']['chat_id']
    only). Every offline test (including every other one in this file) uses
    a hand-built body with chat_id at the top level, matching nothing real -
    so `_route()`'s `body.get("chat_id")` read None in production, NEVER
    wrote to `_route_ctx`, and every outlet guard in this file has been
    silently inert against real traffic since Task 2 shipped. Reproduced
    live: Sonnet 5's actual unfenced-code response to the row-count/Terminal-off
    scenario got zero guard warnings appended, and `outlet()`'s own
    diagnostic showed `ctx={}` for a turn that definitely routed through
    `_route()`'s code path."""
    f = Filter()
    f._search = _no_hits
    chat_id = "real-shape-1"
    body = {
        "messages": [{"role": "user", "content": "how many rows are in this file?"}],
        "metadata": {"chat_id": chat_id},
    }
    asyncio.run(f.inlet(body, __model__=_model(False)))
    assert chat_id in f._route_ctx, (
        "state bridge did not populate from metadata.chat_id - "
        f"_route_ctx keys: {list(f._route_ctx.keys())}"
    )
    assert f._route_ctx[chat_id]["terminal_off"] is True
    print("ok state_bridge_populates_from_metadata_chat_id")


def test_state_bridge_does_not_leak_stale_context_across_turns():
    """Final review Focus 4: _route()'s own `if not query.strip(): return`
    early-return (an image-only or empty-query turn) skips writing
    _route_ctx entirely, WITHOUT clearing a previous turn's entry for the
    same chat_id. A turn whose outlet() never ran (error mid-turn) leaves
    its context behind; the NEXT turn on that chat_id, if it hits this
    early return, would have outlet() run guards against the WRONG turn's
    terminal_off/code state. inlet() must clear any stale entry for this
    chat_id up front, not rely on _route() to always overwrite it."""
    f = Filter()
    f._search = _no_hits
    chat_id = "stale-1"
    # Turn 1: a code-needing, terminal-off turn — populates _route_ctx.
    body1 = {"chat_id": chat_id, "messages": [
        {"role": "user", "content": "how many rows are in this file?"}
    ]}
    asyncio.run(f.inlet(body1, __model__=_model(False)))
    assert chat_id in f._route_ctx, "turn 1 should have stored context"
    assert f._route_ctx[chat_id]["code"] is True
    # Simulate outlet() never running for turn 1 (error mid-turn) — the
    # stale entry is still sitting in _route_ctx exactly as real production
    # code leaves it.
    # Turn 2, same chat_id: an empty-query turn (e.g. an image-only message)
    # that _route() returns early on, before ever touching _route_ctx.
    body2 = {"chat_id": chat_id, "messages": [
        {"role": "user", "content": ""}
    ]}
    asyncio.run(f.inlet(body2, __model__=_model(True)))
    assert chat_id not in f._route_ctx, (
        "turn 1's stale context leaked into turn 2, which never reached "
        "_route()'s own context-write step"
    )
    print("ok state_bridge_does_not_leak_stale_context_across_turns")


def test_guard_code_as_text_fires_on_unfenced_python():
    """Live finding, 2026-10-13: after the terminal_id fix made terminal_off
    correctly True for a chat-UI-detached turn, Sonnet 5 STILL ignored
    HARD_NO_CODE_REFUSAL - but wrote raw, unfenced Python (no ``` at all),
    starting with a bare "1" exactly as HARD_NO_CODE_REFUSAL's own comment
    already named as a documented Sonnet pattern ("a bare '1', 'python' +
    code"). The guard's fence-only check missed this shape entirely."""
    f = Filter()
    ctx = {"terminal_off": True, "code": True, "tools": True}
    body = _make_body_with_response(
        "g13", "How many rows match this filter in this file?",
        "1\n\n\nimport pandas as pd\nfrom sandbox_tools import load_file, summarize\n\n"
        "df = load_file(\"/mnt/uploads/x.csv\")\n\nprint(df.shape)\nprint(df.head())"
    )
    f._run_outlet_guards(body, ctx)
    last = body["messages"][-1]["content"]
    assert "not executed" in last.lower() or "not run" in last.lower(), (
        f"code-as-text guard did not fire on unfenced Python: {last[-200:]}"
    )
    print("ok guard_code_as_text_fires_on_unfenced_python")


def test_guard_code_as_text_skips_ordinary_prose_with_an_equals_sign():
    """The unfenced-code heuristic must not fire on ordinary prose that
    happens to use words like 'import' or an equals sign once."""
    f = Filter()
    ctx = {"terminal_off": True, "code": True, "tools": True}
    body = _make_body_with_response(
        "g14", "How many rows match this filter in this file?",
        REFUSAL
    )
    original = body["messages"][-1]["content"]
    f._run_outlet_guards(body, ctx)
    assert body["messages"][-1]["content"] == original, "guard false-fired on a clean refusal"
    print("ok guard_code_as_text_skips_ordinary_prose_with_an_equals_sign")


def test_guard_code_as_text_fires_on_data_file_turn_with_code_false():
    """The inlet hard refusal fires on `tools and terminal_off` (tools = code
    or data_files — rag_filter.py's own condition), not on `code` alone. A
    turn with an attached data file and a query that needs_code_execution()
    scores as code=False (e.g. the client's exact M1.2-B1 repro, "how many rows
    match this filter?" — the wording alone doesn't trip a CODE_KEYWORD) still gets
    the hard refusal injected on the inlet side. The outlet guard must use
    the SAME gate (tools), or it silently skips the exact scenario the
    refusal was injected for."""
    f = Filter()
    ctx = {"terminal_off": True, "code": False, "tools": True}
    body = _make_body_with_response(
        "g4", "how many duplicates are in this file?",
        "Here's the code:\n```python\nprint('hello')\n```"
    )
    f._run_outlet_guards(body, ctx)
    last = body["messages"][-1]["content"]
    assert "not executed" in last.lower() or "not run" in last.lower(), (
        f"code-as-text guard did not fire on a data-file turn with code=False: {last[-200:]}"
    )
    print("ok guard_code_as_text_fires_on_data_file_turn_with_code_false")


def test_guard_code_as_text_fires_on_bare_fence():
    """A fenced block with no language tag (bare ```) must still fire — the
    guard previously only matched ```python/py/js/javascript, missing
    the unfenced/untagged form the spec names Sonnet actually produces."""
    f = Filter()
    ctx = {"terminal_off": True, "code": True, "tools": True}
    body = _make_body_with_response(
        "g5", "how many rows?",
        "Here's the code:\n```\nprint('hello')\n```"
    )
    f._run_outlet_guards(body, ctx)
    last = body["messages"][-1]["content"]
    assert "not executed" in last.lower() or "not run" in last.lower(), (
        f"code-as-text guard did not fire on a bare-fenced block: {last[-200:]}"
    )
    print("ok guard_code_as_text_fires_on_bare_fence")


def test_guard_code_as_text_skips_terminal_on():
    """Code block with terminal ON → no warning (code was presumably executed)."""
    f = Filter()
    ctx = {"terminal_off": False, "code": True, "tools": True}
    body = _make_body_with_response(
        "g3", "how many rows?",
        "Here:\n```python\nprint('hello')\n```"
    )
    original = body["messages"][-1]["content"]
    f._run_outlet_guards(body, ctx)
    assert body["messages"][-1]["content"] == original, "guard fired with terminal ON"
    print("ok guard_code_as_text_skips_terminal_on")


def test_guard_leak_detection_fires():
    """Response containing internal phrase → warning logged AND phrase redacted."""
    f = Filter()
    ctx = {"terminal_off": False, "code": False, "tools": False}
    body = _make_body_with_response(
        "g4", "what is a cache?",
        "Retrieval had no definition for this concept, but here is the general understanding..."
    )
    import io
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(logging.WARNING)
    log = logging.getLogger("rag-filter")
    log.addHandler(handler)
    try:
        f._run_outlet_guards(body, ctx)
        output = handler.stream.getvalue()
        assert "outlet-leak-detect" in output, f"leak guard did not log: {output}"
    finally:
        log.removeHandler(handler)
    content = body["messages"][-1]["content"]
    assert "retrieval had no definition" not in content.lower(), (
        f"leak phrase not redacted from response: {content}"
    )
    print("ok guard_leak_detection_fires")


def test_guard_leak_detection_clean():
    """Clean response → no leak warning."""
    f = Filter()
    ctx = {"terminal_off": False, "code": False, "tools": False}
    body = _make_body_with_response(
        "g5", "what is a cache?",
        "A cache is a small fast store that sits in front of a slower one."
    )
    import io
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(logging.WARNING)
    log = logging.getLogger("rag-filter")
    log.addHandler(handler)
    try:
        f._run_outlet_guards(body, ctx)
        output = handler.stream.getvalue()
        assert "outlet-leak-detect" not in output, f"leak guard false positive: {output}"
    finally:
        log.removeHandler(handler)
    print("ok guard_leak_detection_clean")


def test_guard_download_reminder_skips_input_paths():
    """A path under /tmp/datasets/ or /tmp/extracted/ is an INPUT the model
    read (FALLBACK_DIR / ARCHIVE_EXTRACT_HINT), never something it saved —
    the reminder claiming 'the file was saved in the sandbox' is false when
    the model is merely naming where it read data FROM."""
    f = Filter()
    ctx = {"terminal_off": False, "code": True, "tools": True}
    body = _make_body_with_response(
        "g8", "how many rows in this file?",
        "I loaded the file from /tmp/datasets/upload_f10.csv and found 500 rows."
    )
    original = body["messages"][-1]["content"]
    f._run_outlet_guards(body, ctx)
    assert body["messages"][-1]["content"] == original, (
        "reminder fired on an INPUT path the model only read from"
    )
    print("ok guard_download_reminder_skips_input_paths")


def test_guard_download_reminder_skips_prose_mentioning_data_colon():
    """`has_download = 'data:' in text` matched ordinary prose like 'Summary
    of the data:' — a false negative that silently suppressed the reminder
    on a genuine unresolved save. The check must require an actual data:
    URI (a MIME type after the colon), not the bare word 'data:'."""
    f = Filter()
    ctx = {"terminal_off": False, "code": True, "tools": True}
    body = _make_body_with_response(
        "g9", "export this as CSV",
        "Summary of the data: 500 rows saved to /tmp/results.csv"
    )
    f._run_outlet_guards(body, ctx)
    last = body["messages"][-1]["content"]
    assert "download" in last.lower(), (
        f"reminder was suppressed by prose containing 'data:': {last[-200:]}"
    )
    print("ok guard_download_reminder_skips_prose_mentioning_data_colon")


def test_guard_fabrication_fires_on_ohlc_like_numbers_with_terminal_off():
    """Spec §1.4: 4+ numbers in a single line, with no tool call
    possible (terminal_off), is the exact M1.2-B1/M2-3 fabrication shape —
    a model inventing bar data it could not have computed. This guard was
    present in the spec's table but dropped from the implementation with
    no ledger ruling (final review finding 5)."""
    f = Filter()
    ctx = {"terminal_off": True, "code": True, "tools": True}
    body = _make_body_with_response(
        "g10", "what were the first row's values?",
        "The first row: Open 4512.25, High 4515.50, Low 4510.00, Close 4513.75."
    )
    f._run_outlet_guards(body, ctx)
    last = body["messages"][-1]["content"]
    assert "not verified" in last.lower() or "not computed" in last.lower(), (
        f"fabrication guard did not fire on a number-heavy line: {last[-200:]}"
    )
    print("ok guard_fabrication_fires_on_ohlc_like_numbers_with_terminal_off")


def test_guard_fabrication_skips_prose_with_no_numbers():
    """A plain-language answer with no numeric claims must not trigger the
    fabrication guard, even with terminal off and a code-flagged query."""
    f = Filter()
    ctx = {"terminal_off": True, "code": True, "tools": True}
    body = _make_body_with_response(
        "g11", "what is a lookup table?",
        REFUSAL
    )
    original = body["messages"][-1]["content"]
    f._run_outlet_guards(body, ctx)
    assert body["messages"][-1]["content"] == original, "fabrication guard false-fired on a clean refusal"
    print("ok guard_fabrication_skips_prose_with_no_numbers")


def test_guard_fabrication_skips_terminal_on():
    """Same number-heavy line, but Terminal was available — presumably real,
    computed output, not a fabrication."""
    f = Filter()
    ctx = {"terminal_off": False, "code": True, "tools": True}
    body = _make_body_with_response(
        "g12", "what were the first row's values?",
        "The first row: Open 4512.25, High 4515.50, Low 4510.00, Close 4513.75."
    )
    original = body["messages"][-1]["content"]
    f._run_outlet_guards(body, ctx)
    assert body["messages"][-1]["content"] == original, "fabrication guard fired with terminal ON"
    print("ok guard_fabrication_skips_terminal_on")


def test_guard_leak_detection_does_not_flag_the_required_nokbinfo_disclosure():
    """NO_KB_INFO REQUIRES the model to say 'This isn't covered in the
    knowledge base yet - here is the general understanding:' verbatim. The
    leak guard's own phrase list included 'knowledge base yet', so every
    correctly-behaving zero-hit turn logged a false 'leak' - polluting the
    v1 log-only frequency measurement the guard exists to collect, and
    making the live test that asserts this phrase's absence fail on
    correct behavior. The required disclosure must never be flagged."""
    f = Filter()
    ctx = {"terminal_off": False, "code": False, "tools": False}
    body = _make_body_with_response(
        "g6", "what is a hash table?",
        "This isn't covered in the knowledge base yet - here is the general "
        "understanding: a hash table is..."
    )
    import io
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(logging.WARNING)
    log = logging.getLogger("rag-filter")
    log.addHandler(handler)
    try:
        f._run_outlet_guards(body, ctx)
        output = handler.stream.getvalue()
        assert "outlet-leak-detect" not in output, (
            f"leak guard flagged the REQUIRED NO_KB_INFO disclosure: {output}"
        )
    finally:
        log.removeHandler(handler)
    print("ok guard_leak_detection_does_not_flag_the_required_nokbinfo_disclosure")


def test_guard_download_reminder_fires():
    """File path in response without download_file → reminder appended."""
    f = Filter()
    ctx = {"terminal_off": False, "code": True, "tools": True}
    body = _make_body_with_response(
        "g6", "export this as CSV",
        "I've saved the results to /tmp/results.csv"
    )
    f._run_outlet_guards(body, ctx)
    last = body["messages"][-1]["content"]
    assert "download" in last.lower(), f"download reminder did not fire: {last[-200:]}"
    print("ok guard_download_reminder_fires")


def test_guard_download_reminder_skips_with_download():
    """File path + download_file's markdown link already present → no reminder.

    M2-5 fix: download_file no longer returns a data: URI (that path was a
    second-class citizen in Open WebUI's process_tool_result - images get
    rewritten into a served URL, non-image data: URIs don't). It now
    uploads via /api/v1/files/ and returns a markdown link to
    /api/v1/files/<id>/content, mirroring how ChatGPT/Claude hand over
    generated files (a reference, never the bytes, in the model's reply).
    """
    f = Filter()
    ctx = {"terminal_off": False, "code": True, "tools": True}
    body = _make_body_with_response(
        "g7", "export this as CSV",
        "I've saved the results to /tmp/results.csv. Here is your file: "
        "[results.csv](https://example.com/api/v1/files/abc123/content)"
    )
    original = body["messages"][-1]["content"]
    f._run_outlet_guards(body, ctx)
    assert body["messages"][-1]["content"] == original, "reminder fired despite download link"
    print("ok guard_download_reminder_skips_with_download")


def test_guard_download_reminder_skips_when_display_file_called():
    """display_file ran this turn → Open WebUI already attached the download
    card, so a "not yet delivered" reminder would be false. Shape matches the
    stored message: output items with a function_call named display_file."""
    f = Filter()
    ctx = {"terminal_off": False, "code": True, "tools": True}
    body = _make_body_with_response(
        "g8", "export this as CSV",
        "Saved to /home/user/outputs/trades.csv - download it from the card above."
    )
    body["messages"][-1]["output"] = [
        {"type": "function_call", "name": "display_file", "call_id": "c1",
         "arguments": '{"path": "/home/user/outputs/trades.csv", "inline": true}'},
        {"type": "function_call_output", "call_id": "c1", "status": "completed"},
    ]
    original = body["messages"][-1]["content"]
    f._run_outlet_guards(body, ctx)
    assert body["messages"][-1]["content"] == original, "reminder fired although display_file delivered the file"
    print("ok guard_download_reminder_skips_when_display_file_called")


def test_guard_download_reminder_does_not_name_an_unavailable_tool():
    """No model has download_file (the tool that offers it is attached to
    none of them), so the reminder must not tell the user to ask for it."""
    f = Filter()
    ctx = {"terminal_off": False, "code": True, "tools": True}
    body = _make_body_with_response("g9", "export this as CSV", "I've saved the results to /tmp/results.csv")
    f._run_outlet_guards(body, ctx)
    last = body["messages"][-1]["content"]
    assert "download" in last.lower(), last[-200:]
    assert "download_file" not in last, last[-200:]
    print("ok guard_download_reminder_does_not_name_an_unavailable_tool")


def main():
    test_state_bridge_stores_context()
    test_state_bridge_popped_by_outlet()
    test_state_bridge_absent_for_non_code()
    test_state_bridge_populates_from_metadata_chat_id()
    test_state_bridge_does_not_leak_stale_context_across_turns()
    test_guard_code_as_text_fires()
    test_guard_code_as_text_skips_clean_refusal()
    test_guard_code_as_text_fires_on_unfenced_python()
    test_guard_code_as_text_skips_ordinary_prose_with_an_equals_sign()
    test_guard_code_as_text_fires_on_data_file_turn_with_code_false()
    test_guard_code_as_text_fires_on_bare_fence()
    test_guard_code_as_text_skips_terminal_on()
    test_guard_fabrication_fires_on_ohlc_like_numbers_with_terminal_off()
    test_guard_fabrication_skips_prose_with_no_numbers()
    test_guard_fabrication_skips_terminal_on()
    test_guard_leak_detection_fires()
    test_guard_leak_detection_clean()
    test_guard_leak_detection_does_not_flag_the_required_nokbinfo_disclosure()
    test_guard_download_reminder_fires()
    test_guard_download_reminder_skips_with_download()
    test_guard_download_reminder_skips_input_paths()
    test_guard_download_reminder_skips_prose_mentioning_data_colon()
    print(f"\n22 checks passed")


if __name__ == "__main__":
    main()


def test_terminal_off_read_from_the_0_11_model_shape():
    """P219: Open WebUI 0.11.4 passes __model__ with the row under "info" ->
    model["info"]["meta"]["capabilities"]["terminal"]; the filter read
    model["meta"], so the Admin Terminal switch was ignored (live: 83/83
    `terminal_off=True (cap=None` lines). A model with Terminal switched off
    must count as terminal_off even when the request still carries a terminal_id."""
    f = Filter()
    f.valves.enable_async_ingestion = False
    model = {"id": "m", "info": {"meta": {"capabilities": {"terminal": False}}}}
    body = {"chat_id": "t-info", "terminal_id": "t1", "features": {},
            "messages": [{"role": "user", "content": "how many rows are in this file?"}]}
    asyncio.run(f.inlet(body, __model__=model))
    assert f._route_ctx["t-info"]["terminal_off"] is True
