"""F10 (17-backlog C5): inlet() never stacks a second injection on a body it
already handled - NO_KB_INFO and OUT_OF_SCOPE_OK say opposite things.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import Filter

TERMINAL_ON = {"id": "m", "meta": {"capabilities": {"terminal": True}}}


def _system_after_inlet(messages, hits=(), sources=""):
    async def _search(*_a, **_k):
        return list(hits), None, sources
    f = Filter()
    f._search = _search
    body = {"chat_id": "s1025", "messages": [dict(m) for m in messages]}
    asyncio.run(f.inlet(body, __model__=TERMINAL_ON))
    return "\n".join(str(m.get("content")) for m in body["messages"]
                     if m.get("role") == "system"), body


# ── F10: inlet is idempotent on a body it already handled ───────────────────

@pytest.mark.parametrize("q,hits", [
    ("What does the knowledge base say about the Gamma Reversal Block?", []),
    ("what does the manual say about backups?", []),
    ("calculate the totals for this file", []),
    ("In Episode 12, what does the speaker draw or mark on the chart around 1:08:00?", []),
])
def test_second_inlet_on_the_same_body_changes_nothing(q, hits):
    async def _search(*_a, **_k):
        return list(hits), None, ""
    f = Filter()
    f._search = _search
    body = {"chat_id": "idem", "messages": [{"role": "user", "content": q}]}
    asyncio.run(f.inlet(body, __model__=TERMINAL_ON))
    once = [dict(m) for m in body["messages"]]
    asyncio.run(f.inlet(body, __model__=TERMINAL_ON))
    assert body["messages"] == once


def test_a_body_resent_with_last_turns_injection_is_rebuilt_not_stacked():
    # If a client ever hands back the injected system message (an edited or
    # regenerated turn), the filter restores the original and routes afresh:
    # never two contradictory blocks (NO_KB_INFO + OUT_OF_SCOPE_OK) in one prompt.
    async def _search(*_a, **_k):
        return [], None, ""
    f = Filter()
    f._search = _search
    sys_msg = {"role": "system", "content": "You are the assistant."}
    body = {"chat_id": "idem2", "messages": [dict(sys_msg), {"role": "user",
            "content": "What does the knowledge base say about the Gamma Reversal Block?"}]}
    asyncio.run(f.inlet(body, __model__=TERMINAL_ON))
    injected = body["messages"][0]["content"]
    # Same system message comes back, the user's message was edited:
    body2 = {"chat_id": "idem2", "messages": [{"role": "system", "content": injected},
             {"role": "user", "content": "what does the manual say about backups?"}]}
    asyncio.run(f.inlet(body2, __model__=TERMINAL_ON))
    system = body2["messages"][0]["content"]
    assert system.count(rag_filter.IDENTITY_GUARD) == 1
    assert rag_filter.OUT_OF_SCOPE_OK not in system
    assert rag_filter.NO_KB_INFO in system
    assert "You are the assistant." in system


def test_data_file_note_is_not_appended_twice():
    msg = {"role": "user", "content": "x"}
    note = rag_filter.DATA_FILE_HINT.format(files="a.csv (sandbox path: /p/a.csv)")
    rag_filter._append_text(msg, note)
    rag_filter._append_text(msg, note)
    assert msg["content"].count("[System note:") == 1
