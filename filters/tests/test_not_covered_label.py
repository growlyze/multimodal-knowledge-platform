"""F4: a gap is labelled "not covered in the knowledge base", as agreed with the client.

CONVERSATIONS 3:29 (MINOR): "The label should read 'not covered in the knowledge
base', as agreed. It currently says 'no document'." OUT_OF_SCOPE_OK told the model
to say "the knowledge base has no document covering it".
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


# ── F4: the agreed label ─────────────────────────────────────────────────────

def test_out_of_scope_kb_request_uses_the_agreed_label():
    assert "no document" not in rag_filter.OUT_OF_SCOPE_OK
    assert "not covered in the knowledge base" in rag_filter.OUT_OF_SCOPE_OK


def test_grounding_rules_name_a_gap_with_the_agreed_label():
    assert "not covered in the knowledge base" in rag_filter.GROUNDING_RULES


def test_no_injected_block_says_no_document():
    names = [n for n in dir(rag_filter) if n.isupper()]
    for n in names:
        v = getattr(rag_filter, n)
        if isinstance(v, str):
            assert "no document covering" not in v, n


def test_the_agreed_label_is_not_logged_as_a_leak():
    # The leak guard is log-only; flagging the label the client asked for would bury
    # real leaks in noise.
    assert "not covered in the knowledge base" not in Filter._LEAK_PHRASES
