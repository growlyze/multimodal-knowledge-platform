"""M4.4 persistent user context: Open WebUI 0.11.4 native Memory (rag_filter 1.44).

Design: docs/superpowers/specs/2026-10-26-persistent-user-context-design.md, option A.
Open WebUI owns storage, per-user isolation and the memory tools; the filter's job is
the prompt contract around them:

  * memories are the user's preferences / defaults / personal notes - never a source
    for a knowledge-base definition or a data figure, never listed under Sources;
  * save only when asked, confirm; never save file contents, data or results, or
    knowledge-base / web text; forget on request;
  * the rule appears ONLY when Open WebUI will really offer memory on this turn
    (mirrors utils/tools.py: features.memory AND builtinTools.memory AND
    capabilities.memory, the last two defaulting to True) - on every other turn the
    prompt carries no memory text (as in 1.43) and keeps P121's "upload it as a file" advice,
    which is true exactly when memory is off.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import Filter

MEM_ON = {"id": "claude-sonnet-5", "info": {"meta": {
    "capabilities": {"terminal": True, "memory": True},
    "builtinTools": {"memory": True, "knowledge": False, "files": False}}}}
# Today's live state on all four picker models (snapshot 2026-10-26).
MEM_OFF = {"id": "claude-sonnet-5", "info": {"meta": {
    "capabilities": {"terminal": True, "memory": False},
    "builtinTools": {"memory": False, "knowledge": False, "files": False}}}}


def _inlet(query, *, model, memory=True):
    async def _search(*_a, **_k):
        return [], None, ""
    f = Filter()
    f._search = _search
    body = {"metadata": {"chat_id": "mem"}, "messages": [{"role": "user", "content": query}],
            "features": {"memory": memory}, "terminal_id": "t1"}
    asyncio.run(f.inlet(body, __model__=model))
    return "\n".join(str(m.get("content")) for m in body["messages"] if m.get("role") == "system")


# ── availability mirrors Open WebUI ─────────────────────────────────────────

@pytest.mark.parametrize("body,model,expected", [
    ({"features": {"memory": True}}, MEM_ON, True),
    ({"features": {"memory": False}}, MEM_ON, False),     # user switched memory off
    ({}, MEM_ON, False),                                  # client sent no toggle
    ({"features": {"memory": True}}, MEM_OFF, False),     # today's live models
    ({"features": {"memory": True}}, {"info": {"meta": {"capabilities": {"memory": True},
      "builtinTools": {"memory": False}}}}, False),       # tools category off
    ({"features": {"memory": True}}, {"info": {"meta": {"capabilities": {"memory": False},
      "builtinTools": {"memory": True}}}}, False),        # capability off
    ({"features": {"memory": True}}, {"info": {"meta": {}}}, True),  # absent = on (0.11.4)
    ({"features": {"memory": True}}, None, True),
])
def test_memory_available_mirrors_open_webui(body, model, expected):
    assert rag_filter._memory_available(body, model) is expected


# ── the rule ────────────────────────────────────────────────────────────────

def test_rule_contract():
    r = rag_filter.MEMORY_RULES
    assert "only when the user asks" in r
    assert "never a source" in r and "Sources" in r
    assert "file contents" in r and "web pages" in r
    assert "forget" in r
    # never echoed as "context" wording (CONTEXT_WORD_RULE), and short: every turn
    assert "context" not in r.lower().replace("<memory_context>", "")
    assert rag_filter.count_tokens(r) < 200


@pytest.mark.parametrize("q", [
    "What is a lookup table?",
    "please remember that I work in metric units",
    "write a script that detects outliers",
    "What does CSV stand for?",
])
def test_rule_rides_every_turn_when_memory_is_on(q):
    assert rag_filter.MEMORY_RULES in _inlet(q, model=MEM_ON)


@pytest.mark.parametrize("q", ["What is a lookup table?", "please store this note"])
def test_no_memory_rule_on_todays_models(q):
    s = _inlet(q, model=MEM_OFF)
    assert rag_filter.MEMORY_RULES not in s
    assert "memory tools" not in s


def test_no_memory_rule_when_the_user_turned_memory_off():
    assert rag_filter.MEMORY_RULES not in _inlet("What is a hash table?", model=MEM_ON, memory=False)


# ── the meta-capability answer follows what is really on ───────────────────

META_Q = "do you have any other data stored in your memory?"


def test_meta_answer_without_memory_keeps_p121_upload_advice():
    s = _inlet(META_Q, model=MEM_OFF)
    assert rag_filter.META_CAPABILITY_RULES in s
    assert "upload it as a file" in s


def test_meta_answer_with_memory_drops_the_upload_workaround():
    s = _inlet(META_Q, model=MEM_ON)
    assert rag_filter.META_CAPABILITY_RULES_MEMORY in s
    assert rag_filter.META_CAPABILITY_RULES not in s
    # P121's workaround wrote personal notes into the SHARED collection (P123)
    assert "upload it as a file" not in s
    assert "Never say you have no database, no memory, or no stored data" in s
    # personal, never shared, never the knowledge base
    assert "other users" in s
