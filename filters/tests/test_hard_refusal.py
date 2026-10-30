"""P81 Blocker 1: hard refusal when Terminal is provably off for this model.

The M1.2 retest: with Terminal off, 7 of 8 model/variant combinations
fabricated numbers, printed fake tool-call syntax as text, or asked
permission to "run" code that was never going to run, instead of declining.
The soft CODE_TOOLS_HINT/_LIGHT instruction ("check your tool list...")
is something a model can choose not to follow. HARD_NO_CODE_REFUSAL fires
instead whenever rag_filter.py can PROVE the fact from the model's own
resolved capabilities (Open WebUI's `__model__` special parameter, passed
through to `_route()` as `model`) rather than asking the model to self-report.

Offline and pure-logic, same pattern as rag_filter.py's own `__main__`
self-check (`_route_once`): `_search` stubbed to no_hits, no network, no DB.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from rag_filter import Filter, HARD_NO_CODE_REFUSAL, CODE_TOOLS_HINT, CODE_TOOLS_LIGHT


def _model(terminal):
    """A minimal model row shaped like Open WebUI's real one (see the live
    Postgres dump this was verified against: model.meta.capabilities.terminal)."""
    return {"id": "test-model", "meta": {"capabilities": {"terminal": terminal}}}


async def _no_hits(*_a, **_kw):
    return [], None, ""


def _route_once(query, **kw):
    f = Filter()
    f._search = _no_hits
    body = {"messages": [{"role": "user", "content": query}]}
    body.update(kw.pop("body_extra", {}))
    asyncio.run(f._route(body, query, kw.pop("data_files", []), **kw))
    system = "\n".join(
        m["content"] for m in body["messages"] if m.get("role") == "system"
    )
    return system


# (query, kwargs) pairs. "how many rows match this filter..." needs
# data_files=[...] attached to reach the tools block - without one it's
# correctly routed as an ordinary knowledge question instead (a different,
# already-tested branch), not a code-execution request.
CODE_QUESTIONS = [
    ("how many rows match this filter in this file?", {"data_files": [{"path": "data.csv"}]}),
    ("run analysis on the attached series", {}),
    ("calculate the totals for this dataset", {}),
]


@pytest.mark.parametrize("query,extra", CODE_QUESTIONS)
def test_hard_refusal_fires_when_terminal_explicitly_off(query, extra):
    system = _route_once(query, model=_model(terminal=False), **extra)
    assert HARD_NO_CODE_REFUSAL in system, system
    assert CODE_TOOLS_HINT not in system
    assert CODE_TOOLS_LIGHT not in system


@pytest.mark.parametrize("query,extra", CODE_QUESTIONS)
def test_soft_hint_still_used_when_terminal_on(query, extra):
    system = _route_once(query, model=_model(terminal=True), **extra)
    assert HARD_NO_CODE_REFUSAL not in system, system
    assert CODE_TOOLS_HINT in system or CODE_TOOLS_LIGHT in system


@pytest.mark.parametrize("query,extra", CODE_QUESTIONS)
def test_soft_hint_used_when_capability_unknown(query, extra):
    """No __model__ passed at all (e.g. a caller that doesn't supply it) must
    fail OPEN to the existing soft-hint behaviour, not silently refuse every
    code question on a model that actually has Terminal access."""
    system = _route_once(query, **extra)
    assert HARD_NO_CODE_REFUSAL not in system, system
    assert CODE_TOOLS_HINT in system or CODE_TOOLS_LIGHT in system


@pytest.mark.parametrize("query,extra", CODE_QUESTIONS)
def test_hard_refusal_fires_when_terminal_id_null(query, extra):
    """Live finding, 2026-10-13: disconnecting the chat UI's Terminal pill
    for a turn does NOT change the model's Admin capability (still True).
    One observed signal: Open WebUI sends `terminal_id: null` at top level."""
    extra = dict(extra, body_extra={"terminal_id": None})
    system = _route_once(query, model=_model(terminal=True), **extra)
    assert HARD_NO_CODE_REFUSAL in system, system
    assert CODE_TOOLS_HINT not in system
    assert CODE_TOOLS_LIGHT not in system


@pytest.mark.parametrize("query,extra", CODE_QUESTIONS)
def test_hard_refusal_fires_when_terminal_id_key_absent_in_real_request(query, extra):
    """Live finding, 2026-10-14: the ACTUAL signal Open WebUI sends when
    Terminal is detached from the chat UI is omitting the `terminal_id`
    key entirely — not sending it as null. Confirmed from DIAG3 logs:
      attached -> body_keys=['stream','model','terminal_id','features',...]
      detached -> body_keys=['stream','model','features',...]
    A real Open WebUI body always includes `features`; an offline test
    body does not. So: `features` present + `terminal_id` absent = Terminal
    was detached for this turn."""
    extra = dict(extra, body_extra={"features": {"web_search": False}})
    system = _route_once(query, model=_model(terminal=True), **extra)
    assert HARD_NO_CODE_REFUSAL in system, system
    assert CODE_TOOLS_HINT not in system
    assert CODE_TOOLS_LIGHT not in system


@pytest.mark.parametrize("query,extra", CODE_QUESTIONS)
def test_soft_hint_used_when_terminal_id_key_absent_in_offline_test(query, extra):
    """A body that never sets the `terminal_id` key AND has no `features`
    (every existing offline test's shape, no real Open WebUI markers) must
    NOT be treated as 'terminal off this turn'. Without `features`, we
    can't distinguish 'Terminal detached' from 'just a test body'."""
    system = _route_once(query, model=_model(terminal=True), **extra)
    assert HARD_NO_CODE_REFUSAL not in system, system
    assert CODE_TOOLS_HINT in system or CODE_TOOLS_LIGHT in system


@pytest.mark.parametrize("query,extra", CODE_QUESTIONS)
def test_soft_hint_used_when_terminal_id_present(query, extra):
    """The normal, Terminal-attached case: a real terminal_id value present
    must keep using the soft hint, not the hard refusal."""
    extra = dict(extra, body_extra={"terminal_id": "9d1a82ca-f354-4e26-8dd7-0f7cf7224754"})
    system = _route_once(query, model=_model(terminal=True), **extra)
    assert HARD_NO_CODE_REFUSAL not in system, system
    assert CODE_TOOLS_HINT in system or CODE_TOOLS_LIGHT in system


def test_hard_refusal_does_not_fire_for_non_code_questions():
    """terminal_off alone must not suppress ordinary knowledge/data-file
    turns that never claimed to need code execution in the first place."""
    system = _route_once("what is a lookup table?", model=_model(terminal=False))
    assert HARD_NO_CODE_REFUSAL not in system, system


def test_hard_refusal_does_not_fire_for_generation_requests_without_code():
    """A generation request with no compute keyword and no data file never
    claims tool access, so the hard refusal must not fire on it."""
    system = _route_once(
        "write a script that detects the documented patterns", model=_model(terminal=False)
    )
    assert HARD_NO_CODE_REFUSAL not in system, system


def test_hard_refusal_message_is_exact_and_rigid():
    """The refusal text itself must be the single allowed sentence, with no
    room for a model to add caveats, code blocks or follow-up suggestions -
    that latitude is exactly what let models narrate fake results."""
    assert (
        "I can't run code right now — code execution isn't available on "
        "this conversation." in HARD_NO_CODE_REFUSAL
    )
    assert "No code block" in HARD_NO_CODE_REFUSAL
    assert "No follow-up suggestion" in HARD_NO_CODE_REFUSAL
