"""P226 (rag_filter 1.45): a follow-up sent after the browser's socket dropped reaches the inlet with
no chat_id and no user message (Open WebUI 0.11.4) - stop it with a readable
message instead of forwarding a system-only payload Anthropic rejects with 400.
"""
import asyncio
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import Filter, _orphaned_followup

SYSTEM = {"role": "system", "content": "You are helpful."}
USER_MSG = {"id": "u2", "parentId": "a1", "role": "user", "content": "and on the 4h?"}


# ── the detector, one case per request shape Open WebUI 0.11.4 produces ─────

def test_orphaned_followup_is_detected():
    # Live repro: no chat_id, session_id None, only the system prompt in
    # messages, the new question carried in metadata.user_message.
    meta = {"chat_id": "", "session_id": None, "user_message": USER_MSG}
    assert _orphaned_followup([SYSTEM], meta) is True


def test_orphaned_followup_with_no_messages_at_all_is_detected():
    meta = {"chat_id": "", "user_message": USER_MSG}
    assert _orphaned_followup([], meta) is True


def test_persisted_turn_rebuilt_from_db_is_not_orphaned():
    # Saved chat_id: middleware rebuilt history from the DB, user message present.
    meta = {"chat_id": "c1", "session_id": "s1", "user_message": USER_MSG}
    msgs = [SYSTEM, {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"}, {"role": "user", "content": "and on the 4h?"}]
    assert _orphaned_followup(msgs, meta) is False


def test_saved_chat_id_with_empty_body_is_not_orphaned():
    # A saved chat_id means history could be rebuilt; an empty body there is a
    # different fault, and "your connection dropped" would mislead the user.
    meta = {"chat_id": "c1", "session_id": "s1", "user_message": USER_MSG}
    assert _orphaned_followup([SYSTEM], meta) is False


def test_api_caller_without_session_id_is_not_orphaned():
    # live_api_client.py: no session_id, no user_message, full history in messages.
    meta = {"chat_id": "", "session_id": None, "user_message": None}
    assert _orphaned_followup([{"role": "user", "content": "hello"}], meta) is False


def test_new_chat_first_turn_is_not_orphaned():
    meta = {"chat_id": "new-uuid", "session_id": "s1", "user_message": USER_MSG}
    assert _orphaned_followup([{"role": "user", "content": "hello"}], meta) is False


def test_temporary_chat_is_not_orphaned():
    # Temporary chats are never saved, so the frontend sends the whole history.
    meta = {"chat_id": "local:abc", "session_id": "s1", "user_message": USER_MSG}
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "and on the 4h?"}]
    assert _orphaned_followup(msgs, meta) is False


@pytest.mark.parametrize("um", [
    None, {}, {"role": "user", "content": ""}, {"role": "user", "content": "   "},
    {"role": "user", "content": [{"type": "text", "text": ""}]}, "not-a-dict",
])
def test_empty_user_message_is_not_orphaned(um):
    assert _orphaned_followup([SYSTEM], {"chat_id": "", "user_message": um}) is False


def test_missing_metadata_is_not_orphaned():
    assert _orphaned_followup([SYSTEM], None) is False


def test_list_content_user_message_counts_as_content():
    um = {"role": "user", "content": [{"type": "text", "text": "chart?"}]}
    assert _orphaned_followup([SYSTEM], {"user_message": um}) is True


# ── the inlet stops the turn, and only that turn ────────────────────────────

def test_inlet_raises_readable_message_and_logs_tag(caplog):
    f = Filter()
    body = {"messages": [dict(SYSTEM)],
            "metadata": {"chat_id": "", "session_id": None, "user_message": USER_MSG}}
    with caplog.at_level(logging.WARNING, logger="rag-filter"):
        with pytest.raises(Exception) as exc:
            asyncio.run(f.inlet(body, __user__={"id": "user-7"}))
    assert str(exc.value) == rag_filter.ORPHANED_FOLLOWUP_MESSAGE
    assert "refresh the page" in str(exc.value)
    lines = [r.getMessage() for r in caplog.records if r.name == "rag-filter"]
    tagged = [m for m in lines if "orphaned follow-up (no chat_id, no user message)" in m]
    assert len(tagged) == 1, lines
    assert "user-7" in tagged[0] and "session_id=absent" in tagged[0]


def test_inlet_reads_dunder_metadata_when_body_has_none():
    f = Filter()
    body = {"messages": [dict(SYSTEM)]}
    meta = {"chat_id": "", "session_id": "sock1", "user_message": USER_MSG}
    with pytest.raises(Exception) as exc:
        asyncio.run(f.inlet(body, __metadata__=meta))
    assert str(exc.value) == rag_filter.ORPHANED_FOLLOWUP_MESSAGE


def test_inlet_api_caller_is_untouched():
    async def _search(*_a, **_k):
        return [], None, ""
    f = Filter()
    f._search = _search
    body = {"messages": [{"role": "user", "content": "hello"}],
            "metadata": {"chat_id": "", "session_id": None, "user_message": None}}
    out = asyncio.run(f.inlet(body))
    assert any(m.get("role") == "user" for m in out["messages"])
