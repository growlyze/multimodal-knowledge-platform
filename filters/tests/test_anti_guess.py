"""Anti-guess: every number in a reply must come from code output (P213, 2026-10-23).

User rule: "calculation by code, not by the model - the model is there for thinking".
Two live cases the same day: gemini-3.8 named a row's timestamp as
8 Sep 12:00 (the tool gave only a count; the real one was 9 Sep 14:50), and Opus said
a file had "85 columns" (the tool printed the column LIST; the model counted 84 names
by eye). A number in the reply's prose that no tool output and no user message
contains is untraced - a guess.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_filter import _untraced_values

TOOL = ("SHAPE (530, 84)\nTotal -1137.5 rows 20\nflag_a 13 flag_b 5 signal 2\n"
        "start 2026-09-09 14:50:00+00:00 end 2026-09-09 15:00:00+00:00 price 7699.5\n"
        "mean volume 2502.94  pct 0.2984  close 30096.75")
USER = "Compute a 9/21 moving-average crossover with a 15-unit floor and 25-unit target"


def test_numbers_copied_from_the_tool_are_traced():
    reply = ("Total -$1,137.50 over 20 rows (13 flagged / 5 flagged-b / 2 signal). "
             "The run at 7,699.50 started on 2026-09-09 at 14:50 UTC.")
    assert _untraced_values(reply, [TOOL], [USER]) == []


def test_rounding_to_the_shown_precision_is_traced():
    assert _untraced_values("Average volume is about 2,503; the jump was 0.30%.", [TOOL], []) == []


def test_user_numbers_are_traced():
    assert _untraced_values("MA 9/21 with a 15-unit floor and 25-unit target.", [TOOL], [USER]) == []


def test_utc_to_new_york_shift_is_traced():
    assert _untraced_values("It started at 10:50 New York time.", [TOOL], []) == []


def test_a_counted_by_eye_number_is_untraced():
    # Opus 2026-10-23: "The file has 85 columns" - the tool never printed 85
    assert "85" in _untraced_values("The file is a chart export with 85 columns.", [TOOL], [])


def test_an_invented_trade_time_is_untraced():
    # gemini-3.8: "the 8 Sep 12:00 UTC trade" - neither the date nor the time was printed
    out = _untraced_values("It was the 8 Sep 12:00 UTC run.", [TOOL], [])
    assert "12:00" in out


def test_code_blocks_links_and_sources_are_not_checked():
    reply = ("Done.\n\n```python\nx = 12345\n```\n[link](https://x.y/a?t=999)\n\n"
             "Sources:\n[1] [Episode 36 (1:39-2:38)](https://youtube.com/watch?v=a&t=99)")
    assert _untraced_values(reply, [TOOL], []) == []


def test_small_list_markers_are_not_flagged():
    assert _untraced_values("1. Load the file\n2. Run it", [TOOL], []) == []


# ---- wiring: the outlet logs untraced values, log only (P213 step 1) ----------
import asyncio
import copy
import logging

from rag_filter import Filter


def _quiet_filter():
    f = Filter()

    async def _none(*_a, **_k):
        return None
    f._persist_downloads = _none
    f._score_answer = lambda _b: None
    return f


def _out_msg(text):
    return {"type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text}]}


def _tool_out(text):
    return {"type": "function_call_output", "call_id": "c1",
            "output": [{"type": "input_text", "text": text}]}


def test_outlet_logs_untraced_values_and_changes_nothing(caplog):
    reply = "The file is a chart export with 85 columns and Total -$1,137.50."
    body = {"chat_id": "ag1", "messages": [
        {"role": "user", "content": "how many columns?"},
        {"role": "assistant", "content": reply,
         "output": [{"type": "function_call", "call_id": "c1", "name": "run_command"},
                    _tool_out(TOOL), _out_msg(reply)]}]}
    before = copy.deepcopy(body["messages"][-1])
    with caplog.at_level(logging.WARNING, logger="rag-filter"):
        out = asyncio.run(_quiet_filter().outlet(body))
    lines = [r.getMessage() for r in caplog.records if "anti-guess" in r.getMessage()]
    assert lines == ["anti-guess: 1 untraced value(s) in reply: ['85']"], lines
    assert out["messages"][-1] == before


def test_outlet_uses_tool_output_from_earlier_turns(caplog):
    reply = "Total was -$1,137.50 over 20 rows."
    body = {"chat_id": "ag2", "messages": [
        {"role": "user", "content": "analyse it"},
        {"role": "assistant", "content": "done", "output": [_tool_out(TOOL), _out_msg("done")]},
        {"role": "user", "content": "and the profit?"},
        {"role": "assistant", "content": reply,
         "output": [_tool_out("printed nothing numeric"), _out_msg(reply)]}]}
    with caplog.at_level(logging.WARNING, logger="rag-filter"):
        asyncio.run(_quiet_filter().outlet(body))
    assert not [r for r in caplog.records if "anti-guess" in r.getMessage()]


def test_outlet_without_tool_output_this_turn_logs_nothing(caplog):
    reply = "The file has 85 columns."
    body = {"chat_id": "ag3", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": reply, "output": [_out_msg(reply)]}]}
    with caplog.at_level(logging.WARNING, logger="rag-filter"):
        asyncio.run(_quiet_filter().outlet(body))
    assert not [r for r in caplog.records if "anti-guess" in r.getMessage()]


def test_outlet_check_never_breaks_the_reply(caplog, monkeypatch):
    import rag_filter

    def boom(*_a):
        raise RuntimeError("x")
    monkeypatch.setattr(rag_filter, "_untraced_values", boom)
    reply = "85 columns."
    body = {"chat_id": "ag4", "messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": reply, "output": [_tool_out(TOOL), _out_msg(reply)]}]}
    out = asyncio.run(_quiet_filter().outlet(body))
    assert out["messages"][-1]["content"] == reply


# ---- layer 2: the sandbox instructions carry the rule (P213 step 2) ----------
import rag_filter as _rf


def test_every_sandbox_block_carries_the_numbers_from_code_rule():
    rule = _rf.NUMBERS_FROM_CODE_RULE
    for phrase in ("output your code printed", "user's message", "conversion",
                   "in code and print it first", "by eye"):
        assert phrase in rule, phrase
    for name in ("CODE_TOOLS_HINT", "CODE_TOOLS_LIGHT", "FILE_READY_HINT",
                 "STORED_DATA_HINT"):
        assert rule in getattr(_rf, name), name
    # FILE_READY_HINT is %-formatted: the rule must survive formatting intact
    assert rule in _rf.FILE_READY_HINT % ("- /mnt/uploads/x.csv", "/x.csv", "/x.csv")
