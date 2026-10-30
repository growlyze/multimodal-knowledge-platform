"""Outlet corrections must reach the SCREEN, not only `content` (P196, 2026-10-23).

Open WebUI 0.11 stores a reply twice: `content` and `output` (the model's message
items). The chat renders `output`. The outlet rewrote `content` only, so every
correction - citation renumbering, the "in the context" rewording, the blank-reply
notice, the download line - was
saved but never shown. Seen live: chat f7cc8cdf (2026-10-22 20:31, Sonnet) shows
Sources [1] [3] [6] on screen while `content` holds the renumbered [1] [2] [3].
Open WebUI passes `output` to the outlet and saves/redraws it when it changes
(utils/middleware.py outlet_filter_handler: output_changed), so syncing it is the fix.
"""
import asyncio
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import Filter, EMPTY_REPLY_NOTICE, _sync_output


def _msg(text, **kw):
    return {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text}], **kw}


def _output_text(output):
    return "\n".join(
        "".join(p.get("text", "") for p in it.get("content") or [])
        for it in output if it.get("type") == "message"
        and "".join(p.get("text", "") for p in it.get("content") or []).strip())


RAW = ("Here is the script.\n\n```python\nprint(summary(rows))\n```\n\n"
       "Sources:\n[1] [Ep 36](https://example.com/watch?v=a&t=1)\n"
       "[3] [Ep 12](https://example.com/watch?v=b&t=2)\n[6] [Boxes](https://x.y/boxes)")


def test_sync_rewrites_the_single_message_item():
    m = {"role": "assistant", "content": RAW.replace("[3]", "[2]").replace("[6]", "[3]"),
         "output": [_msg(RAW)]}
    _sync_output(m, RAW)
    assert _output_text(m["output"]) == m["content"]
    assert m["output"][0]["id"] == "msg_1" and m["output"][0]["status"] == "completed"


def test_sync_keeps_earlier_narration_when_only_the_last_part_changed():
    first, last = "Let me load the file.", "Done.\n\nSources:\n[1] a\n[3] b"
    out = [_msg(first), {"type": "function_call", "name": "run_command"},
           {"type": "function_call_output", "output": "ok"}, _msg(last, id="msg_2")]
    original = first + "\n" + last
    m = {"role": "assistant", "content": first + "\nDone.\n\nSources:\n[1] a\n[2] b", "output": out}
    _sync_output(m, original)
    assert _output_text(m["output"]) == m["content"]
    assert m["output"][0]["content"][0]["text"] == first
    assert [i["type"] for i in m["output"]] == ["message", "function_call", "function_call_output", "message"]


def test_sync_puts_everything_in_the_last_part_when_an_earlier_one_changed():
    a, b = "See [3] above.", "Sources:\n[1] x\n[3] y"
    m = {"role": "assistant", "content": "See [2] above.\nSources:\n[1] x\n[2] y",
         "output": [_msg(a), {"type": "function_call"}, _msg(b, id="msg_2")]}
    _sync_output(m, a + "\n" + b)
    assert _output_text(m["output"]) == m["content"]


def test_sync_adds_a_message_for_a_blank_reply():
    m = {"role": "assistant", "content": EMPTY_REPLY_NOTICE, "output": []}
    _sync_output(m, "")
    assert _output_text(m["output"]) == EMPTY_REPLY_NOTICE


def test_sync_leaves_unchanged_and_output_less_messages_alone():
    m = {"role": "assistant", "content": RAW, "output": [_msg(RAW)]}
    before = copy.deepcopy(m)
    _sync_output(m, RAW)
    assert m == before
    m2 = {"role": "assistant", "content": "x"}
    _sync_output(m2, "y")
    assert "output" not in m2


def test_outlet_end_to_end_screen_shows_the_corrected_sources():
    f = Filter()
    f._route_ctx["c1"] = {"files": []}

    f._score_answer = lambda _b: None

    async def _no_dl(*_a, **_k):
        return None
    f._persist_downloads = _no_dl
    raw = "Gap respected [1][3].\n\nSources:\n[1] [Ep 36](u1)\n[2] [Ep 12](u2)\n[3] [Boxes](u3)"
    body = {"chat_id": "c1", "messages": [
        {"role": "user", "content": "write the summary script"},
        {"role": "assistant", "content": raw, "output": [_msg(raw)]}]}
    out = asyncio.run(f.outlet(body))
    last = out["messages"][-1]
    assert last["content"] != raw, "a guard should have changed the text"
    assert _output_text(last["output"]) == last["content"], "screen and content diverge"
