"""F5: every cited range of one video survives into the Sources list.

CONVERSATIONS 3:29 (MINOR): "In one answer the Sources list shows only one of the
two cited ranges." haystack's format_sources_list keys a source by url/title, so two
passages of one video collapse into ONE entry carrying only the first range, and the
model is told never to add an entry.
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


# ── F5: every cited range of one video survives ─────────────────────────────

EP12 = "https://youtube.com/watch?v=abc"
R1 = "[Workshop Series Episode 12 (24:30–25:26)](%s&t=1470)" % EP12
R2 = "[Workshop Series Episode 12 (43:35–44:29)](%s&t=2615)" % EP12


def _video_hit(citation, start):
    return {"content": "the cache layer is not the last line of defence",
            "source_type": "transcript_segment", "score": 0.8, "citation": citation,
            "metadata": {"source": {"title": "Workshop Series Episode 12"}},
            "start": start}


def test_two_ranges_of_one_video_both_reach_the_sources_block():
    # haystack collapsed them to [1] R1 only (format_sources_list keys by title).
    system, _ = _system_after_inlet(
        [{"role": "user", "content": "Where does the speaker say his cache layer is "
                                     "not the last line of defence?"}],
        [_video_hit(R1, 1470), _video_hit(R2, 2615)], "Sources:\n[1] %s" % R1)
    sources = system[system.rindex("Sources:"):]
    assert R1 in sources and R2 in sources, sources
    assert sources.count(R1) == 1


def test_outlet_keeps_a_range_the_prose_quotes_by_time():
    reply = ("He says it twice: at 24:30 [1] and again at 43:35-44:29, where he "
             "repeats that it is not the last line of defence.\n\nSources:\n"
             "[1] %s\n[2] %s" % (R1, R2))
    out = rag_filter._drop_uncited_sources(reply)
    assert R2 in out, out


def test_outlet_still_drops_a_range_nothing_refers_to():
    other = "[Workshop Series Episode 41 (10:00–11:00)](https://youtube.com/watch?v=z&t=600)"
    reply = "At 24:30 he says it [1].\n\nSources:\n[1] %s\n[2] %s" % (R1, other)
    out = rag_filter._drop_uncited_sources(reply)
    assert R1 in out and other not in out, out


def test_full_outlet_chain_keeps_both_ranges_and_numbers_them():
    reply = ("He says it at 24:30-25:26 [1] and again at 43:35-44:29 [2].\n\n"
             "Sources:\n- [1] %s\n- [2] %s" % (R1, R2))
    f = Filter()
    body = {"chat_id": "c5", "messages": [{"role": "user", "content": "q"},
                                          {"role": "assistant", "content": reply}]}
    f._route_ctx["c5"] = {"terminal_off": False, "tools": False, "code": False,
                          "files": []}
    f._score_answer = lambda *_a, **_k: None
    asyncio.run(f.outlet(body))
    out = body["messages"][-1]["content"]
    assert "- [1] %s" % R1 in out and "- [2] %s" % R2 in out, out
