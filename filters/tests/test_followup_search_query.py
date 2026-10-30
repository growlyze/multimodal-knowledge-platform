"""Follow-up turns search with the conversation, not the bare message (P197, 2026-10-23).

M3's live browser run (12 chats, Sonnet + Opus): haystack logged
`search 'What happened next?' -> 0 results` and `search 'what about at 44:30?' -> 0
results` - _route sent the last user message alone, so a follow-up carried no
video and no topic. 6 of the 7 failed turns were this; on Opus the empty search
also fired the "isn't covered in the knowledge base" label about Episode 12/36,
which ARE in the library. Rule: a follow-up that names no video of its own is
searched as "<previous user message> <follow-up>"; the model still sees only the
user's words. A message that names its own video, or stands on its own, is
searched unchanged.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_filter import _search_query

PREV = "In Episode 36, what happened after the lower level was taken at the low?"
H = [{"role": "user", "content": PREV}, {"role": "assistant", "content": "He waited for displacement."}]

FOLLOW_UPS = [
    "What happened next?",
    "what about at 44:30?",
    "and after that?",
    "What does he do then?",
    "At 25:10?",
    "And around 31 minutes in?",
    "What does he say right before that?",
    "Can you explain that part again?",
    "what about in the other video?",
    "and the same video at 1:02:00?",
    "Why does he do that?",
    "ok, and then?",
]

STANDALONE = [
    "And in Part 4, what is the opening summary?",
    "In Episode 36, what happens at 3:10:00?",
    "What is a data retention policy?",
    "Does any video show a bar chart?",
    "Summarize the video he posted on X.",
    "Write a script that marks the outliers.",
    "What do the videos say about the new format?",
    "In episde 12, around 25 minutes in, what is he forecasting?",
]


@pytest.mark.parametrize("q", FOLLOW_UPS)
def test_follow_up_is_searched_with_the_previous_turn(q):
    s = _search_query(q, H)
    assert s.startswith(PREV) and s.endswith(q), s


@pytest.mark.parametrize("q", STANDALONE)
def test_standalone_message_is_searched_unchanged(q):
    assert _search_query(q, H) == q


def test_no_history_means_no_change():
    assert _search_query("What happened next?", []) == "What happened next?"
    assert _search_query("What happened next?", None) == "What happened next?"


def test_only_the_last_user_turn_is_used_and_it_is_capped():
    long_prev = "In Episode 12, " + "x " * 400
    h = [{"role": "user", "content": "old question about Episode 36"},
         {"role": "assistant", "content": "a"}, {"role": "user", "content": long_prev},
         {"role": "assistant", "content": "b"}]
    s = _search_query("what happened next?", h)
    assert "Episode 36" not in s and s.startswith("In Episode 12") and len(s) <= 600


def _system_after_route(messages, hits=()):
    import asyncio
    import rag_filter

    async def _search(*_a, **_k):
        return list(hits), None, ""
    f = rag_filter.Filter()
    f._search = _search
    body = {"messages": messages}
    asyncio.run(f._route(body, messages[-1]["content"], []))
    return "\n".join(m["content"] for m in body["messages"] if m.get("role") == "system")


def test_empty_search_on_a_named_library_video_never_says_not_covered():
    # Opus, M3 run: "This isn't covered in the knowledge base yet" about Episode 36/12,
    # which are in the library - the search had come back empty (the follow-up bug).
    import rag_filter
    for msgs in (
        [{"role": "user", "content": "In Episode 36, what does he say about the levels at 44:30?"}],
        H + [{"role": "user", "content": "And what does he say about the level then?"}],
    ):
        system = _system_after_route(msgs)
        assert rag_filter.NO_KB_INFO not in system, msgs[-1]
        assert rag_filter.VIDEO_MOMENT_NOT_FOUND in system, msgs[-1]


def test_video_passages_come_with_the_quote_only_shown_times_rule():
    # Opus, M3 run chat 228e0f38 t1: quoted 5:35, 3:08, 3:55 for Episode 36 while the
    # only passage was 2:24; retracted them the next turn.
    import rag_filter
    video = [{"content": "he takes the sell side", "source_type": "transcript_segment",
              "citation": "Training Video Episode 36 (2:24-3:02)", "score": 0.9}]
    text = [{"content": "a doc", "source_type": "pdf", "citation": "doc.pdf - Page 1", "score": 0.9}]
    assert rag_filter.VIDEO_TIME_RULE in rag_filter._context_block(video, "", True)
    assert rag_filter.VIDEO_TIME_RULE not in rag_filter._context_block(text, "", True)


# P197b (M3 combined run 2026-10-23 13:01 UTC): a follow-up to a follow-up lost
# the video - `search 'What happened next? what about at 30:00?' -> 0 results` in a
# chain that began "In Episode 36, ...". The anchor (latest message naming a video or
# standing on its own) must survive any number of follow-ups.
def _chain(*user_turns):
    h = []
    for u in user_turns:
        h += [{"role": "user", "content": u}, {"role": "assistant", "content": "..."}]
    return h


def test_a_chain_of_follow_ups_keeps_the_video_anchor():
    h = _chain("In Episode 36, what happened after the lower level was taken?", "What happened next?")
    s = _search_query("what about at 30:00?", h)
    assert "Episode 36" in s and "What happened next?" in s and s.endswith("what about at 30:00?"), s
    h3 = h + _chain("what about at 30:00?")
    s3 = _search_query("then what?", h3)
    assert "Episode 36" in s3 and s3.endswith("then what?"), s3


def test_the_anchor_is_the_latest_standalone_turn_not_an_older_video():
    h = _chain("In Episode 36, what happened at the open?", "What is a data retention policy?")
    s = _search_query("and then?", h)
    assert "Episode 36" not in s and "data retention policy" in s, s


def test_chain_query_stays_capped():
    h = _chain("In Episode 12, " + "y " * 500, "what about at 10:00?", "and then?")
    assert len(_search_query("what next?", h)) <= 900


# F3 (P202 filter half, sprint 2026-10-25): the exact chain from the M3 test plan.
# 1.34 dropped the NEW message from the search; every turn must carry it, and the
# Episode 36 anchor must survive both follow-ups.
def test_ep36_then_4430_then_what_chain_keeps_anchor_and_new_message():
    first = "In Ep 36, what does he say about the lower level at the open?"
    h = _chain(first)
    s2 = _search_query("what about at 44:30?", h)
    assert s2.startswith(first) and s2.endswith("what about at 44:30?"), s2
    s3 = _search_query("then what?", h + _chain("what about at 44:30?"))
    assert s3.startswith(first) and "44:30" in s3 and s3.endswith("then what?"), s3
