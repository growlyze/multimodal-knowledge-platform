"""F6: a question about the video library never gets the sandbox or the
generation route, and the model is told the videos are not files (sprint
2026-10-25).

The client's M3 pilot test (CONVERSATIONS 3:29): "In Episode 12, what does the speaker
draw or mark on the chart around 1:08:00?" ran match_files/list_files for ~10
minutes, and "Do any of the videos show chart labels like STH ...?" "searched the
sandbox files". The first was routed as a script-generation turn ("draw ... chart"),
the second as a code turn ("chart" is a CODE_KEYWORD). F6 (b): the sandbox hints
forbid filesystem-wide searches and cap the ones that run with `timeout`.
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


# ── F6 (a): video questions never get the sandbox or generation route ─────────

VIDEO_QUESTIONS = [
    "In Episode 12, what does the speaker draw or mark on the chart around 1:08:00?",
    "Do any of the videos show diagram labels like Input, Output or Cache? "
    "In which video and at what time?",
    "Does any video show a bar chart?",
    "In Episode 12, is there a chart with a 'Breakout' label? "
    "What does it show and at what time?",
    "Where does the speaker say his first step is not the last checkpoint "
    "before the release?",
    "According to the videos, what is the first lesson?",
    "What does he plot on the chart in Part 4?",
]


@pytest.mark.parametrize("q", VIDEO_QUESTIONS)
def test_video_question_is_recognised_and_routes_to_neither_sandbox_nor_generation(q):
    assert rag_filter.is_video_question(q)
    assert rag_filter._turn_flags(q, [], False) is False, q


# The other side: these must keep their route. (query, expected code flag,
# is-generation). The first row: a code-generation turn naming a video produces
# the artifact, not a video answer.
NOT_VIDEO = [
    ("Write a script that draws the zones he marks in Episode 12", False, True),
    ("compute the totals from the video on my CSV", True, False),
    ("plot the totals for that run", True, False),
    ("calculate the totals for this file", True, False),
]


@pytest.mark.parametrize("q,code,generation", NOT_VIDEO)
def test_data_and_script_requests_naming_a_video_keep_their_route(q, code, generation):
    assert not rag_filter.is_video_question(q), q
    assert rag_filter._turn_flags(q, [], False) is code, q
    assert rag_filter.is_script_generation_request(q) is generation, q


def test_a_follow_up_in_a_video_conversation_is_a_video_turn():
    h = [{"role": "user", "content": "In Episode 12, what does he say at 1:07:55?"},
         {"role": "assistant", "content": "He explains the gap [1]."}]
    for q in ("what does he draw on the chart then?", "then what?"):
        assert rag_filter.is_video_question(q, h), q
        assert rag_filter._turn_flags(q, h, False) is False, q


def test_an_attached_data_file_still_goes_to_the_sandbox():
    # The file decides on a file turn; the video veto is for file-less questions.
    q = "Do the videos show this chart pattern in my file?"
    assert not rag_filter.is_video_question(q)
    system, _ = _system_after_inlet([{"role": "user", "content": q}])
    assert rag_filter.VIDEO_NO_FILE_TOOLS not in system


def test_video_turn_gets_the_no_file_tools_rule_and_no_sandbox_hint():
    q = VIDEO_QUESTIONS[0]
    hit = {"content": "he marks the low", "source_type": "transcript_segment",
           "score": 0.8, "citation": "[Training Episode 12 (1:07:55–1:08:53)]"
           "(https://example.com/watch?v=a&t=4075)"}
    for hits in ([hit], []):
        system, _ = _system_after_inlet([{"role": "user", "content": q}], hits)
        assert rag_filter.VIDEO_NO_FILE_TOOLS in system
        assert rag_filter.CODE_TOOLS_HINT not in system
        assert rag_filter.CODE_TOOLS_LIGHT not in system


def test_empty_search_on_a_general_video_question_is_labelled_not_covered():
    # "Do any of the videos show ...?" names no single video, so the
    # VIDEO_MOMENT_NOT_FOUND branch missed it and OUT_OF_SCOPE_OK told the model to
    # answer freely from its own knowledge - about what the VIDEOS show.
    system, _ = _system_after_inlet([{"role": "user", "content": VIDEO_QUESTIONS[1]}])
    assert rag_filter.VIDEO_NOT_COVERED in system
    assert rag_filter.OUT_OF_SCOPE_OK not in system
    assert "not covered in the knowledge base" in rag_filter.VIDEO_NOT_COVERED


def test_non_video_turn_does_not_get_the_video_rule():
    system, _ = _system_after_inlet(
        [{"role": "user", "content": "calculate the totals for this file"}])
    assert rag_filter.VIDEO_NO_FILE_TOOLS not in system


def test_video_rule_names_the_file_tools_it_forbids():
    for tool in ("run_command", "list_files", "match_files"):
        assert tool in rag_filter.VIDEO_NO_FILE_TOOLS


# ── F6 (b): the sandbox hints forbid whole-filesystem searches ───────────────

def test_sandbox_hints_forbid_filesystem_wide_searches_and_cap_them():
    for block in (rag_filter.CODE_TOOLS_HINT, rag_filter.CODE_TOOLS_LIGHT):
        assert rag_filter.FILE_SEARCH_RULE in block
    assert "find /" in rag_filter.FILE_SEARCH_RULE
    assert "timeout" in rag_filter.FILE_SEARCH_RULE
