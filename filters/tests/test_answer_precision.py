"""2026-10-27, points 3 and 4: a wrong price must never reach an answer
presented as certain, and every price level taken from a video must come with the
video name and the frame's timestamp so he can check it.

Contract with ocr-precision: an OCR reading that failed a check arrives in the
context as `<value> [uncertain]`, e.g. `20,220.00 [uncertain]`.
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import (
    Filter, _context_block, _passages_for_outlet, _uncertain_value_hits,
    _uncited_video_prices, UNCERTAIN_VALUE_RULE, VIDEO_PRICE_CITE_RULE,
)

FRAME = {
    "source_type": "ocr_frame",
    "citation": "[Training Video Episode 12 @ 46:30](https://youtube.com/watch?v=8GkQfdAXZP0&t=2790)"
                " — Chart labels: 20,220.00 [uncertain] 20,180.25",
    "content": "Chart labels: 20,220.00 [uncertain] 20,180.25 buy stops",
}
TRANSCRIPT = {
    "source_type": "transcript_segment",
    "citation": "[Training Video Episode 12 (46:13–47:10)](https://youtube.com/watch?v=8GkQfdAXZP0&t=2773)",
    "content": "we're looking for price to reach up into 20,180.25 where the old high is",
}
DOC = {"source_type": "document", "citation": "(report.pdf)", "content": "Section notes ..."}

SOURCES_VIDEO = (
    "\n\nSources:\n"
    "- [1] [Training Video Episode 12 (46:13–47:10)](https://youtube.com/watch?v=8GkQfdAXZP0&t=2773)\n"
    "- [2] report.pdf"
)


# ---------------------------------------------------------------- prompt rules

def test_uncertain_rule_only_when_a_marker_is_in_the_context():
    assert UNCERTAIN_VALUE_RULE in _context_block([FRAME], "", True)
    assert UNCERTAIN_VALUE_RULE not in _context_block([TRANSCRIPT, DOC], "", True)


def test_uncertain_rule_says_what_was_asked_and_is_short():
    r = UNCERTAIN_VALUE_RULE.lower()
    assert "[uncertain]" in r and "never" in r and "timestamp" in r
    assert "copy" in r  # the marker itself must not reach the answer
    assert rag_filter.count_tokens(UNCERTAIN_VALUE_RULE) < 90


def test_price_cite_rule_on_video_turns_only():
    assert VIDEO_PRICE_CITE_RULE in _context_block([TRANSCRIPT], "", True)
    assert VIDEO_PRICE_CITE_RULE not in _context_block([DOC], "", True)
    assert rag_filter.count_tokens(VIDEO_PRICE_CITE_RULE) < 60
    # internal-mechanics wording must not be suggested to the user (P1.33 audit)
    assert "passages shown" not in VIDEO_PRICE_CITE_RULE


# ---------------------------------------------------------------- guard 2 (point 3)

def test_uncertain_value_stated_as_fact_is_found_with_its_source():
    p = _passages_for_outlet([FRAME, TRANSCRIPT, DOC])
    hits = _uncertain_value_hits("The high was at 20,220.00 [1]." + SOURCES_VIDEO, p)
    assert hits == [("20,220.00", "Training Video Episode 12 @ 46:30")]


def test_flagged_or_other_numbers_are_not_reported():
    p = _passages_for_outlet([FRAME, TRANSCRIPT])
    assert _uncertain_value_hits("The label looks like 20,220.00 but it is unclear on screen.", p) == []
    assert _uncertain_value_hits("Price reached 20,180.25 [1].", p) == []
    assert _uncertain_value_hits("Nothing numeric here.", p) == []


def test_value_also_read_cleanly_elsewhere_is_not_uncertain():
    clean = dict(TRANSCRIPT, content="the level 20,220.00 held")
    p = _passages_for_outlet([FRAME, clean])
    assert _uncertain_value_hits("It held 20,220.00 [1].", p) == []


def test_rounded_restatement_still_counts():
    p = _passages_for_outlet([FRAME])
    assert _uncertain_value_hits("Target 20220 next.", p) == [
        ("20220", "Training Video Episode 12 @ 46:30")]


def test_code_and_sources_are_not_claims():
    p = _passages_for_outlet([FRAME])
    assert _uncertain_value_hits("```\nlevel = 20220.00\n```\nSee code.", p) == []


# ---------------------------------------------------------------- guard 3 (point 4)

def test_uncited_video_price_is_found():
    p = _passages_for_outlet([TRANSCRIPT, DOC])
    assert _uncited_video_prices("The old high sits at 20,180.25." + SOURCES_VIDEO, p) == ["20,180.25"]


def test_video_price_with_video_citation_passes():
    p = _passages_for_outlet([TRANSCRIPT, DOC])
    assert _uncited_video_prices("The old high sits at 20,180.25 [1]." + SOURCES_VIDEO, p) == []


def test_video_price_cited_only_to_a_document_is_reported():
    p = _passages_for_outlet([TRANSCRIPT, DOC])
    assert _uncited_video_prices("The old high sits at 20,180.25 [2]." + SOURCES_VIDEO, p) == ["20,180.25"]


def test_non_price_numbers_are_ignored():
    p = _passages_for_outlet([dict(TRANSCRIPT, content="step 3 of 12, 2022 lesson at 9:30")])
    assert _uncited_video_prices("Step 3 of 12 in 2022 at 9:30." + SOURCES_VIDEO, p) == []


# ---------------------------------------------------------------- end to end through outlet

def _run(reply, hits, caplog):
    f = Filter()

    async def search(*_a, **_kw):
        return hits, None, ""
    f._search = search
    body = {"chat_id": "ap-1", "messages": [{"role": "user", "content": "What level did he mark in Episode 12 at 46:30?"}]}
    asyncio.run(f.inlet(body, __model__={"id": "m", "meta": {"capabilities": {"terminal": True}}}))
    out = {"chat_id": "ap-1", "messages": [body["messages"][-1],
                                          {"role": "assistant", "content": reply}]}
    with caplog.at_level(logging.WARNING, logger="rag-filter"):
        asyncio.run(f.outlet(out))
    return out["messages"][-1]["content"]


def test_outlet_appends_note_for_uncertain_value_and_keeps_number(caplog):
    reply = "He marked the high at 20,220.00 [1]." + SOURCES_VIDEO
    out = _run(reply, [FRAME, TRANSCRIPT], caplog)
    assert "20,220.00 [1]" in out  # never silently edited
    assert "20,220.00" in out.split("Sources:")[-1]
    assert "Training Video Episode 12 @ 46:30" in out.split("Sources:")[-1]
    assert "uncertain-value:" in caplog.text


def test_outlet_leaves_clean_reply_alone(caplog):
    reply = "He marked the old high at 20,180.25 [1]." + SOURCES_VIDEO
    out = _run(reply, [FRAME, TRANSCRIPT], caplog)
    assert "uncertain-value:" not in caplog.text
    assert "uncited-video-price:" not in caplog.text
    assert out.count("20,180.25") == 1


def test_outlet_logs_uncited_video_price(caplog):
    reply = "He marked the old high at 20,180.25." + SOURCES_VIDEO
    _run(reply, [FRAME, TRANSCRIPT], caplog)
    assert "uncited-video-price:" in caplog.text


def test_percentages_are_not_prices():
    # Replay 2026-10-28: the only upper-bound hits on 28 cached video replies were
    # "labels the 70.5% level" - a ratio, not a price.
    p = _passages_for_outlet([dict(TRANSCRIPT, content="the 70.5% level and 62% retracement")])
    assert _uncited_video_prices("It labels the 70.5% level." + SOURCES_VIDEO, p) == []


def test_note_reaches_the_rendered_output(caplog):
    # P196: the chat renders `output`, not `content`.
    f = Filter()

    async def search(*_a, **_kw):
        return [FRAME, TRANSCRIPT], None, ""
    f._search = search
    body = {"chat_id": "ap-2", "messages": [{"role": "user", "content": "Episode 12 levels at 46:30?"}]}
    asyncio.run(f.inlet(body, __model__={"id": "m", "meta": {"capabilities": {"terminal": True}}}))
    reply = "He marked 20,220.00 [1]." + SOURCES_VIDEO
    msg = {"role": "assistant", "content": reply,
           "output": [{"type": "message", "content": [{"type": "output_text", "text": reply}]}]}
    asyncio.run(f.outlet({"chat_id": "ap-2", "messages": [body["messages"][-1], msg]}))
    shown = msg["output"][-1]["content"][0]["text"]
    assert "Check before relying on this" in shown and "Episode 12 @ 46:30" in shown
