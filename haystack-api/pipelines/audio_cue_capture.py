"""Audio-cue-triggered capture — spec item, confirmed not a research
problem (2026-10-08 research pass, 17-backlog.md C17): true keyword-spotting
is for live audio streams. We already have a full timestamped transcript
from Groq (D65) before the visual phase runs, so this is just matching cue
phrases against text already in hand and noting where they occur.
"""

import logging
import re
from typing import List

logger = logging.getLogger("haystack-api.audio_cue_capture")

# Cue phrases from the original capture spec: what a presenter says just
# before pointing at something on screen worth a frame grab.
_SPEC_CUE_PHRASES = [
    "look here",
    "this one",
    "notice",
    "as you can see",
    "on the screen",
    "this slide",
    "now watch",
    "watch this",
    "see this",
    "right here",
]


# How a presenter verbally points at the screen, mined from pilot
# transcripts (1,809 segments, 2026-10-20): segments containing each phrase,
# e.g. "look at" 50, "in here" 35, "right there" 31, "you can see" 25,
# "right here" 20, "up here" 18, "see how" 17, "this panel" 12, "this row" 11,
# "this item" 8. _SPEC_CUE_PHRASES above matched only 32 segments in total.
# Used by ingestion_quality.check_chart_talk; whether to also capture a frame
# on each is measured before switching on (each frame costs a Gemini call).
CHART_TALK_PHRASES = [
    "look at", "in here", "right there", "you can see", "right here", "this one",
    "you see", "up here", "see how", "this panel", "this row", "this column",
    "this here", "this area", "down here", "this section", "these sections",
    "this range", "this part", "over here", "notice", "now watch", "watch this",
]
# Capture triggers = the spec's phrases + the data-mined screen-pointing ones.
# Measured before switching on (a 9-minute pilot video, 2026-10-20): cue lines
# 2 -> 14, keyframes 106 -> 115 (+8%), +$0.0135 of Gemini for the video.
CUE_PHRASES = sorted(set(_SPEC_CUE_PHRASES) | set(CHART_TALK_PHRASES))

_CHART_TALK_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in CHART_TALK_PHRASES) + r")\b")


def is_chart_talk(text: str) -> bool:
    """True when a transcript line points at something on screen."""
    return bool(_CHART_TALK_RE.search((text or "").lower()))


def find_cue_timestamps(segments) -> List[float]:
    """Scan transcript segments for cue phrases. Returns each matching
    segment's start_time (seconds) — segment-level, not word-level: segments
    run a few seconds, which is precise enough for "grab a nearby
    frame", and matching a multi-word phrase across word boundaries isn't
    worth the complexity this stage doesn't need.
    """
    # Built per call so a caller (or a measurement) can swap CUE_PHRASES.
    cue_re = re.compile(r"\b(" + "|".join(re.escape(p) for p in CUE_PHRASES) + r")\b")
    timestamps = []
    for seg in segments:
        text_lower = (seg.text or "").lower()
        # Whole words: a bare substring let "draw" fire on "drawdown".
        if cue_re.search(text_lower):
            timestamps.append(seg.start_time)
    if timestamps:
        logger.info("found %d audio-cue timestamps: %s", len(timestamps), timestamps)
    return timestamps


def _self_test():
    class FakeSegment:
        def __init__(self, text, start_time):
            self.text = text
            self.start_time = start_time

    # 1. Matches a cue phrase, case-insensitively
    segs = [FakeSegment("So Look Here, this is the highlighted area.", 12.5)]
    assert find_cue_timestamps(segs) == [12.5]
    print("ok 1 — matches a cue phrase case-insensitively")

    # 2. No match -> empty list
    segs2 = [FakeSegment("The report opened with a summary.", 5.0)]
    assert find_cue_timestamps(segs2) == []
    print("ok 2 — no cue phrase present returns an empty list")

    # 3. Multiple segments, only matching ones returned, in order
    segs3 = [
        FakeSegment("nothing special here", 1.0),
        FakeSegment("notice the highlighted row", 4.0),
        FakeSegment("the report continued", 7.0),
        FakeSegment("now watch what happens next", 10.0),
    ]
    assert find_cue_timestamps(segs3) == [4.0, 10.0]
    print("ok 3 — returns only matching segments' timestamps, in order")

    # 4. A segment matching multiple phrases only contributes one timestamp
    segs4 = [FakeSegment("now watch, look here at this panel", 20.0)]
    assert find_cue_timestamps(segs4) == [20.0]
    print("ok 4 — a segment matching several phrases isn't double-counted")

    # 5. Every CUE_PHRASES entry is lowercase (matching is case-folded on
    # text_lower only, not on the phrase list itself)
    for phrase in CUE_PHRASES:
        assert phrase == phrase.lower(), f"{phrase!r} should be lowercase"
    print("ok 5 — CUE_PHRASES entries are all lowercase")

    # 6. Capture triggers on how a presenter actually points at the screen
    # (mined from the pilot transcripts): measured on pWnMpdN_g98, cue lines
    # 2 -> 14, keyframes 106 -> 115 (+8%, +$0.0135 Gemini for 9 min).
    segs6 = [FakeSegment("so look at this row right there", 30.0),
             FakeSegment("you can see the section up here", 45.0)]
    assert find_cue_timestamps(segs6) == [30.0, 45.0], find_cue_timestamps(segs6)
    assert set(CHART_TALK_PHRASES) <= set(CUE_PHRASES)
    print("ok 6 — the data-mined chart-pointing phrases trigger capture")

    # 7. Whole words only: "draw" must not fire on "drawdown", "notice" not on
    # "noticeable" -- the old substring match would have.
    segs7 = [FakeSegment("the drawdown was noticeable", 10.0)]
    assert find_cue_timestamps(segs7) == [], find_cue_timestamps(segs7)
    print("ok 7 — cue phrases match whole words only")

    print("\n7 checks passed")


if __name__ == "__main__":
    _self_test()
