"""Semantic segmentation by topic/concept for video transcripts.

Client said this needed more research before building (2026-10-08). Research
(17-backlog.md C17) found "Unsupervised Audio-Visual Lecture Segmentation"
(WACV 2023, the AVLectures dataset) segments lecture video using transcript
+ visual cues jointly — closer to our content than generic embedding-only
semantic chunkers (Chonkie, EchoGem), which only see transcript text and
ignore the visual-change timeline we already compute in _phase_visual.

This module's boundary rule: a transcript topic shift (cosine similarity
drop between consecutive segment embeddings, same signal
pipelines/chunkers/semantic_chunker.py already uses for general RAG text)
only becomes a real topic boundary when a real screen change — a keyframe
captured for reason "change" or "burst" — happens near it. A transcript
rephrasing with no visual change is a weaker signal than one that coincides
with the presenter actually moving to a new chart/setup.

Takes pre-computed embeddings (e.g. _phase_embed's transcript_vectors)
rather than embedding anything itself — the whole point is not paying for a
second round of API calls just to find topic boundaries.
"""

import logging
from typing import List, Optional

from pipelines.chunkers.semantic_chunker import cosine as _cosine

logger = logging.getLogger("haystack-api.video_topic_segmentation")

# Same default as semantic_chunker.py's SIMILARITY_THRESHOLD — consecutive
# segments below this cosine similarity are a candidate topic shift.
DEFAULT_SIMILARITY_THRESHOLD = 0.5
# How close a real screen change needs to be to a candidate text boundary to
# confirm it as a real topic shift, not just a paraphrase.
DEFAULT_VISUAL_WINDOW_S = 5.0


def segment_by_topic(
    segments,
    vectors: List[List[float]],
    keyframe_timestamps_ms: Optional[List[float]] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    visual_window_s: float = DEFAULT_VISUAL_WINDOW_S,
) -> List[int]:
    """Assign a topic_id (int, starting at 0) to each segment.

    `segments` and `vectors` must be the same length and order (e.g.
    non_empty_segs and transcript_vectors from _phase_index — already
    computed, not re-embedded here).

    Without `keyframe_timestamps_ms`, falls back to transcript-only
    boundaries (same signal as semantic_chunker.py, just applied to video
    segment-level grouping instead of RAG chunk packing) — still useful,
    just without the visual confirmation this module exists to add.

    Never raises: a wrong/missing grouping (single topic for the whole
    video) degrades retrieval precision, a crashed ingestion loses
    everything — same tolerance every other non-critical phase in this
    pipeline already has.
    """
    n = len(segments)
    if n != len(vectors):
        logger.warning("topic segmentation: %d segments but %d vectors — single-topic fallback",
                       n, len(vectors))
        return [0] * n
    if n <= 1:
        return [0] * n

    keyframe_timestamps_ms = keyframe_timestamps_ms or []
    topic_ids = [0] * n
    current_topic = 0
    shifts = 0

    for i in range(1, n):
        sim = _cosine(vectors[i - 1], vectors[i])
        if sim < similarity_threshold:
            boundary_ms = segments[i].start_time * 1000
            if keyframe_timestamps_ms:
                visually_confirmed = any(
                    abs(boundary_ms - kf_ms) <= visual_window_s * 1000
                    for kf_ms in keyframe_timestamps_ms
                )
            else:
                visually_confirmed = True  # no visual signal available — text-only mode
            if visually_confirmed:
                current_topic += 1
                shifts += 1
        topic_ids[i] = current_topic

    logger.info("topic segmentation: %d segments -> %d topics (%d confirmed shifts)",
                n, current_topic + 1, shifts)
    return topic_ids


def _self_test():
    class FakeSegment:
        def __init__(self, start_time):
            self.start_time = start_time

    # 1. Empty input
    assert segment_by_topic([], []) == []
    print("ok 1 — empty input returns an empty list")

    # 2. Single segment is its own topic (0)
    assert segment_by_topic([FakeSegment(0.0)], [[1.0, 0.0]]) == [0]
    print("ok 2 — a single segment is topic 0")

    # 3. Mismatched lengths degrade to single-topic rather than raising
    result = segment_by_topic([FakeSegment(0), FakeSegment(1)], [[1.0, 0.0]])
    assert result == [0, 0], result
    print("ok 3 — mismatched segments/vectors lengths degrade to single-topic")

    # 4. Text-only mode (no keyframe timestamps): a similarity drop alone is
    # enough to create a new topic — same behaviour semantic_chunker.py has.
    segs = [FakeSegment(0.0), FakeSegment(5.0), FakeSegment(10.0)]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]  # orthogonal at index 2
    result2 = segment_by_topic(segs, vectors)
    assert result2 == [0, 0, 1], result2
    print("ok 4 — text-only mode creates a topic on similarity drop alone")

    # 5. With keyframe timestamps given, a text-only similarity drop with NO
    # nearby visual change is NOT confirmed as a topic shift.
    result3 = segment_by_topic(segs, vectors, keyframe_timestamps_ms=[500])  # far from boundary at 10s
    assert result3 == [0, 0, 0], result3
    print("ok 5 — a text shift with no nearby visual change is not confirmed")

    # 6. A text shift WITH a nearby real screen change IS confirmed.
    result4 = segment_by_topic(segs, vectors, keyframe_timestamps_ms=[9800])  # 200ms from the 10s boundary
    assert result4 == [0, 0, 1], result4
    print("ok 6 — a text shift confirmed by a nearby visual change becomes a topic boundary")

    # 7. Multiple confirmed shifts increment the topic id each time
    segs2 = [FakeSegment(t) for t in (0.0, 5.0, 10.0, 15.0)]
    vectors2 = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]  # shifts at 1 and 3
    result5 = segment_by_topic(segs2, vectors2, keyframe_timestamps_ms=[4900, 14900])
    assert result5 == [0, 1, 1, 2], result5
    print("ok 7 — multiple confirmed shifts produce multiple topic ids")

    print("\n7 checks passed")


if __name__ == "__main__":
    _self_test()
