"""Deterministic quality checks for M3 ingestion phases.

Each check returns a label/score/reason that video_ingestion.py writes onto
the phase's span via record_decision() — the same convention retrieval.py
already uses for its confidence gate (`retrieval.decision`/`retrieval.reason`),
just applied to ingestion. Thresholds are grounded in real numbers already in
this repo (Whisper's own decoding defaults, the M3 SSIM calibration), not
guessed, so a "poor" label points at a specific documented expectation.
"""

from dataclasses import dataclass


@dataclass
class QualityCheck:
    label: str
    score: float
    reason: str


# Whisper's own DecodingOptions.logprob_threshold defaults to -1.0 — the
# stock heuristic for "this segment's transcription likely failed". Reused
# here rather than inventing a domain-specific number with no evidence.
ASR_LOGPROB_FLOOR = -1.0

# docs/M3-docs/03-tuning-values.md: SSIM 0.15 (the value this pipeline uses)
# is calibrated to ~40 keyframes/hour; 0.10 (too sensitive) measured
# ~120/hr, 0.20 (too conservative) measured ~25/hr. Outside that explored
# band points at this specific video (near-static, or a detector misfire),
# not at the global threshold being wrong.
KEYFRAMES_PER_HOUR_LOW = 20.0
KEYFRAMES_PER_HOUR_HIGH = 130.0

# Below this fraction of eligible items actually got embedded, something
# upstream silently dropped items (e.g. a Voyage batch failure that degraded
# to an empty list rather than raising).
EMBEDDING_COMPLETENESS_FLOOR = 0.9


def check_asr_quality(segments, duration_s: float = 0.0) -> QualityCheck:
    """segments: list of objects with .confidence (= WhisperX avg_logprob) and .text."""
    if not segments:
        return QualityCheck("empty", 0.0, "no transcript segments produced")
    avg_logprob = sum(s.confidence for s in segments) / len(segments)
    if avg_logprob < ASR_LOGPROB_FLOOR:
        return QualityCheck(
            "low_confidence", avg_logprob,
            f"avg_logprob {avg_logprob:.2f} below Whisper's own "
            f"{ASR_LOGPROB_FLOOR} failure floor")
    if duration_s > 60 and len(segments) < duration_s / 60:
        return QualityCheck(
            "sparse", avg_logprob,
            f"only {len(segments)} segments for {duration_s / 60:.1f} minutes of video")
    return QualityCheck(
        "ok", avg_logprob, f"{len(segments)} segments, avg_logprob {avg_logprob:.2f}")


def check_visual_quality(keyframe_count: int, duration_s: float = 0.0) -> QualityCheck:
    if duration_s <= 0:
        return QualityCheck("unknown", 0.0, "no duration available to judge keyframe density")
    per_hour = keyframe_count / (duration_s / 3600)
    if per_hour < KEYFRAMES_PER_HOUR_LOW:
        return QualityCheck(
            "too_sparse", per_hour,
            f"{per_hour:.0f} keyframes/hr, below the {KEYFRAMES_PER_HOUR_LOW:.0f}/hr floor "
            f"calibrated in docs/M3-docs/03-tuning-values.md")
    if per_hour > KEYFRAMES_PER_HOUR_HIGH:
        return QualityCheck(
            "too_dense", per_hour,
            f"{per_hour:.0f} keyframes/hr, above the {KEYFRAMES_PER_HOUR_HIGH:.0f}/hr ceiling — "
            f"SSIM threshold may be misfiring on this video")
    return QualityCheck("ok", per_hour, f"{per_hour:.0f} keyframes/hr")


def check_embedding_consistency(expected_counts: dict, total_emb: int) -> QualityCheck:
    """expected_counts: {"clip": n, "ocr": n, "od": n, "asr": n} — eligible item
    counts per track, before embedding. Flags silent drops, not just zero output."""
    expected_total = sum(expected_counts.values())
    if expected_total == 0:
        return QualityCheck("empty", 0.0, "no items were eligible for embedding")
    if total_emb == 0:
        return QualityCheck("failed", 0.0, f"expected {expected_total} vectors, got 0")
    ratio = total_emb / expected_total
    if ratio < EMBEDDING_COMPLETENESS_FLOOR:
        return QualityCheck(
            "incomplete", ratio, f"{total_emb}/{expected_total} items embedded ({ratio:.0%})")
    return QualityCheck("ok", ratio, f"{total_emb}/{expected_total} items embedded")


# Frames actually read vs the video's length (sampling is 1 frame/s). Below
# 5% is "nothing could be decoded" -- 0 keyframes on an AV1 file used to
# "succeed" silently (P137); below 90% a partial decode (truncated file).
FRAME_COVERAGE_UNREADABLE = 0.05
FRAME_COVERAGE_PARTIAL = 0.90

# P141: the speaker pointed at the chart all through a video while ~0 keyframes were
# classified as charts (light theme). Heavy chart talk = at least this many
# pointing lines per 10 minutes; "almost no charts" = under this share.
CHART_TALK_PER_10MIN = 3.0
CHART_SHARE_FLOOR = 0.10

# Transcript should reach close to the end (trailing silence/outro allowed),
# and a hole this long mid-video means audio or ASR dropped out.
TRANSCRIPT_END_COVERAGE = 0.85
TRANSCRIPT_MAX_GAP_S = 600.0


def check_frame_coverage(sampled_frames: int, duration_s: float = 0.0) -> QualityCheck:
    if duration_s <= 0:
        return QualityCheck("unknown", 0.0, "no duration available to judge frame coverage")
    coverage = min(1.0, sampled_frames / duration_s)
    if coverage < FRAME_COVERAGE_UNREADABLE:
        return QualityCheck("unreadable", coverage,
                            f"no video frames could be read ({sampled_frames} of ~{duration_s:.0f})")
    if coverage < FRAME_COVERAGE_PARTIAL:
        return QualityCheck("partial", coverage,
                            f"only {coverage:.0%} of the video's frames could be read")
    return QualityCheck("ok", coverage, f"{coverage:.0%} of frames read")


def check_chart_talk(segments, chart_count: int, keyframe_count: int,
                     duration_s: float = 0.0) -> QualityCheck:
    from pipelines.audio_cue_capture import is_chart_talk
    talk = sum(1 for s in segments if is_chart_talk(getattr(s, "text", "")))
    minutes = max(duration_s, 60.0) / 60.0
    per_10min = talk / minutes * 10.0
    share = (chart_count / keyframe_count) if keyframe_count else 0.0
    if per_10min >= CHART_TALK_PER_10MIN and share < CHART_SHARE_FLOOR:
        return QualityCheck("charts_missed", share,
                            f"the speaker points at the chart {talk} times but only "
                            f"{chart_count} of {keyframe_count} keyframes were detected as charts")
    return QualityCheck("ok", share, f"{talk} chart-pointing lines, {chart_count}/{keyframe_count} chart keyframes")


# gemini_down.md (2026-10-22): describe_chart_frame() returns "" on any failure, so
# a Gemini outage turned every chart description into "" and the run still ended
# "done" -- check_embedding_consistency counts only NON-empty descriptions as
# expected, so it agreed with the empty result. This check is the one that notices.
# A failure on every frame of a chart-heavy video is a provider outage, not bad
# luck; a video with only one or two chart frames can lose both to a transient error.
CHART_DESCRIPTIONS_FAIL_MIN = 3


def check_chart_descriptions(chart_count: int, described: int) -> QualityCheck:
    """chart_count: chart keyframes found; described: how many got a non-empty
    vision description. "missing" = none of a chart-heavy video's frames did."""
    if chart_count <= 0:
        return QualityCheck("none", 1.0, "no chart keyframes to describe")
    share = described / chart_count
    if described == 0 and chart_count >= CHART_DESCRIPTIONS_FAIL_MIN:
        return QualityCheck("missing", 0.0,
                            f"0 of {chart_count} chart keyframes got a chart description -- the vision "
                            f"model failed on every frame (see 'vision description failed' in the logs)")
    if described < chart_count:
        return QualityCheck("partial", share,
                            f"{chart_count - described} of {chart_count} chart keyframes have no chart description")
    return QualityCheck("ok", 1.0, f"{described}/{chart_count} chart keyframes described")


def check_transcript_coverage(segments, duration_s: float = 0.0) -> QualityCheck:
    if duration_s <= 0:
        return QualityCheck("unknown", 0.0, "no duration available to judge transcript coverage")
    if not segments:
        return QualityCheck("truncated", 0.0, "no transcript at all")
    ordered = sorted(segments, key=lambda s: s.start_time)
    end = max(s.end_time for s in ordered)
    covered = min(1.0, end / duration_s)
    if covered < TRANSCRIPT_END_COVERAGE:
        return QualityCheck("truncated", covered,
                            f"the transcript stops at {covered:.0%} of the video ({end:.0f}s of {duration_s:.0f}s)")
    gap, at = 0.0, 0.0
    for a, b in zip(ordered, ordered[1:]):
        if b.start_time - a.end_time > gap:
            gap, at = b.start_time - a.end_time, a.end_time
    if gap > TRANSCRIPT_MAX_GAP_S:
        return QualityCheck("gap", covered,
                            f"no transcript for {gap / 60:.0f} min from {at / 60:.0f}:{int(at) % 60:02d}")
    return QualityCheck("ok", covered, f"transcript covers {covered:.0%} of the video")


def _self_test():
    # 1. ASR: empty segments
    qc = check_asr_quality([])
    assert qc.label == "empty", qc
    print("ok 1 — check_asr_quality flags empty segment list")

    # 2. ASR: healthy
    class Seg:
        def __init__(self, text, confidence):
            self.text, self.confidence = text, confidence

    qc = check_asr_quality([Seg("the sla applies", -0.3), Seg("load balancer", -0.2)], duration_s=30)
    assert qc.label == "ok", qc
    print("ok 2 — check_asr_quality passes healthy avg_logprob")

    # 3. ASR: below Whisper's own failure floor
    qc = check_asr_quality([Seg("garbled", -1.8)], duration_s=10)
    assert qc.label == "low_confidence", qc
    print("ok 3 — check_asr_quality flags avg_logprob below -1.0")

    # 4. ASR: sparse for video length
    qc = check_asr_quality([Seg("one word", -0.1)], duration_s=600)  # 10 min, 1 segment
    assert qc.label == "sparse", qc
    print("ok 4 — check_asr_quality flags implausibly few segments for duration")

    # 5. Visual: no duration available
    qc = check_visual_quality(40, duration_s=0)
    assert qc.label == "unknown", qc
    print("ok 5 — check_visual_quality can't judge density without duration")

    # 6. Visual: within the calibrated band (40/hr for a 1-hour video)
    qc = check_visual_quality(40, duration_s=3600)
    assert qc.label == "ok", qc
    print("ok 6 — check_visual_quality passes the calibrated 40/hr rate")

    # 7. Visual: too sparse
    qc = check_visual_quality(5, duration_s=3600)
    assert qc.label == "too_sparse", qc
    print("ok 7 — check_visual_quality flags density below the 20/hr floor")

    # 8. Visual: too dense
    qc = check_visual_quality(200, duration_s=3600)
    assert qc.label == "too_dense", qc
    print("ok 8 — check_visual_quality flags density above the 130/hr ceiling")

    # 9. Embedding: nothing eligible
    qc = check_embedding_consistency({"clip": 0, "ocr": 0, "od": 0, "asr": 0}, 0)
    assert qc.label == "empty", qc
    print("ok 9 — check_embedding_consistency handles zero eligible items")

    # 10. Embedding: complete
    qc = check_embedding_consistency({"clip": 10, "ocr": 5, "od": 5, "asr": 20}, 40)
    assert qc.label == "ok", qc
    print("ok 10 — check_embedding_consistency passes a full match")

    # 11. Embedding: silent partial drop
    qc = check_embedding_consistency({"clip": 10, "ocr": 5, "od": 5, "asr": 20}, 20)
    assert qc.label == "incomplete", qc
    print("ok 11 — check_embedding_consistency flags a silent partial drop")

    # 12. Embedding: total failure despite eligible items
    qc = check_embedding_consistency({"clip": 10, "ocr": 0, "od": 0, "asr": 0}, 0)
    assert qc.label == "failed", qc
    print("ok 12 — check_embedding_consistency flags zero output with eligible items")

    class TSeg:
        def __init__(self, text, start, end):
            self.text, self.start_time, self.end_time, self.confidence = text, start, end, -0.2

    # 13-15. Frame coverage (P137): frames actually read vs the video's length.
    # 0 keyframes "succeeded" silently when OpenCV couldn't decode AV1.
    assert check_frame_coverage(0, 600).label == "unreadable"
    assert check_frame_coverage(20, 600).label == "unreadable"          # 3%
    print("ok 13 — check_frame_coverage: ~nothing read -> unreadable")
    qc = check_frame_coverage(400, 600)
    assert qc.label == "partial" and 0.66 < qc.score < 0.67, qc
    print("ok 14 — check_frame_coverage: part of the video read -> partial")
    assert check_frame_coverage(598, 600).label == "ok"
    assert check_frame_coverage(10, 0).label == "unknown"
    print("ok 15 — check_frame_coverage: full read ok; no duration -> unknown")

    # 16-17. Chart talk vs detected charts (P141): the presenter keeps
    # pointing at the chart but almost no keyframe was classified a chart.
    talk = [TSeg("look at this bar right here", 60.0 * i, 60.0 * i + 5) for i in range(10)]
    qc = check_chart_talk(talk, chart_count=0, keyframe_count=419, duration_s=600)
    assert qc.label == "charts_missed", qc
    print("ok 16 — check_chart_talk: heavy chart talk, ~0 charts detected -> charts_missed")
    assert check_chart_talk(talk, chart_count=397, keyframe_count=418, duration_s=600).label == "ok"
    quiet = [TSeg("welcome back everyone, today is about discipline", 0.0, 5.0)]
    assert check_chart_talk(quiet, chart_count=0, keyframe_count=50, duration_s=600).label == "ok"
    print("ok 17 — check_chart_talk: charts found, or little chart talk -> ok")

    # 18-20. Transcript coverage: audio that stops early or has a long hole.
    full = [TSeg("x", 60.0 * i, 60.0 * i + 55) for i in range(10)]      # 0..595 of 600
    assert check_transcript_coverage(full, 600).label == "ok"
    print("ok 18 — check_transcript_coverage: covers the video -> ok")
    early = [TSeg("x", 60.0 * i, 60.0 * i + 55) for i in range(5)]      # ends at 295 of 600
    qc = check_transcript_coverage(early, 600)
    assert qc.label == "truncated" and qc.score < 0.5, qc
    print("ok 19 — check_transcript_coverage: ends at ~49% -> truncated")
    holed = [TSeg("x", 0.0, 50.0), TSeg("y", 900.0, 1190.0)]           # 850 s silence
    assert check_transcript_coverage(holed, 1200).label == "gap"
    assert check_transcript_coverage([], 600).label == "truncated"
    assert check_transcript_coverage(full, 0).label == "unknown"
    print("ok 20 — check_transcript_coverage: a >10 min hole -> gap; none -> truncated")

    # 21. Chart descriptions (gemini_down.md): a provider outage that blanks every
    # description is "missing", not ok; a few blanks are "partial"; no charts is n/a.
    assert check_chart_descriptions(0, 0).label == "none"
    qc = check_chart_descriptions(40, 0)
    assert qc.label == "missing" and "0 of 40" in qc.reason, qc
    assert check_chart_descriptions(2, 0).label == "partial", "2 frames lost is not an outage verdict"
    qc = check_chart_descriptions(40, 37)
    assert qc.label == "partial" and "3 of 40" in qc.reason, qc
    assert check_chart_descriptions(40, 40).label == "ok"
    print("ok 21 — check_chart_descriptions: all blank -> missing, some -> partial, none to do -> none")

    print("\n21 checks passed")


if __name__ == "__main__":
    _self_test()
