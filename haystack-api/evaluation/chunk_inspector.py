"""
Layer 1 — automated chunk quality inspection.

Runs inside every ingestion job. No LLM, no network: pure rules over the chunks
the chunker just produced, so it costs nothing and can never be skipped.

    report = inspect_chunks(chunks, source_type="transcript", source_text=text)
    report["passed"]     -> False when a hard rule was broken
    report["issues"]     -> [{chunk_index, rule, severity, detail}, ...]

Hard rules (severity "error", these fail the job):
    missing source / source_type metadata
    transcript chunk without start_time / end_time
    a domain term that survives nowhere intact
    a function body split away from its definition

Soft rules (severity "warning", counted and logged):
    chunk under 50 or over 800 tokens
    chunk starting or ending mid-sentence
"""

import logging
import re

from pipelines.chunkers.base import count_tokens, jargon_spans
from pipelines.chunkers.code_chunker import block_ranges

logger = logging.getLogger("haystack-api.evaluation")

MIN_TOKENS = 50
MAX_TOKENS = 800
REQUIRED_METADATA = ("source", "source_type")

# Anything a well-formed sentence, list item or statement may end on.
_SENTENCE_END_CHARS = '.!?…:;"\')]}»'
_STARTS_LOWER = re.compile(r"^[a-z]")
_HAS_PUNCTUATION = re.compile(r"[.!?]")
_CODE_TYPES = {"code"}
_VIDEO_FRAME_TYPES = {"visual_frame", "ocr_frame", "od_frame"}
_VIDEO_TYPES = _VIDEO_FRAME_TYPES | {"transcript_segment"}


def _chunk_parts(chunk):
    """Accept TextChunk objects or plain dicts — both show up in practice."""
    if isinstance(chunk, dict):
        return chunk.get("content", ""), (chunk.get("metadata") or {})
    return chunk.content, (chunk.metadata or {})


def check_chunk(chunk, source_type: str = None, index: int = 0,
                punctuated: bool = True) -> list:
    """Rules that only need the one chunk. Returns a list of issue dicts."""
    content, metadata = _chunk_parts(chunk)
    source_type = source_type or metadata.get("source_type") or "document"
    is_code = source_type in _CODE_TYPES
    issues = []

    def flag(rule, severity, detail):
        issues.append({"chunk_index": index, "rule": rule,
                       "severity": severity, "detail": detail})

    tokens = count_tokens(content)
    if not content.strip():
        flag("empty", "error", "chunk has no content")
        return issues
    if tokens < MIN_TOKENS:
        flag("too_small", "warning", "%d tokens (< %d), probably a fragment"
             % (tokens, MIN_TOKENS))
    if tokens > MAX_TOKENS:
        flag("too_large", "warning", "%d tokens (> %d), a split point was missed"
             % (tokens, MAX_TOKENS))

    for key in REQUIRED_METADATA:
        if not metadata.get(key):
            flag("missing_metadata", "error", "metadata['%s'] is missing" % key)

    if source_type == "transcript":
        for key in ("start_time", "end_time"):
            if metadata.get(key) is None:
                flag("missing_timestamp", "error", "transcript chunk has no %s" % key)

    if source_type == "transcript_segment":
        for key in ("start_time", "end_time", "video_id"):
            if metadata.get(key) is None:
                flag("missing_video_meta", "error",
                     "transcript_segment chunk has no %s" % key)

    if source_type in _VIDEO_FRAME_TYPES:
        if metadata.get("video_id") is None:
            flag("missing_video_meta", "error", "video frame has no video_id")
        if metadata.get("frame_ts") is None and metadata.get("frame_ms") is None:
            flag("missing_video_meta", "error", "video frame has no frame_ts/frame_ms")

    # Mid-sentence checks are meaningless on code and on unpunctuated ASR text.
    if not is_code and punctuated:
        stripped = content.strip()
        if _STARTS_LOWER.match(stripped):
            flag("starts_mid_sentence", "warning", "starts with %r" % stripped[:40])
        if stripped[-1] not in _SENTENCE_END_CHARS:
            flag("ends_mid_sentence", "warning", "ends with %r" % stripped[-40:])

    return issues


def inspect_chunks(chunks: list, source_type: str = None,
                   source_text: str = None, source: str = None) -> dict:
    """Full Layer 1 report for one ingestion job."""
    if not chunks:
        return {"chunks": 0, "passed": False, "errors": 1, "warnings": 0,
                "issues": [{"chunk_index": -1, "rule": "no_chunks", "severity": "error",
                            "detail": "chunker produced nothing"}],
                "counts": {"no_chunks": 1}, "tokens": {}, "source": source}

    punctuated = bool(source_text and _HAS_PUNCTUATION.search(source_text))
    issues = []
    for i, chunk in enumerate(chunks):
        issues.extend(check_chunk(chunk, source_type, i, punctuated))

    issues.extend(_jargon_issues(chunks, source_text))
    if (source_type or "") in _CODE_TYPES:
        issues.extend(_function_issues(chunks, source_text))
    if (source_type or "") in _VIDEO_TYPES:
        issues.extend(_video_issues(chunks, source_type))

    sizes = sorted(count_tokens(_chunk_parts(c)[0]) for c in chunks)
    counts = {}
    for issue in issues:
        counts[issue["rule"]] = counts.get(issue["rule"], 0) + 1
    errors = sum(1 for i in issues if i["severity"] == "error")

    report = {
        "source": source,
        "source_type": source_type,
        "chunks": len(chunks),
        "passed": errors == 0,
        "errors": errors,
        "warnings": len(issues) - errors,
        "counts": counts,
        "tokens": {
            "min": sizes[0],
            "max": sizes[-1],
            "mean": round(sum(sizes) / len(sizes), 1),
            "median": sizes[len(sizes) // 2],
        },
        # A long tail of identical warnings helps nobody; the counts stay complete.
        "issues": issues[:100],
    }
    log = logger.warning if errors else logger.info
    log("chunk inspection %s: %d chunks, %d errors, %d warnings %s",
        source or "?", report["chunks"], report["errors"], report["warnings"], counts)
    return report


def _jargon_issues(chunks: list, source_text: str) -> list:
    """Every protected domain term in the source must survive intact in some chunk.

    This is the boundary test: if "batch job" was cut in half, the term
    exists in the source and in no chunk, and the query for it can never hit.
    """
    if not source_text:
        return []
    in_source = {source_text[s:e].lower() for s, e in jargon_spans(source_text)}
    if not in_source:
        return []

    survived = set()
    for chunk in chunks:
        content = _chunk_parts(chunk)[0]
        survived |= {content[s:e].lower() for s, e in jargon_spans(content)}

    return [{"chunk_index": -1, "rule": "split_jargon", "severity": "error",
             "detail": "%r appears in the source but in no chunk" % term}
            for term in sorted(in_source - survived)]


def _function_issues(chunks: list, source_text: str) -> list:
    """No definition may be separated from its body by a chunk boundary.

    Blocks come from code_chunker.block_ranges so the rule and the chunker agree
    on what one function is; what this actually verifies is that packing and the
    oversize split left every whole block inside a single chunk.
    """
    if not source_text:
        return []

    lines = source_text.splitlines()
    bodies = [_normalise(_chunk_parts(c)[0]) for c in chunks]
    issues = []
    for start, end, name in block_ranges(source_text):
        block = "\n".join(lines[start:end]).strip()
        if not block or count_tokens(block) > MAX_TOKENS:
            continue        # oversized blocks are split on purpose; the size rule covers them
        normalised = _normalise(block)
        if not any(normalised in body for body in bodies):
            issues.append({"chunk_index": -1, "rule": "split_function", "severity": "error",
                           "detail": "body of %r spans a chunk boundary" % (name or "?")})
    return issues


def _video_issues(chunks: list, source_type: str) -> list:
    """Cross-chunk rules for video content: timestamp ordering, empty frames."""
    issues = []

    # Timestamp monotonicity: each chunk's time >= the previous chunk's.
    times = []
    for chunk in chunks:
        _, meta = _chunk_parts(chunk)
        t = meta.get("start_time") or meta.get("frame_ts") or meta.get("frame_ms")
        times.append(float(t) if t is not None else None)

    for i in range(1, len(times)):
        if times[i] is not None and times[i - 1] is not None:
            if times[i] < times[i - 1]:
                issues.append({
                    "chunk_index": i, "rule": "timestamp_not_monotonic",
                    "severity": "error",
                    "detail": "time %.1f < previous %.1f" % (times[i], times[i - 1]),
                })

    # OCR/OD frames with no extracted content are noise.
    if source_type in ("ocr_frame", "od_frame"):
        empty = sum(1 for c in chunks if not _chunk_parts(c)[0].strip())
        if empty > len(chunks) * 0.5:
            issues.append({
                "chunk_index": -1, "rule": "mostly_empty_frames",
                "severity": "warning",
                "detail": "%d of %d frames have no content" % (empty, len(chunks)),
            })

    return issues


def _normalise(text: str) -> str:
    """Whitespace-insensitive comparison — chunk joiners re-flow blank lines."""
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    from pipelines.chunkers import base as chunker_base
    from pipelines.chunkers.base import TextChunk

    # The protected-term list is empty by default (DOMAIN_JARGON); the
    # split_jargon rule only fires for configured terms, so configure some.
    saved_terms = chunker_base.DOMAIN_JARGON
    chunker_base.DOMAIN_JARGON = ["batch job", "service level"]

    good = TextChunk("A batch job is an asynchronous task queued for later. " * 12,
                     {"source": "a.txt", "source_type": "document"})
    assert inspect_chunks([good], "document", good.content)["passed"]

    fragment = TextChunk("job runs", {"source": "a.txt", "source_type": "document"})
    rules = {i["rule"] for i in inspect_chunks([fragment], "document", "It ends. Yes.")["issues"]}
    assert {"too_small", "starts_mid_sentence", "ends_mid_sentence"} <= rules, rules

    no_meta = TextChunk("x" * 400, {})
    assert not inspect_chunks([no_meta], "document")["passed"]

    # "batch job" cut across the boundary -> no chunk holds it intact.
    split = [TextChunk("A long batch", {"source": "a.txt", "source_type": "document"}),
             TextChunk("job runs overnight.", {"source": "a.txt", "source_type": "document"})]
    report = inspect_chunks(split, "document", "A long batch job runs overnight.")
    assert "split_jargon" in report["counts"], report

    # Video frame: missing video_id -> error
    vf = TextChunk("chart at 15:30 " * 30,
                   {"source": "v.mp4", "source_type": "visual_frame",
                    "frame_ts": 930})
    r = inspect_chunks([vf], "visual_frame")
    assert "missing_video_meta" in r["counts"], r

    # Video frame: all metadata present -> passes
    vf2 = TextChunk("chart at 15:30 " * 30,
                    {"source": "v.mp4", "source_type": "visual_frame",
                     "video_id": "abc123", "frame_ts": 930})
    r2 = inspect_chunks([vf2], "visual_frame")
    assert "missing_video_meta" not in r2.get("counts", {}), r2

    # Transcript segment: missing video_id -> error
    ts = TextChunk("he talked about service levels " * 10,
                   {"source": "v.mp4", "source_type": "transcript_segment",
                    "start_time": 0, "end_time": 30})
    r3 = inspect_chunks([ts], "transcript_segment")
    assert "missing_video_meta" in r3["counts"], r3

    # Timestamp monotonicity: out-of-order -> error
    c1 = TextChunk("frame one content " * 20,
                   {"source": "v.mp4", "source_type": "visual_frame",
                    "video_id": "x", "frame_ts": 100})
    c2 = TextChunk("frame two content " * 20,
                   {"source": "v.mp4", "source_type": "visual_frame",
                    "video_id": "x", "frame_ts": 50})
    r4 = inspect_chunks([c1, c2], "visual_frame")
    assert "timestamp_not_monotonic" in r4["counts"], r4

    chunker_base.DOMAIN_JARGON = saved_terms
    print("chunk inspector self-check ok")
