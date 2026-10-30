"""
Chunk payload -> the citation a human can act on.

    format_citation(payload)       one chunk   -> "[Quarterly Report — Page 4](url)"
    format_sources_list(payloads)  many chunks -> the numbered "Sources:" block

Reads the namespaced payload written by pipelines/ingestion.py:

    {"source_type": ..., "collection": ..., "chunk_index": ..., "origin": ...,
     "source": {"url", "title", "filename", "kind", "name",
                "page_number", "row_range", "original_filename"}}

Pre-backfill payloads keep `source` as a filename string with url/title/kind
flat beside it, so both shapes are accepted — every field is optional and a
missing one costs detail, never an exception.
"""

import logging
import re

logger = logging.getLogger("haystack-api.citation")

FALLBACK = "Knowledge Base"

# Keys that live inside the source object. Legacy payloads carry them flat.
SOURCE_KEYS = ("url", "title", "filename", "kind", "name",
               "page_number", "row_range", "original_filename")

_ROW_RANGE = re.compile(r"(\d+)\s*(?:-|–|to)?\s*(\d+)?")


def _source(meta: dict) -> dict:
    """The source object, reconstructed from whichever payload shape arrived."""
    raw = meta.get("source")
    src = dict(raw) if isinstance(raw, dict) else {}
    if isinstance(raw, str) and raw:
        src.setdefault("filename", raw)
    for key in SOURCE_KEYS:
        if src.get(key) in (None, "") and meta.get(key) not in (None, ""):
            src[key] = meta[key]
    # Transcripts historically carried the video name under its own key.
    if not src.get("title") and meta.get("video_title"):
        src["title"] = meta["video_title"]
    return src


def _kind(meta: dict, src: dict) -> str:
    """Normalised dispatch key: source_type, corrected by collection."""
    stype = str(meta.get("source_type") or "").lower()
    if stype in ("structured", "structured_data"):
        return "structured"
    if stype in ("document", "pdf", "transcript", "conversation"):
        return stype
    if stype in ("visual_frame", "ocr_frame", "od_frame", "transcript_segment"):
        return stype
    return stype or "unknown"


def _basename(path) -> str:
    if not path:
        return ""
    return str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _filename(src: dict) -> str:
    """What a person would call this file."""
    return _basename(src.get("original_filename")) or _basename(src.get("filename"))


def _link(label: str, url) -> str:
    return "[%s](%s)" % (label, url) if url else label


def _span(values: list, singular: str, plural: str) -> str:
    """[4] -> 'Page 4'; [4, 5, 7] -> 'Pages 4-7'; [] -> ''."""
    if not values:
        return ""
    low, high = min(values), max(values)
    if low == high:
        return "%s %s" % (singular, low)
    return "%s %s-%s" % (plural, low, high)


def _pages(src: dict) -> list:
    try:
        return [int(src.get("page_number"))]
    except (TypeError, ValueError):
        return []


def _rows(src: dict) -> list:
    """'1500-1600' -> [1500, 1600]. Anything unparseable is dropped."""
    match = _ROW_RANGE.search(str(src.get("row_range") or ""))
    if not match:
        return []
    return [int(g) for g in match.groups() if g is not None]


def _timestamp(value) -> str:
    """Seconds -> M:SS / H:MM:SS. Anything already formatted passes through."""
    try:
        total = int(float(value))
    except (TypeError, ValueError):
        return str(value or "")
    if total >= 3600:
        return "%d:%02d:%02d" % (total // 3600, (total % 3600) // 60, total % 60)
    return "%d:%02d" % (total // 60, total % 60)


def _transcript(src: dict, meta: dict) -> str:
    label = src.get("title") or _filename(src) or FALLBACK
    start = meta.get("start_time", meta.get("start"))
    if start is not None:
        label += " (%s–%s)" % (_timestamp(start),
                                    _timestamp(meta.get("end_time", meta.get("end"))))
    return _link(label, src.get("url"))


def _document(src: dict, pages: list) -> str:
    label = _filename(src) or src.get("title") or str(src.get("url") or "") or FALLBACK
    page = _span(pages, "Page", "Pages")
    return "%s — %s" % (label, page) if page else label


def _structured(src: dict, rows: list) -> str:
    label = _filename(src) or src.get("title") or FALLBACK
    span = _span(rows, "Row", "Rows")
    return "%s — %s" % (label, span) if span else label


def _generic(src: dict) -> str:
    """url -> title -> filename -> "Knowledge Base". Never empty."""
    if src.get("url"):
        return _link(src.get("title") or _filename(src) or str(src["url"]), src["url"])
    return src.get("title") or _filename(src) or FALLBACK


def _video_watch_url(meta: dict, video_id: str, seconds: int = None) -> str:
    """Build the right watch URL for whatever platform this video came from.

    YouTube is the only platform with a stable, re-derivable watch URL and a
    standard seek-by-URL param (&t=). Every other platform (Instagram,
    TikTok, Twitter/X — all of which M3 ingests) has no such thing, so the
    only safe link is the source_url recorded at discovery time, with no
    timestamp appended. `platform` defaults to "youtube" so points ingested
    before this field existed keep citing exactly as before (P103).
    """
    platform = (meta.get("platform") or "youtube").lower()
    if platform == "youtube":
        if not video_id:
            return ""
        url = "https://youtube.com/watch?v=%s" % video_id
        return "%s&t=%d" % (url, seconds) if seconds is not None else url
    return meta.get("source_url") or ""


def _video_frame(src: dict, meta: dict) -> str:
    """Citation for a video keyframe (visual, OCR or OD track)."""
    video_id = meta.get("video_id", "")
    title = src.get("title") or meta.get("title") or "Video"
    frame_ms = meta.get("frame_ts") or meta.get("frame_ms") or meta.get("timestamp_ms") or 0
    # Always milliseconds (video_fusion's payload contract). A "> 1000 means ms"
    # guess turned a frame at 991 ms into "@ 16:31" in a 41-second video.
    seconds = int(float(frame_ms) / 1000)
    url = _video_watch_url(meta, video_id, seconds)
    parts = [_link("%s @ %s" % (title, _timestamp(seconds)), url)]
    ocr_text = meta.get("ocr_text", "")
    if ocr_text:
        parts.append("Screen text: %s" % ocr_text[:200])
    od = meta.get("od_detections")
    if od:
        parts.append("Patterns: %s" % od)
    return " — ".join(parts)


def _video_transcript(src: dict, meta: dict) -> str:
    """Citation for an ASR transcript segment."""
    video_id = meta.get("video_id", "")
    title = src.get("title") or meta.get("title") or "Video"
    start = meta.get("start_time", meta.get("start"))
    end = meta.get("end_time", meta.get("end"))
    if start is not None:
        seconds = int(float(start))
        url = _video_watch_url(meta, video_id, seconds)
        label = "%s (%s–%s)" % (title, _timestamp(start), _timestamp(end or start))
    else:
        url = _video_watch_url(meta, video_id)
        label = title
    return _link(label, url)


def _render(meta: dict, src: dict, pages: list, rows: list) -> str:
    kind = _kind(meta, src)
    if kind == "transcript":
        return _transcript(src, meta)
    if kind in ("visual_frame", "ocr_frame", "od_frame"):
        return _video_frame(src, meta)
    if kind == "transcript_segment":
        return _video_transcript(src, meta)
    if kind in ("document", "pdf"):
        return _document(src, pages)
    if kind == "structured":
        return _structured(src, rows)
    if kind == "conversation":
        # Filtered at ingestion (architecture decision #8) — belt and braces.
        return "Chat History"
    return _generic(src)


def format_citation(chunk_metadata: dict) -> str:
    """One chunk payload -> its citation. Always returns a non-empty string."""
    try:
        meta = chunk_metadata if isinstance(chunk_metadata, dict) else {}
        src = _source(meta)
        return _render(meta, src, _pages(src), _rows(src)) or FALLBACK
    except Exception:
        logger.warning("citation formatting failed — falling back", exc_info=True)
        return FALLBACK


# A video is cited per moment, not per video (F5, sprint 2026-10-25): two passages of
# Episode 12 share its url and title, so they merged into one Sources entry showing
# only the first range -- the reply cited 1:07:55 and 1:09:31, the list showed 1:07:55.
_VIDEO_KINDS = frozenset({"transcript", "transcript_segment", "visual_frame", "ocr_frame", "od_frame"})


def _is_video(meta: dict) -> bool:
    return _kind(meta, _source(meta)) in _VIDEO_KINDS


def _moment(meta: dict) -> str:
    """Where in the video: the passage's start, else the frame's time."""
    for key in ("start_time", "start", "frame_ts", "frame_ms", "timestamp_ms"):
        if meta.get(key) not in (None, ""):
            return "%s=%s" % (key, meta[key])
    return ""


def source_key(chunk_metadata: dict) -> str:
    """What makes two chunks the same source: url, else filename, else title --
    and for a video, also the moment (one entry per time range)."""
    meta = chunk_metadata if isinstance(chunk_metadata, dict) else {}
    src = _source(meta)
    key = str(src.get("url") or src.get("filename") or src.get("title") or FALLBACK)
    if _is_video(meta) and _moment(meta):
        key += "@" + _moment(meta)
    return key


def _dedupe_sources(sources: list) -> list:
    """Collapse duplicate entries for the same page.

    Two entries are the same source when their URLs match exactly or when they
    carry the same title. The survivor keeps its position; page and row numbers
    of the duplicates are merged into it.
    """
    kept, by_url, seen_titles = [], {}, set()

    for entry in sources:
        if _is_video(entry["meta"]):
            # Each moment of a video is its own entry; source_key already merged
            # repeats of the same moment. The title/url collapse below is for
            # duplicated document pages and would fold every range into one.
            kept.append(entry)
            continue
        url = str(entry["src"].get("url") or "")
        title = str(entry["src"].get("title") or "").strip()

        if title and title in seen_titles:
            continue

        if url:
            index = by_url.get(url)
            if index is not None:
                previous = kept[index]
                previous["pages"] += entry["pages"]
                previous["rows"] += entry["rows"]
                continue
            by_url[url] = len(kept)

        if title:
            seen_titles.add(title)
        kept.append(entry)

    return kept


def format_sources_list(chunks: list) -> str:
    """Retrieved chunks -> the numbered "Sources:" block, one entry per source.

    Chunks from the same document collapse into one line with their page (or
    row) numbers merged into a range, and duplicate entries for one page are
    collapsed again by _dedupe_sources. Returns "" when there is nothing to
    cite, so the caller can skip the section entirely.
    """
    merged = {}
    for chunk in chunks or []:
        meta = chunk if isinstance(chunk, dict) else {}
        src = _source(meta)
        entry = merged.setdefault(source_key(meta),
                                  {"meta": meta, "src": src, "pages": [], "rows": []})
        entry["pages"] += _pages(src)
        entry["rows"] += _rows(src)

    lines = ["[%d] %s" % (n, _render(e["meta"], e["src"], e["pages"], e["rows"]) or FALLBACK)
             for n, e in enumerate(_dedupe_sources(list(merged.values())), 1)]
    return "Sources:\n" + "\n".join(lines) if lines else ""


def demo():
    """Offline self-check: python services/citation_formatter.py"""
    web = {"source_type": "code", "collection": "code_docs",
           "source": {"url": "https://docs.example.com/api#run_job",
                      "title": "API Reference / run_job",
                      "filename": "api/run_job.md", "kind": "function",
                      "name": "run_job"}}
    assert format_citation(web) == \
        "[API Reference / run_job](https://docs.example.com/api#run_job)", format_citation(web)

    page = {"source_type": "code_docs", "collection": "code_docs",
            "source": {"url": "https://docs.example.com/guide/setup",
                       "title": "Guide / Setup"}}
    assert format_citation(page) == \
        "[Guide / Setup](https://docs.example.com/guide/setup)"

    pdf = {"source_type": "document",
           "source": {"filename": "uploads/report.pdf", "page_number": 4}}
    assert format_citation(pdf) == "report.pdf — Page 4"
    assert format_citation({"source_type": "document",
                            "source": {"filename": "report.pdf"}}) == "report.pdf"

    csv = {"source_type": "structured",
           "source": {"filename": "uuid_sales_data.csv", "original_filename": "sales_data.csv",
                      "row_range": "1500-1600"}}
    assert format_citation(csv) == "sales_data.csv — Rows 1500-1600"

    video = {"source_type": "transcript",
             "source": {"title": "Training Session 47: Worked Example"},
             "start_time": 872, "end_time": 1125}
    assert format_citation(video) == "Training Session 47: Worked Example (14:32–18:45)"

    # legacy flat payload, missing everything, and junk all land somewhere sane
    assert format_citation({"source_type": "code", "source": "api/run_job.md",
                            "url": "https://docs.example.com/api#run_job", "name": "run_job",
                            "kind": "function"}) \
        == "[run_job.md](https://docs.example.com/api#run_job)"
    assert format_citation({}) == FALLBACK
    assert format_citation(None) == FALLBACK
    assert format_citation({"source_type": "conversation"}) == "Chat History"
    assert format_citation({"source_type": "media",
                            "source": {"filename": "chart.png"}}) == "chart.png"

    # ── Video frame citations (M3) ──
    vf = {"source_type": "visual_frame", "video_id": "dQw4w9WgXcQ",
          "title": "Training Session 47", "frame_ts": 125000}
    vf_cite = format_citation(vf)
    assert "Training Session 47 @ 2:05" in vf_cite, vf_cite
    assert "dQw4w9WgXcQ" in vf_cite, vf_cite
    assert "t=125" in vf_cite, vf_cite

    # OCR frame with on-screen text
    ocrf = {"source_type": "ocr_frame", "video_id": "abc123",
            "title": "Worked Example", "frame_ts": 3661000, "ocr_text": "Level 1.2345"}
    ocrf_cite = format_citation(ocrf)
    assert "1:01:01" in ocrf_cite, ocrf_cite
    assert "Screen text: Level 1.2345" in ocrf_cite, ocrf_cite

    # OD frame with pattern detection
    odf = {"source_type": "od_frame", "video_id": "xyz",
           "title": "Patterns", "frame_ts": 5000, "od_detections": "rectangle"}
    odf_cite = format_citation(odf)
    assert "Patterns: rectangle" in odf_cite, odf_cite

    # A frame under one second is still milliseconds (991 ms was cited "@ 16:31")
    early = format_citation({"source_type": "od_frame", "video_id": "rClFwdAr9DM",
                             "title": "Short", "frame_ms": "991"})
    assert "Short @ 0:00" in early and "t=0" in early, early

    # Transcript segment
    ts = {"source_type": "transcript_segment", "video_id": "seg1",
          "title": "Onboarding Walkthrough", "start_time": 872, "end_time": 1125}
    ts_cite = format_citation(ts)
    assert "Onboarding Walkthrough (14:32–18:45)" in ts_cite, ts_cite
    assert "seg1" in ts_cite, ts_cite

    # Transcript segment with no timestamp
    ts_no = {"source_type": "transcript_segment", "video_id": "seg2",
             "title": "Deep Dive"}
    ts_no_cite = format_citation(ts_no)
    assert "Deep Dive" in ts_no_cite, ts_no_cite

    # Video frame with no video_id falls back gracefully
    vf_empty = {"source_type": "visual_frame", "frame_ts": 1000}
    assert format_citation(vf_empty) != FALLBACK  # still renders something

    listed = format_sources_list([
        web, page,
        {"source_type": "document", "source": {"filename": "report.pdf", "page_number": 4}},
        {"source_type": "document", "source": {"filename": "report.pdf", "page_number": 7}},
    ])
    assert listed.startswith("Sources:\n[1] [API Reference / run_job"), listed
    assert "[3] report.pdf — Pages 4-7" in listed, listed
    assert "[4]" not in listed, "chunks from one file must merge into one entry"
    assert format_sources_list([]) == ""
    print("ok")


if __name__ == "__main__":
    demo()
