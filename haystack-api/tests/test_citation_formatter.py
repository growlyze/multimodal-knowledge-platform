"""
Smallest thing that fails if citations stop being useful.
Run: python tests/test_citation_formatter.py

Same style as the other suites here — no framework, no network. Every field in a
payload is optional, so most of this file is about what happens when they are
missing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from services.citation_formatter import (
    FALLBACK,
    _dedupe_sources,
    format_citation,
    format_sources_list,
    source_key,
)

REFERENCE = {
    "source_type": "code_reference",
    "collection": "code_docs",
    "chunk_index": 0,
    "origin": "scraper",
    "source": {
        "url": "https://docs.example.com/sdk-reference/#run_query",
        "title": "run_query — SDK Reference",
        "filename": "reference/run_query.md",
        "kind": "function",
        "name": "run_query",
        "version": "v2",
        "page_number": None,
        "row_range": None,
        "original_filename": None,
    },
}

GUIDE = {
    "source_type": "code_docs",
    "collection": "code_docs",
    "source": {"url": "https://docs.example.com/guide/type_system",
               "title": "Language / Type system",
               "filename": "guide/language_type_system.md"},
}

PDF = {"source_type": "document", "collection": "documents",
       "source": {"filename": "uploads/report.pdf", "page_number": 4}}

CSV = {"source_type": "structured_data", "collection": "documents",
       "source": {"filename": "b1e4b0ce-0000-4000-8000-2d94ba676c5b_sales_data.csv",
                  "original_filename": "sales_data.csv", "row_range": "1500-1600"}}

VIDEO = {"source_type": "transcript", "collection": "documents",
         "start_time": 872, "end_time": 1125,
         "source": {"title": "Training Session 47: Worked Example",
                    "filename": "session47.mp4"}}

# The same page crawled twice: one URL per run, two chunks, one page. The
# exact URL is what dedupes them (title matching covers renames).
FAQ_ONE = {
    "source_type": "code_docs", "collection": "code_docs",
    "source": {"url": "https://docs.example.com/faq/other-data-and-timeframes/",
               "title": "Other Data and Timeframes"},
}

FAQ_AGAIN = {
    "source_type": "code_docs", "collection": "code_docs",
    "source": {"url": "https://docs.example.com/faq/other-data-and-timeframes/",
               "title": "Other Data and Timeframes"},
}

FAQ_CITATION = ("[Other Data and Timeframes](https://docs.example.com"
                "/faq/other-data-and-timeframes/)")


# ── 1. one citation per source type ─────────────────────

def test_code_reference_links_the_page():
    assert format_citation(REFERENCE) == (
        "[run_query — SDK Reference]"
        "(https://docs.example.com/sdk-reference/#run_query)")

    # A different symbol keeps its own name.
    variable = {"source_type": "code_reference",
                "source": {"url": "https://docs.example.com/#var_close", "title": "close",
                           "kind": "variable", "version": "v2"}}
    assert format_citation(variable) == \
        "[close](https://docs.example.com/#var_close)"


def test_guide_uses_the_page_name():
    assert format_citation(GUIDE) == (
        "[Language / Type system]"
        "(https://docs.example.com/guide/type_system)")

    assert format_citation({"source_type": "code_guide",
                            "source": {"url": "https://docs.example.com/variables",
                                       "title": "Variables"}}) == (
        "[Variables](https://docs.example.com/variables)")


def test_document_names_the_file_and_page():
    assert format_citation(PDF) == "report.pdf — Page 4"
    assert format_citation({"source_type": "document",
                            "source": {"filename": "report.pdf"}}) == "report.pdf"


def test_structured_names_the_rows():
    assert format_citation(CSV) == "sales_data.csv — Rows 1500-1600"
    # No row range known: the file name alone is still a citation.
    assert format_citation({"source_type": "structured_data",
                            "source": {"filename": "sales.csv"}}) == "sales.csv"


def test_transcript_carries_the_timestamp():
    assert format_citation(VIDEO) == "Training Session 47: Worked Example (14:32–18:45)"
    # Over an hour in, and with no timestamps at all.
    assert format_citation({"source_type": "transcript", "start_time": 3872, "end_time": 3905,
                            "source": {"title": "Session 12"}}) == "Session 12 (1:04:32–1:05:05)"
    assert format_citation({"source_type": "transcript",
                            "source": {"title": "Session 12"}}) == "Session 12"


def test_video_citation_uses_source_platform_not_hardcoded_youtube():
    # YouTube still gets a deep link with a timestamp -- unchanged behavior.
    yt = {"source_type": "transcript_segment", "video_id": "abc123", "platform": "youtube",
          "source": {"title": "Onboarding Walkthrough"}, "start_time": 65, "end_time": 70}
    assert format_citation(yt) == \
        "[Onboarding Walkthrough (1:05–1:10)](https://youtube.com/watch?v=abc123&t=65)"

    # A non-YouTube platform (the video pipeline ingests Instagram/TikTok/Twitter
    # too) must link to its own source_url, never a youtube.com URL that never
    # existed. No timestamp param -- these platforms don't support seek-by-URL.
    ig = {"source_type": "transcript_segment", "video_id": "DTn42_UDNaY", "platform": "instagram",
          "source_url": "https://www.instagram.com/reel/DTn42_UDNaY/",
          "source": {"title": "Video by example_channel"}, "start_time": 65, "end_time": 70}
    assert format_citation(ig) == (
        "[Video by example_channel (1:05–1:10)](https://www.instagram.com/reel/DTn42_UDNaY/)")

    # Points ingested before this fix carry no platform field at all --
    # default to youtube so existing citations don't regress.
    legacy = {"source_type": "transcript_segment", "video_id": "legacy1",
              "source": {"title": "Old Video"}, "start_time": 10, "end_time": 20}
    assert format_citation(legacy) == "[Old Video (0:10–0:20)](https://youtube.com/watch?v=legacy1&t=10)"

    # Non-YouTube with no source_url recorded (shouldn't happen post-fix, but
    # must degrade to a label instead of a broken link).
    broken = {"source_type": "transcript_segment", "video_id": "tiktok1", "platform": "tiktok",
              "source": {"title": "TikTok Clip"}, "start_time": 5, "end_time": 8}
    assert format_citation(broken) == "TikTok Clip (0:05–0:08)"

    # Same platform rule applies to keyframe citations (_video_frame).
    ig_frame = {"source_type": "visual_frame", "video_id": "DTn42_UDNaY", "platform": "instagram",
                "source_url": "https://www.instagram.com/reel/DTn42_UDNaY/",
                "source": {"title": "Video by example_channel"}, "frame_ms": 65000}
    assert format_citation(ig_frame) == (
        "[Video by example_channel @ 1:05](https://www.instagram.com/reel/DTn42_UDNaY/)")


def test_conversation_and_unknown_types():
    # Conversations are refused at ingestion; this is the belt-and-braces path.
    assert format_citation({"source_type": "conversation",
                            "source": {"filename": "conversation_4cc_2.txt"}}) == "Chat History"
    assert format_citation({"source_type": "media",
                            "source": {"filename": "chart.png"}}) == "chart.png"
    assert format_citation({"source_type": "whatever"}) == FALLBACK


# ── 2. nothing returns an empty string ──────────────────

def test_missing_fields_never_break_the_citation():
    for payload in ({}, None, "not a dict", [], {"source": None}, {"source": {}},
                    {"source_type": "document", "source": {}},
                    {"source_type": "code_reference", "source": {}},
                    {"source_type": "structured_data", "source": {"row_range": "junk"}},
                    {"source_type": "document", "source": {"page_number": "not a page"}}):
        assert format_citation(payload) == FALLBACK, payload

    # url -> title -> filename, in that order.
    assert format_citation({"source": {"url": "https://x/y", "title": "T", "filename": "f.md"}}) \
        == "[T](https://x/y)"
    assert format_citation({"source": {"title": "T", "filename": "f.md"}}) == "T"
    assert format_citation({"source": {"filename": "dir/f.md"}}) == "f.md"


def test_legacy_flat_payloads_still_cite():
    """Pre-backfill chunks: `source` is the filename, the rest is flat beside it."""
    legacy = {"source": "reference/run_query.md", "source_type": "code_reference",
              "url": "https://docs.example.com/#run_query",
              "title": "run_query — SDK Reference",
              "kind": "function", "name": "run_query", "version": "v2"}
    assert format_citation(legacy) == \
        "[run_query — SDK Reference](https://docs.example.com/#run_query)"

    # A chunk with nothing but a filename is still not empty.
    assert format_citation({"source": "report.pdf", "source_type": "document"}) == "report.pdf"
    # Flat title + url on a code page still links.
    assert format_citation({"source": "guide/alerts.md", "collection": "code_docs",
                            "title": "Concepts / Alerts",
                            "url": "https://docs.example.com/alerts"}) == \
        "[Concepts / Alerts](https://docs.example.com/alerts)"
    # Transcripts used to carry the video name under its own key.
    assert format_citation({"source": "s47.mp4", "source_type": "transcript",
                            "video_title": "Training Session 47", "start_time": 872,
                            "end_time": 905}) == "Training Session 47 (14:32–15:05)"


# ── 3. the Sources block ────────────────────────────────

def test_sources_list_numbers_and_deduplicates():
    listed = format_sources_list([REFERENCE, GUIDE, REFERENCE, PDF])
    lines = listed.splitlines()
    assert lines[0] == "Sources:"
    assert len(lines) == 4, listed              # the repeat collapsed
    assert lines[1].startswith("[1] [run_query — SDK Reference](https://")
    assert lines[2].startswith("[2] [Language / Type system](https://")
    assert lines[3] == "[3] report.pdf — Page 4"


def test_sources_list_merges_page_and_row_ranges():
    pages = [{"source_type": "document", "source": {"filename": "report.pdf",
                                                    "page_number": n}} for n in (7, 4, 5)]
    assert "[1] report.pdf — Pages 4-7" in format_sources_list(pages)

    rows = [{"source_type": "structured_data",
             "source": {"filename": "sales.csv", "row_range": r}}
            for r in ("1-100", "1500-1600")]
    assert "[1] sales.csv — Rows 1-1600" in format_sources_list(rows)


def test_sources_list_is_empty_when_there_is_nothing_to_cite():
    assert format_sources_list([]) == ""
    assert format_sources_list(None) == ""
    # Unciteable chunks still produce a line rather than a blank entry.
    assert format_sources_list([{}, {}]) == "Sources:\n[1] %s" % FALLBACK


def test_sources_list_collapses_duplicate_urls():
    """The same page crawled twice is one source, cited once, in either order."""
    for order in ([FAQ_ONE, FAQ_AGAIN], [FAQ_AGAIN, FAQ_ONE]):
        listed = format_sources_list(order)
        assert listed == "Sources:\n[1] %s" % FAQ_CITATION, listed


def test_sources_list_keeps_the_first_source_with_a_title():
    """Two paths, one page name — the second is noise even when the URLs differ."""
    listed = format_sources_list([
        {"source_type": "code_docs",
         "source": {"url": "https://docs.example.com/guide/concepts/alerts/",
                    "title": "Concepts / Alerts"}},
        {"source_type": "code_docs",
         "source": {"url": "https://docs.example.com/guide/alerts-page/",
                    "title": "Concepts / Alerts"}},
    ])
    assert listed == ("Sources:\n[1] [Concepts / Alerts]"
                      "(https://docs.example.com/guide/concepts/alerts/)"), listed


def test_sources_list_keeps_sources_that_are_genuinely_different():
    listed = format_sources_list([REFERENCE, GUIDE, FAQ_ONE, PDF, CSV])
    assert len(listed.splitlines()) == 6, listed


def test_sources_list_keeps_one_entry_per_video_time_range():
    """F5 (sprint 2026-10-25): two passages of one video shared its url/title, so
    format_sources_list merged them into ONE entry showing only the first range --
    the reply cited 1:07:55 and 1:09:31 of Episode 12, the Sources list showed only
    1:07:55. Video kinds get one entry per time range; the same range twice is
    still one entry, and text kinds group exactly as before."""
    def seg(start, end, vid="8GkQfdAXZP0"):
        return {"source_type": "transcript_segment", "video_id": vid, "platform": "youtube",
                "start_time": start, "end_time": end,
                "source": {"title": "Training Series Episode 12",
                           "url": "https://youtube.com/watch?v=%s" % vid}}
    listed = format_sources_list([seg(4075, 4127), seg(4171, 4230), seg(4075, 4127)])
    assert listed == ("Sources:\n"
                      "[1] [Training Series Episode 12 (1:07:55–1:08:47)](https://youtube.com/watch?v=8GkQfdAXZP0&t=4075)\n"
                      "[2] [Training Series Episode 12 (1:09:31–1:10:30)](https://youtube.com/watch?v=8GkQfdAXZP0&t=4171)"), listed
    frame = {"source_type": "od_frame", "video_id": "8GkQfdAXZP0", "frame_ms": 4134000,
             "source": {"title": "Training Series Episode 12"}}
    frame2 = {**frame, "frame_ms": 4152000}
    assert format_sources_list([frame, frame2, frame]).count("\n[") == 2, format_sources_list([frame, frame2])
    assert "@ 1:08:54" in format_sources_list([frame, frame2]) and "@ 1:09:12" in format_sources_list([frame, frame2])
    # Text kinds: unchanged grouping (one PDF across pages, one reference page).
    pdf5 = {**PDF, "source": {**PDF["source"], "page_number": 5}}
    assert format_sources_list([PDF, pdf5, REFERENCE, REFERENCE]).count("\n[") == 2


def test_dedupe_of_nothing_is_nothing():
    assert _dedupe_sources([]) == []


def test_source_key_groups_by_url_then_filename():
    assert source_key(REFERENCE) == REFERENCE["source"]["url"]
    assert source_key({"source": {"filename": "a.pdf"}}) == "a.pdf"
    assert source_key({"source": "a.pdf"}) == "a.pdf"
    assert source_key({}) == FALLBACK


# ── 4. what /search hands the filter ────────────────────

def test_search_result_carries_citation_and_source_object():
    from routers.search import _to_result

    result = _to_result({"id": "p1", "score": 0.91, "collection": "code_docs",
                         "payload": {"content": "run_query fetches rows", **REFERENCE}})
    assert result.citation.startswith("[run_query — SDK Reference](https://")
    assert result.source == "reference/run_query.md"
    assert result.metadata["source"]["url"].endswith("#run_query")
    assert result.metadata["collection"] == "code_docs"
    assert "content" not in result.metadata

    # Legacy points keep working: `source` stays a plain string.
    legacy = _to_result({"id": "p2", "score": 0.5, "collection": "documents",
                         "payload": {"content": "x", "source": "report.pdf",
                                     "source_type": "document"}})
    assert legacy.source == "report.pdf" and legacy.citation == "report.pdf"


def test_search_result_finds_text_for_every_video_track():
    # P105: M3's video payloads never carry "content" -- only document chunks
    # do. _to_result() used to read nothing but "content", so every video hit
    # reached rag_filter.py with an empty passage body: the model got a
    # citation LINK and no actual substance to ground an answer in, no matter
    # how relevant the hit. Found live 2026-10-14: a real transcript_segment
    # hit scored 0.95 and cited correctly, but the model reported "I don't
    # actually have the transcript... only its title/link was provided."
    from routers.search import _to_result

    transcript = _to_result({"id": "v1", "score": 0.95, "collection": "transcript_segments",
                             "payload": {"source_type": "transcript_segment", "video_id": "abc",
                                         "text": "Because of the lack of confidence..."}})
    assert transcript.content == "Because of the lack of confidence...", transcript.content

    ocr = _to_result({"id": "v2", "score": 0.8, "collection": "ocr_frames",
                      "payload": {"source_type": "ocr_frame", "video_id": "abc",
                                  "ocr_text": "LEVEL 5021.25"}})
    assert ocr.content == "LEVEL 5021.25", ocr.content

    od = _to_result({"id": "v3", "score": 0.7, "collection": "od_frames",
                     "payload": {"source_type": "od_frame", "video_id": "abc",
                                 "od_detections": "highlighted box forming"}})
    assert od.content == "highlighted box forming", od.content

    # A pure visual_frame hit (CLIP image-similarity, no text at all) is a
    # genuinely empty passage -- not a bug, nothing to fall back to.
    visual = _to_result({"id": "v4", "score": 0.6, "collection": "visual_frames",
                         "payload": {"source_type": "visual_frame", "video_id": "abc"}})
    assert visual.content == "", visual.content

    # A document chunk's "content" still wins outright -- no change for M1/M2.
    doc = _to_result({"id": "d1", "score": 0.9, "collection": "documents",
                      "payload": {"source_type": "document", "content": "the real chunk text"}})
    assert doc.content == "the real chunk text", doc.content


# ── runner ──────────────────────────────────────────────

def main():
    tests = [
        test_code_reference_links_the_page,
        test_guide_uses_the_page_name,
        test_document_names_the_file_and_page,
        test_structured_names_the_rows,
        test_transcript_carries_the_timestamp,
        test_video_citation_uses_source_platform_not_hardcoded_youtube,
        test_conversation_and_unknown_types,
        test_missing_fields_never_break_the_citation,
        test_legacy_flat_payloads_still_cite,
        test_sources_list_numbers_and_deduplicates,
        test_sources_list_merges_page_and_row_ranges,
        test_sources_list_is_empty_when_there_is_nothing_to_cite,
        test_sources_list_collapses_duplicate_urls,
        test_sources_list_keeps_the_first_source_with_a_title,
        test_sources_list_keeps_sources_that_are_genuinely_different,
        test_dedupe_of_nothing_is_nothing,
        test_source_key_groups_by_url_then_filename,
        test_sources_list_keeps_one_entry_per_video_time_range,
        test_search_result_carries_citation_and_source_object,
        test_search_result_finds_text_for_every_video_track,
    ]
    for test in tests:
        test()
        print("ok  %s" % test.__name__)
    print("\n%d citation checks passed" % len(tests))


if __name__ == "__main__":
    main()
