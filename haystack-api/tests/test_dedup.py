"""
Duplicate ingestion is refused before it costs anything.
Run: python tests/test_dedup.py

Why this exists: the rag filter re-POSTs every attachment on every chat turn,
and users re-upload the same file under a new name ("x.csv", "x (2).csv",
"x_test.csv" — all live on 2026-10-26). Before pipelines/dedup.py each of those
paid a fresh Voyage embed, and a renamed copy became a second document that
retrieval returned twice. Same stdlib style and the same fakes as
test_ingestion.py; nothing here touches the network.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from pipelines import dedup, ingestion
from pipelines.chunkers.base import TextChunk
from services import embedding_service

from test_ingestion import DOCUMENT, SCRAPED_PAGE, use_fake_backends
from test_retrieval import reset_singletons

COLLECTION = "documents"


class CountingEmbedder:
    """Wraps the stub so a test can prove no paid embed call happened."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0
        self.dimensions = inner.dimensions

    async def embed(self, texts, input_type="document"):
        self.calls += 1
        # A real Voyage call suspends for hundreds of ms. Without this yield
        # the fakes never suspend, gather() runs uploads one after another,
        # and the concurrency check passed with the lock deleted (mutation-
        # tested 2026-10-26) — proving nothing.
        await asyncio.sleep(0.01)
        return await self.inner.embed(texts, input_type=input_type)


def fakes():
    fake = use_fake_backends()
    counter = CountingEmbedder(embedding_service._provider)
    embedding_service._provider = counter
    return fake, counter


def sources(fake) -> set:
    return {p["payload"]["source"]["filename"]
            for p in fake.points.get(COLLECTION, {}).values()}


# ── 1. the hash ignores layout, never meaning ───────────

def test_hash_ignores_whitespace_line_endings_and_unicode_form():
    base = dedup.text_hash("Batch jobs\nrun overnight.")
    assert dedup.text_hash("\n\nBatch  jobs\r\n\r\nrun overnight.  \n\n") == base
    # Leading indentation is NOT layout (see the code checks below): when in
    # doubt the hash says "different" — a wasted embed beats a lost edit.
    assert dedup.text_hash("  Batch jobs\nrun overnight.") != base
    # NFKC: a full-width letter and a ligature normalise to the plain form.
    assert dedup.text_hash("Ｂatch jobs\nrun overnight.") == base
    assert dedup.text_hash("ﬁll") == dedup.text_hash("fill")
    # Case is meaning in code (`Total` is not `total`), so it is NOT folded.
    assert dedup.text_hash("batch jobs\nrun overnight.") != base
    assert dedup.bytes_hash(b"abc") != dedup.bytes_hash(b"abd")
    # Indentation is syntax in code: re-indenting a block is a real edit, and
    # calling it "unchanged" would keep the broken version indexed.
    snippet = "if total > limit:\n    alert(total, limit)\nprint(total)"
    assert dedup.text_hash(snippet) != dedup.text_hash(snippet.replace("    alert", "alert"))
    assert dedup.text_hash(snippet) != dedup.text_hash(snippet.replace("    alert", "  alert"))
    assert dedup.text_hash(snippet) == dedup.text_hash(snippet.replace("limit)", "limit)   "))
    assert base.startswith("sha256:")


# ── 2. same file again: no embed, nothing rewritten ─────

async def test_unchanged_reingest_is_a_free_no_op():
    fake, counter = fakes()
    first = await ingestion.ingest_text(DOCUMENT, "doc.txt", "document", COLLECTION)
    calls, points = counter.calls, dict(fake.points[COLLECTION])

    second = await ingestion.ingest_text(DOCUMENT, "doc.txt", "document", COLLECTION)
    assert second["skipped"] == "unchanged", second
    assert second["chunks_created"] == 0 and second["existing_chunks"] == first["chunks_created"]
    assert counter.calls == calls, "an unchanged file must not be embedded again"
    assert fake.points[COLLECTION] == points
    reset_singletons()


# ── 3. same content, new name: refused, points at the original

async def test_renamed_copy_is_refused_and_names_the_original():
    fake, counter = fakes()
    await ingestion.ingest_text(DOCUMENT, "doc.txt", "document", COLLECTION)
    calls, count = counter.calls, len(fake.points[COLLECTION])

    copy = await ingestion.ingest_text(DOCUMENT + "\n\n", "doc (2).txt", "document", COLLECTION)
    assert copy["skipped"] == "duplicate" and copy["duplicate_of"] == "doc.txt", copy
    assert copy["chunks_created"] == 0
    assert counter.calls == calls and len(fake.points[COLLECTION]) == count
    assert sources(fake) == {"doc.txt"}
    reset_singletons()


# ── 4. same name, new content: still replaced ───────────

async def test_changed_content_under_the_same_name_replaces():
    fake, _ = fakes()
    first = await ingestion.ingest_text(DOCUMENT, "doc.txt", "document", COLLECTION)
    changed = await ingestion.ingest_text(DOCUMENT + "\n\nA new closing paragraph.",
                                          "doc.txt", "document", COLLECTION)
    assert not changed.get("skipped"), changed
    assert changed["replaced"] == first["chunks_created"]
    assert sources(fake) == {"doc.txt"}
    hashes = {p["payload"]["doc_sha256"] for p in fake.points[COLLECTION].values()}
    assert hashes == {dedup.text_hash(DOCUMENT + "\n\nA new closing paragraph.")}, hashes
    reset_singletons()


# ── 5. escape hatches: force, and the kill switch ───────

async def test_force_and_kill_switch_bypass_the_check():
    fake, counter = fakes()
    await ingestion.ingest_text(DOCUMENT, "doc.txt", "document", COLLECTION)
    calls = counter.calls
    # A chunker change needs a deliberate re-index of unchanged files.
    forced = await ingestion.ingest_text(DOCUMENT, "doc.txt", "document", COLLECTION,
                                         {"force_reingest": True})
    assert not forced.get("skipped") and counter.calls == calls + 1, forced
    assert all("force_reingest" not in p["payload"] for p in fake.points[COLLECTION].values())

    original = ingestion.INGEST_DEDUP
    ingestion.INGEST_DEDUP = False
    try:
        copy = await ingestion.ingest_text(DOCUMENT, "copy.txt", "document", COLLECTION)
    finally:
        ingestion.INGEST_DEDUP = original
    assert not copy.get("skipped") and sources(fake) == {"doc.txt", "copy.txt"}
    reset_singletons()


# ── 6. repeated chunks inside one document ──────────────

def test_repeated_chunks_inside_one_document_are_dropped():
    meta = {"source": "a.txt", "source_type": "document"}
    chunks = [TextChunk("Risk 1% per trade.", {**meta, "chunk_index": 0}),
              TextChunk("Disclaimer: not advice.", {**meta, "chunk_index": 1}),
              TextChunk("Risk 1%  per trade.\n", {**meta, "chunk_index": 2}),
              TextChunk("Disclaimer: not advice.", {**meta, "chunk_index": 3})]
    kept, dropped = dedup.drop_repeated_chunks(chunks)
    assert [c.metadata["chunk_index"] for c in kept] == [0, 1] and dropped == 2
    assert all(c.metadata["chunk_sha256"] == dedup.text_hash(c.content) for c in kept)


# ── 7. scraped twice: front matter differs, body is the same

async def test_scraped_page_differing_only_in_front_matter_is_a_duplicate():
    fake, _ = fakes()
    await ingestion.ingest_text(SCRAPED_PAGE, "reference/run_query.md", "code_docs", COLLECTION)
    rescrape = SCRAPED_PAGE.replace("2026-08-27T22:47:51", "2026-10-26T09:00:00")
    again = await ingestion.ingest_text(rescrape, "reference/ta_sma.md", "code_docs", COLLECTION)
    assert again["skipped"] == "duplicate", again
    assert again["duplicate_of"] == "reference/run_query.md"
    reset_singletons()


async def test_rescrape_with_new_citation_metadata_refreshes_it():
    fake, _ = fakes()
    name = "reference/run_query.md"
    await ingestion.ingest_text(SCRAPED_PAGE, name, "code_docs", COLLECTION)
    # Only the scrape time moved: nothing a citation shows changed.
    stamp = SCRAPED_PAGE.replace("2026-08-27T22:47:51", "2026-10-26T09:00:00")
    same = await ingestion.ingest_text(stamp, name, "code_docs", COLLECTION)
    assert same["skipped"] == "unchanged", same
    # The title did change: the body hash alone would call this "unchanged"
    # and the old title would be cited forever (review finding, 2026-10-26).
    retitled = SCRAPED_PAGE.replace("title: run_query", "title: run_query()")
    fresh = await ingestion.ingest_text(retitled, name, "code_docs", COLLECTION)
    assert not fresh.get("skipped"), fresh
    titles = {p["payload"]["source"]["title"] for p in fake.points[COLLECTION].values()}
    assert titles == {"run_query() — SDK Reference"}, titles
    reset_singletons()


# ── 8. file level: identical bytes never reach Tika ─────

async def test_identical_bytes_skip_extraction_entirely():
    fake, counter = fakes()
    store = ingestion.get_store()
    data = DOCUMENT.encode()
    job = await ingestion.create_job("notes.txt", "document", COLLECTION)
    await ingestion.run_job(job["job_id"], "notes.txt", data, "document", COLLECTION, None)
    payload = next(iter(fake.points[COLLECTION].values()))["payload"]
    assert payload["file_sha256"] == dedup.bytes_hash(data), payload

    original = ingestion.extract

    async def must_not_extract(*args, **kwargs):
        raise AssertionError("a byte-identical upload must not be extracted again")

    ingestion.extract = must_not_extract
    try:
        for name, expected in (("notes.txt", "unchanged"), ("notes copy.txt", "duplicate")):
            again = await ingestion.create_job(name, "document", COLLECTION)
            result = await ingestion.run_job(again["job_id"], name, data, "document",
                                             COLLECTION, None)
            assert result["skipped"] == expected, result
            row = await store.get(again["job_id"])
            # A skip is a success the job list can explain, not a failure.
            assert row["status"] == "done" and row["chunks_created"] == 0, row
            assert row["quality"]["skipped"] == expected, row
            if expected == "duplicate":
                assert row["quality"]["duplicate_of"] == "notes.txt", row
    finally:
        ingestion.extract = original
    assert sources(fake) == {"notes.txt"}
    reset_singletons()


# ── 9. a renamed file whose new content copies another file

async def test_duplicate_clears_the_stale_chunks_of_its_own_name():
    fake, _ = fakes()
    await ingestion.ingest_text(DOCUMENT, "a.txt", "document", COLLECTION)
    await ingestion.ingest_text("Old notes about the London session open. " * 40,
                                "b.txt", "document", COLLECTION)
    # b.txt is re-uploaded and now holds exactly a.txt's text: b's old chunks
    # describe a file that no longer exists, so they must not survive the skip.
    result = await ingestion.ingest_text(DOCUMENT, "b.txt", "document", COLLECTION)
    assert result["skipped"] == "duplicate" and result["replaced"] > 0, result
    assert sources(fake) == {"a.txt"}
    reset_singletons()


# ── 10. two uploads of one file at the same moment ──────

async def test_concurrent_identical_uploads_store_one_copy():
    fake, _ = fakes()
    results = await asyncio.gather(*(
        ingestion.ingest_text(DOCUMENT, "race-%d.txt" % i, "document", COLLECTION)
        for i in range(4)))
    stored = [r for r in results if not r.get("skipped")]
    assert len(stored) == 1, results
    assert len(sources(fake)) == 1
    reset_singletons()


async def main():
    tests = [
        test_hash_ignores_whitespace_line_endings_and_unicode_form,
        test_unchanged_reingest_is_a_free_no_op,
        test_renamed_copy_is_refused_and_names_the_original,
        test_changed_content_under_the_same_name_replaces,
        test_force_and_kill_switch_bypass_the_check,
        test_repeated_chunks_inside_one_document_are_dropped,
        test_scraped_page_differing_only_in_front_matter_is_a_duplicate,
        test_rescrape_with_new_citation_metadata_refreshes_it,
        test_identical_bytes_skip_extraction_entirely,
        test_duplicate_clears_the_stale_chunks_of_its_own_name,
        test_concurrent_identical_uploads_store_one_copy,
    ]
    for test in tests:
        result = test()
        if asyncio.iscoroutine(result):
            await result
        print("ok  %s" % test.__name__)
    print("\n%d dedup checks passed" % len(tests))


if __name__ == "__main__":
    asyncio.run(main())
