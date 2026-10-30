"""
Compressed uploads (B1): the safe-extraction layer, the router's archive
routing, gz handling in the structured pipeline, and the /ingest/archive job.
Run: python tests/test_archive.py

No test framework on purpose — same convention as test_extraction.py. These
checks are the ones that rot silently when someone touches file handling: the
routing table, the bomb/traversal guards, and the failure contract everything
downstream depends on (extraction_failed()).

Security cases here are not hypotheticals — each one is an attack shape with a
name: zip-slip ("../evil"), decompression bombs (ratio/member-count/expanded
size), symlink members, nested archives, and declared-size lies.
"""

import asyncio
import gzip
import io
import os
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from extractors.document import extraction_failed
from extractors.router import route_extraction, _get_extension
from pipelines import structured
from pipelines.structured import StructuredError

CSV = (b"time,open,high,low,close,volume\n"
       b"2020-03-11 09:30,8620.5,8630.0,8615.25,8628.75,1420\n"
       b"2020-03-11 09:35,8628.75,8640.0,8626.0,8637.5,1655\n")
MD = b"# Report\n\nA gap is left behind.\n"
CODE = b"use std::collections::HashMap;\n\n// Counts occurrences of each word.\nfn word_count(words: &[&str]) -> HashMap<&str, usize> {\n    let mut counts = HashMap::new();\n    for word in words {\n        *counts.entry(word).or_insert(0) += 1;\n    }\n    counts\n}\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _auth_headers():
    """Bearer header for the protected routes (services/auth.py); sets a test key when none is configured."""
    import config
    config.ADMIN_API_KEY = config.ADMIN_API_KEY or "test-admin-key"
    return {"Authorization": "Bearer %s" % config.ADMIN_API_KEY}



def make_zip(files: dict, method=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def make_targz(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ── containers: zip / tar ───────────────────────────────────────────────────

async def test_zip_members_route_through_their_own_extractors():
    out = await route_extraction("backup.zip", "application/zip", make_zip({
        "data/sales.csv": CSV,
        "notes/report.md": MD,
        "scripts/word_count.rs": CODE,
    }))
    assert "Compressed archive: backup.zip (3 file(s)" in out, out
    assert "data/sales.csv" in out and "Columns (6)" in out, out   # structured summary
    assert "notes/report.md" in out and "A gap is left behind." in out, out
    assert "scripts/word_count.rs" in out and "Rust" in out, out
    assert "1420" in out.split("Sample data", 1)[-1], out        # rows only in the sample
    assert not extraction_failed(out), out


async def test_zip_skips_media_binary_nested_and_empty_members_with_reasons():
    out = await route_extraction("mixed.zip", "", make_zip({
        "good.txt": MD,
        "chart.png": PNG,
        "secret.exe": b"\x00\x01\x02binary",
        "inner.zip": make_zip({"x.txt": b"nested"}),
        "empty.txt": b"",
    }))
    assert "good.txt" in out and "A gap is left behind." in out, out
    assert out.count("Skipped:") == 4, out
    assert "vision model" in out, out                  # media is named, not dumped
    assert "unsupported file type" in out, out         # binary is named
    assert "nested archives are not unpacked" in out, out
    assert "file is empty" in out, out
    assert "binary" not in out, out                    # and no binary leaks as text


async def test_tar_gz_container_is_unpacked():
    out = await route_extraction("archive.tar.gz", "", make_targz({
        "a.txt": MD,
        "sub/b.csv": CSV,
    }))
    assert "Compressed archive: archive.tar.gz (2 file(s)" in out, out
    assert "a.txt" in out and "A gap is left behind." in out, out
    assert "sub/b.csv" in out and "Columns (6)" in out, out


async def test_path_traversal_members_are_refused():
    out = await route_extraction("evil.zip", "", make_zip({
        "../evil.txt": b"escaped",
        "/abs/evil2.txt": b"escaped",
        "C:\\win\\evil3.txt": b"escaped",
        "good.txt": MD,
    }))
    assert out.count("unsafe path") == 3, out
    assert "good.txt" in out and "A gap is left behind." in out, out   # siblings survive
    assert "escaped" not in out, out                           # ...and nothing leaks


async def test_symlink_member_is_skipped():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("link.csv")
        info.external_attr = (0o120777 << 16)      # S_IFLNK
        zf.writestr(info, "/etc/passwd")
        zf.writestr("real.txt", MD)
    out = await route_extraction("links.zip", "", buf.getvalue())
    assert "symbolic link" in out, out
    assert "root:" not in out, out
    assert "real.txt" in out and "A gap is left behind." in out, out


# ── single-stream .gz ───────────────────────────────────────────────────────

async def test_gzip_single_stream_is_the_file_inside():
    out = await route_extraction("data.csv.gz", "", gzip.compress(CSV))
    assert "Compressed archive" not in out, out
    assert "File: data.csv" in out and "Columns (6)" in out, out


async def test_bare_gz_sniffs_its_content():
    out = await route_extraction("dump.gz", "", gzip.compress(b"hello, plain text"))
    assert "hello, plain text" in out, out

    out = await route_extraction("dump.gz", "", make_targz({"a.txt": MD}))
    assert "Compressed archive: dump.tar" in out and "a.txt" in out, out

    out = await route_extraction("dump.gz", "", gzip.compress(b"\x00\x01\x02binary"))
    assert extraction_failed(out), out
    assert "not a supported file type" in out, out


async def test_double_compression_refuses_the_third_layer():
    twice = await route_extraction("data.csv.gz.gz", "",
                                   gzip.compress(gzip.compress(CSV)))
    assert "File: data.csv" in twice, twice        # two layers unwrap fine

    thrice = await route_extraction("data.csv.gz.gz.gz", "",
                                    gzip.compress(gzip.compress(gzip.compress(CSV))))
    assert extraction_failed(thrice), thrice
    assert "compressed more than" in thrice, thrice


# ── refusals: corrupt / empty / unsupported / over the limits ───────────────

async def test_corrupt_archives_fail_with_a_named_reason():
    out = await route_extraction("broken.zip", "", b"PK\x03\x04garbage-not-a-zip")
    assert extraction_failed(out), out
    assert "not a readable ZIP archive" in out, out

    out = await route_extraction("broken.gz", "", b"\x1f\x8b\x08truncated")
    assert extraction_failed(out), out
    assert "not a readable" in out, out


async def test_empty_archives_fail_with_a_named_reason():
    out = await route_extraction("empty.zip", "", make_zip({}))
    assert extraction_failed(out) and "contains no files" in out, out

    out = await route_extraction("empty.gz", "", gzip.compress(b""))
    assert extraction_failed(out) and "zero bytes" in out, out


async def test_rar_and_7z_are_named_unsupported_not_sent_to_tika():
    for name in ("old.rar", "old.7z"):
        out = await route_extraction(name, "", b"Rar!\x1a\x07\x00junk")
        assert extraction_failed(out), (name, out)
        assert "archives are not supported" in out, (name, out)
        assert ".zip" in out, (name, out)          # the way out is stated


async def test_compression_bomb_is_refused():
    bomb = make_zip({"big.csv": b"\x00" * (2 * 1024 * 1024)})
    out = await route_extraction("bomb.zip", "", bomb)
    assert extraction_failed(out), out
    assert "ARCHIVE_MAX_RATIO" in out and "bomb" in out, out

    out = await route_extraction("bomb.gz", "", gzip.compress(b"\x00" * (2 * 1024 * 1024)))
    assert extraction_failed(out) and "ARCHIVE_MAX_RATIO" in out, out


async def test_member_count_and_total_size_limits_are_enforced():
    saved = (config.ARCHIVE_MAX_MEMBERS, config.ARCHIVE_MAX_TOTAL_BYTES)
    try:
        config.ARCHIVE_MAX_MEMBERS = 5
        out = await route_extraction("many.zip", "", make_zip(
            {"f%d.txt" % i: b"x" for i in range(10)}))
        assert extraction_failed(out) and "ARCHIVE_MAX_MEMBERS" in out, out

        config.ARCHIVE_MAX_MEMBERS = saved[0]
        config.ARCHIVE_MAX_TOTAL_BYTES = 1024
        out = await route_extraction("big.zip", "", make_zip(
            {"a.csv": b"x" * 4096, "b.csv": b"y" * 4096}))
        assert extraction_failed(out) and "ARCHIVE_MAX_TOTAL_BYTES" in out, out
    finally:
        config.ARCHIVE_MAX_MEMBERS, config.ARCHIVE_MAX_TOTAL_BYTES = saved


async def test_oversize_member_is_skipped_not_fatal():
    saved = config.ARCHIVE_MAX_MEMBER_BYTES
    try:
        config.ARCHIVE_MAX_MEMBER_BYTES = 100
        out = await route_extraction("mixed.zip", "", make_zip({
            "big.csv": b"x" * 200,
            "small.txt": MD,
        }))
    finally:
        config.ARCHIVE_MAX_MEMBER_BYTES = saved
    assert "small.txt" in out and "A gap is left behind." in out, out
    assert "ARCHIVE_MAX_MEMBER_BYTES" in out, out
    assert "big.csv" in out, out                    # named in the inventory


# ── structured pipeline: data.csv.gz IS data.csv ────────────────────────────

def test_structured_pipeline_treats_csv_gz_as_csv():
    assert structured.detect_format("a/b/data.csv.gz") == "csv"
    assert structured.detect_format("data.csv") == "csv"

    fmt, frames = structured.parse("data.csv.gz", gzip.compress(CSV))
    plain_fmt, plain = structured.parse("data.csv", CSV)
    assert fmt == plain_fmt == "csv"
    df, plain_df = next(iter(frames.values())), next(iter(plain.values()))
    assert list(df.columns) == list(plain_df.columns), (df.columns, plain_df.columns)
    assert len(df) == len(plain_df) == 2

    for bad, expect in (("x.zip", "compressed archive"),
                        ("x.tar.gz", "compressed archive"),
                        ("x.gz", "does not say what is inside")):
        try:
            structured.detect_format(bad)
        except StructuredError as exc:
            assert expect in str(exc), (bad, exc)
        else:
            raise AssertionError("%s should not be a data file" % bad)

    try:
        structured.parse("x.csv.gz", b"\x1f\x8b\x08truncated")
    except StructuredError as exc:
        assert "not a readable" in str(exc), exc
    else:
        raise AssertionError("a corrupt .csv.gz must raise StructuredError")


# ── contracts everything downstream leans on ────────────────────────────────

async def test_failures_use_the_extraction_failed_contract():
    """Every refusal is a failure placeholder: ingestion must never index it."""
    for name, blob in (("broken.zip", b"PKgarbage"),
                       ("old.rar", b"Rar!junk"),
                       ("empty.gz", gzip.compress(b""))):
        out = await route_extraction(name, "", blob)
        assert out.startswith("Document: "), (name, out)
        assert extraction_failed(out), (name, out)


async def test_no_temp_files_are_written():
    """Extraction is in-memory: no scratch files, so no cleanup to forget."""
    before = set(os.listdir(tempfile.gettempdir()))
    await route_extraction("backup.zip", "", make_zip({"a.csv": CSV, "b.md": MD}))
    await route_extraction("data.csv.gz", "", gzip.compress(CSV))
    after = set(os.listdir(tempfile.gettempdir()))
    assert after == before, after - before


async def test_uncompressed_uploads_are_unchanged():
    csv_out = await route_extraction("NQ_2020.csv", "text/csv", CSV)
    assert "Columns (6)" in csv_out and "code interpreter" in csv_out.lower(), csv_out
    code_out = await route_extraction("word_count.rs", "text/plain", CODE)
    assert "Rust" in code_out and "word_count" in code_out, code_out
    img_out = await route_extraction("chart.png", "image/png", PNG)
    assert "Image file" in img_out and "chart.png" in img_out
    txt_out = await route_extraction("notes.txt", "text/plain", MD)
    assert txt_out == MD.decode("utf-8"), txt_out
    assert _get_extension("NQ_2020.csv") == "csv"


# ── /ingest/archive job dispatch ────────────────────────────────────────────

def _offline_store():
    from pipelines import ingestion
    ingestion._store = ingestion.JobStore()
    ingestion._store._degraded = True       # memory store, no asyncpg in tests
    return ingestion


async def test_archive_job_indexes_members_and_skips_the_rest():
    from pipelines import archive_ingestion
    ingestion = _offline_store()
    calls = []

    async def fake_text(text, filename, source_type, collection=None, metadata=None):
        calls.append(("text", filename, source_type, metadata))
        return {"chunks_created": 2}

    async def fake_structured(filename, file_bytes, metadata=None, db=None):
        calls.append(("structured", filename, metadata))
        return {"status": "done", "indexed": 1}

    real = (ingestion.ingest_text, structured.ingest_structured)
    ingestion.ingest_text, structured.ingest_structured = fake_text, fake_structured
    try:
        job = await ingestion.create_job("backup.zip", "archive", config.COLLECTION_DOCUMENTS)
        result = await archive_ingestion.run_archive_job(
            job["job_id"], "backup.zip",
            make_zip({"notes.md": MD, "data.csv": CSV, "evil.exe": b"\x00"}))
    finally:
        ingestion.ingest_text, structured.ingest_structured = real

    assert result["status"] == "done", result
    kinds = {(c[0], c[1]) for c in calls}
    assert ("text", "backup.zip:notes.md") in kinds, calls
    assert ("structured", "backup.zip:data.csv") in kinds, calls
    assert len(calls) == 2, calls                      # evil.exe never dispatched
    assert calls[0][3]["archive"] == "backup.zip", calls

    report = (await ingestion.get_store().get(job["job_id"]))["quality"]
    assert report["kind"] == "archive" and report["members_total"] == 3, report
    assert report["indexed"]["count"] == 2, report
    assert any(s["name"] == "backup.zip:evil.exe" and "unsupported" in s["reason"]
               for s in report["skipped"]["items"]), report


async def test_archive_job_names_members_skipped_as_duplicates():
    """An archive of already-indexed files is a success that says why it
    indexed nothing — not a bare "done, 0 chunks" (pipelines/dedup.py)."""
    from pipelines import archive_ingestion
    ingestion = _offline_store()

    async def fake_text(text, filename, source_type, collection=None, metadata=None):
        return {"chunks_created": 0, "skipped": "duplicate", "duplicate_of": "notes.md"}

    async def fake_structured(filename, file_bytes, metadata=None, db=None):
        return {"status": "done", "indexed": 0, "skipped": "unchanged"}

    real = (ingestion.ingest_text, structured.ingest_structured)
    ingestion.ingest_text, structured.ingest_structured = fake_text, fake_structured
    try:
        job = await ingestion.create_job("again.zip", "archive", config.COLLECTION_DOCUMENTS)
        result = await archive_ingestion.run_archive_job(
            job["job_id"], "again.zip", make_zip({"notes.md": MD, "data.csv": CSV}))
    finally:
        ingestion.ingest_text, structured.ingest_structured = real

    assert result["status"] == "done", result
    items = {i["name"]: i for i in
             (await ingestion.get_store().get(job["job_id"]))["quality"]["indexed"]["items"]}
    assert items["again.zip:notes.md"]["skipped"] == "duplicate", items
    assert items["again.zip:notes.md"]["duplicate_of"] == "notes.md", items
    assert items["again.zip:data.csv"]["skipped"] == "unchanged", items


async def test_archive_job_single_stream_identity_is_the_file_inside():
    from pipelines import archive_ingestion
    ingestion = _offline_store()
    calls = []

    async def fake_structured(filename, file_bytes, metadata=None, db=None):
        calls.append(filename)
        return {"status": "done", "indexed": 1}

    real = structured.ingest_structured
    structured.ingest_structured = fake_structured
    try:
        job = await ingestion.create_job("data.csv.gz", "archive", config.COLLECTION_DOCUMENTS)
        result = await archive_ingestion.run_archive_job(
            job["job_id"], "data.csv.gz", gzip.compress(CSV))
    finally:
        structured.ingest_structured = real

    assert result["status"] == "done", result
    assert calls == ["data.csv"], calls        # same identity as POST /ingest/structured


async def test_archive_job_reports_a_corrupt_archive():
    from pipelines import archive_ingestion
    ingestion = _offline_store()
    job = await ingestion.create_job("broken.zip", "archive", config.COLLECTION_DOCUMENTS)
    result = await archive_ingestion.run_archive_job(job["job_id"], "broken.zip",
                                                     b"PKgarbage")
    assert result["status"] == "failed", result
    stored = await ingestion.get_store().get(job["job_id"])
    assert stored["status"] == "failed" and "ZIP" in stored["error"], stored


async def test_archive_endpoint_over_http():
    """POST /ingest/archive end to end: unpack -> dispatch -> job report."""
    from fastapi.testclient import TestClient
    from services import embedding_service, qdrant_service
    from services.embedding_service import StubProvider
    from test_retrieval import FakeQdrant, fake_qdrant_service, reset_singletons

    reset_singletons()
    fake = FakeQdrant()
    embedding_service._provider = StubProvider(dimensions=8)
    qdrant_service._service = fake_qdrant_service(fake)
    ingestion = _offline_store()
    from main import app

    with TestClient(app, headers=_auth_headers()) as client:
        posted = client.post(
            "/ingest/archive",
            files={"file": ("backup.zip", make_zip({"notes.md": MD,
                                                    "evil.exe": b"\x00"}),
                            "application/zip")})
        assert posted.status_code == 200, posted.text
        job = client.get("/ingest/status/%s" % posted.json()["job_id"]).json()
        assert job["status"] == "done", job
        assert job["chunks_created"] > 0, job
        assert job["quality"]["skipped"]["count"] == 1, job

        sources = {p["payload"]["source"]["filename"]
                   for p in fake.points[config.COLLECTION_DOCUMENTS].values()}
        assert any("backup.zip:notes.md" in s for s in sources), sources

        # A non-compressed file here is a caller bug and answers 400 now.
        plain = client.post("/ingest/archive",
                            files={"file": ("notes.md", MD, "text/markdown")})
        assert plain.status_code == 400 and "not a compressed file" in plain.text, plain.text

        broken = client.post("/ingest/archive",
                             files={"file": ("broken.zip", b"PKgarbage",
                                             "application/zip")})
        assert broken.status_code == 200, broken.text
        job = client.get("/ingest/status/%s" % broken.json()["job_id"]).json()
        assert job["status"] == "failed" and "ZIP" in job["error"], job
    reset_singletons()


async def main():
    tests = [
        test_zip_members_route_through_their_own_extractors,
        test_zip_skips_media_binary_nested_and_empty_members_with_reasons,
        test_tar_gz_container_is_unpacked,
        test_path_traversal_members_are_refused,
        test_symlink_member_is_skipped,
        test_gzip_single_stream_is_the_file_inside,
        test_bare_gz_sniffs_its_content,
        test_double_compression_refuses_the_third_layer,
        test_corrupt_archives_fail_with_a_named_reason,
        test_empty_archives_fail_with_a_named_reason,
        test_rar_and_7z_are_named_unsupported_not_sent_to_tika,
        test_compression_bomb_is_refused,
        test_member_count_and_total_size_limits_are_enforced,
        test_oversize_member_is_skipped_not_fatal,
        test_structured_pipeline_treats_csv_gz_as_csv,
        test_failures_use_the_extraction_failed_contract,
        test_no_temp_files_are_written,
        test_uncompressed_uploads_are_unchanged,
        test_archive_job_indexes_members_and_skips_the_rest,
        test_archive_job_names_members_skipped_as_duplicates,
        test_archive_job_single_stream_identity_is_the_file_inside,
        test_archive_job_reports_a_corrupt_archive,
        test_archive_endpoint_over_http,
    ]
    for test in tests:
        result = test()
        if asyncio.iscoroutine(result):
            await result
        print("ok  %s" % test.__name__)
    print("\n%d archive checks passed" % len(tests))


if __name__ == "__main__":
    asyncio.run(main())
