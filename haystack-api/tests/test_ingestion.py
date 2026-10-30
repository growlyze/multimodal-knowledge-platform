"""
Smallest thing that fails if ingestion or the quality framework breaks.
Run: python tests/test_ingestion.py

No framework, same style as test_retrieval.py — and it reuses that file's
FakeQdrant instead of growing a second one, so both suites exercise the same
REST shapes. Nothing here touches the network.
"""

import re
import asyncio
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import httpx
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

import config
from evaluation import retrieval_evaluator
from evaluation.chunk_inspector import MAX_TOKENS, inspect_chunks
from evaluation.coherence_judge import judge_chunks
from pipelines import ingestion
from pipelines.chunkers.base import TextChunk, count_tokens, find_jargon
from pipelines.chunkers.code_chunker import CodeChunker
from pipelines.chunkers.semantic_chunker import SemanticChunker, parse_timestamp
from pipelines.chunkers.text_chunker import TextChunker
from routers import ingest as ingest_router
from services import embedding_service, qdrant_service
from services.embedding_service import StubProvider

from test_retrieval import FakeQdrant, fake_qdrant_service, reset_singletons


# ── fixtures ────────────────────────────────────────────

DOCUMENT = "\n\n".join(
    "Paragraph %d. The retry window left behind by a failover is a backlog "
    "that the scheduler returns to clear before continuing toward the checkpoint. "
    "The autoscaler ignores manual reservations and quotas, e.g. the "
    "nightly batch sweep takes the backlog first. Only then does the "
    "pipeline deliver work into the reserved pool for a steady throughput." % n
    for n in range(1, 26)
)

TRANSCRIPT = "\n".join(
    ["[00:00:%02d] A batch job is an asynchronous task queued for later." % i
     for i in range(0, 30)]
    + ["[00:01:%02d] The dashboard draws a box on the screen for that level." % i
       for i in range(0, 30)]
)

CODE_SAMPLE = """// Detects a spike: the value jumps above its own recent range.
detectSpike(idx) =>
    jump = close[idx] - close[idx - 1]
    rising = close[idx] > close[idx - 1]
    rising and jump > threshold

// Rolling mean: the average of the last n values.
rollingMean(lookback) =>
    total = close[0] + close[1]
    total > 0 ? total / lookback : na

// Plot the detection on the chart.
plotSignal() =>
    plot(detectSpike(1) ? close : na)
"""


SCRAPED_PAGE = '''---
source: "https://docs.example.com/sdk-reference/#run_query"
title: run_query — SDK Reference
type: code_reference
kind: function
name: run_query
version: v2
scraped_at: "2026-08-27T22:47:51+00:00"
---

# run_query()

The query helper returns the rows matching the filter, capped by the limit.
'''


class TopicEmbedder:
    """Two orthogonal topics — 'batch' text and everything else."""

    dimensions = 2

    async def embed(self, texts, input_type="document"):
        return [[1.0, 0.0] if "batch" in t else [0.0, 1.0] for t in texts]


def _auth_headers():
    """Bearer header for the protected routes (services/auth.py); sets a test key when none is configured."""
    import config
    config.ADMIN_API_KEY = config.ADMIN_API_KEY or "test-admin-key"
    return {"Authorization": "Bearer %s" % config.ADMIN_API_KEY}



def use_fake_backends(dimensions: int = 8) -> FakeQdrant:
    """Point the embedding + Qdrant singletons at offline fakes."""
    reset_singletons()
    fake = FakeQdrant()
    embedding_service._provider = StubProvider(dimensions=dimensions)
    qdrant_service._service = fake_qdrant_service(fake)
    ingestion._store = ingestion.JobStore()
    ingestion._store._degraded = True     # no asyncpg in tests, use the memory store
    return fake


def upload(name: str, data: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=name)


# ── 1. text chunker size ────────────────────────────────

def test_text_chunker_respects_size():
    chunks = TextChunker(max_tokens=500, overlap_tokens=100).chunk(
        DOCUMENT, {"source": "doc.txt", "source_type": "document"})
    assert len(chunks) > 2, len(chunks)

    sizes = [count_tokens(c.content) for c in chunks]
    assert max(sizes) <= 500, sizes
    # Everything but the tail should land in the 400-500 band.
    assert all(400 <= s <= 500 for s in sizes[:-1]), sizes
    assert all(c.metadata["chunk_index"] == i for i, c in enumerate(chunks))
    assert all(c.metadata["source"] == "doc.txt" for c in chunks)

    # Overlap is real: consecutive chunks share a sentence.
    first_tail = chunks[0].content.split(". ")[-2]
    assert first_tail in chunks[1].content, first_tail


# ── 2. text chunker sentence boundaries ─────────────────

def test_text_chunker_preserves_sentences():
    chunks = TextChunker(max_tokens=200, overlap_tokens=40).chunk(
        DOCUMENT, {"source": "doc.txt", "source_type": "document"})
    for chunk in chunks:
        body = chunk.content.strip()
        assert body[-1] in ".!?", body[-60:]
        assert body[0].isupper() or body[0].isdigit(), body[:60]

    # "e.g." must not become a chunk boundary.
    assert not any(c.content.startswith("g.") for c in chunks)

    # A single sentence longer than the target still splits, but only on words.
    monster = "The " + "recalculation " * 400 + "ends here."
    pieces = TextChunker(max_tokens=100, overlap_tokens=0).chunk(
        monster, {"source": "m.txt", "source_type": "document"})
    assert len(pieces) > 3
    assert all(not c.content.startswith(" ") for c in pieces)
    assert "".join(c.content for c in pieces).count("recalculation") == 400


# ── 3. code chunker keeps functions whole ───────────────

def test_code_chunker_preserves_functions():
    from pipelines.chunkers.code_chunker import detect_language

    chunks = CodeChunker(max_tokens=40, overlap_tokens=10).chunk(
        CODE_SAMPLE, {"source": "signals.txt", "source_type": "code"})
    assert len(chunks) > 1, len(chunks)

    for name, body in [("detectSpike", "rising and jump > threshold"),
                       ("rollingMean", "total > 0 ? total / lookback : na")]:
        owners = [c for c in chunks if re.search(r"^%s\([^()]*\)\s*=>" % name, c.content, re.M)]
        assert len(owners) >= 1, name
        assert all(body in c.content for c in owners), name
        # The explanation comment travels with the code.
        assert all("//" in c.content.split(name)[0] for c in owners), name

    assert all(c.metadata["language"] == "code" for c in chunks)
    assert detect_language("word_count.rs") == "Rust"
    assert detect_language("analysis.py") == "Python"
    names = {n for c in chunks for n in c.metadata.get("function_names", [])}
    assert {"detectSpike", "rollingMean", "plotSignal"} <= names, names

    # Layer 1 agrees: nothing was split away from its definition.
    report = inspect_chunks(chunks, "code", CODE_SAMPLE, "signals.txt")
    assert "split_function" not in report["counts"], report["counts"]


# ── 4. semantic chunker topic shift ─────────────────────

async def test_semantic_chunker_detects_topic_shift():
    chunker = SemanticChunker(embedder=TopicEmbedder())
    chunks = await chunker.achunk(TRANSCRIPT, {"source": "walkthrough.txt",
                                               "source_type": "transcript"})
    assert len(chunks) >= 2, len(chunks)
    assert "batch" in chunks[0].content and "dashboard" not in chunks[0].content
    assert any("dashboard" in c.content for c in chunks)

    # Without an embedder it still chunks — just on size, not topic.
    plain = SemanticChunker(embedder=None).chunk(TRANSCRIPT, {"source": "s47.txt",
                                                              "source_type": "transcript"})
    assert plain and all(c.metadata.get("start_time") is not None for c in plain)

    # A failing embedder must not fail the chunker.
    class Broken:
        async def embed(self, texts, input_type="document"):
            raise RuntimeError("provider down")

    survived = await SemanticChunker(embedder=Broken()).achunk(
        TRANSCRIPT, {"source": "s47.txt", "source_type": "transcript"})
    assert survived, "embedding failure must degrade, not explode"


# ── 5. pipeline end to end ──────────────────────────────

async def test_ingestion_pipeline_end_to_end():
    fake = use_fake_backends()
    result = await ingestion.ingest_text(
        DOCUMENT, "doc.txt", "document", "documents")

    assert result["chunks_created"] > 1, result
    points = fake.points["documents"]
    assert len(points) == result["chunks_created"]

    payloads = [p["payload"] for p in points.values()]
    # Filter zone flat, source zone nested — the citation formatter reads the latter.
    assert all(p["content"] and p["source"]["filename"] == "doc.txt" for p in payloads)
    assert all(p["source_type"] == "document" for p in payloads)
    assert all(p["collection"] == "documents" for p in payloads)
    assert all(p["origin"] == ingestion.DEFAULT_ORIGIN for p in payloads)
    assert all("ingested_at" in p and "chunk_index" in p for p in payloads)
    assert all(set(ingestion.SOURCE_KEYS) <= set(p["source"]) for p in payloads)
    assert not any(k in p for p in payloads for k in ("url", "title", "page_number"))
    # Sparse vectors are derived from payload["content"] by qdrant_service.
    assert all("text" in p["vector"] for p in points.values())

    created = fake.collections["documents"]
    assert created["vectors"]["dense"]["size"] == 8
    assert created["sparse_vectors"]["text"]["modifier"] == "idf"

    assert result["quality"]["passed"], result["quality"]
    reset_singletons()


async def test_front_matter_becomes_the_source_object():
    """A scraped page carries its own citation metadata; ingestion files it."""
    meta, body = ingestion.parse_front_matter(SCRAPED_PAGE)
    assert meta["source"].endswith("#run_query") and meta["kind"] == "function"
    assert body.startswith("# run_query()"), body[:40]

    # Missing and unterminated front matter pass the text through untouched.
    assert ingestion.parse_front_matter("no front matter") == ({}, "no front matter")
    assert ingestion.parse_front_matter("---\nnever closed") == ({}, "---\nnever closed")
    assert ingestion.parse_front_matter("") == ({}, "")
    # A document opening with a markdown rule is not front matter.
    assert ingestion.parse_front_matter("---\nJust prose, with a colon: here.\n---\nx") \
        == ({}, "---\nJust prose, with a colon: here.\n---\nx")

    # The reference pages named after operators and annotations are not valid
    # YAML; their metadata is read line by line rather than dropped.
    operators, rest = ingestion.parse_front_matter(
        '---\nsource: "https://docs.example.com/#attr_deprecated"\ntitle: @deprecated — SDK Reference\n'
        "type: code_reference\nkind: annotation\nname: @deprecated\n---\n\n# @deprecated\n")
    assert operators["name"] == "@deprecated" and operators["kind"] == "annotation"
    assert operators["source"] == "https://docs.example.com/#attr_deprecated", operators
    assert rest.startswith("# @deprecated")

    fake = use_fake_backends()
    result = await ingestion.ingest_text(
        SCRAPED_PAGE, "reference/run_query.md", "code_reference",
        config.COLLECTION_CODE, {"origin": "scraper"})

    payload = list(fake.points[config.COLLECTION_CODE].values())[0]["payload"]
    assert result["chunks_created"] == 1, result
    assert "---" not in payload["content"], "front matter must not be indexed"
    # `type` decides the source type, and the YAML `source` key is the URL.
    assert payload["source_type"] == "code_reference"
    assert payload["origin"] == "scraper"
    assert payload["source"]["url"].endswith("#run_query")
    assert payload["source"]["title"] == "run_query — SDK Reference"
    assert payload["source"]["name"] == "run_query" and payload["source"]["kind"] == "function"
    assert payload["source"]["version"] == "v2"
    assert payload["source"]["filename"] == "reference/run_query.md"
    assert payload["source"]["original_filename"] is None
    reset_singletons()


async def test_chat_upload_keeps_the_human_filename():
    uploaded = "24721dfc-d5d6-4471-9ce2-2d94ba676c5b_33.metrics_hourly_EU.csv"
    assert ingestion.strip_uuid_prefix(uploaded) == "33.metrics_hourly_EU.csv"
    assert ingestion.strip_uuid_prefix("plain.csv") == "plain.csv"

    fake = use_fake_backends()
    await ingestion.ingest_text(DOCUMENT, uploaded, "document", "documents",
                                {"origin": "open-webui-chat", "chat_id": "c1"})
    payload = list(fake.points["documents"].values())[0]["payload"]
    assert payload["origin"] == "open-webui-chat"
    assert payload["chat_id"] == "c1", "extra metadata still rides in the filter zone"
    assert payload["source"]["filename"] == uploaded
    assert payload["source"]["original_filename"] == "33.metrics_hourly_EU.csv"
    reset_singletons()


async def test_pipeline_routes_by_source_type():
    assert isinstance(ingestion.select_chunker("document"), TextChunker)
    assert isinstance(ingestion.select_chunker("transcript"), SemanticChunker)
    assert isinstance(ingestion.select_chunker("code"), CodeChunker)
    assert isinstance(ingestion.select_chunker("anything-else"), TextChunker)
    assert ingestion.default_collection("code") == config.COLLECTION_CODE
    assert ingestion.default_collection("transcript") == config.COLLECTION_DOCUMENTS

    fake = use_fake_backends()
    result = await ingestion.ingest_text(CODE_SAMPLE, "signals.txt", "code")
    assert result["collection"] == config.COLLECTION_CODE
    payloads = [p["payload"] for p in fake.points[config.COLLECTION_CODE].values()]
    assert all(p["language"] == "code" for p in payloads)
    reset_singletons()


# ── 6. the endpoint returns before the work happens ─────

async def test_async_ingestion_returns_immediately():
    use_fake_backends()
    started = []

    async def slow_job(job_id, *a, **kw):
        started.append(job_id)
        await asyncio.sleep(0.5)

    original, ingestion.run_job = ingestion.run_job, slow_job
    try:
        background = BackgroundTasks()
        begin = time.perf_counter()
        # Called directly, so FastAPI's Form() defaults are passed explicitly.
        response = await ingest_router.ingest_document(
            background, file=upload("doc.txt", DOCUMENT.encode()),
            source_type="document", collection=None, metadata=None)
        elapsed = time.perf_counter() - begin
    finally:
        ingestion.run_job = original

    assert elapsed < 0.1, "endpoint took %.3fs — it is doing the work inline" % elapsed
    assert response["status"] == "queued" and response["job_id"]
    assert len(background.tasks) == 1, background.tasks
    assert not started, "background task ran before the response was returned"

    job = await ingestion.get_store().get(response["job_id"])
    assert job["status"] == "queued", job
    reset_singletons()


def test_ingest_endpoints_over_http():
    """The whole stack through FastAPI: upload -> job -> status."""
    fake = use_fake_backends()
    from main import app

    with TestClient(app, headers=_auth_headers()) as client:
        posted = client.post("/ingest/transcript",
                             files={"file": ("s47.txt", TRANSCRIPT.encode(), "text/plain")},
                             data={"metadata": json.dumps({"video_id": "s47"})})
        assert posted.status_code == 200, posted.text
        job_id = posted.json()["job_id"]

        # TestClient runs background tasks before returning, so it is done by now.
        status = client.get("/ingest/status/%s" % job_id)
        assert status.status_code == 200, status.text
        job = status.json()
        assert job["status"] == "done", job
        assert job["chunks_created"] > 0 and job["quality"]["passed"], job

        payloads = [p["payload"] for p in fake.points[config.COLLECTION_DOCUMENTS].values()]
        assert all(p["video_id"] == "s47" for p in payloads), "extra metadata was dropped"

        assert client.get("/ingest/status/nope").status_code == 404
        assert client.post("/ingest/document",
                          files={"file": ("empty.txt", b"", "text/plain")}).status_code == 400
        assert client.post("/ingest/document",
                          files={"file": ("a.txt", b"hello there.", "text/plain")},
                          data={"metadata": "not json"}).status_code == 400

        # Architecture decision #8 — chat text never reaches Qdrant, however it
        # is dressed up, but a chat attachment carrying the same origin does.
        chat = json.dumps({"origin": "open-webui-chat", "chat_id": "4cc31daa"})
        assert client.post(
            "/ingest/document",
            files={"file": ("conversation_4cc31daa_2.txt",
                            b"User: what is the batch size. Assistant: it depends.",
                            "text/plain")},
            data={"metadata": chat}).status_code == 400
        assert client.post(
            "/ingest/document", data={"source_type": "conversation", "metadata": chat},
            files={"file": ("notes.txt", b"User: hi", "text/plain")}).status_code == 400
        assert client.post(
            "/ingest/document", data={"metadata": chat},
            files={"file": ("notes2.txt", DOCUMENT.encode(), "text/plain")}).status_code == 200

        sources = {p["payload"]["source"]["filename"]
                   for p in fake.points[config.COLLECTION_DOCUMENTS].values()}
        assert not any(s.startswith("conversation_") for s in sources), sources
        assert "notes2.txt" in sources, sources
    reset_singletons()


# ── 6b. the job list the panel reads ────────────────────

async def test_job_list_shows_uploads_nobody_is_holding():
    """GET /ingest/status lists jobs the caller never had the id for.

    The regression this pins: a CSV posted by the Open WebUI filter is
    fire-and-forget, so when it failed there was no id to poll and the panel
    fell back to "uploads started in this browser session" — which is empty.
    A failed ingestion and an ingestion that never happened looked identical.
    """
    use_fake_backends()
    from main import app

    store = ingestion.get_store()
    done = await ingestion.create_job("sales_2020.csv", "structured_data")
    await store.update(done["job_id"], status="done", chunks_created=1)
    broken = await ingestion.create_job("ops_2021.csv", "structured_data")
    await store.update(broken["job_id"], status="failed",
                       error="could not parse ops_2021.csv as csv")

    with TestClient(app, headers=_auth_headers()) as client:
        listed = client.get("/ingest/status")
        assert listed.status_code == 200, listed.text
        jobs = listed.json()["jobs"]

        names = [job["filename"] for job in jobs]
        assert "sales_2020.csv" in names and "ops_2021.csv" in names, names
        # Newest first — a new upload must not land below yesterday's.
        assert names.index("ops_2021.csv") < names.index("sales_2020.csv"), names

        failed = [job for job in jobs if job["status"] == "failed"]
        assert failed and failed[0]["error"], "the error is the point of the list"
        # The panel polls /ingest/status/{job_id} off whatever the list gave it.
        assert all(job.get("job_id") for job in jobs), jobs

        only = client.get("/ingest/status", params={"status": "failed"}).json()["jobs"]
        assert [job["filename"] for job in only] == ["ops_2021.csv"], only

        assert client.get("/ingest/status", params={"limit": 1}).json()["count"] == 1
        assert client.get("/ingest/status", params={"status": "nope"}).status_code == 400
        # P123: DELETE /admin/documents now writes `deleted`, so it is a state the
        # list must be filterable by — otherwise "what was removed?" is a 400.
        assert client.get("/ingest/status", params={"status": "deleted"}).status_code == 200
        # The fixed path must not be swallowed by /status/{job_id}.
        assert client.get("/ingest/status/nope").status_code == 404


# ── 7. job status tracking ──────────────────────────────

async def test_job_status_tracking():
    use_fake_backends()
    job = await ingestion.create_job("doc.txt", "document")
    assert job["status"] == "queued" and job["collection"] == config.COLLECTION_DOCUMENTS

    store = ingestion.get_store()
    assert (await store.get(job["job_id"]))["status"] == "queued"

    result = await ingestion.run_job(job["job_id"], "doc.txt", DOCUMENT.encode(),
                                     "document", job["collection"], None)
    done = await store.get(job["job_id"])
    assert done["status"] == "done", done
    assert done["chunks_created"] == result["chunks_created"] > 0
    assert done["quality"]["chunks"] == done["chunks_created"]
    assert done["completed_at"]

    # A failure is recorded, not raised at the caller.
    broken = await ingestion.create_job("bad.txt", "document")
    embedding_service._provider = None
    original = (config.EMBEDDING_PROVIDER, config.EMBEDDING_API_KEY)
    config.EMBEDDING_PROVIDER, config.EMBEDDING_API_KEY = "openai", "key"
    try:
        await ingestion.run_job(broken["job_id"], "bad.txt", b"Some text here.",
                                "document", broken["collection"], None)
    finally:
        config.EMBEDDING_PROVIDER, config.EMBEDDING_API_KEY = original
    failed = await store.get(broken["job_id"])
    assert failed["status"] == "failed" and failed["error"], failed
    assert await store.get("no-such-job") is None
    reset_singletons()


# ── 8. re-ingesting replaces, never duplicates ──────────

async def test_duplicate_ingestion():
    fake = use_fake_backends()
    first = await ingestion.ingest_text(DOCUMENT, "doc.txt", "document", "documents")
    # Unchanged content is now skipped before any spend (pipelines/dedup.py,
    # tests/test_dedup.py); force_reingest keeps the replace path itself covered.
    unchanged = await ingestion.ingest_text(DOCUMENT, "doc.txt", "document", "documents")
    assert unchanged["skipped"] == "unchanged" and unchanged["chunks_created"] == 0
    second = await ingestion.ingest_text(DOCUMENT, "doc.txt", "document", "documents",
                                         {"force_reingest": True})

    assert second["replaced"] == first["chunks_created"]
    assert len(fake.points["documents"]) == second["chunks_created"]

    # A different file in the same collection is untouched by the replace.
    await ingestion.ingest_text(CODE_SAMPLE, "signals.txt", "code", "documents")
    total = len(fake.points["documents"])
    await ingestion.ingest_text(DOCUMENT, "doc.txt", "document", "documents")
    assert len(fake.points["documents"]) == total
    reset_singletons()


# ── 11-13. chunk inspector ──────────────────────────────

def test_chunk_inspector_catches_fragments():
    fragment = TextChunk("gap", {"source": "a.txt", "source_type": "document"})
    report = inspect_chunks([fragment], "document", "It is a gap. Yes.")
    assert "too_small" in report["counts"], report
    assert report["tokens"]["min"] == count_tokens("gap")

    empty = TextChunk("   ", {"source": "a.txt", "source_type": "document"})
    assert not inspect_chunks([empty], "document")["passed"]
    assert not inspect_chunks([], "document")["passed"]


def test_chunk_inspector_catches_oversized():
    huge = TextChunk("The batch window holds. " * 400,
                     {"source": "a.txt", "source_type": "document"})
    report = inspect_chunks([huge], "document", huge.content)
    assert "too_large" in report["counts"], report
    assert report["tokens"]["max"] > MAX_TOKENS

    ok = TextChunk("The batch window holds. " * 30,
                   {"source": "a.txt", "source_type": "document"})
    assert inspect_chunks([ok], "document", ok.content)["passed"]


def test_chunk_inspector_catches_mid_sentence_split():
    source = "Price left a gap behind. It filled it later."
    bad = [TextChunk("Price left a gap", {"source": "a.txt", "source_type": "document"}),
           TextChunk("behind. It filled it later.",
                     {"source": "a.txt", "source_type": "document"})]
    rules = {i["rule"] for i in inspect_chunks(bad, "document", source)["issues"]}
    assert "ends_mid_sentence" in rules and "starts_mid_sentence" in rules, rules

    # Metadata rules
    assert "missing_metadata" in inspect_chunks(
        [TextChunk("A sentence that ends properly.", {})], "document")["counts"]
    no_time = TextChunk("A sentence that ends properly.",
                        {"source": "a.txt", "source_type": "transcript"})
    report = inspect_chunks([no_time], "transcript", no_time.content)
    assert report["counts"]["missing_timestamp"] == 2 and not report["passed"]

    # Unpunctuated ASR text is not judged on sentence boundaries.
    asr = TextChunk("so we come into the first section and the numbers climb",
                    {"source": "s.txt", "source_type": "transcript",
                     "start_time": 0.0, "end_time": 9.0})
    quiet = inspect_chunks([asr], "transcript", "raw asr with no punctuation at all")
    assert "ends_mid_sentence" not in quiet["counts"], quiet


# ── 14-15. coherence judge ──────────────────────────────

async def _fake_judge(_client, prompt):
    coherent = "batch job" in prompt and prompt.rstrip().endswith('}')
    good = coherent and "asynchronous" in prompt
    return ('```json\n{"score": %d, "standalone": %s, "concepts_complete": %s,'
            ' "reasoning": "test"}\n```'
            % (5 if good else 1, "true" if good else "false",
               "true" if good else "false"))


async def test_coherence_judge_scores_good_chunk_high():
    chunks = [TextChunk("A batch job is an asynchronous task queued for later.", {})
              for _ in range(12)]
    report = await judge_chunks(chunks, sample_size=10, ask=_fake_judge, seed=7)
    assert report["scored"] == 10 and report["sampled"] == 10
    assert report["average_score"] == 5.0 and report["passed"], report
    assert report["standalone_rate"] == 1.0 and report["concepts_complete_rate"] == 1.0
    assert not report["worst"]


async def test_coherence_judge_scores_bad_chunk_low():
    chunks = [TextChunk("and then the value of the", {}) for _ in range(6)]
    report = await judge_chunks(chunks, sample_size=6, ask=_fake_judge, seed=7)
    assert report["average_score"] == 1.0 and not report["passed"], report
    assert len(report["worst"]) == 6, report["worst"]
    assert "avg 1.00" in report["summary"]

    # A judge that returns garbage is reported, never silently averaged in.
    async def broken(_client, _prompt):
        return "I am not JSON"

    failed = await judge_chunks(chunks, sample_size=3, ask=broken, seed=7)
    assert failed["scored"] == 0 and not failed["passed"] and failed["failures"]
    assert await judge_chunks([], ask=_fake_judge) == {
        "sampled": 0, "passed": True, "summary": "no chunks to judge"}


# ── 16. retrieval evaluator ─────────────────────────────

def _search_transport(hit: bool = True):
    """Fake GET /search that answers each case with its own expected keywords."""
    cases = {c["query"]: c for c in retrieval_evaluator.load_queries()["test_queries"]}

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search", request.url
        case = cases[request.url.params["q"]]
        if not hit:
            return httpx.Response(200, json={"results": []})
        keywords = case.get("expected_keywords_in_result") or ["relevant"]
        collections = case.get("expected_collections") or [config.COLLECTION_DOCUMENTS]
        results = []
        for i in range(10):
            results.append({
                "content": "chunk %d covering %s" % (i, " ".join(keywords)),
                "source": "s.txt",
                "source_type": case.get("expected_source_type", "transcript"),
                "score": 1.0 - i / 10,
                "metadata": {"collection": collections[i % len(collections)],
                             "start_time": 12.0},
            })
        return httpx.Response(200, json={"results": results})

    return httpx.MockTransport(handle)


async def test_retrieval_evaluator_runs_test_queries():
    queries = retrieval_evaluator.load_queries()
    cases = queries["test_queries"]
    assert 20 <= len(cases) <= 40, len(cases)
    assert all(c.get("query") and c.get("difficulty") for c in cases)
    assert {"basic", "jargon-abbreviation", "cross-collection", "code", "specific"} <= \
        {c["difficulty"] for c in cases}

    async with httpx.AsyncClient(base_url="http://api.test",
                                 transport=_search_transport(True)) as client:
        report = await retrieval_evaluator.evaluate(queries, client=client)

    assert report["queries"] == len(cases)
    assert report["passed"] and not report["below_target"], report["below_target"]
    assert report["metrics"]["precision_at_5"] == 1.0
    assert report["metrics"]["mrr"] == 1.0
    assert report["metrics"]["jargon_hit_rate"] == 1.0
    assert report["metrics"]["cross_collection_rate"] == 1.0
    assert not report["failures"]
    assert set(report["by_difficulty"]) == {c["difficulty"] for c in cases}

    # An empty knowledge base must fail loudly, not score as passing.
    async with httpx.AsyncClient(base_url="http://api.test",
                                 transport=_search_transport(False)) as client:
        empty = await retrieval_evaluator.evaluate(queries, client=client)
    assert not empty["passed"] and len(empty["failures"]) == len(cases)
    assert empty["metrics"]["mrr"] == 0.0

    # Off-topic results score zero even though ten came back.
    off_topic = [{"content": "unrelated text about the weather", "source": "w.txt",
                  "source_type": "document", "metadata": {}}] * 10
    score = retrieval_evaluator.score_case(cases[0], off_topic)
    assert score["precision_at_5"] == 0.0 and score["reciprocal_rank"] == 0.0
    assert score["failure"]


# ── 17. domain jargon survives every boundary ───────────

async def test_jargon_preservation_in_chunks():
    from pipelines.chunkers import base as chunker_base

    # The protected-term list is empty by default (DOMAIN_JARGON); a deployment
    # supplies its own. The boundary protection is tested with such a list.
    terms = ["batch job", "service level", "rolling average", "load balancer"]
    saved = chunker_base.DOMAIN_JARGON
    chunker_base.DOMAIN_JARGON = terms
    try:
        text = " ".join(
            "Number %d: the %s matters here because the pipeline depends on it." % (i, terms[i % len(terms)])
            for i in range(120)
        )
        expected = find_jargon(text, terms)
        assert {"batch job", "service level"} <= expected

        for chunker in (TextChunker(max_tokens=120, overlap_tokens=20),
                        SemanticChunker(max_tokens=120, overlap_tokens=20, min_tokens=80)):
            chunks = await chunker.achunk(text, {"source": "j.txt", "source_type": "document"})
            report = inspect_chunks(chunks, "document", text, "j.txt")
            assert "split_jargon" not in report["counts"], (type(chunker).__name__, report)

        # A line break through the middle of a term is repaired, not chunked around.
        broken = "A long batch\njob runs before the service\nlevel review."
        chunks = SemanticChunker(max_tokens=200).chunk(broken, {"source": "j.txt",
                                                                "source_type": "transcript"})
        joined = " ".join(c.content for c in chunks)
        assert "batch job" in joined and "service level" in joined, joined

        # And the abbreviations survive as whole tokens for BM25.
        abbrev_terms = ["sla", "mttr", "rpo", "rbac"]
        chunker_base.DOMAIN_JARGON = abbrev_terms
        abbrev = "The SLA and the MTTR and the RPO and the RBAC all matter. " * 30
        chunks = TextChunker(max_tokens=100, overlap_tokens=20).chunk(
            abbrev, {"source": "a.txt", "source_type": "document"})
        assert {"sla", "mttr", "rpo", "rbac"} <= \
            {t for c in chunks for t in find_jargon(c.content, abbrev_terms)}
    finally:
        chunker_base.DOMAIN_JARGON = saved


# ── 18. transcript timestamps ───────────────────────────

async def test_timestamp_continuity_for_transcript_chunks():
    assert parse_timestamp("[00:14:32] the run starts") == (872.0, "the run starts")
    assert parse_timestamp("12:34 section two") == (754.0, "section two")
    assert parse_timestamp("00:00:01,500 --> 00:00:04,000") == (1.5, "")
    assert parse_timestamp("no timestamp here") == (None, "no timestamp here")

    chunks = await SemanticChunker(embedder=TopicEmbedder()).achunk(
        TRANSCRIPT, {"source": "s47.txt", "source_type": "transcript"})

    starts = [c.metadata["start_time"] for c in chunks]
    ends = [c.metadata["end_time"] for c in chunks]
    assert starts[0] == 0.0, starts
    assert all(s <= e for s, e in zip(starts, ends)), list(zip(starts, ends))
    assert starts == sorted(starts), starts
    # Coverage is continuous: each chunk ends no earlier than the next one starts.
    assert all(ends[i] >= starts[i + 1] for i in range(len(chunks) - 1)), list(zip(starts, ends))

    report = inspect_chunks(chunks, "transcript", TRANSCRIPT, "s47.txt")
    assert "missing_timestamp" not in report["counts"], report

    # SRT survives the same path.
    srt = ("1\n00:00:01,000 --> 00:00:04,000\nThe batch job forms here.\n\n"
           "2\n00:00:04,000 --> 00:00:08,000\nThen the queue drains it.\n")
    srt_chunks = SemanticChunker(max_tokens=200).chunk(srt, {"source": "s.srt",
                                                             "source_type": "transcript"})
    assert srt_chunks and srt_chunks[0].metadata["start_time"] == 1.0, srt_chunks
    assert "1\n" not in srt_chunks[0].content, "SRT index numbers leaked into the text"


# ── live Qdrant (optional) ──────────────────────────────

async def test_live_ingestion():
    """Runs only when QDRANT_URL answers — the one that catches a payload real
    Qdrant rejects but FakeQdrant accepts."""
    import os

    url = os.getenv("QDRANT_URL", config.QDRANT_URL)
    try:
        async with httpx.AsyncClient(timeout=2.0) as probe:
            await probe.get("%s/collections" % url.rstrip("/"))
    except Exception:
        print("  live Qdrant at %s not reachable — skipped" % url)
        return

    collection = "haystack_ingestion_selftest"
    reset_singletons()
    embedding_service._provider = StubProvider(dimensions=8)
    service = qdrant_service.QdrantService(url=url)
    qdrant_service._service = service
    ingestion._store = ingestion.JobStore()
    ingestion._store._degraded = True
    try:
        await ingestion.delete_by_source(service, collection, "live.txt")
        first = await ingestion.ingest_text(DOCUMENT, "live.txt", "document", collection)
        assert first["chunks_created"] > 1, first

        info = await service.collection_info(collection)
        assert info["points_count"] >= first["chunks_created"], info

        hits = await service.search_sparse(collection, "retry window backlog", top_k=5)
        assert hits and hits[0]["payload"]["source"]["filename"] == "live.txt", hits

        second = await ingestion.ingest_text(DOCUMENT, "live.txt", "document", collection)
        assert second["replaced"] == first["chunks_created"], second
        after = await service.collection_info(collection)
        assert after["points_count"] == info["points_count"], (info, after)
        print("  live ingestion OK (%d chunks)" % second["chunks_created"])
    finally:
        await ingestion.delete_by_source(service, collection, "live.txt")
        await service.aclose()
        reset_singletons()


# ── runner ──────────────────────────────────────────────

async def test_a_failed_extraction_is_not_indexed_as_a_document():
    """The extractors answer a failure with prose, because /extract/process must
    return a body or Open WebUI fails the upload. Chunked and embedded, that
    prose becomes a one-chunk document in the knowledge base whose content is an
    error message — retrievable, citable, and wrong. The job has to fail instead.
    """
    import extractors.router as router
    from pipelines.ingestion import ExtractionError, extract

    original = router.route_extraction

    async def failed(filename, content_type, file_bytes):
        return ("Document: %s (30,217 bytes). Extraction failed: 'ascii' codec "
                "can't encode character u2019" % filename)

    router.route_extraction = failed
    try:
        try:
            await extract("report.pdf", b"%PDF", "document")
            raise AssertionError("a failure placeholder was returned as a document")
        except ExtractionError as exc:
            assert "Extraction failed" in str(exc), exc
        # A real document still comes straight back.
        async def fine(filename, content_type, file_bytes):
            return "The quarterly report says revenue rose."
        router.route_extraction = fine
        assert await extract("report.pdf", b"%PDF", "document") == (
            "The quarterly report says revenue rose.")
    finally:
        router.route_extraction = original


async def main():
    tests = [
        test_a_failed_extraction_is_not_indexed_as_a_document,
        test_text_chunker_respects_size,
        test_text_chunker_preserves_sentences,
        test_code_chunker_preserves_functions,
        test_semantic_chunker_detects_topic_shift,
        test_ingestion_pipeline_end_to_end,
        test_front_matter_becomes_the_source_object,
        test_chat_upload_keeps_the_human_filename,
        test_pipeline_routes_by_source_type,
        test_async_ingestion_returns_immediately,
        test_ingest_endpoints_over_http,
        test_job_list_shows_uploads_nobody_is_holding,
        test_job_status_tracking,
        test_duplicate_ingestion,
        test_chunk_inspector_catches_fragments,
        test_chunk_inspector_catches_oversized,
        test_chunk_inspector_catches_mid_sentence_split,
        test_coherence_judge_scores_good_chunk_high,
        test_coherence_judge_scores_bad_chunk_low,
        test_retrieval_evaluator_runs_test_queries,
        test_jargon_preservation_in_chunks,
        test_timestamp_continuity_for_transcript_chunks,
        test_live_ingestion,
    ]
    for test in tests:
        result = test()
        if asyncio.iscoroutine(result):
            await result
        print("ok  %s" % test.__name__)
    print("\n%d ingestion checks passed" % len(tests))


if __name__ == "__main__":
    asyncio.run(main())
