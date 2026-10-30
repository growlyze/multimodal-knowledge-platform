"""
Smallest thing that fails if retrieval breaks. Run: python tests/test_retrieval.py

No framework on purpose. Qdrant and the embedding/rerank APIs are faked with
httpx.MockTransport, so this runs offline and still asserts the exact request
bodies we send — which is what silently rots when an API shape changes.

Set QDRANT_URL to a reachable instance and the live section at the bottom also
runs against real Qdrant.
"""

import asyncio
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

import config
import telemetry_compat
from pipelines import retrieval
from pipelines.retrieval import expand_results, reciprocal_rank_fusion
from routers.search import SearchResponse, search_knowledge
from services import embedding_service, qdrant_service, reranker_service
from services.embedding_service import (
    EmbeddingError,
    StubProvider,
    VoyageProvider,
    create_embedding_provider,
)
from services.qdrant_service import QdrantService, build_filter, text_to_sparse
from services.reranker_service import NoopReranker, VoyageReranker, create_reranker


# ── fake Qdrant ─────────────────────────────────────────

class FakeQdrant:
    """In-memory Qdrant speaking the REST shapes qdrant_service actually sends."""

    def __init__(self):
        self.collections = {}   # name -> config body
        self.points = {}        # name -> {id: {vector, payload}}
        self.seen = []          # (method, path, body) for assertions
        # Real Qdrant rejects bodies over service.max_request_size_mb (default 32,
        # v1.19 config.yaml) -- the fake must too, or an unbatched upsert of a
        # long video's ~3000 keyframes passes here and dies in production.
        self.max_request_bytes = 32 * 1024 * 1024

    def transport(self):
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if len(request.content or b"") > self.max_request_bytes:
            return httpx.Response(413, json={"status": {"error": "Payload too large"}})
        body = json.loads(request.content) if request.content else {}
        self.seen.append((request.method, path, body))
        parts = [p for p in path.split("/") if p]

        if parts[0] != "collections":
            return httpx.Response(404, json={"status": {"error": "not found"}})
        name = parts[1] if len(parts) > 1 else None
        tail = "/".join(parts[2:])

        if tail == "" and request.method == "GET":
            if name not in self.collections:
                return httpx.Response(404, json={"status": {"error": "not found"}})
            # Real Qdrant nests vector config under config.params -- not flat,
            # as an earlier version of this fake had it. See P113.
            return httpx.Response(200, json={"result": {
                "status": "green",
                "points_count": len(self.points.get(name, {})),
                "config": {"params": self.collections[name]},
            }})

        if tail == "" and request.method == "PUT":
            if name in self.collections:
                return httpx.Response(409, json={"status": {"error": "exists"}})
            self.collections[name] = body
            self.points[name] = {}
            return httpx.Response(200, json={"result": True})

        if name not in self.collections:
            return httpx.Response(404, json={"status": {"error": "not found"}})

        if tail == "points" and request.method == "PUT":
            for p in body["points"]:
                self.points[name][p["id"]] = {"vector": p["vector"], "payload": p["payload"]}
            return httpx.Response(200, json={"result": {"status": "completed"}})

        if tail == "points/query":
            return httpx.Response(200, json={"result": {"points": self._query(name, body)}})

        if tail == "points/count":
            matched = [p for p in self.points[name].values()
                       if _matches(p["payload"], body.get("filter"))]
            return httpx.Response(200, json={"result": {"count": len(matched)}})

        if tail == "points/delete":
            keep = {i: p for i, p in self.points[name].items()
                    if not _matches(p["payload"], body.get("filter"))}
            self.points[name] = keep
            return httpx.Response(200, json={"result": {"status": "completed"}})

        if tail == "points/scroll":
            items = [(i, p) for i, p in sorted(self.points[name].items(), key=lambda kv: str(kv[0]))
                     if _matches(p["payload"], body.get("filter"))]
            start = 0 if body.get("offset") is None else \
                next((n for n, (i, _) in enumerate(items) if str(i) == str(body["offset"])), 0)
            limit = body.get("limit", 100)
            page = items[start:start + limit]
            nxt = items[start + limit][0] if len(items) > start + limit else None
            return httpx.Response(200, json={"result": {
                "points": [{"id": i, "payload": p["payload"]} for i, p in page],
                "next_page_offset": nxt,
            }})

        return httpx.Response(404, json={"status": {"error": path}})

    def _query(self, name, body):
        using, query = body["using"], body["query"]
        scored = []
        for pid, point in self.points[name].items():
            if not _matches(point["payload"], body.get("filter")):
                continue
            vec = point["vector"].get(using)
            if vec is None:
                continue
            if using == "text":
                score = _sparse_dot(query, vec)
                if score <= 0:
                    continue  # real Qdrant omits sparse points with no term overlap
            else:
                score = _cosine(query, vec)
            scored.append({"id": pid, "score": score, "payload": point["payload"]})
        scored.sort(key=lambda p: p["score"], reverse=True)
        return scored[:body.get("limit", 10)]


def _lookup(payload, key):
    """Qdrant addresses nested payload fields with dots: "source.filename"."""
    value = payload
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _matches(payload, qf):
    if not qf:
        return True
    for cond in qf.get("must", []):
        value = _lookup(payload, cond["key"])
        if "range" in cond:
            # Qdrant range semantics: a missing field never matches.
            r = cond["range"]
            if value is None or ("gte" in r and value < r["gte"]) or ("lte" in r and value > r["lte"]) \
                    or ("gt" in r and value <= r["gt"]) or ("lt" in r and value >= r["lt"]):
                return False
            continue
        match = cond["match"]
        if "value" in match and value != match["value"]:
            return False
        if "any" in match and value not in match["any"]:
            return False
    return True


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _sparse_dot(a, b):
    bm = dict(zip(b["indices"], b["values"]))
    return sum(v * bm.get(i, 0.0) for i, v in zip(a["indices"], a["values"]))


def fake_qdrant_service(fake: FakeQdrant) -> QdrantService:
    svc = QdrantService(url="http://qdrant.test")
    svc._client = httpx.AsyncClient(base_url=svc.url, transport=fake.transport())
    return svc


def reset_singletons():
    embedding_service._provider = None
    qdrant_service._service = None
    reranker_service._reranker = None


# ── 1. embedding provider factory ───────────────────────

async def test_embedding_provider_factory():
    original = (config.EMBEDDING_PROVIDER, config.EMBEDDING_API_KEY, config.EMBEDDING_DIMENSIONS)
    try:
        config.EMBEDDING_API_KEY = "test-key"
        for name, cls in [("voyage", VoyageProvider), ("stub", StubProvider)]:
            reset_singletons()
            config.EMBEDDING_PROVIDER = name
            assert isinstance(create_embedding_provider(), cls), name

        # Not-yet-implemented providers say so instead of half-working.
        for name in ("openai", "jina", "cohere"):
            reset_singletons()
            config.EMBEDDING_PROVIDER = name
            try:
                create_embedding_provider()
                raise AssertionError(f"{name} should raise NotImplementedError")
            except NotImplementedError:
                pass

        reset_singletons()
        config.EMBEDDING_PROVIDER = "nonsense"
        try:
            create_embedding_provider()
            raise AssertionError("unknown provider must raise")
        except EmbeddingError:
            pass

        # No key -> stub fallback, so the service still boots pre-Voyage.
        reset_singletons()
        config.EMBEDDING_PROVIDER, config.EMBEDDING_API_KEY = "voyage", ""
        assert isinstance(create_embedding_provider(), StubProvider)
    finally:
        config.EMBEDDING_PROVIDER, config.EMBEDDING_API_KEY, config.EMBEDDING_DIMENSIONS = original
        reset_singletons()

    # Stub vectors: right size, unit length, deterministic, finite.
    stub = StubProvider(dimensions=8)
    a, b = await stub.embed(["service level agreement", "service level agreement"])
    (c,) = await stub.embed(["load balancer"])
    assert len(a) == 8 and a == b and a != c
    assert all(math.isfinite(v) for v in a)
    assert abs(math.sqrt(sum(v * v for v in a)) - 1.0) < 1e-6
    assert await stub.embed([]) == []


# ── 2. Voyage embedding request format ──────────────────

async def test_voyage_embedding():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.headers["authorization"] == "Bearer vk-test"
        assert body["model"] == "voyage-4-large" and body["output_dimension"] == 4
        # Deliberately out of order — the provider must re-sort by index.
        data = [{"index": i, "embedding": [float(i)] * 4}
                for i in reversed(range(len(body["input"])))]
        return httpx.Response(200, json={"data": data})

    provider = VoyageProvider(api_key="vk-test", model="voyage-4-large", dimensions=4)
    assert provider.dimensions == 4

    real_client = httpx.AsyncClient

    def patched(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real_client(*a, **kw)

    httpx.AsyncClient = patched
    try:
        vectors = await provider.embed(["a", "b", "c"], input_type="query")
        assert vectors == [[0.0] * 4, [1.0] * 4, [2.0] * 4], vectors
        assert calls[0]["input_type"] == "query"

        # Batching: 300 texts must go out as 3 calls of <= BATCH_SIZE.
        calls.clear()
        await provider.embed([f"t{i}" for i in range(300)])
        assert len(calls) == 3, len(calls)
        assert all(len(c["input"]) <= embedding_service.BATCH_SIZE for c in calls)
        assert sum(len(c["input"]) for c in calls) == 300

        # Long input is truncated, not rejected.
        calls.clear()
        await provider.embed(["x" * (embedding_service.MAX_CHARS + 5000)])
        assert len(calls[0]["input"][0]) == embedding_service.MAX_CHARS
    finally:
        httpx.AsyncClient = real_client

    # A dead API raises rather than returning nothing.
    def boom(request):
        return httpx.Response(401, text="bad key")

    dead = VoyageProvider(api_key="x", dimensions=4)
    httpx.AsyncClient = lambda *a, **kw: real_client(*a, **{**kw, "transport": httpx.MockTransport(boom)})
    try:
        await dead.embed(["a"])
        raise AssertionError("401 must raise")
    except EmbeddingError as exc:
        assert "401" in str(exc)
    finally:
        httpx.AsyncClient = real_client


async def test_upsert_splits_requests_under_the_qdrant_size_cap():
    """A 1h+ video yields thousands of 1024-d keyframe points (~20 KB of JSON
    each); one PUT of all of them exceeds Qdrant's 32 MB request cap and the
    index phase fails after every paid phase already ran."""
    fake = FakeQdrant()
    fake.max_request_bytes = 2 * 1024 * 1024   # scaled down to keep the test fast
    svc = fake_qdrant_service(fake)
    svc.upsert_max_bytes = 1024 * 1024        # the service's budget stays under the cap
    await svc.ensure_collection("video_clip", 1024, enable_sparse=False)

    n = 400   # ~400 x 20 KB = ~8 MB, four times the scaled cap
    vectors = [[0.123456789 + i * 1e-9] * 1024 for i in range(n)]
    await svc.upsert("video_clip", [f"kf-{i}" for i in range(n)], vectors,
                     [{"content": "", "i": i} for i in range(n)],
                     sparse_vectors=[{} for _ in range(n)])

    assert len(fake.points["video_clip"]) == n
    puts = [b for m, p, b in fake.seen if m == "PUT" and p.endswith("/points")]
    assert len(puts) > 1, "expected the upsert to be split into several requests"


# ── 3. collection creation ──────────────────────────────

async def test_qdrant_collection_creation():
    fake = FakeQdrant()
    svc = fake_qdrant_service(fake)

    await svc.ensure_collection("documents", 1024)
    body = fake.collections["documents"]
    assert body["vectors"]["dense"] == {"size": 1024, "distance": "Cosine"}
    assert body["sparse_vectors"]["text"] == {"modifier": "idf"}

    # Idempotent: a second call must not error and must not re-PUT.
    puts = sum(1 for m, p, _ in fake.seen if m == "PUT" and p.endswith("documents"))
    await svc.ensure_collection("documents", 1024)
    assert sum(1 for m, p, _ in fake.seen if m == "PUT" and p.endswith("documents")) == puts

    await svc.ensure_collection("dense_only", 8, enable_sparse=False)
    assert "sparse_vectors" not in fake.collections["dense_only"]

    info = await svc.collection_info("documents")
    assert info["points_count"] == 0 and info["status"] == "green"
    assert await svc.collection_info("never_created") is None
    await svc.aclose()


async def test_ensure_collection_rejects_an_existing_schema_mismatch():
    # P113: ensure_collection() used to return as soon as a collection
    # existed, with no check that its vector schema actually matched the
    # caller's expectation. A collection left over with the wrong named
    # vector (or wrong dimension) let every later upsert fail far away and
    # much later, with an opaque Qdrant 400 ("Not existing vector name
    # error: dense") that named the symptom, not the cause -- this is
    # exactly what happened live on 2026-10-12 to visual_frames.
    fake = FakeQdrant()
    svc = fake_qdrant_service(fake)

    # Simulate a stale/wrong-schema collection: created directly in the fake
    # (bypassing ensure_collection) with a vector named "image", not "dense".
    fake.collections["visual_frames"] = {"vectors": {"image": {"size": 1024, "distance": "Cosine"}}}
    fake.points["visual_frames"] = {}

    try:
        await svc.ensure_collection("visual_frames", 1024)
        raise AssertionError("expected QdrantError for the wrong vector name")
    except qdrant_service.QdrantError as exc:
        assert "visual_frames" in str(exc) and "dense" in str(exc), exc

    # Same video_id collection, this time with "dense" present but the wrong size.
    fake.collections["visual_frames_wrong_size"] = {"vectors": {"dense": {"size": 512, "distance": "Cosine"}}}
    fake.points["visual_frames_wrong_size"] = {}
    try:
        await svc.ensure_collection("visual_frames_wrong_size", 1024)
        raise AssertionError("expected QdrantError for the wrong dimension")
    except qdrant_service.QdrantError as exc:
        assert "1024" in str(exc) and "512" in str(exc), exc

    # A collection whose schema DOES match must still be a silent no-op.
    await svc.ensure_collection("documents", 1024)  # created correctly above
    await svc.ensure_collection("documents", 1024)  # no raise
    await svc.aclose()


# ── 4 + 5. dense and sparse search ──────────────────────

DOCS = [
    ("c1", [1.0, 0.0, 0.0], {"content": "A service level agreement SLA is an uptime commitment in service",
                             "source": "Acme Session 12", "source_type": "transcript",
                             "timestamp": "00:14:32"}),
    ("c2", [0.9, 0.1, 0.0], {"content": "Load balancer LB spreads incoming traffic",
                             "source": "Acme Session 13", "source_type": "transcript"}),
    ("c3", [0.0, 0.0, 1.0], {"content": "plot() draws a series on the chart",
                             "source": "sdk-reference", "source_type": "code"}),
]


async def seeded_qdrant():
    fake = FakeQdrant()
    svc = fake_qdrant_service(fake)
    await svc.ensure_collection("kb", 3)
    await svc.upsert("kb", [d[0] for d in DOCS], [d[1] for d in DOCS], [d[2] for d in DOCS])
    return fake, svc


async def test_dense_search():
    fake, svc = await seeded_qdrant()

    # Point ids are UUID5-mapped, and sparse vectors are derived from content.
    stored = list(fake.points["kb"].values())
    assert all("dense" in p["vector"] and "text" in p["vector"] for p in stored)
    assert len(fake.points["kb"]) == 3

    hits = await svc.search_dense("kb", [1.0, 0.0, 0.0], top_k=3)
    assert [h["payload"]["source"] for h in hits] == ["Acme Session 12", "Acme Session 13",
                                                      "sdk-reference"]
    assert hits[0]["score"] > hits[1]["score"] > hits[2]["score"]
    assert hits[0]["collection"] == "kb"
    assert hits[0]["payload"]["timestamp"] == "00:14:32", "video metadata must survive"

    assert len(await svc.search_dense("kb", [1.0, 0.0, 0.0], top_k=1)) == 1
    await svc.aclose()


async def test_sparse_search():
    _, svc = await seeded_qdrant()

    # Single-word jargon query: BM25 has to carry this, dense cannot.
    hits = await svc.search_sparse("kb", "SLA", top_k=5)
    assert len(hits) == 1 and hits[0]["payload"]["source"] == "Acme Session 12", hits

    hits = await svc.search_sparse("kb", "load balancer", top_k=5)
    assert hits[0]["payload"]["source"] == "Acme Session 13"

    # A query with no usable tokens returns nothing rather than exploding.
    assert await svc.search_sparse("kb", "!!! ???") == []

    # Tokeniser is case- and punctuation-insensitive on both sides.
    q = text_to_sparse("SLA, imbalance.")
    d = text_to_sparse("a service level agreement sla is an imbalance in price")
    assert set(q["indices"]) & set(d["indices"]), "query tokens must overlap the document"
    assert text_to_sparse("")["indices"] == []
    assert text_to_sparse("the the the")["values"] == [3.0], "term frequency must be counted"
    await svc.aclose()


# ── 6. RRF ──────────────────────────────────────────────

def test_rrf_fusion():
    dense = [{"id": "a", "collection": "kb"}, {"id": "b", "collection": "kb"},
             {"id": "c", "collection": "kb"}]
    sparse = [{"id": "c", "collection": "kb"}, {"id": "b", "collection": "kb"},
              {"id": "d", "collection": "kb"}]

    fused = reciprocal_rank_fusion([dense, sparse])
    ids = [f["id"] for f in fused]
    assert set(ids) == {"a", "b", "c", "d"}, ids

    # Exact RRF math: c = 1/63 + 1/61, b = 2/62, a = 1/61, d = 1/63.
    scores = {f["id"]: f["rrf_score"] for f in fused}
    for key, expected in [("a", 1 / 61), ("b", 2 / 62), ("c", 1 / 63 + 1 / 61), ("d", 1 / 63)]:
        assert abs(scores[key] - expected) < 1e-12, (key, scores[key], expected)
    assert ids == ["c", "b", "a", "d"], ids
    # The property that matters: appearing in both lists beats appearing in one.
    assert scores["b"] > scores["a"] > scores["d"]

    # Same id in different collections must stay distinct.
    two = reciprocal_rank_fusion([[{"id": "x", "collection": "a"}],
                                  [{"id": "x", "collection": "b"}]])
    assert len(two) == 2

    assert reciprocal_rank_fusion([[], []]) == []
    # Payload of the first list that produced the doc is preserved.
    kept = reciprocal_rank_fusion([[{"id": "a", "collection": "kb", "payload": {"content": "hi"}}]])
    assert kept[0]["payload"]["content"] == "hi"


# ── 7. reranker ─────────────────────────────────────────

async def test_reranker():
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "rerank-2.5" and body["query"] == "what is an SLA"
        assert body["top_k"] == 2
        # Reverse the incoming order — the pipeline must honour this, not RRF.
        n = len(body["documents"])
        # Atlas/Voyage envelope: {"object": "list", "data": [...], "model", "usage"}
        return httpx.Response(200, json={"object": "list", "data": [
            {"index": n - 1 - i, "relevance_score": 1.0 - i * 0.1} for i in range(body["top_k"])
        ]})

    real_client = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **kw: real_client(*a, **{**kw, "transport": httpx.MockTransport(handler)})
    try:
        ranked = await VoyageReranker(api_key="k", model="rerank-2.5").rerank(
            "what is an SLA", ["doc0", "doc1", "doc2"], top_n=2)
        assert ranked == [(2, 1.0), (1, 0.9)], ranked
    finally:
        httpx.AsyncClient = real_client

    assert await VoyageReranker(api_key="k").rerank("q", [], top_n=5) == []

    # Noop keeps input order and never asks for more than it was given.
    noop = await NoopReranker().rerank("q", ["a", "b"], top_n=5)
    assert [i for i, _ in noop] == [0, 1]

    original = (config.RERANKER_PROVIDER, config.RERANKER_API_KEY)
    try:
        reset_singletons()
        config.RERANKER_PROVIDER, config.RERANKER_API_KEY = "voyage", ""
        assert isinstance(create_reranker(), NoopReranker), "no key must disable, not crash"
    finally:
        config.RERANKER_PROVIDER, config.RERANKER_API_KEY = original
        reset_singletons()


async def test_rerank_reads_the_right_text_field_per_collection_shape():
    """_rerank() used to always read payload["content"] -- but M3's video
    payloads never use that key: transcript segments store text under
    "text", OCR frames under "ocr_text", OD/vision-description frames
    under "od_detections" (see pipelines/video_ingestion.py's _phase_index).
    Every video candidate reranked as an empty string, and Voyage's rerank
    API rejects a batch containing one -- the whole rerank call failed and
    silently fell back to unranked RRF order for every video search. Found
    live 2026-10-12, first real retrieval test against real M3 content.
    """
    captured = {}

    class CapturingReranker:
        calibrated = True

        async def rerank(self, query, documents, top_n=5):
            captured["documents"] = documents
            return [(i, 1.0) for i in range(len(documents))]

    candidates = [
        {"payload": {"content": "an M1/M2 document chunk"}, "rrf_score": 0.1},
        {"payload": {"text": "an ASR transcript segment", "source_type": "transcript_segment"},
         "rrf_score": 0.1},
        {"payload": {"ocr_text": "SLA marked on chart", "source_type": "ocr_frame"},
         "rrf_score": 0.1},
        {"payload": {"od_detections": "a bearish load balancer", "source_type": "od_frame"},
         "rrf_score": 0.1},
        # visual_frames has no text field at all -- pure image similarity.
        # Started actually reaching this function 2026-10-12 once
        # MomentSearchRetriever got wired into search(); without a fallback
        # this reranks as "", and Voyage's rerank API rejects a batch
        # containing even one empty string (same failure mode as the other
        # four, P99), breaking reranking for the WHOLE query again.
        {"payload": {"title": "Acme Mentorship 2024", "source_type": "visual_frame"},
         "rrf_score": 0.1},
    ]
    saved_reranker = reranker_service._reranker
    reranker_service._reranker = CapturingReranker()
    try:
        await retrieval._rerank("query", candidates, top_n=5)
    finally:
        reranker_service._reranker = saved_reranker

    assert captured["documents"] == [
        "an M1/M2 document chunk",
        "an ASR transcript segment",
        "SLA marked on chart",
        "a bearish load balancer",
        "Acme Mentorship 2024",
    ], captured["documents"]


# ── 8 + 9 + 10. endpoint, empty KB, filters ─────────────

async def with_stub_pipeline(fake: FakeQdrant):
    """Point the pipeline at a fake Qdrant + stub embedder + noop reranker."""
    reset_singletons()
    embedding_service._provider = StubProvider(dimensions=8)
    qdrant_service._service = fake_qdrant_service(fake)
    reranker_service._reranker = NoopReranker()
    return qdrant_service._service


async def test_search_endpoint_integration():
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    embedder = embedding_service._provider

    texts = [d[2]["content"] for d in DOCS]
    vectors = await embedder.embed(texts)
    await svc.ensure_collection("documents", 8)
    await svc.upsert("documents", [d[0] for d in DOCS], vectors, [d[2] for d in DOCS])

    resp = await search_knowledge(q=DOCS[0][2]["content"], top_k=2,
                                  collections="documents", source_type=None)
    assert isinstance(resp, SearchResponse)
    assert resp.query == DOCS[0][2]["content"]
    assert resp.collections_searched == ["documents"]
    assert 0 < len(resp.results) <= 2

    top = resp.results[0]
    # Stub embeddings are exact-match only, so the identical text must come first.
    assert top.source == "Acme Session 12", top
    assert top.source_type == "transcript"
    assert isinstance(top.score, float)
    assert top.metadata["timestamp"] == "00:14:32", "citations need video timestamps"
    assert top.metadata["collection"] == "documents"
    assert "content" not in top.metadata, "promoted fields must not be duplicated"

    # Response must serialise for the Open WebUI filter.
    assert json.loads(resp.model_dump_json())["results"][0]["source"] == "Acme Session 12"

    # Cross-collection: a missing second collection must not sink the first.
    resp = await search_knowledge(q="service level agreement", top_k=5,
                                  collections="documents,code_docs",
                                  source_type=None)
    assert resp.collections_searched == ["documents", "code_docs"]
    assert len(resp.results) >= 1
    reset_singletons()


async def test_empty_collection_graceful():
    fake = FakeQdrant()
    await with_stub_pipeline(fake)

    resp = await search_knowledge(q="SLA", top_k=5, collections=None, source_type=None)
    assert resp.results == []
    # No source_type hint now merges text + video (resolve_collections' new
    # default, 2026-10-12) rather than text-only.
    assert resp.collections_searched == retrieval.ALL_COLLECTIONS
    assert await retrieval.search("anything") == []
    reset_singletons()


async def test_search_with_filters():
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    embedder = embedding_service._provider

    texts = [d[2]["content"] for d in DOCS]
    await svc.ensure_collection("documents", 8)
    await svc.upsert("documents", [d[0] for d in DOCS],
                     await embedder.embed(texts), [d[2] for d in DOCS])

    resp = await search_knowledge(q="chart", top_k=10, collections="documents",
                                  source_type="code")
    assert resp.results and all(r.source_type == "code" for r in resp.results), resp.results

    resp = await search_knowledge(q="chart", top_k=10, collections="documents",
                                  source_type="transcript")
    assert resp.results and all(r.source_type == "transcript" for r in resp.results)

    assert build_filter(None) is None and build_filter({"a": None}) is None
    assert build_filter({"a": 1}) == {"must": [{"key": "a", "match": {"value": 1}}]}
    assert build_filter({"a": ["x", "y"]}) == {"must": [{"key": "a", "match": {"any": ["x", "y"]}}]}

    # delete_by_filter reports what it removed and refuses an empty filter.
    assert await svc.delete_by_filter("documents", {"source_type": "code"}) == 1
    assert await svc.delete_by_filter("documents", {"source_type": "code"}) == 0
    assert await svc.delete_by_filter("no_such_collection", {"source_type": "code"}) == 0
    try:
        await svc.delete_by_filter("documents", {})
        raise AssertionError("empty filter must be refused")
    except qdrant_service.QdrantError:
        pass

    points, nxt = await svc.scroll("documents", limit=1)
    assert len(points) == 1 and nxt is not None
    rest, nxt = await svc.scroll("documents", limit=10, offset=nxt)
    assert len(rest) == 1 and nxt is None
    assert await svc.scroll("no_such_collection") == ([], None)
    reset_singletons()


# ── 11-15. topic expansion ──────────────────────────────

# One 5-chunk topic in the new namespaced shape, one 3-chunk topic still in the
# old flat shape (mid-backfill), and a source that is a single chunk.
TOPIC_DOCS = (
    [("lb%d" % i, {"content": "load balancer part %d" % i, "source_type": "document",
                   "chunk_index": i, "source": {"filename": "traffic-load-balancers.md",
                                                "title": "Load Balancers"}})
     for i in range(5)]
    + [("sla%d" % i, {"content": "sla part %d" % i, "source_type": "document",
                      "chunk_index": i, "source": "service-level-agreements.md"})
       for i in range(3)]
    + [("solo", {"content": "one and only chunk", "source_type": "document",
                 "chunk_index": 0, "source": {"filename": "solo.md"}})]
)


async def seeded_topics(collection: str = "kb"):
    """Fake Qdrant holding TOPIC_DOCS, wired in as the process-wide service."""
    reset_singletons()
    fake = FakeQdrant()
    svc = fake_qdrant_service(fake)
    qdrant_service._service = svc
    await svc.ensure_collection(collection, 3)
    await svc.upsert(collection, [d[0] for d in TOPIC_DOCS],
                     [[1.0, 0.0, 0.0]] * len(TOPIC_DOCS), [d[1] for d in TOPIC_DOCS])
    return fake, svc


def hit(chunk_id: str, score: float = 0.9, collection: str = "kb") -> dict:
    payload = next(p for i, p in TOPIC_DOCS if i == chunk_id)
    return {"id": qdrant_service.point_id(chunk_id), "score": score,
            "payload": dict(payload), "collection": collection}


async def test_expand_by_source_finds_siblings():
    _, svc = await seeded_topics()

    chunks = await svc.expand_by_source("kb", ["traffic-load-balancers.md"])
    assert [c["payload"]["chunk_index"] for c in chunks] == [0, 1, 2, 3, 4], chunks
    assert all(c["collection"] == "kb" for c in chunks)

    # Old flat payloads must expand too — the backfill is still rolling, and a
    # namespaced hit in the same query must not shadow them.
    both = await svc.expand_by_source("kb", ["traffic-load-balancers.md", "service-level-agreements.md"])
    assert [qdrant_service.source_of(c["payload"]) for c in both] ==         ["service-level-agreements.md"] * 3 + ["traffic-load-balancers.md"] * 5, both

    assert await svc.expand_by_source("kb", []) == []
    assert await svc.expand_by_source("kb", ["nothing.md"]) == []
    assert await svc.expand_by_source("no_such_collection", ["traffic-load-balancers.md"]) == []
    reset_singletons()


async def test_expansion_deduplicates_primaries():
    await seeded_topics()
    primary = hit("lb2")

    expanded = await expand_results([primary])
    ids = [c["id"] for c in expanded]
    assert len(ids) == len(set(ids)) == 5, expanded
    assert [c["payload"]["chunk_index"] for c in expanded] == [0, 1, 2, 3, 4]

    flags = [c["payload"]["is_primary"] for c in expanded]
    assert flags == [False, False, True, False, False], flags
    assert expanded[2]["score"] == 0.9
    # Siblings inherit the primary's score: rag_filter drops anything under
    # min_relevance_score (0.3) before the context block is built.
    assert all(c["score"] == 0.9 for c in expanded)
    # The caller's chunk is not mutated into a primary behind its back.
    assert "is_primary" not in primary["payload"]

    # Two winners from the same source produce one group, not two.
    expanded = await expand_results([hit("lb0"), hit("lb4", score=0.5)])
    assert len(expanded) == 5
    assert sum(c["payload"]["is_primary"] for c in expanded) == 2
    reset_singletons()


async def test_expansion_respects_max_expansion():
    await seeded_topics()
    primaries = [hit("lb0"), hit("sla1", score=0.4)]

    assert len(await expand_results(primaries)) == 8

    capped = await expand_results(primaries, max_expansion=4)
    assert len(capped) == 4, capped
    # Both winners survive; the two free slots go to the nearest siblings.
    assert sum(c["payload"]["is_primary"] for c in capped) == 2
    assert [(qdrant_service.source_of(c["payload"]), c["payload"]["chunk_index"])
            for c in capped] == [("traffic-load-balancers.md", 0), ("traffic-load-balancers.md", 1),
                                 ("service-level-agreements.md", 0), ("service-level-agreements.md", 1)], capped

    # No room for siblings at all still returns every winner.
    assert len(await expand_results(primaries, max_expansion=2)) == 2
    assert len(await expand_results(primaries, max_expansion=1)) == 1
    reset_singletons()


async def test_expansion_with_no_siblings():
    await seeded_topics()

    only = await expand_results([hit("solo")])
    assert len(only) == 1 and only[0]["payload"]["is_primary"] is True

    assert await expand_results([]) == []
    # A chunk with no source at all cannot be expanded, and must not be dropped.
    orphan = {"id": "x", "score": 0.7, "payload": {"content": "no source"},
              "collection": "kb"}
    assert await expand_results([orphan]) == [
        {**orphan, "payload": {"content": "no source", "is_primary": True}}]
    reset_singletons()


async def test_expansion_config_flag():
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    embedder = embedding_service._provider

    await svc.ensure_collection("kb", 8)
    payloads = [d[1] for d in TOPIC_DOCS]
    await svc.upsert("kb", [d[0] for d in TOPIC_DOCS],
                     await embedder.embed([p["content"] for p in payloads]), payloads)

    original = config.ENABLE_TOPIC_EXPANSION
    try:
        config.ENABLE_TOPIC_EXPANSION = False
        plain = await retrieval.search("load balancer part 2", top_n=1, collections=["kb"])
        assert len(plain) == 1, plain
        assert "is_primary" not in plain[0]["payload"]

        config.ENABLE_TOPIC_EXPANSION = True
        wide = await retrieval.search("load balancer part 2", top_n=1, collections=["kb"])
        assert len(wide) == 5, wide
        assert sum(c["payload"]["is_primary"] for c in wide) == 1

        # And the same through the endpoint, where is_primary rides in metadata.
        resp = await search_knowledge(q="load balancer part 2", top_k=1,
                                      collections="kb", source_type=None)
        assert len(resp.results) == 5
        assert sum(r.metadata["is_primary"] for r in resp.results) == 1
        assert all(r.score > 0 for r in resp.results), "siblings must clear the filter"
    finally:
        config.ENABLE_TOPIC_EXPANSION = original
        reset_singletons()


# ── 20. telemetry_compat safety contract ────────────────

def test_telemetry_compat_never_breaks_retrieval():
    """The property that matters most: retrieval must behave identically whether
    or not the shared telemetry module is reachable (processing/telemetry/ is
    outside haystack-api's build context until the described Dockerfile change
    ships — P37/D48-51, rag-AI/CLAUDE.md). This pins the no-op contract
    directly, independent of whether opentelemetry happens to be installed in
    whatever environment runs this suite; every other test in this file
    exercises retrieval.search() through whichever path telemetry_compat picks,
    and all of them passing is the regression proof that instrumentation didn't
    change behaviour.
    """
    with telemetry_compat.span("t", telemetry_compat.SpanKind.RETRIEVER) as sp:
        sp.set_attribute("k", "v")  # must not raise, real span or not
        telemetry_compat.record_decision(sp, "retrieval", "answered", "reason with a number 0.70")
        telemetry_compat.set_output(sp, "3 results")

    # init_tracing must never raise even against an unreachable endpoint, and
    # returns None rather than a provider when the module isn't importable.
    provider = telemetry_compat.init_tracing("haystack-api-test", endpoint="http://127.0.0.1:1")
    if not telemetry_compat._AVAILABLE:
        assert provider is None


# ── live Qdrant (optional) ──────────────────────────────

async def test_live_qdrant():
    """Runs only when QDRANT_URL answers. This is the one that catches a REST
    body real Qdrant rejects but FakeQdrant happily accepts."""
    url = os.getenv("QDRANT_URL", config.QDRANT_URL)
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            await c.get(f"{url.rstrip('/')}/collections")
    except Exception:
        print(f"  live Qdrant at {url} not reachable — skipped")
        return

    svc = QdrantService(url=url)
    name = "haystack_retrieval_selftest"
    try:
        await svc.delete_by_filter(name, {"source_type": "selftest"})
        await svc.ensure_collection(name, 8)
        embedder = StubProvider(dimensions=8)
        payloads = [{"content": c, "source": s, "source_type": "selftest"}
                    for c, s in [("service level agreement SLA imbalance", "live-1"),
                                 ("plot a series on the chart", "live-2")]]
        vectors = await embedder.embed([p["content"] for p in payloads])
        await svc.upsert(name, ["live-a", "live-b"], vectors, payloads)

        dense = await svc.search_dense(name, vectors[0], top_k=2)
        assert dense and dense[0]["payload"]["source"] == "live-1", dense
        sparse = await svc.search_sparse(name, "SLA", top_k=2)
        assert sparse and sparse[0]["payload"]["source"] == "live-1", sparse
        filtered = await svc.search_dense(name, vectors[0], top_k=2,
                                          filters={"source_type": "nope"})
        assert filtered == []
        assert await svc.delete_by_filter(name, {"source_type": "selftest"}) == 2
        print("  live Qdrant OK")
    finally:
        await svc.aclose()


# ── 16 + 17. relevance floor ────────────────────────────

class FakeReranker:
    """A reranker returning calibrated relevance scores, like Voyage/Cohere do."""

    calibrated = True

    def __init__(self, scores):
        self.scores = scores          # score keyed by chunk content

    async def rerank(self, query, documents, top_n=5):
        ranked = sorted(((i, self.scores.get(d, 0.0)) for i, d in enumerate(documents)),
                        key=lambda p: p[1], reverse=True)
        return ranked[:top_n]


IRRELEVANT_DOCS = [
    ("p1", {"content": "Functions using history referencing must be called on every bar",
            "source": "Execution model", "source_type": "code_docs"}),
    ("p2", {"content": "Wormholes connect distant regions of spacetime",
            "source": "Wormhole.pdf", "source_type": "document"}),
    ("p3", {"content": "Pakistan GDP growth by sector 2019",
            "source": "Pakistan economic development.pdf", "source_type": "document"}),
]


async def _seed_mixed(fake: FakeQdrant):
    """One relevant chunk and two off-topic ones in the same collection."""
    svc = await with_stub_pipeline(fake)
    embedder = embedding_service._provider
    texts = [d[1]["content"] for d in IRRELEVANT_DOCS]
    vectors = await embedder.embed(texts)
    await svc.ensure_collection("documents", 8)
    await svc.upsert("documents", [d[0] for d in IRRELEVANT_DOCS], vectors,
                     [d[1] for d in IRRELEVANT_DOCS])
    return svc


async def test_low_relevance_results_are_dropped():
    """RERANK_TOP_N is a ceiling, not a quota.

    Reported live: a code-reference answer citing three unrelated PDFs and a
    CSV. All three were rerank losers that /search returned
    anyway, and the Sources block cites whatever /search returns.
    """
    fake = FakeQdrant()
    await _seed_mixed(fake)
    reranker_service._reranker = FakeReranker({
        IRRELEVANT_DOCS[0][1]["content"]: 0.91,
        IRRELEVANT_DOCS[1][1]["content"]: 0.02,
        IRRELEVANT_DOCS[2][1]["content"]: 0.01,
    })

    resp = await search_knowledge(q="do functions have to be called on every bar",
                                  top_k=5, collections="documents",
                                  source_type=None)

    sources = {r.source for r in resp.results}
    assert sources == {"Execution model"}, sources
    assert "Wormhole" not in resp.sources, resp.sources
    assert "Pakistan" not in resp.sources, resp.sources
    reset_singletons()


async def test_relevance_floor_only_applies_to_calibrated_scores():
    """A reranker outage must not empty the knowledge base.

    The RRF fallback scores ~1/61 per list — below any relevance threshold. Gating
    on those would turn a reranker outage into "nothing is in the knowledge base"
    for every query, which is the opposite of degrading gracefully.
    """
    fake = FakeQdrant()
    await _seed_mixed(fake)

    class DeadReranker:
        calibrated = True

        async def rerank(self, query, documents, top_n=5):
            raise RuntimeError("502 from rerank endpoint")

    reranker_service._reranker = DeadReranker()
    resp = await search_knowledge(q="history referencing", top_k=5,
                                  collections="documents", source_type=None)
    assert resp.results, "reranker outage must fall back to RRF order, not to nothing"

    # Same for the no-API-key path: NoopReranker's scores are placement, not relevance.
    reranker_service._reranker = NoopReranker()
    resp = await search_knowledge(q="history referencing", top_k=5,
                                  collections="documents", source_type=None)
    assert resp.results, "NoopReranker scores are not relevance — must not be gated"
    reset_singletons()


# ── 18 + 19. where the floor belongs ────────────────────

# Scores below are real Voyage rerank-2.5 output against this corpus, measured on
# the VPS 2026-09-08 once RERANKER_API_KEY was finally set. They are the whole
# argument for the threshold, so they are written down rather than rounded away:
#
#   on-topic   'do functions have to be called on every bar'  0.8750 .. 0.8125
#   on-topic   'is the current bar within regular hours?'     0.8789 .. 0.7695
#   off-topic  'hi'                                           0.6289 .. 0.5469
#   off-topic  'what is the weather today'                    0.3789 .. 0.3066
#   off-topic  'random fun fact about the Roman Empire'       0.3047 .. 0.3008
#
# Lowest true positive 0.7695, highest false positive 0.6289. The band between
# them is empty; MIN_RERANK_SCORE sits in the middle of it.
WORST_TRUE_POSITIVE = 0.7695
BEST_FALSE_POSITIVE = 0.6289


async def test_off_topic_query_drops_the_whole_noise_band():
    """A greeting must not cite the manual.

    The first floor was 0.30, taken from Voyage's published example where
    irrelevant documents scored ~0.26. This corpus does not behave that way: 'hi'
    scored three reference pages 0.6289/0.6055/0.5625, all of which cleared 0.30
    and got cited. The floor has to clear the noise band this reranker actually
    produces, not the one the docs describe.
    """
    fake = FakeQdrant()
    await _seed_mixed(fake)
    reranker_service._reranker = FakeReranker({
        IRRELEVANT_DOCS[0][1]["content"]: BEST_FALSE_POSITIVE,
        IRRELEVANT_DOCS[1][1]["content"]: 0.6055,
        IRRELEVANT_DOCS[2][1]["content"]: 0.5625,
    })

    resp = await search_knowledge(q="hi", top_k=5, collections="documents",
                                  source_type=None)

    assert resp.results == [], [(r.score, r.source) for r in resp.results]
    assert resp.sources == "", resp.sources
    reset_singletons()


async def test_weakest_real_match_survives_the_floor():
    """The floor must not eat genuine answers.

    0.7695 is the lowest score a correct match earned in production, for 'How do I
    check if the current record is within regular business hours?'. A floor above it
    turns a working query into "nothing is in the knowledge base" — the same
    failure the floor exists to prevent, pointing the other way.
    """
    fake = FakeQdrant()
    await _seed_mixed(fake)
    reranker_service._reranker = FakeReranker({
        IRRELEVANT_DOCS[0][1]["content"]: WORST_TRUE_POSITIVE,
        IRRELEVANT_DOCS[1][1]["content"]: 0.02,
        IRRELEVANT_DOCS[2][1]["content"]: 0.01,
    })

    resp = await search_knowledge(q="is the current record within regular business hours",
                                  top_k=5, collections="documents",
                                  source_type=None)

    assert {r.source for r in resp.results} == {"Execution model"},         [(r.score, r.source) for r in resp.results]
    reset_singletons()


# ── 20 + 21. does the floor actually RUN? (P12) ─────────

# Every floor test above installs its reranker by assigning
# reranker_service._reranker directly. That proves _drop_irrelevant() works; it
# cannot prove the floor ever executes in production, and P12 is exactly the case
# where it did not: the floor shipped, deployed, and never once ran, because
# RERANKER_API_KEY was unset, create_reranker() fell back to NoopReranker, and
# NoopReranker.calibrated is False by design. Nothing failed. Nothing logged an
# error. Off-topic chunks were cited for as long as that held.
#
# So the untested link is the FACTORY, not the floor: "is the reranker this
# deployment actually builds one whose scores may be thresholded at all?" These two
# tests pin both answers, and the second one pins the silent half — the condition
# under which the floor is skipped is a configuration value, so it can come back.

async def test_configured_reranker_is_calibrated_so_the_floor_can_fire():
    """With a key set, create_reranker() must return a provider the floor trusts.

    calibrated=False is what silently disables the floor, and it is read off the
    class, so a new provider added without the attribute would inherit
    RerankerProvider.calibrated=True and be thresholded on whatever it returns —
    which is why this asserts the value per provider rather than just truthiness.
    """
    original = (config.RERANKER_PROVIDER, config.RERANKER_API_KEY)
    try:
        for name, expected in [("voyage", True), ("cohere", True), ("none", False)]:
            reranker_service._reranker = None
            config.RERANKER_PROVIDER = name
            config.RERANKER_API_KEY = "test-key"
            built = create_reranker()
            assert built.calibrated is expected, (name, built, built.calibrated)
    finally:
        config.RERANKER_PROVIDER, config.RERANKER_API_KEY = original
        reset_singletons()


async def test_missing_reranker_key_silently_disables_the_floor():
    """The P12 condition itself, asserted rather than remembered.

    This is not a bug to fix here — falling back to RRF order beats emptying the
    knowledge base on every query (test 17 above is the other half of that
    argument). It is a bug to be ABLE TO SEE: the same off-topic chunk that
    test 18 proves is dropped is cited in full when the key is missing, and the
    only difference between the two runs is an environment variable. Whether it is
    set on the VPS right now is a deploy-time fact, not a code fact — Phase 0's
    verify_deploy.sh is where that belongs.
    """
    original = (config.RERANKER_PROVIDER, config.RERANKER_API_KEY)
    try:
        config.RERANKER_PROVIDER = "voyage"
        config.RERANKER_API_KEY = ""
        reranker_service._reranker = None
        assert isinstance(create_reranker(), NoopReranker)
        assert create_reranker().calibrated is False

        # ...and the consequence, end to end: the noise band test 18 drops is kept.
        fake = FakeQdrant()
        await _seed_mixed(fake)
        reranker_service._reranker = None       # force the factory path
        resp = await search_knowledge(q="hi", top_k=5,
                                      collections="documents", source_type=None)
        assert resp.results, (
            "uncalibrated reranker must still answer — an empty result here would "
            "mean a missing key empties the knowledge base"
        )
        assert any("Wormhole" in (r.source or "") for r in resp.results), (
            "the floor is expected NOT to run on an uncalibrated reranker; if this "
            "fails, MIN_RERANK_SCORE is being applied to placement scores"
        )
    finally:
        config.RERANKER_PROVIDER, config.RERANKER_API_KEY = original
        reset_singletons()


# ── comparison-query rewrite (C9) ────────────────────────────
def test_comparison_questions_rewrite_to_keyword_terms():
    """A comparison question must rewrite; an ordinary question must not.

    The second half is the load-bearing half. This fallback exists to add
    recall on a query that retrieved nothing, so any query it touches that
    ALREADY worked would be a regression - `comparison_terms` returning None
    is what guarantees search() leaves those completely alone.
    """
    from pipelines.retrieval import comparison_terms

    assert comparison_terms(
        "what is the difference between strategy.entry and strategy.order?"
    ) == "strategy.entry strategy.order difference"
    assert comparison_terms(
        "What are the differences between ta.sma and ta.ema?"
    ) == "ta.sma ta.ema difference"
    assert comparison_terms("strategy.entry vs strategy.order") ==         "strategy.entry strategy.order difference"
    assert comparison_terms(
        "compare request.security with request.security_lower_tf"
    ) == "request.security request.security_lower_tf difference"

    for ordinary in (
        "what does request.security() do in the SDK?",
        "how do I plot a moving average on a chart in version 6?",
        "what is a service level agreement?",
        "analyse a 9/21 moving-average crossover on this file with a 2% threshold",
    ):
        assert comparison_terms(ordinary) is None, ordinary
    print("ok comparison_questions_rewrite_to_keyword_terms")


def test_resolve_collections():
    from pipelines.retrieval import (
        ALL_COLLECTIONS, DEFAULT_COLLECTIONS, VIDEO_COLLECTIONS, resolve_collections,
    )

    # explicit wins
    assert resolve_collections(explicit=["foo", "bar"]) == ["foo", "bar"]

    # source_type routing
    assert resolve_collections(source_type="video") == VIDEO_COLLECTIONS
    assert resolve_collections(source_type="document") == DEFAULT_COLLECTIONS

    # single collection
    assert resolve_collections(collection="custom_col") == ["custom_col"]

    # default (no args) merges text + video -- a chat question with no
    # source_type hint must be able to reach video content too, not just the
    # document KB. Found live 2026-10-12: rag_filter.py never passes
    # source_type, so this default used to be the entire reason M3 video
    # content was unreachable from the live chat UI.
    assert resolve_collections() == ALL_COLLECTIONS
    assert resolve_collections() == DEFAULT_COLLECTIONS + VIDEO_COLLECTIONS


async def test_video_collections_merge_via_moment_search_retriever():
    """Once MomentSearchRetriever handles all 4 video tracks with the
    correct per-track embedder (visual: Voyage multimodal, D66; ocr/od/asr:
    Voyage text), visual_frames no longer needs excluding from search() --
    it's included correctly through dedicated fusion, not the generic
    single-embedder path that caused the original mismatch. Supersedes the
    old test_visual_frames_excluded_from_generic_search (P94's defensive
    guard); this is the proper fix P94/P99 both called out as still open,
    wired in 2026-10-12.
    """
    import sys
    import types

    fake = FakeQdrant()
    await with_stub_pipeline(fake)

    class FakeDoc:
        def __init__(self, id_, meta):
            self.id = id_
            self.content = ""
            self.meta = meta

    class FakeMomentRetriever:
        def __init__(self, **kwargs):
            pass

        async def run(self, query_embedding, ocr_query_embedding, od_pattern_type=None):
            return {"documents": [
                FakeDoc("vid1:visual:0", {"video_id": "vid1", "frame_ms": 1000,
                                          "source_type": "visual_frame", "wrrf_score": 0.9}),
                FakeDoc("vid1:asr:0", {"video_id": "vid1", "start_time": 5.0,
                                       "text": "service level agreement explained",
                                       "source_type": "transcript_segment", "wrrf_score": 0.8}),
            ]}

    class FakeQueryEmbeddings:
        visual_embedding = [0.1] * 1024
        voyage_embedding = [0.2] * 1024

    fake_video_fusion = types.ModuleType("pipelines.video_fusion")
    fake_video_fusion.MomentSearchRetriever = FakeMomentRetriever
    fake_video_embedder = types.ModuleType("pipelines.video_embedder")
    fake_video_embedder.embed_query = lambda q: FakeQueryEmbeddings()

    sys.modules["pipelines.video_fusion"] = fake_video_fusion
    sys.modules["pipelines.video_embedder"] = fake_video_embedder
    try:
        results = await retrieval.search(
            "service level agreement",
            collections=[config.VIDEO_VISUAL_COLLECTION, config.VIDEO_TRANSCRIPT_COLLECTION],
        )
    finally:
        del sys.modules["pipelines.video_fusion"]
        del sys.modules["pipelines.video_embedder"]

    video_ids = {r["payload"].get("video_id") for r in results}
    assert "vid1" in video_ids, results
    visual_result = next((r for r in results if r["payload"].get("source_type") == "visual_frame"), None)
    assert visual_result is not None, "visual_frames must now be included, not excluded"
    assert visual_result["collection"] == config.VIDEO_VISUAL_COLLECTION, visual_result
    reset_singletons()


async def test_video_search_failure_degrades_to_text_only():
    """If the video embedding/retrieval path fails (e.g. Voyage API down),
    the text portion of a merged search must still return results -- not
    crash the whole query."""
    import sys
    import types

    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    embedder = embedding_service._provider
    await svc.ensure_collection("documents", 8)
    await svc.upsert("documents", ["d1"], await embedder.embed(["service level agreement basics"]),
                     [{"content": "service level agreement basics", "source_type": "document"}])

    def _boom(q):
        raise RuntimeError("voyage api down")

    fake_video_embedder = types.ModuleType("pipelines.video_embedder")
    fake_video_embedder.embed_query = _boom
    sys.modules["pipelines.video_embedder"] = fake_video_embedder
    try:
        results = await retrieval.search(
            "service level agreement basics",
            collections=["documents", config.VIDEO_TRANSCRIPT_COLLECTION],
        )
    finally:
        del sys.modules["pipelines.video_embedder"]

    assert results, "text results must still come back when the video path fails"
    assert results[0]["payload"]["content"] == "service level agreement basics"
    reset_singletons()


async def test_video_single_list_competes_fairly_against_text_dense_sparse_pair():
    """P102: _search_video() returns one already-fused list (D59's internal
    4-track WRRF), while each text collection contributes TWO lists to the
    outer reciprocal_rank_fusion() when hybrid search is on (dense + sparse).
    A text doc ranking well in both of its own collection's lists got roughly
    double the RRF score of an equally well-ranked video hit, which only has
    one list to appear in at all. Found live 2026-10-12: a near-exact
    transcript match scored 0.949 on reranking in isolation, but never
    reached the reranker in production because 20 near-duplicate
    documents candidates -- each counted twice -- filled every fusion
    slot ahead of it. Video's list must count twice, matching the weight a
    text collection gets from its dense+sparse pair.
    """
    fake = FakeQdrant()
    await with_stub_pipeline(fake)

    # Five "junk" text candidates, each returned at identical ranks by BOTH
    # dense and sparse search -- exactly the double-counting that crowded out
    # video in production (there it was boilerplate CSV descriptions).
    junk = [{"id": f"junk{i}", "collection": "documents",
             "payload": {"content": f"junk {i}"}} for i in range(5)]

    async def fake_search_dense(collection, vector, top_k, filters=None):
        return junk if collection == "documents" else []

    async def fake_search_sparse(collection, query_text, top_k, filters=None):
        return junk if collection == "documents" else []

    qdrant_service._service.search_dense = fake_search_dense
    qdrant_service._service.search_sparse = fake_search_sparse

    async def fake_search_video(query, sp=None):
        return [{"id": "vid1:asr:0", "collection": config.VIDEO_TRANSCRIPT_COLLECTION,
                 "payload": {"text": "the real answer", "video_id": "vid1",
                             "source_type": "transcript_segment"}}]

    original = retrieval._search_video
    retrieval._search_video = fake_search_video
    try:
        results = await retrieval.search(
            "anything",
            collections=["documents", config.VIDEO_TRANSCRIPT_COLLECTION],
        )
    finally:
        retrieval._search_video = original

    video_ids = {r["payload"].get("video_id") for r in results}
    assert "vid1" in video_ids, (
        "video hit lost fusion to double-counted text junk despite ranking "
        "first in its own list: " + repr(results)
    )
    reset_singletons()


async def test_timestamp_query_still_searches_text_collections():
    """A query naming a clock time ("the 9:30 open") makes routers/search.py
    add a start_time/end_time window meant for video transcript segments.
    That window was also sent to every TEXT collection, whose chunks carry
    no such fields -- and build_filter() turns the {"$lte": n} operator dict
    into a match-value Qdrant rejects outright (400 "did not match any
    variant of untagged enum MatchInterface"; P100). qdrant_service logs and
    returns [] on a 400, so every timestamp query silently lost the whole
    document KB. Found live 2026-10-16: "what happens at the
    9:30 open" fused 10 candidates (video only) instead of 20, with four 400s
    in the haystack log. Lecture questions name clock times constantly.
    """
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    embedder = embedding_service._provider
    text = "session open at 9:30 new york time"
    await svc.ensure_collection(config.COLLECTION_CODE, 8)
    await svc.upsert(config.COLLECTION_CODE, ["p1"], await embedder.embed([text]),
                     [{"content": text, "source_type": "document"}])

    async def no_video(query, sp=None):
        return []

    original = retrieval._search_video
    retrieval._search_video = no_video
    try:
        resp = await search_knowledge(q=text, top_k=5, collections=None, source_type=None)
    finally:
        retrieval._search_video = original

    text_queries = [body for _, path, body in fake.seen
                    if path.endswith("/points/query")]
    assert text_queries, "text collections were never queried"
    for body in text_queries:
        keys = {c["key"] for c in (body.get("filter") or {}).get("must", [])}
        assert not keys & {"start_time", "end_time"}, (
            "video-only time window sent to a text collection: %r" % body.get("filter"))
    assert [r.content for r in resp.results] == [text], resp.results
    reset_singletons()


async def test_timestamp_query_does_not_mutate_the_module_collection_lists():
    """resolve_collections() hands back the module-level lists themselves, and
    the timestamp branch of routers/search.py appended the video collections
    to whatever it got. One source_type=document request naming a time
    therefore turned DEFAULT_COLLECTIONS into text+video for the life of the
    process -- every later "document" search, and retrieval.search()'s own
    default, quietly included video from then on."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    before = list(retrieval.DEFAULT_COLLECTIONS)

    async def no_video(query, sp=None):
        return []

    original = retrieval._search_video
    retrieval._search_video = no_video
    try:
        await search_knowledge(q="what happens at 9:30", top_k=5, collections=None,
                               source_type="document")
    finally:
        retrieval._search_video = original
    try:
        assert retrieval.DEFAULT_COLLECTIONS == before, retrieval.DEFAULT_COLLECTIONS
    finally:
        retrieval.DEFAULT_COLLECTIONS[:] = before
        reset_singletons()


def test_parse_timestamp():
    from pipelines.retrieval import parse_timestamp

    assert parse_timestamp("at 15:30") == 930
    assert parse_timestamp("1:22:45") == 4965
    assert parse_timestamp("what is SLA") is None
    assert parse_timestamp("at 0:05") == 5
    assert parse_timestamp("5:00 in episode 47") == 300


def test_parse_natural_time_expressions():
    """I-2 (M3 question set, 2026-10-22): "around 25 minutes in" named a moment
    but only h:mm:ss / m:ss parsed, so the moment lookup never ran."""
    from pipelines.retrieval import parse_timestamp

    cases = {
        "In episode 12, around 25 minutes in, what is he forecasting?": 1500,
        "what does he say around minute 44 in episode 12": 2640,
        "at minute 44": 2640,
        "what is on the chart at 44 min": 2640,
        "what does he say 44 min into Episode 12": 2640,
        "at 44m in episode 12": 2640,
        "at 1 hour 8 minutes": 4080,
        "around 1 hr and 8 min into the video": 4080,
        "at 1h08m": 4080,
        "1h 8m in": 4080,
        "what happens at the 25 minute mark": 1500,
        "at the 25-minute mark": 1500,
        "44 minutes 30 seconds in": 2670,
        "at 44m30s": 2670,
        "the 44th minute of Episode 12": 2640,
        "2 hours in": 7200,
    }
    for q, want in cases.items():
        assert parse_timestamp(q) == want, (q, parse_timestamp(q), want)
    # Chart timeframes and durations are not positions in a video.
    for q in ("what is a 15 minute service level agreement", "on the 5 minute chart", "the 1h SLA",
              "on the 15m timeframe", "a 4 hour bar", "for 20 minutes he waits",
              "the first 10 minutes after the open", "5 minutes after the open",
              "what is a service level agreement", "Episode 12", "the 2022 mentorship"):
        assert parse_timestamp(q) is None, (q, parse_timestamp(q))


def test_halfway_resolves_against_the_named_videos_duration():
    """"halfway through" is relative: it needs the named video's duration and is
    None without it. "halfway into the gap" is a price location, not a time."""
    from pipelines.retrieval import parse_timestamp

    assert parse_timestamp("what does he say halfway through Episode 12", duration=4437) == 2218
    assert parse_timestamp("at the halfway point of the March 27 review", duration=6783) == 3391
    assert parse_timestamp("around the halfway mark", duration=551) == 275
    assert parse_timestamp("midway through the video", duration=1000) == 500
    assert parse_timestamp("what does he say halfway through Episode 12") is None
    assert parse_timestamp("price retraced halfway into the gap", duration=4437) is None
    assert parse_timestamp("halfway through the service level agreement", duration=4437) is None


def test_clock_times_of_day_are_not_video_positions():
    """I-3: "In Episode 36, what happened after the spare capacity was taken
    at the 9:30 open?" was read as 9 min 30 s into the video (2026-10-22) and
    returned the end of the 9:11 episode. Clock times of day are not positions."""
    from pipelines.retrieval import parse_timestamp

    for q in ("In Episode 36, what happened after the spare capacity was taken at the 9:30 open?",
              "what happens at 9:30 AM", "at 9:30am", "at 9:30 a.m. in episode 12", "at 8:30 NY time",
              "8:30 New York", "at 2:00 PM", "the 10:00 bar", "the 10 o'clock bar",
              "what happens at 10:00 EST", "the New York open at 9:30", "the 9:50 macro",
              "at 3:00 in the London session", "the 9:30 opening range", "at 4:00 close", "at 9:30 ET",
              "at 14:30 UTC", "in the 8:30 killzone"):
        assert parse_timestamp(q) is None, (q, parse_timestamp(q))
    # Video positions stay video positions -- including after a time of day.
    assert parse_timestamp("In Episode 12, what does the speaker draw around 1:08:00?") == 4080
    assert parse_timestamp("around 25:00 in Episode 12") == 1500
    assert parse_timestamp("after the 9:30 open, what does he say at 25:00 in episode 12") == 1500
    assert parse_timestamp("In the X post video, what is on the chart at 14:32?") == 872
    # Not a clock: ratios, prices, out-of-range fields.
    assert parse_timestamp("a 1:2 risk to reward") is None
    assert parse_timestamp("at 1:75") is None


def test_bare_clock_time_is_a_video_position_only_within_the_named_video():
    """Rule: a bare "9:30" (no time-of-day words around it) is a video position --
    unless the named video is shorter than that, when it cannot be one."""
    from pipelines.retrieval import parse_timestamp

    assert parse_timestamp("In Episode 12, what is said at 9:30?") == 570
    assert parse_timestamp("In Episode 12, what is said at 9:30?", duration=4437) == 570
    assert parse_timestamp("In Episode 36, what is said at 9:30?", duration=551) is None
    assert parse_timestamp("In Episode 36 at 7:30, what is happening?", duration=551) == 450
    assert parse_timestamp("around 25 minutes in", duration=551) is None
    assert parse_timestamp("at 3:10:00", duration=551) is None


async def test_time_of_day_question_does_not_add_a_video_time_window():
    """routers/search.py adds a start_time/end_time window for any parsed time;
    for "the 9:30 open" that window pinned the search to 9:00-10:00 into every
    video. Times of day must reach it as None."""
    seen = []

    async def fake_search(q, top_n=5, collections=None, filters=None):
        seen.append(filters)
        return []

    saved = retrieval.search
    retrieval.search = fake_search
    try:
        await search_knowledge(q="what happens at the 9:30 open", top_k=5, collections=None, source_type=None)
        await search_knowledge(q="what does he say at 25:00", top_k=5, collections=None, source_type=None)
    finally:
        retrieval.search = saved
    assert seen[0] is None or "start_time" not in seen[0], seen
    assert seen[1]["start_time"] == {"$lte": 1530} and seen[1]["end_time"] == {"$gte": 1470}, seen
    reset_singletons()


# ── P152: planned video leg ─────────────────────────────

class _TagReranker:
    """Calibrated fake: scores a text 0.9 when it contains the query string,
    0.1 otherwise -- so a passage only survives the floor when reranked
    against the sub-query it answers."""
    calibrated = True

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.queries = []

    async def rerank(self, query, texts, top_n):
        self.queries.append(query)
        out = []
        for i, t in enumerate(texts):
            s = self.scores.get((query, t), 0.9 if query in t else 0.1)
            out.append((i, s))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_n]


def _video_hit(vid, text, t=0.0):
    return {"id": vid, "collection": config.VIDEO_TRANSCRIPT_COLLECTION,
            "payload": {"video_id": "V1", "text": text, "start_time": t,
                        "source_type": "transcript_segment"}}


def _plan(queries, video_name=None, intent="passage"):
    from pipelines.query_planner import VideoQueryPlan
    return VideoQueryPlan(list(queries), video_name, intent)


async def _with_video_fakes(plan, search_video, coro_fn, catalog=None):
    from pipelines import query_planner
    saved = (query_planner.plan_video_queries, query_planner.indexed_videos, retrieval._search_video)
    query_planner.plan_video_queries = lambda q: plan(q) if callable(plan) else plan

    async def _catalog():
        return catalog or []
    query_planner.indexed_videos = _catalog
    retrieval._search_video = search_video
    try:
        return await coro_fn()
    finally:
        (query_planner.plan_video_queries, query_planner.indexed_videos, retrieval._search_video) = saved


async def test_text_leg_is_unchanged_by_video_planning():
    """The video leg must not change a single text result: the client's retest
    and every M1.2/M2 answer run through the text collections."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    embedder = embedding_service._provider
    await svc.ensure_collection("documents", 8)
    texts = [d[2]["content"] for d in DOCS]
    await svc.upsert("documents", [d[0] for d in DOCS], await embedder.embed(texts), [d[2] for d in DOCS])

    async def no_video(q, sp=None, video_id=None, **kw):
        return []

    q = DOCS[0][2]["content"]
    text_only = await retrieval.search(q, collections=["documents"])
    with_video = await _with_video_fakes(_plan(["a", "b", "c"]), no_video,
                                         lambda: retrieval.search(q, collections=["documents",
                                                                                   config.VIDEO_TRANSCRIPT_COLLECTION]))
    key = lambda rs: [(r["collection"], str(r["id"]), round(float(r["score"]), 9)) for r in rs]
    assert key(with_video) == key(text_only), (key(text_only), key(with_video))
    reset_singletons()


async def test_each_sub_query_is_reranked_against_itself():
    """Reranked against the whole chat message, both passages fall under the
    0.70 floor (P152: 0.654 for the real SLA answer); reranked against the
    sub-query each one answers, both survive."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    reranker_service._reranker = _TagReranker()
    hits = [_video_hit("V1:asr:1", "alpha passage"), _video_hit("V1:asr:2", "beta passage")]

    async def sv(q, sp=None, video_id=None, **kw):
        return list(hits)

    question = "please find two things for me with timestamps"
    planned = await _with_video_fakes(_plan(["alpha", "beta"]), sv,
                                      lambda: retrieval.search(question, collections=[config.VIDEO_TRANSCRIPT_COLLECTION]))
    assert {r["id"] for r in planned} == {"V1:asr:1", "V1:asr:2"}, planned
    unplanned = await _with_video_fakes(_plan([question]), sv,
                                        lambda: retrieval.search(question, collections=[config.VIDEO_TRANSCRIPT_COLLECTION]))
    assert unplanned == [], unplanned
    reset_singletons()


async def test_named_video_filters_the_video_leg():
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    seen = []

    async def sv(q, sp=None, video_id=None, **kw):
        seen.append(video_id)
        return []

    await _with_video_fakes(_plan(["health checks"], video_name="2022 Acme Mentorship Episode 12"), sv,
                            lambda: retrieval.search("x", collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                            catalog=[("8GkQfdAXZP0", "2022 Acme Mentorship Episode 12"),
                                     ("pWnMpdN_g98", "2022 Acme Mentorship Episode 36")])
    # The original question is also searched at once, unfiltered (speculative);
    # the planned sub-query carries the resolved video filter.
    assert "8GkQfdAXZP0" in seen, seen
    seen.clear()
    await _with_video_fakes(_plan(["health checks"], video_name="Episode 1"), sv,
                            lambda: retrieval.search("x", collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                            catalog=[("8GkQfdAXZP0", "2022 Acme Mentorship Episode 12")])
    assert seen and set(seen) == {None}, "an unresolvable name must not filter: " + repr(seen)
    reset_singletons()


async def test_multi_part_keeps_one_passage_per_sub_query():
    """Three topics asked: topic 3's best passage (0.72) must not be crowded
    out by topics 1-2's many high-scoring ones."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    many = {q: [_video_hit("%s%d" % (q, i), "%s passage %d" % (q, i), i) for i in range(4)] for q in ("one", "two")}
    many["three"] = [_video_hit("three0", "three passage 0")]
    scores = {("three", "three passage 0"): 0.72}
    reranker_service._reranker = _TagReranker(scores)

    async def sv(q, sp=None, video_id=None, **kw):
        return list(many[q])

    res = await _with_video_fakes(_plan(["one", "two", "three"]), sv,
                                  lambda: retrieval.search("q", top_n=5, collections=[config.VIDEO_TRANSCRIPT_COLLECTION]))
    ids = [r["id"] for r in res]
    assert "three0" in ids, ids
    assert len(ids) == 5, ids
    reset_singletons()


async def test_video_results_never_displace_text():
    """Default search covers text + video. Video passages are appended after
    the text results, never in place of them: letting them compete for the
    top 5 cut a KB answer from 14 text passages to 2 (measured 2026-10-20)."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    embedder = embedding_service._provider
    await svc.ensure_collection("documents", 8)
    texts = [d[2]["content"] for d in DOCS]
    await svc.upsert("documents", [d[0] for d in DOCS], await embedder.embed(texts), [d[2] for d in DOCS])
    q = DOCS[0][2]["content"]
    # top_n=2: the text leg fills the whole page, as it did in production --
    # with spare slots nothing can be displaced and the test proves nothing.
    text_only = await retrieval.search(q, top_n=2, collections=["documents"])
    hits = {"a": [_video_hit("Va", "a passage")], "b": [_video_hit("Vb", "b passage")]}

    async def sv(sub, sp=None, video_id=None, **kw):
        return list(hits[sub])

    both = await _with_video_fakes(_plan(["a", "b"]), sv,
                                   lambda: retrieval.search(q, top_n=2, collections=["documents",
                                                                                      config.VIDEO_TRANSCRIPT_COLLECTION]))
    key = lambda rs: [(r["collection"], str(r["id"])) for r in rs]
    assert len(text_only) >= 2, text_only
    assert key(both[:len(text_only)]) == key(text_only), (key(text_only), key(both))
    assert [r["id"] for r in both[len(text_only):]] == ["Va", "Vb"], key(both)
    reset_singletons()


async def test_summary_intent_returns_the_video_overview():
    """'Summarize this video' (P152 stage B): no passage scores high against
    an overview question, so the stored per-video overview answers it, first.
    Without one, passage search still runs -- never an empty answer."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await svc.ensure_collection(config.VIDEO_TRANSCRIPT_COLLECTION, 8, enable_sparse=False)
    await svc.upsert(config.VIDEO_TRANSCRIPT_COLLECTION, ["V1:overview"], [[0.1] * 8],
                     [{"video_id": "V1", "title": "Acme 2026 Weekly Review", "text": "Main concepts...",
                       "overview": True, "start_time": 0.0, "source_type": "transcript_segment"}],
                     sparse_vectors=[{}])
    passage = [_video_hit("V1:asr:3", "a passage")]

    async def sv(q, sp=None, video_id=None, **kw):
        return list(passage)

    catalog = [("V1", "Acme 2026 Weekly Review"), ("V2", "2022 Acme Mentorship Episode 12")]
    res = await _with_video_fakes(_plan(["main concepts"], video_name="Acme 2026 Weekly Review", intent="summary"),
                                  sv, lambda: retrieval.search("summarize it", collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=catalog)
    assert res and res[0]["payload"].get("overview") is True and res[0]["payload"]["video_id"] == "V1", res
    assert res[0]["collection"] == config.VIDEO_TRANSCRIPT_COLLECTION
    assert any(r["id"] == "V1:asr:3" for r in res), res

    res2 = await _with_video_fakes(_plan(["main concepts"], video_name="2022 Acme Mentorship Episode 12", intent="summary"),
                                   sv, lambda: retrieval.search("summarize it", collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                   catalog=catalog)
    assert [r["id"] for r in res2] == ["V1:asr:3"], "no overview for V2 -> passage results: " + repr(res2)
    reset_singletons()


async def test_planner_failure_falls_back_to_original_question():
    from pipelines import query_planner
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    seen = []

    async def sv(q, sp=None, video_id=None, **kw):
        seen.append(q)
        return []

    def boom(prompt):
        raise RuntimeError("gemini down")

    saved = (query_planner._generate, retrieval._search_video)
    query_planner._generate, retrieval._search_video = boom, sv
    try:
        out = await retrieval.search("unplannable question here", collections=[config.VIDEO_TRANSCRIPT_COLLECTION])
    finally:
        query_planner._generate, retrieval._search_video = saved
    assert seen == ["unplannable question here"], seen
    assert out == []
    reset_singletons()


async def test_slow_video_leg_is_cut_at_the_deadline():
    """Review finding 1: rag_filter waits 5 s for /search (search_timeout
    default, no valve override deployed) and on timeout drops EVERYTHING,
    text included. The video leg (planner + up to 4 sub-query searches) must
    never hold the response past the deadline: cut it, return text alone."""
    import time as _time
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    embedder = embedding_service._provider
    await svc.ensure_collection("documents", 8)
    texts = [d[2]["content"] for d in DOCS]
    await svc.upsert("documents", [d[0] for d in DOCS], await embedder.embed(texts), [d[2] for d in DOCS])
    q = DOCS[0][2]["content"]
    text_only = await retrieval.search(q, collections=["documents"])
    cancelled = []

    async def slow(sub, sp=None, video_id=None, **kw):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(sub)
            raise
        return []

    saved = retrieval.VIDEO_LEG_DEADLINE_S
    retrieval.VIDEO_LEG_DEADLINE_S = 0.5
    try:
        t0 = _time.perf_counter()
        res = await _with_video_fakes(_plan(["a"]), slow,
                                      lambda: retrieval.search(q, collections=["documents",
                                                                                config.VIDEO_TRANSCRIPT_COLLECTION]))
        elapsed = _time.perf_counter() - t0
        await asyncio.sleep(0)          # let the cancellation land
    finally:
        retrieval.VIDEO_LEG_DEADLINE_S = saved
    assert elapsed < 2.0, elapsed
    key = lambda rs: [(r["collection"], str(r["id"])) for r in rs]
    assert key(res) == key(text_only), (key(text_only), key(res))
    assert "a" in cancelled, "the slow video search must be cancelled, not leaked: %r" % cancelled
    reset_singletons()


async def test_comparison_rescue_still_runs_when_video_passes():
    """Review finding 2: the comparison-terms rescue (C9) recovers a text
    answer when the floor empties the text page. A passing video passage must
    not suppress it, and the rescue must not re-run the whole video leg."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    embedder = embedding_service._provider
    await svc.ensure_collection("documents", 8)
    doc = "alpha beta difference explained"
    await svc.upsert("documents", ["cmp1"], await embedder.embed([doc]),
                     [{"content": doc, "source_type": "document"}])
    reranker_service._reranker = _TagReranker()
    plans = []

    def plan(question):
        plans.append(question)
        return _plan(["alpha video"])

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_video_hit("V1:asr:9", "alpha video passage")]

    res = await _with_video_fakes(plan, sv, lambda: retrieval.search(
        "what is the difference between alpha and beta?",
        collections=["documents", config.VIDEO_TRANSCRIPT_COLLECTION]))
    contents = [r["payload"].get("content") or r["payload"].get("text") for r in res]
    assert doc in contents, "comparison rescue must still recover the text answer: %r" % contents
    assert "alpha video passage" in contents, contents
    assert len(plans) == 1, "the rescue must not run a second planner/video leg: %r" % plans
    reset_singletons()


async def test_search_video_drops_overview_points():
    """Review finding 4: the per-video overview is reachable only through a
    summary question (_fetch_overview). As an ordinary transcript vector it
    would win a passage query's guaranteed slot and be cited at 0:00:00."""
    import sys
    import types

    class FakeDoc:
        def __init__(self, id_, meta):
            self.id, self.content, self.meta = id_, "", meta

    class FakeMoment:
        def __init__(self, **kwargs):
            pass

        async def run(self, query_embedding, ocr_query_embedding, od_pattern_type=None):
            return {"documents": [
                FakeDoc("V1:overview", {"video_id": "V1", "overview": True, "text": "Main concepts",
                                        "source_type": "transcript_segment"}),
                FakeDoc("V1:asr:4", {"video_id": "V1", "text": "the real passage",
                                     "source_type": "transcript_segment"}),
            ]}

    class FakeEmb:
        visual_embedding = [0.1] * 4
        voyage_embedding = [0.2] * 4

    fv = types.ModuleType("pipelines.video_fusion")
    fv.MomentSearchRetriever = FakeMoment
    fe = types.ModuleType("pipelines.video_embedder")
    fe.embed_query = lambda q: FakeEmb()
    sys.modules["pipelines.video_fusion"], sys.modules["pipelines.video_embedder"] = fv, fe
    try:
        out = await retrieval._search_video("load balancer definition")
    finally:
        del sys.modules["pipelines.video_fusion"]
        del sys.modules["pipelines.video_embedder"]
    assert [c["id"] for c in out] == ["V1:asr:4"], out


def test_library_text_lists_videos_with_concepts():
    """'How many videos do you have?' can't be answered by passage similarity
    (0 results live, 2026-10-20). The library answer is built from each
    video's stored overview: title, duration, main-concept headings."""
    videos = [
        {"title": "2022 Acme Mentorship Episode 12", "end_time": 4439.0, "platform": "youtube",
         "text": "Intro line\n### 1. Health Checks and Alerts\nbody\n### 2. Load Balancers\nbody"},
        {"title": "Acme 2026 Weekly Review", "end_time": 6785.0, "platform": "youtube", "text": "no headings here"},
    ]
    text = retrieval._library_text(videos, total=2)
    assert text.startswith("The video knowledge base contains 2 ingested videos"), text
    assert "1. 2022 Acme Mentorship Episode 12 (1:13:59)" in text and "Health Checks and Alerts; Load Balancers" in text, text
    assert "2. Acme 2026 Weekly Review (1:53:05)" in text, text
    # Gemini doesn't keep one format across videos: also "* **Heading:** ..."
    # and "1. **Heading** [0:08:28]" (seen on the live overviews, 2026-10-20).
    bullets = retrieval._library_text([
        {"title": "Ep 36", "end_time": 551.0, "text": "Here is a summary:\n\n* **Identifying Cost Savings:** On the hourly chart [0:00:52].\n\n* **Capacity Raids at the Open:** Algorithms [0:01:39]."},
        {"title": "Commentary", "end_time": 6785.0, "text": "Summary:\n\n1. **Daily Wick as a Range/Gap** [0:08:28]\n   Daily wicks\n\n2. **Time Distortion** [0:13:59]\n   When price"},
    ], total=2)
    assert "Identifying Cost Savings; Capacity Raids at the Open" in bullets, bullets
    assert "Daily Wick as a Range/Gap; Time Distortion" in bullets, bullets
    many = [{"title": "V%d" % i, "end_time": 60.0, "platform": "youtube", "text": ""} for i in range(30)]
    capped = retrieval._library_text(many, total=30)
    assert "25. V24" in capped and "26. V25" not in capped and "...and 5 more" in capped, capped


async def test_library_intent_answers_from_the_overviews():
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await svc.ensure_collection(config.VIDEO_TRANSCRIPT_COLLECTION, 8, enable_sparse=False)
    await svc.upsert(config.VIDEO_TRANSCRIPT_COLLECTION, ["V1:overview", "V2:overview", "V1:asr:0"],
                     [[0.1] * 8] * 3,
                     [{"video_id": "V1", "title": "Episode 12", "text": "### 1. SLA", "overview": True,
                       "end_time": 600.0, "source_type": "transcript_segment"},
                      {"video_id": "V2", "title": "Weekly Review", "text": "### 1. Wicks", "overview": True,
                       "end_time": 1200.0, "source_type": "transcript_segment"},
                      {"video_id": "V1", "title": "Episode 12", "text": "a passage", "start_time": 5.0,
                       "source_type": "transcript_segment"}],
                     sparse_vectors=[{}, {}, {}])
    searched = []

    async def sv(q, sp=None, video_id=None, **kw):
        searched.append(q)
        return []

    res = await _with_video_fakes(_plan(["ingested videos"], intent="library"), sv,
                                  lambda: retrieval.search("how many vodeos u have",
                                                           collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=[("V1", "Episode 12"), ("V2", "Weekly Review")])
    assert len(res) == 1 and res[0]["payload"]["source_type"] == "video_library", res
    body = res[0]["payload"]["text"]
    assert "2 ingested videos" in body and "Episode 12" in body and "Weekly Review" in body, body
    assert res[0]["payload"]["title"] == "Video library (2 videos)"
    assert all(r["payload"]["source_type"] == "video_library" for r in res), res
    reset_singletons()


async def test_slow_planner_still_returns_the_original_questions_video_results():
    """The planner is on the critical path of every video answer. If it is
    slow, the search already run on the original question answers instead of
    nothing (a live question got 0 results, 2026-10-20)."""
    import time as _time
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    reranker_service._reranker = _TagReranker()
    q = "how does a team create a service level agreement"

    def slow_plan(question):
        _time.sleep(1.5)
        return _plan(["planned"])

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_video_hit("V1:asr:7", "how does a team create a service level agreement, explained")] if sub == q else []

    saved = retrieval.PLANNER_BUDGET_S
    retrieval.PLANNER_BUDGET_S = 0.3
    try:
        res = await _with_video_fakes(slow_plan, sv, lambda: retrieval.search(
            q, collections=[config.VIDEO_TRANSCRIPT_COLLECTION]))
    finally:
        retrieval.PLANNER_BUDGET_S = saved
    assert [r["id"] for r in res] == ["V1:asr:7"], res
    reset_singletons()


async def test_empty_planned_searches_fall_back_to_the_original_question():
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    reranker_service._reranker = _TagReranker()
    q = "where does he explain capacity sweeps"

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_video_hit("V1:asr:8", "where does he explain capacity sweeps: here")] if sub == q else []

    res = await _with_video_fakes(_plan(["nothing matches this"]), sv, lambda: retrieval.search(
        q, collections=[config.VIDEO_TRANSCRIPT_COLLECTION]))
    assert [r["id"] for r in res] == ["V1:asr:8"], res
    reset_singletons()


def test_build_filter_turns_operator_dicts_into_range_conditions():
    """P100: {"$lte": n} used to become match-value against a dict, which no
    point can equal -- a time-window filter matched nothing."""
    assert build_filter({"start_time": {"$lte": 945}, "end_time": {"$gte": 885}}) == {"must": [
        {"key": "start_time", "range": {"lte": 945}},
        {"key": "end_time", "range": {"gte": 885}}]}
    assert build_filter({"frame_ms": {"$gt": 1, "$lt": 9}}) == {"must": [
        {"key": "frame_ms", "range": {"gt": 1, "lt": 9}}]}


async def _timed_index(fake, svc):
    """Video V1 (Episode 12) and V2, transcript windows at known times, plus V1's overview."""
    await svc.ensure_collection(config.VIDEO_TRANSCRIPT_COLLECTION, 8)
    rows = [("V1:asr:0", "V1", 0.0, 58.0, "intro"),
            ("V1:asr:1", "V1", 4075.28, 4133.94, "why does the market even create a service level agreement"),
            ("V1:asr:2", "V1", 4127.6, 4171.64, "it goes back up to offer buyers"),
            ("V1:asr:3", "V1", 6000.0, 6058.0, "much later"),
            ("V2:asr:1", "V2", 4070.0, 4130.0, "other video at the same time")]
    ids = [r[0] for r in rows] + ["V1:overview"]
    payloads = [{"video_id": v, "start_time": s, "end_time": e, "text": t,
                 "source_type": "transcript_segment", "title": "2022 Acme Mentorship Episode 12" if v == "V1" else "Other"}
                for _, v, s, e, t in rows]
    payloads.append({"video_id": "V1", "start_time": 0.0, "end_time": 7000.0, "text": "### 1. SLA",
                     "overview": True, "source_type": "transcript_segment"})
    await svc.upsert(config.VIDEO_TRANSCRIPT_COLLECTION, ids, [[0.1] * 8] * len(ids), payloads)


_EP12 = [("V1", "2022 Acme Mentorship Episode 12"), ("V2", "Other")]


async def test_named_video_timestamp_returns_the_passages_at_that_moment():
    """P100: "at 1:07:55 in episode 12" must answer from Episode 12 at 1:07:55.
    Live it returned frames at 26:10, 2:17, 25:50, 40:17 (2026-10-21): the time
    never reached the video leg."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    reranker_service._reranker = _TagReranker()
    q = "what does he explain at 1:07:55 in episode 12"

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_video_hit("V1:asr:3", "much later " + sub)]      # semantic search: the wrong moment

    res = await _with_video_fakes(_plan(["what does he explain"], video_name="episode 12"), sv,
                                  lambda: retrieval.search(q, collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=_EP12)
    # Upserted string ids come back as UUIDs; identify the passage by its payload.
    assert [(r["payload"]["video_id"], r["payload"]["start_time"]) for r in res] == [("V1", 4075.28)], res
    assert res[0]["payload"]["title"] == "2022 Acme Mentorship Episode 12", res
    reset_singletons()


async def test_named_video_timestamp_past_its_end_falls_back_to_the_planned_search():
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    reranker_service._reranker = _TagReranker()
    q = "what does he explain at 3:10:00 in episode 12"

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_video_hit("V1:asr:3", "what does he explain: much later")]

    res = await _with_video_fakes(_plan(["what does he explain"], video_name="episode 12"), sv,
                                  lambda: retrieval.search(q, collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=_EP12)
    assert [r["id"] for r in res] == ["V1:asr:3"], res
    reset_singletons()


async def test_a_time_without_a_named_video_does_not_filter_the_video_leg():
    """"the 9:30 open" is a time of day as often as a video position: with no
    named video there is no moment to look up, so the search is the usual one."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    reranker_service._reranker = _TagReranker()
    q = "what happens at the 9:30 open"

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_video_hit("V1:asr:3", "what happens at the 9:30 open: much later")]

    res = await _with_video_fakes(_plan(["what happens at the 9:30 open"]), sv,
                                  lambda: retrieval.search(q, collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=_EP12)
    assert [r["id"] for r in res] == ["V1:asr:3"], res
    reset_singletons()


async def test_natural_time_in_named_video_returns_the_passages_at_that_moment():
    """I-2: "around 68 minutes in" names the same moment as 1:08:00."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    reranker_service._reranker = _TagReranker()

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_video_hit("V1:asr:3", "much later " + sub)]

    res = await _with_video_fakes(_plan(["what does he explain"], video_name="episode 12"), sv,
                                  lambda: retrieval.search("in episode 12, around 68 minutes in, what does he explain",
                                                           collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=_EP12)
    assert [(r["payload"]["video_id"], r["payload"]["start_time"]) for r in res] == [("V1", 4075.28)], res
    reset_singletons()


async def test_time_of_day_in_named_video_is_not_a_moment():
    """I-3: "...taken at the 9:30 open?" in a named video answered from 9:30 into
    the video. A time of day must leave the planned search in charge."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    await svc.upsert(config.VIDEO_TRANSCRIPT_COLLECTION, ["V1:asr:9"], [[0.1] * 8],
                     [{"video_id": "V1", "start_time": 560.0, "end_time": 600.0, "text": "at nine minutes",
                       "source_type": "transcript_segment", "title": "2022 Acme Mentorship Episode 12"}])
    reranker_service._reranker = _TagReranker()
    q = "In Episode 12, what happened after the spare capacity was taken at the 9:30 open?"

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_video_hit("V1:asr:3", "spare capacity taken: " + sub)]

    res = await _with_video_fakes(_plan(["spare capacity taken"], video_name="Episode 12"), sv,
                                  lambda: retrieval.search(q, collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=_EP12)
    assert [r["id"] for r in res] == ["V1:asr:3"], res
    reset_singletons()


async def test_halfway_in_named_video_uses_its_duration():
    """"halfway through Episode 12" -> the moment at half the video's duration,
    read from the indexed_videos cache (the largest transcript end_time)."""
    from pipelines import query_planner
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    reranker_service._reranker = _TagReranker()

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_video_hit("V1:asr:3", "much later " + sub)]

    query_planner._indexed_cache["durations"] = {"V1": 8150.0}
    try:
        res = await _with_video_fakes(_plan(["what does he explain"], video_name="episode 12"), sv,
                                      lambda: retrieval.search("what does he explain halfway through episode 12",
                                                               collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                      catalog=_EP12)
    finally:
        query_planner._indexed_cache["durations"] = {}
    assert [(r["payload"]["video_id"], r["payload"]["start_time"]) for r in res] == [("V1", 4075.28)], res
    reset_singletons()


def _frame_hit(fid, text, t_ms, source_type="ocr_frame", video="V1"):
    return {"id": fid, "collection": config.VIDEO_OCR_COLLECTION,
            "payload": {"video_id": video, "ocr_text": text, "text": text, "frame_ms": t_ms,
                        "source_type": source_type, "title": "Episode 12"}}


async def test_library_question_about_content_also_returns_matching_passages():
    """the client, 2026-10-22: "Do any of the videos show chart labels like STH, STL,
    ITH, ITL or LTL?" was planned as a library question, so only the video list
    came back and the model said it had no chart frames -- while Episode 12's
    frames carry STH hundreds of times. A library answer must not suppress the
    content search: passages that clear the floor come back with it."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await svc.ensure_collection(config.VIDEO_TRANSCRIPT_COLLECTION, 8, enable_sparse=False)
    await svc.upsert(config.VIDEO_TRANSCRIPT_COLLECTION, ["V1:overview"], [[0.1] * 8],
                     [{"video_id": "V1", "title": "Episode 12", "text": "### 1. System structure", "overview": True,
                       "end_time": 4400.0, "source_type": "transcript_segment"}], sparse_vectors=[{}])
    reranker_service._reranker = _TagReranker()

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_frame_hit("V1:ocr:9", "STH chart labels STH STL ITH", 1716000)] if sub == "STH chart labels" else []

    res = await _with_video_fakes(_plan(["STH chart labels"], intent="library"), sv,
                                  lambda: retrieval.search("Do any of the videos show chart labels like STH?",
                                                           collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=[("V1", "Episode 12")])
    kinds = [r["payload"]["source_type"] for r in res]
    assert "video_library" in kinds and "ocr_frame" in kinds, res
    reset_singletons()


async def test_named_video_time_includes_chart_frames_at_that_moment():
    """the client, 2026-10-22: "In Episode 12, what does the speaker draw or mark on
    the chart around 1:08:00?" -- the transcript at that time is not enough,
    the answer is on the chart. Frame descriptions within the window come too."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    await svc.ensure_collection(config.VIDEO_OD_COLLECTION, 8, enable_sparse=False)
    await svc.upsert(config.VIDEO_OD_COLLECTION, ["V1:od:1", "V1:od:2", "V1:od:3"], [[0.1] * 8] * 3,
                     [{"video_id": "V1", "frame_ms": 4080000, "od_detections": "daily chart, sellside capacity line drawn",
                       "source_type": "od_frame", "title": "2022 Acme Mentorship Episode 12"},
                      {"video_id": "V1", "frame_ms": 900000, "od_detections": "far earlier chart",
                       "source_type": "od_frame", "title": "2022 Acme Mentorship Episode 12"},
                      {"video_id": "V2", "frame_ms": 4080000, "od_detections": "other video same time",
                       "source_type": "od_frame", "title": "Other"}],
                     sparse_vectors=[{}, {}, {}])
    reranker_service._reranker = _TagReranker()

    async def sv(sub, sp=None, video_id=None, **kw):
        return []

    res = await _with_video_fakes(_plan(["what does he draw"], video_name="episode 12"), sv,
                                  lambda: retrieval.search("what does he draw at 1:08:00 in episode 12",
                                                           collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=_EP12)
    frames = [r for r in res if r["payload"].get("source_type") == "od_frame"]
    # od_detections is the field ingestion writes (video_ingestion.py) -- copied from a live payload.
    assert [f["payload"]["od_detections"] for f in frames] == ["daily chart, sellside capacity line drawn"], res
    assert any(r["payload"].get("source_type") == "transcript_segment" for r in res), res
    reset_singletons()


async def _ocr_index(svc):
    await svc.ensure_collection(config.VIDEO_OCR_COLLECTION, 8, enable_sparse=False)
    rows = [("V1", 4079000, "Close position Short 3 @ 4 223.75"),
            ("V1", 4080000, "Close position Short 3 @ 4 223.75"),     # same screen again
            ("V1", 900000, "Account Balance 100 000.00"),
            ("V1", 1200000, "no numbers here"),
            ("V2", 4080000, "other video 9 999.00")]
    await svc.upsert(config.VIDEO_OCR_COLLECTION, ["%s:ocr:%d" % (v, i) for i, (v, _, _) in enumerate(rows)], [[0.1] * 8] * len(rows),
                     [{"video_id": v, "frame_ms": ms, "ocr_text": t, "source_type": "ocr_frame",
                       "title": "2022 Acme Mentorship Episode 12"} for v, ms, t in rows],
                     sparse_vectors=[{}] * len(rows))


async def test_named_video_time_includes_screen_text_at_that_moment():
    """2026-10-29: "the exact entry price ... shown in the order window at 0:20"
    got only the chart description and the answer said the price wasn't in the
    knowledge base; the frame's OCR text read "Short 3 @ 4 223.75"."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    await _ocr_index(svc)
    reranker_service._reranker = _TagReranker()

    async def sv(sub, sp=None, video_id=None, **kw):
        return []

    res = await _with_video_fakes(_plan(["entry price"], video_name="episode 12"), sv,
                                  lambda: retrieval.search("what is the entry price at 1:08:00 in episode 12",
                                                           collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=_EP12)
    texts = [r["payload"]["ocr_text"] for r in res if r["payload"].get("source_type") == "ocr_frame"]
    assert texts == ["Close position Short 3 @ 4 223.75"], res     # one per screen, V1 only, in the window
    reset_singletons()


async def test_price_question_on_a_named_video_returns_its_screen_prices():
    """2026-10-29: "which price levels are shown" on a named video was answered
    from chart descriptions ("near 4227.50") although its frames read exact
    prices. Its distinct priced screens come first; text without a price and
    other videos' screens don't."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    await _ocr_index(svc)
    reranker_service._reranker = _TagReranker()

    async def sv(sub, sp=None, video_id=None, **kw):
        return []

    res = await _with_video_fakes(_plan(["price levels"], video_name="episode 12"), sv,
                                  lambda: retrieval.search("which price levels are shown in episode 12",
                                                           collections=[config.VIDEO_TRANSCRIPT_COLLECTION]),
                                  catalog=_EP12)
    texts = [r["payload"]["ocr_text"] for r in res if r["payload"].get("source_type") == "ocr_frame"]
    assert texts == ["Account Balance 100 000.00", "Close position Short 3 @ 4 223.75"], res
    reset_singletons()


async def test_top_transcript_passages_survive_fusion_crowding():
    """the client, 2026-10-22: a general "what is a service level agreement?" never surfaced
    Episode 12's explanation -- it ranked 31st in the transcript track and the
    four-track fusion filled its 40 slots with chart frames of the frame-heavy
    video. The transcript track's own top passages always reach the reranker."""
    import sys as _sys
    import pipelines.video_fusion, pipelines.video_embedder  # noqa: F401  (ensure loaded)
    # Patch the module objects _search_video's in-function imports will resolve
    # (sys.modules), not a possibly stale reference an earlier test left behind.
    video_fusion = _sys.modules["pipelines.video_fusion"]
    video_embedder = _sys.modules["pipelines.video_embedder"]
    Document = video_fusion.Document          # haystack's, or the module's own fallback

    frames = [Document(id="F%d" % i, content="", meta={"video_id": "Z", "frame_ms": i * 1000,
                                                        "source_type": "od_frame", "description": "frame"})
              for i in range(40)]
    asr = [Document(id="Ep12:asr:90", content="why does the market create a service level agreement",
                    meta={"video_id": "V1", "start_time": 4075.0, "text": "why does the market create a service level agreement",
                          "source_type": "transcript_segment"})]

    class FakeRetriever:
        def __init__(self, **kw):
            pass

        async def run(self, **kw):
            return {"documents": frames, "asr_documents": asr}

    class Emb:
        visual_embedding = [0.1] * 8
        voyage_embedding = [0.1] * 8

    saved = (video_fusion.MomentSearchRetriever, video_embedder.embed_query)
    video_fusion.MomentSearchRetriever = FakeRetriever
    video_embedder.embed_query = lambda q: Emb()
    try:
        cands = await retrieval._search_video("what is a service level agreement")
    finally:
        video_fusion.MomentSearchRetriever, video_embedder.embed_query = saved
    ids = [c["id"] for c in cands]
    assert "Ep12:asr:90" in ids, ids
    assert len(ids) == len(set(ids)), "no duplicates"


# ── M3 scoring: video floor, named video, spoken first, what-next, wrong video ──

def _vhit(vid, video, text, t, source_type="transcript_segment", end=None):
    """A video candidate as _search_video shapes it; frames carry frame_ms (ms), transcripts start/end (s)."""
    if source_type == "transcript_segment":
        payload = {"video_id": video, "text": text, "start_time": t, "end_time": end if end is not None else t + 58,
                   "source_type": source_type, "title": video}
        col = config.VIDEO_TRANSCRIPT_COLLECTION
    else:
        payload = {"video_id": video, "od_detections": text, "frame_ms": int(t * 1000),
                   "source_type": source_type, "title": video}
        col = config.VIDEO_OD_COLLECTION
    return {"id": vid, "collection": col, "payload": payload}


async def _video_only(question, plan, sv, catalog=None):
    return await _with_video_fakes(plan, sv, lambda: retrieval.search(
        question, collections=[config.VIDEO_TRANSCRIPT_COLLECTION]), catalog=catalog)


async def test_video_floor_is_separate_from_the_text_floor():
    """I-4 (M3 Layer 1, 2026-10-22): spoken, machine-transcribed passages that
    answer a paraphrased question rerank below the text floor even when exactly
    right (Ep 36 "which stops did he target": the right window #1 at 0.629).
    Video gets its own, lower floor; the text floor is untouched."""
    assert retrieval.MIN_RERANK_SCORE == 0.70
    assert retrieval.VIDEO_MIN_RERANK_SCORE < retrieval.MIN_RERANK_SCORE
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    mid = round((retrieval.VIDEO_MIN_RERANK_SCORE + retrieval.MIN_RERANK_SCORE) / 2, 3)
    q = "which stops did he target"
    reranker_service._reranker = _TagReranker({(q, "he bought below the old lows"): mid})

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_vhit("E36:asr:5", "E36", "he bought below the old lows", 283.0)]

    res = await _video_only(q, _plan([q]), sv)
    assert [r["id"] for r in res] == ["E36:asr:5"], res
    reset_singletons()


async def test_video_noise_band_still_returns_nothing():
    """The not-covered negatives (Wyckoff, Bollinger, Kubernetes, Gamma Reversal
    Block) must still get no video result: their best video candidates
    measured live sit below VIDEO_MIN_RERANK_SCORE (see the constant)."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    q = "Kubernetes pods"
    noise = retrieval.VIDEO_MIN_RERANK_SCORE - 0.01
    reranker_service._reranker = _TagReranker({(q, "talks about something else"): noise,
                                               (q, "a chart with bars"): noise - 0.05})

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_vhit("M27:asr:9", "M27", "talks about something else", 412.0),
                _vhit("E12:od:3", "E12", "a chart with bars", 333.0, "od_frame")]

    assert await _video_only("What does Acme say about Kubernetes pods in the videos?", _plan([q]), sv) == []
    reset_singletons()


async def test_named_video_keeps_its_best_passages_below_the_floor():
    """I-4: when the user names a video, its best passages are what they mean,
    even below the video floor -- the top NAMED_VIDEO_KEEP above a low absolute
    minimum. Unnamed, the same scores return nothing."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    q = "partial rollout rules"
    lo = retrieval.NAMED_VIDEO_MIN_SCORE
    below = retrieval.VIDEO_MIN_RERANK_SCORE - 0.02
    texts = {"first partial four contracts": below, "second partial": below - 0.01,
             "third weaker": below - 0.02, "junk": lo - 0.05}
    reranker_service._reranker = _TagReranker({(q, t): s for t, s in texts.items()})

    async def sv(sub, sp=None, video_id=None, **kw):
        if sub != q:
            return []
        return [_vhit("E36:%d" % i, "E36", t, 400.0 + i * 45) for i, t in enumerate(texts)]

    named = await _video_only("Which releases break the partial-rollout rules from Episode 36?",
                              _plan([q], video_name="Episode 36"), sv,
                              catalog=[("E36", "2022 Acme Mentorship Episode 36")])
    assert [r["id"] for r in named] == ["E36:0", "E36:1"], named
    unnamed = await _video_only("Which releases break the partial-rollout rules?", _plan([q]), sv)
    assert unnamed == [], unnamed
    reset_singletons()


def test_visual_and_followup_question_detectors():
    for q in ("Do any of the videos show the labels STH or STL on a chart?",
              "What does the speaker draw or mark on the chart around 1:08:00?",
              "Is there a chart where an app settings window covers the price?",
              "Which video shows a 'Friday NDOG' zone?",
              "What was on screen when he described the daily service level agreement?"):
        assert retrieval._is_visual_question(q), q
    for q in ("Why does price come back to a service level agreement?",
              "What is consequent encroachment?",
              "In Episode 36, at what prices did he sell his contracts and how many?"):
        assert not retrieval._is_visual_question(q), q
    for q in ("In Episode 12, what happens after price rebalances the service level agreement?",
              "In Episode 36, what happened after the spare capacity was taken at the 9:30 open?",
              "what did he do next?", "and then what happens",
              # rag_filter 1.34 appends the bare follow-up to its anchor (2026-10-23)
              "what about at 44:30? then what?", "then what?", "what next?", "and after that?"):
        assert retrieval._asks_what_next(q), q
    for q in ("What is a shift in system structure after spare capacity is taken?",
              "What happens at 1:07:55 in episode 12?", "What is a service level agreement?"):
        assert not retrieval._asks_what_next(q), q


async def test_spoken_explanation_precedes_chart_descriptions():
    """I-5: "Why does price come back to a service level agreement?" reranked five
    chart descriptions (0.72-0.79) above Episode 12's spoken explanation
    (0.47-0.59), so the explanation never reached the answer. For a concept
    question the spoken passage comes first; for a visual question the chart
    still wins."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    q = "why does price return to a service level agreement"
    spoken = retrieval.VIDEO_MIN_RERANK_SCORE + 0.01
    # Frames just inside the preference margin above the spoken passage.
    top = spoken + retrieval.SPOKEN_PREFERENCE - 0.01
    scores = {(q, "chart %d" % i): round(top - i * 0.01, 4) for i in range(6)}
    scores[(q, "the system goes back to rebalance the load")] = spoken
    reranker_service._reranker = _TagReranker(scores)

    async def sv(sub, sp=None, video_id=None, **kw):
        if sub != q:
            return []
        return ([_vhit("F%d" % i, "M27", "chart %d" % i, 100.0 + i, "od_frame") for i in range(6)]
                + [_vhit("E12:asr:1", "E12", "the system goes back to rebalance the load", 4075.0)])

    res = await _video_only("Why does price come back to a service level agreement?", _plan([q]), sv)
    assert res and res[0]["id"] == "E12:asr:1", [r["id"] for r in res]
    assert res[0]["score"] == spoken, "the reported score stays the reranker's"
    visual = await _video_only("What is drawn on the chart at the service level agreement?", _plan([q]), sv)
    assert visual[0]["id"] == "F0", [r["id"] for r in visual]
    reset_singletons()


async def test_spoken_preference_does_not_lift_a_weak_transcript_over_a_strong_frame():
    """The preference is a margin, not an override: a transcript far below the
    frames (beyond SPOKEN_PREFERENCE) stays where the reranker put it."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    q = "why does price return to a service level agreement"
    weak = 0.95 - retrieval.SPOKEN_PREFERENCE - 0.05
    assert weak >= retrieval.VIDEO_MIN_RERANK_SCORE, "fixture: the weak passage must clear the floor"
    reranker_service._reranker = _TagReranker({(q, "chart"): 0.95, (q, "spoken"): weak})

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_vhit("F0", "M27", "chart", 100.0, "od_frame"), _vhit("T0", "E12", "spoken", 4075.0)] if sub == q else []

    res = await _video_only("Why does price come back to a service level agreement?", _plan([q]), sv)
    assert [r["id"] for r in res] == ["F0", "T0"], [r["id"] for r in res]
    reset_singletons()


async def test_what_happens_next_returns_the_following_windows():
    """I-7: "In Episode 12, what happens after traffic rebalances the load
    balancer?" matched the 1:07:55 window, but the answer is the window AFTER it
    (1:09:31-1:10:30). A what-next question also gets the next transcript
    windows of the same video; any other question does not."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    q = "price rebalances the service level agreement"
    reranker_service._reranker = _TagReranker()

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_vhit("V1:asr:1", "V1", "price rebalances the service level agreement", 4075.28, end=4133.94)] if sub == q else []

    res = await _video_only("In Episode 12, what happens after price rebalances the service level agreement?",
                            _plan([q], video_name="episode 12"), sv, catalog=_EP12)
    starts = [(r["payload"]["video_id"], r["payload"]["start_time"]) for r in res]
    assert starts[0] == ("V1", 4075.28), starts
    assert ("V1", 4127.6) in starts, "the next window must follow: %r" % starts
    assert ("V1", 6000.0) not in starts and ("V2", 4070.0) not in starts, starts
    plain = await _video_only("In Episode 12, why does price rebalance the service level agreement?",
                              _plan([q], video_name="episode 12"), sv, catalog=_EP12)
    assert [r["payload"]["start_time"] for r in plain] == [4075.28], plain
    reset_singletons()


async def test_what_next_after_a_time_continues_past_that_time():
    """Live 2026-10-23: "...what about at 44:30? then what?" (rag_filter 1.34's
    anchored follow-up) returned only the 44:25 window again -- the moment lookup
    answered first and the what-next windows were never added. A what-next
    question with a time gets the passages at that time AND the ones after."""
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _timed_index(fake, svc)
    reranker_service._reranker = _TagReranker()

    async def sv(sub, sp=None, video_id=None, **kw):
        return []

    res = await _video_only("In Episode 12, what does he explain at 1:07:55? then what?",
                            _plan(["service level agreement"], video_name="episode 12"), sv, catalog=_EP12)
    starts = [r["payload"]["start_time"] for r in res]
    assert 4075.28 in starts and 4127.6 in starts, starts
    assert starts.index(4075.28) < starts.index(4127.6), starts
    plain = await _video_only("In Episode 12, what does he explain at 1:07:55?",
                              _plan(["service level agreement"], video_name="episode 12"), sv, catalog=_EP12)
    assert 6000.0 not in [r["payload"]["start_time"] for r in plain], plain
    reset_singletons()


async def test_wrong_named_video_falls_back_to_the_whole_library():
    """I-6: "In the wrong video, Episode 36, where does he explain the octant
    grid?" filtered to Episode 36 and found nothing (best 0.31) -- the grid is
    explained in the X video. Nothing relevant in the named video -> the same
    sub-queries run unfiltered, so the answer can say where it really is."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    q = "octant grid explanation"
    reranker_service._reranker = _TagReranker(
        {(q, "something unrelated in ep36"): retrieval.NAMED_VIDEO_MIN_SCORE - 0.05})
    seen = []

    async def sv(sub, sp=None, video_id=None, **kw):
        seen.append((sub, video_id))
        if sub != q:
            return []
        if video_id == "E36":
            return [_vhit("E36:asr:1", "E36", "something unrelated in ep36", 52.0)]
        return [_vhit("XP:asr:9", "XP", "the octant grid explanation from the daily imbalance", 447.0)]

    res = await _video_only("In the wrong video, Episode 36, where does he explain the octant grid?",
                            _plan([q], video_name="Episode 36"), sv,
                            catalog=[("E36", "2022 Acme Mentorship Episode 36"), ("XP", "Acme post on X")])
    assert [r["id"] for r in res] == ["XP:asr:9"], res
    assert (q, None) in seen, seen
    reset_singletons()


async def test_named_video_with_an_answer_is_not_widened():
    """The fallback runs only when the named video has nothing above the floor:
    a named video that answers stays the only video searched for that answer."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    q = "octant grid explanation"
    reranker_service._reranker = _TagReranker()
    seen = []

    async def sv(sub, sp=None, video_id=None, **kw):
        seen.append((sub, video_id))
        if sub != q:
            return []
        return [_vhit("%s:asr:1" % (video_id or "XP"), video_id or "XP", "octant grid explanation here", 447.0)]

    # Named by title words, not "X": a platform word resolves only with platform
    # data (I-1), which this fixture doesn't supply.
    res = await _video_only("In the octant grid lesson, where does he explain the octant grid?",
                            _plan([q], video_name="octant grid lesson"), sv,
                            catalog=[("E36", "2022 Acme Mentorship Episode 36"), ("XP", "Acme octant grid lesson")])
    assert [r["id"] for r in res] == ["XP:asr:1"], res
    assert (q, None) not in seen, seen
    reset_singletons()


async def test_visual_question_reranks_the_frame_tracks_top_descriptions():
    """M3 E-class, 2026-10-23: "Is there a chart where an app settings
    window covers the price?" -- the X video's four frames described as exactly
    that ranked 25th-99th in the frame-description track, and the four-track
    fusion kept only its first ~10, all "Symbol Search" pop-ups from another
    video. A visual question's answer IS a frame description, so that track's
    own top VIDEO_FRAME_GUARANTEE reach the reranker; other questions unchanged."""
    import sys as _sys
    import pipelines.video_fusion, pipelines.video_embedder  # noqa: F401
    video_fusion = _sys.modules["pipelines.video_fusion"]
    video_embedder = _sys.modules["pipelines.video_embedder"]
    Document = video_fusion.Document
    fused = [Document(id="F%d" % i, content="", meta={"video_id": "M27", "frame_ms": i * 1000,
                                                       "source_type": "od_frame", "od_detections": "symbol search"})
             for i in range(40)]
    od = fused[:24] + [Document(id="XP:od:760", content="", meta={
        "video_id": "XP", "frame_ms": 760000, "source_type": "od_frame",
        "od_detections": "chart partially obscured by an app settings window"})]

    class FakeRetriever:
        def __init__(self, **kw):
            pass

        async def run(self, **kw):
            return {"documents": fused, "asr_documents": [], "od_documents": od}

    class Emb:
        visual_embedding = [0.1] * 8
        voyage_embedding = [0.1] * 8

    saved = (video_fusion.MomentSearchRetriever, video_embedder.embed_query)
    video_fusion.MomentSearchRetriever = FakeRetriever
    video_embedder.embed_query = lambda q: Emb()
    try:
        plain = await retrieval._search_video("settings window")
        visual = await retrieval._search_video("settings window", frame_guarantee=retrieval.VIDEO_FRAME_GUARANTEE)
    finally:
        video_fusion.MomentSearchRetriever, video_embedder.embed_query = saved
    assert "XP:od:760" not in [c["id"] for c in plain]
    ids = [c["id"] for c in visual]
    assert "XP:od:760" in ids and len(ids) == len(set(ids)), ids
    assert [c for c in visual if c["id"] == "XP:od:760"][0]["collection"] == config.VIDEO_OD_COLLECTION
    assert retrieval.VIDEO_FRAME_GUARANTEE >= 25, "the measured miss sat at rank 25"

    # The leg asks for the guarantee only for a visual question.
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    reranker_service._reranker = _TagReranker()
    asked = []

    async def sv(q, sp=None, video_id=None, **kw):
        asked.append(kw.get("frame_guarantee", 0))
        return []

    await _video_only("Is there a chart where an app settings window covers the price?", _plan(["settings window"]), sv)
    assert asked and set(asked) == {retrieval.VIDEO_FRAME_GUARANTEE}, asked
    asked.clear()
    await _video_only("Why does price come back to a service level agreement?", _plan(["why price returns"]), sv)
    assert asked and set(asked) == {0}, asked
    reset_singletons()


async def test_video_named_only_in_the_question_still_filters():
    """M3 J/D, 2026-10-23: "Summarize the video he posted on X." was planned
    intent=summary with video_name=null, so no overview was fetched and nothing
    came back; "Was the March 27 demo profitable..." named no video either, and
    the right window (0.53) fell under the unnamed floor. When the planner names
    no video, a date or platform reference in the question that resolves to
    exactly ONE indexed video names it. Two different videos referenced
    (a comparison) -> no filter."""
    from pipelines import query_planner
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    reranker_service._reranker = _TagReranker()
    seen = []

    subs = []

    async def sv(q, sp=None, video_id=None, **kw):
        seen.append(video_id)
        subs.append((q, video_id))
        return []

    catalog = [("M27", "Acme 2026 Weekly Review, March 27, 2026"),
               ("E12", "2022 Acme Mentorship Episode 12"), ("XP", "I don't have time for YouTube")]
    saved = dict(query_planner._indexed_cache)
    query_planner._indexed_cache.update(platforms={"M27": "youtube", "E12": "youtube", "XP": "twitter"})
    try:
        await _video_only("Summarize the video he posted on X.", _plan(["summary of the video"], intent="summary"),
                          sv, catalog=catalog)
        assert "XP" in seen, seen
        seen.clear()
        # Once the date has become the filter it is noise in the query: "March 27
        # demo result" ranked garbled OCR and title cards carrying the date above
        # the spoken outcome (0.62 vs 0.40, 2026-10-23).
        await _video_only("Was the March 27 demo successful in the end?", _plan(["March 27 demo outcome"]), sv,
                          catalog=catalog)
        assert "M27" in seen, seen
        assert ("demo outcome", "M27") in subs, subs
        seen.clear()
        await _video_only("Compare how the load balancer is explained in Episode 12 with the March 27 review.",
                          _plan(["load balancer"]), sv, catalog=catalog)
        assert seen and set(seen) == {None}, seen
        seen.clear()
        # m3-053 (sprint 2026-10-25): the planner copied ONE of the two compared videos
        # into video_name (Gemini: "Episode 12" 3/3 runs on the old prompt, "March 27
        # review" 2/3 on the new) and the search was filtered to it -- the other half
        # of the comparison was never searched. Two videos referenced: no filter.
        for named in ("March 27 review", "Episode 12"):
            await _video_only("Compare how the load balancer is explained in Episode 12 with the March 27 review.",
                              _plan(["load balancer"], video_name=named), sv, catalog=catalog)
            assert seen and set(seen) == {None}, (named, seen)
            seen.clear()
        await _video_only("What is a service level agreement?", _plan(["service level agreement"]), sv, catalog=catalog)
        assert seen and set(seen) == {None}, seen
    finally:
        query_planner._indexed_cache.clear()
        query_planner._indexed_cache.update(saved)
    reset_singletons()


async def test_wrong_named_video_fallback_keeps_the_best_elsewhere():
    """I-6, 2026-10-23: the widened search for "In the wrong video, Episode 36,
    where does he explain the octant grid?" found the X video's octant passages
    at 0.52-0.57 -- some runs one sub-query cleared the 0.55 floor, some none,
    and the question returned nothing. The user named a video, so the topic is
    expected to exist: the widened results keep their best like a named video
    (top NAMED_VIDEO_KEEP down to NAMED_VIDEO_MIN_SCORE). Unrelated junk below
    that minimum still returns nothing."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    q = "octant grid"
    mid = round((retrieval.NAMED_VIDEO_MIN_SCORE + retrieval.VIDEO_MIN_RERANK_SCORE) / 2, 3)
    reranker_service._reranker = _TagReranker({
        (q, "something unrelated in ep36"): retrieval.NAMED_VIDEO_MIN_SCORE - 0.1,
        (q, "the octants the quadrants of the daily range"): mid,
        (q, "noise elsewhere"): retrieval.NAMED_VIDEO_MIN_SCORE - 0.05})

    async def sv(sub, sp=None, video_id=None, **kw):
        if video_id == "E36":
            return [_vhit("E36:asr:1", "E36", "something unrelated in ep36", 52.0)]
        return [_vhit("XP:asr:9", "XP", "the octants the quadrants of the daily range", 566.0),
                _vhit("M27:asr:3", "M27", "noise elsewhere", 791.0)]

    catalog = [("E36", "2022 Acme Mentorship Episode 36"), ("XP", "Acme post on X")]
    res = await _video_only("In the wrong video, Episode 36, where does he explain the octant grid?",
                            _plan([q], video_name="Episode 36"), sv, catalog=catalog)
    assert [r["id"] for r in res] == ["XP:asr:9"], res
    reset_singletons()


def test_combined_follow_up_uses_the_latest_time():
    """rag_filter 1.33 searches a follow-up that names no video as "<previous
    user message> <follow-up>". The latest message comes last, so the LAST
    position in the query is the one asked about: "...around 25 minutes in ...
    what about at 44:30?" is 44:30, not 25:00. Times of day and the duration
    rule still skip a time; the "9:30 open ... at 25:00" case is unchanged."""
    from pipelines.retrieval import parse_timestamp
    q ="In episde 12, around 25 minutes in, what is he forecasting? what about at 44:30?"
    assert parse_timestamp(q) == 2670, parse_timestamp(q)
    assert parse_timestamp(q, duration=8150.0) == 2670
    # 44:30 is past a 30-minute video's end: the earlier time inside it stands.
    assert parse_timestamp(q, duration=1800.0) == 1500
    assert parse_timestamp("what does he say at 15:30? and what about after the 9:30 open?") == 930
    assert parse_timestamp("after the 9:30 open, what does he say at 25:00") == 1500
    assert parse_timestamp("In Episode 36, what happens at 2:00? what about 9:30 AM") == 120


async def _x_video_index(svc):
    """The X video (XP) with an overview, for the planner-fallback tests."""
    await svc.ensure_collection(config.VIDEO_TRANSCRIPT_COLLECTION, 8, enable_sparse=False)
    await svc.upsert(config.VIDEO_TRANSCRIPT_COLLECTION, ["XP:overview"], [[0.1] * 8],
                     [{"video_id": "XP", "title": "I don't have time for YouTube", "text": "Main concepts: octants",
                       "overview": True, "start_time": 0.0, "source_type": "transcript_segment"}],
                     sparse_vectors=[{}])


_X_CATALOG = [("M27", "Acme 2026 Weekly Review, March 27, 2026"),
              ("E12", "2022 Acme Mentorship Episode 12"), ("XP", "I don't have time for YouTube")]


async def test_over_budget_planner_still_resolves_the_named_video():
    """Round 3, issue 2 (live 2026-10-23 12:11 UTC, six parallel chats): the
    planner went over PLANNER_BUDGET_S and the leg fell back to the original
    question searched across every video. "Summarize the video he posted on X."
    matches no passage across the library, so it returned 0 results and the reply
    said there is no X video. The video reference and the "summarize" cue are in
    the question itself -- resolving them needs no model: the X video's overview
    and its own passages must come back without the planner."""
    import time as _time
    from pipelines import query_planner
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _x_video_index(svc)
    reranker_service._reranker = _TagReranker()
    seen = []

    def slow_plan(question):
        _time.sleep(1.0)
        return _plan(["planned"], intent="summary")

    async def sv(sub, sp=None, video_id=None, **kw):
        seen.append(video_id)
        return [_video_hit("XP:asr:1", "octant passage " + sub)] if video_id == "XP" else []

    saved = (retrieval.PLANNER_BUDGET_S, dict(query_planner._indexed_cache))
    retrieval.PLANNER_BUDGET_S = 0.2
    query_planner._indexed_cache.update(platforms={"M27": "youtube", "E12": "youtube", "XP": "twitter"})
    try:
        res = await _video_only("Summarize the video he posted on X.", slow_plan, sv, catalog=_X_CATALOG)
    finally:
        retrieval.PLANNER_BUDGET_S = saved[0]
        query_planner._indexed_cache.clear()
        query_planner._indexed_cache.update(saved[1])
    assert "XP" in seen, seen
    assert res and res[0]["payload"].get("overview") is True and res[0]["payload"]["video_id"] == "XP", res
    assert any(r["id"] == "XP:asr:1" for r in res), res
    reset_singletons()


async def test_failed_planner_still_resolves_the_named_video():
    """Same when the planner fails outright (plan == the original question):
    the deterministic reference still filters. A question naming no video
    keeps the unfiltered original-question search exactly as before."""
    from pipelines import query_planner
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _x_video_index(svc)
    reranker_service._reranker = _TagReranker()
    seen = []

    async def sv(sub, sp=None, video_id=None, **kw):
        seen.append((sub, video_id))
        return [_video_hit("XP:asr:1", "octant passage " + sub)] if video_id == "XP" else []

    saved = dict(query_planner._indexed_cache)
    query_planner._indexed_cache.update(platforms={"M27": "youtube", "E12": "youtube", "XP": "twitter"})
    try:
        q = "Summarize the video he posted on X."
        res = await _video_only(q, _plan([q]), sv, catalog=_X_CATALOG)
        assert res and res[0]["payload"].get("overview") is True, res
        seen.clear()
        q2 = "where does he explain capacity sweeps"
        await _video_only(q2, _plan([q2]), sv, catalog=_X_CATALOG)
        assert seen == [(q2, None)], seen
        # A definition question (H2) adds the lecturer's why-query -- still unfiltered.
        seen.clear()
        q3 = "what is a service level agreement"
        await _video_only(q3, _plan([q3]), sv, catalog=_X_CATALOG)
        assert sorted(seen) == [("what is a service level agreement", None),
                                ("why does the market create a service level agreement", None)], seen
    finally:
        query_planner._indexed_cache.clear()
        query_planner._indexed_cache.update(saved)
    reset_singletons()


async def test_failed_planner_resolves_a_video_named_by_title():
    """Live 2026-10-29 11:35 UTC: the planner failed (Gemini over 1.5 s, Claude
    fallback timed out) on "* **Pattern Recognition - Fiber RTO NYO and Asian
    Session [aQrd75xwBS4]** tell me about this". The question names the video by
    TITLE, not date/episode/platform, so nothing filtered to it and the chat got
    five bare keyframes. Now: the title resolves, "tell me about this" is a
    summary, and the video's overview comes first."""
    from pipelines import query_planner
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await svc.ensure_collection(config.VIDEO_TRANSCRIPT_COLLECTION, 8, enable_sparse=False)
    title = "2017-10-10 - Pattern Recognition - Cache RTO Sizing and Regional Rollout [aQrd75xwBS4]"
    await svc.upsert(config.VIDEO_TRANSCRIPT_COLLECTION, ["FB:overview"], [[0.1] * 8],
                     [{"video_id": "FB", "title": title, "text": "Main concepts: RTO on Fiber",
                       "overview": True, "start_time": 0.0, "source_type": "transcript_segment"}],
                     sparse_vectors=[{}])
    reranker_service._reranker = _TagReranker()
    seen = []

    async def sv(sub, sp=None, video_id=None, **kw):
        seen.append((sub, video_id))
        return [_video_hit("FB:asr:1", "fiber passage " + sub)] if video_id == "FB" else []

    catalog = _X_CATALOG + [("FB", title)]
    q = "*   **Pattern Recognition - Cache RTO Sizing and Regional Rollout [aQrd75xwBS4]** tell me about this"
    res = await _video_only(q, _plan([q]), sv, catalog=catalog)
    assert res and res[0]["payload"].get("overview") is True and res[0]["payload"]["video_id"] == "FB", res
    # (the raw question is also searched unfiltered in parallel, by design)
    assert any(vid == "FB" for _, vid in seen), seen
    seen.clear()
    q2 = "In Pattern Recognition - Cache RTO Sizing and Regional Rollout, where is the RTO drawn?"
    res = await _video_only(q2, _plan([q2]), sv, catalog=catalog)
    assert any(vid == "FB" for _, vid in seen), seen                   # filtered to the video...
    assert not (res and res[0]["payload"].get("overview")), res         # ...but a specific question gets passages
    reset_singletons()


async def _short_index(svc):
    """Episode 12 cut to 9:11 (551 s): windows at 0:00, 3:00-3:58 and the end."""
    await svc.ensure_collection(config.VIDEO_TRANSCRIPT_COLLECTION, 8)
    rows = [("V1:asr:0", 0.0, 58.0, "intro"), ("V1:asr:3", 180.0, 238.0, "at three ten he marks the low"),
            ("V1:asr:9", 500.0, 551.0, "the end")]
    await svc.upsert(config.VIDEO_TRANSCRIPT_COLLECTION, [r[0] for r in rows], [[0.1] * 8] * len(rows),
                     [{"video_id": "V1", "start_time": s, "end_time": e, "text": t,
                       "source_type": "transcript_segment", "title": "2022 Acme Mentorship Episode 12"}
                      for _, s, e, t in rows])


async def test_time_past_the_end_of_the_named_video_says_so():
    """Round 3, issue 5: "In Episode 36, what happens at 3:10:00?" -- Episode 36
    is 9:11 long, and both models answered vaguely because nothing told them.
    A note result states the length and that the time is past the end; when
    the digits also read as m:ss inside the video (3:10 -> 190 s) the passages
    there follow it. A time with no such reading gets the note plus the usual
    planned search. Inside the video: unchanged. No named video: no note."""
    from pipelines import query_planner
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await _short_index(svc)
    reranker_service._reranker = _TagReranker()

    async def sv(sub, sp=None, video_id=None, **kw):
        return [_video_hit("V1:asr:9", "what happens: the end")]

    saved = dict(query_planner._indexed_cache)
    query_planner._indexed_cache["durations"] = {"V1": 551.0}
    try:
        res = await _video_only("In Episode 12, what happens at 3:10:00?",
                                _plan(["what happens"], video_name="Episode 12"), sv, catalog=_EP12)
        note = res[0]["payload"]
        assert note["source_type"] == "video_library", res
        assert "9:11" in note["text"] and "3:10:00" in note["text"] and "past the end" in note["text"], note
        assert "2022 Acme Mentorship Episode 12" in note["text"] and "3:10 " in note["text"], note
        assert any(r["payload"].get("start_time") == 180.0 for r in res[1:]), res

        res2 = await _video_only("In Episode 12, what happens at 4:00:30?",
                                 _plan(["what happens"], video_name="Episode 12"), sv, catalog=_EP12)
        assert res2[0]["payload"]["source_type"] == "video_library", res2
        assert "4:00:30" in res2[0]["payload"]["text"], res2
        assert [r["id"] for r in res2[1:]] == ["V1:asr:9"], res2

        res3 = await _video_only("In Episode 12, what happens at 3:10?",
                                 _plan(["what happens"], video_name="Episode 12"), sv, catalog=_EP12)
        assert all(r["payload"].get("source_type") != "video_library" for r in res3), res3
        assert [r["payload"]["start_time"] for r in res3] == [180.0], res3

        res4 = await _video_only("what happens at 3:10:00?", _plan(["what happens"]), sv, catalog=_EP12)
        assert all(r["payload"].get("source_type") != "video_library" for r in res4), res4
    finally:
        query_planner._indexed_cache.clear()
        query_planner._indexed_cache.update(saved)
    reset_singletons()


_FOUR = [("E12", "2022 Acme Mentorship Episode 12"), ("E36", "2022 Acme Mentorship Episode 36"),
         ("M27", "Acme 2026 Weekly Review, March 27, 2026"), ("XP", "I don't have time for YouTube")]


def _hit_in(vid, text):
    return {"id": vid + ":asr:1", "collection": config.VIDEO_TRANSCRIPT_COLLECTION,
            "payload": {"video_id": vid, "text": text, "start_time": 10.0, "source_type": "transcript_segment"}}


async def test_the_other_video_excludes_the_named_one():
    """Round 4 (live 2026-10-23): rag_filter 1.33 searched the follow-up as
    "Summarize the video he posted on X. and in the other video?" -- the X video
    resolved again, only it came back (1 result), and both Claude models said "I
    only have one video in the knowledge base" (4 are indexed). The video named
    before "the other video" is an EXCLUSION: every video is searched, the named
    one's results dropped. Summary with several others left -> the library item
    (minus the excluded video) so the model can list them and ask which one;
    exactly one other -> its overview. Enforced in code whatever the planner says,
    and with no plan at all."""
    from pipelines import query_planner
    fake = FakeQdrant()
    svc = await with_stub_pipeline(fake)
    await svc.ensure_collection(config.VIDEO_TRANSCRIPT_COLLECTION, 8, enable_sparse=False)
    await svc.upsert(config.VIDEO_TRANSCRIPT_COLLECTION, ["XP:overview", "E12:overview"], [[0.1] * 8] * 2,
                     [{"video_id": "XP", "title": "I don't have time for YouTube", "text": "### 1. Octants",
                       "overview": True, "start_time": 0.0, "end_time": 900.0, "source_type": "transcript_segment"},
                      {"video_id": "E12", "title": "2022 Acme Mentorship Episode 12", "text": "### 1. SLA",
                       "overview": True, "start_time": 0.0, "end_time": 600.0, "source_type": "transcript_segment"}],
                     sparse_vectors=[{}, {}])
    reranker_service._reranker = _TagReranker()
    seen = []

    async def sv(sub, sp=None, video_id=None, **kw):
        seen.append(video_id)
        vids = [video_id] if video_id else [v for v, _ in _FOUR]
        return [_hit_in(v, sub + " in " + v) for v in vids]

    saved = dict(query_planner._indexed_cache)
    query_planner._indexed_cache.update(platforms={"E12": "youtube", "E36": "youtube", "M27": "youtube",
                                                   "XP": "twitter"})
    try:
        q = "Summarize the video he posted on X. and in the other video?"
        for plan in (_plan(["summary of the video"], video_name="the video he posted on X", intent="summary"),
                     _plan(["summary of the video"], intent="summary"), _plan([q])):
            seen.clear()
            res = await _video_only(q, plan, sv, catalog=_FOUR)
            assert res and set(seen) == {None}, (plan, seen)
            assert all(r["payload"].get("video_id") != "XP" for r in res), (plan, res)
            lib = res[0]["payload"]
            assert lib["source_type"] == "video_library", (plan, res)
            assert "Episode 12" in lib["text"] and "Episode 36" in lib["text"] and "March 27" in lib["text"], lib
            assert "YouTube" not in lib["text"], "the excluded video is not listed: " + lib["text"]

        q2 = "In Episode 36, what is a service level agreement? and in the other video?"
        for plan in (_plan(["what is a service level agreement"], video_name="Episode 36"), _plan(["what is a service level agreement"])):
            seen.clear()
            res = await _video_only(q2, plan, sv, catalog=_FOUR)
            assert set(seen) == {None}, seen
            vids = {r["payload"].get("video_id") for r in res}
            assert "E36" not in vids and vids & {"E12", "M27", "XP"}, res
            assert all(r["payload"].get("source_type") != "video_library" for r in res), res

        seen.clear()
        res = await _video_only("Compare Episode 12 with the other video.",
                                _plan(["load balancer"], video_name="Episode 12"), sv, catalog=_FOUR)
        vids = {r["payload"].get("video_id") for r in res}
        assert "E12" not in vids and vids, res

        # Only one other video: it IS the other one -- its overview answers.
        two = [_FOUR[0], _FOUR[3]]
        res = await _video_only("Not Episode 12 -- summarize the other video.",
                                _plan(["summary"], video_name="Episode 12", intent="summary"), sv, catalog=two)
        assert res[0]["payload"].get("overview") is True and res[0]["payload"]["video_id"] == "XP", res
        assert all(r["payload"].get("video_id") != "E12" for r in res), res

        # No "other": the named video still filters, exactly as before.
        seen.clear()
        await _video_only("Summarize the video he posted on X.", _plan(["summary"], intent="summary"), sv,
                          catalog=_FOUR)
        assert "XP" in seen, seen
    finally:
        query_planner._indexed_cache.clear()
        query_planner._indexed_cache.update(saved)
    reset_singletons()


# ── H2 (sprint 2026-10-25): a general definition question surfaces the defining passage ──

def test_strongest_sub_query_leads_the_merged_video_results():
    """H2 (live 2026-10-25, every trace since 10-23): "According to the videos, what is
    a service level agreement?" planned ["what is a service level agreement", "why does the market create
    a service level agreement"]. The first sub-query's best was a March 27 passage at 0.70, the
    second's Episode 12 1:07:55 at 0.89 -- and _merge put the heads in sub-query order,
    so the model read March 27 first and in one of two chats never cited Episode 12.
    The heads are ordered by score; a pinned result (overview, library list, note)
    still comes first whatever a passage scores."""
    m27 = {**_vhit("M27:asr:1185", "M27", "sla mention", 1185.0), "score": 0.70, "rank_score": 0.85}
    m27b = {**_vhit("M27:asr:5420", "M27", "sla mention 2", 5420.0), "score": 0.68, "rank_score": 0.83}
    e12 = {**_vhit("E12:asr:4075", "E12", "why the market creates a sla", 4075.0), "score": 0.89, "rank_score": 1.04}
    out = retrieval._merge([], [[m27, m27b], [e12]], 5)
    assert [c["id"] for c in out] == ["E12:asr:4075", "M27:asr:1185", "M27:asr:5420"], [c["id"] for c in out]
    overview = {"id": "E12:overview", "collection": config.VIDEO_TRANSCRIPT_COLLECTION, "score": 1.0,
                "payload": {"video_id": "E12", "overview": True, "text": "concepts", "source_type": "transcript_segment"}}
    out = retrieval._merge([], [[overview], [m27], [e12]], 5)
    assert [c["id"] for c in out] == ["E12:overview", "E12:asr:4075", "M27:asr:1185"], [c["id"] for c in out]
    library = {"id": "video-library", "collection": config.VIDEO_TRANSCRIPT_COLLECTION, "score": 1.0,
               "payload": {"text": "4 videos", "source_type": "video_library"}}
    out = retrieval._merge([], [[library], [e12]], 5)
    assert [c["id"] for c in out] == ["video-library", "E12:asr:4075"], [c["id"] for c in out]


async def test_definition_question_always_searches_the_lecture_explanation():
    """H2 / m3-002: the videos explain a concept (why it comes about) rather
    than define it; only "why does the market create a service level agreement" finds Episode
    12 1:07:55 (0.89, rank 1 -- "what is a service level agreement" never reaches it, "why does
    price come back to a SLA" ranks it 25th at 0.47). The planner adds that query
    only when it follows its prompt, and not at all when it is over budget (live
    2026-10-25 18:50: "What is a service level agreement?" ran unplanned). The query is
    now added in code: with a plan that lacks it, and with no plan at all."""
    fake = FakeQdrant()
    await with_stub_pipeline(fake)
    reranker_service._reranker = _TagReranker()
    why = "why does the market create a service level agreement"
    seen = []

    async def sv(sub, sp=None, video_id=None, **kw):
        seen.append(sub)
        return [_vhit("E12:asr:4075", "E12", why + ", to rebalance", 4075.0)] if sub == why else []

    for question in ("According to the videos, what is a service level agreement?",
                     "Why does price come back to a service level agreement?"):
        seen.clear()
        res = await _video_only(question, _plan(["what is a service level agreement"]), sv)
        assert why in seen, (question, seen)
        assert [r["id"] for r in res] == ["E12:asr:4075"], (question, res)
        # Planner over budget / failed: the plan is the question itself.
        seen.clear()
        res = await _video_only(question, lambda q: _plan([q]), sv)
        assert why in seen, ("unplanned", question, seen)
        assert [r["id"] for r in res] == ["E12:asr:4075"], ("unplanned", question, res)
    # A question that defines nothing is searched exactly as before.
    seen.clear()
    await _video_only("In episode 12, what is he forecasting?", lambda q: _plan([q]), sv)
    assert why not in seen and not any(s.startswith("why does the market create") for s in seen), seen
    reset_singletons()


async def main():
    tests = [
        test_embedding_provider_factory, test_voyage_embedding,
        test_qdrant_collection_creation, test_ensure_collection_rejects_an_existing_schema_mismatch,
        test_upsert_splits_requests_under_the_qdrant_size_cap,
        test_dense_search, test_sparse_search,
        test_rrf_fusion, test_reranker,
        test_rerank_reads_the_right_text_field_per_collection_shape,
        test_search_endpoint_integration,
        test_empty_collection_graceful, test_search_with_filters,
        test_expand_by_source_finds_siblings, test_expansion_deduplicates_primaries,
        test_expansion_respects_max_expansion, test_expansion_with_no_siblings,
        test_expansion_config_flag,
        test_low_relevance_results_are_dropped,
        test_relevance_floor_only_applies_to_calibrated_scores,
        test_off_topic_query_drops_the_whole_noise_band,
        test_weakest_real_match_survives_the_floor,
        test_configured_reranker_is_calibrated_so_the_floor_can_fire,
        test_missing_reranker_key_silently_disables_the_floor,
        test_telemetry_compat_never_breaks_retrieval,
        test_comparison_questions_rewrite_to_keyword_terms,
        test_resolve_collections,
        test_video_collections_merge_via_moment_search_retriever,
        test_video_search_failure_degrades_to_text_only,
        test_video_single_list_competes_fairly_against_text_dense_sparse_pair,
        test_timestamp_query_still_searches_text_collections,
        test_timestamp_query_does_not_mutate_the_module_collection_lists,
        test_parse_timestamp,
        test_parse_natural_time_expressions,
        test_halfway_resolves_against_the_named_videos_duration,
        test_clock_times_of_day_are_not_video_positions,
        test_bare_clock_time_is_a_video_position_only_within_the_named_video,
        test_time_of_day_question_does_not_add_a_video_time_window,
        test_text_leg_is_unchanged_by_video_planning,
        test_each_sub_query_is_reranked_against_itself,
        test_named_video_filters_the_video_leg,
        test_multi_part_keeps_one_passage_per_sub_query,
        test_planner_failure_falls_back_to_original_question,
        test_video_results_never_displace_text,
        test_summary_intent_returns_the_video_overview,
        test_slow_video_leg_is_cut_at_the_deadline,
        test_comparison_rescue_still_runs_when_video_passes,
        test_search_video_drops_overview_points,
        test_library_text_lists_videos_with_concepts,
        test_library_intent_answers_from_the_overviews,
        test_build_filter_turns_operator_dicts_into_range_conditions,
        test_named_video_timestamp_returns_the_passages_at_that_moment,
        test_named_video_timestamp_past_its_end_falls_back_to_the_planned_search,
        test_a_time_without_a_named_video_does_not_filter_the_video_leg,
        test_natural_time_in_named_video_returns_the_passages_at_that_moment,
        test_time_of_day_in_named_video_is_not_a_moment,
        test_halfway_in_named_video_uses_its_duration,
        test_library_question_about_content_also_returns_matching_passages,
        test_named_video_time_includes_chart_frames_at_that_moment,
        test_named_video_time_includes_screen_text_at_that_moment,
        test_price_question_on_a_named_video_returns_its_screen_prices,
        test_top_transcript_passages_survive_fusion_crowding,
        test_slow_planner_still_returns_the_original_questions_video_results,
        test_empty_planned_searches_fall_back_to_the_original_question,
        test_video_floor_is_separate_from_the_text_floor,
        test_video_noise_band_still_returns_nothing,
        test_named_video_keeps_its_best_passages_below_the_floor,
        test_visual_and_followup_question_detectors,
        test_spoken_explanation_precedes_chart_descriptions,
        test_spoken_preference_does_not_lift_a_weak_transcript_over_a_strong_frame,
        test_what_happens_next_returns_the_following_windows,
        test_what_next_after_a_time_continues_past_that_time,
        test_wrong_named_video_falls_back_to_the_whole_library,
        test_named_video_with_an_answer_is_not_widened,
        test_visual_question_reranks_the_frame_tracks_top_descriptions,
        test_video_named_only_in_the_question_still_filters,
        test_wrong_named_video_fallback_keeps_the_best_elsewhere,
        test_combined_follow_up_uses_the_latest_time,
        test_over_budget_planner_still_resolves_the_named_video,
        test_failed_planner_still_resolves_the_named_video,
        test_failed_planner_resolves_a_video_named_by_title,
        test_time_past_the_end_of_the_named_video_says_so,
        test_the_other_video_excludes_the_named_one,
        test_strongest_sub_query_leads_the_merged_video_results,
        test_definition_question_always_searches_the_lecture_explanation,
        test_live_qdrant,
    ]
    for test in tests:
        result = test()
        if asyncio.iscoroutine(result):
            await result
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} retrieval checks passed")


if __name__ == "__main__":
    asyncio.run(main())
