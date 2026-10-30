"""
Qdrant access. INGESTION, STRUCTURED and ADMIN sessions import from here.

Talks to Qdrant's REST API with httpx rather than qdrant-client, because
qdrant-client is still commented out in requirements.txt (INFRA owns that file)
and every call we need is one HTTP request. If INFRA enables the package later
this module is the only thing that has to change.

Collections carry two vectors per point:
    "dense"  — embedding-model vectors, cosine
    "text"   — sparse term vector with Qdrant's Idf modifier, for exact-term recall
               (domain jargon: abbreviations, product names, multi-word terms)

Public surface (do not break):
    QdrantService, get_qdrant(), text_to_sparse(), point_id(),
    source_of(), chunk_index()
"""

import hashlib
import json
import logging
import re
import time
import uuid
from collections import Counter

import httpx

import config

logger = logging.getLogger("haystack-api.qdrant")

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "text"
TIMEOUT = httpx.Timeout(30.0, connect=5.0)

# Qdrant rejects request bodies over service.max_request_size_mb (default 32).
# A long video's keyframes alone are ~3000 x ~20 KB of JSON per 1024-d point,
# so one unbatched upsert died with 413 at the index phase -- after every paid
# phase (ASR, Gemini, Voyage) had already run. Half the cap leaves headroom for
# the per-point size estimate being a little low.
UPSERT_MAX_BYTES = 16 * 1024 * 1024
SCROLL_PAGE = 256    # points per scroll request when expanding a source
EXPAND_LIMIT = 1000  # ceiling on one expansion, so a pathologically split file
                     # cannot walk a whole collection into memory

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class QdrantError(RuntimeError):
    """Qdrant rejected a write. Reads degrade to empty instead of raising."""


def text_to_sparse(text: str) -> dict:
    """Text -> {"indices": [...], "values": [...]} for the sparse vector field.

    Tokens are hashed to uint32 slots and weighted by term frequency; Qdrant's
    Idf modifier supplies the document-frequency half at query time.

    ponytail: this is IDF-weighted term frequency, not full BM25 — no length
    normalisation or k1 saturation, so very long chunks score slightly high.
    Fine for jargon recall. Upgrade path is fastembed's Bm25 model, which needs
    INFRA to add the package.
    """
    counts = Counter(_TOKEN_RE.findall(text.lower()))
    if not counts:
        return {"indices": [], "values": []}
    items = sorted(
        (int.from_bytes(hashlib.blake2b(t.encode(), digest_size=4).digest(), "big"), float(c))
        for t, c in counts.items()
    )
    return {"indices": [i for i, _ in items], "values": [v for _, v in items]}


def point_id(raw) -> str:
    """Qdrant only accepts uint64 or UUID ids. Map any other string
    deterministically so the same chunk id always lands on the same point."""
    if isinstance(raw, int):
        return raw
    text = str(raw)
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


def source_of(payload: dict) -> str:
    """The file a chunk came from, in either payload shape.

    Points written before the namespacing carry `source` as the filename string;
    new ones nest it under `source.filename`. Empty when neither is set — callers
    treat that as "not expandable" rather than grouping unrelated chunks together.
    """
    source = (payload or {}).get("source")
    if isinstance(source, dict):
        return str(source.get("filename") or "")
    return str(source or "")


def chunk_index(payload: dict) -> int:
    """Position of a chunk inside its source. Missing or unparseable sorts first."""
    try:
        return int((payload or {}).get("chunk_index") or 0)
    except (TypeError, ValueError):
        return 0


_RANGE_OPS = {"$gte": "gte", "$lte": "lte", "$gt": "gt", "$lt": "lt"}


def build_filter(filters: dict) -> dict:
    """{"source_type": "transcript", "video_id": ["a", "b"]} -> Qdrant filter.

    Scalars become match-value, lists become match-any, and an operator dict
    ({"$gte": 885, "$lte": 945}) becomes a range condition. Returns None when
    empty so callers can pass it straight through.
    """
    if not filters:
        return None
    must = []
    for field, value in filters.items():
        if value is None:
            continue
        if isinstance(value, dict):
            # P100: without this branch an operator dict fell through to
            # match-value -- comparing a number field against a dict, which
            # matches no point (or 400s), so every time-window filter was a
            # silent no-op.
            unknown = set(value) - set(_RANGE_OPS)
            if unknown:
                raise ValueError("unsupported filter operator(s) %r on %r" % (sorted(unknown), field))
            must.append({"key": field, "range": {_RANGE_OPS[op]: v for op, v in value.items()}})
        elif isinstance(value, (list, tuple, set)):
            must.append({"key": field, "match": {"any": list(value)}})
        else:
            must.append({"key": field, "match": {"value": value}})
    return {"must": must} if must else None


class QdrantService:
    def __init__(self, url: str = None):
        self.url = (url or config.QDRANT_URL).rstrip("/")
        self._client = None
        self.upsert_max_bytes = UPSERT_MAX_BYTES

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.url, timeout=TIMEOUT)
        return self._client

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, **kw) -> dict:
        resp = await self.client.request(method, path, **kw)
        if resp.status_code >= 400:
            raise QdrantError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json().get("result")

    # ── collections ──────────────────────────────────────

    async def ensure_collection(self, name: str, dimensions: int,
                                enable_sparse: bool = True) -> None:
        """Create the collection if it does not exist. Idempotent.

        Validates an EXISTING collection's "dense" vector against what the
        caller expects (P113) -- this used to return on existence alone, so
        a collection left over with a different name or dimension let every
        later upsert fail far away from here, with an opaque Qdrant 400
        ("Not existing vector name error: dense") that named the symptom,
        not the cause. Happened live 2026-10-12 to `visual_frames`.
        """
        existing = await self.collection_info(name)
        if existing is not None:
            vectors = ((existing.get("config") or {}).get("params") or {}).get("vectors") or {}
            dense = vectors.get(DENSE_VECTOR)
            if dense is None or dense.get("size") != dimensions:
                found = f"{DENSE_VECTOR!r} sized {dense.get('size')}" if dense else \
                    f"no {DENSE_VECTOR!r} vector (has: {sorted(vectors) or 'none'!r})"
                raise QdrantError(
                    f"collection {name!r} already exists with an incompatible schema "
                    f"({found}, expected {DENSE_VECTOR!r} sized {dimensions}) -- delete "
                    f"and recreate it rather than upsert into a mismatched collection"
                )
            return

        body = {"vectors": {DENSE_VECTOR: {"size": dimensions, "distance": "Cosine"}}}
        if enable_sparse:
            body["sparse_vectors"] = {SPARSE_VECTOR: {"modifier": "idf"}}

        resp = await self.client.put(f"/collections/{name}", json=body)
        # 409 == someone else created it between our check and this call.
        if resp.status_code >= 400 and resp.status_code != 409:
            raise QdrantError(f"create collection {name} -> {resp.status_code}: {resp.text[:300]}")
        logger.info("created collection %s (%d dims, sparse=%s)", name, dimensions, enable_sparse)

    async def create_keyword_index(self, collection: str, field: str) -> None:
        """Keyword payload index on one field. Re-creating an existing one is a
        no-op in Qdrant, so callers need not track it. wait=false: building the
        index on a big collection must not hold up the ingest that asked."""
        await self._request("PUT", f"/collections/{collection}/index",
                            params={"wait": "false"},
                            json={"field_name": field, "field_schema": "keyword"})

    async def collection_info(self, collection: str) -> dict:
        """Point count, status and config. None if the collection does not exist."""
        resp = await self.client.get(f"/collections/{collection}")
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise QdrantError(f"collection_info {collection} -> {resp.status_code}")
        return resp.json().get("result")

    # ── writes ───────────────────────────────────────────

    async def upsert(self, collection: str, ids: list, vectors: list[list[float]],
                     payloads: list[dict], sparse_vectors: list[dict] = None) -> None:
        """Insert or update points. Payloads carry all chunk metadata.

        sparse_vectors is optional; when omitted it is derived from each
        payload's "content" so callers cannot forget to make BM25 work.
        """
        if not ids:
            return
        if not (len(ids) == len(vectors) == len(payloads)):
            raise QdrantError(
                f"length mismatch: {len(ids)} ids, {len(vectors)} vectors, {len(payloads)} payloads"
            )
        if sparse_vectors is not None and len(sparse_vectors) != len(ids):
            raise QdrantError(
                f"length mismatch: {len(ids)} ids, {len(sparse_vectors)} sparse vectors"
            )

        points = []
        for i, (pid, vec, payload) in enumerate(zip(ids, vectors, payloads)):
            vector = {DENSE_VECTOR: vec}
            sparse = (
                sparse_vectors[i] if sparse_vectors
                else text_to_sparse(payload.get("content", ""))
            )
            if sparse and sparse.get("indices"):
                vector[SPARSE_VECTOR] = sparse
            points.append({"id": point_id(pid), "vector": vector, "payload": payload})

        started = time.perf_counter()
        batches = 0
        for batch in self._size_batches(points):
            await self._request("PUT", f"/collections/{collection}/points",
                                params={"wait": "true"}, json={"points": batch})
            batches += 1
        logger.info("upserted %d points into %s in %d request(s), %.0fms",
                    len(points), collection, batches,
                    (time.perf_counter() - started) * 1000)

    def _size_batches(self, points: list):
        """Yield runs of points whose JSON stays under upsert_max_bytes."""
        batch, size = [], 0
        for point in points:
            n = len(json.dumps(point, separators=(",", ":")))
            if batch and size + n > self.upsert_max_bytes:
                yield batch
                batch, size = [], 0
            batch.append(point)
            size += n
        if batch:
            yield batch

    async def delete_by_filter(self, collection: str, filter_dict: dict) -> int:
        """Delete points matching filter. Returns the number deleted."""
        qf = build_filter(filter_dict)
        if qf is None:
            raise QdrantError("delete_by_filter needs a filter — refusing to wipe a collection")
        if await self.collection_info(collection) is None:
            return 0

        counted = await self._request(
            "POST", f"/collections/{collection}/points/count",
            json={"filter": qf, "exact": True},
        )
        count = (counted or {}).get("count", 0)
        if count:
            await self._request("POST", f"/collections/{collection}/points/delete",
                                params={"wait": "true"}, json={"filter": qf})
        logger.info("deleted %d points from %s", count, collection)
        return count

    # ── reads ────────────────────────────────────────────

    async def search_dense(self, collection: str, vector: list[float],
                           top_k: int = 50, filters: dict = None) -> list[dict]:
        """Dense vector search. Returns [{id, score, payload}, ...]."""
        return await self._query(collection, vector, DENSE_VECTOR, top_k, filters)

    async def search_sparse(self, collection: str, query_text: str,
                            top_k: int = 50, filters: dict = None) -> list[dict]:
        """Sparse / BM25-style search. Returns [{id, score, payload}, ...]."""
        sparse = text_to_sparse(query_text)
        if not sparse["indices"]:
            return []
        return await self._query(collection, sparse, SPARSE_VECTOR, top_k, filters)

    async def _query(self, collection: str, query, using: str,
                     top_k: int, filters: dict) -> list[dict]:
        body = {"query": query, "using": using, "limit": top_k, "with_payload": True}
        qf = build_filter(filters)
        if qf:
            body["filter"] = qf

        resp = await self.client.post(f"/collections/{collection}/points/query", json=body)
        if resp.status_code == 404:
            # Nothing indexed yet — an empty KB is not an error.
            logger.info("collection %s does not exist, returning no results", collection)
            return []
        if resp.status_code >= 400:
            logger.error("%s search on %s -> %s: %s",
                         using, collection, resp.status_code, resp.text[:300])
            return []

        points = (resp.json().get("result") or {}).get("points", [])
        return [
            {"id": p["id"], "score": p.get("score", 0.0),
             "payload": p.get("payload") or {}, "collection": collection}
            for p in points
        ]

    async def scroll(self, collection: str, limit: int = 100, offset: str = None,
                     filters: dict = None) -> tuple[list[dict], str]:
        """Paginated listing. Returns (points, next_offset)."""
        body = {"limit": limit, "with_payload": True}
        if offset is not None:
            body["offset"] = offset
        qf = build_filter(filters)
        if qf:
            body["filter"] = qf

        resp = await self.client.post(f"/collections/{collection}/points/scroll", json=body)
        if resp.status_code == 404:
            return [], None
        if resp.status_code >= 400:
            raise QdrantError(f"scroll {collection} -> {resp.status_code}: {resp.text[:300]}")

        result = resp.json().get("result") or {}
        points = [
            {"id": p["id"], "payload": p.get("payload") or {}}
            for p in result.get("points", [])
        ]
        return points, result.get("next_page_offset")


    async def expand_by_source(self, collection: str, source_values: list[str],
                               exclude_ids: list = None) -> list[dict]:
        """Every chunk belonging to `source_values`, ordered by (source, chunk_index).

        Both payload shapes are queried and merged by point id, rather than
        stopping at the first key that returns anything: the backfill is rolling,
        so one collection can hold a namespaced file and a not-yet-converted one
        at the same time, and a first-key-wins fallback would silently never
        expand the second. Same two-key pass ingestion.delete_by_source makes.

        exclude_ids drops points the caller already has, so a primary chunk is
        never returned twice.
        """
        if not source_values:
            return []
        skip = {str(i) for i in (exclude_ids or ())}
        found = {}
        for key in ("source.filename", "source"):
            offset = None
            while len(found) < EXPAND_LIMIT:
                page, offset = await self.scroll(
                    collection, limit=SCROLL_PAGE, offset=offset,
                    filters={key: list(source_values)},
                )
                found.update((str(p["id"]), {**p, "collection": collection})
                             for p in page if str(p["id"]) not in skip)
                if offset is None:
                    break
        return sorted(found.values(),
                      key=lambda p: (source_of(p["payload"]), chunk_index(p["payload"])))


_service = None


def get_qdrant() -> QdrantService:
    """Shared instance — one httpx connection pool for the process."""
    global _service
    if _service is None:
        _service = QdrantService()
    return _service
