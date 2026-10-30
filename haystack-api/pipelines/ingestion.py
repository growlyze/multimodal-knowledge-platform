"""
Ingestion pipeline — extract, chunk, inspect, embed, upsert.

    /ingest/* endpoint  ->  create_job()  -> job id returned immediately
                            BackgroundTasks -> run_job() does the work

One file becomes one job. Re-ingesting the same filename into the same
collection replaces its chunks instead of duplicating them, because the job
deletes every point with that `source` before upserting the new ones.

Content that is already indexed is not ingested again (pipelines/dedup.py):
identical bytes are caught before extraction, identical text before chunking,
and the job is marked done with `skipped` = "unchanged" (same filename) or
"duplicate" (+ `duplicate_of`, another filename) and nothing is embedded.

Layer 1 quality inspection runs on every job and its report is stored with the
job. Layer 2 (LLM coherence judge) runs only when ENABLE_COHERENCE_JUDGE=true.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone

try:
    import yaml
except ImportError:              # pragma: no cover - depends on the image build
    # ponytail: PyYAML is in requirements.txt but INFRA owns the image, so a
    # container built before that line still has to ingest. The fallback parser
    # below reads the scraper's own `key: value` front matter, which is all we
    # produce today. Delete it once every deployed image has PyYAML.
    yaml = None

import config
from evaluation.chunk_inspector import inspect_chunks
from pipelines import dedup
from pipelines.chunkers.code_chunker import CodeChunker
from pipelines.chunkers.semantic_chunker import SemanticChunker
from pipelines.chunkers.text_chunker import TextChunker

logger = logging.getLogger("haystack-api.ingest")

# Layer 2 is an LLM call per sampled chunk — off by default so bulk ingestion
# of 622 hours of transcripts does not run up an API bill.
ENABLE_COHERENCE_JUDGE = os.getenv("ENABLE_COHERENCE_JUDGE", "false").lower() == "true"
COHERENCE_SAMPLE_SIZE = int(os.getenv("COHERENCE_SAMPLE_SIZE", "20"))

# Kill switch for pipelines/dedup.py — off restores replace-by-filename only.
# Read here like the two knobs above rather than in config.py (infra-owned).
INGEST_DEDUP = os.getenv("INGEST_DEDUP", "true").lower() == "true"
# Metadata flag that re-ingests unchanged content on purpose — the only way to
# re-index a file after a chunker or embedding change, since its hash did not move.
FORCE_KEY = "force_reingest"

CODE_SOURCE_TYPES = {"code"}


def select_chunker(source_type: str):
    """Chunking strategy for a source type. Unknown types get the text chunker."""
    if source_type == "transcript":
        return SemanticChunker()
    if source_type in CODE_SOURCE_TYPES:
        return CodeChunker()
    return TextChunker()


def default_collection(source_type: str) -> str:
    return (config.COLLECTION_CODE if source_type in CODE_SOURCE_TYPES
            else config.COLLECTION_DOCUMENTS)


def chunk_id(collection: str, filename: str, index: int) -> str:
    """Deterministic, so a re-ingest overwrites the same points."""
    return "%s:%s:%d" % (collection, filename, index)


# ── payload schema ──────────────────────────────────────
# Two zones. The filter zone stays flat because Qdrant indexes and filters on
# it; everything a citation needs is nested under "source", where only
# services/citation_formatter.py reads it.

FILTER_KEYS = ("source_type", "collection", "chunk_index", "ingested_at", "origin")
SOURCE_KEYS = ("url", "title", "filename", "kind", "name", "version",
               "page_number", "row_range", "original_filename")

DEFAULT_ORIGIN = "api"

# Open WebUI names uploads "<uuid>_<the name the user picked>".
_UUID_PREFIX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_", re.I)


def strip_uuid_prefix(filename: str) -> str:
    """'24721dfc-...-2d94ba676c5b_33.esh23.csv' -> '33.esh23.csv'.

    Anything without that prefix — including a scraped page's relative path —
    comes back unchanged, which is how build_payload knows there was no rename.
    """
    return _UUID_PREFIX.sub("", str(filename or ""))


def parse_front_matter(text: str) -> tuple:
    """'---\\nkey: value\\n---\\nbody' -> ({"key": "value"}, "body").

    Missing or malformed front matter returns ({}, text) — a page that does not
    have any is the normal case, not an error.
    """
    if not text or not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block, body = text[3:end], text[end + 4:].lstrip("\n")

    meta = None
    try:
        meta = yaml.safe_load(block) if yaml else None
    except Exception as exc:
        logger.debug("front matter is not valid YAML (%s: %s)", type(exc).__name__, exc)
    if not isinstance(meta, dict):
        # Some reference pages are titled after operators and annotations —
        # "title: @version=", "name: =", "title: *= — ..." — which YAML refuses.
        # The scraper writes one `key: value` per line, so read it that way
        # rather than lose the citation metadata for those pages.
        meta = _simple_front_matter(block)
    # A document that opens with a markdown rule parses as YAML too ("Note: hi"
    # is a mapping), so the keys have to look like keys before we believe it.
    if not meta or not all(_is_key(k) for k in meta):
        return {}, text
    return {str(k): v for k, v in meta.items() if v not in (None, "")}, body


def _is_key(key) -> bool:
    return isinstance(key, str) and bool(key.strip()) and " " not in key.strip()


def _simple_front_matter(block: str) -> dict:
    """One `key: value` per line, JSON-quoted where the value is ambiguous."""
    meta = {}
    for line in block.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not sep or not _is_key(key):
            return {}
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        meta[key] = value
    return meta


def front_matter_metadata(front: dict) -> dict:
    """Scraper front matter -> ingestion metadata.

    The YAML key `source` is the page URL (confusing, but it is what the scraper
    writes), and `type` is the source type.
    """
    meta = {k: v for k, v in front.items() if k not in ("source", "type")}
    if front.get("source"):
        meta["url"] = front["source"]
    if front.get("type"):
        meta["source_type"] = front["type"]
    return meta


def build_payload(content: str, chunk_meta: dict, collection: str,
                  ingested_at: str) -> dict:
    """Chunk metadata -> the namespaced Qdrant payload.

    The chunkers keep working in flat metadata (chunk_inspector and the code
    chunker both read `source` from it); the split into zones happens here, once,
    on the way into Qdrant.
    """
    meta = dict(chunk_meta or {})
    filename = meta.pop("source", None)
    source = {key: meta.pop(key, None) for key in SOURCE_KEYS}
    source["filename"] = filename
    if not source["original_filename"]:
        stripped = strip_uuid_prefix(filename)
        source["original_filename"] = stripped if stripped != filename else None

    return {"content": content,
            "source_type": meta.pop("source_type", None),
            "collection": collection,
            "ingested_at": meta.pop("ingested_at", None) or ingested_at,
            "origin": meta.pop("origin", None) or DEFAULT_ORIGIN,
            **meta,                 # chunk_index, token_count, language, timestamps...
            "source": source}


async def delete_by_source(qdrant, collection: str, filename: str,
                           extra: dict = None) -> int:
    """Delete every chunk of one file, in both payload shapes.

    Points written before the namespacing carry `source` as the filename string;
    new ones carry it under `source.filename`. Re-ingest has to clear both or a
    file ingested twice across the change would keep its old chunks forever.
    """
    removed = 0
    for key in ("source.filename", "source"):
        removed += await qdrant.delete_by_filter(collection, {key: filename, **(extra or {})})
    return removed


# ── job status tracking ─────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source_type TEXT NOT NULL,
    collection TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    chunks_created INTEGER DEFAULT 0,
    quality JSONB,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
)
"""

_FIELDS = ("id", "filename", "source_type", "collection", "status",
           "chunks_created", "quality", "error", "created_at", "completed_at")

# A job list nobody reads past, and a cap on what one request can ask for.
DEFAULT_JOB_LIST = 50
MAX_JOB_LIST = 200


def _decode_job_row(row) -> dict:
    """One job row -> the JSON shape the panel reads, from either backend.

    Postgres hands back datetimes and a JSON string; the in-memory fallback
    already holds ISO strings and a dict. Both go through here so a degraded
    store and a healthy one are indistinguishable to the caller.

    `job_id` rides along beside `id` because the panel polls
    /ingest/status/{job_id} for anything still running, and it reads that key
    off whatever the list handed it.
    """
    out = dict(row)
    for key in ("created_at", "completed_at"):
        value = out.get(key)
        if value is not None and not isinstance(value, str):
            out[key] = value.isoformat()
    if isinstance(out.get("quality"), str):
        try:
            out["quality"] = json.loads(out["quality"])
        except ValueError:
            # A quality report that will not parse is not worth failing a status
            # poll over — the status and the error are what the caller came for.
            out["quality"] = None
    if out.get("id") and not out.get("job_id"):
        out["job_id"] = out["id"]
    return out


class JobStore:
    """PostgreSQL-backed job status, with an in-process fallback.

    ponytail: asyncpg is not in requirements.txt yet (INFRA owns that file), so
    the store degrades to a dict and the API still works end to end. The moment
    the package is installed and DATABASE_URL resolves, it uses Postgres with
    no code change. The fallback is per-process — status is lost on restart and
    wrong behind more than one worker.
    """

    def __init__(self, dsn: str = None):
        self.dsn = dsn or config.DATABASE_URL
        self._pool = None
        self._memory = {}
        self._degraded = None      # None = not decided yet
        self._lock = asyncio.Lock()

    async def _pg(self):
        """Connection pool, or None when Postgres is unavailable."""
        if self._degraded:
            return None
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is not None or self._degraded:
                return self._pool
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4)
                async with self._pool.acquire() as conn:
                    await conn.execute(SCHEMA)
                self._degraded = False
                logger.info("ingestion_jobs table ready")
            except Exception as exc:
                self._degraded = True
                logger.warning("job status falling back to memory (%s: %s)",
                               type(exc).__name__, exc)
            return self._pool

    async def create(self, job_id: str, filename: str, source_type: str,
                     collection: str) -> dict:
        row = {"id": job_id, "filename": filename, "source_type": source_type,
               "collection": collection, "status": "queued", "chunks_created": 0,
               "quality": None, "error": None,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "completed_at": None}
        self._memory[job_id] = row
        pool = await self._pg()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO ingestion_jobs (id, filename, source_type, collection, status)"
                    " VALUES ($1, $2, $3, $4, 'queued')"
                    " ON CONFLICT (id) DO UPDATE SET status = 'queued', error = NULL",
                    job_id, filename, source_type, collection)
        return row

    async def update(self, job_id: str, **fields) -> None:
        row = self._memory.setdefault(job_id, {"id": job_id})
        row.update(fields)
        if fields.get("status") in ("done", "failed"):
            row["completed_at"] = datetime.now(timezone.utc).isoformat()

        pool = await self._pg()
        if not pool:
            return
        allowed = {k: v for k, v in fields.items()
                   if k in _FIELDS and k not in ("id", "created_at")}
        if not allowed:
            return
        if allowed.get("quality") is not None:
            allowed["quality"] = json.dumps(allowed["quality"])
        sets = ", ".join("%s = $%d" % (k, i + 2) for i, k in enumerate(allowed))
        if fields.get("status") in ("done", "failed"):
            sets += ", completed_at = NOW()"
        async with pool.acquire() as conn:
            await conn.execute("UPDATE ingestion_jobs SET %s WHERE id = $1" % sets,
                               job_id, *allowed.values())

    async def mark_deleted(self, filename: str, collection: str) -> int:
        """Flip every finished job for this file+collection to `deleted`.

        P123: a file removed through DELETE /admin/documents kept its job rows at
        status=done, chunks_created=N, so GET /ingest/status/<id> went on vouching
        for content that was no longer in Qdrant — and that "done" is what sent a
        two-day hunt looking for a data-loss bug in ingestion. Only `done` rows
        move: a failed job never indexed anything, and a queued/processing one
        will write fresh chunks after this delete and report on those itself.
        """
        note = "removed via DELETE /admin/documents at %s" % (
            datetime.now(timezone.utc).isoformat())
        marked = 0
        for row in self._memory.values():
            if (row.get("filename") == filename and row.get("collection") == collection
                    and row.get("status") == "done"):
                row.update(status="deleted", error=note)
                marked += 1

        pool = await self._pg()
        if not pool:
            return marked
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE ingestion_jobs SET status = 'deleted', error = $3"
                " WHERE filename = $1 AND collection = $2 AND status = 'done'",
                filename, collection, note)
        # asyncpg answers "UPDATE <n>"; the database is the authority when it is up.
        try:
            return int(str(result).rsplit(" ", 1)[-1])
        except ValueError:
            return marked

    async def get(self, job_id: str):
        pool = await self._pg()
        if pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM ingestion_jobs WHERE id = $1", job_id)
            if row:
                return _decode_job_row(row)
        memory = self._memory.get(job_id)
        return _decode_job_row(memory) if memory else None

    async def list(self, limit: int = DEFAULT_JOB_LIST, status: str = None) -> list:
        """Recent jobs, newest first. Optionally only those in one state.

        This is the only place a chat-triggered ingestion is ever visible. The
        upload happens inside Open WebUI, the filter posts it and forgets it, and
        nobody in the admin panel is holding the job id — so without a list, a
        CSV that failed to ingest and a CSV nobody ever uploaded look exactly
        the same from the panel. That is the whole reason the missing structured
        ingestion took a live experiment to diagnose.
        """
        limit = max(1, min(int(limit), MAX_JOB_LIST))
        pool = await self._pg()
        if pool:
            sql, args = "SELECT * FROM ingestion_jobs", []
            if status:
                args.append(status)
                sql += " WHERE status = $%d" % len(args)
            args.append(limit)
            # NULLS LAST so a row whose created_at default has not resolved sorts
            # with the oldest rather than displacing the newest.
            sql += " ORDER BY created_at DESC NULLS LAST LIMIT $%d" % len(args)
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *args)
            return [_decode_job_row(row) for row in rows]

        # Ties are not hypothetical here: datetime.now() resolves to ~15ms on
        # Windows, so two uploads in the same tick carry the same created_at, and
        # a stable sort then hands them back in dict order — oldest first, the
        # exact opposite of what this promises. Insertion order breaks the tie,
        # because a later insert is a later job.
        numbered = list(enumerate(self._memory.values()))
        if status:
            numbered = [(i, row) for i, row in numbered if row.get("status") == status]
        numbered.sort(key=lambda pair: (pair[1].get("created_at") or "", pair[0]),
                      reverse=True)
        return [_decode_job_row(row) for _, row in numbered[:limit]]


_store = None


def get_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store


# ── the pipeline ────────────────────────────────────────

async def create_job(filename: str, source_type: str, collection: str = None) -> dict:
    """Register a queued job and hand back its id. Never touches Qdrant."""
    collection = collection or default_collection(source_type)
    job_id = str(uuid.uuid4())
    await get_store().create(job_id, filename, source_type, collection)
    logger.info("job %s queued: %s (%s -> %s)", job_id, filename, source_type, collection)
    return {"job_id": job_id, "filename": filename, "source_type": source_type,
            "collection": collection, "status": "queued"}


async def run_job(job_id: str, filename: str, file_bytes: bytes, source_type: str,
                  collection: str, metadata: dict = None) -> dict:
    """The background half. Marks the job done or failed; never raises."""
    store = get_store()
    started = time.perf_counter()
    try:
        await store.update(job_id, status="processing")
        metadata = dict(metadata or {})
        skipped = None
        if INGEST_DEDUP and not metadata.get(FORCE_KEY):
            # Before extraction: the filter re-posts every attachment on every
            # chat turn, and a byte-identical repost must cost no Tika call and
            # no embedding. Hit -> the text check below is never reached.
            metadata["file_sha256"] = dedup.bytes_hash(file_bytes)
            skipped = await _skip_if_indexed(collection, filename, "file_sha256",
                                             metadata["file_sha256"])
        if skipped:
            result = skipped
        else:
            text = await extract(filename, file_bytes, source_type)
            result = await ingest_text(text, filename, source_type, collection, metadata)
        await store.update(job_id, status="done",
                           chunks_created=result["chunks_created"],
                           quality=result["quality"])
        logger.info("job %s done: %d chunks in %.1fs",
                    job_id, result["chunks_created"], time.perf_counter() - started)
        return result
    except Exception as exc:
        logger.exception("job %s failed: %s", job_id, filename)
        await store.update(job_id, status="failed", error="%s: %s" % (type(exc).__name__, exc))
        return {"job_id": job_id, "status": "failed", "error": str(exc)}


class ExtractionError(RuntimeError):
    """Extraction produced a failure placeholder rather than a document.

    run_job turns this into a failed job with the reason attached, which is what
    GET /ingest/status/{id} then shows.
    """


async def extract(filename: str, file_bytes: bytes, source_type: str) -> str:
    """Bytes -> text. Reuses the extractors the EXTRACTION session already built
    (imported, never edited) so PDF/DOCX still go through Tika."""
    if source_type in ("transcript", "chat"):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")

    from extractors.document import extraction_failed
    from extractors.router import route_extraction

    text = await route_extraction(filename, "", file_bytes)
    # The extractors answer a failure with prose, because /extract/process must
    # return a body or Open WebUI fails the upload. Indexing that prose puts a
    # one-chunk "document" in the knowledge base whose content is an error
    # message, retrievable and citable ever after — so the job fails instead.
    if extraction_failed(text):
        raise ExtractionError(text)
    return text


async def ingest_text(text: str, filename: str, source_type: str,
                      collection: str = None, metadata: dict = None) -> dict:
    """Chunk -> inspect -> embed -> upsert. The part worth testing in isolation."""
    # Scraped markdown carries its own metadata; the caller's wins on conflict.
    front, text = parse_front_matter(text)
    metadata = {**front_matter_metadata(front), **(metadata or {})}
    source_type = metadata.pop("source_type", None) or source_type
    collection = collection or default_collection(source_type)
    force = bool(metadata.pop(FORCE_KEY, False))
    check = INGEST_DEDUP and not force and bool(dedup.normalize(text))
    if check:
        metadata["doc_sha256"] = dedup.text_hash(text)
        metadata["meta_sha256"] = dedup.meta_hash(front)
        metadata["hash_version"] = dedup.HASH_VERSION
        # Held until the upsert lands: two uploads of one text in the same
        # second must not both pass the check below and both embed.
        async with dedup.hold(metadata["doc_sha256"]):
            skipped = await _skip_if_indexed(collection, filename, "doc_sha256",
                                             metadata["doc_sha256"],
                                             meta=metadata["meta_sha256"])
            if skipped:
                return skipped
            return await _index(text, filename, source_type, collection, metadata)
    return await _index(text, filename, source_type, collection, metadata)


async def _skip_if_indexed(collection: str, filename: str, field: str,
                           value: str, meta: str = None) -> dict:
    """The skip result when `value` is already in `collection`, else None.

    A duplicate also clears this filename's own older chunks: "b.txt"
    re-uploaded with a.txt's exact text no longer holds what b.txt's old
    chunks say, and keeping them would answer from a file that is gone.
    """
    from services.qdrant_service import get_qdrant

    qdrant = get_qdrant()
    if await qdrant.collection_info(collection) is None:
        return None
    await dedup.ensure_hash_indexes(qdrant, collection)
    found = await dedup.find_existing(qdrant, collection, field, value, filename)
    if not found:
        return None
    if found["kind"] == "unchanged" and meta and found["meta"] != {meta}:
        # Same body, new citation metadata (a re-scrape whose title moved):
        # fall through to the normal replace so the payload is rewritten.
        return None
    removed = 0
    if found["kind"] == "duplicate":
        removed = await delete_by_source(qdrant, collection, filename)
    report = dedup.skip_report(found, field)
    logger.info("skipped %s: %s%s (%s)", filename, found["kind"],
                " of %s" % found["source"] if found["kind"] == "duplicate" else "",
                field)
    return {"filename": filename, "collection": collection, "chunks_created": 0,
            "replaced": removed, "quality": report, **report}


async def _index(text: str, filename: str, source_type: str, collection: str,
                 metadata: dict) -> dict:
    base = {"source": filename, "source_type": source_type, **metadata}

    chunker = select_chunker(source_type)
    chunks = await chunker.achunk(text, base)
    chunks, repeated = dedup.drop_repeated_chunks(chunks)
    if repeated:
        logger.info("%s: dropped %d repeated chunk(s)", filename, repeated)
    if not chunks:
        logger.warning("%s produced no chunks — nothing to index", filename)
        return {"filename": filename, "collection": collection, "chunks_created": 0,
                "quality": inspect_chunks([], source_type, text, filename)}

    quality = inspect_chunks(chunks, source_type, text, filename)

    from services.embedding_service import create_embedding_provider
    from services.qdrant_service import get_qdrant

    embedder = create_embedding_provider()
    qdrant = get_qdrant()

    vectors = await embedder.embed([c.content for c in chunks], input_type="document")
    await qdrant.ensure_collection(collection, embedder.dimensions)

    # Replace, don't duplicate: the same file ingested twice must not double up.
    removed = await delete_by_source(qdrant, collection, filename)
    if removed:
        logger.info("replaced %d existing chunks for %s", removed, filename)

    ingested_at = datetime.now(timezone.utc).isoformat()
    ids, payloads = [], []
    for i, chunk in enumerate(chunks):
        ids.append(chunk_id(collection, filename, i))
        payloads.append(build_payload(chunk.content, chunk.metadata, collection, ingested_at))
    await qdrant.upsert(collection, ids, vectors, payloads)

    result = {"filename": filename, "collection": collection,
              "chunks_created": len(chunks), "replaced": removed, "quality": quality}

    if ENABLE_COHERENCE_JUDGE:
        result["coherence"] = await _judge(chunks)
    return result


async def _judge(chunks: list):
    """Layer 2 is a quality signal, not a gate — a judge failure never fails a job."""
    try:
        from evaluation.coherence_judge import judge_chunks
        report = await judge_chunks(chunks, sample_size=COHERENCE_SAMPLE_SIZE)
        if not report.get("passed", True):
            logger.warning("coherence below target: %s", report.get("summary"))
        return report
    except Exception as exc:
        logger.warning("coherence judge skipped: %s", exc)
        return {"error": str(exc)}
