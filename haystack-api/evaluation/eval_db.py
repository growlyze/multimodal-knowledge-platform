"""
Schema and connection for the evaluation tables.

Six `eval_*` tables in the same PostgreSQL database as everything else. They
auto-create on first use, and nothing in this module raises: evaluation runs
inside the chat request path, so a dead database has to cost a log line, not an
answer.

    db = get_eval_db()
    ok   = await db.execute("INSERT INTO eval_query_log ...", *args)   # False on failure
    rows = await db.fetch("SELECT ...")                                # [] on failure

DBService (services/db_service.py) refuses to touch anything outside `data_*`,
so this keeps its own small pool rather than widening that guard.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from uuid import uuid4

import config

logger = logging.getLogger("haystack-api.evaluation")

RETRY_COOLDOWN = 60.0       # seconds before dialling a database that just failed
POOL_MAX = 3

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS eval_query_log (
        id TEXT PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT NOW(),
        user_id TEXT,
        query TEXT NOT NULL,
        results_count INT DEFAULT 0,
        top_score FLOAT,
        collections_searched TEXT,
        latency_ms INT,
        had_code_execution BOOLEAN DEFAULT FALSE,
        had_file_attachment BOOLEAN DEFAULT FALSE,
        empty_result BOOLEAN DEFAULT FALSE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_answer_scores (
        id TEXT PRIMARY KEY,
        query_log_id TEXT,
        timestamp TIMESTAMP DEFAULT NOW(),
        faithfulness FLOAT,
        answer_relevancy FLOAT,
        hallucination_count INT,
        total_claims INT,
        citation_accuracy FLOAT,
        judge_model TEXT,
        judge_reasoning TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_ingestion_quality (
        id TEXT PRIMARY KEY,
        job_id TEXT,
        timestamp TIMESTAMP DEFAULT NOW(),
        filename TEXT,
        total_chunks INT,
        avg_chunk_size FLOAT,
        oversized_count INT DEFAULT 0,
        fragment_count INT DEFAULT 0,
        mid_sentence_splits INT DEFAULT 0,
        coherence_avg FLOAT,
        errors TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_test_runs (
        id TEXT PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT NOW(),
        precision_at_5 FLOAT,
        recall_at_10 FLOAT,
        mrr FLOAT,
        ndcg FLOAT,
        jargon_hit_rate FLOAT,
        total_queries INT,
        failed_queries INT,
        details TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_system_metrics (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT NOW(),
        period TEXT DEFAULT 'hourly',
        search_latency_p50 INT,
        search_latency_p95 INT,
        embedding_latency_avg INT,
        error_count INT DEFAULT 0,
        query_count INT DEFAULT 0,
        empty_result_count INT DEFAULT 0,
        voyage_429_count INT DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_coverage_gaps (
        id SERIAL PRIMARY KEY,
        query_pattern TEXT NOT NULL,
        occurrence_count INT DEFAULT 1,
        first_seen TIMESTAMP DEFAULT NOW(),
        last_seen TIMESTAMP DEFAULT NOW(),
        resolved BOOLEAN DEFAULT FALSE
    )
    """,
    "CREATE INDEX IF NOT EXISTS eval_query_log_ts ON eval_query_log (timestamp)",
    "CREATE INDEX IF NOT EXISTS eval_answer_scores_ts ON eval_answer_scores (timestamp)",
    "CREATE INDEX IF NOT EXISTS eval_test_runs_ts ON eval_test_runs (timestamp)",
    # ON CONFLICT (query_pattern) needs this; the table itself is verbatim from the spec.
    "CREATE UNIQUE INDEX IF NOT EXISTS eval_coverage_gaps_pattern "
    "ON eval_coverage_gaps (query_pattern)",
    # ── M3: Video ingestion tables ───────────────────────
    # video_catalog before video_ingest_jobs (FK dependency)
    """
    CREATE TABLE IF NOT EXISTS video_catalog (
        video_id       TEXT PRIMARY KEY,
        title          TEXT NOT NULL,
        duration       INTEGER NOT NULL,
        upload_date    DATE,
        chapter_count  INTEGER DEFAULT 0,
        thumbnail_url  TEXT,
        channel_id     TEXT NOT NULL,
        discovered_at  TIMESTAMPTZ DEFAULT NOW(),
        source_url     TEXT,
        platform       TEXT NOT NULL DEFAULT 'youtube'
    )
    """,
    # Non-YouTube rows (pasted via /video/fetch-url) need source_url to
    # re-derive a download URL later — video_id alone can't do it off
    # YouTube. ADD COLUMN IF NOT EXISTS so this applies to tables created
    # before these columns existed, same as a fresh CREATE TABLE above.
    "ALTER TABLE video_catalog ADD COLUMN IF NOT EXISTS source_url TEXT",
    "ALTER TABLE video_catalog ADD COLUMN IF NOT EXISTS platform TEXT NOT NULL DEFAULT 'youtube'",
    """
    CREATE TABLE IF NOT EXISTS video_ingest_jobs (
        id             SERIAL PRIMARY KEY,
        video_id       TEXT NOT NULL REFERENCES video_catalog(video_id),
        status         TEXT NOT NULL DEFAULT 'new',
        queue_position INTEGER,
        phase          TEXT,
        phase_progress REAL DEFAULT 0.0,
        retry_count    INTEGER DEFAULT 0,
        error_message  TEXT,
        started_at     TIMESTAMPTZ,
        completed_at   TIMESTAMPTZ,
        created_at     TIMESTAMPTZ DEFAULT NOW(),
        updated_at     TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    # Ingestion warnings ("only 64% of frames could be read", "transcript stops
    # at 49%", "charts missed"): shown in the admin panel, so a problem isn't
    # only a span label nobody reads (P137, 2026-10-20).
    "ALTER TABLE video_ingest_jobs ADD COLUMN IF NOT EXISTS warnings TEXT",
    """
    CREATE TABLE IF NOT EXISTS video_phase_checkpoints (
        video_id       TEXT NOT NULL,
        phase          TEXT NOT NULL,
        status         TEXT NOT NULL,
        output_path    TEXT,
        item_count     INTEGER,
        started_at     TIMESTAMPTZ,
        completed_at   TIMESTAMPTZ,
        PRIMARY KEY (video_id, phase)
    )
    """,
    # D68: safety/guardrail layer — running cost ledger checked against
    # config.MAX_INGESTION_SPEND_USD before each new video starts.
    """
    CREATE TABLE IF NOT EXISTS video_ingest_costs (
        video_id       TEXT PRIMARY KEY,
        groq_cost_usd  REAL NOT NULL DEFAULT 0.0,
        voyage_cost_usd REAL NOT NULL DEFAULT 0.0,
        vision_cost_usd REAL NOT NULL DEFAULT 0.0,
        recorded_at    TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    # Measured per-run usage (pipelines/usage_meter.py): exact Gemini tokens
    # incl. thinking, Voyage tokens + pixels, Groq billed seconds, plus exact
    # wall-clock per stage. video_ingest_costs only ever held flat estimates,
    # which could not answer "actual cost per stage, including tokens".
    """
    CREATE TABLE IF NOT EXISTS video_ingest_usage (
        video_id       TEXT PRIMARY KEY,
        usage          JSONB NOT NULL,
        phase_seconds  JSONB NOT NULL DEFAULT '{}',
        prices_as_of   TEXT NOT NULL,
        total_usd      REAL NOT NULL,
        status         TEXT NOT NULL,
        recorded_at    TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    # Per-question retrieval cost (P161 item 3, pipelines/retrieval_cost.py):
    # one row per /search -- Voyage query embed, video embed_query, rerank,
    # Gemini planner -- as a usage_meter.SearchUsageMeter summary. Kept apart
    # from video_ingest_usage (ingestion) and chat-model spend on purpose.
    # Raw rows are pruned after 90 days by the writer itself.
    """
    CREATE TABLE IF NOT EXISTS retrieval_usage (
        id             TEXT PRIMARY KEY,
        recorded_at    TIMESTAMPTZ DEFAULT NOW(),
        query_log_id   TEXT,
        user_id        TEXT,
        usage          JSONB NOT NULL,
        total_usd      REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS retrieval_usage_ts ON retrieval_usage (recorded_at)",
]

_FAILED = object()


def new_id() -> str:
    return uuid4().hex


def cutoff(hours: float) -> datetime:
    """Naive local timestamp `hours` ago — the same clock NOW() writes with."""
    return datetime.now() - timedelta(hours=hours)


class EvalDB:
    """asyncpg-backed. `pool` is injectable so tests can run without PostgreSQL."""

    def __init__(self, dsn: str = None, pool=None):
        self.dsn = dsn or config.DATABASE_URL
        self._pool = pool
        self._ready = False
        self._lock = asyncio.Lock()
        self._retry_at = 0.0

    async def pool(self):
        """The pool with every `eval_*` table guaranteed, or None if unreachable."""
        if self._ready:
            return self._pool
        if time.monotonic() < self._retry_at:
            return None
        async with self._lock:
            if self._ready:
                return self._pool
            try:
                if self._pool is None:
                    import asyncpg
                    self._pool = await asyncpg.create_pool(
                        self.dsn, min_size=1, max_size=POOL_MAX)
                async with self._pool.acquire() as conn:
                    for statement in SCHEMA:
                        await conn.execute(statement)
            except Exception as exc:
                # ponytail: flat cooldown, not backoff — a dead database costs one
                # log line a minute and every caller already has a default to
                # return. Swap in backoff only if that minute ever matters.
                self._retry_at = time.monotonic() + RETRY_COOLDOWN
                logger.warning("eval tables unavailable (%s: %s) — retrying in %.0fs",
                               type(exc).__name__, exc, RETRY_COOLDOWN)
                return None
            self._ready = True
            logger.info("eval_* tables ready")
            return self._pool

    async def _run(self, method: str, sql: str, args, default):
        pool = await self.pool()
        if pool is None:
            return default
        try:
            async with pool.acquire() as conn:
                return await getattr(conn, method)(sql, *args)
        except Exception as exc:
            logger.warning("eval %s failed (%s: %s) | %s",
                           method, type(exc).__name__, exc, " ".join(sql.split())[:120])
            return default

    async def execute(self, sql: str, *args) -> bool:
        """True when the statement ran."""
        return await self._run("execute", sql, args, _FAILED) is not _FAILED

    async def fetch(self, sql: str, *args) -> list:
        return [dict(row) for row in await self._run("fetch", sql, args, [])]

    async def fetchrow(self, sql: str, *args):
        row = await self._run("fetchrow", sql, args, None)
        return dict(row) if row is not None else None

    async def fetchval(self, sql: str, *args):
        return await self._run("fetchval", sql, args, None)

    async def aclose(self):
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception as exc:
                logger.warning("closing eval pool: %s", exc)
        self._pool, self._ready = None, False


_db = None


def get_eval_db() -> EvalDB:
    """Shared instance — one pool for the process."""
    global _db
    if _db is None:
        _db = EvalDB()
    return _db


async def log_ingestion_quality(job_id: str, filename: str, report: dict,
                                coherence_avg: float = None, db: EvalDB = None) -> str:
    """Store a Layer 1 chunk_inspector report against an ingestion job.

    INGESTION calls this with the dict inspect_chunks() already returns, so
    nothing in that pipeline has to change shape.
    """
    counts = report.get("counts") or {}
    log_id = new_id()
    await (db or get_eval_db()).execute(
        """
        INSERT INTO eval_ingestion_quality
            (id, job_id, filename, total_chunks, avg_chunk_size, oversized_count,
             fragment_count, mid_sentence_splits, coherence_avg, errors)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
        log_id, job_id, filename,
        int(report.get("chunks") or 0),
        float((report.get("tokens") or {}).get("mean") or 0.0),
        int(counts.get("too_large", 0)),
        int(counts.get("too_small", 0) + counts.get("empty", 0)),
        int(counts.get("starts_mid_sentence", 0) + counts.get("ends_mid_sentence", 0)),
        coherence_avg,
        json.dumps(counts) if report.get("errors") else None,
    )
    return log_id
