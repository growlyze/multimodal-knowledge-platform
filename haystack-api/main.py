"""
Haystack RAG API — the single backend that handles:
  1. Content extraction (Open WebUI calls /extract)
  2. Ingestion to Qdrant (filter calls /ingest)
  3. Retrieval from Qdrant (filter calls /search)
  4. Admin operations (KB manager calls /admin)
"""

import asyncio
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from telemetry_compat import flush, init_tracing
from services.auth import AdminKeyNotConfigured, admin_key_not_configured_handler
# `eval` aliased — the router module name shadows the builtin otherwise.
from routers import admin, costs_retrieval, extract, ingest, search, video
from routers import eval as eval_router
from routers import dashboard as dashboard_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("haystack-api")


# P181 follow-up (approved 2026-10-25): the first video question after a restart
# ran unplanned -- building the planner's Gemini/Claude clients took 3.1-4.2 s, over
# retrieval.PLANNER_BUDGET_S. Warmed in the background: startup never waits for it,
# and a slow or failing provider only costs a log line (the first question then
# degrades exactly as it did before this existed).
PLANNER_WARM_TIMEOUT_S = 30.0


async def _warm_planner():
    try:
        from pipelines import query_planner
        result = await asyncio.wait_for(query_planner.warm_up(), PLANNER_WARM_TIMEOUT_S)
        logger.info("query planner warmed: %s", result)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("query planner warm-up failed (%s: %s) -- the first video question "
                       "may run unplanned", type(exc).__name__, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("Starting Haystack RAG API")
    logger.info(f"Embedding: {config.EMBEDDING_PROVIDER}/{config.EMBEDDING_MODEL}")
    logger.info(f"Qdrant: {config.QDRANT_URL}")
    logger.info(f"Hybrid search: {config.ENABLE_HYBRID_SEARCH}")

    # Reads PHOENIX_API_KEY from this service's own existing env_file (.env) — no
    # new env var. Provider is flushed at shutdown below: BatchSpanProcessor
    # holds spans for up to 5s, and without an explicit flush, whatever's queued
    # when the container stops is silently lost on every restart/deploy.
    tracer_provider = init_tracing("haystack-api")

    # Create upload directory
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    # Open the PostgreSQL pool now, which is also what creates `data_catalog`.
    # Doing it here rather than leaving it to the first CSV upload means a
    # database misconfiguration shows up in the startup log, not as one failed
    # ingestion job hours later. Non-fatal on purpose: everything except
    # structured data works without PostgreSQL, and the pool retries on use.
    try:
        from services.db_service import get_db
        await get_db().pool()
        logger.info("PostgreSQL ready (data_catalog present)")
    except Exception as exc:
        logger.warning("PostgreSQL unavailable at startup (%s: %s) — structured "
                       "ingestion will retry the connection per upload",
                       type(exc).__name__, exc)

    # TODO: Initialize Qdrant collections if they don't exist
    # TODO: Initialize embedding client
    # TODO: Initialize reranker client

    warm_task = asyncio.create_task(_warm_planner())
    # Drive auto-approve / auto-delete (both OFF until switched on in the panel).
    from routers.video import drive_autopilot_loop
    autopilot_task = asyncio.create_task(drive_autopilot_loop())

    yield

    logger.info("Shutting down Haystack RAG API")
    if not warm_task.done():
        warm_task.cancel()
    autopilot_task.cancel()
    flush(tracer_provider)


app = FastAPI(
    title="Haystack RAG API",
    description="Growlyze Multimodal AI Chat — content extraction, ingestion, "
                "retrieval, and admin API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # CORS_ORIGINS env, comma-separated
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fail-closed auth (services/auth.py): 503 {"error": ...} when no API key is
# configured, instead of the default 500 wrap.
app.add_exception_handler(AdminKeyNotConfigured, admin_key_not_configured_handler)

# ── Mount routers ───────────────────────────────────────
app.include_router(extract.router, prefix="/extract", tags=["extraction"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(search.router, prefix="/search", tags=["retrieval"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(admin.data_router, prefix="/admin", tags=["admin"])
app.include_router(eval_router.router, prefix="/eval", tags=["evaluation"])
app.include_router(video.router, prefix="/video", tags=["video"])
# GET /costs/retrieval (P161). The router carries its own /costs prefix so
# routers/costs.py (GET /costs/chat) can mount alongside it.
app.include_router(costs_retrieval.router)
app.include_router(dashboard_router.router)  # GET /dashboard/* (own prefix)


@app.get("/health")
async def health():
    return {"status": "ok", "embedding": config.EMBEDDING_MODEL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
