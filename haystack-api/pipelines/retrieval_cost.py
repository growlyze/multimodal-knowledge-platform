"""Per-question retrieval cost (P161 item 3): store and summarise.

routers/search.py opens a usage_meter.search_scope() around every search; every
paid call inside it (Voyage query embed, video embed_query, Voyage rerank, the
Gemini planner on a cache miss) records into that one in-memory meter. This
module writes the meter to `retrieval_usage` and reads it back for
GET /costs/retrieval. Retrieval spend is its own table on purpose: ingestion
lives in video_ingest_usage and chat-model spend in its own ledger, and the
admin Costs page shows the three side by side, never summed into one figure.

Off the request path, and never able to fail a search ("degrade, never break
the chat"): schedule() starts a task and returns at once. The task waits
PERSIST_GRACE_S before reading the meter, because some paid calls finish AFTER
the search has answered -- a planner past its 3 s budget, or a video leg cut at
VIDEO_LEG_DEADLINE_S, keeps running in its thread (asyncio.to_thread cannot be
cancelled) and is still billed. 12 s covers the planner's 10 s Gemini deadline.
A restart inside that window loses those rows; nothing else does.

Retention: raw rows older than RETENTION_DAYS are deleted from the same task,
at most once per PRUNE_EVERY_S per process -- there is no scheduler in this
service, and the write path is the one place guaranteed to run while rows
accumulate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from evaluation.eval_db import get_eval_db, new_id
from pipelines import usage_meter

logger = logging.getLogger("haystack-api.retrieval_cost")

PERSIST_GRACE_S = 12.0
RETENTION_DAYS = 90
PRUNE_EVERY_S = 3600.0

# Strong references: the event loop only keeps weak ones to tasks, so an
# unreferenced fire-and-forget task can be garbage-collected mid-flight.
_pending: set = set()
_last_prune = 0.0


def schedule(meter, query_log_id, user_id) -> None:
    """Persist this search's usage in the background. Never raises."""
    try:
        task = asyncio.get_running_loop().create_task(persist(meter, query_log_id, user_id))
    except Exception:
        logger.warning("could not schedule retrieval usage write", exc_info=True)
        return
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def persist(meter, query_log_id, user_id, db=None, grace_s: float = None) -> None:
    """Write one row (and maybe prune). Swallows every failure: EvalDB already
    turns a dead database into a no-op with a one-minute cooldown."""
    global _last_prune
    try:
        grace = PERSIST_GRACE_S if grace_s is None else grace_s
        if grace > 0:
            await asyncio.sleep(grace)
        db = db or get_eval_db()
        summary = meter.summary()
        await db.execute(
            "INSERT INTO retrieval_usage (id, query_log_id, user_id, usage, total_usd) "
            "VALUES ($1, $2, $3, $4::jsonb, $5)",
            new_id(), query_log_id, user_id, json.dumps(summary), float(summary["total_usd"]))
        now = time.monotonic()
        if now - _last_prune >= PRUNE_EVERY_S or _last_prune == 0.0:
            _last_prune = now
            await db.execute(
                "DELETE FROM retrieval_usage WHERE recorded_at < NOW() - make_interval(days => $1)",
                RETENTION_DAYS)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("retrieval usage write failed -- search unaffected", exc_info=True)


NOTE = ("Measured per search from each API's reported usage (Voyage tokens, Gemini tokens) x the "
        "official list price; Voyage's free token allowance means the invoice can be lower. "
        "Gemini planner calls served from its cache cost nothing and are not counted. "
        "Raw rows are kept %d days." % RETENTION_DAYS)


async def summarize(days: int, db=None) -> dict:
    """GET /costs/retrieval's body: totals, per service, per day (the daily
    rollup), average per search. available=false when the database is down."""
    db = db or get_eval_db()
    empty_services = [{"service": s, "requests": 0, "tokens": 0, "usd": 0.0}
                      for s in usage_meter.SEARCH_SERVICES]
    out = {"available": False, "period_days": days, "total_usd": 0.0, "searches": 0,
           "by_service": empty_services, "by_day": [], "per_search_avg_usd": 0.0,
           "prices_as_of": usage_meter.PRICES["retrieval_as_of"],
           "unit_prices": [u for u in usage_meter.unit_prices(retrieval=True)
                           if u["model"] in ("voyage-4-large", "voyage-multimodal-3.5",
                                             "rerank-2.5", "gemini-3.5-flash-lite")],
           "note": NOTE}
    if await db.pool() is None:
        out["note"] = "Cost database unreachable -- " + NOTE
        return out

    # total_usd is REAL (float4) per row; summed as float8 so thousands of
    # sub-cent rows don't drift. Validated on the live PostgreSQL 16 in a
    # rolled-back transaction, 2026-10-21.
    since = "recorded_at >= NOW() - make_interval(days => $1)"
    totals = await db.fetchrow(
        "SELECT COUNT(*) AS searches, COALESCE(SUM(total_usd::float8), 0) AS total_usd "
        "FROM retrieval_usage WHERE " + since, days) or {}
    services = await db.fetch(
        "SELECT s.key AS service, SUM((s.value->>'requests')::bigint) AS requests, "
        "SUM((s.value->>'tokens')::bigint) AS tokens, SUM((s.value->>'usd')::float8) AS usd "
        "FROM retrieval_usage, jsonb_each(usage->'services') AS s WHERE " + since
        + " GROUP BY s.key", days)
    by_day = await db.fetch(
        "SELECT to_char(date_trunc('day', recorded_at), 'YYYY-MM-DD') AS day, "
        "SUM(total_usd::float8) AS usd FROM retrieval_usage WHERE " + since
        + " GROUP BY day ORDER BY day", days)

    found = {r["service"]: r for r in services}
    searches = int(totals.get("searches") or 0)
    total = float(totals.get("total_usd") or 0.0)
    out.update({
        "available": True,
        "searches": searches,
        "total_usd": total,
        "per_search_avg_usd": total / searches if searches else 0.0,
        "by_service": [{"service": s,
                        "requests": int((found.get(s) or {}).get("requests") or 0),
                        "tokens": int((found.get(s) or {}).get("tokens") or 0),
                        "usd": float((found.get(s) or {}).get("usd") or 0.0)}
                       for s in usage_meter.SEARCH_SERVICES],
        "by_day": [{"date": r["day"], "usd": float(r["usd"] or 0.0)} for r in by_day],
    })
    return out
