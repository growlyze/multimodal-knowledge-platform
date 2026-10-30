"""
/eval — evaluation data for the admin panel.

Read-only except for the two buttons: run the retrieval test set, and mark a
coverage gap resolved. Everything degrades to empty rather than 500ing, because
the evaluation page must still render when PostgreSQL or the judge is down.

    GET  /eval/dashboard      metric cards
    GET  /eval/queries        recent query log
    GET  /eval/coverage       knowledge gaps
    POST /eval/coverage/resolve
    POST /eval/run-test-set   Layer 3 against the live index
    GET  /eval/test-runs      history of those runs
    GET  /eval/system         live counters
    GET  /eval/answer-scores  recent judge verdicts
    POST /eval/log-answer     the filter hands back an answer to be judged
"""

import asyncio
import json
import logging
import math
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from pydantic import BaseModel

from evaluation import retrieval_evaluator
from evaluation.answer_evaluator import MAX_ANSWER_CHARS, AnswerEvaluator, should_sample
from evaluation.coverage_analyzer import CoverageAnalyzer
from evaluation.eval_db import get_eval_db, new_id
from evaluation.query_logger import QueryLogger
from evaluation.system_metrics import get_metrics
from services.auth import require_admin, require_admin_or_loader

logger = logging.getLogger("haystack-api.eval")
# Every endpoint except POST /log-answer needs the admin key; log-answer is
# what the chat filter posts answers back to, so it takes either key
# (services/auth.py). Dependencies are per-route for exactly that reason.
router = APIRouter()

MAX_DETAILS_CHARS = 200_000     # a test run's JSON report, capped before it hits TEXT


def _logger() -> QueryLogger:
    return QueryLogger()


# ── answer scoring ──────────────────────────────────────

class AnswerLog(BaseModel):
    """What the RAG filter posts once the model has answered."""
    query: str
    answer: str
    log_id: str                     # from the SearchResponse that produced the context
    chunks: List[str] = []
    model: Optional[str] = None


async def _judge(payload: AnswerLog) -> None:
    """Run the judge and store the verdict against the query log row."""
    try:
        await AnswerEvaluator().evaluate(
            payload.query, payload.answer[:MAX_ANSWER_CHARS], payload.chunks,
            model=payload.model, query_log_id=payload.log_id)
    except Exception:
        logger.warning("answer judge failed for %s", payload.log_id, exc_info=True)


@router.post("/log-answer", dependencies=[Depends(require_admin_or_loader)])
async def log_answer(payload: AnswerLog, background_tasks: BackgroundTasks):
    """Score one answer for faithfulness, on the EVAL_SAMPLE_RATE sample.

    The filter posts here because the two halves live in different containers: the
    answer only exists in Open WebUI, the judge and its database only exist here.

    Returns immediately. The sampling coin flip is synchronous so the caller learns
    whether it was picked, but the judge call itself (one LLM round trip, up to 60s)
    runs in the background — the chat must never wait on evaluation.
    """
    if not payload.answer.strip():
        return {"sampled": False, "reason": "empty answer"}
    if not should_sample():
        return {"sampled": False}
    background_tasks.add_task(_judge, payload)
    return {"sampled": True, "log_id": payload.log_id}


# ── dashboard ───────────────────────────────────────────

@router.get("/dashboard", dependencies=[Depends(require_admin)])
async def dashboard(hours: float = Query(24, gt=0)):
    """Aggregate metrics for the dashboard cards."""
    stats, answers, gaps = await asyncio.gather(
        _logger().get_stats(hours),
        AnswerEvaluator().get_summary(),
        CoverageAnalyzer().get_gaps(limit=5),
    )
    system = get_metrics().get_current()
    return {
        "retrieval": {
            "empty_result_rate": stats["empty_result_rate"],
            "avg_top_score": stats["avg_top_score"],
            "query_count_24h": stats["query_count"],
            "avg_latency_ms": stats["avg_latency_ms"],
        },
        "answer_quality": answers,
        "system": {
            "search_latency_p50": system["search_latency_p50"],
            "search_latency_p95": system["search_latency_p95"],
            "error_rate": system["error_rate"],
            "uptime_hours": round(system["uptime_seconds"] / 3600, 1),
        },
        "coverage": {
            "gap_count": len(gaps),
            "top_gaps": [{"pattern": g["pattern"], "count": g["count"]} for g in gaps],
        },
    }


# ── query log ───────────────────────────────────────────

@router.get("/queries", dependencies=[Depends(require_admin)])
async def recent_queries(limit: int = Query(50, ge=1, le=1000)):
    """Recent query log for inspection."""
    rows = await _logger().get_recent(limit)
    return {"queries": rows, "count": len(rows)}


# ── coverage ────────────────────────────────────────────

@router.get("/coverage", dependencies=[Depends(require_admin)])
async def coverage_gaps(min_count: int = Query(3, ge=1),
                        limit: int = Query(50, ge=1, le=500),
                        include_resolved: bool = Query(False)):
    """Knowledge gaps — frequently empty queries."""
    gaps = await CoverageAnalyzer().get_gaps(min_count, limit, include_resolved)
    return {"gaps": gaps, "count": len(gaps), "min_count": min_count}


@router.post("/coverage/resolve", dependencies=[Depends(require_admin)])
async def resolve_gap(pattern: str = Query(..., min_length=1),
                      resolved: bool = Query(True)):
    """Mark a gap handled once its content has been ingested."""
    ok = await CoverageAnalyzer().mark_resolved(pattern, resolved)
    return {"pattern": pattern, "resolved": resolved, "updated": ok}


# ── test set (Layer 3) ──────────────────────────────────

def ndcg_from_scores(scores: list) -> float:
    """Binary-relevance NDCG over the first relevant rank of each query.

    ponytail: one relevant item per query, so IDCG is 1 and the whole thing is
    mean(1/log2(rank+1)). Graded NDCG needs per-result relevance labels, which
    the test set does not have — add them to test_queries.json first.
    """
    if not scores:
        return 0.0
    total = sum(1.0 / math.log2(s["first_relevant_rank"] + 1)
                for s in scores if s.get("first_relevant_rank"))
    return round(total / len(scores), 4)


async def run_test_set(run_id: str = None, url: str = None, top_k: int = 10,
                       db=None, evaluate=None) -> dict:
    """Run test_queries.json through the live /search and store the result."""
    run_id = run_id or new_id()
    db = db or get_eval_db()
    evaluate = evaluate or retrieval_evaluator.evaluate
    try:
        report = await evaluate(url=url or retrieval_evaluator.API_URL, top_k=top_k)
    except Exception as exc:
        logger.exception("test set run %s failed", run_id)
        return {"id": run_id, "error": "%s: %s" % (type(exc).__name__, exc)}

    metrics = report["metrics"]
    row = {
        "id": run_id,
        "precision_at_5": metrics["precision_at_5"],
        "recall_at_10": metrics["recall_at_10"],
        "mrr": metrics["mrr"],
        "ndcg": ndcg_from_scores(report["scores"]),
        "jargon_hit_rate": metrics["jargon_hit_rate"],
        "total_queries": report["queries"],
        "failed_queries": len(report["failures"]),
        "passed": report["passed"],
        "below_target": report["below_target"],
    }
    details = json.dumps({k: report[k] for k in
                          ("metrics", "targets", "below_target", "by_difficulty",
                           "failures", "passed")}, default=str)[:MAX_DETAILS_CHARS]
    await db.execute(
        """
        INSERT INTO eval_test_runs
            (id, precision_at_5, recall_at_10, mrr, ndcg, jargon_hit_rate,
             total_queries, failed_queries, details)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        run_id, row["precision_at_5"], row["recall_at_10"], row["mrr"], row["ndcg"],
        row["jargon_hit_rate"], row["total_queries"], row["failed_queries"], details,
    )
    logger.info("test set run %s: %s", run_id, row)
    return row


@router.post("/run-test-set", dependencies=[Depends(require_admin)])
async def trigger_test_set(background_tasks: BackgroundTasks,
                           top_k: int = Query(10, ge=1, le=50)):
    """Trigger test set evaluation. Returns the job ID it will be stored under."""
    run_id = new_id()
    background_tasks.add_task(run_test_set, run_id, None, top_k)
    return {"job_id": run_id, "status": "running",
            "poll": "/eval/test-runs — the row appears when the run finishes"}


@router.get("/test-runs", dependencies=[Depends(require_admin)])
async def test_run_history(limit: int = Query(10, ge=1, le=100)):
    """History of test set evaluation runs."""
    rows = await get_eval_db().fetch(
        """
        SELECT id, timestamp, precision_at_5, recall_at_10, mrr, ndcg,
               jargon_hit_rate, total_queries, failed_queries
        FROM eval_test_runs ORDER BY timestamp DESC LIMIT $1
        """,
        limit,
    )
    return {"runs": rows, "count": len(rows)}


@router.get("/test-runs/{run_id}", dependencies=[Depends(require_admin)])
async def test_run_detail(run_id: str):
    """One run with its full per-query report."""
    row = await get_eval_db().fetchrow("SELECT * FROM eval_test_runs WHERE id = $1", run_id)
    if row is None:
        return {"id": run_id, "status": "not_found"}
    row["details"] = json.loads(row["details"]) if row.get("details") else None
    return row


# ── system ──────────────────────────────────────────────

@router.get("/system", dependencies=[Depends(require_admin)])
async def system_health(history: int = Query(0, ge=0, le=168)):
    """Current system metrics, plus the last N flushed hours on request."""
    current = get_metrics().get_current()
    if history:
        current["history"] = await get_metrics().history(history)
    return current


# ── answer quality ──────────────────────────────────────

@router.get("/answer-scores", dependencies=[Depends(require_admin)])
async def answer_scores(limit: int = Query(20, ge=1, le=200),
                        hours: float = Query(24 * 7, gt=0)):
    """Recent answer quality evaluations."""
    evaluator = AnswerEvaluator()
    scores, summary = await asyncio.gather(evaluator.get_recent(limit),
                                           evaluator.get_summary(hours))
    return {"scores": scores, "summary": summary, "count": len(scores)}
