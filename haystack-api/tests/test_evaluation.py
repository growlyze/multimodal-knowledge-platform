"""
Smallest thing that fails if evaluation breaks. Run: python tests/test_evaluation.py

No framework, matching the other test files here. PostgreSQL is replaced by
stdlib sqlite3 behind an asyncpg-shaped pool, so the real SQL in every module
actually executes — a broken column name fails here, not in production. The
judge is a fake callable; nothing reaches the network.
"""

import asyncio
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import answer_evaluator, eval_db, system_metrics
from evaluation.answer_evaluator import AnswerEvaluator
from evaluation.coverage_analyzer import CoverageAnalyzer, extract_pattern
from evaluation.eval_db import EvalDB
from evaluation.query_logger import QueryLogger
from evaluation.system_metrics import SystemMetrics, percentile
from routers import eval as eval_router


# ── asyncpg-shaped pool over sqlite ─────────────────────

_DEFAULT_NOW = re.compile(r"DEFAULT\s+NOW\(\)", re.I)
_NOW = re.compile(r"NOW\(\)", re.I)
_SERIAL = re.compile(r"\bSERIAL\s+PRIMARY\s+KEY\b", re.I)
_PARAM = re.compile(r"\$\d+")
_ADD_COLUMN_IF_NOT_EXISTS = re.compile(r"ADD COLUMN IF NOT EXISTS", re.I)


def translate(sql: str, args: tuple):
    """PostgreSQL dialect -> sqlite. Only what our own statements use."""
    sql = _DEFAULT_NOW.sub("DEFAULT (datetime('now','localtime'))", sql)
    sql = _NOW.sub("datetime('now','localtime')", sql)
    sql = _SERIAL.sub("INTEGER PRIMARY KEY AUTOINCREMENT", sql)
    # sqlite's ADD COLUMN has no IF NOT EXISTS — FakeConn.execute below
    # swallows the resulting "duplicate column" error instead, which is
    # the same idempotency Postgres gets from the clause itself.
    sql = _ADD_COLUMN_IF_NOT_EXISTS.sub("ADD COLUMN", sql)
    sql = _PARAM.sub("?", sql)          # our params are always numbered in order
    bound = [v.strftime("%Y-%m-%d %H:%M:%S") if isinstance(v, datetime)
             else int(v) if isinstance(v, bool) else v for v in args]
    return sql, bound


class FakeConn:
    def __init__(self, raw):
        self.raw = raw

    async def execute(self, sql, *args):
        translated, bound = translate(sql, args)
        try:
            self.raw.execute(translated, bound)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise
        self.raw.commit()
        return "OK"

    async def fetch(self, sql, *args):
        return self.raw.execute(*translate(sql, args)).fetchall()

    async def fetchrow(self, sql, *args):
        return self.raw.execute(*translate(sql, args)).fetchone()

    async def fetchval(self, sql, *args):
        row = await self.fetchrow(sql, *args)
        return row[0] if row is not None else None


class FakePool:
    """Enough of an asyncpg pool for EvalDB. `fail=True` plays a dead database."""

    def __init__(self, fail=False):
        self.fail = fail
        self.raw = sqlite3.connect(":memory:")
        self.raw.row_factory = sqlite3.Row

    def acquire(self):
        pool = self

        class _Acquire:
            async def __aenter__(self):
                if pool.fail:
                    raise ConnectionError("postgres is down")
                return FakeConn(pool.raw)

            async def __aexit__(self, *exc):
                return False

        return _Acquire()

    async def close(self):
        self.raw.close()


def fresh_db(fail=False) -> EvalDB:
    return EvalDB(dsn="sqlite://memory", pool=FakePool(fail=fail))


def install(db: EvalDB):
    """Point every singleton at this database — the routers use them directly."""
    eval_db._db = db
    system_metrics._metrics = SystemMetrics(db=db)
    return db


# ── M3 measured ingestion usage ─────────────────────────

async def test_record_usage_round_trips_through_real_sql():
    """record_usage's SQL runs against the real schema (sqlite here). The
    cost_tracker self-test only captures arguments, which would pass even if
    the table or the statement were broken -- the gap that let a Postgres-only
    DEFAULT silently disable this whole fake DB during development."""
    import json
    from pipelines.cost_tracker import record_usage, total_spend
    from pipelines.usage_meter import UsageMeter
    db = fresh_db()
    m = UsageMeter("vid-u")
    m.record_groq(600.0)
    m.record_voyage_text(5000)
    usage = m.summary()
    await record_usage(db, "vid-u", usage, {"asr": 41.0}, "done")
    await record_usage(db, "vid-u", usage, {"asr": 40.0}, "done")   # upsert, not duplicate
    row = await db.fetchrow("SELECT usage, phase_seconds, prices_as_of, total_usd, status "
                            "FROM video_ingest_usage WHERE video_id = $1", "vid-u")
    assert row is not None
    assert json.loads(row["usage"])["groq"]["audio_seconds"] == 600.0
    assert json.loads(row["phase_seconds"]) == {"asr": 40.0}
    assert row["status"] == "done" and row["prices_as_of"] == usage["prices_as_of"]
    assert abs(await total_spend(db) - usage["total_usd"]) < 1e-6


async def test_video_jobs_warnings_column_exists_on_a_fresh_database():
    """The warnings column (P137) must exist after schema init on a FRESH
    database. A migration placed before its table's CREATE fails there, and
    EvalDB swallows the error, so only a read-back proves it."""
    db = fresh_db()
    await db.execute("INSERT INTO video_ingest_jobs (video_id, status, warnings) VALUES ($1, $2, $3)",
                     "vw1", "done", "transcript stops at 49%")
    row = await db.fetchrow("SELECT warnings FROM video_ingest_jobs WHERE video_id = $1", "vw1")
    assert row is not None and row["warnings"] == "transcript stops at 49%", row


# ── query logger ────────────────────────────────────────

async def test_query_logger_stores_and_retrieves():
    log = QueryLogger(fresh_db())
    log_id = await log.log_query("what is a service level agreement", 5, 0.82,
                                 ["documents"], latency_ms=140,
                                 user_id="alice", had_file_attachment=True)
    assert log_id and len(log_id) == 32

    rows = await log.get_recent(10)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["id"] == log_id
    assert row["query"] == "what is a service level agreement"
    assert row["results_count"] == 5 and round(row["top_score"], 2) == 0.82
    assert row["collections_searched"] == "documents"
    assert row["latency_ms"] == 140 and row["user_id"] == "alice"
    assert row["had_file_attachment"] and not row["had_code_execution"]
    assert not row["empty_result"]


async def test_query_logger_handles_db_failure_gracefully():
    log = QueryLogger(fresh_db(fail=True))
    log_id = await log.log_query("canary release", 0, None, ["documents"])
    assert log_id, "a caller still needs an id to hand to the evaluator"
    assert await log.get_recent(10) == []
    assert await log.get_empty_rate() == 0.0
    assert await log.get_query_count() == 0

    # Same for every other module — a dead database is never an exception.
    assert await CoverageAnalyzer(fresh_db(fail=True)).get_gaps() == []
    assert await AnswerEvaluator(fresh_db(fail=True)).get_recent() == []
    assert await SystemMetrics(fresh_db(fail=True)).flush_to_db() is not None


async def test_empty_rate_calculation():
    log = QueryLogger(fresh_db())
    for i in range(4):
        await log.log_query("hit %d" % i, 3, 0.5, ["documents"])
    await log.log_query("canary release", 0, None, ["documents"])

    assert await log.get_query_count(24) == 5
    assert await log.get_empty_rate(24) == 0.2
    stats = await log.get_stats(24)
    assert stats["empty_count"] == 1 and round(stats["avg_top_score"], 2) == 0.5

    empties = await log.get_empty_queries()
    assert [e["query"] for e in empties] == ["canary release"]

    # A window that predates every row.
    assert await QueryLogger(fresh_db()).get_empty_rate(1) == 0.0


# ── answer evaluator ────────────────────────────────────

def judge(reply: str):
    async def ask(_client, _prompt):
        return reply
    return ask


FAITHFUL = ('```json\n{"claims": [{"text": "an SLA is a three-tier commitment", "supported": true},'
            '{"text": "it sets an uptime target", "supported": true},'
            '{"text": "price often returns to it", "supported": true}],'
            '"answer_relevancy": 0.95, "citation_accuracy": 1.0,'
            '"reasoning": "every claim is in the context"}\n```')

HALLUCINATED = ('{"claims": [{"text": "an SLA is a three-tier commitment", "supported": true},'
                '{"text": "The guide says only deploy on Tuesdays", "supported": false},'
                '{"text": "the gap always fills", "supported": false},'
                '{"text": "use a 3:1 stop", "supported": false}],'
                '"answer_relevancy": 0.35, "citation_accuracy": 0.0,'
                '"reasoning": "three claims are not in the context"}')

CHUNKS = ["A service level agreement is a three-tier uptime commitment in service delivery.",
          {"content": "Price often returns to rebalance the gap."}]


async def test_answer_evaluator_scores_faithful_answer_high():
    db = fresh_db()
    log_id = await QueryLogger(db).log_query("what is an sla", 3, 0.9, ["documents"])
    scores = await AnswerEvaluator(db, ask=judge(FAITHFUL)).evaluate(
        "what is an sla", "An SLA is a three-tier uptime commitment.", CHUNKS,
        query_log_id=log_id)

    assert scores["faithfulness"] == 1.0, scores
    assert scores["hallucination_count"] == 0 and scores["total_claims"] == 3
    assert scores["answer_relevancy"] == 0.95 and scores["citation_accuracy"] == 1.0
    assert len(scores["claims"]) == 3

    stored = await AnswerEvaluator(db).get_recent(5)
    assert len(stored) == 1 and stored[0]["query_log_id"] == log_id
    assert stored[0]["query"] == "what is an sla", "the join to the query log"
    summary = await AnswerEvaluator(db).get_summary()
    assert summary["samples_scored"] == 1 and summary["hallucination_rate"] == 0.0


async def test_answer_evaluator_scores_hallucinated_answer_low():
    db = fresh_db()
    scores = await AnswerEvaluator(db, ask=judge(HALLUCINATED)).evaluate(
        "what is an sla", "Only deploy on Tuesdays.", CHUNKS, query_log_id="q1")

    assert scores["faithfulness"] == 0.25, scores
    assert scores["hallucination_count"] == 3 and scores["total_claims"] == 4
    assert scores["answer_relevancy"] < 0.5 and scores["citation_accuracy"] == 0.0

    summary = await AnswerEvaluator(db).get_summary()
    assert summary["faithfulness_avg"] == 0.25
    assert summary["hallucination_rate"] == 0.75, summary


async def test_answer_evaluator_handles_judge_failure():
    async def down(_client, _prompt):
        raise TimeoutError("judge timed out")

    scores = await AnswerEvaluator(fresh_db(), ask=down).evaluate("q", "a", CHUNKS)
    assert scores["faithfulness"] is None and "judge timed out" in scores["error"]

    garbage = await AnswerEvaluator(fresh_db(), ask=judge("sorry, I can't")).evaluate(
        "q", "a", CHUNKS)
    assert garbage["faithfulness"] is None and "no JSON" in garbage["error"]

    empty = await AnswerEvaluator(fresh_db(), ask=judge(FAITHFUL)).evaluate("q", "  ", CHUNKS)
    assert empty["error"] == "empty answer"

    # Sampling is a coin flip, not a call.
    assert answer_evaluator.should_sample(rate=0.0) is False
    assert answer_evaluator.should_sample(rate=1.0) is True
    assert await AnswerEvaluator(fresh_db(), ask=judge(FAITHFUL)).sample_and_evaluate(
        "q", "a", CHUNKS, rate=0.0) is None


async def test_answer_judge_is_off_by_default():
    """2026-10-21 decision: the answer-quality judge stays off. With
    EVAL_SAMPLE_RATE unset the default is 0.0, and 0.0 must mean no judge call
    ever -- not "rarely" -- through both the evaluator and POST /eval/log-answer."""
    import importlib
    import os
    from fastapi import BackgroundTasks

    saved_env = os.environ.pop("EVAL_SAMPLE_RATE", None)
    try:
        importlib.reload(answer_evaluator)
        assert answer_evaluator.SAMPLE_RATE == 0.0, answer_evaluator.SAMPLE_RATE
        assert not any(answer_evaluator.should_sample() for _ in range(20_000))

        calls = []

        async def counting(_client, _prompt):
            calls.append(1)
            return FAITHFUL

        for _ in range(200):
            assert await AnswerEvaluator(fresh_db(), ask=counting).sample_and_evaluate(
                "q", "an answer", CHUNKS) is None
        assert calls == []

        # The route imported should_sample by name: reload it too, as a fresh
        # process would, then check nothing is scheduled.
        importlib.reload(eval_router)
        tasks = BackgroundTasks()
        out = await eval_router.log_answer(
            eval_router.AnswerLog(query="q", answer="an answer", log_id="L1"), tasks)
        assert out == {"sampled": False} and tasks.tasks == [], (out, tasks.tasks)

        # A negative or zero override is off as well, never an error.
        assert answer_evaluator.should_sample(rate=-1.0) is False
    finally:
        if saved_env is not None:
            os.environ["EVAL_SAMPLE_RATE"] = saved_env
        importlib.reload(answer_evaluator)
        importlib.reload(eval_router)


# ── system metrics ──────────────────────────────────────

async def test_system_metrics_percentiles():
    metrics = SystemMetrics(fresh_db())
    for ms in range(1, 101):
        metrics.record_search_latency(ms)
    for ms in (100, 200, 300):
        metrics.record_embedding_latency(ms)
    metrics.record_query()
    metrics.record_query()
    metrics.record_error()

    current = metrics.get_current()
    assert current["search_latency_p50"] == 50 and current["search_latency_p95"] == 95
    assert current["embedding_latency_avg"] == 200
    assert current["error_rate"] == 0.5 and current["uptime_seconds"] >= 0

    assert percentile([], 50) is None
    assert percentile([7], 95) == 7
    assert percentile([1, 2, 3, 4], 50) == 2


async def test_system_metrics_flush_and_reset():
    db = fresh_db()
    metrics = SystemMetrics(db)
    metrics.record_search_latency(120)
    metrics.record_search_latency(480)
    metrics.record_query()
    metrics.record_empty_result()
    metrics.record_voyage_429()

    flushed = await metrics.flush_to_db()
    assert flushed["search_latency_p50"] == 120 and flushed["search_latency_p95"] == 480
    assert flushed["query_count"] == 1 and flushed["voyage_429_count"] == 1

    after = metrics.get_current()
    assert after["query_count"] == 0 and after["search_latency_p50"] is None

    history = await metrics.history(5)
    assert len(history) == 1 and history[0]["empty_result_count"] == 1
    assert history[0]["period"] == "hourly"

    # An idle hour writes nothing.
    assert await metrics.flush_to_db() is not None
    assert len(await metrics.history(5)) == 1


# ── coverage analyzer ───────────────────────────────────

async def test_coverage_analyzer_groups_similar_queries():
    db = fresh_db()
    analyzer = CoverageAnalyzer(db)
    for query in ("What is the Canary Release?",
                  "explain canary release entries",
                  "CANARY RELEASE setup on ACME",
                  "How do feature flags work?"):
        await analyzer.record_empty_query(query)

    assert extract_pattern("What is the Canary Release?") == "canary release"
    assert await analyzer.record_empty_query("???") == ""

    gaps = await analyzer.get_gaps(min_count=1)
    assert [(g["pattern"], g["count"]) for g in gaps] == [
        ("Canary Release", 3), ("Feature Flags", 1)], gaps

    assert [g["pattern"] for g in await analyzer.get_gaps(min_count=3)] == ["Canary Release"]
    assert await analyzer.get_gaps(min_count=4) == []

    # Backfill rebuilds the same gaps from the query log alone.
    log = QueryLogger(db)
    await log.log_query("what is a canary release", 0, None, [])
    assert await CoverageAnalyzer(db).backfill(24) == 1
    assert (await analyzer.get_gaps(min_count=1))[0]["count"] == 4


async def test_coverage_analyzer_mark_resolved():
    db = fresh_db()
    analyzer = CoverageAnalyzer(db)
    for _ in range(3):
        await analyzer.record_empty_query("what is the canary release")

    assert await analyzer.mark_resolved("Canary Release") is True
    assert await analyzer.get_gaps(min_count=1) == []

    still_there = await analyzer.get_gaps(min_count=1, include_resolved=True)
    assert len(still_there) == 1 and still_there[0]["resolved"] is True

    # Resolving is sticky — a later occurrence counts but stays hidden.
    await analyzer.record_empty_query("canary release entry model")
    assert await analyzer.get_gaps(min_count=1) == []
    assert (await analyzer.get_gaps(min_count=1, include_resolved=True))[0]["count"] == 4

    assert await analyzer.mark_resolved("Canary Release", resolved=False) is True
    assert len(await analyzer.get_gaps(min_count=1)) == 1


# ── endpoints ───────────────────────────────────────────

async def test_dashboard_endpoint_returns_all_sections():
    db = install(fresh_db())
    log = QueryLogger(db)
    for i in range(3):
        await log.log_query("service level agreement %d" % i, 4, 0.7 + i / 100, ["documents"],
                            latency_ms=100 + i)
    log_id = await log.log_query("what is the canary release", 0, None, ["documents"])
    analyzer = CoverageAnalyzer(db)
    for _ in range(3):
        await analyzer.record_empty_query("what is the canary release")
    await AnswerEvaluator(db, ask=judge(HALLUCINATED)).evaluate(
        "what is an sla", "answer", CHUNKS, query_log_id=log_id)
    system_metrics.get_metrics().record_search_latency(120)
    system_metrics.get_metrics().record_query()

    board = await eval_router.dashboard(24)     # Query() defaults only bind under FastAPI
    assert set(board) == {"retrieval", "answer_quality", "system", "coverage"}
    assert board["retrieval"]["query_count_24h"] == 4
    assert board["retrieval"]["empty_result_rate"] == 0.25
    assert board["retrieval"]["avg_top_score"] is not None
    assert board["answer_quality"]["samples_scored"] == 1
    assert board["answer_quality"]["faithfulness_avg"] == 0.25
    assert board["system"]["search_latency_p50"] == 120
    assert board["system"]["uptime_hours"] >= 0
    assert board["coverage"]["gap_count"] == 1
    assert board["coverage"]["top_gaps"][0] == {"pattern": "Canary Release", "count": 3}

    assert (await eval_router.recent_queries(50))["count"] == 4
    assert (await eval_router.coverage_gaps(3, 50, False))["count"] == 1
    assert (await eval_router.answer_scores(20, 24 * 7))["count"] == 1
    assert (await eval_router.system_health(0))["query_count"] == 1

    resolved = await eval_router.resolve_gap("Canary Release", True)
    assert resolved["updated"] and (await eval_router.coverage_gaps(1, 50, False))["count"] == 0

    # Dashboard still renders with the database gone — this is the whole point.
    install(fresh_db(fail=True))
    dead = await eval_router.dashboard(24)
    assert dead["retrieval"]["query_count_24h"] == 0
    assert dead["coverage"]["gap_count"] == 0
    assert dead["answer_quality"]["samples_scored"] == 0


async def test_test_set_runner_computes_metrics():
    db = install(fresh_db())
    report = {
        "queries": 4,
        "metrics": {"precision_at_5": 0.8, "recall_at_10": 0.85, "mrr": 0.75,
                    "jargon_hit_rate": 0.9, "cross_collection_rate": 1.0},
        "targets": {"precision_at_5": 0.8},
        "below_target": {},
        "by_difficulty": {"easy": {"queries": 4}},
        "failures": [{"query": "canary release", "failure": "no relevant result in 10 returned"}],
        "passed": True,
        "scores": [{"first_relevant_rank": 1}, {"first_relevant_rank": 3},
                   {"first_relevant_rank": 7}, {"first_relevant_rank": None}],
    }

    async def fake_evaluate(url=None, top_k=10):
        assert top_k == 10
        return report

    row = await eval_router.run_test_set(db=db, evaluate=fake_evaluate)
    # 1/log2(2) + 1/log2(4) + 1/log2(8) + 0, over 4 queries
    assert row["ndcg"] == round((1 + 0.5 + 1 / 3) / 4, 4) == 0.4583, row
    assert row["precision_at_5"] == 0.8 and row["mrr"] == 0.75
    assert row["total_queries"] == 4 and row["failed_queries"] == 1

    history = await eval_router.test_run_history(10)
    assert history["count"] == 1 and history["runs"][0]["ndcg"] == row["ndcg"]

    detail = await eval_router.test_run_detail(row["id"])
    assert detail["details"]["metrics"]["mrr"] == 0.75
    assert (await eval_router.test_run_detail("nope"))["status"] == "not_found"

    assert eval_router.ndcg_from_scores([]) == 0.0
    assert eval_router.ndcg_from_scores([{"first_relevant_rank": None}]) == 0.0

    async def boom(url=None, top_k=10):
        raise RuntimeError("search is down")

    failed = await eval_router.run_test_set(db=db, evaluate=boom)
    assert "search is down" in failed["error"]
    assert (await eval_router.test_run_history(10))["count"] == 1, "no row for a failed run"


async def main():
    tests = [
        test_video_jobs_warnings_column_exists_on_a_fresh_database,
        test_record_usage_round_trips_through_real_sql,
        test_query_logger_stores_and_retrieves,
        test_query_logger_handles_db_failure_gracefully,
        test_empty_rate_calculation,
        test_answer_evaluator_scores_faithful_answer_high,
        test_answer_evaluator_scores_hallucinated_answer_low,
        test_answer_evaluator_handles_judge_failure,
        test_answer_judge_is_off_by_default,
        test_system_metrics_percentiles,
        test_system_metrics_flush_and_reset,
        test_coverage_analyzer_groups_similar_queries,
        test_coverage_analyzer_mark_resolved,
        test_dashboard_endpoint_returns_all_sections,
        test_test_set_runner_computes_metrics,
    ]
    for test in tests:
        await test()
        print("ok  %s" % test.__name__)
    print("\n%d evaluation checks passed" % len(tests))


if __name__ == "__main__":
    asyncio.run(main())
