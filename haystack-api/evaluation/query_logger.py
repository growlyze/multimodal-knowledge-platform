"""
Every search query, logged. Zero cost, always on.

The RAG filter calls this once per message (Session C wires it in); the admin
panel reads it back through /eval/queries and /eval/dashboard. Empty results are
the signal that matters — they are what coverage_analyzer clusters into gaps.

    logger = QueryLogger()
    log_id = await logger.log_query("what is a service level agreement", 5, 0.82,
                                    ["documents"], latency_ms=140)

Nothing here raises. A failed write returns the id anyway, so the caller can
still hand it to AnswerEvaluator without branching.
"""

from evaluation.eval_db import EvalDB, cutoff, get_eval_db, new_id

MAX_QUERY_CHARS = 2000      # a pasted CSV in the query box is not worth storing whole


class QueryLogger:
    def __init__(self, db: EvalDB = None):
        self.db = db or get_eval_db()

    async def log_query(self, query: str, results_count: int, top_score: float,
                        collections_searched: list = None, latency_ms: int = 0,
                        user_id: str = None, had_code_execution: bool = False,
                        had_file_attachment: bool = False) -> str:
        """Log a query. Returns log_id for linking to answer scores."""
        log_id = new_id()
        await self.db.execute(
            """
            INSERT INTO eval_query_log
                (id, user_id, query, results_count, top_score, collections_searched,
                 latency_ms, had_code_execution, had_file_attachment, empty_result)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            log_id, user_id, (query or "")[:MAX_QUERY_CHARS], int(results_count or 0),
            float(top_score) if top_score is not None else None,
            ",".join(collections_searched or []),
            int(latency_ms or 0), bool(had_code_execution), bool(had_file_attachment),
            not results_count,
        )
        return log_id

    async def get_recent(self, limit: int = 100) -> list:
        """Recent queries, newest first, for dashboard display."""
        return await self.db.fetch(
            """
            SELECT id, timestamp, user_id, query, results_count, top_score,
                   collections_searched, latency_ms, had_code_execution,
                   had_file_attachment, empty_result
            FROM eval_query_log ORDER BY timestamp DESC LIMIT $1
            """,
            max(1, int(limit)),
        )

    async def get_stats(self, hours: float = 24) -> dict:
        """count / empty rate / average top score over one window, in one round trip."""
        row = await self.db.fetchrow(
            """
            SELECT COUNT(*) AS queries,
                   SUM(CASE WHEN empty_result THEN 1 ELSE 0 END) AS empty,
                   AVG(top_score) AS avg_top_score,
                   AVG(latency_ms) AS avg_latency_ms
            FROM eval_query_log WHERE timestamp >= $1
            """,
            cutoff(hours),
        ) or {}
        queries = int(row.get("queries") or 0)
        empty = int(row.get("empty") or 0)
        return {
            "hours": hours,
            "query_count": queries,
            "empty_count": empty,
            "empty_result_rate": round(empty / queries, 4) if queries else 0.0,
            "avg_top_score": (round(float(row["avg_top_score"]), 4)
                              if row.get("avg_top_score") is not None else None),
            "avg_latency_ms": (round(float(row["avg_latency_ms"]))
                               if row.get("avg_latency_ms") is not None else None),
        }

    async def get_empty_rate(self, hours: float = 24) -> float:
        """Fraction of queries with no results in the last N hours."""
        return (await self.get_stats(hours))["empty_result_rate"]

    async def get_query_count(self, hours: float = 24) -> int:
        """Total queries in the last N hours."""
        return (await self.get_stats(hours))["query_count"]

    async def get_empty_queries(self, hours: float = 24, limit: int = 500) -> list:
        """The queries that came back empty — coverage_analyzer's raw material."""
        return await self.db.fetch(
            """
            SELECT query, timestamp FROM eval_query_log
            WHERE empty_result AND timestamp >= $1
            ORDER BY timestamp DESC LIMIT $2
            """,
            cutoff(hours), max(1, int(limit)),
        )
