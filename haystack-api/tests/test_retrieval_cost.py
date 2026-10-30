"""
Per-question retrieval cost (P161 item 3): every paid call /search makes is
metered into one SearchUsageMeter, persisted to retrieval_usage off the request
path, and summarised by GET /costs/retrieval.

    python tests/test_retrieval_cost.py

No PostgreSQL, no Voyage, no Gemini: the HTTP helpers, the voyageai client and
the Gemini call are replaced at their boundaries; the database is a FakeDB that
records the SQL it was handed.
"""
import asyncio
import os
import sys
import time
from types import SimpleNamespace

# Run as `python tests/test_retrieval_cost.py`: put the package root on the path
# (three sibling suites forget this and need PYTHONPATH=., see CLAUDE.md).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EVAL_DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")

import config
from pipelines import query_planner, retrieval, retrieval_cost, usage_meter, video_embedder
from routers import costs_retrieval
from routers import search as search_router
from services import embedding_service, reranker_service

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


class FakeDB:
    """Records every statement; fetch()/fetchrow() answer from canned rows by
    matching a substring of the SQL. `alive=False` is a dead database."""

    def __init__(self, alive=True, rows=None, hang=False):
        self.alive, self.rows, self.hang = alive, rows or {}, hang
        self.executed = []

    async def pool(self):
        return "pool" if self.alive else None

    async def execute(self, sql, *args):
        if self.hang:
            await asyncio.sleep(3600)
        self.executed.append((" ".join(sql.split()), args))
        return self.alive

    async def fetch(self, sql, *args):
        for key, rows in self.rows.items():
            if key in sql:
                return rows
        return []

    async def fetchrow(self, sql, *args):
        rows = await self.fetch(sql, *args)
        return rows[0] if rows else None


def _svc(meter, name):
    return meter.summary()["services"][name]


# ── service boundaries ──────────────────────────────────

@check("Voyage text query embedding is metered from usage.total_tokens")
async def _():
    async def fake_post(client, url, payload, headers, attempts=4):
        return {"data": [{"index": 0, "embedding": [0.1]}], "usage": {"total_tokens": 9}}

    saved = embedding_service.post_with_retry
    embedding_service.post_with_retry = fake_post
    try:
        provider = embedding_service.VoyageProvider(api_key="k", model="voyage-4-large", dimensions=1)
        await provider.embed(["outside any search"], input_type="query")   # not metered: no scope
        with usage_meter.search_scope() as m:
            await provider.embed(["what is an sla"], input_type="query")
    finally:
        embedding_service.post_with_retry = saved
    assert _svc(m, "voyage_query_embed") == {
        "requests": 1, "tokens": 9, "usd": 9 * usage_meter.PRICES["voyage_text_per_m_tokens"] / 1e6}


@check("a Voyage response without usage still counts the request (0 tokens)")
async def _():
    async def fake_post(client, url, payload, headers, attempts=4):
        return {"data": [{"index": 0, "embedding": [0.1]}]}

    saved = embedding_service.post_with_retry
    embedding_service.post_with_retry = fake_post
    try:
        with usage_meter.search_scope() as m:
            await embedding_service.VoyageProvider(api_key="k", dimensions=1).embed(["q"], "query")
    finally:
        embedding_service.post_with_retry = saved
    assert _svc(m, "voyage_query_embed")["requests"] == 1


@check("Voyage rerank is metered from usage.total_tokens at the rerank-2.5 rate")
async def _():
    async def fake_post(client, url, payload, headers, attempts=4):
        return {"data": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}],
                "usage": {"total_tokens": 4000}}

    saved = reranker_service.post_with_retry
    reranker_service.post_with_retry = fake_post
    try:
        with usage_meter.search_scope() as m:
            ranking = await reranker_service.VoyageReranker(api_key="k").rerank("q", ["a", "b"], 2)
            await reranker_service.VoyageReranker(api_key="k").rerank("q", ["a", "b"], 2)
    finally:
        reranker_service.post_with_retry = saved
    assert ranking == [(1, 0.9), (0, 0.4)], "metering must not change the ranking"
    r = _svc(m, "voyage_rerank")
    assert (r["requests"], r["tokens"]) == (2, 8000), r
    assert abs(r["usd"] - 8000 * 0.05 / 1e6) < 1e-15, r


@check("video embed_query meters both Voyage calls as voyage_video_query")
async def _():
    class FakeClient:
        def __init__(self, **kw):
            pass

        def embed(self, texts, model=None, input_type=None):
            return SimpleNamespace(embeddings=[[0.0] * 4], total_tokens=5)

        def multimodal_embed(self, inputs, model=None, input_type=None):
            return SimpleNamespace(embeddings=[[0.0] * 4], text_tokens=6, image_pixels=0,
                                   total_tokens=6)

    sys.modules["voyageai"] = SimpleNamespace(Client=FakeClient)
    saved_key = config.EMBEDDING_API_KEY
    config.EMBEDDING_API_KEY = "fake"
    video_embedder._voyage_limiter._interval = 0.0
    try:
        with usage_meter.search_scope() as m:
            await asyncio.to_thread(video_embedder.embed_query, "sla")   # as retrieval calls it
    finally:
        del sys.modules["voyageai"]
        config.EMBEDDING_API_KEY = saved_key
    v = _svc(m, "voyage_video_query")
    assert (v["requests"], v["tokens"]) == (2, 11), v
    p = usage_meter.PRICES
    assert abs(v["usd"] - (5 * p["voyage_text_per_m_tokens"] + 6 * p["voyage_mm_per_m_tokens"]) / 1e6) < 1e-15


@check("the Gemini planner is metered on a real call, not on a cache hit")
async def _():
    calls = []

    def fake_response(prompt):
        calls.append(prompt)
        return SimpleNamespace(
            parsed=SimpleNamespace(model_dump=lambda: {"queries": ["sla"], "video_name": None,
                                                       "intent": "passage"}),
            usage_metadata=SimpleNamespace(prompt_token_count=300, candidates_token_count=20,
                                           thoughts_token_count=0))

    saved = query_planner._gemini_response
    query_planner._gemini_response = fake_response
    query_planner._cache.clear()
    try:
        with usage_meter.search_scope() as m:
            await asyncio.to_thread(query_planner.plan_video_queries, "explain the sla please")
            await asyncio.to_thread(query_planner.plan_video_queries, "explain the sla please")
    finally:
        query_planner._gemini_response = saved
        query_planner._cache.clear()
    assert len(calls) == 1
    g = _svc(m, "gemini_planner")
    assert (g["requests"], g["tokens"]) == (1, 320), g
    assert abs(g["usd"] - (300 * 0.30 + 20 * 2.50) / 1e6) < 1e-15, g


@check("real retrieval.search: text embed + rerank (incl. the comparison rescue) land in one meter")
async def _():
    class TinyQdrant:
        async def search_dense(self, name, vector, k, filters):
            return [{"id": "c1", "collection": name, "score": 0.5,
                     "payload": {"content": "strategy.entry places an order"}}]

        async def search_sparse(self, name, query, k, filters):
            return []

    reranks = []

    async def fake_post(client, url, payload, headers, attempts=4):
        if url.endswith("/rerank"):
            reranks.append(payload["query"])
            # Below the 0.70 floor -> empty page -> comparison rescue -> search() again
            return {"data": [{"index": 0, "relevance_score": 0.5}], "usage": {"total_tokens": 100}}
        return {"data": [{"index": 0, "embedding": [0.1]}], "usage": {"total_tokens": 12}}

    saved = (embedding_service.post_with_retry, reranker_service.post_with_retry,
             embedding_service._provider, reranker_service._reranker, retrieval.get_qdrant,
             retrieval.search)
    embedding_service.post_with_retry = reranker_service.post_with_retry = fake_post
    embedding_service._provider = embedding_service.VoyageProvider(api_key="k", dimensions=1)
    reranker_service._reranker = reranker_service.VoyageReranker(api_key="k")
    retrieval.get_qdrant = lambda: TinyQdrant()
    retrieval.search = _REAL_SEARCH          # the rescue calls retrieval's own search()
    try:
        with usage_meter.search_scope() as m:
            await _REAL_SEARCH("what is the difference between strategy.entry and strategy.order?",
                               collections=["documents"])
    finally:
        (embedding_service.post_with_retry, reranker_service.post_with_retry,
         embedding_service._provider, reranker_service._reranker, retrieval.get_qdrant,
         retrieval.search) = saved
    assert len(reranks) == 2, reranks                      # original + rescued query
    s = m.summary()["services"]
    assert (s["voyage_query_embed"]["requests"], s["voyage_query_embed"]["tokens"]) == (2, 24), s
    assert (s["voyage_rerank"]["requests"], s["voyage_rerank"]["tokens"]) == (2, 200), s


_REAL_SEARCH = retrieval.search


# ── /search: one row per search, off the request path ───

def _patch_db(db):
    for module in (retrieval_cost, search_router):
        module.get_eval_db = lambda: db


async def _settle():
    """Let fire-and-forget persistence tasks finish (grace is 0 in tests)."""
    for _ in range(3):
        pending = list(retrieval_cost._pending)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)


def _rows(db):
    return [args for sql, args in db.executed if sql.startswith("INSERT INTO retrieval_usage")]


async def _call(q, user_id="u1"):
    return await search_router.search_knowledge(
        q=q, top_k=5, collections=None, source_type=None, user_id=user_id,
        had_code_execution=False, had_file_attachment=False, in_scope=True)


HIT = {"score": 0.9, "collection": "documents", "id": "c1",
       "payload": {"content": "an SLA is a three-tier uptime commitment", "source": "guide.pdf"}}


@check("/search stores one retrieval_usage row with log id, user and total")
async def _():
    import json

    async def fake_search(q, **kw):
        usage_meter.record_search_voyage_text("voyage_query_embed", 10)
        usage_meter.record_search_rerank(2000)
        return [HIT]

    db = FakeDB()
    _patch_db(db)
    retrieval.search = fake_search
    resp = await _call("what is an sla")
    await _settle()
    rows = _rows(db)
    assert len(rows) == 1, db.executed
    row_id, log_id, user_id, usage_json, total = rows[0]
    usage = json.loads(usage_json)
    # Linked to the eval_query_log row this same search created.
    assert log_id is not None and log_id == resp.log_id and user_id == "u1", (log_id, resp.log_id)
    assert usage["services"]["voyage_rerank"]["tokens"] == 2000
    assert abs(total - (10 * 0.12 + 2000 * 0.05) / 1e6) < 1e-15
    assert resp.results and resp.results[0].content.startswith("an SLA")


@check("two concurrent /search calls each store only their own usage")
async def _():
    async def fake_search(q, **kw):
        n = 1 if q == "small" else 1000
        for _ in range(10):
            await asyncio.sleep(0.001)          # the two searches interleave here
            await asyncio.to_thread(usage_meter.record_search_rerank, n)
        return [HIT]

    db = FakeDB()
    _patch_db(db)
    retrieval.search = fake_search
    await asyncio.gather(_call("small", "a"), _call("large", "b"))
    await _settle()
    import json
    by_user = {r[2]: json.loads(r[3])["services"]["voyage_rerank"]["tokens"] for r in _rows(db)}
    assert by_user == {"a": 10, "b": 10_000}, by_user


@check("persisting adds no latency: a hung database never delays the search")
async def _():
    async def fake_search(q, **kw):
        usage_meter.record_search_rerank(1)
        return [HIT]

    db = FakeDB(hang=True)
    _patch_db(db)
    retrieval.search = fake_search
    started = time.perf_counter()
    resp = await asyncio.wait_for(_call("q"), timeout=2.0)
    assert time.perf_counter() - started < 1.0 and resp.results
    for t in list(retrieval_cost._pending):
        t.cancel()
    await asyncio.sleep(0)


@check("a dead database: search answers, nothing raised")
async def _():
    async def fake_search(q, **kw):
        usage_meter.record_search_rerank(1)
        return [HIT]

    db = FakeDB(alive=False)
    _patch_db(db)
    retrieval.search = fake_search
    resp = await _call("q")
    await _settle()
    assert resp.results


@check("a failed search still stores the usage it had already paid for")
async def _():
    from fastapi import HTTPException

    async def failing_search(q, **kw):
        usage_meter.record_search_voyage_text("voyage_query_embed", 7)
        raise RuntimeError("qdrant down")

    db = FakeDB()
    _patch_db(db)
    retrieval.search = failing_search
    try:
        await _call("q")
        raise AssertionError("expected a 502")
    except HTTPException as exc:
        assert exc.status_code == 502
    await _settle()
    rows = _rows(db)
    assert len(rows) == 1 and rows[0][1] is None, rows


@check("raw rows are pruned past 90 days, at most once per interval")
async def _():
    db = FakeDB()
    retrieval_cost._last_prune = 0.0
    m = usage_meter.SearchUsageMeter()
    await retrieval_cost.persist(m, None, None, db=db, grace_s=0)
    await retrieval_cost.persist(m, None, None, db=db, grace_s=0)
    deletes = [(s, a) for s, a in db.executed if s.startswith("DELETE FROM retrieval_usage")]
    assert len(deletes) == 1, db.executed
    assert deletes[0][1] == (retrieval_cost.RETENTION_DAYS,) and retrieval_cost.RETENTION_DAYS == 90


@check("the retrieval_usage table is created with the other eval tables")
async def _():
    from evaluation import eval_db
    ddl = " ".join(" ".join(s.split()) for s in eval_db.SCHEMA)
    assert "CREATE TABLE IF NOT EXISTS retrieval_usage" in ddl
    for col in ("query_log_id TEXT", "user_id TEXT", "usage JSONB NOT NULL", "total_usd REAL NOT NULL",
                "recorded_at TIMESTAMPTZ"):
        assert col in ddl, col
    assert "ON retrieval_usage (recorded_at)" in ddl


# ── GET /costs/retrieval ────────────────────────────────

@check("GET /costs/retrieval: totals, per-service, per-day, average")
async def _():
    db = FakeDB(rows={
        "COUNT(*)": [{"searches": 4, "total_usd": 0.002}],
        "jsonb_each": [
            {"service": "voyage_rerank", "requests": 6, "tokens": 30000, "usd": 0.0015},
            {"service": "voyage_query_embed", "requests": 4, "tokens": 40, "usd": 0.0000048},
        ],
        "GROUP BY day": [{"day": "2026-10-20", "usd": 0.0005}, {"day": "2026-10-21", "usd": 0.0015}],
    })
    costs_retrieval.get_eval_db = lambda: db
    out = await costs_retrieval.retrieval_costs(days=30)
    assert out["available"] is True and out["period_days"] == 30
    assert out["searches"] == 4 and abs(out["total_usd"] - 0.002) < 1e-12
    assert abs(out["per_search_avg_usd"] - 0.0005) < 1e-12
    names = [s["service"] for s in out["by_service"]]
    assert names == list(usage_meter.SEARCH_SERVICES), names       # all four, fixed order
    planner = out["by_service"][names.index("gemini_planner")]
    assert planner == {"service": "gemini_planner", "requests": 0, "tokens": 0, "usd": 0.0}
    assert out["by_day"] == [{"date": "2026-10-20", "usd": 0.0005}, {"date": "2026-10-21", "usd": 0.0015}]
    assert "list price" in out["note"] and out["unit_prices"]


@check("GET /costs/retrieval with the database down: available=false, zeros")
async def _():
    costs_retrieval.get_eval_db = lambda: FakeDB(alive=False)
    out = await costs_retrieval.retrieval_costs(days=7)
    assert out["available"] is False and out["searches"] == 0 and out["total_usd"] == 0.0
    assert out["per_search_avg_usd"] == 0.0 and out["by_day"] == [] and out["period_days"] == 7


@check("GET /costs/retrieval is mounted at /costs/retrieval")
async def _():
    import main
    paths = set(main.app.openapi()["paths"])
    assert "/costs/retrieval" in paths, sorted(paths)


async def main_async():
    retrieval_cost.PERSIST_GRACE_S = 0.0
    saved_search = retrieval.search
    ok = 0
    try:
        for name, fn in CHECKS:
            await fn()
            ok += 1
            print("ok %d %s" % (ok, name))
    finally:
        retrieval.search = saved_search
    print("\n%d retrieval cost checks passed" % ok)


if __name__ == "__main__":
    asyncio.run(main_async())
