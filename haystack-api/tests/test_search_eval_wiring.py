"""
Where evaluation plugs into the request path.

Query logging lives in the /search route, not in the Open WebUI filter: the filter
runs in another container and cannot import `evaluation`. Answer scoring is the
mirror image — the answer only exists in the filter, so it comes back over HTTP to
POST /eval/log-answer.

The case that matters most here is evaluation being *down*. Search must behave
identically, and a dead eval database must not hand out a log_id that would buy an
LLM judge call with no query row to attach it to.

    python tests/test_search_eval_wiring.py

No PostgreSQL, no Qdrant, no judge: retrieval is stubbed and the eval database is
deliberately unreachable.
"""
import asyncio
import os

os.environ.setdefault("EVAL_DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")

from evaluation import eval_db
from pipelines import retrieval
from routers import eval as eval_router
from routers import search as search_router

HITS = [
    {"score": 0.91, "collection": "documents", "id": "c1",
     "payload": {"content": "an SLA is a three-tier uptime commitment", "source": "guide.pdf",
                 "source_type": "document"}},
    {"score": 0.60, "collection": "documents", "id": "c2",
     "payload": {"content": "he draws the gap here", "source": "s47.mp4",
                 "source_type": "transcript"}},
]


async def main():
    calls = {}

    # --- 1. eval DB unreachable: search still answers, logging degrades to None ---
    async def fake_search(q, **kw):
        calls["query"] = q
        return HITS

    retrieval.search = fake_search
    resp = await search_router.search_knowledge(
        q="what is a service level agreement", top_k=5, collections=None, source_type=None,
        user_id="u7", had_code_execution=True, had_file_attachment=False)

    assert calls["query"] == "what is a service level agreement"
    assert len(resp.results) == 2, resp
    assert resp.results[0].content.startswith("an SLA")
    assert resp.log_id is None, "a dead eval DB must yield no log id, not an error"
    print("ok  search returns results with the eval database down")

    # --- 2. the same path with a working eval DB records the query ---
    logged, empties = [], []

    class FakeDB:
        async def pool(self):
            return "a live pool"

        async def execute(self, sql, *args):
            if "eval_query_log" in sql:
                logged.append(args)
            elif "eval_coverage_gaps" in sql:
                empties.append(args)
            return True

    # Each module did `from evaluation.eval_db import get_eval_db`, so each holds its
    # own reference — patch all three, not just the source module.
    fake = FakeDB()
    from evaluation import coverage_analyzer, query_logger
    for module in (eval_db, query_logger, coverage_analyzer, search_router):
        module.get_eval_db = lambda: fake

    resp = await search_router.search_knowledge(
        q="what is a service level agreement", top_k=5, collections=None, source_type=None,
        user_id="u7", had_code_execution=True, had_file_attachment=False)

    assert resp.log_id, "a healthy eval DB must return a log id"
    assert len(logged) == 1, logged
    row = logged[0]
    assert row[2] == "what is a service level agreement", row      # query
    assert row[3] == 2, row                               # results_count
    assert abs(row[4] - 0.91) < 1e-6, row                 # top_score = max, not first
    assert row[6] >= 0, row                               # latency_ms recorded
    assert row[7] is True, row                            # had_code_execution
    assert row[9] is False, row                           # empty_result
    assert not empties, "a non-empty search is not a coverage gap"
    print("ok  query logged with score, latency and context flags")

    # --- 3. an empty search feeds the coverage analyzer ---
    async def no_hits(q, **kw):
        return []

    retrieval.search = no_hits
    resp = await search_router.search_knowledge(
        q="what is the canary release", top_k=5, collections=None, source_type=None,
        user_id=None, had_code_execution=False, had_file_attachment=False)

    assert resp.results == []
    assert logged[-1][9] is True, "empty_result must be set"
    assert logged[-1][4] is None, "no results means no top score"
    assert empties == [("canary release",)], empties
    print("ok  empty search recorded as a coverage gap")

    # --- 3b. an out-of-scope miss is NOT a coverage gap ---
    #
    # The gap list drives what gets ingested next. A question the corpus was never
    # meant to answer ("who was Einstein?") returns zero hits for a reason that no
    # amount of ingestion fixes, and filing it alongside "what is the canary release"
    # turns the dashboard's top_gaps panel into a log of off-topic questions.
    # rag_filter decides this — it is the only side that knows whether the query
    # was about the knowledge base — and sends it as in_scope.
    before = len(empties)
    resp = await search_router.search_knowledge(
        q="who was Einstein", top_k=5, collections=None, source_type=None,
        user_id=None, had_code_execution=False, had_file_attachment=False,
        in_scope=False)

    assert resp.results == []
    assert len(empties) == before, (
        "an out-of-scope miss was recorded as a coverage gap: %r" % (empties[before:],)
    )
    assert logged[-1][9] is True, "it is still an empty result, and still logged"
    print("ok  an out-of-scope miss is logged but is not a coverage gap")
    # --- 4. the answer endpoint: sampled path schedules the judge, and only once ---
    class Tasks:
        def __init__(self):
            self.scheduled = []

        def add_task(self, fn, payload):
            self.scheduled.append((fn, payload))

    payload = eval_router.AnswerLog(
        query="what is a service level agreement", answer="An SLA is a three-tier uptime commitment.",
        log_id="log-1", chunks=["an SLA is a three-tier uptime commitment"])

    eval_router.should_sample = lambda *a, **k: True
    tasks = Tasks()
    out = await eval_router.log_answer(payload, tasks)
    assert out == {"sampled": True, "log_id": "log-1"}, out
    assert len(tasks.scheduled) == 1
    print("ok  sampled answer schedules the judge in the background")

    eval_router.should_sample = lambda *a, **k: False
    tasks = Tasks()
    out = await eval_router.log_answer(payload, tasks)
    assert out == {"sampled": False}, out
    assert not tasks.scheduled, "an unsampled answer must not cost an LLM call"
    print("ok  unsampled answer costs nothing")

    eval_router.should_sample = lambda *a, **k: True
    blank = eval_router.AnswerLog(query="q", answer="   ", log_id="log-2")
    tasks = Tasks()
    out = await eval_router.log_answer(blank, tasks)
    assert out["sampled"] is False and not tasks.scheduled, out
    print("ok  empty answer is never judged")

    # --- 5. the judge failing must not escape the background task ---
    class Boom:
        async def evaluate(self, *a, **kw):
            raise RuntimeError("judge exploded")

    eval_router.AnswerEvaluator = lambda *a, **kw: Boom()
    await eval_router._judge(payload)          # must not raise
    print("ok  a broken judge stays contained")

    print("\n7 wiring checks passed")


if __name__ == "__main__":
    asyncio.run(main())
