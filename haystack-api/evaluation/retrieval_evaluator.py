"""
Layer 3 — retrieval quality evaluation.

Runs every query in test_queries.json through GET /search and scores the
results. This is the metric that actually tells you whether ingestion worked:
Layer 1 and 2 can both be green while the right chunk is still unretrievable.

    python -m evaluation.retrieval_evaluator --url http://localhost:8000
    python -m evaluation.retrieval_evaluator --out report.json

Metrics, and what a bad number points at:

    precision@5   how much of the top 5 is on topic      -> reranker / chunking
    recall@10     expected keywords found in the top 10  -> chunk size, coverage
    MRR           rank of the first relevant result      -> fusion weighting
    jargon hits   do domain-abbreviation queries return anything -> BM25 sparse vectors
    cross-coll    do multi-collection queries reach both -> collection routing
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import httpx

logger = logging.getLogger("haystack-api.evaluation")

API_URL = os.getenv("HAYSTACK_API_URL", "http://localhost:8000")
QUERIES_PATH = Path(__file__).with_name("test_queries.json")
TIMEOUT = httpx.Timeout(30.0, connect=5.0)

DEFAULT_TARGETS = {"precision_at_5": 0.8, "recall_at_10": 0.85,
                   "mrr": 0.7, "jargon_hit_rate": 0.9}


def load_queries(path=None) -> dict:
    with open(path or QUERIES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower())


def keywords_found(result: dict, keywords: list) -> set:
    """Which expected keywords appear in this result's content."""
    haystack = _norm(result.get("content", ""))
    return {k for k in keywords if _norm(k).strip() and _norm(k).strip() in haystack}


def is_relevant(result: dict, case: dict) -> bool:
    """A result counts as relevant when it satisfies every stated expectation.

    ponytail: keyword containment stands in for a human relevance label. It is
    strict enough to catch a retriever that returns the wrong topic, and it
    costs nothing to re-run. Upgrade path: swap this one function for graded
    human labels or an LLM judge — nothing else in the file has to change.
    """
    wanted_type = case.get("expected_source_type")
    if wanted_type and result.get("source_type") != wanted_type:
        return False

    collections = case.get("expected_collections")
    if collections and _collection_of(result) not in collections:
        return False

    if case.get("expected_has_timestamp") and \
            (result.get("metadata") or {}).get("start_time") is None:
        return False

    keywords = case.get("expected_keywords_in_result") or []
    if keywords:
        return bool(keywords_found(result, keywords))
    return True


def _collection_of(result: dict):
    return (result.get("metadata") or {}).get("collection")


def score_case(case: dict, results: list) -> dict:
    """Per-query metrics over one ranked result list."""
    top5, top10 = results[:5], results[:10]
    relevant5 = [r for r in top5 if is_relevant(r, case)]
    relevant10 = [r for r in top10 if is_relevant(r, case)]

    precision5 = len(relevant5) / 5.0 if top5 else 0.0

    keywords = case.get("expected_keywords_in_result") or []
    if keywords:
        covered = set()
        for result in top10:
            covered |= keywords_found(result, keywords)
        recall10 = len(covered) / len(keywords)
    else:
        recall10 = 1.0 if relevant10 else 0.0

    rank = next((i + 1 for i, r in enumerate(results) if is_relevant(r, case)), 0)

    score = {
        "query": case["query"],
        "difficulty": case.get("difficulty", "unspecified"),
        "results": len(results),
        "precision_at_5": round(precision5, 3),
        "recall_at_10": round(recall10, 3),
        "reciprocal_rank": round(1.0 / rank, 3) if rank else 0.0,
        "first_relevant_rank": rank or None,
    }

    collections = case.get("expected_collections")
    if collections:
        seen = {_collection_of(r) for r in top10}
        score["collections_hit"] = sorted(c for c in seen if c)
        score["cross_collection_ok"] = all(c in seen for c in collections)
    if not rank:
        score["failure"] = "no relevant result in %d returned" % len(results)
    return score


async def run_query(client: httpx.AsyncClient, case: dict, top_k: int = 10) -> list:
    """GET /search for one case. A failed call scores zero, it does not crash the run."""
    params = {"q": case["query"], "top_k": top_k}
    collections = case.get("expected_collections")
    if collections:
        params["collections"] = ",".join(collections)
    try:
        resp = await client.get("/search", params=params)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as exc:
        logger.warning("search failed for %r: %s", case["query"], exc)
        return []


async def evaluate(queries: dict = None, client: httpx.AsyncClient = None,
                   url: str = None, top_k: int = 10) -> dict:
    """Run the whole test set. Pass `client` to point at a fake transport."""
    queries = queries or load_queries()
    cases = queries["test_queries"]
    targets = {**DEFAULT_TARGETS, **(queries.get("targets") or {})}

    owned = client is None
    client = client or httpx.AsyncClient(base_url=url or API_URL, timeout=TIMEOUT)
    try:
        # Sequential on purpose: every query hits the embedding and rerank APIs,
        # and a 30-query burst is the fastest way to get rate limited.
        scores = [score_case(case, await run_query(client, case, top_k)) for case in cases]
    finally:
        if owned:
            await client.aclose()

    return build_report(scores, targets)


def build_report(scores: list, targets: dict = None) -> dict:
    targets = {**DEFAULT_TARGETS, **(targets or {})}
    n = len(scores) or 1

    def mean(key):
        return round(sum(s[key] for s in scores) / n, 3)

    jargon = [s for s in scores if s["difficulty"] == "jargon-abbreviation"]
    cross = [s for s in scores if "cross_collection_ok" in s]

    metrics = {
        "precision_at_5": mean("precision_at_5"),
        "recall_at_10": mean("recall_at_10"),
        "mrr": mean("reciprocal_rank"),
        "jargon_hit_rate": (round(sum(1 for s in jargon if s["first_relevant_rank"])
                                  / len(jargon), 3) if jargon else None),
        "cross_collection_rate": (round(sum(1 for s in cross if s["cross_collection_ok"])
                                        / len(cross), 3) if cross else None),
    }
    failures = [s for s in scores if s.get("failure")]

    report = {
        "queries": len(scores),
        "metrics": metrics,
        "targets": targets,
        "below_target": {k: metrics[k] for k, floor in targets.items()
                         if metrics.get(k) is not None and metrics[k] < floor},
        "by_difficulty": _by_difficulty(scores),
        "failures": failures,
        "scores": scores,
    }
    report["passed"] = not report["below_target"]
    log = logger.info if report["passed"] else logger.warning
    log("retrieval evaluation: p@5=%s recall@10=%s mrr=%s jargon=%s (%d failures)",
        metrics["precision_at_5"], metrics["recall_at_10"], metrics["mrr"],
        metrics["jargon_hit_rate"], len(failures))
    return report


def _by_difficulty(scores: list) -> dict:
    groups = {}
    for score in scores:
        groups.setdefault(score["difficulty"], []).append(score)
    return {
        name: {
            "queries": len(group),
            "precision_at_5": round(sum(s["precision_at_5"] for s in group) / len(group), 3),
            "recall_at_10": round(sum(s["recall_at_10"] for s in group) / len(group), 3),
            "mrr": round(sum(s["reciprocal_rank"] for s in group) / len(group), 3),
        }
        for name, group in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser(description="Layer 3 retrieval evaluation")
    parser.add_argument("--url", default=API_URL, help="haystack-api base URL")
    parser.add_argument("--queries", default=None, help="path to a test_queries.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default=None, help="write the full report as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    report = asyncio.run(evaluate(load_queries(args.queries), url=args.url, top_k=args.top_k))

    print(json.dumps({k: report[k] for k in
                      ("queries", "metrics", "targets", "below_target",
                       "by_difficulty", "passed")}, indent=2))
    for failure in report["failures"]:
        print("FAIL  %-60s %s" % (failure["query"][:60], failure["failure"]))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("full report -> %s" % args.out)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
