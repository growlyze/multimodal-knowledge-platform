"""
LLM-as-judge for answer quality — the layer above retrieval.

Retrieval metrics say the right chunk came back. This says the model actually
used it: every claim in the answer is checked against the retrieved context, and
anything unsupported is a hallucination.

    evaluator = AnswerEvaluator()
    scores = await evaluator.sample_and_evaluate(query, answer, chunks,
                                                 query_log_id=log_id)   # None if not sampled

Sampled, not universal — EVAL_SAMPLE_RATE of queries, because this is one
extra LLM call per evaluation. Default 0.0 = off (2026-10-21 decision). Raw httpx and two providers for the same reason
coherence_judge uses them: `anthropic` is not in requirements.txt (INFRA owns
that file) and the judge has to stay swappable.

Never raises. A judge that is down, throttled or babbling returns a dict with
`error` set and the scores None; the chat pipeline neither knows nor cares.
"""

import json
import logging
import os
import random
import re

import httpx

from evaluation.eval_db import EvalDB, cutoff, get_eval_db, new_id

logger = logging.getLogger("haystack-api.evaluation")

PROVIDER = os.getenv("EVAL_JUDGE_PROVIDER", "anthropic").lower()
MODEL = os.getenv("EVAL_JUDGE_MODEL", "claude-haiku-4-5-20251001")
API_KEY = os.getenv("EVAL_JUDGE_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
BASE_URL = os.getenv("EVAL_JUDGE_BASE_URL", "")
# OFF by default (decision 2026-10-21, P161 item 5): the answer-quality judge is
# one extra paid LLM call per sampled chat turn and the user decided it stays
# off. Was 0.05 with ANTHROPIC_API_KEY set live, i.e. armed to buy a Haiku call
# on 5% of turns (eval_answer_scores still had 0 rows on 2026-10-21, so no spend
# was observed). 0.0 means never (should_sample short-circuits);
# set EVAL_SAMPLE_RATE explicitly to turn it back on.
SAMPLE_RATE = float(os.getenv("EVAL_SAMPLE_RATE", "0.0"))

MAX_CHUNKS = 12
MAX_CHUNK_CHARS = 2000
MAX_ANSWER_CHARS = 8000
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

JUDGE_PROMPT = """You are grading a RAG answer against the context it was given.

Break the ANSWER into its distinct factual claims. For each claim decide whether
the CONTEXT supports it. A claim is unsupported if the context does not state it,
even when the claim is true in general — that is a hallucination here.

Then rate:
- answer_relevancy 0-1: does the answer address the QUESTION that was asked?
- citation_accuracy 0-1: of any sources the answer cites, how many are actually
  in the context and actually say what the answer attributes to them?
  Use 1.0 when the answer cites nothing.

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
{answer}

Respond with JSON only:
{{"claims": [{{"text": "...", "supported": true}}],
  "answer_relevancy": 0.0-1.0,
  "citation_accuracy": 0.0-1.0,
  "reasoning": "one or two sentences"}}"""

_JSON = re.compile(r"\{.*\}", re.DOTALL)


# ── providers ───────────────────────────────────────────

async def _ask_anthropic(client: httpx.AsyncClient, prompt: str) -> str:
    resp = await client.post(
        (BASE_URL or "https://api.anthropic.com") + "/v1/messages",
        headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": 2048,
              "messages": [{"role": "user", "content": prompt}]},
    )
    resp.raise_for_status()
    blocks = resp.json().get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


async def _ask_openai(client: httpx.AsyncClient, prompt: str) -> str:
    resp = await client.post(
        (BASE_URL or "https://api.openai.com/v1") + "/chat/completions",
        headers={"Authorization": "Bearer %s" % API_KEY,
                 "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": 2048,
              "messages": [{"role": "user", "content": prompt}]},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


PROVIDERS = {"anthropic": _ask_anthropic, "openai": _ask_openai}


# ── scoring ─────────────────────────────────────────────

def build_context(chunks: list) -> str:
    """Retrieved chunks -> numbered context block. Accepts strings or search hits."""
    lines = []
    for i, chunk in enumerate(chunks[:MAX_CHUNKS], 1):
        text = chunk if isinstance(chunk, str) else (
            chunk.get("content") or chunk.get("text") or "")
        lines.append("[%d] %s" % (i, text[:MAX_CHUNK_CHARS]))
    return "\n\n".join(lines)


def parse_verdict(text: str) -> dict:
    """Pull the JSON verdict out of a reply that may be fenced or chatty."""
    match = _JSON.search(text or "")
    if not match:
        raise ValueError("no JSON object in judge reply: %r" % (text or "")[:200])
    data = json.loads(match.group())

    claims = [{"text": str(c.get("text", ""))[:300], "supported": bool(c.get("supported"))}
              for c in (data.get("claims") or []) if isinstance(c, dict)]
    total = len(claims)
    supported = sum(1 for c in claims if c["supported"])

    def unit(key, default=None):
        value = data.get(key, default)
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    return {
        # Computed from the claims rather than asked for: the judge counts
        # inconsistently, the division does not.
        "faithfulness": round(supported / total, 3) if total else unit("faithfulness"),
        "answer_relevancy": unit("answer_relevancy"),
        "hallucination_count": total - supported,
        "total_claims": total,
        "claims": claims,
        "citation_accuracy": unit("citation_accuracy", 1.0),
        "judge_reasoning": str(data.get("reasoning", ""))[:1000],
        "judge_model": MODEL,
    }


def failed(reason: str) -> dict:
    """The shape callers get when the judge could not score anything."""
    logger.warning("answer evaluation failed: %s", reason)
    return {"faithfulness": None, "answer_relevancy": None, "hallucination_count": None,
            "total_claims": None, "claims": [], "citation_accuracy": None,
            "judge_reasoning": "", "judge_model": MODEL, "error": reason[:300]}


def should_sample(rate: float = None) -> bool:
    """True for the fraction of queries worth spending a judge call on.
    A rate of 0 (the default) or below is off outright, not a coin flip."""
    rate = SAMPLE_RATE if rate is None else rate
    if rate <= 0.0:
        return False
    return random.random() < rate


class AnswerEvaluator:
    def __init__(self, db: EvalDB = None, ask=None):
        self.db = db or get_eval_db()
        self.ask = ask              # async (client, prompt) -> str; tests inject a fake

    async def evaluate(self, query: str, answer: str, retrieved_chunks: list,
                       model: str = None, query_log_id: str = None) -> dict:
        """Score an answer for faithfulness and relevancy. Stores when query_log_id is given."""
        ask = self.ask or PROVIDERS.get(PROVIDER)
        if ask is None:
            return failed("unknown EVAL_JUDGE_PROVIDER %r" % PROVIDER)
        if self.ask is None and not API_KEY:
            return failed("EVAL_JUDGE_API_KEY is not set")
        if not (answer or "").strip():
            return failed("empty answer")

        prompt = JUDGE_PROMPT.format(query=query, context=build_context(retrieved_chunks or []),
                                     answer=(answer or "")[:MAX_ANSWER_CHARS])
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                scores = parse_verdict(await ask(client, prompt))
        except Exception as exc:
            return failed("%s: %s" % (type(exc).__name__, exc))

        scores["judge_model"] = model or MODEL
        if query_log_id is not None:
            await self.store(query_log_id, scores)
        return scores

    async def sample_and_evaluate(self, query: str, answer: str, retrieved_chunks: list,
                                  query_log_id: str = None, rate: float = None):
        """evaluate() on EVAL_SAMPLE_RATE of calls, None on the rest.

        This is the one the filter should call — otherwise every caller
        reimplements the sampling coin flip.
        """
        if not should_sample(rate):
            return None
        return await self.evaluate(query, answer, retrieved_chunks,
                                   query_log_id=query_log_id)

    async def store(self, query_log_id: str, scores: dict) -> str:
        score_id = new_id()
        await self.db.execute(
            """
            INSERT INTO eval_answer_scores
                (id, query_log_id, faithfulness, answer_relevancy, hallucination_count,
                 total_claims, citation_accuracy, judge_model, judge_reasoning)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            score_id, query_log_id, scores.get("faithfulness"),
            scores.get("answer_relevancy"), scores.get("hallucination_count"),
            scores.get("total_claims"), scores.get("citation_accuracy"),
            scores.get("judge_model"), scores.get("judge_reasoning"),
        )
        return score_id

    async def get_recent(self, limit: int = 20) -> list:
        return await self.db.fetch(
            """
            SELECT s.*, q.query FROM eval_answer_scores s
            LEFT JOIN eval_query_log q ON q.id = s.query_log_id
            ORDER BY s.timestamp DESC LIMIT $1
            """,
            max(1, int(limit)),
        )

    async def get_summary(self, hours: float = 24 * 7) -> dict:
        """Dashboard aggregates. A week by default — 5% of traffic is a thin sample."""
        row = await self.db.fetchrow(
            """
            SELECT COUNT(*) AS samples,
                   AVG(faithfulness) AS faithfulness_avg,
                   AVG(answer_relevancy) AS relevancy_avg,
                   AVG(citation_accuracy) AS citation_avg,
                   SUM(hallucination_count) AS hallucinations,
                   SUM(total_claims) AS claims
            FROM eval_answer_scores WHERE timestamp >= $1
            """,
            cutoff(hours),
        ) or {}

        def avg(key):
            return round(float(row[key]), 3) if row.get(key) is not None else None

        claims = int(row.get("claims") or 0)
        return {
            "samples_scored": int(row.get("samples") or 0),
            "faithfulness_avg": avg("faithfulness_avg"),
            "relevancy_avg": avg("relevancy_avg"),
            "citation_accuracy_avg": avg("citation_avg"),
            "hallucination_rate": (round(int(row.get("hallucinations") or 0) / claims, 4)
                                   if claims else 0.0),
        }


if __name__ == "__main__":
    import asyncio

    faithful = ('```json\n{"claims": [{"text": "a batch job runs asynchronously", "supported": true},' 
                '{"text": "it is queued for later", "supported": true}],'
                '"answer_relevancy": 0.9, "citation_accuracy": 1.0, "reasoning": "ok"}\n```')
    hallucinated = ('{"claims": [{"text": "a batch job runs synchronously", "supported": true},' 
                    '{"text": "the docs say to run it on Tuesdays", "supported": false},' 
                    '{"text": "the queue always empties", "supported": false}],'
                    '"answer_relevancy": 0.4, "citation_accuracy": 0.0, "reasoning": "no"}')

    async def main():
        good = await AnswerEvaluator(ask=lambda c, p: _reply(faithful)).evaluate(
            "what is a webhook", "an answer", ["a chunk"])
        assert good["faithfulness"] == 1.0 and good["hallucination_count"] == 0, good

        bad = await AnswerEvaluator(ask=lambda c, p: _reply(hallucinated)).evaluate(
            "what is a webhook", "an answer", ["a chunk"])
        assert bad["faithfulness"] == round(1 / 3, 3) and bad["hallucination_count"] == 2, bad

        async def boom(_client, _prompt):
            raise httpx.ConnectError("judge is down")

        broken = await AnswerEvaluator(ask=boom).evaluate("q", "a", ["c"])
        assert broken["faithfulness"] is None and "judge is down" in broken["error"], broken
        print("answer evaluator self-check ok")

    async def _reply(text):
        return text

    asyncio.run(main())
