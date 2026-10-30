"""
Layer 2 — LLM-as-judge for semantic coherence.

Scores a random sample of chunks 1-5 and asks two yes/no questions the size and
boundary rules in Layer 1 cannot answer: is the chunk understandable on its own,
and is any domain concept cut off?

Off by default (ENABLE_COHERENCE_JUDGE) — it is one API call per sampled chunk,
which is not something to run while ingesting 622 hours of transcripts.

Raw httpx rather than an SDK: `anthropic` is not in requirements.txt (INFRA owns
that file) and the provider has to stay swappable, the same way the embedding
and reranker providers are.

    report = await judge_chunks(chunks, sample_size=20)
"""

import asyncio
import json
import logging
import os
import random
import re

import httpx

logger = logging.getLogger("haystack-api.evaluation")

PROVIDER = os.getenv("COHERENCE_JUDGE_PROVIDER", "anthropic").lower()
MODEL = os.getenv("COHERENCE_JUDGE_MODEL", "claude-opus-5")
API_KEY = os.getenv("COHERENCE_JUDGE_API_KEY", "")
BASE_URL = os.getenv("COHERENCE_JUDGE_BASE_URL", "")
CONCURRENCY = int(os.getenv("COHERENCE_JUDGE_CONCURRENCY", "5"))

# Targets from the RAG plan. Below any of these, the chunking needs tuning.
TARGET_SCORE = 3.5
TARGET_STANDALONE = 0.80
TARGET_CONCEPTS = 0.90

MAX_CHUNK_CHARS = 6000
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

COHERENCE_PROMPT = """Score this text chunk from 1-5 on semantic coherence:

1 = Fragment — mid-sentence start/end, no complete thought
2 = Poor — multiple unrelated topics jumbled together
3 = Acceptable — mostly one topic but with noise
4 = Good — one clear topic, clean boundaries
5 = Excellent — self-contained, one coherent concept

Also answer:
- Could someone understand this chunk WITHOUT reading the surrounding context? (yes/no)
- Does this chunk contain domain-specific concepts? If yes, are any concepts cut off? (yes/no)

Chunk:
\"\"\"
{chunk_content}
\"\"\"

Respond with JSON only: \
{{"score": N, "standalone": true/false, "concepts_complete": true/false, "reasoning": "..."}}"""


class JudgeError(RuntimeError):
    """The judge could not be reached or is not configured."""


# ── providers ───────────────────────────────────────────

async def _ask_anthropic(client: httpx.AsyncClient, prompt: str) -> str:
    resp = await client.post(
        (BASE_URL or "https://api.anthropic.com") + "/v1/messages",
        headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": 1024,
              "messages": [{"role": "user", "content": prompt}]},
    )
    resp.raise_for_status()
    # content is a list of blocks; thinking blocks are present on some models.
    blocks = resp.json().get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


async def _ask_openai(client: httpx.AsyncClient, prompt: str) -> str:
    resp = await client.post(
        (BASE_URL or "https://api.openai.com/v1") + "/chat/completions",
        headers={"Authorization": "Bearer %s" % API_KEY,
                 "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": 1024,
              "messages": [{"role": "user", "content": prompt}]},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


PROVIDERS = {"anthropic": _ask_anthropic, "openai": _ask_openai}


# ── judging ─────────────────────────────────────────────

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str) -> dict:
    """Pull the JSON verdict out of an LLM reply that may be fenced or chatty."""
    match = _JSON.search(text or "")
    if not match:
        raise ValueError("no JSON object in judge reply: %r" % (text or "")[:200])
    data = json.loads(match.group())
    score = float(data.get("score", 0))
    if not 1 <= score <= 5:
        raise ValueError("score %r out of range" % data.get("score"))
    return {
        "score": score,
        "standalone": bool(data.get("standalone")),
        "concepts_complete": bool(data.get("concepts_complete", True)),
        "reasoning": str(data.get("reasoning", ""))[:500],
    }


async def judge_chunks(chunks: list, sample_size: int = 20, ask=None,
                       seed: int = None) -> dict:
    """Score a random sample and compare the averages against the targets.

    `ask` is an async callable (client, prompt) -> str; tests inject a fake one.
    """
    if not chunks:
        return {"sampled": 0, "passed": True, "summary": "no chunks to judge"}

    ask = ask or PROVIDERS.get(PROVIDER)
    if ask is None:
        raise JudgeError("unknown COHERENCE_JUDGE_PROVIDER %r" % PROVIDER)
    if ask in PROVIDERS.values() and not API_KEY:
        raise JudgeError("COHERENCE_JUDGE_API_KEY is not set")

    rng = random.Random(seed)
    sample = rng.sample(chunks, min(sample_size, len(chunks)))
    gate = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async def one(index, chunk):
            content = chunk.content if hasattr(chunk, "content") else chunk.get("content", "")
            async with gate:
                try:
                    reply = await ask(client, COHERENCE_PROMPT.format(
                        chunk_content=content[:MAX_CHUNK_CHARS]))
                    verdict = parse_verdict(reply)
                except Exception as exc:
                    logger.warning("judge failed on sample %d: %s", index, exc)
                    return {"index": index, "error": str(exc)[:200]}
                verdict["index"] = index
                verdict["preview"] = content[:120]
                return verdict

        verdicts = await asyncio.gather(*(one(i, c) for i, c in enumerate(sample)))

    return summarise(verdicts)


def summarise(verdicts: list) -> dict:
    """Aggregate verdicts into the report the ingestion job stores."""
    scored = [v for v in verdicts if "error" not in v]
    if not scored:
        return {"sampled": len(verdicts), "scored": 0, "passed": False,
                "summary": "every judge call failed",
                "failures": [v for v in verdicts if "error" in v][:10]}

    n = len(scored)
    average = sum(v["score"] for v in scored) / n
    standalone = sum(1 for v in scored if v["standalone"]) / n
    complete = sum(1 for v in scored if v["concepts_complete"]) / n

    report = {
        "sampled": len(verdicts),
        "scored": n,
        "average_score": round(average, 2),
        "standalone_rate": round(standalone, 3),
        "concepts_complete_rate": round(complete, 3),
        "targets": {"average_score": TARGET_SCORE, "standalone_rate": TARGET_STANDALONE,
                    "concepts_complete_rate": TARGET_CONCEPTS},
        "passed": (average >= TARGET_SCORE and standalone >= TARGET_STANDALONE
                   and complete >= TARGET_CONCEPTS),
        # Anything at 2 or below is a specific chunk worth looking at by hand.
        "worst": sorted([v for v in scored if v["score"] <= 2.0],
                        key=lambda v: v["score"])[:10],
        "failures": [v for v in verdicts if "error" in v][:10],
    }
    report["summary"] = ("avg %.2f (target %.1f), standalone %.0f%% (target %.0f%%), "
                         "concepts complete %.0f%% (target %.0f%%)"
                         % (average, TARGET_SCORE, standalone * 100, TARGET_STANDALONE * 100,
                            complete * 100, TARGET_CONCEPTS * 100))
    log = logger.info if report["passed"] else logger.warning
    log("coherence judge: %s", report["summary"])
    return report


if __name__ == "__main__":
    from pipelines.chunkers.base import TextChunk

    async def fake_ask(_client, prompt):
        good = "batch job" in prompt and "asynchronous" in prompt
        return ('```json\n{"score": %d, "standalone": %s, "concepts_complete": %s,'
                ' "reasoning": "demo"}\n```'
                % (5 if good else 1, "true" if good else "false",
                   "true" if good else "false"))

    chunks = [TextChunk("A batch job is an asynchronous task queued for later.", {})] * 9
    report = asyncio.run(judge_chunks(chunks, sample_size=9, ask=fake_ask, seed=1))
    assert report["passed"] and report["average_score"] == 5.0, report

    bad = [TextChunk("gap and then the", {})] * 5
    report = asyncio.run(judge_chunks(bad, sample_size=5, ask=fake_ask, seed=1))
    assert not report["passed"] and report["worst"], report
    print("coherence judge self-check ok")
