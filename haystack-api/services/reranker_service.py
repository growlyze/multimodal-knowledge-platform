"""
Cross-encoder reranking, the last stage of retrieval.

Public surface (do not break):
    RerankerProvider, create_reranker()
"""

import logging
import time
from abc import ABC, abstractmethod

import httpx

import config
from pipelines import usage_meter
from services.embedding_service import post_with_retry

logger = logging.getLogger("haystack-api.reranker")

TIMEOUT = httpx.Timeout(30.0, connect=10.0)
MAX_DOC_CHARS = 8_000


class RerankerProvider(ABC):
    # Whether `score` is a relevance judgement that can be compared against a
    # threshold. True for real cross-encoders; False where the score only encodes
    # placement. pipelines/retrieval.py drops low-relevance results only when this
    # is set — gating on a placement number would delete good results.
    calibrated = True

    @abstractmethod
    async def rerank(self, query: str, documents: list[str],
                     top_n: int = 5) -> list[tuple[int, float]]:
        """Rerank documents for query. Returns [(original_index, score), ...]."""
        ...


class VoyageReranker(RerankerProvider):
    URL = f"{config.VOYAGE_BASE_URL}/rerank"

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or config.RERANKER_API_KEY
        self.model = model or config.RERANKER_MODEL

    async def rerank(self, query, documents, top_n=5):
        if not documents:
            return []
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            data = await post_with_retry(client, self.URL, {
                "query": query,
                "documents": [d[:MAX_DOC_CHARS] for d in documents],
                "model": self.model,
                "top_k": min(top_n, len(documents)),
            }, {"Authorization": f"Bearer {self.api_key}"})
        logger.info("reranked %d docs in %.0fms",
                    len(documents), (time.perf_counter() - started) * 1000)
        logger.debug("rerank raw response: %s", data)
        # P161: Voyage bills rerank by tokens (query x docs + docs) and reports
        # them in usage.total_tokens; recorded only inside a /search scope.
        usage_meter.record_search_rerank((data.get("usage") or {}).get("total_tokens", 0))
        # Atlas/Voyage return {"object","data","model","usage"}; Cohere returns
        # "results". Do not merge these — verified live against ai.mongodb.com.
        return [(d["index"], d["relevance_score"]) for d in data["data"]]


class CohereReranker(RerankerProvider):
    URL = "https://api.cohere.com/v2/rerank"

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or config.RERANKER_API_KEY
        self.model = model or config.RERANKER_MODEL

    async def rerank(self, query, documents, top_n=5):
        if not documents:
            return []
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            data = await post_with_retry(client, self.URL, {
                "query": query,
                "documents": [d[:MAX_DOC_CHARS] for d in documents],
                "model": self.model,
                "top_n": min(top_n, len(documents)),
            }, {"Authorization": f"Bearer {self.api_key}"})
        return [(d["index"], d["relevance_score"]) for d in data["results"]]


class NoopReranker(RerankerProvider):
    """Pass-through when reranking is disabled or no key is configured.
    Keeps RRF order and hands back descending placeholder scores."""

    # The scores below are "position 3 of 20", not "30% relevant". Nothing may
    # threshold them: with no key configured every query would fail the floor.
    calibrated = False

    async def rerank(self, query, documents, top_n=5):
        return [(i, 1.0 - i / max(len(documents), 1)) for i in range(min(top_n, len(documents)))]


PROVIDERS = {
    "voyage": VoyageReranker,
    "cohere": CohereReranker,
    "none": NoopReranker,
}

_reranker = None


def create_reranker() -> RerankerProvider:
    """Factory on config.RERANKER_PROVIDER. Cached — one instance per process.

    Falls back to NoopReranker when no API key is set, so search still returns
    RRF-ranked results instead of failing.
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    name = config.RERANKER_PROVIDER.lower()
    if name != "none" and not config.RERANKER_API_KEY:
        logger.warning("RERANKER_API_KEY not set — reranking disabled, RRF order stands")
        name = "none"

    cls = PROVIDERS.get(name)
    if cls is None:
        logger.warning("unknown RERANKER_PROVIDER '%s' — reranking disabled", name)
        cls = NoopReranker
    _reranker = cls()
    logger.info("reranker: %s", name)
    return _reranker
