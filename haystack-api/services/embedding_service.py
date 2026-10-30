"""
Embedding providers. INGESTION and STRUCTURED sessions import from here.

Public surface (do not break):
    EmbeddingProvider           — abstract base
    create_embedding_provider() — factory, cached per process
    post_with_retry()           — shared HTTP retry helper (reranker_service uses it too)
"""

import asyncio
import hashlib
import logging
import math
import struct
import time
from abc import ABC, abstractmethod

import httpx

import config
from pipelines import usage_meter

logger = logging.getLogger("haystack-api.embedding")

# voyage-4 takes 32K tokens; ~4 chars/token with slack. Truncating beats a 400 from the API.
MAX_CHARS = 100_000
BATCH_SIZE = 128
TIMEOUT = httpx.Timeout(60.0, connect=10.0)


class EmbeddingError(RuntimeError):
    """Provider failed. Callers surface this — never swallow it into empty results."""


async def post_with_retry(client: httpx.AsyncClient, url: str, payload: dict,
                          headers: dict, attempts: int = 4) -> dict:
    """POST with exponential backoff on 429/5xx. Shared by embedding + reranker."""
    for attempt in range(attempts):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                delay = 2 ** attempt
                logger.warning("%s -> %s, retrying in %ss", url, resp.status_code, delay)
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"{url} returned {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == attempts - 1:
                raise EmbeddingError(f"{url} unreachable: {exc}") from exc
            await asyncio.sleep(2 ** attempt)
    raise EmbeddingError(f"{url} failed after {attempts} attempts")


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        """Embed texts, return vectors. Batches internally.

        input_type is "document" at ingest time and "query" at search time —
        providers that support asymmetric embedding retrieve better with it.
        Callers that do not care can omit it.
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensions, for collection creation."""
        ...


class VoyageProvider(EmbeddingProvider):
    """Default. Uses config.EMBEDDING_API_KEY, config.EMBEDDING_MODEL."""

    URL = f"{config.VOYAGE_BASE_URL}/embeddings"

    def __init__(self, api_key=None, model=None, dimensions=None):
        self.api_key = api_key or config.EMBEDDING_API_KEY
        self.model = model or config.EMBEDDING_MODEL
        self._dimensions = dimensions or config.EMBEDDING_DIMENSIONS
        if not self.api_key:
            raise EmbeddingError("EMBEDDING_API_KEY is not set")

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        if not texts:
            return []
        texts = [t[:MAX_CHARS] for t in texts]
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        vectors: list[list[float]] = []

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                data = await post_with_retry(client, self.URL, {
                    "input": batch,
                    "model": self.model,
                    "input_type": input_type,
                    "output_dimension": self._dimensions,
                }, headers)
                # P161: per-question retrieval cost. A no-op outside a /search
                # scope, so document/CSV ingestion embeds (which also come
                # through here) are not counted as chat retrieval.
                usage_meter.record_search_voyage_text(
                    "voyage_query_embed", (data.get("usage") or {}).get("total_tokens", 0))
                # Response order is not guaranteed — sort by the echoed index.
                vectors.extend(
                    d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])
                )

        logger.info("embedded %d texts (%s) in %.0fms",
                    len(texts), input_type, (time.perf_counter() - started) * 1000)
        if len(vectors) != len(texts):
            raise EmbeddingError(f"expected {len(texts)} vectors, got {len(vectors)}")
        return vectors


class StubProvider(EmbeddingProvider):
    """No API key needed. Deterministic hash-based vectors so the whole pipeline
    runs end to end before Voyage is provisioned.

    ponytail: identical text matches itself and nothing else — enough to exercise
    upsert/search/RRF/rerank wiring, useless for real semantic recall. Swap by
    setting EMBEDDING_PROVIDER=voyage plus a real EMBEDDING_API_KEY.
    """

    def __init__(self, dimensions=None):
        self._dimensions = dimensions or config.EMBEDDING_DIMENSIONS

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        return [self._hash_vector(t) for t in texts]

    def _hash_vector(self, text: str) -> list[float]:
        raw = b""
        seed = text.encode("utf-8", "ignore")
        while len(raw) < self._dimensions * 4:
            seed = hashlib.sha256(seed).digest()
            raw += seed
        vals = struct.unpack(f">{self._dimensions}f", raw[:self._dimensions * 4])
        # Raw float bytes can decode to nan/inf — clamp, then normalise for cosine.
        vals = [v if math.isfinite(v) else 0.0 for v in vals]
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]


class _NotImplementedProvider(EmbeddingProvider):
    NAME = ""

    def __init__(self, *_a, **_kw):
        raise NotImplementedError(f"{self.NAME} embedding provider not implemented yet")

    async def embed(self, texts, input_type="document"):
        raise NotImplementedError

    @property
    def dimensions(self):
        raise NotImplementedError


class OpenAIProvider(_NotImplementedProvider):
    NAME = "OpenAI"


class JinaProvider(_NotImplementedProvider):
    NAME = "Jina"


class CohereProvider(_NotImplementedProvider):
    NAME = "Cohere"


PROVIDERS = {
    "voyage": VoyageProvider,
    "openai": OpenAIProvider,
    "jina": JinaProvider,
    "cohere": CohereProvider,
    "stub": StubProvider,
}

_provider = None


def create_embedding_provider() -> EmbeddingProvider:
    """Factory on config.EMBEDDING_PROVIDER. Cached — one instance per process.

    Falls back to StubProvider when no API key is configured, so the service
    boots and /search works before Voyage is provisioned.
    """
    global _provider
    if _provider is not None:
        return _provider

    name = config.EMBEDDING_PROVIDER.lower()
    if name != "stub" and not config.EMBEDDING_API_KEY:
        logger.warning(
            "EMBEDDING_API_KEY not set — falling back to StubProvider. "
            "Retrieval results are meaningless until a real key is configured."
        )
        name = "stub"

    cls = PROVIDERS.get(name)
    if cls is None:
        raise EmbeddingError(
            f"unknown EMBEDDING_PROVIDER '{name}', expected one of {sorted(PROVIDERS)}"
        )
    _provider = cls()
    logger.info("embedding provider: %s (%d dims)", name, _provider.dimensions)
    return _provider
