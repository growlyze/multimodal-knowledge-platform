"""Dual embedding — Voyage multimodal for frames, Voyage text for the other tracks.

D66: visual frames used to go through local CLIP (ViT-B/32, 512d, its own
embedding space). That required `open-clip-torch` — a heavy torch-based
dependency the M3 install list already struggles with — and kept
visual_frames out of step with the other three collections, which all use
Voyage text embeddings (1024d). Voyage's multimodal endpoint embeds images
and text into that same shared space, so this file now has one embedding
provider end to end, no local model to load.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import config
from pipelines import usage_meter

logger = logging.getLogger("haystack-api.video_embedder")


def _voyage_client():
    """Every Voyage client in this file is built explicitly, never bare
    voyageai.Client() — the project's Voyage key is issued via MongoDB Atlas
    (rag-AI/CLAUDE.md: "Atlas keys 403 on api.voyageai.com"), so relying on
    the SDK's own default base_url silently 403s in production."""
    import voyageai
    return voyageai.Client(api_key=config.EMBEDDING_API_KEY, base_url=config.VOYAGE_BASE_URL)


@dataclass
class QueryEmbeddings:
    """Both embedding types needed for a 4-track video search."""
    visual_embedding: List[float]   # Voyage multimodal — visual_frames collection
    voyage_embedding: List[float]   # 1024d — ocr/od/transcript collections


class VoyageRateLimiter:
    """Simple token-bucket rate limiter for Voyage API calls."""

    def __init__(self, rpm: int = 200):
        self._interval = 60.0 / rpm
        self._last_call = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()


_voyage_limiter = VoyageRateLimiter()


def embed_texts_voyage(
    texts: List[str],
    batch_size: int = 96,
    model: str = None,
    on_batch=None,
) -> List[List[float]]:
    """Embed text strings using Voyage API. Returns list of 1024d vectors.

    `on_batch()`, when given, runs before each batch's paid request: the
    ingestion orchestrator counts batches with it (embed progress) and uses
    it as a cancel checkpoint -- whatever it raises stops the next request.
    """
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "voyage-4-large")

    client = _voyage_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if on_batch is not None:
            on_batch()
        _voyage_limiter.wait()
        result = client.embed(batch, model=model, input_type="document")
        usage_meter.record_voyage_text(getattr(result, "total_tokens", 0))
        all_embeddings.extend(result.embeddings)
        logger.debug("embedded batch %d/%d (%d texts)",
                     i // batch_size + 1,
                     (len(texts) + batch_size - 1) // batch_size,
                     len(batch))

    logger.info("embedded %d texts via Voyage (%dd)", len(texts), len(all_embeddings[0]) if all_embeddings else 0)
    return all_embeddings


def embed_query_voyage(text: str, model: str = None) -> List[float]:
    """Embed a single query string using Voyage. Returns 1024d vector."""
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "voyage-4-large")

    client = _voyage_client()
    _voyage_limiter.wait()
    result = client.embed([text], model=model, input_type="query")
    # Chat retrieval cost (P161); a no-op outside a /search scope. Never the
    # ingestion meter: a query embed must not land on a video's bill (check 10).
    usage_meter.record_search_voyage_text("voyage_video_query", getattr(result, "total_tokens", 0))
    return result.embeddings[0]


def embed_frames_visual(
    frames: List[np.ndarray],
    batch_size: int = 96,
    model: str = None,
    on_batch=None,
) -> List[List[float]]:
    """Embed BGR keyframes via Voyage multimodal. Returns one vector per frame.

    `on_batch()` -- same contract as embed_texts_voyage's.

    No local model/torch dependency — frames go straight to the same Voyage
    provider as every other M3 track.
    """
    from PIL import Image

    if model is None:
        model = os.getenv("VOYAGE_MULTIMODAL_MODEL", config.VOYAGE_MULTIMODAL_MODEL)

    client = _voyage_client()
    all_embeddings = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        if on_batch is not None:
            on_batch()
        # Each inner list is one embeddable item; a single-image list embeds
        # just that image, same shape as embed_texts_voyage's per-text batching.
        pil_batch = [[Image.fromarray(frame_bgr[:, :, ::-1])] for frame_bgr in batch]
        _voyage_limiter.wait()
        result = client.multimodal_embed(pil_batch, model=model, input_type="document")
        usage_meter.record_voyage_multimodal(getattr(result, "text_tokens", 0),
                                             getattr(result, "image_pixels", 0), len(batch))
        all_embeddings.extend(result.embeddings)

    logger.info("Voyage-multimodal-embedded %d frames (%dd)",
                len(frames), len(all_embeddings[0]) if all_embeddings else 0)
    return all_embeddings


def embed_query_visual(text: str, model: str = None) -> List[float]:
    """Embed a text query into the same multimodal space as visual frames."""
    if model is None:
        model = os.getenv("VOYAGE_MULTIMODAL_MODEL", config.VOYAGE_MULTIMODAL_MODEL)

    client = _voyage_client()
    _voyage_limiter.wait()
    result = client.multimodal_embed([[text]], model=model, input_type="query")
    usage_meter.record_search_voyage_multimodal("voyage_video_query", getattr(result, "text_tokens", 0),
                                                getattr(result, "image_pixels", 0))
    return result.embeddings[0]


def embed_query(text: str) -> QueryEmbeddings:
    """Produce both embedding types needed for a 4-track video search query."""
    visual_emb = embed_query_visual(text)
    voyage_emb = embed_query_voyage(text)
    return QueryEmbeddings(visual_embedding=visual_emb, voyage_embedding=voyage_emb)


def _self_test():
    """Offline tests — no API calls or model loading needed."""

    # 1. QueryEmbeddings dataclass
    qe = QueryEmbeddings(visual_embedding=[1.0] * 1024, voyage_embedding=[2.0] * 1024)
    assert len(qe.visual_embedding) == 1024
    assert len(qe.voyage_embedding) == 1024
    assert qe.visual_embedding[0] == 1.0
    assert qe.voyage_embedding[0] == 2.0
    print("ok 1 — QueryEmbeddings dataclass")

    # 2. VoyageRateLimiter: successive calls respect interval
    limiter = VoyageRateLimiter(rpm=6000)  # fast for testing
    t0 = time.monotonic()
    limiter.wait()
    limiter.wait()
    t1 = time.monotonic()
    assert t1 - t0 < 1.0, "rate limiter should be fast at 6000 rpm"
    print("ok 2 — VoyageRateLimiter basic timing")

    # 3. VoyageRateLimiter: low rpm adds delay
    slow_limiter = VoyageRateLimiter(rpm=60)  # 1 per second
    t0 = time.monotonic()
    slow_limiter.wait()
    slow_limiter.wait()
    t1 = time.monotonic()
    assert t1 - t0 >= 0.9, f"expected ~1s delay, got {t1-t0:.2f}s"
    print("ok 3 — VoyageRateLimiter enforces delay at low rpm")

    # 4. Module-level _voyage_limiter exists
    assert isinstance(_voyage_limiter, VoyageRateLimiter)
    print("ok 4 — module-level rate limiter instance")

    # 5. embed_texts_voyage is callable (won't actually call API)
    assert callable(embed_texts_voyage)
    assert callable(embed_frames_visual)
    assert callable(embed_query)
    print("ok 5 — all public functions are callable")

    # 6-8. Every Voyage client is built with an explicit api_key/base_url —
    # the project's Voyage key is issued via MongoDB Atlas (rag-AI/CLAUDE.md:
    # "Atlas keys 403 on api.voyageai.com"), so voyageai.Client()'s own
    # default of api.voyageai.com would silently 403 in production.
    import sys
    import types

    captured_clients = []

    class FakeResult:
        # Real voyageai 0.5.0 shapes: EmbeddingsObject has total_tokens;
        # MultimodalEmbeddingsObject has text_tokens, image_pixels, total_tokens.
        def __init__(self, n, dim, total_tokens=0, text_tokens=0, image_pixels=0):
            self.embeddings = [[0.1] * dim for _ in range(n)]
            self.total_tokens = total_tokens
            self.text_tokens = text_tokens
            self.image_pixels = image_pixels

    class FakeClient:
        def __init__(self, **kwargs):
            captured_clients.append(kwargs)

        def embed(self, texts, model=None, input_type=None):
            return FakeResult(len(texts), 1024, total_tokens=7 * len(texts))

        def multimodal_embed(self, inputs, model=None, input_type=None):
            return FakeResult(len(inputs), 1024, image_pixels=16 * len(inputs),
                              total_tokens=len(inputs))

    fake_voyageai = types.ModuleType("voyageai")
    fake_voyageai.Client = FakeClient
    sys.modules["voyageai"] = fake_voyageai
    saved_key = config.EMBEDDING_API_KEY
    config.EMBEDDING_API_KEY = "fake-test-key"  # dev machines have no real key set
    try:
        embed_texts_voyage(["hello"])
        embed_frames_visual([np.zeros((4, 4, 3), dtype=np.uint8)])
        embed_query_visual("a query")
    finally:
        del sys.modules["voyageai"]
        config.EMBEDDING_API_KEY = saved_key

    assert len(captured_clients) == 3, captured_clients
    for kwargs in captured_clients:
        assert kwargs.get("api_key") == "fake-test-key", f"missing explicit api_key: {kwargs}"
        assert kwargs.get("base_url"), f"missing explicit base_url (Atlas routing): {kwargs}"
    print("ok 6 — embed_texts_voyage builds its client with explicit api_key/base_url")
    print("ok 7 — embed_frames_visual builds its client with explicit api_key/base_url")
    print("ok 8 — embed_query_visual builds its client with explicit api_key/base_url")

    # 9. embed_frames_visual returns one embedding per input frame
    sys.modules["voyageai"] = fake_voyageai
    try:
        vecs = embed_frames_visual([
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
        ])
    finally:
        del sys.modules["voyageai"]
    assert len(vecs) == 2, vecs
    assert len(vecs[0]) == 1024, vecs[0]
    print("ok 9 — embed_frames_visual returns one 1024d vector per frame")

    # 10. Ingestion embeds are metered from the response's own usage; query
    # embeds (chat search, same process) are not billed to the video.
    from pipelines import usage_meter
    sys.modules["voyageai"] = fake_voyageai
    config.EMBEDDING_API_KEY = "fake-test-key"
    meter = usage_meter.start("embed-test")
    try:
        embed_texts_voyage(["a", "b", "c"])
        embed_frames_visual([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)
        embed_query_voyage("a query")
        embed_query_visual("a query")
    finally:
        usage_meter.stop()
        del sys.modules["voyageai"]
        config.EMBEDDING_API_KEY = saved_key
    v = meter.summary()["voyage"]
    assert (v["text_tokens"], v["mm_pixels"], v["mm_images"]) == (21, 32, 2), v
    print("ok 10 — ingestion embeds metered from response usage; query embeds not")

    # 11. on_batch() runs before every paid batch (both embedders), so the
    # ingestion orchestrator can count batches for its progress bar and stop
    # a cancelled job between batches: its exception stops the loop before
    # the next request.
    class _Stop(Exception):
        pass

    for name, fn, items in (
            ("text", embed_texts_voyage, ["a", "b", "c", "d", "e"]),
            ("frames", embed_frames_visual, [np.zeros((4, 4, 3), dtype=np.uint8)] * 5)):
        ticks = []

        def _on_batch():
            ticks.append(1)
            if len(ticks) == 2:
                raise _Stop()

        sys.modules["voyageai"] = fake_voyageai
        config.EMBEDDING_API_KEY = "fake-test-key"
        meter = usage_meter.start("embed-batch-test")
        try:
            try:
                fn(items, batch_size=2, on_batch=_on_batch)
                raise AssertionError("%s: on_batch's exception must propagate" % name)
            except _Stop:
                pass
        finally:
            usage_meter.stop()
            del sys.modules["voyageai"]
            config.EMBEDDING_API_KEY = saved_key
        v = meter.summary()["voyage"]
        assert len(ticks) == 2, (name, ticks)
        # exactly one batch was sent before the stop, not all three
        sent = v["mm_requests"] if name == "frames" else v["text_requests"]
        assert sent == 1, (name, v)
    print("ok 11 — on_batch runs before each paid batch and can stop the next one")

    print("\n11 checks passed")


if __name__ == "__main__":
    _self_test()
