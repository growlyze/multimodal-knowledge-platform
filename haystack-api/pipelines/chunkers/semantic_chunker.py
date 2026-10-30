"""
Semantic chunker — video transcripts.

Boundaries come from topic shift, not from character counts: consecutive
sentences are embedded and a drop in cosine similarity marks a new topic.

  * 300-600 tokens, 50-100 overlap (config.CHUNK_SIZE_TRANSCRIPT / _OVERLAP_)
  * timestamp-aware — every chunk carries start_time / end_time
  * Domain-term-aware — a protected term from DOMAIN_JARGON is never cut
    across a boundary
  * no embedder available -> degrades to TextChunker behaviour, timestamps kept

The pipeline calls `achunk()`; `chunk()` is the synchronous fallback path.
"""

import logging
import math
import os
import re

import config
from pipelines.chunkers.base import (
    BaseChunker,
    DOMAIN_JARGON,
    Unit,
    count_tokens,
    jargon_spans,
    pack_units,
    split_long_unit,
    split_sentences,
    units_to_chunks,
)

logger = logging.getLogger("haystack-api.ingest")

# Cosine similarity below this between neighbouring sentences = new topic.
SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.5"))
# Never force a topic break that would leave a fragment behind.
MIN_CHUNK_TOKENS = int(os.getenv("SEMANTIC_MIN_CHUNK_TOKENS", "50"))
# Gaps in the bottom quartile are *preferred* break points (soft boundaries).
SOFT_BOUNDARY_PERCENTILE = 0.25

# [00:14:32]  (1:02:03)  00:14:32.500  00:00:01,000 --> 00:00:04,000  12:34
_TIMESTAMP = re.compile(
    r"^\s*[\[\(<]?\s*(\d{1,3}):(\d{2})(?::(\d{2}))?(?:[.,](\d{1,3}))?\s*[\]\)>]?"
    r"(?:\s*-->\s*[\d:.,]+)?\s*[-:]?\s*"
)
_SRT_INDEX = re.compile(r"^\s*\d+\s*$")


def parse_timestamp(line: str):
    """('[00:14:32] price runs higher') -> (872.0, 'price runs higher').

    Returns (None, line) when the line carries no timestamp.
    """
    match = _TIMESTAMP.match(line)
    if not match:
        return None, line
    a, b, c, frac = match.groups()
    # Two fields is MM:SS, three is HH:MM:SS.
    hours, minutes, seconds = (int(a), int(b), int(c)) if c else (0, int(a), int(b))
    total = hours * 3600 + minutes * 60 + seconds
    if frac:
        total += int(frac) / (10 ** len(frac))
    rest = line[match.end():].strip()
    return float(total), rest


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class SemanticChunker(BaseChunker):
    def __init__(self, max_tokens: int = None, overlap_tokens: int = None,
                 min_tokens: int = None, jargon: list = None,
                 embedder=None, similarity_threshold: float = None):
        self.max_tokens = max_tokens or config.CHUNK_SIZE_TRANSCRIPT
        self.overlap_tokens = (
            config.CHUNK_OVERLAP_TRANSCRIPT if overlap_tokens is None else overlap_tokens
        )
        self.min_tokens = min_tokens or 300
        # None = use the deployment's DOMAIN_JARGON, resolved at call time by
        # jargon_spans()/split_long_unit(); an explicit list overrides.
        self.jargon = None if jargon is None else [t.lower() for t in jargon]
        self.embedder = embedder
        self.threshold = (
            SIMILARITY_THRESHOLD if similarity_threshold is None else similarity_threshold
        )

    # ── sync path (no embeddings) ───────────────────────

    def chunk(self, text: str, metadata: dict = None) -> list:
        """Timestamp- and jargon-aware chunking without topic detection."""
        units = self._units(text)
        if not units:
            return []
        packed = pack_units(units, self.max_tokens, self.overlap_tokens, self.min_tokens)
        return units_to_chunks(packed, metadata)

    # ── async path (embeddings drive the boundaries) ────

    async def achunk(self, text: str, metadata: dict = None) -> list:
        units = self._units(text)
        if not units:
            return []

        embedder = self.embedder
        if embedder is None:
            try:
                from services.embedding_service import create_embedding_provider
                embedder = create_embedding_provider()
            except Exception as exc:
                logger.warning("semantic chunker: no embedder (%s) — text fallback", exc)
                embedder = None

        if embedder is not None and len(units) > 1:
            try:
                await self._mark_topic_shifts(units, embedder)
            except Exception as exc:
                # Chunking must not fail because an embedding API hiccuped.
                logger.warning("semantic chunker: topic detection failed (%s) — text fallback",
                               exc)

        packed = pack_units(units, self.max_tokens, self.overlap_tokens, self.min_tokens)
        chunks = units_to_chunks(packed, metadata)
        logger.info("semantic chunker: %d sentences -> %d chunks", len(units), len(chunks))
        return chunks

    async def _mark_topic_shifts(self, units: list, embedder) -> None:
        """Set .hard on real topic shifts, .boundary on the weakest gaps.

        ponytail: embeds one vector per sentence, which roughly doubles the
        ingest embedding bill for transcripts. If that matters, embed sliding
        windows of 3 sentences instead — same code path, fewer calls.
        """
        vectors = await embedder.embed([u.text for u in units], input_type="document")
        if len(vectors) != len(units):
            raise ValueError("embedder returned %d vectors for %d sentences"
                             % (len(vectors), len(units)))

        sims = [cosine(vectors[i], vectors[i + 1]) for i in range(len(units) - 1)]
        if not sims:
            return
        # Bottom quartile, but never above the mean: on uniform text there is no
        # dip to prefer, and marking every gap would shred it into fragments.
        percentile = sorted(sims)[max(0, int(len(sims) * SOFT_BOUNDARY_PERCENTILE) - 1)]
        soft_cut = min(percentile, sum(sims) / len(sims))

        running = units[0].tokens
        shifts = 0
        for i, sim in enumerate(sims):
            nxt = units[i + 1]
            if sim < self.threshold and running >= MIN_CHUNK_TOKENS:
                nxt.hard = True
                shifts += 1
                running = nxt.tokens
                continue
            if sim < soft_cut:
                nxt.boundary = True
            running += nxt.tokens
        logger.info("semantic chunker: %d topic shifts over %d gaps", shifts, len(sims))

    # ── units ───────────────────────────────────────────

    def _units(self, text: str) -> list:
        """Transcript text -> timestamped, jargon-safe sentence units."""
        if not text or not text.strip():
            return []

        units = []
        for start_time, segment in self._segments(text):
            meta = {}
            if start_time is not None:
                meta = {"start_time": start_time, "end_time": start_time}
            for sentence in split_sentences(segment):
                for piece in split_long_unit(sentence, self.max_tokens, self.jargon):
                    units.append(Unit(text=piece, meta=dict(meta)))

        self._close_timestamps(units)
        return self._merge_split_jargon(units)

    def _segments(self, text: str) -> list:
        """[(seconds_or_None, text), ...] — one entry per transcript line."""
        segments, current_time, buffer = [], None, []
        for line in text.splitlines():
            if not line.strip() or _SRT_INDEX.match(line):
                continue
            stamp, rest = parse_timestamp(line)
            if stamp is not None:
                if buffer:
                    segments.append((current_time, " ".join(buffer)))
                    buffer = []
                current_time = stamp
                if rest:
                    buffer.append(rest)
            else:
                buffer.append(line.strip())
        if buffer:
            segments.append((current_time, " ".join(buffer)))
        return segments

    @staticmethod
    def _close_timestamps(units: list) -> None:
        """end_time of a unit is where the next timestamped unit starts."""
        next_start = None
        for unit in reversed(units):
            if "start_time" not in unit.meta:
                continue
            unit.meta["end_time"] = (
                next_start if next_start is not None else unit.meta["start_time"]
            )
            next_start = unit.meta["start_time"]

    def _merge_split_jargon(self, units: list) -> list:
        """Glue neighbours back together when a jargon phrase straddles them.

        Transcripts break lines mid-phrase ("...a service level" / "agreement
        forms"), and a chunk boundary there would make the term unretrievable.
        """
        if len(units) < 2:
            return units

        merged = [units[0]]
        for unit in units[1:]:
            previous = merged[-1]
            joined = "%s %s" % (previous.text, unit.text)
            split_point = len(previous.text) + 1
            if any(s < split_point < e for s, e in jargon_spans(joined, self.jargon)):
                previous.text = joined
                previous.tokens = count_tokens(joined)
                for key, value in unit.meta.items():
                    if key.startswith("end_") or key not in previous.meta:
                        previous.meta[key] = value
                continue
            merged.append(unit)
        return merged


if __name__ == "__main__":
    import asyncio

    class FakeEmbedder:
        """Two topics: vectors point in orthogonal directions."""
        async def embed(self, texts, input_type="document"):
            return [[1.0, 0.0] if "batch" in t else [0.0, 1.0]
                    for t in texts]

    transcript = "\n".join(
        ["[00:00:%02d] A batch job is an asynchronous task queued for later." % i
         for i in range(0, 20)]
        + ["[00:01:%02d] The dashboard renders a line on the chart." % i for i in range(0, 20)]
    )
    out = asyncio.run(SemanticChunker(embedder=FakeEmbedder()).achunk(
        transcript, {"source": "walkthrough.txt", "source_type": "transcript"}))
    assert len(out) >= 2, "topic shift not detected"
    assert all("start_time" in c.metadata and "end_time" in c.metadata for c in out)
    assert "batch" in out[0].content and "dashboard" not in out[0].content, out[0].content
    print("%d chunks, times: %s" % (
        len(out), [(c.metadata["start_time"], c.metadata["end_time"]) for c in out]))
