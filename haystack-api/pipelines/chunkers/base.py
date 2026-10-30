"""
Chunker interface plus the text primitives every chunker shares.

Public surface (INGESTION owns this; evaluation/ imports it too):
    TextChunk, BaseChunker, Unit
    count_tokens(), split_sentences(), split_long_unit(), pack_units()
    DOMAIN_JARGON, jargon_spans(), find_jargon()
"""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ── token counting ──────────────────────────────────────
# ponytail: 4 chars/token is the standard English heuristic and needs no
# dependency. Every size target, chunker and quality check goes through this
# one function, so they stay self-consistent even though it is approximate.
# Upgrade path: tiktoken behind the same name (INFRA has to add the package).

def count_tokens(text: str) -> int:
    """Approximate token count. ~4 characters per token."""
    return max(1, round(len(text) / 4))


# ── Domain terminology ──────────────────────────────────
# Multi-word terms must never be cut in half by a chunk boundary. The
# abbreviations are single tokens and are here for retrieval evaluation.
# Empty by default: each deployment lists its own protected terms in
# DOMAIN_JARGON (comma-separated), e.g.
# DOMAIN_JARGON="service level agreement,churn rate,sla,kpi"
DOMAIN_JARGON = [
    t.strip().lower()
    for t in os.getenv("DOMAIN_JARGON", "").split(",")
    if t.strip()
]


def _jargon_pattern(terms):
    # Longest first so a multi-word term wins over its shorter parts.
    return re.compile(
        r"\b(?:%s)\b" % "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)),
        re.IGNORECASE,
    )


def jargon_spans(text: str, terms: list = None) -> list:
    """[(start, end), ...] character spans of every jargon occurrence.

    `terms=None` reads the deployment's DOMAIN_JARGON at call time; an empty
    list (the default) protects nothing and matches nothing.
    """
    if terms is None:
        terms = DOMAIN_JARGON
    if not terms:
        return []
    return [m.span() for m in _jargon_pattern(terms).finditer(text)]


def find_jargon(text: str, terms: list = None) -> set:
    """Lowercased jargon terms present in text."""
    return {text[s:e].lower() for s, e in jargon_spans(text, terms)}


# ── sentence splitting ──────────────────────────────────
# Split after . ! ? … when followed by whitespace, and at newlines —
# transcripts and lists often carry no terminal punctuation at all.

_SENTENCE_END = re.compile(r'(?<=[.!?…])["\')\]]*\s+(?=\S)|\n+')

# A "sentence end" right after one of these is an abbreviation, not an end.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "approx",
    "fig", "no", "inc", "ltd", "co", "e.g", "i.e", "a.m", "p.m", "u.s", "est",
}
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")


def split_sentences(text: str) -> list:
    """Split into sentences. Never returns empty strings.

    Abbreviations ("e.g.", "Mr.") are merged back, because a chunk that starts
    at "g. the introduction ..." is unusable in a retrieval result.
    """
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()]
    merged = []
    for part in parts:
        if merged:
            tail = _LAST_WORD.search(merged[-1])
            # "...e.g." or a lone initial "J." — glue the next part back on.
            if tail and (tail.group(1).lower().rstrip(".") in _ABBREV or len(tail.group(1)) == 1):
                merged[-1] = "%s %s" % (merged[-1], part)
                continue
        merged.append(part)
    return merged


def split_long_unit(text: str, max_tokens: int, terms: list = None) -> list:
    """Last-resort split of a single unit already larger than max_tokens.

    Cuts on whitespace only, and never inside a domain-term phrase — the cut
    moves back to the start of the term instead.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    spans = jargon_spans(text, terms)
    pieces, start, budget = [], 0, 0
    for match in re.finditer(r"\S+", text):
        word_tokens = count_tokens(match.group()) + 1
        if budget and budget + word_tokens > max_tokens:
            cut = _safe_cut(match.start(), start, spans)
            if cut > start:
                pieces.append(text[start:cut].strip())
                start, budget = cut, 0
        budget += word_tokens

    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return [p for p in pieces if p] or [text]


def _safe_cut(offset: int, start: int, spans: list) -> int:
    """Move a cut point out of the middle of a jargon phrase."""
    for s, e in spans:
        if s < offset < e:
            return s if s > start else e
    return offset


# ── chunk packing ───────────────────────────────────────

@dataclass
class Unit:
    """One indivisible piece of text plus what the packer needs to know."""
    text: str
    tokens: int = 0
    boundary: bool = False      # prefer starting a new chunk here (paragraph, topic shift)
    hard: bool = False          # must start a new chunk here (function definition)
    meta: dict = field(default_factory=dict)   # merged into the chunk metadata

    def __post_init__(self):
        if not self.tokens:
            self.tokens = count_tokens(self.text)


@dataclass
class TextChunk:
    content: str
    metadata: dict


def pack_units(units: list, max_tokens: int, overlap_tokens: int,
               min_tokens: int = 0) -> list:
    """Greedily pack units into chunks, carrying an overlap tail between them.

    Units are never split. Whatever the caller passes in *is* the boundary
    guarantee — sentences for prose, blocks for code — which is what keeps
    chunks from ending mid-sentence or mid-function.

    Returns [[Unit, ...], ...] so callers can build metadata from unit.meta.
    """
    if not units:
        return []
    if not min_tokens:
        min_tokens = int(max_tokens * 0.75)

    chunks, current, budget = [], [], 0
    for unit in units:
        if current and (unit.hard
                        or budget + unit.tokens > max_tokens
                        or (unit.boundary and budget >= min_tokens)):
            chunks.append(current)
            current = [] if unit.hard else _overlap_tail(current, overlap_tokens)
            budget = sum(u.tokens for u in current)
        current.append(unit)
        budget += unit.tokens

    if current:
        chunks.append(current)
    return _drop_contained(chunks)


def _overlap_tail(units: list, overlap_tokens: int) -> list:
    """Trailing units of a finished chunk that carry into the next one."""
    if overlap_tokens <= 0 or len(units) < 2:
        return []
    tail, total = [], 0
    for unit in reversed(units[1:]):     # never carry the whole chunk forward
        if total + unit.tokens > overlap_tokens:
            break
        tail.insert(0, unit)
        total += unit.tokens
    return tail


def _drop_contained(chunks: list) -> list:
    """A trailing chunk made only of overlap adds nothing but a duplicate.

    Identity, not equality: repeated sentences are common in transcripts and
    equal-by-value Units would collapse the whole document into one chunk.
    """
    while len(chunks) > 1:
        previous = {id(u) for u in chunks[-2]}
        if not all(id(u) in previous for u in chunks[-1]):
            break
        chunks.pop()
    return chunks


def units_to_chunks(packed: list, base_metadata: dict = None, joiner: str = " ") -> list:
    """[[Unit, ...], ...] -> [TextChunk, ...] with index and token metadata."""
    chunks = []
    for units in packed:
        content = joiner.join(u.text for u in units).strip()
        if not content:
            continue
        meta = dict(base_metadata or {})
        for unit in units:
            for key, value in unit.meta.items():
                # First unit wins for start_*, last wins for end_* — the natural
                # semantics for timestamps and line numbers.
                if key.startswith("end_") or key not in meta:
                    meta[key] = value
        meta["chunk_index"] = len(chunks)
        meta["token_count"] = count_tokens(content)
        chunks.append(TextChunk(content=content, metadata=meta))
    return chunks


# ── interface ───────────────────────────────────────────

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict = None) -> list:
        """Split text into chunks with metadata."""
        ...

    async def achunk(self, text: str, metadata: dict = None) -> list:
        """Async entry point. Only SemanticChunker overrides it (it embeds);
        everything else runs the sync path, so the pipeline has one call site."""
        return self.chunk(text, metadata)
