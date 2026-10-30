"""
Recursive text chunker — PDFs, DOCX, plain text.

Paragraph boundaries are preferred split points, sentence boundaries are the
fallback, and a word-level split only happens for a single sentence that is
already over the maximum on its own.

Target 400-500 tokens, 100 token overlap (config.CHUNK_SIZE_TEXT / _OVERLAP_TEXT).
"""

import logging
import re

import config
from pipelines.chunkers.base import (
    BaseChunker,
    Unit,
    count_tokens,
    pack_units,
    split_long_unit,
    split_sentences,
    units_to_chunks,
)

logger = logging.getLogger("haystack-api.ingest")

_PARAGRAPH = re.compile(r"\n\s*\n+")
# Tika emits "page N" markers on some PDFs; cheap page numbers when present.
_PAGE_MARKER = re.compile(r"^\s*(?:page\s+)?(\d{1,4})\s*$", re.IGNORECASE)
# A real page boundary: Tika's XHTML wraps every PDF page in <div class="page">,
# and its text output separates them with a form feed. Neither survives every
# extraction path — when there is no boundary at all, page_number stays null.
_PAGE_BREAK = re.compile(r"<div\s+class=\"page\"[^>]*>|\f")


class TextChunker(BaseChunker):
    def __init__(self, max_tokens: int = None, overlap_tokens: int = None,
                 min_tokens: int = None):
        self.max_tokens = max_tokens or config.CHUNK_SIZE_TEXT
        self.overlap_tokens = (
            config.CHUNK_OVERLAP_TEXT if overlap_tokens is None else overlap_tokens
        )
        self.min_tokens = min_tokens or int(self.max_tokens * 0.8)

    def chunk(self, text: str, metadata: dict = None) -> list:
        if not text or not text.strip():
            return []
        units = self._units(text)
        packed = pack_units(units, self.max_tokens, self.overlap_tokens, self.min_tokens)
        chunks = units_to_chunks(packed, metadata)
        logger.info("text chunker: %d chars -> %d chunks (max=%d, overlap=%d)",
                    len(text), len(chunks), self.max_tokens, self.overlap_tokens)
        return chunks

    def _units(self, text: str) -> list:
        """Page -> paragraph -> sentence -> (only if needed) word-level units."""
        units = []
        for page, section in _pages(text):
            for paragraph in _PARAGRAPH.split(section):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                marker = _PAGE_MARKER.match(paragraph)
                if marker:
                    page = int(marker.group(1))
                    continue

                meta = {"page_number": page} if page is not None else {}
                first = True
                for sentence in split_sentences(paragraph):
                    for piece in split_long_unit(sentence, self.max_tokens):
                        units.append(Unit(text=piece, boundary=first, meta=dict(meta)))
                        first = False
        return units


def _pages(text: str) -> list:
    """[(page_number_or_None, text), ...] — one entry per PDF page.

    A document with no page boundaries is one section with no number, so
    source.page_number ends up null rather than wrong.
    """
    sections = _PAGE_BREAK.split(text)
    if len(sections) == 1:
        return [(None, text)]
    if not sections[0].strip():
        sections.pop(0)             # the marker opened the document
    return list(enumerate(sections, start=1))


def chunk_text(text: str, metadata: dict = None, **kw) -> list:
    """Convenience wrapper — the pipeline uses the class."""
    return TextChunker(**kw).chunk(text, metadata)


if __name__ == "__main__":
    sample = ("The retry policy is only a safety net. " * 40 + "\n\n"
              + "A batch window is the stretch of time where background jobs may run. " * 40)
    out = TextChunker(max_tokens=500, overlap_tokens=100).chunk(
        sample, {"source": "demo.txt", "source_type": "document"})
    assert out, "no chunks produced"
    assert all(count_tokens(c.content) <= 500 for c in out), \
        [count_tokens(c.content) for c in out]
    assert all(c.content[-1] in ".!?" for c in out[:-1]), "chunk ended mid-sentence"
    assert all(c.metadata.get("page_number") is None for c in out), "no markers, no pages"

    paged = TextChunker(max_tokens=100, overlap_tokens=0).chunk(
        "\f" + "Page one text. " * 40 + "\f" + "Page two text. " * 40, {})
    pages = {c.metadata.get("page_number") for c in paged}
    assert pages == {1, 2}, pages
    print("%d chunks: %s" % (len(out), [count_tokens(c.content) for c in out]))
