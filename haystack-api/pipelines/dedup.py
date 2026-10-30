"""
Duplicate detection for ingestion — content hashes, checked before any spend.

Why this exists (2026-10-26, measured live): the rag filter re-POSTs every
attachment on every chat turn, so one PDF in a ten-turn chat was extracted and
embedded ten times; and users re-upload the same file under a new name
(`CME_MINI_NQZ2026, 5.csv` + `proba CME_MINI_NQZ2026, 5.csv`,
`DL_ES1_test.csv` + `CME_MINI_DL_ES1!, 5 (2).csv`), which the old
replace-by-filename rule cannot see — each copy became a second document.

Two keys, both SHA-256, both stored flat in the Qdrant payload of every chunk
(and, for structured files, in the data_catalog row's metadata):

* ``file_sha256`` — the uploaded bytes. Checked before extraction, so a
  byte-identical re-upload never reaches Tika, the chunker or Voyage.
* ``doc_sha256``  — the extracted text after normalize(). Catches the same
  content in different bytes: a re-exported PDF, a page scraped twice with a
  new ``scraped_at`` (front matter is stripped before hashing).

On a hit: same filename -> "unchanged", nothing happens; different filename ->
"duplicate", nothing is stored and the job names the file it duplicates.

Deliberately NOT done (see KB 13-decision-log.md, the D-row for this module):
cross-document chunk dedup and near-duplicate (MinHash) matching. Measured,
the knowledge base holds 25 redundant chunks out of 3834 — shared code
boilerplate that belongs to each page's own citation — and a near-match on
reference pages differing in one function name would delete real content.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import unicodedata

logger = logging.getLogger("haystack-api.dedup")

# Bump when normalize() changes: every stored hash was computed with the old
# rule, and a silent change would make every re-upload look new (wasted spend)
# or, worse, make different text look identical. Stored with each chunk.
HASH_VERSION = 1

# The fields duplicate lookups filter on. Indexed as keywords so a lookup stays
# one index probe instead of a collection scan as the collections grow.
HASH_FIELDS = ("file_sha256", "doc_sha256")

_SPACES = re.compile(r"[ \t\f\v ]+")
_BLANK_LINES = re.compile(r"\n{2,}")


def normalize(text: str) -> str:
    """Layout-insensitive, meaning-preserving form of a text, for hashing only.

    NFKC folds full-width letters and ligatures, CRLF/CR become LF, runs of
    spaces after a line's indentation collapse, trailing spaces and blank-line
    runs go. Two things are kept because they are meaning, not layout: case
    (most code is case-sensitive, `Total` is not `total`, and so are identifiers) and
    each line's leading indentation (Python-style blocks are delimited by it, so a
    re-indented script is a real edit, never "unchanged").
    The stored chunk text is never this — only its hash is.
    """
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        body = line.lstrip(" \t")
        indent = line[:len(line) - len(body)]
        lines.append(indent + _SPACES.sub(" ", body).rstrip() if body else "")
    return _BLANK_LINES.sub("\n", "\n".join(lines)).strip("\n")


def text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def bytes_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data or b"").hexdigest()


# Front-matter keys that change on every scrape without changing anything a
# citation shows — hashing them would make every re-scrape look "changed".
VOLATILE_META = frozenset({"scraped_at"})


def meta_hash(front: dict) -> str:
    """Hash of a page's citation metadata (front matter), volatile keys out.

    doc_sha256 hashes the body only, so a re-scrape whose title or URL moved
    would read "unchanged" and keep citing the old title forever. Same
    filename + same body + a different meta_sha256 is therefore re-indexed.
    """
    stable = {str(k): str(v) for k, v in (front or {}).items() if k not in VOLATILE_META}
    blob = json.dumps(stable, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def drop_repeated_chunks(chunks: list) -> tuple:
    """Keep the first chunk of each distinct text; returns (kept, n_dropped).

    Within ONE document only — a repeated disclaimer or page footer is pure
    noise there. Each kept chunk gets its ``chunk_sha256`` in metadata, so a
    future cross-document report can group on it without re-reading text.
    """
    seen, kept = set(), []
    for chunk in chunks:
        digest = text_hash(chunk.content)
        if digest in seen:
            continue
        seen.add(digest)
        chunk.metadata["chunk_sha256"] = digest
        kept.append(chunk)
    return kept, len(chunks) - len(kept)


# ── concurrency ────────────────────────────────────────
# A check-then-write against Qdrant races: two uploads of the same file in the
# same second both see "not there" and both embed. haystack-api runs a single
# uvicorn worker (Dockerfile CMD has no --workers), so a per-process lock keyed
# on the hash closes that window. ponytail: if it ever runs several workers,
# this becomes a PostgreSQL `INSERT ... ON CONFLICT DO NOTHING` claim instead.

_locks = {}     # key -> [lock, holders]


@contextlib.asynccontextmanager
async def hold(key: str):
    entry = _locks.setdefault(key, [asyncio.Lock(), 0])
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if not entry[1]:
            _locks.pop(key, None)


# ── lookups ────────────────────────────────────────────

_indexed = set()


async def ensure_hash_indexes(qdrant, collection: str) -> None:
    """Best effort, once per collection per process — a lookup works without
    the index (Qdrant scans), it is only slower, so a failure never fails a job."""
    if collection in _indexed:
        return
    try:
        for field in HASH_FIELDS:
            await qdrant.create_keyword_index(collection, field)
    except Exception as exc:
        # Not remembered, so the next ingest into this collection retries.
        logger.debug("payload index on %s not created: %s", collection, exc)
        return
    _indexed.add(collection)


async def find_existing(qdrant, collection: str, field: str, value: str,
                        filename: str) -> dict:
    """Who already holds this hash in this collection? None when nobody does.

    Returns {"kind": "unchanged"|"duplicate", "source": <filename holding it>,
    "chunks": <how many points carry it>}. "unchanged" wins when this same
    filename is among the holders, so a file is never called a duplicate of
    itself just because a copy of it also exists under another name.
    """
    from services.qdrant_service import source_of

    holders, metas, offset = {}, set(), None
    while True:
        points, offset = await qdrant.scroll(collection, limit=256, offset=offset,
                                             filters={field: value})
        for point in points:
            name = source_of(point["payload"])
            holders[name] = holders.get(name, 0) + 1
            if name == filename:
                metas.add(point["payload"].get("meta_sha256"))
        if not offset:
            break
    if not holders:
        return None
    if filename in holders:
        return {"kind": "unchanged", "source": filename, "chunks": holders[filename],
                "meta": metas}
    source = sorted(holders)[0]
    return {"kind": "duplicate", "source": source, "chunks": holders[source]}


def skip_report(found: dict, field: str) -> dict:
    """The job's quality report for a skip — says why, and what it matched."""
    report = {"skipped": found["kind"], "matched_on": field,
              "existing_chunks": found["chunks"]}
    if found["kind"] == "duplicate":
        report["duplicate_of"] = found["source"]
    return report
