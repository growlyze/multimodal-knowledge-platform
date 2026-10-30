"""
/admin — Knowledge base management endpoints.

Called by the admin dashboard (dashboard/, server-side proxy) or directly.

/data/search and /data/table are the pair `data_tools.load_dataset()` uses from
inside the code sandbox: search finds the table that holds a dataset, table
hands back its rows. The sandbox has no database credentials of its own (locked
architecture decision #4), so this is the only way it reaches stored data.

/tables, /tables/{name} and DELETE /tables/{name} are the same catalog seen by a
person instead: the panel's Spreadsheet data page lists, previews and drops.

/storage is the latest row ops/storage_monitor.sh (a host-side systemd timer) wrote
into ops_storage_reports, graded per probe, for the panel's Storage section.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

import config
from services.auth import require_admin, require_data_reader
from services.db_service import DBError, TABLE_PREFIX, get_db, quote
from services.qdrant_service import (QdrantError, chunk_index, get_qdrant,
                                    source_of)

logger = logging.getLogger("haystack-api.admin")
router = APIRouter(dependencies=[Depends(require_admin)])
# Read-only dataset endpoints the code sandbox calls (data_tools.load_dataset).
# Mounted under /admin like `router`, but also accept DATA_API_KEY.
data_router = APIRouter(dependencies=[Depends(require_data_reader)])

MAX_RESULTS = 50
# QUERY_ROW_CAP (1000) is sized for a text-to-SQL answer someone reads. An analysis
# in the sandbox needs the whole dataset — e.g. 211K rows of three years of 5-minute
# data. This is a PAGE size: above it the reply says `truncated: true` and
# load_dataset() (data_tools) fetches the next page by offset. Before that it read
# page one only and silently analysed a fraction of a larger table (2026-10-22).
# ponytail: 250K rows is ~40MB of JSON and takes seconds. If datasets outgrow it,
# the fix is a binary format (parquet bytes) on this endpoint, not a bigger number.
DEFAULT_ROW_LIMIT = 250_000
MAX_ROW_LIMIT = 1_000_000

_WORD = re.compile(r"[a-z0-9]+")

# Words every catalog row contains, which therefore carry no signal: TABLE_PREFIX
# puts "data" in every table name, and build_summary puts the rest in every summary.
# Without this, "gold tick data 1987" scores a hit on an unrelated table and
# load_dataset() analyses it — a match on boilerplate is indistinguishable from a real one.
# ponytail: a fixed list, like coverage_analyzer's STOPWORDS. It only has to cover
# build_summary's own vocabulary, which is ours and changes when we change it.
CATALOG_NOISE = frozenset("""
data file files csv tsv txt xlsx xls json jsonl parquet format structured
table tables postgresql sql row rows column columns col sheet sheets
sample samples numeric range ranges date dates year years covered
symbol symbols instrument instruments min max mean index entry summary
open high low close volume time timestamp datetime
""".split())


def _words(text) -> set:
    return set(_WORD.findall(str(text or "").lower()))


def _score(row: dict, tokens: list) -> int:
    """How well one catalog row answers the query. 0 means not a match.

    Whole words, never substrings: as a substring "AT" matches "data",
    "columns" and half of every summary. load_dataset() takes hits[0] and
    analyses it, so a sloppy match here silently runs on the wrong dataset —
    which looks like a result rather than like an error.
    """
    metadata = row.get("metadata") or {}
    symbols = {str(s).lower() for s in (metadata.get("symbols") or [])}
    names = _words(row.get("filename")) | _words(row.get("table_name"))
    # Names this same file was also uploaded under. Dedup records a byte-identical
    # re-upload as an alias on the original row instead of loading it a second
    # time, so without this load_dataset("DL_AT1_test") stopped finding the data
    # the user uploaded under exactly that name.
    for alias in metadata.get("aliases") or []:
        names |= _words(alias)
    summary = _words(row.get("summary"))

    score = 0
    for token in tokens:
        if token in symbols:
            score += 10         # the symbol itself — what "ACME" actually means
        elif token in names:
            score += 5          # ACME_2020_2023.csv
        elif token in summary:
            score += 1          # a year, a column name, a word from the sample rows
    return score


def _entry(row: dict) -> dict:
    """One catalog row in the shape load_dataset() reads.

    `table_name` is the load-bearing field — it is what /data/table is keyed on.
    """
    metadata = row.get("metadata") or {}
    symbols = [str(s) for s in (metadata.get("symbols") or [])]
    return {
        "table_name": row.get("table_name"),
        "filename": row.get("filename"),
        "sheet": row.get("sheet"),
        "format": row.get("format"),
        "symbol": symbols[0] if symbols else None,
        "symbols": symbols,
        "row_count": row.get("row_count"),
        "columns": row.get("columns") or [],
        "date_range": metadata.get("date_range") or {},
        "created_at": row.get("created_at"),
        "summary": row.get("summary"),
    }


@data_router.get("/data/search")
async def search_datasets(
    q: str = Query(..., description='Dataset name, symbol or keyword, e.g. "sales", "sensors 2022"'),
    limit: int = Query(5, ge=1, le=MAX_RESULTS),
):
    """Find stored datasets by symbol, filename or summary text.

    Returns a bare JSON array, best match first, so load_dataset() can take
    hits[0] and trust it. An empty array with HTTP 200 means nothing matched —
    an empty catalog is a normal answer, not an error. A database that is *down*
    is a different thing and does raise, so it shows up in monitoring rather than
    looking like "no data exists".
    """
    try:
        rows = await get_db().list_tables()
    except DBError as exc:
        logger.warning("dataset search unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Data catalog unavailable") from exc

    tokens = sorted(_words(q) - CATALOG_NOISE)
    if not tokens:
        # A bare ?q= lists the catalog newest-first, which is what someone poking
        # at the endpoint by hand actually wants.
        matches = [(0, row) for row in rows]
    else:
        matches = [(s, row) for s, row in ((_score(row, tokens), row) for row in rows) if s]

    # list_tables() already comes back newest-first and sort is stable, so ties on
    # score keep that order — the newest upload of a dataset wins over last year's.
    matches.sort(key=lambda pair: pair[0], reverse=True)
    logger.info("dataset search %r -> %d/%d catalog rows", q, len(matches), len(rows))
    return [_entry(row) for _, row in matches[:limit]]


@data_router.get("/data/table/{table_name}")
async def read_dataset_table(
    table_name: str,
    limit: int = Query(DEFAULT_ROW_LIMIT, ge=1, le=MAX_ROW_LIMIT),
    offset: int = Query(0, ge=0),
):
    """Rows of one stored dataset, for load_dataset() to build a DataFrame from.

    The table name is resolved against `data_catalog` rather than trusted: an
    identifier cannot be a bind parameter, so the only names that reach SQL are
    ones this service created. Anything else is a 404, not an injection.

    Rows come in physical order (ORDER BY ctid): load_dataset() pages through a
    table above `limit` with OFFSET, and without a fixed order PostgreSQL may
    return each page from a different starting point (synchronized seq scans), so
    pages could overlap or skip rows. These tables are written once and never
    updated, so ctid order is stable. load_dataset() still sorts by time after.
    """
    try:
        catalog = await get_db().list_tables()
    except DBError as exc:
        logger.warning("dataset read unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Data catalog unavailable") from exc

    match = next((r for r in catalog if r.get("table_name") == table_name), None)
    if match is None or not str(table_name).startswith(TABLE_PREFIX):
        raise HTTPException(status_code=404, detail="No such dataset: %s" % table_name)

    try:
        # limit + 1 so a full page can be told apart from a truncated one.
        rows = await get_db().query(
            "SELECT * FROM %s ORDER BY ctid LIMIT $1 OFFSET $2" % quote(match["table_name"]),
            limit + 1, offset, limit=limit + 1)
    except DBError as exc:
        logger.warning("dataset read failed for %s: %s", table_name, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    truncated = len(rows) > limit
    if truncated:
        rows = rows[:limit]
        logger.info("%s page at offset %d: %d rows of %s, more follow "
                    "(load_dataset() fetches the next page)",
                    table_name, offset, limit, match.get("row_count"))
    return {
        "table_name": match["table_name"],
        "filename": match.get("filename"),
        "row_count": len(rows),
        "total_rows": match.get("row_count"),
        "offset": offset,
        "truncated": truncated,
        "rows": rows,
    }


# ── data tables (the admin panel's Spreadsheet data page) ───
#
# /data/* above is the sandbox's read path and answers in the shape
# load_dataset() wants. These three are the panel's, and answer in the shape a
# person reads: one flat row per table, no summary text, sizes and dates spelled
# out. Same catalog underneath — a different audience, not different data.

TABLE_PREVIEW_ROWS = 100
MAX_PREVIEW_ROWS = 1000


def _table_row(row: dict) -> dict:
    """One catalog row, flattened for a dataframe.

    Nested values are joined into strings rather than left as lists: the panel
    renders this straight into st.dataframe, where a list cell prints as a repr.
    The long `summary` is dropped entirely — it is paragraphs, and it is already
    served by /data/search for the sandbox that needs it.
    """
    metadata = row.get("metadata") or {}
    dates = metadata.get("date_range") or {}
    columns = row.get("columns") or []
    return {
        "table_name": row.get("table_name"),
        "filename": row.get("filename"),
        "sheet": row.get("sheet"),
        "format": row.get("format"),
        "rows": row.get("row_count"),
        "columns": row.get("col_count") or len(columns),
        "symbols": ", ".join(str(s) for s in (metadata.get("symbols") or [])),
        "from": (dates.get("start") or "")[:10],
        "to": (dates.get("end") or "")[:10],
        "column_names": ", ".join(str(c.get("name")) for c in columns
                                  if isinstance(c, dict)),
        "created_at": row.get("created_at"),
    }


async def _catalog_row(table_name: str) -> dict:
    """Resolve a table name against data_catalog, or 404.

    The name is never trusted straight into SQL — an identifier cannot be a bind
    parameter, so the only names that reach a query are ones this service wrote.
    """
    try:
        catalog = await get_db().list_tables()
    except DBError as exc:
        logger.warning("data catalog unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Data catalog unavailable") from exc

    match = next((r for r in catalog if r.get("table_name") == table_name), None)
    if match is None or not str(table_name).startswith(TABLE_PREFIX):
        raise HTTPException(status_code=404, detail="No such data table: %s" % table_name)
    return match


@router.get("/tables")
async def list_data_tables():
    """Every spreadsheet loaded into the database, newest first."""
    try:
        rows = await get_db().list_tables()
    except DBError as exc:
        logger.warning("data catalog unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Data catalog unavailable") from exc
    return {"count": len(rows), "tables": [_table_row(r) for r in rows]}


@router.get("/tables/{table_name}")
async def preview_data_table(
    table_name: str,
    limit: int = Query(TABLE_PREVIEW_ROWS, ge=1, le=MAX_PREVIEW_ROWS),
):
    """The first rows of one data table, for the panel's preview pane.

    Capped far below /data/table's limit on purpose: this one is rendered in a
    browser, not fed to an analysis.
    """
    match = await _catalog_row(table_name)
    try:
        rows = await get_db().query(
            "SELECT * FROM %s LIMIT $1" % quote(match["table_name"]), limit, limit=limit)
    except DBError as exc:
        logger.warning("preview failed for %s: %s", table_name, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"table_name": match["table_name"], "filename": match.get("filename"),
            "total_rows": match.get("row_count"), "rows": rows,
            "summary": match.get("summary")}


@router.delete("/tables/{table_name}")
async def drop_data_table(table_name: str):
    """Drop one data table, its catalog row, and the Qdrant summary that names it.

    The summary is the reason this is not just DROP TABLE. It is the chunk that
    tells the model "the answer is in data_nq_20260826101500" — left behind, it
    keeps saying so about a table that no longer exists, and text-to-SQL fails on
    a question the knowledge base claims it can answer. Deleting by `table_name`
    rather than by filename is what keeps the other sheets of the same workbook
    intact.
    """
    from pipelines.structured import SOURCE_TYPE as STRUCTURED

    match = await _catalog_row(table_name)

    removed = 0
    try:
        removed = await get_qdrant().delete_by_filter(
            config.COLLECTION_DOCUMENTS,
            {"table_name": match["table_name"], "source_type": STRUCTURED})
    except (QdrantError, httpx.HTTPError) as exc:
        # Not fatal: an orphaned summary is a stale pointer, an undropped table
        # is rows nobody can see. Drop the table and say the chunk survived.
        logger.warning("could not remove the summary chunk for %s: %s", table_name, exc)

    try:
        await get_db().drop_table(match["table_name"])
    except DBError as exc:
        logger.warning("drop failed for %s: %s", table_name, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info("dropped data table %s (%s), %d summary chunk(s)",
                table_name, match.get("filename"), removed)
    return {"table_name": match["table_name"], "filename": match.get("filename"),
            "dropped": True, "chunks_removed": removed}

# ── knowledge base (Qdrant) ─────────────────────────────
# The panel's dashboard and browser both need "how many *documents*", which
# Qdrant does not store — it stores chunks. So the counts come from grouping the
# points by their source filename, which means scrolling the collection.
#
# Two payload shapes have to be read at once. Points written before the
# namespacing carry `source` as the filename string with url/title flat beside
# it; pipelines/ingestion.py now nests them under a `source` object.
# scripts/backfill_metadata.py converts the old ones, and it may not have run.

SCROLL_BATCH = 1000
# ponytail: a full scroll per request, no cache. 5.4K points is ~6 round trips to
# a container-local Qdrant. If a collection grows past MAX_SCAN the fix is a
# facet request (Qdrant /facet on an indexed source.filename), not a bigger cap.
MAX_SCAN = 50_000
CONTENT_PREVIEW = 200

# Enough payload to group by source. Asking for named keys rather than the whole
# payload keeps `content` — the bulk of every point — off the wire.
SOURCE_FIELDS = ["source", "source_type", "ingested_at", "url", "title", "scraped_at",
                 "video_id"]


def _source_fields(payload: dict) -> dict:
    """One chunk's payload -> the source fields the panel shows, in either schema.

    source_of() already resolves the filename across both shapes; this adds the
    citation fields beside it, which moved into the `source` object at the same
    time the filename did.
    """
    source = payload.get("source")
    if isinstance(source, dict):
        url, title = source.get("url"), source.get("title")
    else:
        url, title = payload.get("url"), payload.get("title")
    filename = source_of(payload)
    if not filename and payload.get("video_id"):
        # Video points carry no `source`; without this every video collection
        # grouped into one "(unknown source)" and the dashboard counted 1 per
        # collection however many videos were in (2026-10-21).
        filename = str(payload["video_id"])
    return {
        "filename": filename,
        "video_id": payload.get("video_id"),
        "url": url,
        "title": title,
        "source_type": payload.get("source_type"),
        "ingested_at": payload.get("ingested_at") or payload.get("scraped_at"),
    }


def _group_by_source(points: list, collection: str) -> list:
    """Chunks -> one entry per source file, alphabetical.

    `ingested_at` is the earliest chunk's. Both writers stamp it with
    datetime.isoformat() in UTC, so the strings sort the same way the instants
    do; a value in some other format only mis-sorts a file against itself.
    """
    documents = {}
    for point in points:
        fields = _source_fields(point.get("payload") or {})
        name = fields["filename"] or "(unknown source)"
        document = documents.get(name)
        if document is None:
            documents[name] = document = {
                "source": name, "filename": name, "collection": collection,
                "chunk_count": 0, "source_type": None, "url": None,
                "title": None, "ingested_at": None,
            }
            if fields["video_id"]:
                document["video_id"] = str(fields["video_id"])
        document["chunk_count"] += 1
        stamp = fields["ingested_at"]
        if stamp and (document["ingested_at"] is None or stamp < document["ingested_at"]):
            document["ingested_at"] = stamp
        # First non-empty wins: these describe the file, so every chunk agrees,
        # but a half-backfilled file can carry them on some chunks and not others.
        for key in ("source_type", "url", "title"):
            if not document[key] and fields[key]:
                document[key] = fields[key]
    return sorted(documents.values(), key=lambda d: d["source"])


async def _scroll_sources(qdrant, collection: str) -> list:
    """Every point in a collection, payload trimmed to SOURCE_FIELDS.

    QdrantService.scroll() always asks for the whole payload, and counting
    documents does not need `content` — 5.4K chunks of it is megabytes per
    dashboard load. Naming the fields is the only reason this posts directly.
    """
    points, offset = [], None
    while True:
        body = {"limit": SCROLL_BATCH, "with_payload": SOURCE_FIELDS}
        if offset is not None:
            body["offset"] = offset

        response = await qdrant.client.post(
            "/collections/%s/points/scroll" % collection, json=body)
        if response.status_code == 404:
            return points            # nothing indexed there yet — not an error
        if response.status_code >= 400:
            raise QdrantError("scroll %s -> %s: %s"
                              % (collection, response.status_code, response.text[:300]))

        result = response.json().get("result") or {}
        points += result.get("points") or []
        offset = result.get("next_page_offset")
        if offset is None:
            return points
        if len(points) >= MAX_SCAN:
            logger.warning("%s has more than %d points — document list truncated",
                           collection, MAX_SCAN)
            return points


async def _collection_names(qdrant) -> list:
    """Every collection Qdrant holds, the two configured ones first.

    The configured pair is always listed even when it does not exist yet, so an
    empty documents reads as "0 chunks" rather than vanishing.
    """
    known = [config.COLLECTION_DOCUMENTS, config.COLLECTION_CODE]
    try:
        response = await qdrant.client.get("/collections")
        response.raise_for_status()
        found = [c["name"] for c in
                 (response.json().get("result") or {}).get("collections") or []]
    except Exception as exc:
        logger.warning("cannot list Qdrant collections (%s: %s) — using the "
                       "configured pair only", type(exc).__name__, exc)
        return known
    return known + sorted(name for name in found if name not in known)


async def _documents_in(qdrant, collection: str) -> list:
    return _group_by_source(await _scroll_sources(qdrant, collection), collection)


@router.get("/stats")
async def get_stats():
    """What is actually in the knowledge base, per collection.

    `points_count` is Qdrant's own number; `documents` counts the distinct source
    files behind those points, which is what the dashboard means by a document.
    A collection Qdrant does not have comes back status "missing" rather than
    being dropped, so a configured-but-empty one still shows up.
    """
    qdrant = get_qdrant()
    try:
        collections, total_chunks, total_documents = [], 0, 0
        # One video spans four collections (frames, on-screen text, chart
        # descriptions, transcript); the headline counts it once.
        videos = set()
        for name in await _collection_names(qdrant):
            info = await qdrant.collection_info(name)
            if info is None:
                collections.append({"name": name, "points_count": 0,
                                    "documents": 0, "status": "missing"})
                continue
            points = info.get("points_count") or 0
            grouped = await _documents_in(qdrant, name)
            collections.append({"name": name, "points_count": points,
                                "documents": len(grouped),
                                "status": info.get("status") or "unknown"})
            total_chunks += points
            for document in grouped:
                if document.get("video_id"):
                    videos.add(document["video_id"])
                else:
                    total_documents += 1
    # httpx.HTTPError, not OSError: a refused connection arrives as
    # httpx.ConnectError, which does not inherit from OSError.
    except (QdrantError, httpx.HTTPError) as exc:
        logger.warning("knowledge base stats unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Vector store unavailable") from exc

    try:
        structured_tables = len(await get_db().list_tables())
    except DBError as exc:
        # The vector store answered; a database that has not come up yet should
        # not blank the whole dashboard.
        logger.warning("structured table count unavailable: %s", exc)
        structured_tables = 0

    return {"collections": collections,
            "total_documents": total_documents + len(videos),
            "total_videos": len(videos),
            "total_chunks": total_chunks,
            "postgresql": {"structured_tables": structured_tables}}


@router.get("/documents")
async def list_documents(
    collection: Optional[str] = Query(None, description="All collections when omitted"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=200),
):
    """Source files in the knowledge base — documents, not chunks."""
    qdrant = get_qdrant()
    try:
        names = [collection] if collection else await _collection_names(qdrant)
        documents = [doc for name in names for doc in await _documents_in(qdrant, name)]
    except (QdrantError, httpx.HTTPError) as exc:
        logger.warning("document listing unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Vector store unavailable") from exc

    start = (page - 1) * per_page
    return {
        "collection": collection,
        "documents": documents[start:start + per_page],
        "total": len(documents),
        "page": page,
        "per_page": per_page,
        "pages": (len(documents) + per_page - 1) // per_page,
    }


@router.get("/documents/{source_filename:path}")
async def get_document(
    source_filename: str,
    collection: Optional[str] = Query(None, description="All collections when omitted"),
):
    """Every chunk of one source file, in chunk order.

    `:path` because a scraped source filename is a relative path —
    "reference/functions_ta_sma.md" — and would otherwise stop at the slash.
    """
    qdrant = get_qdrant()
    try:
        names = [collection] if collection else await _collection_names(qdrant)
        chunks = []
        for name in names:
            # expand_by_source() is the retriever's own source lookup: it queries
            # both payload shapes, merges by point id and orders by chunk_index,
            # which is exactly this listing. One ordering for both, not two.
            for point in await qdrant.expand_by_source(name, [source_filename]):
                payload = point.get("payload") or {}
                content = str(payload.get("content") or "")
                chunks.append({
                    "id": point["id"],
                    "collection": point.get("collection", name),
                    "chunk_index": payload.get("chunk_index"),
                    "content": content[:CONTENT_PREVIEW],
                    "truncated": len(content) > CONTENT_PREVIEW,
                    **_source_fields(payload),
                })
    except (QdrantError, httpx.HTTPError) as exc:
        logger.warning("chunk listing unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Vector store unavailable") from exc

    if not chunks:
        raise HTTPException(status_code=404,
                            detail="No such document: %s" % source_filename)

    # Already ordered per collection; re-sort only matters when several hold the
    # same filename. chunk_index() is the shared rule — a missing index sorts
    # first, the same way the retriever expands it.
    chunks.sort(key=lambda c: chunk_index(c))
    return {"source": source_filename, "chunk_count": len(chunks), "chunks": chunks}


@router.delete("/documents/{source_filename:path}")
async def delete_document(
    source_filename: str,
    collection: Optional[str] = Query(None, description="All collections when omitted"),
):
    """Remove one source file from the knowledge base — every chunk of it.

    `:path` and the collection default match get_document, so the panel deletes
    exactly the document it just listed.

    Two things this does not do the obvious way:

    * It looks the file up before deleting. delete_by_source() reports how many
      points it removed, but 0 removed and "no such document" are the same
      number, and the panel has to tell a stale row from a failed delete.
    * A structured upload is more than its chunks — its rows are PostgreSQL
      tables with a catalog row pointing at them. Deleting the summary chunk
      alone would leave those tables behind: still queryable by text-to-SQL,
      no longer visible anywhere in this panel. Those files go out through
      structured.delete_dataset(), which drops all three.
    """
    # Imported here rather than at module scope: this is the one endpoint that
    # writes through the pipelines, and pipelines.structured pulls in the whole
    # ingestion module behind it.
    from pipelines.ingestion import delete_by_source
    from pipelines.structured import SOURCE_TYPE as STRUCTURED, delete_dataset

    qdrant = get_qdrant()
    try:
        names = [collection] if collection else await _collection_names(qdrant)
        found = {name: await qdrant.expand_by_source(name, [source_filename])
                 for name in names}
    except (QdrantError, httpx.HTTPError) as exc:
        logger.warning("delete lookup unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Vector store unavailable") from exc

    if not any(found.values()):
        raise HTTPException(status_code=404,
                            detail="No such document: %s" % source_filename)

    structured = any((point.get("payload") or {}).get("source_type") == STRUCTURED
                     for points in found.values() for point in points)

    removed, cleared, tables = 0, [], []
    try:
        for name, points in found.items():
            if not points:
                continue
            count = await delete_by_source(qdrant, name, source_filename)
            removed += count
            cleared.append(name)
        if structured:
            # Its Qdrant summary is already gone; this drops the tables and the
            # catalog row, and reports 0 chunks removed for exactly that reason.
            dataset = await delete_dataset(source_filename, db=get_db(), qdrant=qdrant)
            tables = dataset["tables_dropped"]
    except (QdrantError, httpx.HTTPError) as exc:
        # Part of the file may already be gone. Say so rather than implying the
        # whole delete failed — the panel's next refresh shows what survived.
        logger.warning("delete of %s failed after %d chunk(s): %s",
                       source_filename, removed, exc)
        raise HTTPException(
            status_code=503,
            detail="Vector store unavailable — %d chunk(s) were removed before it "
                   "stopped responding" % removed) from exc
    except DBError as exc:
        logger.warning("dropped chunks for %s but not its tables: %s",
                       source_filename, exc)
        raise HTTPException(
            status_code=503,
            detail="Removed %d chunk(s), but the database is unavailable so its "
                   "data tables are still there" % removed) from exc

    # P123: without this the file's ingestion jobs keep saying status=done after
    # its chunks are gone. Degrade, never break — the delete already happened,
    # so a job store that will not answer is logged, not turned into a 503.
    from pipelines.ingestion import get_store
    jobs_marked = 0
    for name in cleared:
        try:
            jobs_marked += await get_store().mark_deleted(source_filename, name)
        except Exception as exc:
            logger.warning("deleted %s but could not mark its jobs in %s: %s",
                           source_filename, name, exc)

    logger.info("deleted %s: %d chunk(s) from %s, %d table(s), %d job(s) marked deleted",
                source_filename, removed, ", ".join(cleared) or "nowhere", len(tables),
                jobs_marked)
    return {"source": source_filename, "chunks_removed": removed,
            "collections": cleared, "tables_dropped": tables}


@router.post("/reindex")
async def reindex_collection(collection: str = Query(config.COLLECTION_DOCUMENTS)):
    """Reindex all documents in a collection (e.g. after changing embedding model)."""
    # TODO: Implement reindexing
    return {"status": "not_implemented", "collection": collection}


# ── storage monitor (P160) ──────────────────────────────
# ops/storage_monitor.sh runs hourly on the host, measures every store (host disk,
# sandbox tmpfs, Postgres, Qdrant, Open WebUI data, docker logs and build cache)
# and writes ONE row per run. Nothing here measures anything: this container
# cannot see the host disk or the sandbox's tmpfs, which is the whole reason the
# collector is a host script. This endpoint only grades what it wrote.

# Defaults for a percentage probe that names no thresholds of its own. A probe may
# carry warn_pct / alert_pct (the host disk uses 75/90) and warn_bytes / alert_bytes
# (Postgres, Qdrant, the video work dir: stores with no capacity of their own, on
# which a percentage could never fire). The script is where probes are defined, so
# it is also where their own limits live — one place to touch when adding a store.
STORAGE_WARN_PCT = 70
STORAGE_ALERT_PCT = 80
# The timer fires hourly. Two missed runs means it is dead or failing, and a dead
# timer is worse than a full disk: the last report keeps saying "12% used" forever.
STORAGE_STALE_AFTER = timedelta(hours=2)

_STATUS_RANK = {"ok": 0, "warn": 1, "alert": 2, "error": 3}


def _storage_now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonb(value):
    """asyncpg returns jsonb as text unless a codec is registered; db_service registers none."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _at_least(value, limit) -> bool:
    return value is not None and limit is not None and value >= limit


def _probe_status(probe: dict) -> str:
    """Worst of the percentage grade and the absolute-bytes grade."""
    if probe.get("error"):
        # The script records a probe it could not measure instead of dropping
        # it — an absent row would read as "nothing to worry about".
        return "error"
    pct, used = probe.get("pct"), probe.get("used_bytes")
    warn_pct = probe.get("warn_pct", STORAGE_WARN_PCT)
    alert_pct = probe.get("alert_pct", STORAGE_ALERT_PCT)
    if _at_least(pct, alert_pct) or _at_least(used, probe.get("alert_bytes")):
        return "alert"
    if _at_least(pct, warn_pct) or _at_least(used, probe.get("warn_bytes")):
        return "warn"
    return "ok"


def _storage_thresholds() -> dict:
    return {"warn_pct": STORAGE_WARN_PCT, "alert_pct": STORAGE_ALERT_PCT,
            "stale_after_seconds": int(STORAGE_STALE_AFTER.total_seconds())}


def storage_report(row, now: datetime) -> dict:
    """One ops_storage_reports row (or None) -> what the panel renders.

    `alert` is the banner condition: any probe at its alert level, any probe
    that failed, or no report newer than STORAGE_STALE_AFTER.
    """
    if not row:
        return {"stale": True, "alert": True, "worst": "error", "reported_at": None,
                "age_seconds": None, "probes": [], "cleanup": None,
                "thresholds": _storage_thresholds()}

    reported_at = row.get("reported_at")
    if isinstance(reported_at, str):
        reported_at = datetime.fromisoformat(reported_at)
    if reported_at is not None and reported_at.tzinfo is None:
        reported_at = reported_at.replace(tzinfo=timezone.utc)   # timestamptz is UTC
    age = (now - reported_at) if reported_at else None
    stale = age is None or age > STORAGE_STALE_AFTER

    probes = [dict(p, status=_probe_status(p)) for p in (_jsonb(row.get("probes")) or [])]
    worst = max((p["status"] for p in probes), key=_STATUS_RANK.get, default="ok")
    return {
        "stale": stale,
        "alert": stale or _STATUS_RANK[worst] >= _STATUS_RANK["alert"],
        "worst": worst,
        "reported_at": reported_at.isoformat() if reported_at else None,
        "age_seconds": int(age.total_seconds()) if age is not None else None,
        "probes": probes,
        "cleanup": _jsonb(row.get("cleanup")),
        "thresholds": _storage_thresholds(),
    }


@router.get("/storage")
async def get_storage():
    """Latest storage report, graded.

    503 when the database itself is down — a different failure from "the monitor
    never ran" (stale), and the panel must render neither as a healthy report.
    """
    db = get_db()
    try:
        # The monitor creates the table on its first run. Until then this is
        # "no report yet" (stale), not a 500 on an undefined table.
        present = await db.query(
            "SELECT to_regclass('public.ops_storage_reports') IS NOT NULL AS present")
        rows = []
        if present and present[0].get("present"):
            rows = await db.query(
                "SELECT id, reported_at, probes, cleanup FROM ops_storage_reports"
                " ORDER BY reported_at DESC LIMIT 1")
    except DBError as exc:
        logger.warning("storage report unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Storage report unavailable") from exc
    return storage_report(rows[0] if rows else None, _storage_now())


# ── backup status (see README.md, Backups) ──────────────────────────
#
# The same pattern as storage: ops/backup.sh writes one
# ops_backup_reports row per run (running -> success/failed), this endpoint
# reads and grades it, the panel's Backups section renders it. The agreed
# warning: a last successful backup older than 26 h means nightly runs have
# stopped and nobody noticed.

BACKUP_STALE_AFTER = timedelta(hours=26)


def _ts(value) -> Optional[datetime]:
    """Row timestamp -> aware UTC datetime (or None). asyncpg hands timestamptz
    back as datetime; tests and json paths may hand us iso text."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)   # timestamptz is UTC
    return value


def _backup_run(row) -> dict:
    started = _ts(row.get("started_at"))
    finished = _ts(row.get("finished_at"))
    error = row.get("error")
    if isinstance(error, str):
        error = error[:300]   # panel-safe size; the script writes short step names
    return {
        "started_at": started.isoformat() if started else None,
        "finished_at": finished.isoformat() if finished else None,
        "status": row.get("status") or ("success" if row.get("ok") else "failed"),
        "ok": bool(row.get("ok")),
        "snapshot_id": row.get("snapshot_id"),
        "bytes_added": row.get("bytes_added"),
        # Written by the script as a plain failing-step description — never
        # secrets — so it is safe to show in the panel as-is.
        "error": error,
        "detail": _jsonb(row.get("detail")),
    }


def backup_report(latest, last_success, now: datetime) -> dict:
    """Latest run + latest success (or None, None) -> what the panel renders.

    `alert` is the banner condition: no successful backup at all, the last
    successful backup older than BACKUP_STALE_AFTER (26 h), or the most recent
    run failed.
    """
    run = _backup_run(latest) if latest else None
    success = _backup_run(last_success) if last_success else None

    success_at = _ts(last_success.get("started_at")) if last_success else None
    success_age = (now - success_at) if success_at else None
    stale = success_age is None or success_age > BACKUP_STALE_AFTER

    status = run["status"] if run else "never"
    return {
        "status": status,
        "stale": stale,
        "alert": stale or status == "failed",
        "last_success_at": success_at.isoformat() if success_at else None,
        "success_age_seconds": int(success_age.total_seconds()) if success_age is not None else None,
        "latest_run": run,
        "last_success": success,
        "thresholds": {"stale_after_seconds": int(BACKUP_STALE_AFTER.total_seconds())},
    }


@router.get("/backup")
async def get_backup():
    """Latest backup run and latest success, graded.

    503 when the database is down — a different failure from "the backup has
    never run" (stale), and the panel must render neither as healthy.
    """
    db = get_db()
    try:
        # The backup script creates the table on its first run. Until then this
        # is "never backed up" (stale), not a 500 on an undefined table.
        present = await db.query(
            "SELECT to_regclass('public.ops_backup_reports') IS NOT NULL AS present")
        latest, success = [], []
        if present and present[0].get("present"):
            latest = await db.query(
                "SELECT id, started_at, finished_at, ok, status, snapshot_id,"
                " bytes_added, error, detail FROM ops_backup_reports"
                " ORDER BY started_at DESC LIMIT 1")
            success = await db.query(
                "SELECT id, started_at, finished_at, ok, status, snapshot_id,"
                " bytes_added, error, detail FROM ops_backup_reports"
                " WHERE ok ORDER BY started_at DESC LIMIT 1")
    except DBError as exc:
        logger.warning("backup report unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Backup report unavailable") from exc
    return backup_report(latest[0] if latest else None,
                         success[0] if success else None,
                         _storage_now())
