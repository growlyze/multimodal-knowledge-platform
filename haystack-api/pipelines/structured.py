"""
Structured data pipeline — CSV / TSV / XLSX / JSON / Parquet.

    file bytes -> pandas -> PostgreSQL table(s)
                         -> metadata summary -> embedded -> Qdrant

Tabular data is never chunked and embedded row by row: 211K rows make 211K
useless vectors. One summary chunk per table goes into `documents`, so
"do we have ACME data from 2020?" retrieves the summary, and the summary names
the PostgreSQL table that answers "what was the close on March 15?".

Entry points:
    ingest_structured(filename, file_bytes)  -> load + index, returns table names
    run_structured_job(job_id, ...)          -> the same, reported as an /ingest job
    delete_dataset(filename)                 -> drops the tables AND the Qdrant chunk
"""

import io
import json
import re
import logging
import time
from datetime import datetime, timezone

import config
from pipelines import dedup, ingestion
from pipelines.ingestion import build_payload, delete_by_source
from services.db_service import DBError, get_db, table_name_for

logger = logging.getLogger("haystack-api.structured")

SOURCE_TYPE = "structured_data"

# Extension -> canonical format name. Everything else is not our problem.
FORMATS = {
    "csv": "csv", "txt": "csv",
    "tsv": "tsv", "tab": "tsv",
    "xlsx": "xlsx", "xlsm": "xlsx", "xls": "xlsx",
    "json": "json", "jsonl": "jsonl", "ndjson": "jsonl",
    "parquet": "parquet", "pq": "parquet",
}

DATE_HINTS = ("time", "date", "timestamp", "datetime", "session")
SYMBOL_HINTS = ("symbol", "ticker", "instrument", "contract", "pair")

MAX_STAT_COLUMNS = 10       # a summary nobody reads past
MAX_SAMPLE_COLUMNS = 15
MAX_SYMBOLS = 20
DATE_SNIFF_ROWS = 1000
# "...-0400" / "...-04:00" at the end of a timestamp: it states its own UTC offset.
_OFFSET_TAIL = re.compile(r"[+-]\d{2}:?\d{2}$")


class StructuredError(ValueError):
    """The file could not be parsed as tabular data. Message goes to the user."""


# ── parsing ─────────────────────────────────────────────

def unwrap_compressed(filename: str, file_bytes: bytes) -> tuple:
    """("data.csv.gz", bytes) -> ("data.csv", decompressed bytes).

    A single-stream .gz/.bz2 holds exactly one file, so the data pipeline
    treats it as the file inside it: same format detection, same `data_` table
    naming, same replace-on-reupload identity as the uncompressed upload.
    Containers (.zip/.tar/...) are not data files and are refused here by name.
    """
    from extractors import archive

    name = archive.basename(filename)
    suffix = archive.compression_suffix(name)
    if suffix is None:
        return filename, file_bytes
    if suffix not in archive.SINGLE_STREAM_SUFFIXES:
        raise StructuredError(
            "%s is a compressed archive, not a data file - use /ingest/archive "
            "for it, or upload the data file inside it directly" % name)
    try:
        inner, blob = archive.decompress_single(name, file_bytes)
    except archive.ArchiveError as exc:
        raise StructuredError(str(exc)) from exc
    return inner, blob


def detect_format(filename: str) -> str:
    from extractors import archive

    name = archive.basename(filename)
    base, comp = archive.split_compression(name)
    if comp is not None and comp not in archive.SINGLE_STREAM_SUFFIXES:
        raise StructuredError(
            "%s is a compressed archive, not a data file - use /ingest/archive "
            "for it, or upload the data file inside it directly" % name)
    if comp is not None and "." not in base:
        raise StructuredError(
            "%s is compressed but its name does not say what is inside - "
            "rename it (e.g. %s.csv.gz) or upload the file inside it directly"
            % (name, base))
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    fmt = FORMATS.get(ext)
    if fmt is None:
        raise StructuredError(
            "%s is not a supported data file (expected one of: %s)"
            % (filename, ", ".join(sorted(set(FORMATS))))
        )
    return fmt


def parse(filename: str, file_bytes: bytes) -> tuple:
    """bytes -> (format, {sheet_name_or_None: DataFrame}).

    Raises StructuredError with a message worth showing a user. Ragged rows are
    skipped rather than fatal — one bad line in 211K should not lose the file.
    """
    if not file_bytes:
        raise StructuredError("%s is empty" % filename)

    filename, file_bytes = unwrap_compressed(filename, file_bytes)
    fmt = detect_format(filename)
    try:
        frames = _PARSERS[fmt](file_bytes)
    except StructuredError:
        raise
    except Exception as exc:
        raise StructuredError("could not parse %s as %s — %s: %s"
                              % (filename, fmt, type(exc).__name__, exc)) from exc

    frames = {name: df for name, df in frames.items()
              if df is not None and len(df.columns) and len(df)}
    if not frames:
        raise StructuredError("%s parsed to zero rows — nothing to load" % filename)

    for df in frames.values():
        coerce_dates(df)
    return fmt, frames


def _read_delimited(file_bytes: bytes, sep: str) -> dict:
    import pandas as pd

    df = pd.read_csv(
        io.BytesIO(file_bytes), sep=sep,
        encoding="utf-8", encoding_errors="replace",
        on_bad_lines="skip",        # ragged rows are dropped, not fatal
    )
    return {None: df}


def _read_excel(file_bytes: bytes) -> dict:
    import pandas as pd
    # sheet_name=None -> every sheet, each becoming its own PostgreSQL table.
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)


def _read_json(file_bytes: bytes) -> dict:
    import pandas as pd

    data = json.loads(file_bytes.decode("utf-8", "replace"))
    if isinstance(data, list):
        return {None: pd.json_normalize(data)}
    if isinstance(data, dict):
        # {"rows": [...]} is the common wrapper — unwrap the single list value.
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1 and all(isinstance(i, dict) for i in lists[0]):
            return {None: pd.json_normalize(lists[0])}
        return {None: pd.json_normalize(data)}
    raise StructuredError("JSON root is %s, expected an object or array"
                          % type(data).__name__)


def _read_jsonl(file_bytes: bytes) -> dict:
    import pandas as pd
    return {None: pd.read_json(io.BytesIO(file_bytes), lines=True)}


def _read_parquet(file_bytes: bytes) -> dict:
    import pandas as pd
    return {None: pd.read_parquet(io.BytesIO(file_bytes))}


_PARSERS = {
    "csv": lambda b: _read_delimited(b, ","),
    "tsv": lambda b: _read_delimited(b, "\t"),
    "xlsx": _read_excel,
    "json": _read_json,
    "jsonl": _read_jsonl,
    "parquet": _read_parquet,
}


def coerce_dates(df) -> list:
    """Turn date-ish text columns into real datetimes, in place.

    Worth the pass: it makes `Time` a PostgreSQL TIMESTAMP, so "the close on
    March 15" is a WHERE clause instead of string matching, and it gives the
    summary a real date range.

    ponytail: name hint + a 1000-row sniff, not full type inference. A date
    column named `col_7` stays TEXT.
    """
    import pandas as pd
    from pandas.api import types as pt

    converted = []
    for col in df.columns:
        dtype = df[col].dtype
        # Anything already typed is left alone. Tested against dtype rather than
        # `is_object_dtype` because pandas 3 gives text columns a `str` dtype.
        if (pt.is_datetime64_any_dtype(dtype) or pt.is_numeric_dtype(dtype)
                or pt.is_bool_dtype(dtype)):
            continue
        if not any(h in str(col).lower() for h in DATE_HINTS):
            continue
        sniff, _ = parse_dates(df[col].head(DATE_SNIFF_ROWS))
        if sniff.notna().mean() < 0.9:
            continue
        parsed, mixed = parse_dates(df[col])
        if mixed:
            # New York offsets on both sides of a DST change (-0400 and -0500,
            # every one of the client's 1-minute files). Kept as the file's own text
            # on purpose: a TIMESTAMPTZ comes back from PostgreSQL in UTC, and
            # load_dataset() reads the -04/-05 offsets to know the data is New
            # York time (P187) - converting here would erase exactly that.
            # date_range() still parses it, so the summary keeps its years.
            continue
        if parsed.notna().mean() >= 0.9:
            df[col] = parsed
            converted.append(col)
    return converted


def parse_dates(values):
    """values -> (datetimes, mixed_offsets). Never raises on mixed offsets.

    Timestamps carrying two different UTC offsets in one column cannot share a
    single fixed-offset dtype. pandas 3 raises "Mixed timezones detected" for
    them - out of coerce_dates, which failed the whole upload of every file that
    spans a DST change - and pandas 2.2 returns an object column of mixed
    datetimes that nothing downstream recognises as a date. Both are retried
    with utc=True: one instant per row, comparable across the change.
    """
    import warnings

    import pandas as pd
    from pandas.api import types as pt

    # Offsets checked first, from the text: pandas 2.2 spends 26 s on 1.2M
    # mixed-offset rows building the object column it then gives up on, against
    # 7 s for the utc=True parse (a client's 1.2M-row CSV part 1, 2026-10-22).
    text = values.dropna().astype(str).str.strip()
    if len(text) and text.head(DATE_SNIFF_ROWS).str.contains(_OFFSET_TAIL).all():
        tails = text.str.slice(-6).str.replace(":", "", regex=False).str.slice(-5)
        if tails.nunique() > 1:
            return pd.to_datetime(values, errors="coerce", utc=True), True

    try:
        with warnings.catch_warnings():
            # pandas 2.2's "mixed time zones will raise in a future version".
            warnings.simplefilter("ignore", FutureWarning)
            parsed = pd.to_datetime(values, errors="coerce")
        if pt.is_datetime64_any_dtype(parsed.dtype):
            return parsed, False
    except ValueError:
        pass
    return pd.to_datetime(values, errors="coerce", utc=True), True


# ── metadata summary ────────────────────────────────────

def _first_matching(columns, hints):
    for col in columns:
        if any(h in str(col).lower() for h in hints):
            return col
    return None


def date_range(df) -> dict:
    """{"column", "start", "end", "years"} for the first datetime column, or {}."""
    from pandas.api import types as pt

    stamps = [c for c in df.columns if pt.is_datetime64_any_dtype(df[c].dtype)]
    if stamps:
        col = _first_matching(stamps, DATE_HINTS) or stamps[0]
        series = df[col].dropna()
    else:
        # A date column coerce_dates left as text because its rows carry mixed
        # offsets. Without this the summary had no range and no "Years
        # covered", so a search for "ACME 2018" scored all three parts of a
        # 3-part series the same and load_dataset() took the newest one.
        col = _first_matching([c for c in df.columns if not pt.is_numeric_dtype(df[c].dtype)],
                              DATE_HINTS)
        if col is None:
            return {}
        parsed, mixed = parse_dates(df[col])
        if not mixed or parsed.notna().mean() < 0.9:
            return {}
        series = parsed.dropna()
    if series.empty:
        return {}
    start, end = series.min(), series.max()
    return {
        "column": str(col),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "years": list(range(int(start.year), int(end.year) + 1)),
    }


def symbols(df) -> list:
    """Distinct values of a symbol/ticker column, if there is one."""
    col = _first_matching(df.columns, SYMBOL_HINTS)
    if col is None:
        return []
    values = df[col].dropna().unique()
    if len(values) > MAX_SYMBOLS:
        return []                      # a high-cardinality column is not a symbol
    return [str(v) for v in values]


def build_summary(filename: str, fmt: str, table: str, df, columns: list,
                  sheet: str = None, dates: dict = None, tickers: list = None) -> str:
    """The text that gets embedded. Everything a router needs to decide
    "yes, that data exists, and here is the table that holds it"."""
    from pandas.api import types as pt

    dates = dates if dates is not None else date_range(df)
    tickers = tickers if tickers is not None else symbols(df)

    lines = [
        "Structured data file: %s" % filename,
        "Format: %s | PostgreSQL table: %s" % (fmt.upper(), table),
    ]
    if sheet:
        lines.append("Sheet: %s" % sheet)
    lines.append("Rows: %s | Columns: %d" % (format(len(df), ","), len(columns)))
    lines.append("Columns: %s" % ", ".join(
        "%s (%s)" % (c["name"], c["type"]) for c in columns))

    if tickers:
        lines.append("Symbols / instruments: %s" % ", ".join(tickers))
    if dates:
        lines.append("Date range (%s): %s to %s" % (dates["column"], dates["start"], dates["end"]))
        # Spelled out so BM25 matches a bare year in the question.
        lines.append("Years covered: %s" % ", ".join(str(y) for y in dates["years"]))

    numeric = [c for c in df.columns if pt.is_numeric_dtype(df[c].dtype)
               and not pt.is_bool_dtype(df[c].dtype)][:MAX_STAT_COLUMNS]
    if numeric:
        lines.append("Numeric ranges:")
        for col in numeric:
            series = df[col].dropna()
            if series.empty:
                continue
            lines.append("  %s: min %s, max %s, mean %s"
                         % (col, _num(series.min()), _num(series.max()), _num(series.mean())))

    sample = df.iloc[:5, :MAX_SAMPLE_COLUMNS]
    lines += ["", "Sample rows (first 5 of %s):" % format(len(df), ","),
              sample.to_string(index=False, max_colwidth=28)]
    lines += ["",
              'Row-level questions about this data are answered by querying the '
              'PostgreSQL table "%s" with SQL, or with the Code Interpreter '
              '(pandas). This summary is an index entry, not the data itself.' % table]
    return "\n".join(lines)


def _num(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return format(round(value, 4), ",g")


# ── the pipeline ────────────────────────────────────────

async def ingest_structured(filename: str, file_bytes: bytes, metadata: dict = None,
                            db=None) -> dict:
    """Load a data file into PostgreSQL and index its summary in Qdrant.

    Never raises — the caller is a background task and a bad upload should come
    back as a message, not a traceback. Re-uploading the same filename with
    changed bytes replaces both the tables and the Qdrant chunk instead of
    duplicating them; byte-identical bytes are skipped (see _skip_if_loaded)
    unless metadata carries ingestion.FORCE_KEY.
    """
    started = time.perf_counter()
    db = db or get_db()
    result = {"filename": filename, "source_type": SOURCE_TYPE, "tables": []}
    # Copied, and the force flag popped, before anything is stored: the flag is
    # an instruction for this one upload, and persisted into the catalog row or
    # the summary payload it would read as a property of the data forever.
    metadata = dict(metadata or {})
    force = bool(metadata.pop(ingestion.FORCE_KEY, False))

    try:
        # data.csv.gz is data.csv: same tables, same replace-on-reupload
        # identity as the uncompressed upload (B1). Hashed after unwrapping for
        # the same reason - gzip stamps a header mtime, so two compressions of
        # one CSV differ in bytes while holding identical data.
        filename, file_bytes = unwrap_compressed(filename, file_bytes)
        result["filename"] = filename
        if not ingestion.INGEST_DEDUP:
            return await _load(filename, file_bytes, metadata, db, result, started)

        # Stored even on a forced upload, so the next ordinary one can match it.
        metadata["file_sha256"] = dedup.bytes_hash(file_bytes)
        metadata["hash_version"] = dedup.HASH_VERSION
        # Held through the load, not just the lookup: two uploads of one file in
        # the same second (two chat tabs, or the filter's re-post racing the
        # first upload) must not both find nothing and both create tables.
        async with dedup.hold(metadata["file_sha256"]):
            if not force:
                skipped = await _skip_if_loaded(filename, metadata["file_sha256"],
                                                db, result, started)
                if skipped:
                    return skipped
            return await _load(filename, file_bytes, metadata, db, result, started)

    except (StructuredError, DBError) as exc:
        logger.warning("structured ingestion failed for %s: %s", filename, exc)
        error = str(exc)
    except Exception as exc:
        logger.exception("structured ingestion crashed for %s", filename)
        error = "%s: %s" % (type(exc).__name__, exc)
    # `tables` is kept on the failure path too: the rows may already be loaded
    # and the caller needs to know that before retrying.
    result.update(status="failed", error=error,
                  elapsed_ms=round((time.perf_counter() - started) * 1000))
    return result


async def _load(filename: str, file_bytes: bytes, metadata: dict, db, result: dict,
                started: float) -> dict:
    """Parse, replace this filename's tables, record them, index the summaries.

    Raises; ingest_structured() turns any failure into a failed result.
    """
    fmt, frames = parse(filename, file_bytes)
    result["format"] = fmt

    # Replace, don't duplicate: drop whatever this filename produced last time.
    result["replaced_tables"] = await db.drop_by_filename(filename)

    now, used = datetime.now(timezone.utc), set()
    for index, (sheet, df) in enumerate(frames.items()):
        table = table_name_for(filename, sheet, now=now)
        if table in used:
            # Sheets 'Sales' and 'sales' sanitize to the same name — the second
            # would silently overwrite the first without this.
            table = table_name_for(filename, "%s_%d" % (sheet or "sheet", index), now=now)
        used.add(table)
        created = await db.create_table_from_df(table, df)
        # Describe the table, not the file: the summary is what an LLM reads
        # before writing SQL, so every name in it must be the SQL name.
        df = df.rename(columns={c["source_name"]: c["name"] for c in created["columns"]})
        dates, tickers = date_range(df), symbols(df)
        summary = build_summary(filename, fmt, table, df, created["columns"],
                                sheet=sheet, dates=dates, tickers=tickers)
        entry = {
            "table_name": table,
            "sheet": sheet,
            "row_count": created["row_count"],
            "columns": created["columns"],
            "date_range": dates,
            "symbols": tickers,
            "summary": summary,
        }
        # metadata carries file_sha256/hash_version when dedup is on: the
        # catalog row is where the next upload of these bytes looks for them.
        await db.record(table, filename, fmt, created["row_count"],
                        created["columns"], summary, sheet=sheet,
                        metadata={"date_range": dates, "symbols": tickers, **metadata})
        result["tables"].append(entry)

    result["indexed"] = await index_summaries(filename, fmt, result["tables"], metadata)
    result.update(
        status="done",
        table_name=result["tables"][0]["table_name"],
        row_count=sum(t["row_count"] for t in result["tables"]),
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
    logger.info("%s -> %d table(s), %d rows in %dms", filename,
                len(result["tables"]), result["row_count"], result["elapsed_ms"])
    return result


async def _skip_if_loaded(filename: str, file_sha256: str, db, result: dict,
                          started: float) -> dict:
    """The skip result when these exact bytes are already loaded, else None.

    Why (measured live 2026-10-26): the rag filter re-POSTs every attachment on
    every chat turn, and each re-post dropped the tables, recreated them under a
    new timestamped name and re-embedded the summary - a query written against
    turn 3's table name met a table turn 4 had dropped. And the same file
    uploaded under a second name ("DL_ES1_test.csv" / "CME_MINI_DL_ES1!, 5
    (2).csv") became a second set of tables and a second summary competing in
    search, so load_dataset() picked between two copies of one file.

    Same filename -> "unchanged": nothing is touched. Other filename ->
    "duplicate": nothing is created; this name is recorded as an alias on the
    original's catalog rows so /admin/data/search still finds the data by the
    name the user uploaded it under.

    ponytail: an unfiltered catalog scan matched in Python - the catalog holds
    one row per uploaded table (tens), and metadata is JSON in TEXT with no index
    to probe. Becomes a WHERE on a real column if the catalog ever grows.
    """
    rows = [row for row in await db.list_tables()
            if (row.get("metadata") or {}).get("file_sha256") == file_sha256]
    if not rows:
        return None

    own = [row for row in rows if row["filename"] == filename]
    replaced = []
    if own:
        kind, source, matched = "unchanged", filename, own
    else:
        kind = "duplicate"
        # sorted() for the reason dedup.find_existing sorts: bytes held under two
        # names must name the same original on every re-post.
        source = sorted({row["filename"] for row in rows})[0]
        matched = [row for row in rows if row["filename"] == source]
        # This name may have held different data before. Its tables and summary
        # describe content it no longer holds, and left behind they would answer
        # questions from a file that is gone.
        replaced = await db.drop_by_filename(filename)
        await _delete_summaries(filename)
        for row in matched:
            meta = dict(row.get("metadata") or {})
            aliases = list(meta.get("aliases") or [])
            if filename in aliases:
                continue        # the filter's every-turn re-post: already recorded
            meta["aliases"] = aliases + [filename]
            # The row's own values passed back whole: record() upserts the
            # complete row, so anything not passed would be blanked.
            await db.record(row["table_name"], row["filename"], row.get("format"),
                            row.get("row_count") or 0, row.get("columns") or [],
                            row.get("summary"), sheet=row.get("sheet"), metadata=meta)

    tables = [{"table_name": row["table_name"], "sheet": row.get("sheet"),
               "row_count": row.get("row_count") or 0,
               "columns": row.get("columns") or [],
               "date_range": (row.get("metadata") or {}).get("date_range") or {},
               "symbols": (row.get("metadata") or {}).get("symbols") or []}
              for row in matched]
    # The text pipeline's skip shape (ingestion._skip_if_indexed), so the panel
    # reads one set of keys whichever pipeline skipped the file.
    report = {"skipped": kind, "matched_on": "file_sha256",
              "existing_chunks": len(matched)}
    if kind == "duplicate":
        report["duplicate_of"] = source
    # status "done" + indexed 0 is what archive_ingestion and run_structured_job
    # count as success: a skipped re-post is not a failed upload.
    result.update(
        status="done", format=matched[0].get("format"), tables=tables,
        table_name=tables[0]["table_name"],
        row_count=sum(t["row_count"] for t in tables),
        indexed=0, chunks_created=0, collection=config.COLLECTION_DOCUMENTS,
        replaced_tables=replaced, quality=report, **report,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
    logger.info("skipped %s: %s%s (file_sha256) -> %s", filename, kind,
                " of %s" % source if kind == "duplicate" else "",
                ", ".join(t["table_name"] for t in tables))
    return result


async def _delete_summaries(filename: str) -> int:
    from services.qdrant_service import get_qdrant

    # Scoped to our source_type, as index_summaries scopes it, so a same-named
    # document's text chunks survive.
    return await delete_by_source(get_qdrant(), config.COLLECTION_DOCUMENTS, filename,
                                  {"source_type": SOURCE_TYPE})


async def index_summaries(filename: str, fmt: str, entries: list,
                          metadata: dict = None) -> int:
    """One Qdrant point per table. Replaces any previous points for this file."""
    if not entries:
        return 0

    from services.embedding_service import create_embedding_provider
    from services.qdrant_service import get_qdrant

    embedder = create_embedding_provider()
    qdrant = get_qdrant()
    collection = config.COLLECTION_DOCUMENTS

    await qdrant.ensure_collection(collection, embedder.dimensions)
    # Scoped to our source_type so a same-named document's chunks survive.
    await delete_by_source(qdrant, collection, filename, {"source_type": SOURCE_TYPE})

    vectors = await embedder.embed([e["summary"] for e in entries], input_type="document")
    ingested_at = datetime.now(timezone.utc).isoformat()
    ids, payloads = [], []
    for index, entry in enumerate(entries):
        ids.append(summary_point_id(filename, entry["sheet"]))
        # One summary covers the whole table, so the citation names every row.
        rows = entry["row_count"]
        payloads.append(build_payload(entry["summary"], {
            "source": filename,
            "source_type": SOURCE_TYPE,
            "row_range": "1-%d" % rows if rows else None,
            "chunk_index": index,
            "table_name": entry["table_name"],
            "sheet": entry["sheet"],
            "format": fmt,
            "columns": [c["name"] for c in entry["columns"]],
            "row_count": rows,
            "date_range": entry["date_range"],
            "symbols": entry["symbols"],
            **(metadata or {}),
        }, collection, ingested_at))
    await qdrant.upsert(collection, ids, vectors, payloads)
    return len(ids)


def summary_point_id(filename: str, sheet: str = None) -> str:
    """Deterministic, so a re-upload overwrites the same point."""
    return "structured:%s:%s" % (filename, sheet or "0")


async def delete_dataset(filename: str, db=None, qdrant=None) -> dict:
    """Remove a dataset entirely: PostgreSQL tables, catalog rows, Qdrant chunk.

    `qdrant` is injectable for the same reason `db` is: /admin's delete already
    holds a service instance and hands it down, so one request talks to one
    client rather than reaching for the module-level singleton mid-call.
    """
    from services.qdrant_service import get_qdrant

    db = db or get_db()
    qdrant = qdrant or get_qdrant()
    dropped = await db.drop_by_filename(filename)
    removed = await delete_by_source(qdrant, config.COLLECTION_DOCUMENTS, filename,
                                     {"source_type": SOURCE_TYPE})
    logger.info("deleted dataset %s: %d table(s), %d chunk(s)",
                filename, len(dropped), removed)
    return {"filename": filename, "tables_dropped": dropped, "chunks_removed": removed}


# ── the background job ──────────────────────────────────

async def run_structured_job(job_id: str, filename: str, file_bytes: bytes,
                             metadata: dict = None, db=None) -> dict:
    """The background half of POST /ingest/structured.

    ingest_structured() already swallows its own failures, so nothing here is
    error handling — it is translation. /ingest/status/{id} serves one job shape
    for every source type, and the panel reads `chunks_created` off it, so the
    summary count goes there: one point per table is genuinely what this
    ingestion adds to Qdrant.
    """
    from pipelines.ingestion import get_store

    store = get_store()
    await store.update(job_id, status="processing")
    result = await ingest_structured(filename, file_bytes, metadata, db=db)
    report = job_report(result)

    if result["status"] == "done":
        await store.update(job_id, status="done",
                           chunks_created=result.get("indexed", 0), quality=report)
    else:
        # `report` is kept on the failure path deliberately: a file that loaded
        # three sheets and died on the fourth has real tables in PostgreSQL, and
        # whoever retries needs to see them.
        await store.update(job_id, status="failed",
                           error=result.get("error"), quality=report)
    return result


def job_report(result: dict) -> dict:
    """The compact form of an ingestion result, for the job's `quality` column.

    Deliberately not the whole result: every table carries its full summary
    text, and the job row is read by a status poll, not by a model.
    """
    tables = result.get("tables") or []
    report = {
        "kind": "structured",
        "format": result.get("format"),
        "row_count": sum(t["row_count"] for t in tables),
        "summaries_indexed": result.get("indexed", 0),
        "replaced_tables": result.get("replaced_tables") or [],
        "tables": [{"table_name": t["table_name"], "sheet": t["sheet"],
                    "row_count": t["row_count"],
                    "columns": [c["name"] for c in t["columns"]]}
                   for t in tables],
    }
    # A dedup skip says why nothing was loaded and which file holds the data;
    # without these the job row reads "done, 0 summaries" - a silent no-op.
    for key in ("skipped", "matched_on", "existing_chunks", "duplicate_of"):
        if key in result:
            report[key] = result[key]
    return report
