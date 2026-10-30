"""
PostgreSQL access for structured data. STRUCTURED writes it, ADMIN imports it.

One uploaded data file becomes one real table:

    NQ_2020-03-11.csv  ->  data_nq_2020_03_11_20260826101500

plus a row in `data_catalog` mapping filename -> table_name -> metadata, so the
Qdrant summary chunk and the admin panel can both find the table again.

Shares the database with Open WebUI. Everything we create is prefixed `data_`
and nothing here ever writes outside that prefix — see `_assert_ours`.

Public surface (do not break):
    DBService, get_db(), DBError, sanitize_identifier(), table_name_for()
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

import config

logger = logging.getLogger("haystack-api.db")

TABLE_PREFIX = "data_"
CATALOG_TABLE = "data_catalog"
MAX_IDENT = 63              # PostgreSQL identifier limit
QUERY_ROW_CAP = 1000        # a text-to-SQL answer nobody reads past
QUERY_TIMEOUT_MS = 30_000

CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS data_catalog (
    table_name TEXT PRIMARY KEY,
    filename   TEXT NOT NULL,
    sheet      TEXT,
    format     TEXT,
    row_count  BIGINT,
    col_count  INTEGER,
    columns    TEXT,
    summary    TEXT,
    metadata   TEXT,
    created_at TIMESTAMP DEFAULT NOW()
)
"""
# ponytail: columns/metadata are JSON in TEXT, not JSONB — the catalog holds one
# row per uploaded file, so nothing needs to index inside them. Switch to JSONB
# if the admin panel ever filters on a key.

CATALOG_INDEX = "CREATE INDEX IF NOT EXISTS data_catalog_filename ON data_catalog (filename)"


class DBError(RuntimeError):
    """Postgres is unreachable or rejected a statement. Never swallowed —
    structured ingestion without a database has nowhere to put the rows."""


# ── identifiers ─────────────────────────────────────────

_NOT_IDENT = re.compile(r"[^a-z0-9_]+")


def sanitize_identifier(raw, fallback: str = "col", maxlen: int = MAX_IDENT) -> str:
    """Any string -> a safe lowercase SQL identifier.

    Table and column names cannot be passed as bind parameters, so they are
    interpolated into DDL. This whitelist is the only thing between an uploaded
    filename and the SQL string — keep it a whitelist.
    """
    ident = _NOT_IDENT.sub("_", str(raw).strip().lower()).strip("_")
    if not ident:
        ident = fallback
    elif ident[0].isdigit():
        ident = "%s_%s" % (fallback, ident)
    return ident[:maxlen]


def quote(ident: str) -> str:
    """Double-quote an already-sanitized identifier. Rejects anything that is not."""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", ident or ""):
        raise DBError("refusing to build SQL with identifier %r" % ident)
    return '"%s"' % ident


def table_name_for(filename: str, sheet: str = None, now: datetime = None) -> str:
    """data_{sanitized filename}_{timestamp}, capped at PostgreSQL's 63 chars."""
    stem = filename.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if sheet:
        stem = "%s_%s" % (stem, sheet)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d%H%M%S")
    budget = MAX_IDENT - len(TABLE_PREFIX) - len(stamp) - 1
    return "%s%s_%s" % (TABLE_PREFIX, sanitize_identifier(stem, "file")[:budget], stamp)


def _assert_ours(table: str) -> str:
    """Guard on every write path: we only ever touch our own tables.

    Open WebUI's 44 tables live in this same database. A bug that builds the
    wrong table name must fail loudly here, not drop `user`.
    """
    if not table.startswith(TABLE_PREFIX):
        raise DBError("refusing to touch %r — not a %s* table" % (table, TABLE_PREFIX))
    return table


# ── pandas -> PostgreSQL types ──────────────────────────

def column_definitions(df) -> list:
    """[(original_name, sql_name, sql_type), ...] with duplicate names de-duped."""
    from pandas.api import types as pt

    out, seen = [], set()
    for original in df.columns:
        name = sanitize_identifier(original, "col")
        if name in seen:                       # 'Open' and 'open' collide once lowered
            n = 2
            while "%s_%d" % (name[:MAX_IDENT - 3], n) in seen:
                n += 1
            name = "%s_%d" % (name[:MAX_IDENT - 3], n)
        seen.add(name)

        dtype = df[original].dtype
        if pt.is_bool_dtype(dtype):
            sql_type = "BOOLEAN"
        elif pt.is_integer_dtype(dtype):
            sql_type = "BIGINT"
        elif pt.is_float_dtype(dtype):
            sql_type = "DOUBLE PRECISION"
        elif pt.is_datetime64_any_dtype(dtype):
            sql_type = "TIMESTAMPTZ" if getattr(dtype, "tz", None) else "TIMESTAMP"
        else:
            sql_type = "TEXT"
        out.append((original, name, sql_type))
    return out


def _column_values(series, sql_type: str) -> list:
    """One DataFrame column -> Python natives asyncpg can bind. NaN/NaT -> None.

    Per column rather than per row: pandas already stores columns contiguously,
    so `.tolist()` does the numpy -> Python conversion in one C-level pass.
    """
    import pandas as pd

    isna = list(pd.isna(series))
    if sql_type in ("TIMESTAMP", "TIMESTAMPTZ"):
        values = list(series.dt.to_pydatetime())
    elif sql_type == "TEXT":
        values = [None if bad else str(v) for v, bad in zip(series.tolist(), isna)]
    else:
        values = series.tolist()
    return [None if bad else v for v, bad in zip(values, isna)]


def dataframe_records(df, columns: list):
    """Lazy row iterator. zip() streams, so a 200K-row file is never fully
    materialised as tuples."""
    return zip(*[_column_values(df[original], sql_type)
                 for original, _, sql_type in columns])


# ── read-only query guard ───────────────────────────────

_READ_ONLY_START = ("select", "with", "explain", "show", "table", "values")
# Open WebUI's credential tables. The read-only transaction stops writes; this
# stops an LLM-written query from reading password hashes back into a chat.
_BLOCKED = re.compile(r"\b(auth|oauth_session|api_key)\b", re.IGNORECASE)


def assert_read_only(sql: str) -> str:
    """Reject anything that is not a single read statement.

    ponytail: belt (this check) and braces (a READ ONLY transaction in
    `DBService.query`). Neither stops a SELECT against Open WebUI's other
    tables — the real fix is a PostgreSQL role with SELECT granted only on
    data_*, which needs INFRA to provision it.
    """
    stripped = (sql or "").strip().rstrip(";").strip()
    if not stripped:
        raise DBError("empty query")
    if ";" in stripped:
        raise DBError("only one statement per query")
    if stripped.split(None, 1)[0].lower() not in _READ_ONLY_START:
        raise DBError("only read statements are allowed, got: %s" % stripped[:60])
    hit = _BLOCKED.search(stripped)
    if hit:
        raise DBError("query references a protected table: %s" % hit.group(0))
    return stripped


# ── the service ─────────────────────────────────────────

class DBService:
    """asyncpg-backed. `pool` is injectable so tests can run without PostgreSQL."""

    def __init__(self, dsn: str = None, pool=None):
        self.dsn = dsn or config.DATABASE_URL
        self._pool = pool
        self._ready = False
        self._lock = asyncio.Lock()

    async def pool(self):
        """Connection pool with `data_catalog` guaranteed to exist."""
        if self._pool is not None and self._ready:
            return self._pool
        async with self._lock:
            if self._pool is not None and self._ready:
                return self._pool
            if self._pool is None:
                try:
                    import asyncpg
                except ImportError as exc:      # INFRA owns requirements.txt
                    raise DBError("asyncpg is not installed: %s" % exc) from exc
                try:
                    self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
                except Exception as exc:
                    raise DBError("cannot reach PostgreSQL: %s: %s"
                                  % (type(exc).__name__, exc)) from exc
            async with self._pool.acquire() as conn:
                await conn.execute(CATALOG_SCHEMA)
                await conn.execute(CATALOG_INDEX)
            self._ready = True
            logger.info("data_catalog ready")
            return self._pool

    async def aclose(self):
        if self._pool is not None:
            await self._pool.close()
            self._pool, self._ready = None, False

    # ── tables ──────────────────────────────────────────

    async def create_table_from_df(self, name: str, df) -> dict:
        """Create `name` from the DataFrame's schema and COPY every row in.

        Returns {table_name, row_count, columns}. Replaces the table if it
        already exists — the caller owns the name, so a collision is a re-run.
        """
        _assert_ours(name)
        columns = column_definitions(df)
        if not columns:
            raise DBError("%s has no columns" % name)

        ddl = "CREATE TABLE %s (%s)" % (
            quote(name),
            ", ".join("%s %s" % (quote(sql_name), sql_type)
                      for _, sql_name, sql_type in columns),
        )
        started = time.perf_counter()
        pool = await self.pool()
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS %s" % quote(name))
            await conn.execute(ddl)
            if len(df):
                # COPY, not INSERT — 211K rows in one round trip instead of 211K.
                await conn.copy_records_to_table(
                    name,
                    records=dataframe_records(df, columns),
                    columns=[sql_name for _, sql_name, _ in columns],
                )
        logger.info("loaded %d rows x %d cols into %s in %.1fs",
                    len(df), len(columns), name, time.perf_counter() - started)
        return {
            "table_name": name,
            "row_count": int(len(df)),
            "columns": [{"name": sql_name, "source_name": str(original), "type": sql_type}
                        for original, sql_name, sql_type in columns],
        }

    async def drop_table(self, name: str) -> bool:
        """Drop one data table and forget it in the catalog."""
        _assert_ours(name)
        pool = await self.pool()
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS %s" % quote(name))
            await conn.execute("DELETE FROM data_catalog WHERE table_name = $1", name)
        logger.info("dropped %s", name)
        return True

    async def query(self, sql: str, *args, limit: int = QUERY_ROW_CAP) -> list:
        """Run one read-only statement and return rows as dicts."""
        stripped = assert_read_only(sql)
        pool = await self.pool()
        async with pool.acquire() as conn:
            async with conn.transaction(readonly=True):
                await conn.execute("SET LOCAL statement_timeout = %d" % QUERY_TIMEOUT_MS)
                rows = await conn.fetch(stripped, *args)
        return [dict(r) for r in rows[:limit]]

    async def row_count(self, name: str) -> int:
        _assert_ours(name)
        pool = await self.pool()
        async with pool.acquire() as conn:
            return int(await conn.fetchval("SELECT COUNT(*) FROM %s" % quote(name)))

    # ── catalog ─────────────────────────────────────────

    async def record(self, table_name: str, filename: str, fmt: str, row_count: int,
                     columns: list, summary: str, sheet: str = None,
                     metadata: dict = None) -> None:
        """Upsert the filename -> table_name mapping."""
        pool = await self.pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO data_catalog"
                " (table_name, filename, sheet, format, row_count, col_count,"
                "  columns, summary, metadata)"
                " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)"
                " ON CONFLICT (table_name) DO UPDATE SET"
                " filename = $2, sheet = $3, format = $4, row_count = $5,"
                " col_count = $6, columns = $7, summary = $8, metadata = $9",
                table_name, filename, sheet, fmt, int(row_count), len(columns),
                json.dumps(columns), summary, json.dumps(metadata or {}))

    async def list_tables(self, filename: str = None) -> list:
        """Catalog rows, newest first. Filter to one filename when given.

        Returns dicts rather than bare names: the admin panel and the query
        router both need the columns and row counts, and they are already here.
        """
        pool = await self.pool()
        sql = "SELECT * FROM data_catalog"
        args = ()
        if filename:
            sql += " WHERE filename = $1"
            args = (filename,)
        sql += " ORDER BY created_at DESC"
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [_decode_catalog_row(dict(r)) for r in rows]

    async def drop_by_filename(self, filename: str) -> list:
        """Drop every table this filename produced. Returns the names dropped."""
        dropped = []
        for row in await self.list_tables(filename):
            await self.drop_table(row["table_name"])
            dropped.append(row["table_name"])
        return dropped


def _decode_catalog_row(row: dict) -> dict:
    for key in ("columns", "metadata"):
        if isinstance(row.get(key), str):
            try:
                row[key] = json.loads(row[key])
            except ValueError:
                pass
    created = row.get("created_at")
    if hasattr(created, "isoformat"):
        row["created_at"] = created.isoformat()
    return row


_service = None


def get_db() -> DBService:
    """Shared instance — one connection pool for the process."""
    global _service
    if _service is None:
        _service = DBService()
    return _service
