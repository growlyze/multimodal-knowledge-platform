"""
Smallest thing that fails if structured ingestion breaks.
Run: python tests/test_structured.py

No framework, same style as test_retrieval.py, and it reuses that file's
FakeQdrant rather than growing a second one. PostgreSQL is faked with stdlib
sqlite3 (see FakePool) so the suite runs with no services at all; set
TEST_DATABASE_URL to also exercise the real thing.
"""

import asyncio
import io
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pandas as pd

import config
from pipelines import ingestion
from pipelines.structured import (SOURCE_TYPE, StructuredError, build_summary,
                                  coerce_dates, date_range, delete_dataset,
                                  detect_format, ingest_structured, job_report,
                                  parse, symbols)
from services import db_service, embedding_service, qdrant_service
from services.db_service import (DBError, DBService, quote, sanitize_identifier,
                                 table_name_for)
from services.embedding_service import StubProvider

from test_retrieval import FakeQdrant, fake_qdrant_service, reset_singletons


# ── a PostgreSQL stand-in ───────────────────────────────

class FakeConn:
    """The slice of asyncpg's connection API that db_service actually uses,
    on top of sqlite3.

    ponytail: a dialect shim, not an emulator — it rewrites $n placeholders,
    NOW() and statement_timeout, and leans on sqlite's permissive type
    affinities for BIGINT/DOUBLE PRECISION/TIMESTAMPTZ. Enough to prove rows
    land and come back; `test_live_postgres` covers the real dialect.
    """

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.statements = []

    def _sql(self, sql: str) -> str:
        self.statements.append(sql)
        # ctid is PostgreSQL's physical row address; read_dataset_table orders
        # pages by it (P184). sqlite's rowid means the same for a write-once
        # table - without this every test reading a table back died on
        # "no such column: ctid" once paging landed.
        return (sql.replace("$", "?").replace("NOW()", "CURRENT_TIMESTAMP")
                .replace("ORDER BY ctid", "ORDER BY rowid"))

    @staticmethod
    def _args(args):
        return tuple(a.isoformat() if isinstance(a, datetime) else a for a in args)

    async def execute(self, sql, *args):
        if "statement_timeout" in sql:      # PostgreSQL-only knob
            self.statements.append(sql)
            return
        self.db.execute(self._sql(sql), self._args(args))
        self.db.commit()

    async def fetch(self, sql, *args):
        cur = self.db.execute(self._sql(sql), self._args(args))
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]

    async def fetchval(self, sql, *args):
        row = self.db.execute(self._sql(sql), self._args(args)).fetchone()
        return row[0] if row else None

    async def copy_records_to_table(self, table, records=None, columns=None):
        sql = 'INSERT INTO "%s" (%s) VALUES (%s)' % (
            table, ", ".join('"%s"' % c for c in columns), ", ".join("?" * len(columns)))
        self.statements.append("COPY %s" % table)
        self.db.executemany(sql, (self._args(r) for r in records))
        self.db.commit()

    def transaction(self, readonly=False):
        conn = self

        class _Txn:
            async def __aenter__(self):
                conn.statements.append("BEGIN READONLY=%s" % readonly)
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Txn()


class FakePool:
    # check_same_thread=False because TestClient runs the app — and therefore the
    # background ingestion task — in its own portal thread, while the pool is
    # built here in the test's. asyncpg has no such rule; sqlite does.
    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn = FakeConn(self.db)

    def acquire(self):
        pool = self

        class _Acquire:
            async def __aenter__(self):
                return pool.conn

            async def __aexit__(self, *exc):
                return False

        return _Acquire()

    async def close(self):
        self.db.close()

    @property
    def statements(self):
        return self.conn.statements


def _auth_headers():
    """Bearer header for the protected routes (services/auth.py); sets a test key when none is configured."""
    import config
    config.ADMIN_API_KEY = config.ADMIN_API_KEY or "test-admin-key"
    return {"Authorization": "Bearer %s" % config.ADMIN_API_KEY}



async def fresh_pipeline():
    """A DBService on sqlite plus a fake Qdrant and a stub embedder."""
    fake = FakeQdrant()
    embedding_service._provider = StubProvider(dimensions=8)
    qdrant_service._service = fake_qdrant_service(fake)
    return DBService(pool=FakePool()), fake


def fresh_app():
    """The same fakes, installed as the process singletons the routers reach for.

    ingest_structured() takes an injectable `db`; the endpoint behind it does
    not, and neither does /admin — they call get_db(). So a test that goes
    through FastAPI replaces the singleton instead of passing an argument.

    Synchronous, unlike fresh_pipeline(), because its callers are: TestClient
    runs the app in its own portal, and a test that also called asyncio.run()
    would be starting a second loop inside the runner's.
    """
    reset_singletons()
    fake = FakeQdrant()
    db = DBService(pool=FakePool())
    embedding_service._provider = StubProvider(dimensions=8)
    qdrant_service._service = fake_qdrant_service(fake)
    db_service._service = db
    ingestion._store = ingestion.JobStore()
    ingestion._store._degraded = True      # no asyncpg here — use the memory store
    return db, fake


# ── fixtures ────────────────────────────────────────────

def series_csv(rows: int = 200, step_minutes: int = 5, symbol: str = "ACME",
           start=datetime(2020, 3, 11, 9, 30)) -> bytes:
    """5-minute time-series rows (open/high/low/close/volume) as an uploaded CSV."""
    lines = ["Symbol,Time,Open,High,Low,Close,Volume"]
    for i in range(rows):
        stamp = start + timedelta(minutes=step_minutes * i)
        base = 9000 + (i % 500) * 1.25
        lines.append("%s,%s,%.2f,%.2f,%.2f,%.2f,%d" % (
            symbol, stamp.strftime("%Y-%m-%d %H:%M:%S"),
            base, base + 5.5, base - 4.25, base + 1.75, 100 + i))
    return ("\n".join(lines) + "\n").encode()


CSV_2x1 = b"a,b\n1,2\n"
HEADERS_ONLY = b"Symbol,Time,Close\n"


def multi_sheet_xlsx() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.read_csv(io.BytesIO(series_csv(30))).to_excel(writer, sheet_name="ACME", index=False)
        pd.read_csv(io.BytesIO(series_csv(20, symbol="AT"))).to_excel(
            writer, sheet_name="AT Regional", index=False)
    return buf.getvalue()


# ── 1. CSV -> PostgreSQL ────────────────────────────────

async def test_csv_to_postgres():
    db, _ = await fresh_pipeline()
    result = await ingest_structured("ACME_2020-03-11.csv", series_csv(250), db=db)

    assert result["status"] == "done", result
    assert result["format"] == "csv"
    assert result["row_count"] == 250, result["row_count"]

    table = result["table_name"]
    assert table.startswith("data_acme_2020_03_11_"), table
    assert len(table) <= 63, table
    assert await db.row_count(table) == 250, "rows in PostgreSQL must match the file"

    # Column types survive the round trip: Time became a real timestamp.
    types = {c["name"]: c["type"] for c in result["tables"][0]["columns"]}
    assert types["symbol"] == "TEXT" and types["time"] == "TIMESTAMP", types
    assert types["close"] == "DOUBLE PRECISION" and types["volume"] == "BIGINT", types

    rows = await db.query('SELECT * FROM "%s" LIMIT 3' % table)
    assert len(rows) == 3 and rows[0]["symbol"] == "ACME", rows


# ── 2. multi-sheet Excel ────────────────────────────────

async def test_xlsx_to_postgres():
    db, _ = await fresh_pipeline()
    result = await ingest_structured("regional.xlsx", multi_sheet_xlsx(), db=db)

    assert result["status"] == "done", result
    assert len(result["tables"]) == 2, "each sheet becomes its own table"
    by_sheet = {t["sheet"]: t for t in result["tables"]}
    assert set(by_sheet) == {"ACME", "AT Regional"}, by_sheet.keys()
    assert by_sheet["ACME"]["row_count"] == 30
    assert by_sheet["AT Regional"]["row_count"] == 20

    # The sheet name is sanitized into the table name, not dropped.
    assert "at_regional" in by_sheet["AT Regional"]["table_name"], by_sheet
    for entry in result["tables"]:
        assert await db.row_count(entry["table_name"]) == entry["row_count"]
    assert by_sheet["AT Regional"]["symbols"] == ["AT"], by_sheet["AT Regional"]


# ── 3. summary quality ──────────────────────────────────

async def test_metadata_summary_quality():
    db, _ = await fresh_pipeline()
    # ~6-day rows so the file genuinely spans 2020 -> 2023, like the real one.
    result = await ingest_structured(
        "ACME_2020-03-11_to_2023-03-11.csv", series_csv(200, step_minutes=60 * 24 * 6), db=db)
    assert result["status"] == "done", result
    summary = result["tables"][0]["summary"]

    for needle in ("ACME_2020-03-11_to_2023-03-11.csv", "Format: CSV",
                   "Rows: 200", "Columns: 7", result["table_name"]):
        assert needle in summary, "summary missing %r:\n%s" % (needle, summary)
    for column in ("symbol", "time", "open", "high", "low", "close", "volume"):
        assert column in summary, "summary missing column %s" % column

    assert "Symbols / instruments: ACME" in summary, summary
    assert "Date range (time): 2020-03-11" in summary, summary
    # Years spelled out so a bare "2020" in the question hits BM25.
    assert "Years covered: 2020, 2021, 2022, 2023" in summary, summary
    assert "Numeric ranges:" in summary and "close: min" in summary, summary
    assert "Sample rows (first 5 of 200):" in summary, summary
    assert summary.count("\n") >= 15, "summary is too thin to route on"

    dates = result["tables"][0]["date_range"]
    assert dates["column"] == "time" and dates["start"].startswith("2020-03-11")


# ── 4. summary lands in Qdrant ──────────────────────────

async def test_metadata_stored_in_qdrant():
    db, fake = await fresh_pipeline()
    result = await ingest_structured("ACME_2020-03-11.csv", series_csv(120), db=db)
    assert result["indexed"] == 1, result

    points = list(fake.points[config.COLLECTION_DOCUMENTS].values())
    assert len(points) == 1, "one summary chunk per table, never one per row"
    payload = points[0]["payload"]
    assert payload["source"]["filename"] == "ACME_2020-03-11.csv"
    # The citation names the rows this summary stands for.
    assert payload["source"]["row_range"] == "1-120", payload["source"]
    assert payload["source_type"] == "structured_data"
    assert payload["table_name"] == result["table_name"]
    assert payload["content"] == result["tables"][0]["summary"]
    assert payload["row_count"] == 120 and payload["format"] == "csv"
    assert payload["columns"][:2] == ["symbol", "time"], payload["columns"]
    assert payload["date_range"]["start"].startswith("2020-03-11")
    assert payload["symbols"] == ["ACME"]

    # Both vectors present, so hybrid search can find it by meaning and by "ACME".
    assert "dense" in points[0]["vector"] and "text" in points[0]["vector"]

    hits = await qdrant_service.get_qdrant().search_sparse(
        config.COLLECTION_DOCUMENTS, "ACME data 2020", top_k=5)
    assert hits and hits[0]["payload"]["table_name"] == result["table_name"], hits


# ── 5. the catalog ──────────────────────────────────────

async def test_data_catalog_tracks_tables():
    db, _ = await fresh_pipeline()
    a = await ingest_structured("ACME_2020.csv", series_csv(40), db=db)
    b = await ingest_structured("YM_2021.csv", series_csv(60, symbol="MB"), db=db)

    catalog = await db.list_tables()
    assert len(catalog) == 2, catalog
    mapping = {row["filename"]: row for row in catalog}
    assert mapping["ACME_2020.csv"]["table_name"] == a["table_name"]
    assert mapping["YM_2021.csv"]["table_name"] == b["table_name"]

    row = mapping["ACME_2020.csv"]
    assert row["row_count"] == 40 and row["col_count"] == 7 and row["format"] == "csv"
    assert [c["name"] for c in row["columns"]][:2] == ["symbol", "time"], row["columns"]
    assert "Structured data file: ACME_2020.csv" in row["summary"]
    assert row["metadata"]["symbols"] == ["ACME"], row["metadata"]

    assert len(await db.list_tables(filename="YM_2021.csv")) == 1
    assert await db.list_tables(filename="nope.csv") == []


# ── 6. re-upload replaces ───────────────────────────────

async def test_duplicate_upload_replaces():
    db, fake = await fresh_pipeline()
    first = await ingest_structured("ACME.csv", series_csv(50), db=db)
    second = await ingest_structured("ACME.csv", series_csv(80), db=db)

    assert second["status"] == "done", second
    assert await db.list_tables(filename="ACME.csv") == await db.list_tables(), \
        "the old catalog row must be gone, not shadowed"
    assert len(await db.list_tables()) == 1, "one row per file, not one per upload"
    assert (await db.list_tables())[0]["row_count"] == 80

    if second["table_name"] != first["table_name"]:
        assert first["table_name"] in second["replaced_tables"], second
        try:
            await db.row_count(first["table_name"])
            raise AssertionError("the superseded table should have been dropped")
        except sqlite3.OperationalError:
            pass
    assert await db.row_count(second["table_name"]) == 80

    points = fake.points[config.COLLECTION_DOCUMENTS]
    assert len(points) == 1, "re-upload must replace the Qdrant chunk, not add one"
    assert list(points.values())[0]["payload"]["row_count"] == 80


# ── 7. delete cleans both stores ────────────────────────

async def test_drop_table_cleans_both():
    db, fake = await fresh_pipeline()
    result = await ingest_structured("regional.xlsx", multi_sheet_xlsx(), db=db)
    assert len(fake.points[config.COLLECTION_DOCUMENTS]) == 2

    deleted = await delete_dataset("regional.xlsx", db=db)
    assert len(deleted["tables_dropped"]) == 2, deleted
    assert deleted["chunks_removed"] == 2, deleted
    assert await db.list_tables() == [], "catalog rows must go too"
    assert fake.points[config.COLLECTION_DOCUMENTS] == {}, "Qdrant metadata must go too"

    for entry in result["tables"]:
        try:
            await db.row_count(entry["table_name"])
            raise AssertionError("%s survived the delete" % entry["table_name"])
        except sqlite3.OperationalError:
            pass


# ── 8. 100K rows ────────────────────────────────────────

async def test_large_csv_performance():
    db, _ = await fresh_pipeline()
    blob = series_csv(100_000)
    started = time.perf_counter()
    result = await ingest_structured("ACME_big.csv", blob, db=db)
    elapsed = time.perf_counter() - started

    assert result["status"] == "done", result
    assert result["row_count"] == 100_000
    assert await db.row_count(result["table_name"]) == 100_000
    print("    100K rows (%.1f MB) in %.1fs" % (len(blob) / 1e6, elapsed))
    assert elapsed < 60, "100K rows took %.1fs — something went per-row" % elapsed

    # The real regression this guards: one bulk COPY, not 100K INSERTs.
    inserts = [s for s in db._pool.statements if s.startswith(("COPY", "INSERT"))]
    assert len(inserts) <= 3, "expected a bulk load, got %d statements" % len(inserts)


# ── 9. bad input degrades, never crashes ────────────────

async def test_malformed_csv_graceful():
    db, _ = await fresh_pipeline()

    # a) ragged rows: the good rows still load.
    ragged = (b"Symbol,Time,Close\n"
              b"ACME,2020-03-11 09:30:00,9000.5\n"
              b"ACME,2020-03-11 09:35:00,9001.5,EXTRA,JUNK\n"
              b"ACME,2020-03-11 09:40:00,9002.5\n")
    result = await ingest_structured("ragged.csv", ragged, db=db)
    assert result["status"] == "done", result
    assert result["row_count"] == 2, "the ragged row is skipped, the file is not lost"

    # b) unreadable / empty / unsupported: a message, not a traceback.
    for name, blob, expect in [
        ("empty.csv", b"", "empty"),
        ("headers_only.csv", b"Symbol,Time,Close\n", "zero rows"),
        ("broken.json", b"{not json at all", "could not parse"),
        ("notes.docx", b"PK\x03\x04binary", "not a supported data file"),
    ]:
        bad = await ingest_structured(name, blob, db=db)
        assert bad["status"] == "failed", "%s should have failed: %s" % (name, bad)
        assert expect in bad["error"], "%s -> unhelpful error %r" % (name, bad["error"])
        assert bad["tables"] == [], bad

    # A bad upload leaves nothing behind.
    assert len(await db.list_tables()) == 1, await db.list_tables()


# ── 10. other formats ───────────────────────────────────

async def test_json_tsv_parquet_formats():
    db, _ = await fresh_pipeline()
    rows = [{"Symbol": "AT", "Time": "2021-01-0%d 09:30:00" % d, "Close": 3700.0 + d}
            for d in range(1, 6)]

    cases = [("t.json", json.dumps(rows).encode(), "json"),
             ("t.jsonl", b"\n".join(json.dumps(r).encode() for r in rows), "jsonl"),
             ("t.tsv", series_csv(12).replace(b",", b"\t"), "tsv")]
    try:
        import pyarrow  # noqa: F401
        buf = io.BytesIO()
        pd.DataFrame(rows).to_parquet(buf, index=False)
        cases.append(("t.parquet", buf.getvalue(), "parquet"))
    except ImportError:
        print("    (parquet skipped — pyarrow not installed)")

    for name, blob, fmt in cases:
        result = await ingest_structured(name, blob, db=db)
        assert result["status"] == "done", (name, result)
        assert result["format"] == fmt and result["row_count"] in (5, 12), result
        assert await db.row_count(result["table_name"]) == result["row_count"]

    # {"rows": [...]} wrappers are unwrapped rather than flattened to one row.
    wrapped = await ingest_structured("w.json", json.dumps({"rows": rows}).encode(), db=db)
    assert wrapped["row_count"] == 5, wrapped


# ── 11. identifiers are a trust boundary ────────────────

def test_identifiers_are_sanitized():
    hostile = 'acme"; DROP TABLE "user"; --.csv'
    table = table_name_for(hostile)
    assert table.startswith("data_") and len(table) <= 63
    assert set(table) <= set("abcdefghijklmnopqrstuvwxyz0123456789_"), table
    quote(table)                                     # must not raise

    assert sanitize_identifier("2020 Close") == "col_2020_close"
    assert sanitize_identifier("!!!") == "col"
    assert len(sanitize_identifier("x" * 200)) == 63

    for bad in ['a"b', "a;b", "DROP", "", "1a", None]:
        try:
            quote(bad)
            raise AssertionError("quote() accepted %r" % bad)
        except DBError:
            pass


async def test_query_is_read_only():
    db, _ = await fresh_pipeline()
    await ingest_structured("ACME.csv", series_csv(10), db=db)

    for bad in ["DROP TABLE data_catalog", "DELETE FROM data_catalog",
                "SELECT 1; DROP TABLE data_catalog", "SELECT * FROM auth", ""]:
        try:
            await db.query(bad)
            raise AssertionError("query() allowed %r" % bad)
        except DBError:
            pass

    rows = await db.query("SELECT filename FROM data_catalog")
    assert rows == [{"filename": "ACME.csv"}], rows
    assert "BEGIN READONLY=True" in db._pool.statements


# ── 12. parsing helpers ─────────────────────────────────

def test_format_detection_and_date_coercion():
    assert detect_format("a/b/ACME.CSV") == "csv"
    assert detect_format("x.xlsm") == "xlsx" and detect_format("x.ndjson") == "jsonl"
    for name in ("x.pdf", "x", "x.zip"):
        try:
            detect_format(name)
            raise AssertionError("%s should not be a data file" % name)
        except StructuredError:
            pass

    df = pd.DataFrame({"Time": ["2020-03-11 09:30:00", "2020-03-12 09:30:00"],
                       "Label": ["a", "b"], "Note": ["2020-03-11", "x"]})
    assert coerce_dates(df) == ["Time"], "only date-named columns are converted"
    assert date_range(df)["column"] == "Time"
    assert date_range(pd.DataFrame({"a": [1]})) == {}

    # A high-cardinality 'symbol' column is an id, not an instrument.
    assert symbols(pd.DataFrame({"Symbol": ["ACME"] * 5 + ["AT"] * 5})) == ["ACME", "AT"]
    assert symbols(pd.DataFrame({"Symbol": [str(i) for i in range(50)]})) == []
    assert symbols(pd.DataFrame({"a": [1, 2]})) == []

    fmt, frames = parse("x.csv", series_csv(3))
    assert fmt == "csv" and len(frames[None]) == 3


async def test_new_york_offsets_across_dst_ingest():
    """the client's 1-minute files stamp every row with New York's own offset, -0400
    in summer and -0500 in winter, so any file spanning a DST change mixes the
    two (all 9 continuous parts and every per-contract file, 2026-10-22).
    pandas 3 raises "Mixed timezones detected" out of coerce_dates and the
    whole upload became a failed job; the pinned pandas 2.2.3 instead returned
    an object column, so `time` was stored as TEXT with no date range - the
    summary lost "Years covered", and "AT 2018" could no longer tell part 1
    of a 3-part series from part 3."""
    lines = ["Time,Open,High,Low,Close,Volume,symbol",
             "2016-11-04 15:59:00-0400,2085.0,2086.0,2084.5,2085.5,10,ATZ16",
             "2016-11-06 18:00:00-0500,2090.0,2091.0,2089.5,2090.5,12,ATZ16",
             "2017-03-17 09:30:00-0400,2380.0,2381.0,2379.5,2380.5,15,ATH17"]
    blob = ("\n".join(lines) + "\n").encode()
    db, _ = await fresh_pipeline()
    result = await ingest_structured("AT_1min_continuous_NY_part1_of_3.csv", blob, db=db)
    assert result["status"] == "done", result
    assert await db.row_count(result["table_name"]) == 3
    entry = result["tables"][0]
    dates = entry["date_range"]
    # The span in UTC - an unambiguous instant, whichever offset each row had.
    assert dates and dates["start"].startswith("2016-11-04T19:59"), dates
    assert dates["end"].startswith("2017-03-17T13:30"), dates
    assert dates["years"] == [2016, 2017], dates
    assert "Years covered: 2016, 2017" in entry["summary"], entry["summary"]
    # The stored text keeps each row's own offset: load_dataset() reads that to
    # know the file is New York time (P187) - a UTC rewrite would erase it.
    rows = await db.query('SELECT time FROM "%s"' % result["table_name"])
    stored = sorted(str(r["time"]) for r in rows)
    assert stored[0].startswith("2016-11-04 15:59:00-04"), stored
    assert stored[1].startswith("2016-11-06 18:00:00-05"), stored


def test_summary_survives_odd_frames():
    """No date column, no symbol column, no numerics — still a usable summary."""
    df = pd.DataFrame({"note": ["alpha", "beta"], "tag": ["x", "y"]})
    text = build_summary("notes.csv", "csv", "data_notes_1", df,
                         [{"name": "note", "type": "TEXT"}, {"name": "tag", "type": "TEXT"}])
    assert "Rows: 2 | Columns: 2" in text and "data_notes_1" in text
    assert "Date range" not in text and "Numeric ranges" not in text
    assert "Sample rows" in text


# ── 13. the endpoint the filter actually calls ──────────

def test_ingest_endpoint_over_http():
    """POST /ingest/structured -> rows in PostgreSQL, a summary in Qdrant, a job.

    The unit tests above call ingest_structured() directly. This is the wiring
    between it and the Open WebUI filter, which is where a CSV upload used to
    land nowhere: the endpoint parsed the form and answered "not implemented".
    """
    from fastapi.testclient import TestClient

    _, fake = fresh_app()
    from main import app

    with TestClient(app, headers=_auth_headers()) as client:
        posted = client.post(
            "/ingest/structured",
            files={"file": ("ACME_2020-03-11.csv", series_csv(120), "text/csv")},
            data={"metadata": json.dumps({"origin": "open-webui-chat",
                                          "chat_id": "c12"})})
        assert posted.status_code == 200, posted.text
        queued = posted.json()
        assert queued["status"] == "queued" and queued["job_id"], queued
        assert queued["format"] == "csv" and queued["bytes"] > 0, queued
        assert queued["source_type"] == SOURCE_TYPE, queued

        # TestClient runs background tasks before returning, so it is done by now.
        job = client.get("/ingest/status/%s" % queued["job_id"]).json()
        assert job["status"] == "done", job
        assert job["chunks_created"] == 1, "one summary point per table"
        report = job["quality"]
        assert report["kind"] == "structured" and report["row_count"] == 120, report
        table = report["tables"][0]["table_name"]
        assert table.startswith("data_acme_2020_03_11_"), table

        # the rows really are in the database — read back the way the sandbox
        # reads them, so this covers the load path as well as the write
        stored = client.get("/admin/data/table/%s" % table).json()
        assert stored["total_rows"] == 120 and len(stored["rows"]) == 120, stored

        # ...and the summary really is in the vector store, carrying the chat
        # metadata the filter sent, so an upload can be traced back to its chat.
        points = list(fake.points[config.COLLECTION_DOCUMENTS].values())
        assert len(points) == 1, points
        payload = points[0]["payload"]
        assert payload["table_name"] == table, payload
        assert payload["source_type"] == SOURCE_TYPE, payload
        assert payload["chat_id"] == "c12", "metadata was dropped"
        assert table in payload["content"], payload["content"][:200]

        # The 400s. All of them are knowable before a job exists, so none leaves
        # a failed job behind for someone to find later and wonder about.
        assert client.post("/ingest/structured",
                           files={"file": ("empty.csv", b"", "text/csv")}
                           ).status_code == 400
        assert client.post("/ingest/structured",
                           files={"file": ("notes.docx", b"PK-not-a-spreadsheet")}
                           ).status_code == 400
        assert client.post("/ingest/structured",
                           files={"file": ("a.csv", CSV_2x1)},
                           data={"metadata": "not json"}).status_code == 400
        # Architecture decision #8 applies here too — chat text is never indexed,
        # whichever endpoint it is posted to.
        assert client.post(
            "/ingest/structured",
            files={"file": ("conversation_4cc31daa_2.txt", CSV_2x1)},
            data={"metadata": json.dumps({"origin": "open-webui-chat"})}
        ).status_code == 400
    reset_singletons()


def test_failed_upload_is_a_failed_job_not_a_crash():
    """A file that parses to nothing fails the job and says why in the job row."""
    from fastapi.testclient import TestClient

    fresh_app()
    from main import app

    with TestClient(app, headers=_auth_headers()) as client:
        posted = client.post("/ingest/structured",
                             files={"file": ("headers.csv", HEADERS_ONLY)})
        assert posted.status_code == 200, posted.text
        job = client.get("/ingest/status/%s" % posted.json()["job_id"]).json()
        assert job["status"] == "failed", job
        assert "zero rows" in job["error"], job["error"]
        assert job["quality"]["tables"] == [], job["quality"]
    reset_singletons()


# ── 14. what the panel and the sandbox each read back ───

def test_admin_sees_the_upload():
    """One CSV, then the reads that all have to find it.

    /data/search and /data/table are load_dataset()'s pair; /tables and
    /tables/{name} are the panel's. Same catalog underneath, so they must agree.
    """
    from fastapi.testclient import TestClient

    _, fake = fresh_app()
    from main import app

    with TestClient(app, headers=_auth_headers()) as client:
        posted = client.post("/ingest/structured",
                             files={"file": ("ACME_2020.csv", series_csv(40), "text/csv")})
        assert posted.status_code == 200, posted.text

        # the panel's list
        listing = client.get("/admin/tables").json()
        assert listing["count"] == 1, listing
        row = listing["tables"][0]
        table = row["table_name"]
        assert row["filename"] == "ACME_2020.csv" and row["rows"] == 40, row
        assert row["symbols"] == "ACME" and row["from"] == "2020-03-11", row
        assert "close" in row["column_names"], row
        assert "summary" not in row, "the panel lists tables, not paragraphs"

        # the panel's preview
        preview = client.get("/admin/tables/%s" % table, params={"limit": 5}).json()
        assert len(preview["rows"]) == 5 and preview["total_rows"] == 40, preview
        assert preview["rows"][0]["symbol"] == "ACME", preview["rows"][0]
        assert client.get("/admin/tables/data_nope").status_code == 404
        # and never a table we did not create, however it is dressed up
        assert client.get("/admin/tables/pg_user").status_code == 404

        # load_dataset()'s pair: find it by name, then read every row
        hits = client.get("/admin/data/search", params={"q": "ACME_2020.csv"}).json()
        assert [h["table_name"] for h in hits] == [table], hits
        rows = client.get("/admin/data/table/%s" % table).json()["rows"]
        assert len(rows) == 40 and rows[0]["close"] > 0, rows[0]

        # the dashboard metric
        assert client.get("/admin/stats").json()["postgresql"]["structured_tables"] == 1

        # deleting drops the rows AND the summary that points at them
        dropped = client.delete("/admin/tables/%s" % table).json()
        assert dropped["dropped"] and dropped["chunks_removed"] == 1, dropped
        assert client.get("/admin/tables").json()["count"] == 0
        assert not fake.points[config.COLLECTION_DOCUMENTS], "orphaned summary chunk"
        assert client.delete("/admin/tables/%s" % table).status_code == 404
    reset_singletons()


def test_load_dataset_reads_the_uploaded_csv():
    """The last link: data_tools.load_dataset() against the real endpoints.

    Mode B end to end. The tests above prove the rows and the catalog are
    right; this one proves the sandbox's own loader makes a usable DataFrame out
    of them, with the `time` column becoming the index.

    data_tools ships in the code-execution image rather than this one, so it
    is imported from the repo and skipped when it is not there. Only the socket
    is faked: load_dataset's own URL building, JSON handling and normalisation
    all run, over the payloads the admin router really returns.
    """
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(ROOT.parent / "code-execution"))
    try:
        from data_tools import loader as dt_loader
    except ImportError as exc:
        print("    (skipped — data_tools not importable: %s)" % exc)
        return

    fresh_app()
    from main import app

    with TestClient(app, headers=_auth_headers()) as client:
        assert client.post("/ingest/structured",
                           files={"file": ("ACME_2020.csv", series_csv(40), "text/csv")}
                           ).status_code == 200

        def through_test_client(url, timeout=None):
            assert url.startswith("http://haystack-api:8000/"), url
            response = client.get(url[len("http://haystack-api:8000"):])
            return response.json() if response.status_code == 200 else None

        real_get, dt_loader._api_get = dt_loader._api_get, through_test_client
        try:
            df = dt_loader.load_dataset("ACME_2020", api_url="http://haystack-api:8000")
        finally:
            dt_loader._api_get = real_get

    assert len(df) == 40, "load_dataset came back with %d of 40 rows" % len(df)
    assert isinstance(df.index, pd.DatetimeIndex), df.index
    cols = {str(c).lower() for c in df.columns}
    assert {"open", "high", "low", "close"} <= cols, df.columns
    reset_singletons()


def test_job_report_is_small():
    """The job row is polled, so it must not carry the summary paragraphs."""
    report = job_report({
        "status": "done", "format": "csv", "indexed": 2,
        "replaced_tables": ["data_old"],
        "tables": [{"table_name": "data_a", "sheet": None, "row_count": 3,
                    "columns": [{"name": "close", "source_name": "Close",
                                 "type": "DOUBLE PRECISION"}],
                    "summary": "x" * 5000, "date_range": {}, "symbols": []}],
    })
    assert report == {"kind": "structured", "format": "csv", "row_count": 3,
                      "summaries_indexed": 2, "replaced_tables": ["data_old"],
                      "tables": [{"table_name": "data_a", "sheet": None,
                                  "row_count": 3, "columns": ["close"]}]}, report
    assert len(json.dumps(report)) < 400, "the job row is a status, not a payload"


# ── 15. duplicate detection ─────────────────────────────
# Measured live 2026-10-26: the filter re-POSTs every attachment on every chat
# turn, so each turn dropped and recreated a file's tables under a new
# timestamped name and re-embedded its summary; and one file re-uploaded under
# a second name ("DL_AT1_test.csv" / "CME_MINI_DL_ES1!, 5 (2).csv") became two
# sets of tables and two summaries for load_dataset() to choose between.

def count_embeds():
    """Wrap the stub embedder so a test can prove a skip spent nothing."""
    provider = embedding_service._provider
    calls = []
    real = provider.embed

    async def counting(texts, input_type="document"):
        calls.append(len(texts))
        return await real(texts, input_type=input_type)

    provider.embed = counting
    return calls


async def test_repost_of_same_file_is_unchanged():
    db, fake = await fresh_pipeline()
    first = await ingest_structured("ACME.csv", series_csv(50), {"chat_id": "c1"}, db=db)
    assert first["status"] == "done", first
    row = (await db.list_tables())[0]
    # The keys a later upload is matched on, stored where the next one looks.
    assert row["metadata"]["file_sha256"].startswith("sha256:"), row["metadata"]
    assert row["metadata"]["hash_version"] == 1, row["metadata"]
    payload = list(fake.points[config.COLLECTION_DOCUMENTS].values())[0]["payload"]
    assert payload["file_sha256"] == row["metadata"]["file_sha256"], payload

    embeds = count_embeds()
    second = await ingest_structured("ACME.csv", series_csv(50), {"chat_id": "c1"}, db=db)
    # status done + indexed 0: archive_ingestion and run_structured_job both
    # treat that as success, not as a failed member.
    assert second["status"] == "done" and second["indexed"] == 0, second
    assert second["skipped"] == "unchanged" and second["matched_on"] == "file_sha256", second
    assert second["table_name"] == first["table_name"], "the table must keep its name"
    assert [t["table_name"] for t in second["tables"]] == [first["table_name"]], second
    assert not embeds, "an unchanged re-post must not embed"
    assert len(await db.list_tables()) == 1
    assert await db.row_count(first["table_name"]) == 50, "rows must be untouched"
    assert len(fake.points[config.COLLECTION_DOCUMENTS]) == 1

    report = job_report(second)
    assert report["skipped"] == "unchanged", report
    assert report["tables"][0]["table_name"] == first["table_name"], report


async def test_same_bytes_under_another_name_is_an_alias():
    db, fake = await fresh_pipeline()
    blob = series_csv(60, symbol="AT")
    original = await ingest_structured("DL_AT1_test.csv", blob, db=db)
    assert original["status"] == "done", original

    embeds = count_embeds()
    copy = "CME_MINI_DL_ES1!, 5 (2).csv"
    dup = await ingest_structured(copy, blob, db=db)
    assert dup["status"] == "done" and dup["indexed"] == 0, dup
    assert dup["skipped"] == "duplicate" and dup["duplicate_of"] == "DL_AT1_test.csv", dup
    assert dup["table_name"] == original["table_name"], "the job names where the data lives"
    assert not embeds, "a duplicate must not embed"

    catalog = await db.list_tables()
    assert [r["filename"] for r in catalog] == ["DL_AT1_test.csv"], catalog
    assert catalog[0]["metadata"]["aliases"] == [copy], catalog[0]["metadata"]
    # The original row keeps everything it had; only the alias is added.
    assert catalog[0]["metadata"]["symbols"] == ["AT"], catalog[0]["metadata"]
    assert catalog[0]["row_count"] == 60 and catalog[0]["summary"], catalog[0]
    assert len(fake.points[config.COLLECTION_DOCUMENTS]) == 1, "no second summary"

    # The filter re-posts the copy every turn: the alias is recorded once.
    again = await ingest_structured(copy, blob, db=db)
    assert again["skipped"] == "duplicate", again
    assert (await db.list_tables())[0]["metadata"]["aliases"] == [copy]

    report = job_report(dup)
    assert report["skipped"] == "duplicate" and report["duplicate_of"] == "DL_AT1_test.csv"
    assert report["tables"][0]["table_name"] == original["table_name"], report


async def test_duplicate_clears_the_names_old_content():
    """b.csv used to hold other data; re-uploaded with a.csv's bytes, its old
    tables and summary describe content that name no longer holds."""
    db, fake = await fresh_pipeline()
    await ingest_structured("a.csv", series_csv(30), db=db)
    old_b = await ingest_structured("b.csv", series_csv(20, symbol="MB"), db=db)
    assert len(fake.points[config.COLLECTION_DOCUMENTS]) == 2

    dup = await ingest_structured("b.csv", series_csv(30), db=db)
    assert dup["skipped"] == "duplicate" and dup["duplicate_of"] == "a.csv", dup
    assert dup["replaced_tables"] == [old_b["table_name"]], dup
    assert [r["filename"] for r in await db.list_tables()] == ["a.csv"]
    points = list(fake.points[config.COLLECTION_DOCUMENTS].values())
    assert [p["payload"]["source"]["filename"] for p in points] == ["a.csv"], points


async def test_duplicate_workbook_aliases_every_sheet():
    db, _ = await fresh_pipeline()
    blob = multi_sheet_xlsx()
    await ingest_structured("regional.xlsx", blob, db=db)
    dup = await ingest_structured("regional copy.xlsx", blob, db=db)
    assert dup["skipped"] == "duplicate" and len(dup["tables"]) == 2, dup
    rows = await db.list_tables()
    assert len(rows) == 2 and all(r["metadata"]["aliases"] == ["regional copy.xlsx"]
                                  for r in rows), rows
    # Each sheet keeps its own sheet name and summary through the re-record.
    assert {r["sheet"] for r in rows} == {"ACME", "AT Regional"}, rows


async def test_force_reingest_bypasses_dedup_and_is_not_stored():
    db, fake = await fresh_pipeline()
    first = await ingest_structured("ACME.csv", series_csv(40), db=db)
    forced = await ingest_structured("ACME.csv", series_csv(40),
                                     {ingestion.FORCE_KEY: True}, db=db)
    assert forced["status"] == "done" and not forced.get("skipped"), forced
    assert forced["indexed"] == 1 and first["table_name"] in forced["replaced_tables"]
    row = (await db.list_tables())[0]
    assert ingestion.FORCE_KEY not in row["metadata"], row["metadata"]
    payload = list(fake.points[config.COLLECTION_DOCUMENTS].values())[0]["payload"]
    assert ingestion.FORCE_KEY not in payload, payload


async def test_dedup_switch_off_restores_plain_replace():
    db, _ = await fresh_pipeline()
    saved = ingestion.INGEST_DEDUP
    ingestion.INGEST_DEDUP = False
    try:
        await ingest_structured("ACME.csv", series_csv(40), db=db)
        second = await ingest_structured("ACME.csv", series_csv(40), db=db)
        assert not second.get("skipped") and second["indexed"] == 1, second
        dup = await ingest_structured("ACME copy.csv", series_csv(40), db=db)
        assert not dup.get("skipped") and len(await db.list_tables()) == 2, dup
    finally:
        ingestion.INGEST_DEDUP = saved


class SlowCatalogDB(DBService):
    """list_tables() that yields to the loop, as asyncpg genuinely does.

    The sqlite fake never suspends, so without this two gathered uploads run
    one after the other and a missing lock could not be seen.
    """

    async def list_tables(self, filename: str = None) -> list:
        await asyncio.sleep(0.01)
        return await super().list_tables(filename)


async def test_simultaneous_identical_uploads_make_one_set_of_tables():
    _, fake = await fresh_pipeline()
    db = SlowCatalogDB(pool=FakePool())
    blob = series_csv(40)
    a, b = await asyncio.gather(
        ingest_structured("ACME_EXPORT_2026, 5.csv", blob, db=db),
        ingest_structured("proba ACME_EXPORT_2026, 5.csv", blob, db=db))
    assert a["status"] == b["status"] == "done", (a, b)
    assert sorted([bool(a.get("skipped")), bool(b.get("skipped"))]) == [False, True], (a, b)
    assert len(await db.list_tables()) == 1, await db.list_tables()
    assert len(fake.points[config.COLLECTION_DOCUMENTS]) == 1


async def test_skipped_job_is_done_with_the_reason():
    from pipelines.structured import run_structured_job

    db, _ = fresh_app()
    blob = series_csv(30)
    first = await ingestion.create_job("AT.csv", SOURCE_TYPE)
    await run_structured_job(first["job_id"], "AT.csv", blob, db=db)
    second = await ingestion.create_job("AT (1).csv", SOURCE_TYPE)
    await run_structured_job(second["job_id"], "AT (1).csv", blob, db=db)

    job = await ingestion.get_store().get(second["job_id"])
    assert job["status"] == "done" and job["chunks_created"] == 0, job
    quality = job["quality"]
    assert quality["skipped"] == "duplicate" and quality["duplicate_of"] == "AT.csv", quality
    assert quality["tables"] and quality["tables"][0]["table_name"].startswith("data_at_"), quality
    reset_singletons()


# ── 16. the real thing, when it is reachable ────────────

async def test_live_postgres():
    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        print("    (skipped — set TEST_DATABASE_URL to run against real PostgreSQL)")
        return
    db = DBService(dsn=dsn)
    name = "structured_selftest.csv"
    try:
        embedding_service._provider = StubProvider(dimensions=8)
        qdrant_service._service = fake_qdrant_service(FakeQdrant())
        result = await ingest_structured(name, series_csv(5000), db=db)
        assert result["status"] == "done", result
        assert await db.row_count(result["table_name"]) == 5000
        rows = await db.query(
            'SELECT MAX(close) AS hi FROM "%s" WHERE time >= $1' % result["table_name"],
            datetime(2020, 3, 11))
        assert rows and rows[0]["hi"] > 0, rows
        print("    live PostgreSQL OK: %s" % result["table_name"])
    except DBError as exc:
        print("    (skipped — %s)" % exc)
    finally:
        try:
            await db.drop_by_filename(name)
        finally:
            await db.aclose()


async def main():
    tests = [
        test_csv_to_postgres,
        test_xlsx_to_postgres,
        test_metadata_summary_quality,
        test_metadata_stored_in_qdrant,
        test_data_catalog_tracks_tables,
        test_duplicate_upload_replaces,
        test_drop_table_cleans_both,
        test_large_csv_performance,
        test_malformed_csv_graceful,
        test_json_tsv_parquet_formats,
        test_identifiers_are_sanitized,
        test_query_is_read_only,
        test_format_detection_and_date_coercion,
        test_new_york_offsets_across_dst_ingest,
        test_summary_survives_odd_frames,
        test_ingest_endpoint_over_http,
        test_failed_upload_is_a_failed_job_not_a_crash,
        test_admin_sees_the_upload,
        test_load_dataset_reads_the_uploaded_csv,
        test_job_report_is_small,
        test_repost_of_same_file_is_unchanged,
        test_same_bytes_under_another_name_is_an_alias,
        test_duplicate_clears_the_names_old_content,
        test_duplicate_workbook_aliases_every_sheet,
        test_force_reingest_bypasses_dedup_and_is_not_stored,
        test_dedup_switch_off_restores_plain_replace,
        test_simultaneous_identical_uploads_make_one_set_of_tables,
        test_skipped_job_is_done_with_the_reason,
        test_live_postgres,
    ]
    for test in tests:
        result = test()
        if asyncio.iscoroutine(result):
            await result
        print("ok  %s" % test.__name__)
    print("\n%d structured checks passed" % len(tests))


if __name__ == "__main__":
    asyncio.run(main())
