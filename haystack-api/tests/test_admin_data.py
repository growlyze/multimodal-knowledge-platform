"""
/admin/data/search and /admin/data/table — the pair the code sandbox loads data through.

The case worth guarding is a wrong match rather than no match. load_dataset() takes
hits[0] and analyses it, so "AT" matching the ACME dataset does not raise anything —
it returns a plausible-looking result computed on the wrong instrument.

    python tests/test_admin_data.py

No PostgreSQL: the catalog and one table are faked at the DBService boundary.
"""

import asyncio

from fastapi import HTTPException

from routers import admin

# Summaries copied in shape from pipelines.structured.build_summary, including the
# words that make naive substring matching fail: "data", "Format" and "Date" all contain
# "at", "Symbols" contains "mb", "low" contains "lo".
ACME_SUMMARY = (
    "Structured data file: ACME_2020_2023.csv\n"
    "Format: CSV | PostgreSQL table: data_acme_2020_2023_20250831\n"
    "Rows: 211,043 | Columns: 7\n"
    "Columns: symbol, time, open, high, low, close, volume\n"
    "Symbols / instruments: ACME\n"
    "Date range (time): 2020-03-11 to 2023-03-11\n"
    "Numeric ranges:\n  close: min 6628, max 16767\n"
    "Sample rows (first 5 of 211,043): readings follow\n"
)
AT_SUMMARY = (
    "Structured data file: AT_regional_2022.csv\n"
    "Format: CSV | PostgreSQL table: data_at_regional_2022_20250830\n"
    "Rows: 98,204 | Columns: 7\n"
    "Columns: symbol, time, open, high, low, close, volume\n"
    "Symbols / instruments: AT\n"
    "Date range (time): 2022-01-03 to 2022-12-30\n"
)

CATALOG = [
    {   # newest first, the order list_tables() guarantees
        "table_name": "data_acme_2020_2023_20250831",
        "filename": "ACME_2020_2023.csv",
        "sheet": None, "format": "csv", "row_count": 211043,
        "columns": [{"name": "close", "type": "double precision"},
                    {"name": "time", "type": "timestamp"}],
        "summary": ACME_SUMMARY,
        "created_at": "2025-08-31T10:00:00",
        "metadata": {"symbols": ["ACME"],
                     "date_range": {"start": "2020-03-11T00:00:00",
                                    "end": "2023-03-11T00:00:00"}},
    },
    {
        "table_name": "data_at_regional_2022_20250830",
        "filename": "AT_regional_2022.csv",
        "sheet": None, "format": "csv", "row_count": 98204,
        "columns": [{"name": "close", "type": "double precision"}],
        "summary": AT_SUMMARY,
        "created_at": "2025-08-30T10:00:00",
        "metadata": {"symbols": ["AT"],
                     "date_range": {"start": "2022-01-03T00:00:00",
                                    "end": "2022-12-30T00:00:00"}},
    },
]

ROWS = [{"symbol": "ACME", "time": "2020-03-11T09:30:00", "close": 8000 + i}
        for i in range(50)]


class FakeDB:
    def __init__(self, catalog=None, rows=None, broken=False):
        self.catalog = CATALOG if catalog is None else catalog
        self.rows = ROWS if rows is None else rows
        self.broken = broken
        self.sql = []

    async def list_tables(self, filename=None):
        if self.broken:
            from services.db_service import DBError
            raise DBError("connection refused")
        return list(self.catalog)

    async def query(self, sql, *args, limit=1000):
        self.sql.append(sql)
        take, offset = (args + (None, 0))[:2]
        rows = self.rows[offset:]
        return rows[:take] if take else rows


def use(db):
    admin.get_db = lambda: db
    return db


async def raises(status, coro):
    try:
        await coro
    except HTTPException as exc:
        assert exc.status_code == status, exc.status_code
        return exc
    raise AssertionError("expected HTTP %d, got a normal return" % status)


# ── search ──────────────────────────────────────────────

async def test_symbol_finds_its_own_dataset():
    use(FakeDB())
    for symbol, table in (("ACME", "data_acme_2020_2023_20250831"),
                          ("AT", "data_at_regional_2022_20250830")):
        hits = await admin.search_datasets(q=symbol, limit=5)
        assert hits, "%s found nothing" % symbol
        # This is the assertion that matters: hits[0] is what gets analysed.
        assert hits[0]["table_name"] == table, (symbol, hits[0]["table_name"])
        assert hits[0]["symbol"] == symbol


async def test_short_symbol_does_not_match_prose():
    """"AT" appears inside "data", "Format" and "Date"; "mb" inside "Symbols"."""
    use(FakeDB(catalog=[CATALOG[0]]))          # ACME only — no AT dataset exists
    for query in ("AT", "MB", "LO"):
        assert await admin.search_datasets(q=query, limit=5) == [], \
            "%s matched the ACME dataset through substring noise" % query


async def test_ranking_prefers_symbol_then_filename():
    use(FakeDB())
    # "regional" appears only in the AT filename, "2022" only in its summary/name.
    hits = await admin.search_datasets(q="regional", limit=5)
    assert hits[0]["filename"] == "AT_regional_2022.csv", hits

    # A symbol hit outranks a filename hit even when both match.
    hits = await admin.search_datasets(q="ACME regional", limit=5)
    assert hits[0]["symbol"] == "ACME", hits
    assert len(hits) == 2, "both should still be returned, just ordered"


async def test_no_match_is_an_empty_list_not_an_error():
    use(FakeDB())
    assert await admin.search_datasets(q="weather station data 1987", limit=5) == []
    # An empty catalog is the same answer, not a 404.
    use(FakeDB(catalog=[]))
    assert await admin.search_datasets(q="ACME", limit=5) == []


async def test_blank_query_lists_the_catalog():
    use(FakeDB())
    hits = await admin.search_datasets(q="   ", limit=5)
    assert [h["filename"] for h in hits] == ["ACME_2020_2023.csv", "AT_regional_2022.csv"]


async def test_limit_is_honoured():
    use(FakeDB())
    assert len(await admin.search_datasets(q="ACME AT", limit=1)) == 1
    assert len(await admin.search_datasets(q="ACME AT", limit=5)) == 2


async def test_catalog_boilerplate_is_not_a_match():
    """Every table name starts with "data_" and every summary says "Structured
    data file" — matching on that vocabulary is matching on nothing."""
    use(FakeDB())
    for noise in ("data", "csv file", "rows columns", "close volume"):
        hits = await admin.search_datasets(q=noise, limit=5)
        assert len(hits) == len(CATALOG), \
            "%r should be treated as no query, not as a match" % noise

    # A word that is neither boilerplate nor in the catalog stays a real token,
    # so it correctly matches nothing rather than listing everything.
    assert await admin.search_datasets(q="rows and columns", limit=5) == []


async def test_database_down_is_503_not_empty():
    """An outage must not read as 'no data exists' — that hides the failure."""
    use(FakeDB(broken=True))
    await raises(503, admin.search_datasets(q="ACME", limit=5))


async def test_response_shape_is_what_load_dataset_reads():
    """Mirror data_tools.loader.load_dataset's own parsing, exactly."""
    use(FakeDB())
    hits = await admin.search_datasets(q="ACME", limit=5)

    if isinstance(hits, dict):                                  # its wrapper handling
        hits = hits.get("results") or hits.get("datasets") or hits.get("tables") or []
    match = hits[0] if isinstance(hits, list) and hits else None
    assert isinstance(match, dict), match
    table = match.get("table_name") or match.get("table") or match.get("name")
    assert table == "data_acme_2020_2023_20250831", table

    for field in ("filename", "symbol", "date_range", "row_count"):
        assert field in match, field
    assert match["date_range"]["start"] == "2020-03-11T00:00:00"
    assert match["row_count"] == 211043


# ── table ───────────────────────────────────────────────

async def test_table_returns_rows_load_dataset_can_frame():
    db = use(FakeDB())
    payload = await admin.read_dataset_table("data_acme_2020_2023_20250831",
                                             limit=1000, offset=0)
    rows = None
    for key in ("rows", "data", "records", "results"):          # _rows_to_frame
        value = payload.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            rows = value
            break
    assert rows is not None and len(rows) == 50, payload.get("row_count")
    assert rows[0]["close"] == 8000
    assert payload["truncated"] is False
    assert payload["total_rows"] == 211043
    assert 'FROM "data_acme_2020_2023_20250831"' in db.sql[0], db.sql


async def test_truncation_is_flagged():
    use(FakeDB())
    payload = await admin.read_dataset_table("data_acme_2020_2023_20250831",
                                             limit=10, offset=0)
    assert payload["truncated"] is True, "a clipped table must say so"
    assert len(payload["rows"]) == 10


async def test_offset_pages():
    use(FakeDB())
    payload = await admin.read_dataset_table("data_acme_2020_2023_20250831",
                                             limit=5, offset=20)
    assert payload["rows"][0]["close"] == 8020, payload["rows"][0]


async def test_pages_have_a_stable_order():
    # load_dataset() pages through tables above 250K rows with OFFSET. Without an
    # ORDER BY, PostgreSQL may return rows in any order (a synchronized seq scan
    # can start mid-table), so two pages could overlap or skip rows.
    db = FakeDB()
    use(db)
    await admin.read_dataset_table("data_acme_2020_2023_20250831", limit=5, offset=20)
    assert "ORDER BY ctid" in db.sql[-1], db.sql[-1]


async def test_unknown_table_is_404():
    use(FakeDB())
    await raises(404, admin.read_dataset_table("data_not_a_thing", limit=10, offset=0))


async def test_table_name_cannot_escape_the_catalog():
    """Identifiers cannot be bind parameters, so the catalog is the allowlist."""
    use(FakeDB())
    for hostile in ('data_x"; DROP TABLE data_catalog; --',
                    "auth", "users", "data_catalog",
                    "../../etc/passwd", "data_acme_2020_2023_20250831 UNION SELECT 1"):
        db = use(FakeDB())
        await raises(404, admin.read_dataset_table(hostile, limit=10, offset=0))
        assert not db.sql, "a rejected name must never reach SQL: %r" % hostile


async def test_alias_finds_the_original_table():
    """A byte-identical re-upload under another name is recorded as an alias on
    the original's catalog row instead of becoming a second table (dedup). The
    user still asks for it by the name they uploaded, so the alias must match."""
    row = dict(CATALOG[0], filename="ACME_EXPORT_2026, 5.csv",
               metadata=dict(CATALOG[0]["metadata"],
                             aliases=["proba ACME_EXPORT_2026, 5.csv"]))
    use(FakeDB(catalog=[row, CATALOG[1]]))
    hits = await admin.search_datasets(q="proba", limit=5)
    assert [h["table_name"] for h in hits] == [row["table_name"]], hits
    # A row with no aliases key (every row written before dedup) still scores.
    hits = await admin.search_datasets(q="AT_regional_2022", limit=5)
    assert hits[0]["table_name"] == CATALOG[1]["table_name"], hits


async def main():
    tests = [
        test_symbol_finds_its_own_dataset,
        test_short_symbol_does_not_match_prose,
        test_ranking_prefers_symbol_then_filename,
        test_no_match_is_an_empty_list_not_an_error,
        test_blank_query_lists_the_catalog,
        test_limit_is_honoured,
        test_catalog_boilerplate_is_not_a_match,
        test_database_down_is_503_not_empty,
        test_response_shape_is_what_load_dataset_reads,
        test_table_returns_rows_load_dataset_can_frame,
        test_truncation_is_flagged,
        test_pages_have_a_stable_order,
        test_offset_pages,
        test_unknown_table_is_404,
        test_table_name_cannot_escape_the_catalog,
        test_alias_finds_the_original_table,
    ]
    for test in tests:
        await test()
        print("ok  %s" % test.__name__)
    print("\n%d admin data checks passed" % len(tests))


if __name__ == "__main__":
    asyncio.run(main())
