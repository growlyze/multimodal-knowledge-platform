"""
/admin/stats, /admin/documents and /admin/documents/{source} — what the panel
counts the knowledge base with.

The case worth guarding is the two payload schemas. scripts/backfill_metadata.py
turns `source` from a filename string into an object, and it may not have run
when this deploys, so every one of these endpoints has to read a collection that
is half converted and still report one document per file rather than two.

    python tests/test_admin_kb.py

No Qdrant and no PostgreSQL: both are faked at the service boundary.
"""

import asyncio
import json

import httpx

from fastapi import HTTPException

from routers import admin
from services.qdrant_service import QdrantService, source_of

STAMP = "2025-08-28T10:00:00+00:00"
LATER = "2025-08-28T11:00:00+00:00"


def new_point(pid, filename, index, ingested_at=STAMP, content="new schema chunk"):
    """A point pipelines/ingestion.py writes today."""
    return {"id": pid, "payload": {
        "content": content,
        "source_type": "code_reference",
        "collection": "code_docs",
        "ingested_at": ingested_at,
        "origin": "scraper",
        "chunk_index": index,
        "source": {"filename": filename, "url": "https://docs.example.com/sdk-reference/#run_query",
                   "title": "run_query — SDK Reference", "kind": "function",
                   "name": "run_query", "version": "v2"},
    }}


def old_point(pid, filename, index, content="old schema chunk"):
    """A point written before the namespacing — `source` is the filename."""
    return {"id": pid, "payload": {
        "content": content,
        "source": filename,
        "url": "https://docs.example.com/guide/alerts",
        "title": "Alerts",
        "source_type": "code_docs",
        "scraped_at": STAMP,
        "chunk_index": index,
    }}


POINTS = {
    "code_docs": [
        new_point(1, "reference/run_query.md", 0, LATER),
        new_point(2, "reference/run_query.md", 1, STAMP),      # earliest wins
        new_point(3, "reference/run_query.md", 2, LATER, "x" * 500),
        old_point(4, "guide/alerts.md", 0),
        old_point(5, "guide/alerts.md", 1),
    ],
    "documents": [],
}


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP %d" % self.status_code)


def matches(payload, clause):
    """The subset of Qdrant filtering these endpoints use: match value or any.

    `source` on a namespaced point is an object, and Qdrant does not match a
    scalar against an object — the whole reason both keys have to be queried.
    """
    key, match = clause["key"], clause["match"]
    wanted = match["any"] if "any" in match else [match["value"]]
    if key == "source.filename":
        source = payload.get("source")
        return isinstance(source, dict) and source.get("filename") in wanted
    found = payload.get(key)
    return not isinstance(found, dict) and found in wanted


class FakeClient:
    """Stands in for QdrantService.client. `broken` fails the way httpx does —
    with ConnectError, which is not an OSError.
    """

    def __init__(self, store, collections=None, broken=False):
        self.store = store
        self.collections = collections if collections is not None else list(store)
        self.broken = broken
        self.payload_asks = []
        self.deleted = []

    async def get(self, path):
        if self.broken:
            raise httpx.ConnectError("connection refused")
        if path == "/collections":
            return FakeResponse(200, {"result": {
                "collections": [{"name": n} for n in self.collections]}})
        name = path.rsplit("/", 1)[-1]
        if name not in self.collections:
            return FakeResponse(404)
        return FakeResponse(200, {"result": {
            "points_count": len(self.store.get(name, [])), "status": "green"}})

    async def post(self, path, json=None):
        if self.broken:
            raise httpx.ConnectError("connection refused")
        name = path.split("/")[2]
        if name not in self.collections:
            return FakeResponse(404)
        self.payload_asks.append(json.get("with_payload"))

        points = self.select(name, json.get("filter"))
        asked = json.get("with_payload")
        if isinstance(asked, list):
            # Qdrant returns only the named payload keys; so must the fake, or a
            # field missing from SOURCE_FIELDS goes unnoticed.
            points = [dict(p, payload={k: v for k, v in p["payload"].items() if k in asked})
                      for p in points]
        # Every fixture fits one page; paging itself is Qdrant's, not ours.
        return FakeResponse(200, {"result": {"points": points, "next_page_offset": None}})

    def select(self, name, qfilter):
        points = self.store.get(name, [])
        for clause in (qfilter or {}).get("must") or []:
            points = [p for p in points if matches(p["payload"], clause)]
        return points

    async def request(self, method, path, params=None, json=None):
        """QdrantService._request()'s socket — count and delete go through here.

        Deleting really removes from `store`, so a test can assert what survived
        rather than only what the endpoint claimed.
        """
        if self.broken:
            raise httpx.ConnectError("connection refused")
        name = path.split("/")[2]
        if name not in self.collections:
            return FakeResponse(404)

        matched = self.select(name, (json or {}).get("filter"))
        if path.endswith("/points/count"):
            return FakeResponse(200, {"result": {"count": len(matched)}})
        if path.endswith("/points/delete"):
            self.deleted += [p["id"] for p in matched]
            gone = {id(p) for p in matched}
            self.store[name] = [p for p in self.store.get(name, [])
                                if id(p) not in gone]
            return FakeResponse(200, {"result": {"status": "completed"}})
        raise AssertionError("unexpected Qdrant call: %s %s" % (method, path))


class FakeDB:
    def __init__(self, tables=2, broken=False):
        self.tables, self.broken = tables, broken
        self.dropped = []

    async def list_tables(self, filename=None):
        if self.broken:
            from services.db_service import DBError
            raise DBError("connection refused")
        return [{"table_name": "data_%d" % i} for i in range(self.tables)]

    async def drop_by_filename(self, filename):
        # db_service's own drop_by_filename is list_tables + drop_table; the
        # names it returns are all this endpoint reports.
        self.dropped = [row["table_name"] for row in await self.list_tables(filename)]
        return self.dropped


class FakeQdrant(QdrantService):
    """The real service with a fake socket under it, so collection_info() and
    expand_by_source() are the shipping code rather than a second copy of it."""

    def __init__(self, client):
        super().__init__("http://fake")
        self._client = client

    @property
    def client(self):
        return self._client


def use(store=None, collections=None, broken=False, tables=2, db_broken=False):
    # A copy of POINTS, never POINTS itself: delete_document really removes from
    # the store, and a shared fixture would leak that into the next test.
    if store is None:
        store = {name: list(points) for name, points in POINTS.items()}
    client = FakeClient(store, collections, broken)
    qdrant = FakeQdrant(client)
    # One FakeDB for the whole request, not one per get_db() call, so a test can
    # read back what was dropped.
    client.db = FakeDB(tables, db_broken)
    admin.get_qdrant = lambda: qdrant
    admin.get_db = lambda: client.db
    return client


async def raises(status, coro):
    try:
        await coro
    except HTTPException as exc:
        assert exc.status_code == status, exc.status_code
        return exc
    raise AssertionError("expected HTTP %d, got a normal return" % status)


# ── payload shapes ──────────────────────────────────────

async def test_both_schemas_yield_the_same_fields():
    namespaced = admin._source_fields(new_point(1, "reference/run_query.md", 0)["payload"])
    assert namespaced["filename"] == "reference/run_query.md"
    assert namespaced["url"] == "https://docs.example.com/sdk-reference/#run_query"
    assert namespaced["title"].startswith("run_query")
    assert namespaced["ingested_at"] == STAMP

    flat = admin._source_fields(old_point(4, "guide/alerts.md", 0)["payload"])
    assert flat["filename"] == "guide/alerts.md"
    assert flat["url"] == "https://docs.example.com/guide/alerts"
    assert flat["title"] == "Alerts"
    assert flat["source_type"] == "code_docs"
    assert flat["ingested_at"] == STAMP, "scraped_at stands in for ingested_at"

    assert admin._source_fields({})["filename"] == "", "an empty payload is not a crash"


async def test_a_half_backfilled_file_is_one_document_not_two():
    mixed = [new_point(1, "guide/alerts.md", 0), old_point(2, "guide/alerts.md", 1)]
    grouped = admin._group_by_source(mixed, "code_docs")
    assert len(grouped) == 1, grouped
    assert grouped[0]["chunk_count"] == 2


async def test_grouping_keeps_the_earliest_stamp():
    grouped = {d["source"]: d for d in
               admin._group_by_source(POINTS["code_docs"], "code_docs")}
    assert grouped["reference/run_query.md"]["ingested_at"] == STAMP, "not the LATER one"
    assert grouped["reference/run_query.md"]["chunk_count"] == 3
    assert grouped["guide/alerts.md"]["chunk_count"] == 2


async def test_a_source_less_chunk_does_not_vanish_silently():
    grouped = admin._group_by_source([{"id": 9, "payload": {"content": "orphan"}}], "c")
    assert [d["source"] for d in grouped] == ["(unknown source)"]


# ── /admin/stats ────────────────────────────────────────

async def test_stats_counts_chunks_and_documents():
    use()
    stats = await admin.get_stats()
    by_name = {c["name"]: c for c in stats["collections"]}
    assert by_name["code_docs"]["points_count"] == 5
    assert by_name["code_docs"]["documents"] == 2, "5 chunks, 2 source files"
    assert by_name["code_docs"]["status"] == "green"
    assert stats["total_chunks"] == 5
    assert stats["total_documents"] == 2
    assert stats["postgresql"]["structured_tables"] == 2


async def test_stats_does_not_pull_chunk_content_over_the_wire():
    client = use()
    await admin.get_stats()
    assert client.payload_asks, "the collections were never scrolled"
    for ask in client.payload_asks:
        assert ask == admin.SOURCE_FIELDS, ask
        assert "content" not in ask


async def test_a_configured_collection_that_does_not_exist_still_shows():
    use(collections=["code_docs"])
    stats = await admin.get_stats()
    missing = [c for c in stats["collections"] if c["name"] == "documents"]
    assert missing and missing[0] == {"name": "documents", "points_count": 0,
                                      "documents": 0, "status": "missing"}, missing
    assert stats["total_chunks"] == 5


async def test_an_unconfigured_collection_is_counted_too():
    store = dict(POINTS, scratch=[old_point(7, "notes.md", 0)])
    use(store)
    stats = await admin.get_stats()
    assert [c["name"] for c in stats["collections"]][:2] == [
        "documents", "code_docs"], "the configured pair leads"
    assert {c["name"] for c in stats["collections"]} == {
        "documents", "code_docs", "scratch"}
    assert stats["total_chunks"] == 6


async def test_qdrant_down_is_503_not_zero():
    use(broken=True)
    # A dead vector store must not render as an empty knowledge base — that is
    # exactly the "0 documents" this endpoint was fixed to stop showing.
    await raises(503, admin.get_stats())


async def test_a_down_database_does_not_blank_the_vector_counts():
    use(db_broken=True)
    stats = await admin.get_stats()
    assert stats["total_chunks"] == 5
    assert stats["postgresql"]["structured_tables"] == 0


# ── /admin/documents ────────────────────────────────────

async def test_documents_lists_sources_not_chunks():
    use()
    page = await admin.list_documents(collection="code_docs", page=1, per_page=20)
    assert page["total"] == 2, page
    names = [d["source"] for d in page["documents"]]
    assert names == ["guide/alerts.md", "reference/run_query.md"], names
    alerts, sma = page["documents"]
    assert alerts["chunk_count"] == 2 and alerts["url"] == "https://docs.example.com/guide/alerts"
    assert sma["chunk_count"] == 3 and sma["title"].startswith("run_query")
    assert sma["source_type"] == "code_reference"
    assert sma["collection"] == "code_docs"


async def test_documents_spans_every_collection_when_none_is_named():
    use(dict(POINTS, scratch=[old_point(7, "notes.md", 0)]))
    page = await admin.list_documents(collection=None, page=1, per_page=20)
    assert page["total"] == 3, page
    assert {d["collection"] for d in page["documents"]} == {"code_docs", "scratch"}


async def test_pages_do_not_overlap_or_drop_a_document():
    use()
    first = await admin.list_documents(collection="code_docs", page=1, per_page=1)
    second = await admin.list_documents(collection="code_docs", page=2, per_page=1)
    third = await admin.list_documents(collection="code_docs", page=3, per_page=1)
    assert [d["source"] for d in first["documents"]] == ["guide/alerts.md"]
    assert [d["source"] for d in second["documents"]] == ["reference/run_query.md"]
    assert third["documents"] == [], "past the end is empty, not an error"
    assert first["pages"] == 2 and first["total"] == 2


async def test_an_empty_collection_is_an_empty_list_not_an_error():
    use()
    page = await admin.list_documents(collection="documents", page=1, per_page=20)
    assert page == {"collection": "documents", "documents": [], "total": 0,
                    "page": 1, "per_page": 20, "pages": 0}, page


async def test_documents_qdrant_down_is_503():
    use(broken=True)
    await raises(503, admin.list_documents(collection="code_docs",
                                           page=1, per_page=20))


# ── /admin/documents/{source} ───────────────────────────

async def test_chunks_of_a_namespaced_source():
    use()
    detail = await admin.get_document("reference/run_query.md", collection="code_docs")
    assert detail["chunk_count"] == 3, detail
    assert [c["chunk_index"] for c in detail["chunks"]] == [0, 1, 2], "chunk order"
    assert detail["chunks"][0]["content"] == "new schema chunk"
    assert detail["chunks"][0]["url"] == "https://docs.example.com/sdk-reference/#run_query"
    assert detail["chunks"][0]["ingested_at"] == LATER


async def test_chunks_of_an_old_flat_source():
    use()
    detail = await admin.get_document("guide/alerts.md", collection="code_docs")
    assert detail["chunk_count"] == 2, detail
    assert detail["chunks"][0]["title"] == "Alerts"
    assert detail["chunks"][0]["content"] == "old schema chunk"


async def test_content_is_truncated_and_says_so():
    use()
    detail = await admin.get_document("reference/run_query.md", collection="code_docs")
    long = detail["chunks"][2]
    assert len(long["content"]) == admin.CONTENT_PREVIEW == 200
    assert long["truncated"] is True
    assert detail["chunks"][0]["truncated"] is False


async def test_a_file_in_both_schemas_is_not_counted_twice():
    # The state a backfill is halfway through: one filename, both payload shapes.
    use({"code_docs": [new_point(1, "guide/alerts.md", 1),
                             old_point(2, "guide/alerts.md", 0)]})
    detail = await admin.get_document("guide/alerts.md", collection="code_docs")
    assert detail["chunk_count"] == 2, detail
    assert [c["id"] for c in detail["chunks"]] == [2, 1], "ordered by chunk_index"


async def test_unknown_source_is_404():
    use()
    await raises(404, admin.get_document("reference/nope.md", collection="code_docs"))


async def test_a_chunk_without_an_index_sorts_the_way_the_retriever_expands_it():
    # qdrant_service.chunk_index() reads a missing index as 0, so it sorts first.
    # The panel and the retriever must agree; one ordering, not two.
    point = old_point(9, "guide/alerts.md", 0)
    point["payload"].pop("chunk_index")
    use({"code_docs": [point, old_point(8, "guide/alerts.md", 1)]})
    detail = await admin.get_document("guide/alerts.md", collection="code_docs")
    assert [c["id"] for c in detail["chunks"]] == [9, 8], detail
    assert detail["chunks"][0]["chunk_index"] is None, "the real value is still reported"


# ── DELETE /admin/documents/{source} ────────────────────

async def test_delete_removes_every_chunk_of_one_file_and_nothing_else():
    client = use()
    result = await admin.delete_document("guide/alerts.md",
                                         collection="code_docs")
    assert result["chunks_removed"] == 2, result
    assert result["collections"] == ["code_docs"], result
    survivors = [source_of(p["payload"]) for p in client.store["code_docs"]]
    assert survivors == ["reference/run_query.md"] * 3, survivors


async def test_delete_clears_both_payload_schemas():
    # The half-backfilled case: one filename, one namespaced point and one flat.
    # A delete that queried a single key would leave the other behind forever.
    client = use({"code_docs": [new_point(1, "guide/alerts.md", 1),
                                      old_point(2, "guide/alerts.md", 0)]})
    result = await admin.delete_document("guide/alerts.md",
                                         collection="code_docs")
    assert result["chunks_removed"] == 2, result
    assert client.store["code_docs"] == [], client.store


async def test_delete_spans_every_collection_when_none_is_named():
    client = use({"code_docs": [old_point(1, "shared.md", 0)],
                  "documents": [old_point(2, "shared.md", 0)]})
    result = await admin.delete_document("shared.md", collection=None)
    assert result["chunks_removed"] == 2, result
    assert sorted(result["collections"]) == ["code_docs", "documents"]
    assert not any(client.store.values()), client.store


async def test_deleting_in_one_collection_leaves_the_other_copy():
    client = use({"code_docs": [old_point(1, "shared.md", 0)],
                  "documents": [old_point(2, "shared.md", 0)]})
    await admin.delete_document("shared.md", collection="code_docs")
    assert client.store["code_docs"] == []
    assert len(client.store["documents"]) == 1, "a named collection is a scope"


async def test_deleting_an_unknown_source_is_404_not_a_silent_success():
    # 0 chunks removed and "no such document" are the same number, so the
    # lookup has to happen before the delete or a typo reports success.
    client = use()
    await raises(404, admin.delete_document("reference/nope.md",
                                            collection="code_docs"))
    assert client.deleted == [], "nothing may be deleted on the way to a 404"


async def test_delete_qdrant_down_is_503():
    use(broken=True)
    await raises(503, admin.delete_document("guide/alerts.md",
                                            collection="code_docs"))


async def test_deleting_a_dataset_drops_its_tables_too():
    # A structured upload is a summary chunk in Qdrant plus PostgreSQL tables.
    # Removing the chunk alone leaves tables text-to-SQL can still query and
    # this panel can no longer see.
    point = old_point(1, "sales.csv", 0)
    point["payload"]["source_type"] = "structured_data"
    client = use({"documents": [point]})
    result = await admin.delete_document("sales.csv", collection="documents")
    assert result["chunks_removed"] == 1, result
    assert result["tables_dropped"] == ["data_0", "data_1"], result
    assert client.db.dropped == ["data_0", "data_1"]


async def test_deleting_a_document_does_not_touch_the_data_tables():
    client = use()
    result = await admin.delete_document("guide/alerts.md",
                                         collection="code_docs")
    assert result["tables_dropped"] == [], result
    assert client.db.dropped == [], "only structured_data owns tables"


async def test_a_dataset_whose_tables_will_not_drop_says_what_survived():
    # The chunks are already gone by then. Reporting a flat failure would send
    # someone looking for a document that is no longer there.
    point = old_point(1, "sales.csv", 0)
    point["payload"]["source_type"] = "structured_data"
    use({"documents": [point]}, db_broken=True)
    error = await raises(503, admin.delete_document("sales.csv",
                                                    collection="documents"))
    assert "data tables are still there" in error.detail, error.detail


def memory_job_store():
    """The real JobStore, pinned to its in-memory fallback — no PostgreSQL."""
    from pipelines import ingestion
    store = ingestion.JobStore(dsn="postgresql://unused")
    store._degraded = True
    ingestion._store = store
    return store


async def test_deleting_a_document_marks_its_ingestion_jobs_deleted():
    # P123: kestrel_test.txt was removed through this endpoint (the /kb/ panel's
    # "Yes, delete") minutes after each upload, yet GET /ingest/status/<id> kept
    # answering status=done, chunks_created=1 for both jobs. That "done" is what
    # sent the investigation hunting for a data-loss bug in ingestion — the job
    # row has to stop claiming the content is there once it is not.
    store = memory_job_store()
    await store.create("job-kept", "guide/alerts.md", "document", "code_docs")
    await store.update("job-kept", status="done", chunks_created=2)
    await store.create("job-other", "reference/run_query.md", "document", "code_docs")
    await store.update("job-other", status="done", chunks_created=3)
    await store.create("job-elsewhere", "guide/alerts.md", "document", "documents")
    await store.update("job-elsewhere", status="done", chunks_created=1)

    use()
    await admin.delete_document("guide/alerts.md", collection="code_docs")

    gone = await store.get("job-kept")
    assert gone["status"] == "deleted", gone
    assert "DELETE /admin/documents" in (gone["error"] or ""), gone
    assert (await store.get("job-other"))["status"] == "done", "another file's job"
    assert (await store.get("job-elsewhere"))["status"] == "done", \
        "a named collection is a scope for the job rows too"


async def test_mark_deleted_updates_the_postgres_rows_status_reads():
    # GET /ingest/status reads Postgres first, so flipping only the in-process
    # dict would leave the row the endpoint actually serves still saying done.
    from pipelines import ingestion
    from tests.test_structured import FakePool
    store = ingestion.JobStore(dsn="postgresql://unused")
    pool = FakePool()
    async with pool.acquire() as conn:
        await conn.execute(ingestion.SCHEMA)
    store._pool, store._degraded = pool, False
    await store.create("pg-job", "kestrel_test.txt", "document", "documents")
    await store.update("pg-job", status="done", chunks_created=1)
    await store.create("pg-failed", "kestrel_test.txt", "document", "documents")
    await store.update("pg-failed", status="failed", error="extraction failed")
    store._memory.clear()           # prove the answer comes from the table

    await store.mark_deleted("kestrel_test.txt", "documents")

    rows = {r["id"]: r for r in await pool.conn.fetch("SELECT * FROM ingestion_jobs")}
    row = rows["pg-job"]
    assert row["status"] == "deleted", row
    assert "DELETE /admin/documents" in row["error"], row
    failed = rows["pg-failed"]
    assert failed["status"] == "failed" and failed["error"] == "extraction failed", \
        "a job that never indexed anything keeps its own failure"
    await pool.close()


async def test_a_job_store_failure_never_fails_the_delete():
    # Degrade, never break: the chunks are already gone by the time the job rows
    # are touched, so a dead job store must not turn a real delete into a 503.
    class BrokenStore:
        async def mark_deleted(self, filename, collection):
            raise RuntimeError("job store down")

    from pipelines import ingestion
    ingestion._store = BrokenStore()
    client = use()
    result = await admin.delete_document("guide/alerts.md", collection="code_docs")
    assert result["chunks_removed"] == 2, result
    assert [source_of(p["payload"]) for p in client.store["code_docs"]] == \
        ["reference/run_query.md"] * 3
    memory_job_store()      # never leave a later test reaching for real Postgres


def video_point(pid, video_id, title, source_type="visual_frame"):
    """A point video ingestion writes: no `source`, identified by video_id."""
    return {"id": pid, "payload": {"video_id": video_id, "title": title, "source_type": source_type,
                                   "frame_ms": pid * 1000, "platform": "youtube"}}


def _video_store():
    return dict(POINTS,
                visual_frames=[video_point(10, "V1", "Episode 12"), video_point(11, "V1", "Episode 12"),
                               video_point(12, "V1", "Episode 12"), video_point(13, "V2", "Episode 36")],
                transcript_segments=[video_point(20, "V1", "Episode 12", "transcript_segment"),
                                     video_point(21, "V2", "Episode 36", "transcript_segment")])


async def test_stats_counts_each_ingested_video_once():
    """Video points carry video_id, not `source`, so each video collection used
    to collapse into one "(unknown source)" document, and the dashboard's
    headline added 1 per video collection -- 4 whatever the number of videos
    (2026-10-21: "1,155 sources" was right only because 4 videos were in)."""
    use(_video_store())
    stats = await admin.get_stats()
    by_name = {c["name"]: c for c in stats["collections"]}
    assert by_name["visual_frames"]["documents"] == 2, by_name["visual_frames"]
    assert by_name["transcript_segments"]["documents"] == 2, by_name["transcript_segments"]
    assert stats["total_videos"] == 2, stats
    assert stats["total_documents"] == 2 + 2, "2 source files + 2 videos, each video once"


async def test_video_documents_are_named_by_title():
    use(_video_store())
    grouped = await admin._documents_in(admin.get_qdrant(), "visual_frames")
    assert [(d["title"], d["chunk_count"]) for d in grouped] == [("Episode 12", 3), ("Episode 36", 1)], grouped


async def main():
    tests = [
        test_stats_counts_each_ingested_video_once,
        test_video_documents_are_named_by_title,
        test_deleting_a_document_marks_its_ingestion_jobs_deleted,
        test_mark_deleted_updates_the_postgres_rows_status_reads,
        test_a_job_store_failure_never_fails_the_delete,
        test_both_schemas_yield_the_same_fields,
        test_a_half_backfilled_file_is_one_document_not_two,
        test_grouping_keeps_the_earliest_stamp,
        test_a_source_less_chunk_does_not_vanish_silently,
        test_stats_counts_chunks_and_documents,
        test_stats_does_not_pull_chunk_content_over_the_wire,
        test_a_configured_collection_that_does_not_exist_still_shows,
        test_an_unconfigured_collection_is_counted_too,
        test_qdrant_down_is_503_not_zero,
        test_a_down_database_does_not_blank_the_vector_counts,
        test_documents_lists_sources_not_chunks,
        test_documents_spans_every_collection_when_none_is_named,
        test_pages_do_not_overlap_or_drop_a_document,
        test_an_empty_collection_is_an_empty_list_not_an_error,
        test_documents_qdrant_down_is_503,
        test_chunks_of_a_namespaced_source,
        test_chunks_of_an_old_flat_source,
        test_content_is_truncated_and_says_so,
        test_a_file_in_both_schemas_is_not_counted_twice,
        test_unknown_source_is_404,
        test_a_chunk_without_an_index_sorts_the_way_the_retriever_expands_it,
        test_delete_removes_every_chunk_of_one_file_and_nothing_else,
        test_delete_clears_both_payload_schemas,
        test_delete_spans_every_collection_when_none_is_named,
        test_deleting_in_one_collection_leaves_the_other_copy,
        test_deleting_an_unknown_source_is_404_not_a_silent_success,
        test_delete_qdrant_down_is_503,
        test_deleting_a_dataset_drops_its_tables_too,
        test_deleting_a_document_does_not_touch_the_data_tables,
        test_a_dataset_whose_tables_will_not_drop_says_what_survived,
    ]
    for test in tests:
        await test()
        print("ok  %s" % test.__name__)
    print("\n%d admin knowledge base checks passed" % len(tests))


if __name__ == "__main__":
    asyncio.run(main())
