"""
End-to-end check of the structured (CSV/XLSX) path, one boundary at a time.

    # from the VPS host, no rebuild needed — the script goes in over stdin:
    docker compose exec -T haystack-api python - < haystack-api/scripts/verify_structured.py

    # once the image carries this file:
    docker compose exec haystack-api python scripts/verify_structured.py
    docker compose exec haystack-api python scripts/verify_structured.py --keep

Uploads a small CSV, then asks each component in turn whether it actually
arrived: the job store, PostgreSQL, Qdrant, and the two admin endpoints the
panel draws from. Every step prints PASS or FAIL with the number it saw, so a
run says *where* the chain breaks rather than that it broke.

Why a live upload instead of inspecting the last real one: a file that never
made it leaves nothing to inspect, which is the failure mode this exists to
diagnose. A synthetic file with a known row count turns "nothing is there" into
"nothing arrived at step 4".

The test file is deleted through DELETE /admin/documents at the end, which is
also the only way to exercise that the delete drops the PostgreSQL tables and
the Qdrant summary together. --keep leaves it in place to look at in the panel.

Costs one embedding call. Safe to re-run: each run uses a fresh filename, and
nothing here touches a file it did not upload.
"""

import argparse
import asyncio
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path

# `python - < script` is the no-rebuild way to run this, and that leaves no
# __file__ to locate the package from — fall back to the image's WORKDIR.
ROOT = (Path(__file__).resolve().parent.parent if "__file__" in globals()
        else Path("/app"))
sys.path.insert(0, str(ROOT))

import config

API = "http://localhost:8000"
POLL_SECONDS = 2.0
POLL_ATTEMPTS = 45          # 90s: a small file embeds fast, a throttled key does not

# Symbol and Time are deliberate: they are what date_range() and symbols() key
# off, so a pass here means the summary carries the fields BM25 matches a bare
# year and a symbol against, not just that rows landed.
TEST_CSV = (
    "Symbol,Time,Open,High,Low,Close,Volume\n"
    "ACME,2026-03-15 09:30:00,18000.25,18010.50,17995.00,18005.75,1250\n"
    "ACME,2026-03-15 09:31:00,18005.75,18020.00,18001.25,18018.50,980\n"
    "ACME,2026-03-16 09:30:00,18018.50,18040.75,18015.00,18035.25,1440\n"
).encode("utf-8")
TEST_ROWS = 3


class Report:
    """Ordered PASS/FAIL lines plus the first failure, which is the answer."""

    def __init__(self):
        self.first_failure = None
        self.steps = 0

    def ok(self, step: str, detail: str = "") -> bool:
        self.steps += 1
        print("  PASS  %-34s %s" % (step, detail))
        return True

    def fail(self, step: str, detail: str = "") -> bool:
        self.steps += 1
        print("  FAIL  %-34s %s" % (step, detail))
        if self.first_failure is None:
            self.first_failure = (step, detail)
        return False

    def note(self, step: str, detail: str = "") -> None:
        print("  ----  %-34s %s" % (step, detail))


def check_build(report: Report) -> bool:
    """Is this container serving the wired endpoint or the old stub?

    The stub answered 200 with {"status": "not_implemented"}, and the Open WebUI
    filter logs only the status code — so on the old build every upload looks
    like a success from the outside and writes nothing. Reading the deployed
    source is the one check that cannot be fooled by that.
    """
    try:
        from routers import ingest
        source = inspect.getsource(ingest.ingest_structured)
    except Exception as exc:
        return report.fail("build: /ingest/structured", "could not inspect: %s" % exc)

    if "not_implemented" in source:
        return report.fail(
            "build: /ingest/structured",
            "STUB — this image predates 7d4ce7e; rebuild haystack-api")
    return report.ok("build: /ingest/structured", "wired")


def check_config(report: Report) -> None:
    """Config that fails silently later, surfaced now.

    Neither is fatal to the run — the point is that a pass at step 5 with a stub
    embedder is a summary nobody will ever retrieve, and that is worth seeing on
    the same screen as the pass.
    """
    dsn = config.DATABASE_URL or ""
    shown = dsn
    if "@" in dsn and "//" in dsn:
        head, tail = dsn.split("//", 1)
        creds, host = tail.split("@", 1)
        user = creds.split(":", 1)[0]
        shown = "%s//%s:***@%s" % (head, user, host)
    if "changeme" in dsn:
        report.fail("config: DATABASE_URL",
                    "%s — the config.py fallback, DATABASE_URL is not set" % shown)
    else:
        report.ok("config: DATABASE_URL", shown)

    try:
        from services.embedding_service import create_embedding_provider
        provider = create_embedding_provider()
        name = type(provider).__name__
        if "stub" in name.lower():
            report.fail("config: embedding provider",
                        "%s — EMBEDDING_API_KEY unset, every vector written is noise" % name)
        else:
            report.ok("config: embedding provider",
                      "%s (%s, %d dims)" % (name, config.EMBEDDING_MODEL,
                                            provider.dimensions))
    except Exception as exc:
        report.fail("config: embedding provider", "%s: %s" % (type(exc).__name__, exc))


async def upload(client, filename: str, report: Report):
    """POST the file. Returns the job id, or None when the endpoint refused it."""
    try:
        response = await client.post(
            "%s/ingest/structured" % API,
            files={"file": (filename, TEST_CSV)},
            data={"metadata": '{"origin": "verify_structured"}'},
            timeout=60.0)
    except Exception as exc:
        report.fail("POST /ingest/structured", "%s: %s" % (type(exc).__name__, exc))
        return None

    body = {}
    try:
        body = response.json()
    except ValueError:
        pass

    # The stub's tell: HTTP 200 and no job to poll. Named explicitly because
    # "no job_id" on a 200 is otherwise a confusing thing to read.
    if body.get("status") == "not_implemented":
        report.fail("POST /ingest/structured",
                    "200 not_implemented — old build, nothing was written")
        return None
    if response.status_code >= 400:
        report.fail("POST /ingest/structured",
                    "%d %s" % (response.status_code, response.text[:160]))
        return None
    job_id = body.get("job_id")
    if not job_id:
        report.fail("POST /ingest/structured",
                    "%d but no job_id: %s" % (response.status_code, str(body)[:160]))
        return None

    report.ok("POST /ingest/structured",
              "job %s, format %s" % (job_id, body.get("format")))
    return job_id


async def wait_for_job(client, job_id: str, report: Report) -> dict:
    """Poll until the job leaves the queue. Returns the final job record."""
    job = {}
    for _ in range(POLL_ATTEMPTS):
        try:
            response = await client.get("%s/ingest/status/%s" % (API, job_id), timeout=15.0)
            job = response.json() if response.status_code == 200 else {}
        except Exception as exc:
            report.fail("GET /ingest/status", "%s: %s" % (type(exc).__name__, exc))
            return {}
        if job.get("status") in ("done", "failed"):
            break
        await asyncio.sleep(POLL_SECONDS)

    status = job.get("status")
    if status == "done":
        report.ok("job status", "done, %s summary chunk(s) indexed"
                  % job.get("chunks_created"))
    elif status == "failed":
        # The error is the whole point of the run when it gets this far: the
        # rows may already be in PostgreSQL and the embedding is what died.
        report.fail("job status", "failed — %s" % (job.get("error") or "no error recorded"))
    else:
        report.fail("job status",
                    "still %s after %ds" % (status or "unknown",
                                            int(POLL_SECONDS * POLL_ATTEMPTS)))
    return job


async def check_postgres(filename: str, report: Report, expect: bool = True) -> list:
    """Catalog rows for this file, and a live count from each table it names."""
    from services.db_service import get_db

    db = get_db()
    try:
        rows = await db.list_tables(filename=filename)
    except Exception as exc:
        report.fail("postgres: data_catalog", "%s: %s" % (type(exc).__name__, exc))
        return []

    if not expect:
        if rows:
            report.fail("postgres: after delete",
                        "%d catalog row(s) survived" % len(rows))
        else:
            report.ok("postgres: after delete", "catalog clear")
        return rows

    if not rows:
        return report.fail("postgres: data_catalog", "no row for %s" % filename) or []

    detail = []
    for row in rows:
        table = row.get("table_name")
        try:
            live = await db.row_count(table)
        except Exception as exc:
            detail.append("%s (count failed: %s)" % (table, exc))
            continue
        detail.append("%s = %d row(s)" % (table, live))
        if live != TEST_ROWS:
            report.fail("postgres: row count",
                        "%s holds %d, expected %d" % (table, live, TEST_ROWS))
    report.ok("postgres: data_catalog", "; ".join(detail))
    return rows


async def check_qdrant(filename: str, report: Report, expect: bool = True) -> list:
    """Summary points for this file, queried the way the panel's delete does.

    expand_by_source reads both payload shapes, so this cannot pass or fail for
    the wrong reason on a collection the backfill has only partly converted.
    """
    from services.qdrant_service import get_qdrant

    try:
        points = await get_qdrant().expand_by_source(config.COLLECTION_DOCUMENTS, [filename])
    except Exception as exc:
        report.fail("qdrant: summary point", "%s: %s" % (type(exc).__name__, exc))
        return []

    if not expect:
        if points:
            report.fail("qdrant: after delete", "%d point(s) survived" % len(points))
        else:
            report.ok("qdrant: after delete", "collection clear")
        return points

    if not points:
        return report.fail(
            "qdrant: summary point",
            "nothing in %s for %s" % (config.COLLECTION_DOCUMENTS, filename)) or []

    payload = points[0].get("payload") or {}
    report.ok("qdrant: summary point",
              "%d point(s), source_type=%s, table=%s"
              % (len(points), payload.get("source_type"), payload.get("table_name")))

    # The summary is what the retriever matches on. An empty one is a point that
    # exists and can never be found, which a count alone would call a pass.
    content = payload.get("content") or ""
    if len(content) < 80:
        report.fail("qdrant: summary text", "%d chars — too short to retrieve" % len(content))
    else:
        report.ok("qdrant: summary text", "%d chars" % len(content))
    return points


async def check_panel(client, filename: str, report: Report, expect: bool = True) -> None:
    """The two endpoints the admin panel actually calls.

    Steps 4 and 5 can pass while this fails — that is the difference between
    "it is stored" and "you can see it", which is the second half of the ask.
    """
    label = "admin: KB browser" if expect else "admin: KB browser after delete"
    try:
        response = await client.get("%s/admin/documents" % API, timeout=30.0,
                                    params={"collection": config.COLLECTION_DOCUMENTS,
                                            "per_page": 200})
        body = response.json()
    except Exception as exc:
        report.fail(label, "%s: %s" % (type(exc).__name__, exc))
        return

    documents = body.get("documents") or []
    match = next((d for d in documents
                  if (d.get("source") or d.get("filename")) == filename), None)
    total = body.get("total", len(documents))

    if expect:
        if match:
            report.ok(label, "listed as %s, %s chunk(s), of %s document(s)"
                      % (match.get("source_type"), match.get("chunk_count"), total))
        else:
            # per_page=200 is above the panel's default 50, so a miss here is a
            # real absence rather than the alphabetical paging that hides a file
            # from page 1.
            report.fail(label, "not among %s document(s) in %s"
                        % (total, config.COLLECTION_DOCUMENTS))
    else:
        if match:
            report.fail(label, "still listed after delete")
        else:
            report.ok(label, "gone, %s document(s) remain" % total)

    label = "admin: spreadsheet page" if expect else "admin: spreadsheet after delete"
    try:
        response = await client.get("%s/admin/tables" % API, timeout=30.0)
        listing = response.json()
        tables = listing.get("tables", listing) if isinstance(listing, dict) else listing
    except Exception as exc:
        report.fail(label, "%s: %s" % (type(exc).__name__, exc))
        return

    mine = [t for t in (tables or []) if t.get("filename") == filename]
    if expect:
        if mine:
            report.ok(label, "%s" % ", ".join(t.get("table_name", "?") for t in mine))
        else:
            report.fail(label, "not among %d table(s)" % len(tables or []))
    else:
        if mine:
            report.fail(label, "%d table(s) still listed" % len(mine))
        else:
            report.ok(label, "gone, %d table(s) remain" % len(tables or []))


async def delete(client, filename: str, report: Report) -> None:
    """Remove the test file the way the panel's button does."""
    try:
        response = await client.delete(
            "%s/admin/documents/%s" % (API, filename), timeout=60.0,
            params={"collection": config.COLLECTION_DOCUMENTS})
    except Exception as exc:
        report.fail("DELETE /admin/documents", "%s: %s" % (type(exc).__name__, exc))
        return

    if response.status_code != 200:
        report.fail("DELETE /admin/documents",
                    "%d %s" % (response.status_code, response.text[:160]))
        return

    body = response.json()
    dropped = body.get("tables_dropped") or []
    report.ok("DELETE /admin/documents",
              "%s chunk(s), %d table(s): %s"
              % (body.get("chunks_removed"), len(dropped), ", ".join(dropped) or "none"))


async def run(keep: bool) -> int:
    import httpx

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    filename = "kbcheck_%s.csv" % stamp
    report = Report()

    print("\nStructured ingestion check — %s" % filename)
    print("collection: %s | qdrant: %s\n" % (config.COLLECTION_DOCUMENTS, config.QDRANT_URL))

    print("build and config")
    wired = check_build(report)
    check_config(report)
    if not wired:
        # Uploading to the stub proves nothing further and leaves a job that
        # never existed. Stop at the finding that explains everything after it.
        print("\nStopping: the container is serving the pre-7d4ce7e stub.")
        print("Fix: git pull to f9eb26b or later on the VPS, then")
        print("     docker compose up -d --build haystack-api")
        return 1

    async with httpx.AsyncClient() as client:
        print("\ningestion")
        job_id = await upload(client, filename, report)
        if job_id is None:
            return 1
        job = await wait_for_job(client, job_id, report)

        print("\nstorage")
        await check_postgres(filename, report)
        await check_qdrant(filename, report)

        print("\nadmin panel")
        await check_panel(client, filename, report)

        if keep:
            print("\n--keep: %s left in place. Delete it from the panel when done."
                  % filename)
        else:
            print("\ndelete")
            await delete(client, filename, report)
            await check_postgres(filename, report, expect=False)
            await check_qdrant(filename, report, expect=False)
            await check_panel(client, filename, report, expect=False)

        if job.get("status") == "failed":
            report.note("job error", str(job.get("error"))[:300])

    print("\n%s" % ("-" * 68))
    if report.first_failure is None:
        print("ALL %d CHECKS PASSED — upload lands in PostgreSQL and Qdrant, the "
              "panel shows it," % report.steps)
        print("and deleting it clears both stores.")
        return 0
    step, detail = report.first_failure
    print("FIRST FAILURE: %s — %s" % (step, detail))
    print("Everything after that point is downstream of it; fix this one first.")
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--keep", action="store_true",
                        help="leave the test file in place instead of deleting it")
    args = parser.parse_args(argv)
    return asyncio.run(run(args.keep))


if __name__ == "__main__":
    sys.exit(main())
