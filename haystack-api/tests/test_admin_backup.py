"""GET /admin/backup — the latest ops_backup_reports row and the latest
successful one, graded (stale = last success older than 26 h).

Mirrors test_admin_storage.py: FakeDB instead of a database, the module's own
clock injected, assertions on behaviour including "a failed backup is never
reported as a successful recovery point".

Run: cd haystack-api && python tests/test_admin_backup.py
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from routers import admin

NOW = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)


def run(started_ago=timedelta(minutes=30), ok=True, status=None, snapshot_id="ab12cd34",
        error=None, detail=None, as_text=False):
    started = NOW - started_ago
    row = {"id": 1, "started_at": started, "finished_at": started + timedelta(minutes=5),
           "ok": ok, "status": status or ("success" if ok else "failed"),
           "snapshot_id": snapshot_id if ok else None,
           "bytes_added": 18432000 if ok else None,
           # The script writes a failing-step description here, never secrets.
           "error": error if error is not None else (None if ok else "pg_dump failed"),
           "detail": detail if detail is not None else
                     {"stage_dump_bytes": 41234567, "collections": 6}}
    if as_text:   # asyncpg hands jsonb back as text (db_service registers no codec)
        text_row = dict(row)
        text_row["detail"] = json.dumps(text_row["detail"])
        return text_row
    return row


class FakeDB:
    def __init__(self, latest=None, success=None, table=True, broken=False):
        self.latest, self.success = latest, success
        self.table, self.broken = table, broken
        self.sql = []

    async def query(self, sql, *args, limit=1000):
        if self.broken:
            from services.db_service import DBError
            raise DBError("connection refused")
        self.sql.append(sql)
        if "to_regclass" in sql:
            return [{"present": self.table}]
        assert self.table, "selected from a table that does not exist"
        if "WHERE ok" in sql:
            return [self.success] if self.success else []
        return [self.latest] if self.latest else []


def use(db):
    admin.get_db = lambda: db
    admin._storage_now = lambda: NOW
    return db


async def test_thresholds_are_module_constants():
    assert admin.BACKUP_STALE_AFTER == timedelta(hours=26)


async def test_a_fresh_success_is_healthy():
    report = admin.backup_report(run(), run(), NOW)
    assert report["stale"] is False and report["alert"] is False, report
    assert report["status"] == "success" and report["latest_run"]["ok"] is True
    assert report["success_age_seconds"] == 30 * 60


async def test_no_run_at_all_is_stale_and_alerts():
    report = admin.backup_report(None, None, NOW)
    assert report["stale"] is True and report["alert"] is True
    assert report["status"] == "never" and report["latest_run"] is None
    assert report["last_success_at"] is None and report["success_age_seconds"] is None


async def test_success_older_than_26_hours_is_stale():
    ok = run(started_ago=timedelta(hours=27))
    report = admin.backup_report(ok, ok, NOW)
    assert report["stale"] is True and report["alert"] is True, report
    assert report["success_age_seconds"] == 27 * 3600


async def test_success_at_26_hours_is_still_fresh():
    ok = run(started_ago=timedelta(hours=26))
    report = admin.backup_report(ok, ok, NOW)
    assert report["stale"] is False and report["alert"] is False, report


async def test_a_failed_run_alerts_even_with_a_fresh_success():
    latest = run(started_ago=timedelta(minutes=10), ok=False)
    last_ok = run(started_ago=timedelta(hours=20))
    report = admin.backup_report(latest, last_ok, NOW)
    assert report["status"] == "failed" and report["alert"] is True, report
    assert report["stale"] is False, report   # data is still protected
    assert report["last_success"]["ok"] is True


async def test_a_failed_run_is_never_reported_as_a_recovery_point():
    latest = run(started_ago=timedelta(minutes=10), ok=False, snapshot_id=None)
    report = admin.backup_report(latest, None, NOW)
    assert report["latest_run"]["ok"] is False
    assert report["latest_run"]["snapshot_id"] is None, report
    assert report["stale"] is True, report    # no success -> stale


async def test_failure_detail_is_displayed_but_only_what_the_script_wrote():
    latest = run(ok=False, error="qdrant: could not create snapshot for ocr_frames")
    report = admin.backup_report(latest, latest, NOW)
    assert report["latest_run"]["error"] == \
        "qdrant: could not create snapshot for ocr_frames"


async def test_failure_detail_is_truncated_for_display():
    latest = run(ok=False, error="x" * 500)
    report = admin.backup_report(latest, latest, NOW)
    assert report["latest_run"]["error"] == "x" * 300, len(report["latest_run"]["error"])


async def test_jsonb_returned_as_text_is_decoded():
    report = admin.backup_report(run(as_text=True), run(as_text=True), NOW)
    assert report["latest_run"]["detail"]["collections"] == 6, report


async def test_snapshot_id_and_sizes_are_exposed():
    report = admin.backup_report(run(), run(), NOW)
    assert report["latest_run"]["snapshot_id"] == "ab12cd34"
    assert report["latest_run"]["bytes_added"] == 18432000


async def test_threshold_is_reported_for_the_panel():
    report = admin.backup_report(None, None, NOW)
    assert report["thresholds"]["stale_after_seconds"] == 26 * 3600


async def test_endpoint_reads_latest_and_latest_success():
    db = use(FakeDB(latest=run(started_ago=timedelta(minutes=5), ok=False),
                    success=run(started_ago=timedelta(hours=2))))
    report = await admin.get_backup()
    assert report["status"] == "failed" and report["stale"] is False, report
    assert report["last_success"]["ok"] is True
    assert any("ORDER BY started_at DESC" in s for s in db.sql), db.sql
    assert any("WHERE ok" in s for s in db.sql), db.sql


async def test_endpoint_before_the_first_run_is_stale_not_500():
    db = use(FakeDB(latest=None, success=None, table=False))
    report = await admin.get_backup()
    assert report["stale"] is True and report["status"] == "never"
    assert len(db.sql) == 1, db.sql   # only the to_regclass probe


async def test_endpoint_with_an_empty_table_is_stale():
    db = use(FakeDB(latest=None, success=None))
    report = await admin.get_backup()
    assert report["stale"] is True and report["latest_run"] is None
    assert len(db.sql) == 3, db.sql


async def test_database_down_is_503_not_a_healthy_report():
    use(FakeDB(broken=True))
    try:
        await admin.get_backup()
    except HTTPException as exc:
        assert exc.status_code == 503, exc.status_code
        return
    raise AssertionError("expected HTTP 503, got a normal return")


async def main():
    tests = [
        test_thresholds_are_module_constants,
        test_a_fresh_success_is_healthy,
        test_no_run_at_all_is_stale_and_alerts,
        test_success_older_than_26_hours_is_stale,
        test_success_at_26_hours_is_still_fresh,
        test_a_failed_run_alerts_even_with_a_fresh_success,
        test_a_failed_run_is_never_reported_as_a_recovery_point,
        test_failure_detail_is_displayed_but_only_what_the_script_wrote,
        test_failure_detail_is_truncated_for_display,
        test_jsonb_returned_as_text_is_decoded,
        test_snapshot_id_and_sizes_are_exposed,
        test_threshold_is_reported_for_the_panel,
        test_endpoint_reads_latest_and_latest_success,
        test_endpoint_before_the_first_run_is_stale_not_500,
        test_endpoint_with_an_empty_table_is_stale,
        test_database_down_is_503_not_a_healthy_report,
    ]
    for test in tests:
        await test()
        print("ok  %s" % test.__name__)
    print("\n%d admin backup checks passed" % len(tests))


if __name__ == "__main__":
    asyncio.run(main())
