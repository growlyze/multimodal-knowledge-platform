"""
/admin/storage — the latest row ops/storage_monitor.sh wrote, graded per probe.

The cases worth guarding are the silent ones. A full tmpfs fails writes with no
warning (P160), so a probe at 80% has to come back as "alert", not as a number
someone has to notice. And a dead timer writes nothing at all — the last report
then just keeps saying "12% used" forever — so a report older than two hours, or
none at all, has to come back `stale`, never as a healthy-looking old answer.

    PYTHONPATH=. python tests/test_admin_storage.py

No PostgreSQL: the report table is faked at the DBService boundary.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from routers import admin

NOW = datetime(2026, 10, 21, 12, 0, 0, tzinfo=timezone.utc)


def probe(name, pct, used=1, total=100, **extra):
    return dict({"name": name, "label": name.replace("_", " "), "used_bytes": used,
                 "total_bytes": total, "pct": pct, "path": "/" + name}, **extra)


def row(probes, age=timedelta(minutes=10), cleanup=None, as_text=False):
    cleanup = {"files_deleted": 3, "bytes_deleted": 4096} if cleanup is None else cleanup
    return {"id": 7, "reported_at": NOW - age,
            # asyncpg hands jsonb back as text unless a codec is registered, and
            # db_service registers none — the endpoint must decode it itself.
            "probes": json.dumps(probes) if as_text else probes,
            "cleanup": json.dumps(cleanup) if as_text else cleanup}


class FakeDB:
    def __init__(self, rows=None, table=True, broken=False):
        self.rows = [] if rows is None else rows
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
        return self.rows[:1]


def use(db):
    admin.get_db = lambda: db
    admin._storage_now = lambda: NOW
    return db


async def test_thresholds_grade_each_probe():
    report = admin.storage_report(row([probe("a", 69.9), probe("b", 70.0), probe("c", 79.9),
                                       probe("d", 80.0), probe("e", 100.0)]), NOW)
    assert [p["status"] for p in report["probes"]] == ["ok", "warn", "warn", "alert", "alert"], report


async def test_thresholds_are_module_constants():
    assert (admin.STORAGE_WARN_PCT, admin.STORAGE_ALERT_PCT) == (70, 80)
    assert admin.STORAGE_STALE_AFTER == timedelta(hours=2)


async def test_a_probe_can_carry_its_own_percent_thresholds():
    # The host disk warns at 75 / alerts at 90 (one disk under everything, so
    # 80% of 415 GB is still 80 GB of headroom); the tmpfs probes keep 70/80.
    host = dict(warn_pct=75, alert_pct=90)
    report = admin.storage_report(row([probe("h1", 74.9, **host), probe("h2", 80.0, **host),
                                       probe("h3", 90.0, **host)]), NOW)
    assert [p["status"] for p in report["probes"]] == ["ok", "warn", "alert"], report
    assert report["alert"] is True


GB = 1000 ** 3


async def test_an_absolute_byte_threshold_grades_a_store_with_no_total():
    # Postgres, Qdrant, the video work dir: no capacity of their own, so a
    # percentage can never fire. Only a byte ceiling makes them alertable.
    pg = dict(total=None, alert_bytes=5 * GB)
    report = admin.storage_report(row([probe("pg_small", None, used=4 * GB, **pg),
                                       probe("pg_big", None, used=5 * GB, **pg)]), NOW)
    assert [p["status"] for p in report["probes"]] == ["ok", "alert"], report


async def test_a_warn_only_byte_threshold_never_alerts():
    cache = dict(total=None, warn_bytes=20 * GB)
    report = admin.storage_report(row([probe("build_cache", None, used=50 * GB, **cache)]), NOW)
    assert report["probes"][0]["status"] == "warn" and report["alert"] is False, report


async def test_the_worse_of_percent_and_bytes_wins():
    both = probe("both", 10.0, used=30 * GB, total=300 * GB, alert_bytes=25 * GB)
    assert admin.storage_report(row([both]), NOW)["probes"][0]["status"] == "alert"


async def test_a_store_without_a_capacity_is_ok_not_an_alert():
    # Postgres/Qdrant/log sizes have no ceiling of their own — the filesystem
    # probe under them carries the percentage.
    report = admin.storage_report(row([probe("postgres", None, used=5, total=None)]), NOW)
    assert report["probes"][0]["status"] == "ok", report
    assert report["alert"] is False


async def test_a_failed_probe_is_error_and_raises_the_alert():
    # The script records a probe that could not be measured rather than dropping
    # it — a missing row would read as "nothing to worry about".
    failed = probe("sandbox_tmp", None, used=None, total=None, error="container not running")
    report = admin.storage_report(row([probe("host_root", 12.0), failed]), NOW)
    assert [p["status"] for p in report["probes"]] == ["ok", "error"], report
    assert report["alert"] is True and report["worst"] == "error"


async def test_alert_is_raised_at_eighty_percent_not_at_seventy():
    assert admin.storage_report(row([probe("a", 79.9)]), NOW)["alert"] is False
    assert admin.storage_report(row([probe("a", 80.0)]), NOW)["alert"] is True
    assert admin.storage_report(row([probe("a", 75.0)]), NOW)["worst"] == "warn"


async def test_a_fresh_report_is_not_stale():
    report = admin.storage_report(row([probe("a", 1.0)], age=timedelta(minutes=59)), NOW)
    assert report["stale"] is False and report["alert"] is False, report
    assert report["age_seconds"] == 59 * 60
    assert report["reported_at"] == (NOW - timedelta(minutes=59)).isoformat()


async def test_a_report_older_than_two_hours_is_stale_and_alerts():
    report = admin.storage_report(row([probe("a", 1.0)], age=timedelta(hours=2, seconds=1)), NOW)
    assert report["stale"] is True and report["alert"] is True, report
    # The old numbers are still returned — they are the best we have — but
    # flagged, so the panel can say how old they are.
    assert report["probes"][0]["pct"] == 1.0


async def test_no_report_at_all_is_stale():
    report = admin.storage_report(None, NOW)
    assert report["stale"] is True and report["alert"] is True
    assert report["probes"] == [] and report["reported_at"] is None and report["cleanup"] is None


async def test_jsonb_returned_as_text_is_decoded():
    report = admin.storage_report(row([probe("a", 85.0)], as_text=True), NOW)
    assert report["probes"][0]["status"] == "alert", report
    assert report["cleanup"] == {"files_deleted": 3, "bytes_deleted": 4096}


async def test_endpoint_reads_the_newest_row():
    db = use(FakeDB(rows=[row([probe("host_root", 12.0)])]))
    report = await admin.get_storage()
    assert report["stale"] is False and report["probes"][0]["status"] == "ok", report
    assert any("ORDER BY reported_at DESC" in s and "LIMIT 1" in s for s in db.sql), db.sql


async def test_endpoint_before_the_first_run_is_stale_not_500():
    # The monitor creates the table on its first run; until then the endpoint
    # must say "no report", not fail on an undefined table.
    db = use(FakeDB(table=False))
    report = await admin.get_storage()
    assert report["stale"] is True and report["probes"] == [], report
    assert len(db.sql) == 1, db.sql


async def test_endpoint_with_an_empty_table_is_stale():
    use(FakeDB(rows=[]))
    report = await admin.get_storage()
    assert report["stale"] is True and report["reported_at"] is None


async def test_database_down_is_503_not_a_healthy_report():
    use(FakeDB(broken=True))
    try:
        await admin.get_storage()
    except HTTPException as exc:
        assert exc.status_code == 503, exc.status_code
        return
    raise AssertionError("expected HTTP 503, got a normal return")


async def main():
    tests = [
        test_thresholds_grade_each_probe,
        test_thresholds_are_module_constants,
        test_a_probe_can_carry_its_own_percent_thresholds,
        test_an_absolute_byte_threshold_grades_a_store_with_no_total,
        test_a_warn_only_byte_threshold_never_alerts,
        test_the_worse_of_percent_and_bytes_wins,
        test_a_store_without_a_capacity_is_ok_not_an_alert,
        test_a_failed_probe_is_error_and_raises_the_alert,
        test_alert_is_raised_at_eighty_percent_not_at_seventy,
        test_a_fresh_report_is_not_stale,
        test_a_report_older_than_two_hours_is_stale_and_alerts,
        test_no_report_at_all_is_stale,
        test_jsonb_returned_as_text_is_decoded,
        test_endpoint_reads_the_newest_row,
        test_endpoint_before_the_first_run_is_stale_not_500,
        test_endpoint_with_an_empty_table_is_stale,
        test_database_down_is_503_not_a_healthy_report,
    ]
    for test in tests:
        await test()
        print("ok  %s" % test.__name__)
    print("\n%d admin storage checks passed" % len(tests))


if __name__ == "__main__":
    asyncio.run(main())
