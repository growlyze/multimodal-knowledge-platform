"""
Drive intake (V6) and Drive auto-delete (V7) — offline checks.

the client's local fetcher uploads videos to a Google Drive folder; the server picks
them up from there. Everything here runs against FakeDriveLister and an
in-memory FakeDB, so nothing touches rclone, Drive or PostgreSQL.

    PYTHONPATH=. python tests/test_drive_intake.py

The failures guarded against are the expensive, silent ones:
  - a "scan" that queues something (ingestion is client-triggered only),
  - a queue call without explicit ids/confirm that bulk-queues a folder (D68),
  - a cleanup that deletes a Drive file whose video is NOT safely ingested,
  - a misconfigured server that throws a 500 instead of saying what is missing.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from pipelines import drive_intake as di
from routers import video

# Never read or write a real upload-folder choice (e.g. inside the container).
_TMP = tempfile.mkdtemp(prefix="drive-intake-test-")
os.environ["DRIVE_INTAKE_FOLDER_FILE"] = os.path.join(_TMP, "folder.json")

NOW = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)

_count = 0


def ok(name):
    global _count
    _count += 1
    print("ok %d %s" % (_count, name))


def run(coro):
    return asyncio.run(coro)


# ── Fakes ────────────────────────────────────────────────

def files():
    return [
        di.DriveFile("AAA111", "Product lesson 1.mp4", 1000, "video/mp4", "2026-10-24T10:00:00Z"),
        di.DriveFile("BBB222", "Product lesson 2.mkv", 2000, "video/x-matroska", ""),
        di.DriveFile("CCC333", "Lesson 3.MOV", 3000, "", ""),
        di.DriveFile("DDD444", "Lesson 4.mp4", 4000, "video/mp4", ""),
        di.DriveFile("EEE555", "Lesson 5.webm", 5000, "video/webm", ""),
        di.DriveFile("NOTES1", "notes.txt", 10, "text/plain", ""),
        di.DriveFile("PDF001", "slides.pdf", 10, "application/pdf", ""),
        # extension unknown but Drive says it is a video -> still a video
        di.DriveFile("MIME01", "clip", 10, "video/mp4", ""),
    ]


class FakeDB:
    """video_catalog + video_ingest_jobs + video_settings, matched on SQL text."""

    def __init__(self, catalog=None, jobs=None, settings=None, alive=True):
        self.catalog = dict(catalog or {})          # video_id -> row
        self.jobs = list(jobs or [])                # rows with id, video_id, status, ...
        self.settings = dict(settings or {})
        self.alive = alive
        self.sql = []

    async def pool(self):
        return object() if self.alive else None

    def _latest(self, vid):
        rows = [j for j in self.jobs if j["video_id"] == vid]
        return rows[-1] if rows else None

    async def fetch(self, sql, *a):
        self.sql.append(sql)
        if "LEFT JOIN LATERAL" in sql:
            out = []
            for vid in a[0]:
                if vid in self.catalog:
                    j = self._latest(vid)
                    out.append({"video_id": vid,
                                "status": j["status"] if j else None,
                                "completed_at": j.get("completed_at") if j else None})
            return out
        if "FROM video_settings" in sql:
            return [{"key": k, "value": v} for k, v in self.settings.items()]
        raise AssertionError("unexpected fetch: %s" % sql)

    async def fetchrow(self, sql, *a):
        self.sql.append(sql)
        if "MAX(queue_position)" in sql:
            pos = [j.get("queue_position") or 0 for j in self.jobs]
            return {"next_pos": (max(pos) if pos else 0) + 1}
        if "COUNT(*)" in sql:
            return {"cnt": sum(1 for j in self.jobs if j["status"] in ("queued", "processing"))}
        if "FROM video_ingest_jobs WHERE video_id" in sql:
            j = self._latest(a[0])
            return {"id": j["id"], "status": j["status"]} if j else None
        raise AssertionError("unexpected fetchrow: %s" % sql)

    async def execute(self, sql, *a):
        self.sql.append(sql)
        if sql.startswith("INSERT INTO video_catalog"):
            self.catalog.setdefault(a[0], {"video_id": a[0], "title": a[1],
                                           "source_url": a[2], "platform": a[3]})
        elif sql.startswith("INSERT INTO video_ingest_jobs"):
            self.jobs.append({"id": len(self.jobs) + 1, "video_id": a[0],
                              "status": "queued", "queue_position": a[1]})
        elif sql.startswith("UPDATE video_ingest_jobs SET status = 'queued'"):
            for j in self.jobs:
                if j["id"] == a[0]:
                    j["status"], j["queue_position"] = "queued", a[1]
        elif "video_settings" in sql and sql.lstrip().startswith("INSERT"):
            self.settings[a[0]] = a[1]
        elif "CREATE TABLE" in sql:
            pass
        else:
            raise AssertionError("unexpected execute: %s" % sql)
        return True


def vid(fid):
    return di.video_id_for(fid)


def db_with_states():
    """AAA new; BBB queued; CCC processing; DDD done 3 days ago; EEE failed."""
    cat = {vid(f): {"video_id": vid(f)} for f in ("BBB222", "CCC333", "DDD444", "EEE555")}
    jobs = [
        {"id": 1, "video_id": vid("BBB222"), "status": "queued", "queue_position": 1},
        {"id": 2, "video_id": vid("CCC333"), "status": "processing", "queue_position": 2},
        {"id": 3, "video_id": vid("DDD444"), "status": "done", "queue_position": 3,
         "completed_at": NOW - timedelta(days=3)},
        {"id": 4, "video_id": vid("EEE555"), "status": "failed", "queue_position": 4},
    ]
    return FakeDB(cat, jobs)


def use(db, lister):
    video.get_eval_db = lambda: db
    video._drive_lister = lambda: lister
    video._drive_now = lambda: NOW
    return db


def unconfigured():
    raise di.DriveIntakeNotConfigured("drive intake not configured: DRIVE_INTAKE_FOLDER_ID unset")


def status_of(call):
    try:
        run(call)
    except HTTPException as exc:
        return exc.status_code, str(exc.detail)
    return None, ""


# ── V6: identification + classification ──────────────────

def test_ids_and_video_detection():
    assert vid("AbC-_9") == "gd_AbC-_9"
    assert di.file_id_for("gd_AbC-_9") == "AbC-_9"
    # A YouTube id must never be mistaken for a Drive file (cleanup keys on this).
    assert di.file_id_for("z6AxPV2NQ9U") is None
    videos = [f.file_id for f in files() if di.is_video_file(f)]
    assert videos == ["AAA111", "BBB222", "CCC333", "DDD444", "EEE555", "MIME01"], videos
    ok("drive file id <-> video_id is reversible; non-video files are not videos")


def test_classify():
    db = db_with_states()
    index = run(di.job_index(db, [vid(f.file_id) for f in files()]))
    result = di.classify(files(), index)
    states = {v["file_id"]: v["state"] for v in result["videos"]}
    assert states == {"AAA111": "new", "BBB222": "queued", "CCC333": "processing",
                      "DDD444": "ingested", "EEE555": "failed", "MIME01": "new"}, states
    assert sorted(i["name"] for i in result["ignored"]) == ["notes.txt", "slides.pdf"]
    assert result["counts"]["new"] == 2 and result["counts"]["ingested"] == 1
    ok("classification: new / queued / processing / ingested / failed; non-video ignored")


def test_catalogued_without_job_is_new():
    db = FakeDB({vid("AAA111"): {"video_id": vid("AAA111")}}, [])
    index = run(di.job_index(db, [vid("AAA111")]))
    result = di.classify(files()[:1], index)
    assert result["videos"][0]["state"] == "new", result
    ok("a catalogued Drive video that was never queued still counts as new")


# ── V6: endpoints ─────────────────────────────────────────

def test_scan_is_dry_run():
    db = use(db_with_states(), di.FakeDriveLister(files()))
    before = (dict(db.catalog), [dict(j) for j in db.jobs])
    out = run(video.drive_intake_scan())
    assert out["dry_run"] is True
    assert (db.catalog, db.jobs) == before, "scan changed the database"
    assert not any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for s in db.sql), db.sql
    assert len(out["videos"]) == 6 and len(out["ignored"]) == 2
    ok("POST /video/drive-intake/scan queues nothing and writes nothing")


def test_scan_unconfigured_503():
    use(db_with_states(), None)
    video._drive_lister = unconfigured
    code, detail = status_of(video.drive_intake_scan())
    assert code == 503 and "not configured" in detail, (code, detail)
    ok("scan with rclone/folder unconfigured -> 503 naming what is missing")


def test_scan_db_down_503():
    use(FakeDB(alive=False), di.FakeDriveLister(files()))
    code, _ = status_of(video.drive_intake_scan())
    # A dead eval DB would otherwise classify every file as "new" and invite
    # re-queuing videos that are already ingested.
    assert code == 503, code
    ok("scan with the eval DB down -> 503, not 'everything is new'")


def test_queue_requires_ids_and_confirm():
    db = use(db_with_states(), di.FakeDriveLister(files()))
    code, _ = status_of(video.drive_intake_queue(video.DriveQueueRequest(file_ids=["AAA111"])))
    assert code == 400, code
    code, _ = status_of(video.drive_intake_queue(
        video.DriveQueueRequest(file_ids=["AAA111"], confirm=False)))
    assert code == 400, code
    code, _ = status_of(video.drive_intake_queue(video.DriveQueueRequest(file_ids=[], confirm=True)))
    assert code == 400, code
    assert len(db.jobs) == 4 and vid("AAA111") not in db.catalog
    ok("queue refuses without confirm=true or without explicit file ids")


def test_queue_unconfigured_503():
    use(db_with_states(), None)
    video._drive_lister = unconfigured
    code, detail = status_of(video.drive_intake_queue(
        video.DriveQueueRequest(file_ids=["AAA111"], confirm=True)))
    assert code == 503 and "not configured" in detail, (code, detail)
    ok("queue with drive intake unconfigured -> 503, not an exception")


def test_queue_new_and_skips():
    db = use(db_with_states(), di.FakeDriveLister(files()))
    out = run(video.drive_intake_queue(video.DriveQueueRequest(
        file_ids=["AAA111", "BBB222", "DDD444", "EEE555", "NOTES1", "NOPE99"], confirm=True)))
    by = {d["file_id"]: d for d in out["details"]}
    assert by["AAA111"]["status"] == "queued"
    assert by["BBB222"]["status"] == "skipped" and by["BBB222"]["reason"] == "queued"
    assert by["DDD444"]["status"] == "skipped" and by["DDD444"]["reason"] == "ingested"
    assert by["EEE555"]["status"] == "queued", "a failed Drive video can be retried"
    assert by["NOTES1"]["status"] == "not_video"
    assert by["NOPE99"]["status"] == "not_found"
    assert out["queued"] == 2
    row = db.catalog[vid("AAA111")]
    # Same catalog shape the queue worker reads: platform routes the download
    # phase to Drive, source_url is what the citation links to.
    assert row["platform"] == di.PLATFORM == "gdrive", row
    assert row["title"] == "Product lesson 1", row
    assert "AAA111" in row["source_url"], row
    # Retry reuses the failed row instead of inserting a duplicate (P97).
    assert sum(1 for j in db.jobs if j["video_id"] == vid("EEE555")) == 1
    assert [j["status"] for j in db.jobs if j["video_id"] == vid("EEE555")] == ["queued"]
    ok("queue: new + failed queued, queued/ingested/non-video/unknown skipped, no duplicate rows")


def test_queue_respects_caps():
    db = use(FakeDB(), di.FakeDriveLister(files()))
    saved = video.config.MAX_IN_FLIGHT_VIDEOS
    video.config.MAX_IN_FLIGHT_VIDEOS = 1
    try:
        out = run(video.drive_intake_queue(video.DriveQueueRequest(
            file_ids=["AAA111", "BBB222"], confirm=True)))
    finally:
        video.config.MAX_IN_FLIGHT_VIDEOS = saved
    assert out["queued"] == 1, out
    assert out["details"][1]["reason"] == "in_flight_limit_reached", out
    saved = video.config.MAX_QUEUE_BATCH_SIZE
    video.config.MAX_QUEUE_BATCH_SIZE = 1
    try:
        code, _ = status_of(video.drive_intake_queue(video.DriveQueueRequest(
            file_ids=["AAA111", "BBB222"], confirm=True)))
    finally:
        video.config.MAX_QUEUE_BATCH_SIZE = saved
    assert code == 400, code
    ok("queue obeys the D68 in-flight cap and batch-size cap")


def test_download_for_ingest():
    lister = di.FakeDriveLister(files())
    path = di.download_for_ingest(vid("BBB222"), "/work", lister=lister)
    assert path.replace("\\", "/") == "/work/gd_BBB222.mkv", path
    assert lister.downloads == [("BBB222", path)], lister.downloads
    try:
        di.download_for_ingest(vid("GONE00"), "/work", lister=lister)
    except RuntimeError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("missing Drive file must fail the download phase")
    ok("download phase copies the Drive file by id to work_dir/<video_id>.<ext>")


def test_acquisition_routes_gdrive():
    from pipelines import video_acquisition as acq
    calls = []
    saved = di.download_for_ingest
    di.download_for_ingest = lambda video_id, output_dir, lister=None: (
        calls.append((video_id, output_dir)) or "/w/%s.mp4" % video_id)
    try:
        p = acq.download_video("gd_AAA111", "/w", platform="gdrive",
                               source_url="https://drive.google.com/file/d/AAA111/view")
    finally:
        di.download_for_ingest = saved
    assert p == "/w/gd_AAA111.mp4" and calls == [("gd_AAA111", "/w")], (p, calls)
    ok("video_acquisition.download_video sends platform=gdrive to Drive, never to yt-dlp")


# ── rclone command construction ──────────────────────────

class Runner:
    def __init__(self, stdout="[]", returncode=0):
        self.calls, self.stdout, self.returncode = [], stdout, returncode

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)

        class R:
            pass
        r = R()
        r.returncode, r.stdout, r.stderr = self.returncode, self.stdout, "boom"
        return r


def test_rclone_lister_commands():
    listing = json.dumps([
        {"ID": "AAA111", "Name": "a.mp4", "Size": 5, "MimeType": "video/mp4",
         "ModTime": "2026-10-24T10:00:00Z", "IsDir": False},
        {"ID": "DIR", "Name": "sub", "Size": -1, "MimeType": "inode/directory", "IsDir": True},
    ])
    runner = Runner(listing)
    lister = di.RcloneDriveLister("FOLDER9", remote="gdrive:", binary="rclone",
                                  config_path="/rc/rclone.conf", runner=runner)
    got = lister.list_files()
    assert [f.file_id for f in got] == ["AAA111"], got
    cmd = runner.calls[0]
    assert cmd[:2] == ["rclone", "lsjson"] and "gdrive:" in cmd, cmd
    assert cmd[cmd.index("--drive-root-folder-id") + 1] == "FOLDER9", cmd
    assert cmd[cmd.index("--config") + 1] == "/rc/rclone.conf", cmd
    lister.download("AAA111", "/w/gd_AAA111.mp4")
    cmd = runner.calls[-1]
    assert cmd[1:5] == ["backend", "copyid", "gdrive:", "AAA111"], cmd
    assert "/w/gd_AAA111.mp4" in cmd
    # deletefile is name-based; it must re-list and refuse an ambiguous name.
    lister.delete("AAA111", "a.mp4")
    cmd = runner.calls[-1]
    assert cmd[1:3] == ["deletefile", "gdrive:a.mp4"], cmd
    assert "--drive-use-trash=false" in cmd, "auto-delete is permanent, not the Drive trash"
    assert cmd[cmd.index("--drive-root-folder-id") + 1] == "FOLDER9", cmd
    dup = json.dumps([{"ID": "AAA111", "Name": "a.mp4", "IsDir": False},
                      {"ID": "ZZZ999", "Name": "a.mp4", "IsDir": False}])
    lister2 = di.RcloneDriveLister("FOLDER9", runner=Runner(dup))
    try:
        lister2.delete("AAA111", "a.mp4")
    except di.DriveDeleteRefused:
        pass
    else:
        raise AssertionError("deleted by an ambiguous name")
    try:
        di.RcloneDriveLister("F", runner=Runner("", returncode=1)).list_files()
    except RuntimeError as exc:
        assert "rclone" in str(exc)
    else:
        raise AssertionError("a failed rclone call must raise, not look like an empty folder")
    ok("rclone lister: lsjson/copyid/deletefile carry folder id + config; ambiguous delete refused")


def test_lister_from_env():
    try:
        di.lister_from_env({}, which=lambda b: "/usr/bin/rclone")
    except di.DriveIntakeNotConfigured as exc:
        assert "DRIVE_INTAKE_FOLDER_ID" in str(exc)
    else:
        raise AssertionError("no folder id must be 'not configured'")
    try:
        di.lister_from_env({"DRIVE_INTAKE_FOLDER_ID": "F"}, which=lambda b: None)
    except di.DriveIntakeNotConfigured as exc:
        assert "rclone" in str(exc)
    else:
        raise AssertionError("missing rclone binary must be 'not configured'")
    lister = di.lister_from_env({"DRIVE_INTAKE_FOLDER_ID": "1qrq",
                                 "DRIVE_INTAKE_RCLONE_REMOTE": "gd2:"},
                                which=lambda b: "/usr/bin/rclone", exists=lambda p: True)
    assert lister.folder_id == "1qrq" and lister.remote == "gd2:"
    ok("lister_from_env: folder id and rclone binary required, folder/remote configurable")


def test_rclone_config_is_used_from_a_private_writable_copy():
    """V6 activation (2026-10-25, measured on the VPS with an expired-token copy of the
    host conf): with the conf bind-mounted read-only, rclone refreshed the token and
    listed the folder, but logged "ERROR : Failed to save config after 10 tries ...
    rename /run/rclone/rclone.conf ...: device or resource busy" on every call -- a
    single-file bind mount can never be replaced by rename, ro or not. rclone runs on
    a private copy instead: the refreshed token lands there, the host file (shared
    with the backups) is never written."""
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "mounted.conf")
        with open(src, "w") as f:
            f.write("[gdrive]\ntype = drive\ntoken = {\"expiry\":\"2020\"}\n")
        work = os.path.join(tmp, "work")
        env = {"DRIVE_INTAKE_FOLDER_ID": "F", "DRIVE_INTAKE_RCLONE_CONFIG": src,
               "DRIVE_INTAKE_RCLONE_WORKDIR": work}
        lister = di.lister_from_env(env, which=lambda b: "/usr/bin/rclone")
        copy = lister.config_path
        assert copy != src and copy.startswith(work), copy
        assert open(copy).read() == open(src).read()
        if os.name == "posix":
            assert oct(os.stat(copy).st_mode & 0o777) == "0o600", oct(os.stat(copy).st_mode)
        # rclone saved a refreshed token into the copy: the next lister keeps it...
        with open(copy, "w") as f:
            f.write("refreshed")
        later = os.path.getmtime(src) + 10
        os.utime(copy, (later, later))
        assert open(di.lister_from_env(env, which=lambda b: "/usr/bin/rclone").config_path).read() == "refreshed"
        # ...until the mounted file changes (re-authorised on the host): copied again.
        with open(src, "w") as f:
            f.write("[gdrive]\ntype = drive\ntoken = {\"expiry\":\"2030\"}\n")
        os.utime(src, (later + 10, later + 10))
        assert "2030" in open(di.lister_from_env(env, which=lambda b: "/usr/bin/rclone").config_path).read()
        assert "2030" in open(src).read(), "the mounted file is only ever read"
        # An unreadable / missing mount is still 'not configured', never a crash.
        try:
            di.lister_from_env({**env, "DRIVE_INTAKE_RCLONE_CONFIG": os.path.join(tmp, "nope.conf")},
                               which=lambda b: "/usr/bin/rclone")
        except di.DriveIntakeNotConfigured as exc:
            assert "not mounted" in str(exc)
        else:
            raise AssertionError("a missing config must be 'not configured'")
    ok("rclone config: used from a private 0600 copy, re-copied only when the mount changes")


def test_panel_signin_override_runs_the_copy_on_the_connected_account():
    """The admin panel's Google Drive sign-in (/data/backup-drives/primary.json)
    is swapped into the private copy the same way ops/backup_drives.py applies it
    for the backup: token and folder replaced, and the issuing OAuth client kept -- so scans,
    downloads and auto-approve/auto-delete run on the connected account. No
    override -> the server's own config, byte for byte; an injecting override is
    ignored, never applied."""
    import json
    import os
    import tempfile
    base = ('[gdrive]\ntype = drive\nclient_id = CID\nclient_secret = CS\nscope = drive\n'
            'root_folder_id = OLDFOLDER\ntoken = {"refresh_token":"old"}\n')
    rec = {"token": {"access_token": "a", "refresh_token": "new"}, "root_folder_id": "NEWFOLDER",
           "account": "[EMAIL]"}
    with tempfile.TemporaryDirectory() as tmp:
        src, override = (os.path.join(tmp, n) for n in ("mounted.conf", "primary.json"))
        work = os.path.join(tmp, "work")
        with open(src, "w") as f:
            f.write(base)
        env = {"DRIVE_INTAKE_FOLDER_ID": "F", "DRIVE_INTAKE_RCLONE_CONFIG": src,
               "DRIVE_INTAKE_RCLONE_WORKDIR": work, "DRIVE_INTAKE_DRIVE_OVERRIDE": override}

        def copy_text():
            return open(di.lister_from_env(env, which=lambda b: "/usr/bin/rclone").config_path).read()

        assert copy_text() == base, "no panel sign-in -> the server's config, unchanged"
        with open(override, "w") as f:
            json.dump(rec, f)
        text = copy_text()
        assert '"refresh_token":"new"' in text and "NEWFOLDER" in text, text
        assert "client_id = CID" in text and "scope = drive" in text, "client and scope kept"
        assert "OLDFOLDER" not in text and '"old"' not in text, "the server's token/folder replaced"
        # a web-client sign-in must be refreshed as the client that issued it
        with open(override, "w") as f:
            json.dump(dict(rec, client_id="WEBID", client_secret="WEBSEC"), f)
        later = os.path.getmtime(override) + 10
        os.utime(override, (later, later))
        text = copy_text()
        assert "client_id = WEBID" in text and "client_secret = WEBSEC" in text, text
        assert open(src).read() == base, "the mounted file is only ever read"
        # rclone refreshed the token in the copy: kept until the override changes
        copy = di.lister_from_env(env, which=lambda b: "/usr/bin/rclone").config_path
        with open(copy, "w") as f:
            f.write("refreshed")
        assert copy_text() == "refreshed", "rclone's own refresh in the copy is not rebuilt away"
        # a different account signs in -> rebuilt with that account
        rec2 = dict(rec, token={"refresh_token": "newer"}, root_folder_id="NEWERFOLDER")
        with open(override, "w") as f:
            json.dump(rec2, f)
        later = os.path.getmtime(override) + 10
        os.utime(override, (later, later))
        text = copy_text()
        assert '"refresh_token":"newer"' in text and "NEWERFOLDER" in text, text
        # the sign-in goes away (override removed) -> back to the server's remote
        os.remove(override)
        assert copy_text() == base
        # an injecting override is ignored, never applied
        with open(override, "w") as f:
            json.dump(dict(rec, root_folder_id="F\nclient_secret = stolen"), f)
        assert copy_text() == base, "a multi-line value must never reach the config"
        with open(override, "w") as f:
            json.dump(dict(rec, client_secret="S\ninjected = 1"), f)
        assert copy_text() == base, "a multi-line client value must never reach the config"
    ok("panel sign-in: the copy runs on the connected account (token+folder, client kept); "
       "refresh kept; switch and sign-out rebuild; injection refused")


def test_apply_override_fills_missing_keys():
    text = di.apply_override("[other]\ntype = local\n[gdrive]\ntype = drive\n", "gdrive",
                             {"token": {"refresh_token": "t"}, "root_folder_id": "F"})
    assert "token = " in text and "root_folder_id = F" in text and "[other]\ntype = local" in text
    assert di.load_override("/nonexistent/primary.json") == {}, "no override -> {}"
    text = di.apply_override("[gdrive]\ntype = drive\n", "gdrive",
                             {"token": {"refresh_token": "t"}, "root_folder_id": "F",
                              "client_id": "W", "client_secret": "S"})
    assert "client_id = W" in text and "client_secret = S" in text, "the issuing client is applied too"
    ok("apply_override: a section without token/folder gets both added; a missing file is none")


# ── V7: auto-delete ──────────────────────────────────────

def cleanup_db(done_age_hours, status="done"):
    jobs = [{"id": 1, "video_id": vid("DDD444"), "status": status, "queue_position": 1,
             "completed_at": (NOW - timedelta(hours=done_age_hours))
             if done_age_hours is not None else None}]
    return FakeDB({vid("DDD444"): {"video_id": vid("DDD444")}}, jobs)


def test_cleanup_window_boundary():
    for age, expect in ((47.99, []), (48, ["DDD444"]), (72, ["DDD444"])):
        rep = run(di.run_cleanup(cleanup_db(age), di.FakeDriveLister(files()),
                                 enabled=False, window_hours=48, dry_run=True, now=NOW))
        got = [d["file_id"] for d in rep["would_delete"]]
        assert got == expect, (age, got)
    ok("cleanup window: 47.99 h kept, 48 h and older selected")


def test_cleanup_only_done():
    for status in ("failed", "processing", "queued", "new", "mystery"):
        rep = run(di.run_cleanup(cleanup_db(100, status), di.FakeDriveLister(files()),
                                 enabled=True, window_hours=48, dry_run=True, now=NOW))
        assert rep["would_delete"] == [], (status, rep["would_delete"])
    # done but completed_at missing: cannot prove the window passed -> keep
    rep = run(di.run_cleanup(cleanup_db(None), di.FakeDriveLister(files()),
                             enabled=True, window_hours=48, dry_run=True, now=NOW))
    assert rep["would_delete"] == []
    # unmatched files (AAA, notes.txt, ...) are never selected
    rep = run(di.run_cleanup(cleanup_db(100), di.FakeDriveLister(files()),
                             enabled=True, window_hours=48, dry_run=True, now=NOW))
    assert [d["file_id"] for d in rep["would_delete"]] == ["DDD444"]
    kept = {k["file_id"]: k["reason"] for k in rep["kept"]}
    assert "AAA111" in kept and "NOTES1" in kept, kept
    ok("cleanup selects only done jobs past the window; failed/in-progress/unknown/unmatched kept")


def test_cleanup_dry_run_deletes_nothing():
    lister = di.FakeDriveLister(files())
    rep = run(di.run_cleanup(cleanup_db(100), lister, enabled=True, window_hours=48,
                             dry_run=True, now=NOW))
    assert lister.deleted == [] and rep["deleted"] == [] and rep["dry_run"] is True
    ok("cleanup dry run deletes nothing")


def test_cleanup_toggle_off_refuses():
    lister = di.FakeDriveLister(files())
    try:
        run(di.run_cleanup(cleanup_db(100), lister, enabled=False, window_hours=48,
                           dry_run=False, now=NOW))
    except di.DriveCleanupDisabled:
        pass
    else:
        raise AssertionError("real delete with the toggle OFF")
    assert lister.deleted == []
    rep = run(di.run_cleanup(cleanup_db(100), lister, enabled=True, window_hours=48,
                             dry_run=False, now=NOW))
    assert lister.deleted == ["DDD444"] and [d["file_id"] for d in rep["deleted"]] == ["DDD444"]
    ok("real cleanup refuses with toggle OFF, deletes only the selected file with it ON")


def test_cleanup_window_clamped():
    assert di.clamp_hours(1) == di.AUTODELETE_MIN_HOURS == 24
    assert di.clamp_hours(1000) == di.AUTODELETE_MAX_HOURS == 48
    assert di.clamp_hours("x") == di.AUTODELETE_DEFAULT_HOURS == 48
    s = di.parse_settings({})
    assert s == {"autodelete_enabled": False, "autodelete_hours": 48, "autoapprove_enabled": False}, s
    s = di.parse_settings({"drive_autodelete_enabled": "true", "drive_autodelete_hours": "30"})
    assert s == {"autodelete_enabled": True, "autodelete_hours": 30, "autoapprove_enabled": False}, s
    ok("settings default OFF / 48 h; window clamped to the agreed 24-48 h")


def test_cleanup_endpoint_respects_toggle():
    db = cleanup_db(100)
    lister = di.FakeDriveLister(files())
    use(db, lister)
    out = run(video.drive_intake_cleanup(video.DriveCleanupRequest()))
    assert out["dry_run"] is True and [d["file_id"] for d in out["would_delete"]] == ["DDD444"]
    code, _ = status_of(video.drive_intake_cleanup(video.DriveCleanupRequest(dry_run=False)))
    assert code == 409 and lister.deleted == [], code
    run(video.drive_intake_save_settings(video.DriveIntakeSettings(
        autodelete_enabled=True, autodelete_hours=24)))
    got = run(video.drive_intake_get_settings())
    assert {k: got[k] for k in ("autodelete_enabled", "autodelete_hours")} == \
        {"autodelete_enabled": True, "autodelete_hours": 24}, got
    out = run(video.drive_intake_cleanup(video.DriveCleanupRequest(dry_run=False)))
    assert lister.deleted == ["DDD444"], lister.deleted
    ok("POST /video/drive-intake/cleanup: dry run by default, 409 while toggle OFF")


def test_cleanup_endpoint_unconfigured():
    use(cleanup_db(100), None)
    video._drive_lister = unconfigured
    code, _ = status_of(video.drive_intake_cleanup(video.DriveCleanupRequest()))
    assert code == 503, code
    ok("cleanup with drive intake unconfigured -> 503")


def test_settings_do_not_touch_auto_check_keys():
    db = use(FakeDB(settings={"auto_check_enabled": "true"}), di.FakeDriveLister([]))
    run(video.drive_intake_save_settings(video.DriveIntakeSettings(autodelete_enabled=True, autodelete_hours=48)))
    assert db.settings["auto_check_enabled"] == "true"
    assert set(db.settings) == {"auto_check_enabled", "drive_autodelete_enabled",
                                "drive_autodelete_hours"}, db.settings
    ok("drive settings use their own keys; the existing /video/settings keys are untouched")


# ── upload folder chosen in the admin panel ──────────────

def _clear_folder():
    try:
        os.remove(os.environ["DRIVE_INTAKE_FOLDER_FILE"])
    except OSError:
        pass


def test_selected_folder_precedence():
    _clear_folder()
    env = {"DRIVE_INTAKE_FOLDER_ID": "ENVFOLDER0001"}
    assert di.selected_folder(env) == {"folder_id": "ENVFOLDER0001", "name": "", "source": "env",
                                       "env_folder_id": "ENVFOLDER0001"}
    assert di.selected_folder({})["source"] == ""
    di.save_selected_folder("PANELFOLDER01", "Uploads", env)
    sel = di.selected_folder(env)
    assert sel["folder_id"] == "PANELFOLDER01" and sel["name"] == "Uploads" and sel["source"] == "panel"
    if os.name == "posix":
        assert os.stat(di.folder_file(env)).st_mode & 0o777 == 0o600
    lister = di.lister_from_env(env, which=lambda b: "/usr/bin/rclone", exists=lambda p: True)
    assert lister.folder_id == "PANELFOLDER01", "scan/queue/cleanup/download all follow the panel choice"
    assert di.lister_from_env(env, which=lambda b: "/usr/bin/rclone", folder_id="OTHERFOLDER1").folder_id \
        == "OTHERFOLDER1"
    for bad in ("", "x", "../../etc/passwd", "a b c d e f g h", None, 5):
        try:
            di.save_selected_folder(bad, "", env)
        except ValueError:
            pass
        else:
            raise AssertionError("bad folder id accepted: %r" % (bad,))
    with open(di.folder_file(env), "w") as fh:
        fh.write("{broken")
    assert di.selected_folder(env)["folder_id"] == "ENVFOLDER0001", "a damaged file falls back to the env"
    _clear_folder()
    ok("upload folder: panel choice wins over DRIVE_INTAKE_FOLDER_ID, validated, 0600, damaged file ignored")


def test_folder_endpoint():
    _clear_folder()
    listed = []

    def folder_lister(fid):
        listed.append(fid)
        return di.FakeDriveLister(files(), folder_id=fid)
    video._drive_folder_lister = folder_lister
    use(FakeDB(), di.FakeDriveLister([]))
    req = video.DriveFolderRequest
    assert status_of(video.drive_intake_set_folder(req(folder_id="NEWFOLDER0001")))[0] == 400, "confirm needed"
    assert status_of(video.drive_intake_set_folder(req(folder_id="../x", confirm=True)))[0] == 400
    use(db_with_states(), di.FakeDriveLister([]))
    code, detail = status_of(video.drive_intake_set_folder(req(folder_id="NEWFOLDER0001", confirm=True)))
    assert code == 409 and "queued or processing" in detail, (code, detail)
    assert not os.path.exists(os.environ["DRIVE_INTAKE_FOLDER_FILE"])

    class Broken(di.FakeDriveLister):
        def list_files(self):
            raise RuntimeError("403 insufficient permissions")
    video._drive_folder_lister = lambda fid: Broken([], folder_id=fid)
    use(FakeDB(), di.FakeDriveLister([]))
    assert status_of(video.drive_intake_set_folder(req(folder_id="NEWFOLDER0001", confirm=True)))[0] == 502
    assert not os.path.exists(os.environ["DRIVE_INTAKE_FOLDER_FILE"]), "an unlistable folder is never saved"

    video._drive_folder_lister = folder_lister
    out = run(video.drive_intake_set_folder(req(folder_id="NEWFOLDER0001", name="Uploads", confirm=True)))
    assert out["intake_folder_id"] == "NEWFOLDER0001" and out["intake_folder_source"] == "panel"
    assert listed[-1] == "NEWFOLDER0001"
    got = run(video.drive_intake_get_settings())
    assert got["intake_folder_id"] == "NEWFOLDER0001" and got["intake_folder_name"] == "Uploads"
    _clear_folder()
    ok("POST /drive-intake/folder: confirm required, id validated, refused while Drive videos are in "
       "flight or the folder cannot be listed; GET settings reports it")


# ── autopilot: auto-approve + auto-delete toggles ─────────

def test_settings_save_only_what_is_sent():
    db = use(FakeDB(settings={"drive_autoapprove_enabled": "true"}), di.FakeDriveLister([]))
    out = run(video.drive_intake_save_settings(video.DriveIntakeSettings(autodelete_enabled=True,
                                                                         autodelete_hours=30)))
    assert db.settings["drive_autoapprove_enabled"] == "true", "the auto-delete form left auto-approve alone"
    assert out["autodelete_enabled"] and out["autodelete_hours"] == 30 and out["autoapprove_enabled"]
    run(video.drive_intake_save_settings(video.DriveIntakeSettings(autoapprove_enabled=False)))
    assert db.settings["drive_autoapprove_enabled"] == "false" and db.settings["drive_autodelete_enabled"] == "true"
    assert di.settings_pairs() == {}
    ok("drive settings: each toggle saved on its own; one form never resets the other")


def test_autopilot_off_does_nothing():
    lister = di.FakeDriveLister(files())
    db = use(FakeDB(), lister)
    started = []
    video._process_queue = lambda d: started.append(d) or asyncio.sleep(0)
    out = run(video.drive_autopilot_round(now=NOW))
    assert out == {"skipped": "both toggles off"} and db.jobs == [] and lister.deleted == [] and not started
    ok("autopilot with both toggles OFF: no scan, no queue, no delete")


def test_autopilot_autoapprove_queues_new_only():
    lister = di.FakeDriveLister(files())
    db = db_with_states()
    db.settings["drive_autoapprove_enabled"] = "true"
    use(db, lister)
    started = []

    async def fake_worker(d):
        started.append(d)
    video._process_queue = fake_worker
    video.ingest_video = video.ingest_video or (lambda *a, **k: None)
    saved = video._worker_active
    video._worker_active = False

    async def go():
        res = await video.drive_autopilot_round(now=NOW)
        await asyncio.sleep(0)               # let the started worker task run
        return res
    out = run(go())
    video._worker_active = saved
    queued = sorted(j["video_id"] for j in db.jobs if j["status"] == "queued")
    # new: AAA111 + MIME01 (videos); EEE555 is failed -> never auto-retried; BBB222 was already queued
    assert out["queued"] == 2, out
    assert vid("AAA111") in queued and vid("MIME01") in queued and vid("EEE555") not in queued
    assert started, "the worker is started after queuing"
    assert lister.deleted == [], "auto-delete is a separate toggle"
    ok("auto-approve ON: queues only NEW videos (never failed ones), starts the worker; deletes nothing")


def test_autopilot_autodelete_uses_window():
    lister = di.FakeDriveLister(files())
    db = cleanup_db(100)
    db.settings.update({"drive_autodelete_enabled": "true", "drive_autodelete_hours": "48"})
    use(db, lister)
    out = run(video.drive_autopilot_round(now=NOW))
    assert out["deleted"] == 1 and lister.deleted == ["DDD444"] and out["queued"] == 0
    ok("auto-delete ON: removes only ingested videos past the window; queues nothing")


def test_autopilot_status_in_settings():
    use(FakeDB(), di.FakeDriveLister([]))
    video._autopilot.update(last_run_at="2026-10-26T12:00:00+00:00", last_error="")
    got = run(video.drive_intake_get_settings())
    assert got["autopilot"]["last_run_at"] == "2026-10-26T12:00:00+00:00" and "autoapprove_enabled" in got
    ok("GET settings reports both toggles and the autopilot's last run")


def test_signed_out_stops_intake():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        env = {"DRIVE_INTAKE_FOLDER_ID": "ENVFOLDER0001", "DRIVE_INTAKE_DRIVE_OVERRIDE": os.path.join(tmp, "primary.json")}
        lister = di.lister_from_env(env, which=lambda b: "/usr/bin/rclone", exists=lambda p: True)
        assert lister.folder_id, "no sign-out file: the server's remote is used as before"
        open(os.path.join(tmp, "signed-out.json"), "w").write("{}")
        try:
            di.lister_from_env(env, which=lambda b: "/usr/bin/rclone", exists=lambda p: True)
        except di.DriveIntakeNotConfigured as exc:
            assert "not connected" in str(exc)
        else:
            raise AssertionError("a signed-out Drive was still used for uploads")
    ok("signed out in the panel -> uploads stop (scan/queue/download refuse), no fallback to the old account")


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\n%d checks passed" % _count)


if __name__ == "__main__":
    main()
