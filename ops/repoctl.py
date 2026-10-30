#!/usr/bin/env python3
"""repoctl.py — run ONE operator request from the dashboard's
"Backup & restore" section, on the VPS host.

The panel runs in a container and must never hold the backup repository's
secrets (RESTIC_PASSWORD, the rclone OAuth token), so it cannot talk to restic
itself. It drops one request file in /data/backup-drives/ctl/ (the directory it
already shares with the backup's Drive sign-in) and reads the result from the
same directory:

    request.json    panel -> host   {"id","cmd","snapshot","paths",
                                     "requested_at","by"}
    status.json     host -> panel   {"id","cmd","state","step","started_at",
                                     "finished_at","result","error",
                                     "progress" (0-100, or null when unknown)}

ops/repoctl.path starts this script the moment request.json appears;
the request is claimed (renamed) on arrival, so a request is run once.

Commands — all read-only on the repository. Nothing here ever forgets,
prunes, re-tags or edits a backup; a restore can never destroy a recovery
point. A request names the folder it is about ("repo": daily | weekly |
monthly — the three subfolders of the backup folder, daily when absent).

    backup-now      run ops/backup.sh — the nightly job's own script.
                    It takes the nightly job's own lock itself, so a manual
                    run and the 03:03 scheduled run can never overlap or
                    double-push ("another backup run holds ...").
    snapshots       restic snapshots --json -> the recovery points.
    inspect <id>    is this recovery point available and readable before a
                    restore? snapshot record + repository check + the
                    MANIFEST.json it was taken with + its file inventory.
    find-deleted    which knowledge-base DOCUMENTS are in backup <id> but gone
                    from the live knowledge base now (deleted since). The
                    backup's Qdrant snapshots are loaded into TEMPORARY
                    collections (ops/qdrant_docs.py), compared, and dropped.
    restore-docs    bring selected deleted documents back from <id>: their
                    sections into the live collection (vectors and payload as
                    they were) and, for spreadsheets, their PostgreSQL tables
                    and data_catalog rows from the backup's database dump.
                    Additive only: a document that has ANY section live, a
                    table that exists, a catalog row that exists — skipped,
                    never overwritten. Verified afterwards.
    restore-files   restore SELECTED files from <id> into the live knowledge
                    base file storage. Additive only:
                      * only the two file roots below are restorable this way;
                      * a file that exists on the server is NEVER overwritten
                        (existing data is newer data) — it is skipped and
                        reported;
                      * every restored file is verified against the size the
                        snapshot lists before it is moved into place.
    restore-full    DISASTER recovery from <id>: replaces the live Postgres
                    database, the six Qdrant collections and
                    /data/haystack-uploads with the backup's. Needs the typed
                    confirmation "RESTORE" and a fresh safety backup (a
                    milestone in the daily folder) taken by the same request
                    first — if that backup fails, nothing is touched. Config
                    files (.env, compose.yml, ...) are only hash-compared,
                    never written. See cmd_restore_full for the step order.
    Document files restored here are re-added to the knowledge base
    through the ordinary Upload & add pipeline when their records are missing —
    that pipeline recognises identical content and never embeds it twice.

Secrets: RESTIC_PASSWORD and the rclone token are used and never logged, never
written to the status file (which holds file names, sizes and counts only).
The panel's connected Google Drive account is honoured exactly as the nightly
backup honours it (ops/backup_drives.py applied to a temporary rclone config,
removed afterwards).

Usage (on the VPS, as root):
    python3 ops/repoctl.py                  # what repoctl.service runs
    python3 ops/repoctl.py --self-test      # offline checks: no restic, no docker
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

# ops/catalog.py (same directory) — the backup catalog shared with the panel.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog as backup_catalog                       # noqa: E402

try:
    import fcntl                                  # the host is Linux; the self
except ImportError:                               # check also runs elsewhere
    fcntl = None

INSTALL_DIR = os.environ.get("INSTALL_DIR", "/opt/gmac")
STACK_NAME = os.environ.get("STACK_NAME", "gmac")
ENV_FILE = os.environ.get("ENV_FILE", os.path.join(INSTALL_DIR, ".env"))
BACKUP_SCRIPT = os.environ.get("BACKUP_SCRIPT", os.path.join(INSTALL_DIR, "ops/backup.sh"))
DRIVES_HELPER = os.environ.get("DRIVES_HELPER", os.path.join(INSTALL_DIR, "ops/backup_drives.py"))
CTL_DIR = os.environ.get("REPOCTL_DIR", "/data/backup-drives/ctl")
LOCK_FILE = os.environ.get("LOCK_FILE", "/run/lock/gmac-backup.lock")
RUN_DIR = os.environ.get("DRIVE_RUN_DIR", "/run")
WORK_PARENT = os.environ.get("REPOCTL_WORK_PARENT", "/var/tmp")
# The only paths restore-files may put data back into: the knowledge base's
# own file storage. The snapshot also carries /opt/gmac config (.env and
# friends) — those are never restorable through this path. Open WebUI chat
# uploads were dropped on 2026-10-27: they are no longer backed up, and an
# older snapshot that still holds them must not write them back.
RESTORE_ROOTS = ("/data/haystack-uploads",)
MANIFEST_PATH = "/var/tmp/gmac-backup/MANIFEST.json"
STAGING = "/var/tmp/gmac-backup"
# Document restore (find-deleted / restore-docs): the two document collections,
# reached the same way ops/backup.sh reaches Qdrant and Postgres.
DOC_COLLECTIONS = (os.environ.get("COLLECTION_DOCUMENTS", "documents"),
                   os.environ.get("COLLECTION_CODE", "code_docs"))
DOCS_HELPER = os.environ.get("DOCS_HELPER", os.path.join(INSTALL_DIR, "ops/qdrant_docs.py"))
HAYSTACK = os.environ.get("HAYSTACK_CONTAINER", STACK_NAME + "-haystack")
POSTGRES = os.environ.get("POSTGRES_CONTAINER", STACK_NAME + "-postgres")
PG_USER = os.environ.get("PG_USER", "openwebui")
PG_DB = os.environ.get("PG_DB", "openwebui")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
CONTAINER_TMP = "/tmp/gmac-restore"
# The three subfolders of the backup folder are three restic repositories
# (daily / weekly / monthly); a request says which one it is about
# ("repo": daily | weekly | monthly, daily when absent).
TARGETS = ("daily", "weekly", "monthly")
# A manual backup may take minutes to finish after this service exits (it IS
# the run); read/restore requests hold the shared lock while they work.
LOCK_WAIT_S = float(os.environ.get("REPOCTL_LOCK_WAIT_S", "5"))

# Full restore, phases before the database is touched. repoctl.service
# has TimeoutStartSec=4h; if these hit it, systemd kills the run mid-phase and
# Python's finally never restarts Open WebUI (ExecStopPost does, but the run
# reports nothing). Kept so their worst case sum, 1h15 + 20m + 1h45 = 3h20,
# fails with a reason inside this process, well before the unit's 4h.
FULL_READ_TIMEOUT_S = 75 * 60          # restic restore of the staging tree + kb files
FULL_SHA_TIMEOUT_S = 20 * 60           # sha256sum -c MANIFEST.sha256
FULL_SAFETY_TIMEOUT_S = 105 * 60       # backup.sh --target daily (dump + qdrant + upload)
SNAPSHOT_FIELDS = ("id", "short_id", "time", "tags", "size", "summary")
# Full restore (restore-full). Names and lists come from the same places the
# backup and the drill take them (compose.yml container_name, the COLLECTIONS
# and REQUIRED_TABLES of ops/backup.sh / ops/restore-drill.sh), so
# a restore replaces exactly what a backup captured.
OPENWEBUI = os.environ.get("OPENWEBUI_CONTAINER", STACK_NAME + "-open-webui")
# Stopped for the restore: the two writers of Postgres/Qdrant/kb files. A chat
# or an ingest landing mid-restore would be lost or half-applied. Postgres and
# Qdrant themselves keep running — the restore talks to them.
FULL_RESTORE_STOP = (OPENWEBUI, HAYSTACK)
FULL_COLLECTIONS = tuple(os.environ.get(
    "COLLECTIONS", "documents code_docs visual_frames ocr_frames od_frames "
                   "transcript_segments").split())
REQUIRED_TABLES = tuple(os.environ.get("REQUIRED_TABLES", "user chat function").split())
QDRANT_HELPER = os.environ.get("QDRANT_HELPER", os.path.join(INSTALL_DIR, "ops/qdrant_backup.py"))
# The file trees a full restore replaces. Deliberately NOT RESTORE_ROOTS: that
# list governs single-file restores and changes independently.
FULL_RESTORE_FILES = ("/data/haystack-uploads",)
# Open WebUI chat uploads stopped being backed up on 2026-10-27. Snapshots
# from before still carry them; a full restore must never write them back
# (they would resurrect attachments users deleted since).
IGNORED_SOURCES = ("/data/openwebui/uploads",)
FULL_RESTORE_PHRASE = "RESTORE"


class RepoctlError(Exception):
    """A request failed; the message is safe for the panel (never a secret)."""


class BusyError(RepoctlError):
    """Another backup or restore holds the lock; try again when it finishes."""


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# ── pure helpers (self-tested) ────────────────────────────────────────────

def load_request(path: str) -> dict:
    """The request record, validated. Raises RepoctlError on anything else."""
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
    except FileNotFoundError:
        raise RepoctlError("no request file")
    except (OSError, ValueError) as exc:
        raise RepoctlError("the request file is unreadable (%s)" % type(exc).__name__)
    if not isinstance(rec, dict):
        raise RepoctlError("the request is not a JSON object")
    if rec.get("cmd") not in ("backup-now", "snapshots", "inspect", "restore-files",
                              "find-deleted", "restore-docs", "restore-full", "rebuild-catalog",
                              "find-deleted-chats", "restore-chats",
                              "find-deleted-videos", "restore-videos"):
        raise RepoctlError("unknown command %r" % rec.get("cmd"))
    if rec.get("repo") and rec.get("repo") not in TARGETS:
        raise RepoctlError("unknown backup folder %r (daily, weekly or monthly)"
                           % str(rec.get("repo"))[:20])
    if rec.get("cmd") in ("inspect", "restore-files", "find-deleted", "restore-docs", "restore-full",
                          "find-deleted-chats", "restore-chats",
                          "find-deleted-videos", "restore-videos"):
        if not valid_snapshot_id(rec.get("snapshot")):
            raise RepoctlError("no usable recovery point id in the request")
    if rec.get("cmd") in ("restore-chats", "restore-videos"):
        key = "chats" if rec.get("cmd") == "restore-chats" else "videos"
        items = rec.get(key)
        if not isinstance(items, list) or not items or len(items) > 500 \
                or any(not valid_item_id(i) for i in items):
            raise RepoctlError("no usable %s were selected" % key)
    if rec.get("cmd") == "restore-full":
        check_full_request(rec)
    if rec.get("cmd") == "restore-docs":
        docs = rec.get("docs")
        if not isinstance(docs, list) or not docs:
            raise RepoctlError("no documents were selected")
        for d in docs:
            if not valid_doc(d):
                raise RepoctlError("refusing document %r" % str(d)[:120])
    if rec.get("cmd") == "restore-files":
        paths = rec.get("paths")
        if not isinstance(paths, list) or not paths:
            raise RepoctlError("no files were selected")
        for p in paths:
            if not safe_restore_path(p):
                raise RepoctlError("refusing to restore %r — only files under the knowledge "
                                   "base's own storage can be restored here" % str(p)[:120])
    return rec


def valid_snapshot_id(value) -> bool:
    """A restic short id (or 'latest') — never a flag, never a path."""
    return isinstance(value, str) and (value == "latest" or (
        value.isalnum() and 6 <= len(value) <= 64))


def safe_restore_path(path) -> bool:
    """True for an absolute file path inside one of the two restore roots."""
    if not isinstance(path, str) or not path or "\n" in path or "\r" in path:
        return False
    if not path.startswith("/") or ".." in path.split("/"):
        return False
    return any(path == root or path.startswith(root + "/") for root in RESTORE_ROOTS)


def check_full_request(rec: dict) -> None:
    """A full restore replaces the live database: it needs the typed phrase,
    exactly, and a concrete recovery point. 'latest' is refused because the
    safety backup taken first would itself become 'latest' (in the daily
    folder) — the restore would silently put back the present."""
    if rec.get("confirm") != FULL_RESTORE_PHRASE:
        raise RepoctlError("a full restore needs the typed confirmation %s" % FULL_RESTORE_PHRASE)
    if not valid_snapshot_id(rec.get("snapshot")) or rec.get("snapshot") == "latest":
        raise RepoctlError("a full restore needs a specific recovery point, not 'latest'")
    # Optional, default ON. Only a literal false turns the safety backup off
    # (panel toggle, 2026-10-28: it takes minutes and Drive space during testing).
    if "safety_backup" in rec and not isinstance(rec["safety_backup"], bool):
        raise RepoctlError("safety_backup must be true or false")


def valid_doc(d) -> bool:
    """{"collection": one of DOC_COLLECTIONS, "name": a document name}."""
    if not isinstance(d, dict) or d.get("collection") not in DOC_COLLECTIONS:
        return False
    name = d.get("name")
    return isinstance(name, str) and 0 < len(name) <= 500 and "\n" not in name and "\r" not in name


def valid_table(name) -> bool:
    """Only the structured-data tables (services/db_service.py prefixes data_)."""
    import re
    return isinstance(name, str) and bool(re.fullmatch(r"data_[a-z0-9_]{1,60}", name))


def valid_item_id(value) -> bool:
    """A chat / video id from the database: a bounded opaque string, never a
    flag, a path or a line break."""
    return isinstance(value, str) and 0 < len(value) <= 300 and "\n" not in value \
        and "\r" not in value and not value.startswith("-")


def epoch_ms_iso(value) -> str:
    """Open WebUI's bigint timestamps -> an ISO time ("" when unknown). Rows
    carry SECONDS or MILLISECONDS depending on age — and both must read as
    the same real date, never as 1970."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if v <= 0:
        return ""
    while v > 1e11:                    # ms / µs / ns -> seconds
        v /= 1000
    try:
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return ""


# ── deleted chats / videos (find + restore, never overwriting) ────────────

def copy_block(copy_text: str, table: str) -> tuple:
    """(columns, rows) of ONE table's COPY block in pg_restore --data-only
    output. Rows are raw tab-separated lines: COPY escapes any tab inside a
    value as \\t, so splitting a row on real tabs is exact. The name match is
    exact — 'chat' must never pick up 'chat_message' or 'chat_file'."""
    cols, rows, inside = [], [], False
    for line in (copy_text or "").splitlines():
        if line.startswith("COPY "):
            head = line.split(" FROM ")[0]
            # Reserved-word names arrive quoted (public."user"): split the
            # schema off first, THEN drop the quotes.
            name = head[5:].split("(")[0].strip().split(".")[-1].strip('"')
            if name == table and "(" in head:
                cols = [c.strip() for c in head[head.index("(") + 1:head.rindex(")")].split(",")]
                inside = True
            continue
        if inside:
            if line == "\\.":
                inside = False
                continue
            rows.append(line)
    return cols, rows


def row_field(row: str, columns: list, name: str) -> str:
    """One column of a COPY row, by name ("" when the shape is unexpected)."""
    try:
        return row.split("\t")[columns.index(name)]
    except (ValueError, IndexError):
        return ""


def deleted_rows(rows, live_keys, columns: list, key: str) -> list:
    """The backup's rows the live database no longer has — what a restore can
    bring back (rows that still exist are never part of a restore)."""
    live = set(live_keys or [])
    return [r for r in rows or [] if row_field(r, columns, key) not in live]


def restore_rows_sql(table: str, columns: list, rows: list, conflict: str) -> str:
    """INSERT the backup's rows only where they do not exist yet (ON CONFLICT
    DO NOTHING): restoring NEVER overwrites a row that is live now. Same shape
    as catalog_sql — a temp table first, so an interrupted COPY can never
    leave half a chat behind."""
    if not rows or not columns:
        return ""
    quoted = '"%s"' % table
    lines = ["BEGIN;",
             "CREATE TEMP TABLE restore_rows (LIKE %s INCLUDING DEFAULTS) ON COMMIT DROP;" % quoted,
             "COPY restore_rows (%s) FROM stdin;" % ", ".join(columns)]
    lines.extend(rows)
    lines.append("\\.")
    lines.append("INSERT INTO %s (%s) SELECT %s FROM restore_rows ON CONFLICT (%s) DO NOTHING;"
                 % (quoted, ", ".join(columns), ", ".join(columns), conflict))
    lines.append("COMMIT;")
    return "\n".join(lines) + "\n"


def chat_restore_sql(tables: dict, chat_ids: list) -> str:
    """SQL that brings back the selected chats and everything that hangs on
    them: their accounts, their messages, their file links — each row only
    when it is missing live. Tables come from the backup's dump
    ({name: (columns, rows)})."""
    wanted = {str(c) for c in chat_ids or []}
    out = []
    for table in ("user", "chat", "chat_message", "chat_file"):
        cols, rows = tables.get(table) or ([], [])
        if table == "chat":
            picked = [r for r in rows if row_field(r, cols, "id") in wanted]
        elif table == "chat_message":
            picked = [r for r in rows if row_field(r, cols, "chat_id") in wanted]
        elif table == "chat_file":
            picked = [r for r in rows if row_field(r, cols, "chat_id") in wanted]
        else:                                   # user: everyone in these chats
            users = {row_field(r, cols2, "user_id")
                     for t2, (cols2, rows2) in tables.items() if t2 in ("chat", "chat_message")
                     for r in rows2 if (t2 == "chat" and row_field(r, cols2, "id") in wanted)
                     or (t2 == "chat_message" and row_field(r, cols2, "chat_id") in wanted)}
            picked = [r for r in rows if row_field(r, cols, "id") in users]
        out.append(restore_rows_sql(table, cols, picked, "id"))
    return "".join(out)


def chat_summary(row: str, columns: list, users: dict, messages: dict) -> dict:
    """One deleted chat as the panel lists it: what it was called, whose it
    was (the account), when it was last touched and how much is in it."""
    chat_id = row_field(row, columns, "id")
    user_id = row_field(row, columns, "user_id")
    account = users.get(user_id) or {}
    return {"id": chat_id, "title": row_field(row, columns, "title") or "(untitled chat)",
            "user_id": user_id, "account": account.get("name") or account.get("email") or "unknown account",
            "email": account.get("email") or "",
            "when": epoch_ms_iso(row_field(row, columns, "updated_at")),
            "messages": messages.get(chat_id, 0)}


def video_restore_sql(columns: list, rows: list) -> str:
    """The video catalog entries of a selection. Processed content (frames,
    transcripts) is vector data and comes back with a full restore — said on
    every video restore, never implied."""
    return restore_rows_sql("video_catalog", columns, rows, "video_id")


def deleted_docs(backup_docs: list, live_docs: list, collection: str) -> list:
    """Documents in the backup with no section live any more."""
    live = {d["name"] for d in live_docs}
    return [dict(d, collection=collection) for d in backup_docs if d["name"] not in live]


def catalog_sql(copy_text: str, tables: list) -> str:
    """SQL that puts back ONLY the catalog rows of `tables`, never touching an
    existing row: the backup's data_catalog COPY block goes into a temporary
    table first. `copy_text` is pg_restore's --data-only output."""
    lines, block, inside = [], [], False
    for line in copy_text.splitlines():
        if line.startswith("COPY ") and "data_catalog" in line.split(" FROM ")[0]:
            cols = line[line.index("("):line.index(")") + 1]
            block.append("COPY restore_catalog %s FROM stdin;" % cols)
            inside = True
            continue
        if inside:
            block.append(line)
            if line == "\\.":
                inside = False
    if not block:
        return ""
    names = ", ".join("'%s'" % t for t in tables if valid_table(t))
    lines.append("BEGIN;")
    lines.append("CREATE TEMP TABLE restore_catalog (LIKE data_catalog INCLUDING DEFAULTS) ON COMMIT DROP;")
    lines.extend(block)
    lines.append("INSERT INTO data_catalog SELECT * FROM restore_catalog WHERE table_name IN (%s) "
                 "ON CONFLICT (table_name) DO NOTHING;" % (names or "NULL"))
    lines.append("COMMIT;")
    return "\n".join(lines) + "\n"


def glob_escape(path: str) -> str:
    """restic --include pattern that matches this path literally (its pattern
    language uses * ? [...] — 'CME_MINI_DL_ES1!, 5.csv' style names included)."""
    out = []
    for ch in path:
        out.append("[%s]" % ch if ch in "*?[" else ch)
    return "".join(out)


def snapshot_row(rec: dict) -> dict:
    """One recovery point as the panel shows it (no host paths, no secrets)."""
    summary = rec.get("summary") or {}
    return {"id": rec.get("short_id") or rec.get("id", ""),
            "time": (rec.get("time") or "").replace("T", " ").replace("Z", "")[:16],
            "tags": sorted(t for t in (rec.get("tags") or []) if t != "gmac-backup"),
            "size_bytes": rec.get("size") or summary.get("total_bytes_processed") or 0,
            "files": summary.get("total_files_processed", 0)}


def parse_ndjson(text: str) -> list:
    """`restic ls --json` is newline-delimited JSON (the snapshot record first,
    then one node per line) — parse it tolerantly."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def inventory_rows(nodes: list) -> tuple:
    """([{'path','size'}] for the restorable files in a snapshot, count of the
    rest) — only the two file roots are offered back; everything else in the
    snapshot (staging, config) is counted but never restorable here."""
    rows, other = [], 0
    for entry in nodes or []:
        path, size = entry.get("path", ""), int(entry.get("size") or 0)
        if entry.get("type") != "file":
            continue
        if safe_restore_path(path):
            rows.append({"path": path, "size": size})
        else:
            other += 1
    rows.sort(key=lambda r: r["path"])
    return rows, other


def plan_restore(inventory: list, selected: list) -> dict:
    """What a restore would do: 'restore' (in the backup and selectable),
    'missing' (selected but not in this recovery point). Existing files are
    decided server-side at apply time and are never overwritten."""
    by_path = {row["path"]: row for row in inventory}
    plan = {"restore": [], "missing": []}
    for path in selected:
        row = by_path.get(path)
        (plan["restore"] if row else plan["missing"]).append(row or {"path": path})
    return plan


def manifest_summary(manifest: dict) -> dict:
    """The parts of MANIFEST.json the panel shows (counts and versions; the
    file entries carry names and hashes only)."""
    out = {}
    for key in ("created_at", "git_revision", "hostname", "collections",
                "versions", "totals"):
        if key in manifest:
            out[key] = manifest[key]
    entries = manifest.get("files") or []
    out["file_count"] = len(entries)
    return out


# Who asked and when, copied into every status write of the current request,
# so the panel can say "started 16:24 UTC by the operator" -- a status shown after a
# page refresh must never look like the refresh started it.
CURRENT = {"by": "", "requested_at": ""}


def status_doc(cmd: str, state: str, request_id: str = "", step: str = "",
               result=None, error: str = "", sub: float = 0.0) -> dict:
    return {"id": request_id, "cmd": cmd, "state": state, "step": step,
            "at": now_iso(), "result": result or {}, "error": error,
            "progress": progress_pct(cmd, state, step, sub),
            "by": CURRENT["by"], "requested_at": CURRENT["requested_at"]}


# ── progress (the panel's percentage bar) ─────────────────────────────────
# Without a number the panel showed only a step name for minutes at a time
# (reading a backup re-downloads the ~650 MB database dump from Drive), and an
# operator could not tell a slow run from a stuck one (2026-10-28). Each
# command's steps, in order, with the percentage at which each one starts; the
# step's own measured progress (`sub`, 0..1: restic's percent_done, or the
# backup script's log stage) fills the band up to the next step's start. The
# bands are rough weights of how long each step takes, not a time estimate.
PROGRESS_BANDS = {
    "restore-full": [("checking the request", 0), ("reading the backup", 2),
                     ("stopping Open WebUI", 35), ("taking a safety backup", 37),
                     ("stopping the backend", 70), ("restoring the database", 72),
                     ("restoring the knowledge-base files", 85),
                     ("restoring the vector store", 88), ("verifying", 96)],
    "backup-now": [("", 0)],
    "snapshots": [("", 0)],
    "inspect": [("", 0)],
    "rebuild-catalog": [("", 0)],
}
# Every other command reads part of the backup, then works on it locally.
# "" matches any step, so whatever follows the read lands in the last band.
DEFAULT_BANDS = [("reading the backup", 0), ("", 85)]


def progress_pct(cmd: str, state: str, step: str = "", sub: float = 0.0):
    """0-100 for the panel, or None when there is nothing honest to show."""
    if state == "done":
        return 100
    if state != "running":
        return None
    if step == "started":           # main()'s first write, before any command step
        return 0
    bands = PROGRESS_BANDS.get(cmd, DEFAULT_BANDS)
    for i, (name, start) in enumerate(bands):
        if step.startswith(name):
            end = bands[i + 1][1] if i + 1 < len(bands) else 100
            # Never 100 while running: the last few % are the result write-up.
            return min(99, int(start + (end - start) * min(max(sub, 0.0), 1.0)))
    return None


# The backup script's log lines -> how far through its run it is. Measured on
# the 2026-10-28 runs: the dump is about a third of the local work, the upload
# to Drive the largest single part of a run that has data to send.
BACKUP_STAGES = [("postgres: dumping", 0.02), ("postgres: dump ok", 0.30),
                 ("qdrant: snapshotting", 0.32), ("manifest: writing", 0.45),
                 ("verify: all staged", 0.50), ("drive:", 0.52),
                 ("restic: backing up", 0.55), ("restic: snapshot", 0.85),
                 ("restic: check", 0.88), ("retention", 0.93), ("catalog", 0.97)]


def backup_stage(line: str, current: float) -> float:
    """The fraction a backup-script log line marks; never moves backwards."""
    text = line.split(" ", 1)[1] if line[:4].isdigit() and " " in line else line
    for prefix, frac in BACKUP_STAGES:
        if text.startswith(prefix):
            return max(current, frac)
    return current


# ── database restore: rows that point at nothing ──────────────────────────
# pg_restore prints a comment header before each object; the first foreign
# key's header is where the orphan cleanup goes (data loaded, keys not yet on).
FK_MARKER = b"; Type: FK CONSTRAINT;"
_FK_RE = re.compile(r"ALTER TABLE ONLY (\S+)\s+ADD CONSTRAINT (\S+) FOREIGN KEY \(([^)]*)\) "
                    r"REFERENCES (\S+?)\(([^)]*)\)")


def foreign_keys(post_data_sql: str) -> list:
    """[(table, name, [cols], ref_table, [ref_cols])] from pg_restore's post-data SQL."""
    out = []
    for m in _FK_RE.finditer(post_data_sql):
        out.append((m.group(1), m.group(2), [c.strip() for c in m.group(3).split(",")],
                    m.group(4), [c.strip() for c in m.group(5).split(",")]))
    return out


def orphan_cleanup_sql(fks: list) -> str:
    """One DELETE per foreign key: rows whose (non-null) key matches nothing.
    The count is printed as 'orphans <n>' (psql -q still prints SELECT output)."""
    if not fks:
        return ""
    parts = []
    for table, _name, cols, ref, refcols in fks:
        notnull = " AND ".join("t.%s IS NOT NULL" % c for c in cols)
        match = " AND ".join("r.%s = t.%s" % (rc, c) for rc, c in zip(refcols, cols))
        parts.append("DELETE FROM %s t WHERE %s AND NOT EXISTS (SELECT 1 FROM %s r WHERE %s)"
                     % (table, notnull, ref, match))
    ctes = ", ".join("d%d AS (%s RETURNING 1)" % (i, p) for i, p in enumerate(parts))
    total = " + ".join("(SELECT count(*) FROM d%d)" % i for i in range(len(parts)))
    return "\n-- orphan cleanup (repoctl)\nWITH %s SELECT 'orphans ' || (%s);\n" % (ctes, total)


# Every backup's dump is taken DURING that backup, so it holds that run's own
# ops_backup_reports row still saying 'running'. Restored, that row showed
# "Last run: Running, started 11:50" for good (2026-10-28) — no run was going:
# a full restore holds the backup lock. That run SUCCEEDED — its snapshot is the
# one being restored — so it is recorded as a success with that snapshot id.
# (First version marked it 'failed': the dashboard then said "The most recent
# backup run failed: interrupted", which was false.) Same transaction. The
# name is schema-qualified: pg_dump's SQL empties search_path, and the bare
# name failed every restore with 'relation does not exist' (2026-10-28).
def stale_runs_sql(snapshot_id: str) -> str:
    sid = snapshot_id if re.fullmatch(r"[0-9a-f]{8,64}", snapshot_id or "") else ""
    return """
DO $$ BEGIN
  IF to_regclass('public.ops_backup_reports') IS NOT NULL THEN
    UPDATE public.ops_backup_reports SET status = 'success', ok = true, error = NULL,
           finished_at = COALESCE(finished_at, now()), snapshot_id = NULLIF('%s', '')
     WHERE status = 'running';
  END IF;
END $$;
""" % sid


# The backup history (ops_backup_reports) lives in the same database, so a full
# restore rewound it to the backup's moment: the 13:03, 14:12 and 15:39 runs of
# 2026-10-28 vanished from the dashboard after the 15:33 restore. The live rows
# are read before the restore and every one the backup lacks is put back, in
# the same transaction. Explicit columns: the table's shape is fixed by
# ops/backup.sh.
RUN_COLUMNS = "id, started_at, finished_at, ok, status, snapshot_id, bytes_added, error, detail"


def keep_runs_sql(copy_text: str) -> str:
    """SQL re-adding the live backup-history rows a restore would lose. Empty
    when there were none (or the live table does not exist)."""
    if not copy_text.strip():
        return ""
    return ("\nCREATE TEMP TABLE _keep_runs (id int, started_at timestamptz, finished_at "
            "timestamptz, ok boolean, status text, snapshot_id text, bytes_added bigint, "
            "error text, detail jsonb);\n"
            "COPY _keep_runs (%s) FROM stdin;\n%s\\.\n"
            "DO $$ BEGIN\n"
            "  IF to_regclass('public.ops_backup_reports') IS NOT NULL THEN\n"
            "    INSERT INTO public.ops_backup_reports (%s) SELECT %s FROM _keep_runs k\n"
            "     WHERE NOT EXISTS (SELECT 1 FROM public.ops_backup_reports r WHERE r.id = k.id);\n"
            "    PERFORM setval(pg_get_serial_sequence('public.ops_backup_reports', 'id'),\n"
            "                   GREATEST((SELECT max(id) FROM public.ops_backup_reports), 1));\n"
            "  END IF;\n"
            "END $$;\n" % (RUN_COLUMNS, copy_text if copy_text.endswith("\n") else copy_text + "\n",
                           RUN_COLUMNS, RUN_COLUMNS))


def orphans_removed(psql_out: str) -> int:
    m = re.search(r"orphans (\d+)", psql_out)
    return int(m.group(1)) if m else 0


def known_record(catalog: dict, snap_id: str) -> bool:
    """True when the catalog already describes this snapshot from its manifest
    (it has contents). A bare record (manifest missing or unreadable last
    time) is read again."""
    for rec in (catalog or {}).get("snapshots") or []:
        ids = {str(rec.get("id") or ""), str(rec.get("short_id") or "")}
        if (snap_id in ids or str(rec.get("id") or "").startswith(snap_id)
                or snap_id.startswith(str(rec.get("short_id") or "-"))) and rec.get("contains"):
            return True
    return False


# ── files and locks (side effects, tiny and total) ────────────────────────

def write_json(path: str, obj: dict) -> None:
    # This runs as ROOT into /data/backup-drives/ctl, a directory the admin
    # panel container can write. A plain O_CREAT|O_TRUNC open would follow a
    # symlink planted at the (predictable) tmp name and truncate whatever it
    # points to — /etc/shadow included. O_EXCL refuses any existing name and
    # O_NOFOLLOW refuses a symlink outright; a leftover (link or stale file)
    # is unlinked — which removes the link itself, never its target — and the
    # create retried once.
    tmp = "%s.%d.tmp" % (path, os.getpid())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(tmp, flags, 0o600)
    except FileExistsError:
        os.unlink(tmp)
        fd = os.open(tmp, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def write_status(cmd: str, state: str, request_id: str = "", step: str = "",
                 result=None, error: str = "", sub: float = 0.0) -> dict:
    doc = status_doc(cmd, state, request_id, step, result, error, sub)
    LAST_STATUS.clear()
    LAST_STATUS.update(cmd=cmd, state=state, request_id=request_id, step=step, result=result,
                       error=error, sub=sub, written=time.monotonic())
    try:
        os.makedirs(CTL_DIR, mode=0o700, exist_ok=True)
        write_json(os.path.join(CTL_DIR, "status.json"), doc)
    except OSError as exc:
        sys.stderr.write("repoctl: cannot write status.json: %s\n" % exc)
    return doc


# The last status written, so a long step can report its own progress without
# every caller threading cmd / request id / result through to restic.
LAST_STATUS = {}


def tick(sub: float) -> None:
    """Move the current step's bar to `sub` (0..1). At most one write every 2 s
    (restic reports several times a second) and only while a step is running."""
    last = LAST_STATUS
    if last.get("state") != "running" or sub <= last.get("sub", 0.0):
        return
    if time.monotonic() - last.get("written", 0.0) < 2.0 and sub < 1.0:
        return
    write_status(last["cmd"], "running", last["request_id"], last["step"], last["result"],
                 last["error"], sub)


# ── the backup catalog (catalog.json) ─────────────────────────────────────
# One record per recovery point; the panel's "Restore backup" list reads the
# LOCAL copy. Written by ops/backup.sh at the end of every run; here
# after a full restore's safety backup and by rebuild-catalog.

def catalog_path() -> str:
    return os.environ.get("BACKUP_CATALOG_FILE") or os.path.join(CTL_DIR, "catalog.json")


def catalog_safety_record(safety_id: str, request_id: str) -> None:
    """Record the safety snapshot for what it is: the way back from this
    restore. The backup script's own catalog write (with the manifest) is
    richer — the merge keeps its facts; this only guarantees the record is
    there and says what it protects."""
    rec = backup_catalog.record_from_manifest({}, {
        "id": safety_id, "short_id": safety_id[:8], "folder": "daily",
        "kind": "safety", "status": "unverified",
        "note": "safety backup before restore %s" % request_id,
        "taken_at": now_iso()})
    merged = backup_catalog.merge_record(backup_catalog.read_catalog_file(catalog_path()), rec)
    os.makedirs(CTL_DIR, mode=0o700, exist_ok=True)
    write_json(catalog_path(), merged)


def _catalog_upload(env) -> None:
    """The catalog's Drive copy next to the repositories is a spare for
    disaster recovery; the panel reads the local one. Best effort: an upload
    problem is one log line, never a failed request."""
    repo = (env.get("RESTIC_REPOSITORY") or "")
    root = repo[len("rclone:"):] if repo.startswith("rclone:") else ""
    root = root.rsplit("/", 1)[0] if "/" in root else ""
    if not root:
        return
    try:
        subprocess.run(["rclone", "copyto", catalog_path(), root + "/catalog.json"],
                       env=env, capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write("repoctl: catalog upload skipped (%s)\n" % type(exc).__name__)


class JobLock:
    """The nightly backup's own lock: one repository job at a time."""

    def __init__(self, path: str = LOCK_FILE, wait_s: float = LOCK_WAIT_S):
        self.path, self.wait_s, self.fd = path, wait_s, None

    def __enter__(self):
        if fcntl is None:                         # self-test machines only
            raise RepoctlError("file locks are unavailable on this machine")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600)
        deadline = time.time() + self.wait_s
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.time() >= deadline:
                    os.close(self.fd)
                    self.fd = None
                    raise BusyError("another backup or restore is running — try again when it "
                                    "finishes")
                time.sleep(0.2)

    def __exit__(self, *exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
        return False


def env_get(key: str) -> str:
    """One key from /opt/gmac/.env (same reader as backup.sh)."""
    try:
        with open(ENV_FILE, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


class RepoEnv:
    """RESTIC_* plus the rclone config restic's rclone should use: the panel's
    connected Google Drive account when there is one (the same override the
    nightly backup applies, ops/backup_drives.py), else the server's own. The
    backup folder holds one repository per subfolder (daily/weekly/monthly) -
    `target` is the subfolder this request is about."""

    def __init__(self, target: str = "daily"):
        self.env = dict(os.environ)
        self.env.setdefault("RESTIC_REPOSITORY", env_get("RESTIC_REPOSITORY"))
        self.env.setdefault("RESTIC_PASSWORD", env_get("RESTIC_PASSWORD"))
        root = (self.env.get("RESTIC_REPOSITORY") or "").rstrip("/")
        if root and target in TARGETS:
            self.env["RESTIC_REPOSITORY"] = root + "/" + target
        self.tmp_conf = ""
        self.source = "server"
        self.note = ""

    def setup(self):
        base = os.environ.get("RCLONE_BASE_CONFIG", "/root/.config/rclone/rclone.conf")
        override = os.environ.get("BACKUP_DRIVE_OVERRIDE", "/data/backup-drives/primary.json")
        repo = self.env.get("RESTIC_REPOSITORY") or ""
        rest = repo.split(":", 1)[1] if repo.startswith("rclone:") else ""   # <remote>:<path>
        remote = rest.split(":", 1)[0] if rest else ""
        if not os.path.exists(override) and \
                os.path.exists(os.path.join(os.path.dirname(override), "signed-out.json")):
            # Signed out on the handler page: NOT connected -- never fall back to
            # the server's own remote (the previous account).
            raise RepoctlError("Google Drive is not connected (signed out on the Drive Backup & Upload "
                               "Handler page) — sign in with Google first")
        if not remote or not os.path.exists(override):
            return self
        try:
            sys.path.insert(0, os.path.dirname(DRIVES_HELPER))
            import backup_drives
            rec = backup_drives.load_override(override)
            if not rec:
                return self
            os.makedirs(RUN_DIR, exist_ok=True)
            fd, self.tmp_conf = _mkstemp0600(RUN_DIR, "gmac-rclone.")
            with open(base, encoding="utf-8") as fh:
                text = fh.read()
            os.write(fd, backup_drives.apply_override(text, remote, rec).encode())
            os.close(fd)
            self.env["RCLONE_CONFIG"] = self.tmp_conf
            self.source = "panel"
        except Exception as exc:                       # noqa: BLE001 - fall back
            self.note = "the panel's Google Drive sign-in could not be applied (%s)" % type(exc).__name__
            self.close()
        return self

    def close(self):
        if self.tmp_conf:
            try:
                os.remove(self.tmp_conf)
            except OSError:
                pass
            self.tmp_conf = ""


def _mkstemp0600(directory: str, prefix: str):
    import tempfile
    return tempfile.mkstemp(prefix=prefix, dir=directory)


# ── restic ────────────────────────────────────────────────────────────────

def restic(env: dict, *args, timeout: float = 1800, progress: bool = False):
    """One restic call -> (rc, stdout, stderr). The password travels only in
    the environment; the journal records the arguments, never secrets."""
    cmd = ["restic"] + list(args)
    # The folder in brackets: "unlock" three times in a row read as a retry
    # loop; it is one call per folder (daily/weekly/monthly).
    folder = ((env or {}).get("RESTIC_REPOSITORY") or "").rstrip("/").rsplit("/", 1)[-1]
    sys.stderr.write("repoctl: %s%s\n" % (" ".join(args[:3]), (" [%s]" % folder) if folder else ""))
    if progress:
        return _streamed(cmd + ["--json"], env, timeout, _restic_fraction, "restic %s" % args[0])
    try:
        run = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RepoctlError("restic %s failed to run (%s)" % (args[0], type(exc).__name__))
    return run.returncode, run.stdout, run.stderr


def _restic_fraction(line: str, current: float) -> float:
    """restic --json status line -> percent_done (0..1); anything else keeps `current`."""
    if '"percent_done"' not in line:
        return current
    try:
        return max(current, float(json.loads(line).get("percent_done") or 0.0))
    except (ValueError, AttributeError, TypeError):
        return current


def _streamed(cmd: list, env, timeout: float, fraction, what: str):
    """Run cmd reading stdout line by line, ticking the panel's bar with
    fraction(line, so_far) -> 0..1. Same (rc, stdout, stderr) as subprocess.run;
    stderr drains on a thread so a chatty command can never fill the pipe and
    hang. The timeout kills the command, as subprocess.run's would."""
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
    except OSError as exc:
        raise RepoctlError("%s failed to run (%s)" % (what, type(exc).__name__))
    err = []
    drain = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)
    drain.start()
    killer = threading.Timer(timeout, proc.kill)
    killer.start()
    out, so_far = [], 0.0
    try:
        for line in proc.stdout:
            out.append(line)
            so_far = fraction(line.strip(), so_far)
            tick(so_far)
        rc = proc.wait()
    finally:
        timed_out = not killer.is_alive()
        killer.cancel()
    drain.join(timeout=30)
    if timed_out:
        raise RepoctlError("%s failed to run (TimeoutExpired)" % what)
    return rc, "".join(out), "".join(err)


# ── commands ──────────────────────────────────────────────────────────────

def cmd_backup_now(request: dict, target: str = "daily") -> dict:
    """Run the backup job's own script into the chosen folder (daily by
    default — the folder the three-times-a-day timer uses). It holds its own
    lock, so this can never overlap or double-push against a scheduled run."""
    if not os.path.exists(BACKUP_SCRIPT):
        raise RepoctlError("%s is missing" % BACKUP_SCRIPT)
    write_status("backup-now", "running", request.get("id", ""), "starting the backup")
    rc, stdout, stderr = _streamed(["bash", BACKUP_SCRIPT, "--target", target], None, 4 * 3600,
                                   backup_stage, "the backup script")
    run = subprocess.CompletedProcess([], rc, stdout, stderr)
    tail = [l for l in (run.stdout or "").splitlines() if l.strip()]
    if run.returncode != 0:
        detail = (run.stdout or "") + "\n" + (run.stderr or "")
        errors = [l for l in detail.splitlines()
                  if ("ERROR" in l or "Fatal" in l or "lock" in l.lower()) and l.strip()]
        raise RepoctlError(_why(errors[-1] if errors else
                                "the backup failed (exit %d)" % run.returncode))
    snapshot = ""
    for line in tail:
        if "backup complete: snapshot " in line:
            snapshot = line.split("snapshot ", 1)[1].strip()
    return {"snapshot_id": snapshot, "log": tail[-3:]}


def _why(message: str) -> str:
    """One safe, actionable line: the real message, plus what to do about a
    repository lock (the classic cause of a failed check)."""
    msg = message.split("ERROR:", 1)[-1].strip()[:300] or "unknown error"
    if "lock" in msg.lower() or "already locked" in msg.lower():
        msg += (" — if no backup or restore is running, `restic unlock` on the server "
                "clears a stale lock")
    return msg


def cmd_snapshots(env: dict) -> dict:
    rc, out, err = restic(env.env, "snapshots", "--json", timeout=120)
    if rc != 0:
        raise RepoctlError(_restic_why(err, "could not read the backup repository"))
    rows = [snapshot_row(r) for r in json.loads(out or "[]")]
    rows.sort(key=lambda r: r["time"], reverse=True)
    return {"snapshots": rows, "count": len(rows), "drive_source": env.source,
            "drive_note": env.note}


def _restic_why(err: str, fallback: str) -> str:
    for line in (err or "").splitlines():
        if "Fatal:" in line or "error" in line.lower() or "lock" in line.lower():
            return _why(line.replace("Fatal:", "").strip() or fallback)
    return fallback


def _one_snapshot(env: dict, snapshot: str) -> dict:
    rc, out, _ = restic(env.env, "snapshots", snapshot, "--json", timeout=120)
    recs = json.loads(out or "[]") if rc == 0 else []
    if not recs:
        raise RepoctlError("recovery point %s is not in the repository (kept backups only "
                           "survive retention — see the list)" % snapshot[:12])
    return recs[-1]


def cmd_inspect(env: dict, request: dict) -> dict:
    """Available + readable before any restore: the snapshot record, a
    repository check, the manifest it was taken with and what it holds."""
    snapshot = request["snapshot"]
    step = "reading the recovery point"
    write_status("inspect", "running", request.get("id", ""), step)
    rec = _one_snapshot(env, snapshot)
    rc, _, err = restic(env.env, "check", timeout=3600)
    if rc != 0:
        raise RepoctlError("the repository check failed — this recovery point cannot be "
                           "trusted: %s" % _restic_why(err, "restic check failed"))
    manifest = {}
    rc, out, _ = restic(env.env, "dump", snapshot, MANIFEST_PATH, timeout=600)
    if rc == 0:
        try:
            manifest = manifest_summary(json.loads(out))
        except ValueError:
            manifest = {"file_count": 0, "note": "the manifest is not readable"}
    write_status("inspect", "running", request.get("id", ""), "listing the files")
    rc, out, err = restic(env.env, "ls", snapshot, "--json", timeout=600)
    if rc != 0:
        raise RepoctlError("the file inventory could not be read: %s"
                           % _restic_why(err, "restic ls failed"))
    files, other = inventory_rows(parse_ndjson(out))
    return {"snapshot": snapshot_row(rec), "readable": True,
            "checks": ["repository check passed", "manifest read", "file inventory read"],
            "manifest": manifest, "files": files, "restorable_files": len(files),
            "other_files": other, "drive_source": env.source, "drive_note": env.note}


def cmd_restore_files(env: dict, request: dict) -> dict:
    """Additive restore of selected files: verified in scratch first, moved
    into place only where the file is missing, verified again afterwards."""
    snapshot, selected = request["snapshot"], request["paths"]
    request_id = request.get("id", "")
    rec = _one_snapshot(env, snapshot)
    rc, out, err = restic(env.env, "ls", snapshot, "--json", timeout=600)
    if rc != 0:
        raise RepoctlError("the file inventory could not be read: %s"
                           % _restic_why(err, "restic ls failed"))
    inventory, _ = inventory_rows(parse_ndjson(out))
    plan = plan_restore(inventory, list(selected))
    if plan["missing"] and not plan["restore"]:
        raise RepoctlError("none of the selected files are in recovery point %s"
                           % snapshot[:12])
    targets = [row["path"] for row in plan["restore"]]
    write_status("restore-files", "running", request_id,
                 "restoring %d file(s) from %s" % (len(targets), snapshot[:12]))
    work = os.path.join(WORK_PARENT, "gmac-repoctl.%d" % os.getpid())
    try:
        os.makedirs(work, mode=0o700, exist_ok=True)
        include = []
        for path in targets:
            include += ["--include", glob_escape(path)]
        rc, _, err = restic(env.env, "restore", snapshot, "--target", work, "--verify", *include,
                            timeout=3600)
        if rc != 0:
            raise RepoctlError("restic could not restore the selection: %s"
                               % _restic_why(err, "restic restore failed"))
        restored, skipped, failed = [], [], []
        for row in plan["restore"]:
            src = os.path.join(work, row["path"].lstrip("/"))
            live = row["path"]
            try:
                if not os.path.isfile(src):
                    failed.append({"path": live, "why": "not present in the restored set"})
                    continue
                if os.path.getsize(src) != row["size"]:
                    failed.append({"path": live, "why": "restored size does not match the "
                                                        "snapshot (corrupt?)"})
                    continue
                if os.path.exists(live):
                    skipped.append({"path": live, "size": os.path.getsize(live),
                                    "why": "already exists — existing data is never overwritten"})
                    continue
                os.makedirs(os.path.dirname(live), exist_ok=True)
                os.replace(src, live)
                if os.path.getsize(live) != row["size"]:
                    failed.append({"path": live, "why": "the live copy does not verify"})
                    continue
                restored.append({"path": live, "size": row["size"]})
            except OSError as exc:
                failed.append({"path": live, "why": type(exc).__name__})
        write_status("restore-files", "running", request_id,
                     "%d restored, %d skipped" % (len(restored), len(skipped)))
        return {"snapshot": snapshot_row(rec), "restored": restored,
                "skipped_existing": skipped, "failed": failed,
                "missing_from_backup": [m["path"] for m in plan["missing"]],
                "verified": len(restored), "drive_source": env.source, "drive_note": env.note}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ── deleted documents (find-deleted / restore-docs) ───────────────────────

def _docker(*args, stdin=None, timeout=600, text=True):
    try:
        return subprocess.run(["docker"] + list(args), input=stdin, capture_output=True,
                              text=text, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RepoctlError("docker %s failed to run (%s)" % (args[0], type(exc).__name__))


def qdocs(*args, timeout=1800):
    """ops/qdrant_docs.py inside the haystack container (Qdrant is not on the host)."""
    with open(DOCS_HELPER, encoding="utf-8") as fh:
        script = fh.read()
    run = _docker("exec", "-i", "-e", "QDRANT_URL=" + QDRANT_URL, HAYSTACK, "python3", "-", *args,
                  stdin=script, timeout=timeout)
    if run.returncode != 0:
        raise RepoctlError("the vector store step failed: %s"
                           % ((run.stderr or "").strip().splitlines() or ["unknown error"])[-1][:300])
    return json.loads(run.stdout or "null")


def _restore_staging(env: dict, snapshot: str, work: str, names: list) -> None:
    include = []
    for n in names:
        include += ["--include", glob_escape(os.path.join(STAGING, n))]
    rc, _, err = restic(env.env, "restore", snapshot, "--target", work, *include, timeout=3600,
                        progress=True)
    if rc != 0:
        raise RepoctlError("the backup could not be read: %s" % _restic_why(err, "restic restore failed"))


def _load_collections(work: str, collections: list, tag: str) -> dict:
    """Backup snapshot of each collection -> a temporary collection. {live: tmp}."""
    qdocs("cleanup")                              # leftovers of an interrupted run
    if _docker("exec", HAYSTACK, "mkdir", "-p", CONTAINER_TMP).returncode != 0:
        raise RepoctlError("could not prepare the vector store container")
    tmps = {}
    for c in collections:
        src = os.path.join(work, STAGING.lstrip("/"), "qdrant", "%s.snapshot" % c)
        if not os.path.isfile(src):
            raise RepoctlError("this backup has no %s snapshot" % c)
        dest = "%s/%s.snapshot" % (CONTAINER_TMP, c)
        if _docker("cp", src, "%s:%s" % (HAYSTACK, dest), timeout=1800).returncode != 0:
            raise RepoctlError("could not hand the %s snapshot to the vector store" % c)
        tmp = "restore_tmp_%s_%s" % (c, tag)
        qdocs("load", tmp, dest)
        tmps[c] = tmp
        _docker("exec", HAYSTACK, "rm", "-f", dest)
    return tmps


def _drop_collections(tmps: dict) -> None:
    for tmp in tmps.values():
        try:
            qdocs("drop", tmp, timeout=300)
        except RepoctlError as exc:
            sys.stderr.write("repoctl: %s\n" % exc)
    _docker("exec", HAYSTACK, "rm", "-rf", CONTAINER_TMP)


def cmd_find_deleted(env: RepoEnv, request: dict) -> dict:
    snapshot, request_id = request["snapshot"], request.get("id", "")
    rec = _one_snapshot(env, snapshot)
    work = os.path.join(WORK_PARENT, "gmac-repoctl.%d" % os.getpid())
    tmps = {}
    try:
        os.makedirs(work, mode=0o700, exist_ok=True)
        write_status("find-deleted", "running", request_id, "reading the backup")
        _restore_staging(env, snapshot, work, ["qdrant/%s.snapshot" % c for c in DOC_COLLECTIONS])
        write_status("find-deleted", "running", request_id, "comparing with the live knowledge base")
        tmps = _load_collections(work, list(DOC_COLLECTIONS), str(os.getpid()))
        found, totals = [], {}
        for c in DOC_COLLECTIONS:
            backup_docs = qdocs("sources", tmps[c])
            live_docs = qdocs("sources", c)
            totals[c] = {"backup": len(backup_docs), "live": len(live_docs)}
            found += deleted_docs(backup_docs, live_docs, c)
        return {"snapshot": snapshot_row(rec), "deleted": found, "totals": totals,
                "drive_source": env.source, "drive_note": env.note}
    finally:
        _drop_collections(tmps)
        shutil.rmtree(work, ignore_errors=True)


def _psql(sql: str, timeout=600):
    run = _docker("exec", "-i", POSTGRES, "psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-At",
                  "-U", PG_USER, "-d", PG_DB, stdin=sql, timeout=timeout)
    if run.returncode != 0:
        raise RepoctlError("the database step failed: %s"
                           % ((run.stderr or "").strip().splitlines() or ["unknown error"])[-1][:300])
    return run.stdout or ""


def _restore_tables(dump: str, tables: list) -> dict:
    """Missing structured tables + their catalog rows from the backup's dump.
    Existing tables / rows are never touched."""
    tables = [t for t in tables if valid_table(t)]
    if not tables:
        return {"restored": [], "existing": []}
    names = ", ".join("'%s'" % t for t in tables)
    existing = set(_psql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' "
                         "AND table_name IN (%s);" % names).split())
    todo = [t for t in tables if t not in existing]
    if todo:
        args = []
        for t in todo:
            args += ["-t", t]
        with open(dump, "rb") as fh:
            run = subprocess.run(["docker", "exec", "-i", POSTGRES, "pg_restore", "-f", "-", "--no-owner",
                                  "--no-acl", *args], stdin=fh, capture_output=True, timeout=3600)
        if run.returncode != 0:
            raise RepoctlError("pg_restore of the tables failed: %s"
                               % (run.stderr.decode("utf-8", "replace").strip().splitlines() or ["?"])[-1][:300])
        _psql(run.stdout.decode("utf-8"), timeout=3600)
    with open(dump, "rb") as fh:
        run = subprocess.run(["docker", "exec", "-i", POSTGRES, "pg_restore", "-f", "-", "--data-only",
                              "-t", "data_catalog"], stdin=fh, capture_output=True, timeout=3600)
    if run.returncode == 0:
        sql = catalog_sql(run.stdout.decode("utf-8"), tables)
        if sql:
            _psql(sql)
    return {"restored": todo, "existing": sorted(existing)}


def cmd_restore_docs(env: RepoEnv, request: dict) -> dict:
    snapshot, request_id = request["snapshot"], request.get("id", "")
    docs = request["docs"]
    rec = _one_snapshot(env, snapshot)
    collections = sorted({d["collection"] for d in docs})
    work = os.path.join(WORK_PARENT, "gmac-repoctl.%d" % os.getpid())
    tmps = {}
    try:
        os.makedirs(work, mode=0o700, exist_ok=True)
        write_status("restore-docs", "running", request_id, "reading the backup")
        wanted = ["qdrant/%s.snapshot" % c for c in collections]
        if "documents" in collections:
            wanted.append("postgres/openwebui.dump")     # spreadsheets' tables
        _restore_staging(env, snapshot, work, wanted)
        tmps = _load_collections(work, collections, str(os.getpid()))
        results = []
        for c in collections:
            names = [d["name"] for d in docs if d["collection"] == c]
            write_status("restore-docs", "running", request_id,
                         "restoring %d document(s) into %s" % (len(names), c))
            out = qdocs("copy", tmps[c], c, json.dumps(names))
            for name in names:
                row = {"name": name, "collection": c, "sections": out["copied"].get(name, 0),
                       "tables": [], "status": ""}
                if name in out["skipped"]:
                    row["status"] = "already in the knowledge base — left as it is"
                elif name in out["missing"]:
                    row["status"] = "not in this backup"
                else:
                    tables = out["tables"].get(name) or []
                    if tables:
                        write_status("restore-docs", "running", request_id,
                                     "restoring the data tables of %s" % name[:60])
                        t = _restore_tables(os.path.join(work, STAGING.lstrip("/"), "postgres",
                                                         "openwebui.dump"), tables)
                        row["tables"] = t["restored"] + t["existing"]
                    row["status"] = "restored"
                results.append(row)
        # verify: every restored document is searchable live again
        for c in collections:
            live = {d["name"]: d for d in qdocs("sources", c)}
            for row in results:
                if row["collection"] == c and row["status"] == "restored":
                    got = live.get(row["name"], {}).get("chunks", 0)
                    row["verified"] = got >= row["sections"] > 0
                    if not row["verified"]:
                        row["status"] = "restored, but only %d of %d sections are live" % (got, row["sections"])
        return {"snapshot": snapshot_row(rec), "documents": results,
                "restored": sum(1 for r in results if r.get("verified")),
                "drive_source": env.source, "drive_note": env.note}
    finally:
        _drop_collections(tmps)
        shutil.rmtree(work, ignore_errors=True)


# ── deleted chats and videos (find-deleted-chats / restore-chats / ...) ────
# What the backup holds and the live database lost: chats WITH the account
# they belonged to, video catalog entries. "find" is strictly read-only;
# "restore" inserts only rows that are missing — a row that exists live is
# never overwritten (restore_rows_sql, ON CONFLICT DO NOTHING).

def _dump_tables(dump: str, tables: list) -> dict:
    """{table: (columns, rows)} from the backup's pg dump — pg_restore prints
    to stdout; the live database is not touched."""
    out = {}
    for table in tables:
        with open(dump, "rb") as fh:
            run = subprocess.run(["docker", "exec", "-i", POSTGRES, "pg_restore", "-f", "-",
                                  "--data-only", "-t", table], stdin=fh, capture_output=True,
                                 timeout=3600)
        if run.returncode != 0:
            raise RepoctlError("could not read %s from the backup's database dump: %s"
                               % (table, (run.stderr or b"").decode("utf-8", "replace")
                                  .strip().splitlines()[-1:][0][:200] if run.stderr else "unknown"))
        out[table] = copy_block(run.stdout.decode("utf-8", "replace"), table)
    return out


def _live_keys(sql: str) -> list:
    return [ln.strip() for ln in _psql(sql).splitlines() if ln.strip()]


def cmd_find_deleted_chats(env, request: dict) -> dict:
    """Chats in the backup the live database no longer has — each with the
    account it belonged to. READ-ONLY: nothing is written anywhere."""
    snapshot, request_id = request["snapshot"], request.get("id", "")
    rec = _one_snapshot(env, snapshot)
    work = os.path.join(WORK_PARENT, "gmac-repoctl.%d" % os.getpid())
    try:
        os.makedirs(work, mode=0o700, exist_ok=True)
        write_status("find-deleted-chats", "running", request_id, "reading the backup")
        _restore_staging(env, snapshot, work, ["postgres/openwebui.dump"])
        dump = os.path.join(work, STAGING.lstrip("/"), "postgres", "openwebui.dump")
        tables = _dump_tables(dump, ["chat", "chat_message", "user"])
        chat_cols, chat_rows = tables["chat"]
        write_status("find-deleted-chats", "running", request_id, "comparing with the live database")
        live = _live_keys("SELECT id FROM chat;")
        gone = deleted_rows(chat_rows, live, chat_cols, "id")
        users = {}
        ucols, urows = tables["user"]
        for r in urows:
            users[row_field(r, ucols, "id")] = {"name": row_field(r, ucols, "name"),
                                                "email": row_field(r, ucols, "email")}
        messages = {}
        mcols, mrows = tables["chat_message"]
        for r in mrows:
            cid = row_field(r, mcols, "chat_id")
            messages[cid] = messages.get(cid, 0) + 1
        deleted = [chat_summary(r, chat_cols, users, messages) for r in gone]
        deleted.sort(key=lambda d: d["when"], reverse=True)
        return {"snapshot": snapshot_row(rec), "deleted": deleted,
                "totals": {"backup_chats": len(chat_rows), "live_chats": len(live),
                           "deleted": len(deleted)},
                "drive_source": env.source, "drive_note": env.note}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_restore_chats(env, request: dict) -> dict:
    """Bring back the selected chats: the chat itself, its messages, its file
    links and its account — each row ONLY when it is missing live. A chat
    that exists now is never overwritten; nothing else changes."""
    snapshot, request_id = request["snapshot"], request.get("id", "")
    chats = [str(c) for c in request.get("chats") or []]
    rec = _one_snapshot(env, snapshot)
    work = os.path.join(WORK_PARENT, "gmac-repoctl.%d" % os.getpid())
    try:
        os.makedirs(work, mode=0o700, exist_ok=True)
        write_status("restore-chats", "running", request_id, "reading the backup")
        _restore_staging(env, snapshot, work, ["postgres/openwebui.dump"])
        dump = os.path.join(work, STAGING.lstrip("/"), "postgres", "openwebui.dump")
        tables = _dump_tables(dump, ["user", "chat", "chat_message", "chat_file"])
        sql = chat_restore_sql(tables, chats)
        if not sql.strip():
            raise RepoctlError("none of the selected chats are in this backup")
        write_status("restore-chats", "running", request_id,
                     "restoring %d chat(s)" % len(chats))
        _psql(sql, timeout=3600)
        live = set(_live_keys("SELECT id FROM chat;"))
        restored = [c for c in chats if c in live]
        missing = [c for c in chats if c not in live]
        return {"snapshot": snapshot_row(rec), "requested": len(chats),
                "restored": restored, "not_in_backup": missing, "verified": not missing,
                "drive_source": env.source, "drive_note": env.note}
    finally:
        shutil.rmtree(work, ignore_errors=True)


VIDEO_NOTE = ("processed video data (frames, transcripts) is vector data and comes back "
              "only with a full restore")


def cmd_find_deleted_videos(env, request: dict) -> dict:
    """Video catalog entries in the backup that the live catalog lost."""
    snapshot, request_id = request["snapshot"], request.get("id", "")
    rec = _one_snapshot(env, snapshot)
    work = os.path.join(WORK_PARENT, "gmac-repoctl.%d" % os.getpid())
    try:
        os.makedirs(work, mode=0o700, exist_ok=True)
        write_status("find-deleted-videos", "running", request_id, "reading the backup")
        _restore_staging(env, snapshot, work, ["postgres/openwebui.dump"])
        dump = os.path.join(work, STAGING.lstrip("/"), "postgres", "openwebui.dump")
        cols, rows = _dump_tables(dump, ["video_catalog"])["video_catalog"]
        live = _live_keys("SELECT video_id FROM video_catalog;")
        gone = deleted_rows(rows, live, cols, "video_id")
        deleted = [{"id": row_field(r, cols, "video_id"), "title": row_field(r, cols, "title"),
                    "when": row_field(r, cols, "upload_date") or ""} for r in gone]
        deleted.sort(key=lambda d: d["title"])
        return {"snapshot": snapshot_row(rec), "deleted": deleted, "note": VIDEO_NOTE,
                "totals": {"backup": len(rows), "live": len(live), "deleted": len(deleted)},
                "drive_source": env.source, "drive_note": env.note}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_restore_videos(env, request: dict) -> dict:
    """The selected video catalog entries — only rows missing live."""
    snapshot, request_id = request["snapshot"], request.get("id", "")
    wanted = {str(v) for v in request.get("videos") or []}
    rec = _one_snapshot(env, snapshot)
    work = os.path.join(WORK_PARENT, "gmac-repoctl.%d" % os.getpid())
    try:
        os.makedirs(work, mode=0o700, exist_ok=True)
        write_status("restore-videos", "running", request_id, "reading the backup")
        _restore_staging(env, snapshot, work, ["postgres/openwebui.dump"])
        dump = os.path.join(work, STAGING.lstrip("/"), "postgres", "openwebui.dump")
        cols, rows = _dump_tables(dump, ["video_catalog"])["video_catalog"]
        picked = [r for r in rows if row_field(r, cols, "video_id") in wanted]
        if not picked:
            raise RepoctlError("none of the selected videos are in this backup")
        _psql(video_restore_sql(cols, picked), timeout=3600)
        live = set(_live_keys("SELECT video_id FROM video_catalog;"))
        restored = [v for v in wanted if v in live]
        return {"snapshot": snapshot_row(rec), "restored": sorted(restored),
                "verified": len(restored) == len(wanted), "note": VIDEO_NOTE,
                "drive_source": env.source, "drive_note": env.note}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ── full restore (restore-full) ───────────────────────────────────────────

class RestoreFailed(RepoctlError):
    """A full restore stopped part-way. `result` carries the steps done and
    the safety snapshot id, so the panel can show the way back."""

    def __init__(self, message: str, result: dict):
        RepoctlError.__init__(self, message)
        self.result = result


def full_restore_plan(manifest: dict, restored_root: str) -> dict:
    """What a full restore will replace, from the backup restored under
    `restored_root`. Raises RepoctlError when the backup cannot rebuild
    everything — refused BEFORE any container is stopped."""
    stage = os.path.join(restored_root, STAGING.lstrip("/"))
    db = manifest.get("database") or {}
    arts = {a.get("file"): a for a in db.get("artifacts") or [] if isinstance(a, dict)}
    dump_rel = "postgres/%s.dump" % (db.get("name") or PG_DB)
    dump = os.path.join(stage, dump_rel)
    if dump_rel not in arts or not os.path.isfile(dump):
        raise RepoctlError("this backup has no database dump — it cannot be fully restored")
    points = {c.get("name"): c.get("points_count")
              for c in (manifest.get("qdrant") or {}).get("collections") or [] if isinstance(c, dict)}
    collections = []
    for c in FULL_COLLECTIONS:
        rel = "qdrant/%s.snapshot" % c
        path = os.path.join(stage, rel)
        sha = (arts.get(rel) or {}).get("sha256") or ""
        if not os.path.isfile(path) or not sha:
            # A partial vector store would leave the knowledge base pointing at
            # documents whose sections are gone; refuse the whole restore.
            raise RepoctlError("this backup has no %s snapshot — it cannot be fully restored" % c)
        collections.append({"name": c, "snapshot": path, "sha256": sha, "points": points.get(c)})
    files, ignored = [], []
    listed = {e.get("path"): e for e in manifest.get("source_files") or [] if isinstance(e, dict)}
    for path in listed:
        if path not in FULL_RESTORE_FILES:
            ignored.append(path)
    for path in FULL_RESTORE_FILES:
        entry = listed.get(path) or {}
        restored = os.path.join(restored_root, path.lstrip("/"))
        if not os.path.isdir(restored):
            if entry.get("files"):
                raise RepoctlError("%s is listed in the backup but did not restore" % path)
            # Empty (or absent) at backup time: restoring that point in time
            # means an empty tree, not "keep whatever is live now".
            os.makedirs(restored, exist_ok=True)
        files.append({"path": path, "restored": restored, "files": entry.get("files")})
    return {"database": dump, "collections": collections, "files": files,
            "ignored": sorted(set(ignored) | set(IGNORED_SOURCES)),
            "config": [c for c in manifest.get("config") or [] if isinstance(c, dict)]}


def verify_full(manifest: dict, live_counts: dict, table_counts: dict) -> list:
    """Mismatches after a full restore (empty list = verified): every
    collection's live point count must equal the backup's MANIFEST.json, and
    the irreplaceable tables (accounts, chats, pasted filters) must hold rows."""
    out = []
    want = {c.get("name"): c.get("points_count")
            for c in (manifest.get("qdrant") or {}).get("collections") or [] if isinstance(c, dict)}
    for name in FULL_COLLECTIONS:
        if name not in want:
            continue                       # the backup never recorded it: nothing to compare
        got = live_counts.get(name)
        if got != want[name]:
            out.append("collection %s: %s points live, the backup recorded %s" % (name, got, want[name]))
    for t in REQUIRED_TABLES:
        rows = table_counts.get(t)
        if rows is None:
            out.append("table %s: missing after the restore" % t)
        elif rows <= 0:
            out.append("table %s: empty after the restore" % t)
    return out


def _file_sha(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Ops:
    """Every side effect of cmd_restore_full — docker, restic, the backup
    script, the file swap. The self-test swaps in _FakeOps, so the order and
    the failure handling are tested without touching a real container."""

    def lock_free(self) -> bool:
        """The nightly backup's lock, probed and released at once (the drill's
        `flock -n`). The safety backup must take that very lock itself."""
        if fcntl is None:
            raise RepoctlError("file locks are unavailable on this machine")
        os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
        fd = os.open(LOCK_FILE, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        except OSError:
            return False
        finally:
            os.close(fd)

    def hold_lock(self):
        return JobLock()

    def snapshot(self, env, snapshot: str) -> dict:
        return _one_snapshot(env, snapshot)

    def safety_backup(self, tag: str):
        if not os.path.exists(BACKUP_SCRIPT):
            raise RepoctlError("%s is missing" % BACKUP_SCRIPT)
        try:
            return _streamed(["bash", BACKUP_SCRIPT, "--target", "daily", "--tag", tag], None,
                             FULL_SAFETY_TIMEOUT_S, backup_stage, "the backup script")
        except RepoctlError as exc:
            return 1, "", "ERROR: the backup script did not finish (%s)" % exc

    def catalog_safety(self, safety_id: str, request_id: str) -> None:
        catalog_safety_record(safety_id, request_id)

    def restic_restore(self, env, snapshot: str, work: str, includes: list) -> None:
        args = []
        for pattern in includes:
            args += ["--include", pattern]
        rc, _, err = restic(env.env, "restore", snapshot, "--target", work, "--verify", *args,
                            timeout=FULL_READ_TIMEOUT_S, progress=True)
        if rc != 0:
            raise RepoctlError("the backup could not be read: %s" % _restic_why(err, "restic restore failed"))

    def sha_check(self, stage: str) -> bool:
        run = subprocess.run(["sha256sum", "-c", "--quiet", "MANIFEST.sha256"], cwd=stage,
                             capture_output=True, text=True, timeout=FULL_SHA_TIMEOUT_S)
        return run.returncode == 0

    def running(self, container: str) -> bool:
        run = _docker("inspect", "-f", "{{.State.Running}}", container, timeout=60)
        return run.returncode == 0 and (run.stdout or "").strip() == "true"

    def stop(self, container: str) -> None:
        if _docker("stop", "-t", "60", container, timeout=180).returncode != 0:
            raise RepoctlError("could not stop %s" % container)

    def start(self, container: str) -> None:
        if _docker("start", container, timeout=180).returncode != 0:
            raise RepoctlError("could not start %s" % container)

    def pg_restore(self, dump: str, database: str = "", snapshot_id: str = "") -> int:
        """The dump replaces the database, all of it or none of it. Returns how
        many orphaned rows the backup held and the restore left out.

        pg_restore's SQL is streamed into ONE psql transaction (ON_ERROR_STOP,
        so any failure rolls back and the live database stays exactly as it
        was). Just before the foreign keys are re-added, rows that point at
        nothing are deleted: on 2026-10-28 every backup held 4 chat_file rows
        whose chat no longer existed, so re-adding chat_file_chat_id_fkey
        failed and NO full restore of ANY backup could succeed. A link from a
        file to a chat that does not exist is meaningless; refusing the whole
        restore over it is not. --clean --if-exists / --no-owner / --no-acl
        as before."""
        db = database or PG_DB
        fks = foreign_keys(self._pg_sql(dump, ["--schema-only", "--section=post-data"]))
        cleanup = orphan_cleanup_sql(fks)
        keep = keep_runs_sql(self._live_runs(db))
        psql = subprocess.Popen(["docker", "exec", "-i", POSTGRES, "psql", "-U", PG_USER, "-d", db,
                                 "-q", "-X", "-v", "ON_ERROR_STOP=1", "--single-transaction",
                                 "-v", "VERBOSITY=terse"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = []
        reader = threading.Thread(target=lambda: out.append(psql.stdout.read()), daemon=True)
        errs = []
        drainer = threading.Thread(target=lambda: errs.append(psql.stderr.read()), daemon=True)
        reader.start(); drainer.start()
        with open(dump, "rb") as fh:
            src = subprocess.Popen(["docker", "exec", "-i", POSTGRES, "pg_restore", "-f", "-", "--clean",
                                    "--if-exists", "--no-owner", "--no-acl"],
                                   stdin=fh, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            injected = False
            try:
                for line in src.stdout:
                    if not injected and FK_MARKER in line:
                        psql.stdin.write(cleanup.encode())
                        injected = True
                    psql.stdin.write(line)
                if not injected:
                    psql.stdin.write(cleanup.encode())
                psql.stdin.write(stale_runs_sql(snapshot_id).encode())
                psql.stdin.write(keep.encode())
                psql.stdin.close()
            except BrokenPipeError:
                pass                                   # psql stopped on an error; reported below
            src_err = src.stderr.read()
            src_rc = src.wait()
        try:
            rc = psql.wait(timeout=4 * 3600)
        except subprocess.TimeoutExpired:
            psql.kill()
            raise RepoctlError("the database restore did not finish (TimeoutExpired)")
        reader.join(timeout=30); drainer.join(timeout=30)
        err = b"".join(errs).decode("utf-8", "replace")
        if src_rc != 0 or rc != 0:
            lines = [l for l in (err + "\n" + src_err.decode("utf-8", "replace")).splitlines()
                     if "ERROR" in l or "error" in l]
            raise RepoctlError("pg_restore failed (the database was rolled back): %s"
                               % ((lines or err.strip().splitlines() or ["?"])[0][:300]))
        return orphans_removed(b"".join(out).decode("utf-8", "replace"))

    def _live_runs(self, db: str) -> str:
        """The live backup-history rows as COPY text; "" when the table is
        absent or unreadable (never a reason to refuse the restore)."""
        run = subprocess.run(["docker", "exec", POSTGRES, "psql", "-X", "-U", PG_USER, "-d", db, "-c",
                              "COPY (SELECT %s FROM public.ops_backup_reports) TO STDOUT" % RUN_COLUMNS],
                             capture_output=True, timeout=120)
        return run.stdout.decode("utf-8", "replace") if run.returncode == 0 else ""

    def _pg_sql(self, dump: str, args: list) -> str:
        with open(dump, "rb") as fh:
            run = subprocess.run(["docker", "exec", "-i", POSTGRES, "pg_restore", "-f", "-"] + args,
                                 stdin=fh, capture_output=True, timeout=3600)
        if run.returncode != 0:
            raise RepoctlError("the backup's database could not be read")
        return run.stdout.decode("utf-8", "replace")

    def _helper(self, *args, timeout=3600):
        with open(QDRANT_HELPER, encoding="utf-8") as fh:
            script = fh.read()
        return _docker("exec", "-i", "-e", "QDRANT_URL=" + QDRANT_URL, HAYSTACK, "python3", "-", *args,
                       stdin=script, timeout=timeout)

    def refresh_planner(self):
        """POST /video/planner/refresh inside the backend container: the number
        of searchable videos, or None (never a reason to fail the restore). The
        backend may still be starting, so a few tries."""
        code = ("import json,urllib.request as u;"
                "r=u.urlopen(u.Request('http://localhost:8000/video/planner/refresh',method='POST'),timeout=60);"
                "print(json.load(r).get('videos'))")
        for _ in range(6):
            try:
                run = _docker("exec", HAYSTACK, "python3", "-c", code, timeout=90)
            except RepoctlError:
                time.sleep(5)
                continue
            out = (run.stdout or "").strip()
            if run.returncode == 0 and out.isdigit():
                return int(out)
            time.sleep(5)
        sys.stderr.write("repoctl: planner refresh failed; it refreshes itself within 5 minutes\n")
        return None

    def qdrant_restore(self, collection: str, path: str, sha: str) -> None:
        # The helper runs inside gmac-haystack (Qdrant is not on the host), so
        # the snapshot file is handed into that container first.
        if _docker("exec", HAYSTACK, "mkdir", "-p", CONTAINER_TMP).returncode != 0:
            raise RepoctlError("could not prepare the vector store container")
        dest = "%s/%s.snapshot" % (CONTAINER_TMP, collection)
        try:
            if _docker("cp", path, "%s:%s" % (HAYSTACK, dest), timeout=3600).returncode != 0:
                raise RepoctlError("could not hand the %s snapshot to the vector store" % collection)
            run = self._helper("restore", collection, dest, sha)
            if run.returncode != 0:
                raise RepoctlError("Qdrant refused the %s snapshot (or its checksum): %s" % (
                    collection, ((run.stderr or "").strip().splitlines() or ["?"])[-1][:300]))
        finally:
            _docker("exec", HAYSTACK, "rm", "-f", dest)

    def qdrant_counts(self) -> dict:
        run = self._helper("list", timeout=600)
        if run.returncode != 0:
            return {}
        try:
            return {c.get("name"): c.get("points_count") for c in json.loads(run.stdout or "[]")}
        except ValueError:
            return {}

    def table_counts(self, tables) -> dict:
        out = {}
        for t in tables:
            try:
                got = _psql('SELECT count(*) FROM public."%s";' % t.replace('"', "")).strip()
                out[t] = int(got.splitlines()[-1]) if got else None
            except (RepoctlError, ValueError):
                out[t] = None
        return out

    def swap_files(self, restored: str, live: str, keep: str) -> None:
        """Move the live tree's CONTENTS aside into `keep`, then the restored
        contents in. The directory itself stays: it is bind-mounted into
        gmac-haystack (compose.yml), and a renamed directory would leave the
        container holding the old one. A failure moves everything back."""
        if os.path.lexists(keep):
            raise RepoctlError("%s already exists — refusing to overwrite an earlier pre-restore copy" % keep)
        os.makedirs(live, exist_ok=True)
        os.makedirs(keep, mode=0o700)
        moved_out, moved_in, in_flight = [], [], None
        try:
            for name in sorted(os.listdir(live)):
                os.rename(os.path.join(live, name), os.path.join(keep, name))    # same filesystem
                moved_out.append(name)
            for name in sorted(os.listdir(restored)):
                # the restore staged under /var/tmp: maybe another filesystem
                # (shutil.move then copies, and a failed copy leaves a
                # half-made live/<name> that is in neither list below)
                in_flight = name
                shutil.move(os.path.join(restored, name), os.path.join(live, name))
                moved_in.append(name)
                in_flight = None
        except OSError as exc:
            for name in moved_in + ([in_flight] if in_flight else []):
                _remove_entry(os.path.join(live, name))
            # One rename back that fails must not abandon the rest: stopping
            # at the first error left every later old file stranded in `keep`
            # with an error that claimed they "were put back".
            stuck = []
            for name in moved_out:
                try:
                    os.rename(os.path.join(keep, name), os.path.join(live, name))
                except OSError:
                    stuck.append(name)
            if stuck:
                raise RepoctlError("the knowledge-base files could not be swapped (%s), and %d old "
                                   "file(s) could not be moved back from %s: %s — move them into %s by "
                                   "hand" % (type(exc).__name__, len(stuck), keep,
                                             ", ".join(stuck[:20]), live))
            raise RepoctlError("the knowledge-base files could not be swapped (%s) — the live files "
                               "were put back (the emptied copy is %s)" % (type(exc).__name__, keep))

    def config_hash(self, path: str) -> str:
        """Same digest the backup's MANIFEST.json records: a file's sha256, or
        for a directory the sha256 of `find | sort | xargs sha256sum`."""
        import hashlib
        try:
            if os.path.isdir(path):
                files = sorted(os.path.join(b, n) for b, _d, ns in os.walk(path) for n in ns)
                text = "".join("%s  %s\n" % (_file_sha(f), f) for f in files)
                return hashlib.sha256(text.encode()).hexdigest()
            return _file_sha(path)
        except OSError:
            return ""


def _remove_entry(path: str) -> None:
    """Remove a file, directory or symlink without following the link:
    rmtree on a symlink to a directory would delete the files it points at
    (or refuse), so a link is unlinked and only a real directory recursed."""
    if not os.path.lexists(path):
        return
    if os.path.islink(path):
        try:
            os.unlink(path)
        except OSError:                   # a directory link on Windows test machines
            os.rmdir(path)
    elif not os.path.isdir(path):
        _quiet_remove(path)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _quiet_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _safety_id(out: str) -> str:
    """The snapshot id from the backup script's own last line
    ("backup complete: snapshot <id>"); '' when it did not complete."""
    found = ""
    for line in (out or "").splitlines():
        if "backup complete: snapshot " in line:
            found = line.split("snapshot ", 1)[1].strip()
    return found if valid_snapshot_id(found) else ""


def _take_safety(ops, tag, state, result, request_id, done):
    """The full restore's safety backup: the way back from it."""
    rc, out, err = ops.safety_backup(tag)
    safety = _safety_id(out) if rc == 0 else ""
    if not safety:
        lines = [l for l in ((out or "") + "\n" + (err or "")).splitlines()
                 if ("ERROR" in l or "Fatal" in l) and l.strip()]
        raise RestoreFailed("the safety backup failed, so the restore did not start — nothing was "
                            "replaced: %s" % _why(lines[-1] if lines else
                                                  "the backup failed (exit %d)" % rc), result)
    state["safety"] = result["safety_snapshot"] = safety
    done("snapshot %s (daily folder, kept)" % safety)
    # The catalog says what this snapshot is for. Best effort: the backup
    # script recorded it (with its manifest) already; a catalog problem
    # never stops the restore.
    try:
        ops.catalog_safety(safety, request_id)
    except Exception as exc:                    # noqa: BLE001
        sys.stderr.write("repoctl: catalog: safety record skipped (%s)\n" % type(exc).__name__)


def cmd_restore_full(env, request: dict, ops=None) -> dict:
    """Replace the database, the vectors and the kb files with a backup's.

    Order (each step is written to status.json as it starts):
      1. check     typed RESTORE, a concrete recovery point, the backup lock
                   free ("a backup is running" otherwise — nothing stopped);
                   the point is resolved to its full id NOW, before step 3
                   adds a newer snapshot.
      2. read      restic restore of the staging tree + kb files into a work
                   dir; sha256sum -c MANIFEST.sha256; plan (refuses a backup
                   that cannot rebuild everything, before anything stops).
      3. safety    gmac-open-webui stopped (no chat can be written that the
                   safety backup would miss), then a milestone backup of the
                   present (daily folder, tag pre-restore-<request id>, never
                   pruned). Fails -> abort, nothing replaced, open-webui
                   started again. Its id is the way back from everything below.
      4. stop      gmac-haystack (the other writer), under the backup lock.
      5. database  pg_restore --clean, in one transaction.
      6. files     /data/haystack-uploads swapped (live copy kept aside) —
                   while haystack is still stopped (plan step 7, moved up:
                   haystack bind-mounts that directory).
      7. vectors   haystack started (the Qdrant helper runs in it); each of the
                   six collections recovered from its snapshot, checksum
                   verified by Qdrant.
      8. verify    open-webui started; point counts == MANIFEST.json, required
                   tables non-empty; config files hash-compared ONLY.
    Every container this run stopped is started again whatever happens."""
    ops = ops or Ops()
    check_full_request(request)
    request_id = str(request.get("id", ""))
    tag = "pre-restore-" + (re.sub(r"[^A-Za-z0-9-]", "", request_id)[:40] or str(os.getpid()))
    steps, stopped = [], []
    state = {"step": "", "safety": "", "keep": ""}
    result = {"snapshot": {}, "safety_snapshot": "", "steps": steps, "config": [], "verified": False,
              "mismatches": [], "ignored": []}

    def begin(name: str) -> None:
        state["step"] = name
        write_status("restore-full", "running", request_id, name, result=result)

    def done(detail: str = "") -> None:
        steps.append({"step": state["step"], "ok": True, "detail": detail})

    # Only the check runs before the try: nothing is stopped yet.
    begin("checking the request")
    if not ops.lock_free():
        raise BusyError("a backup is running — start the full restore when it finishes "
                        "(nothing was changed)")
    rec = ops.snapshot(env, request["snapshot"])
    full_id = rec.get("id") or request["snapshot"]
    result["snapshot"] = snapshot_row(rec)
    done("recovery point %s" % result["snapshot"].get("id", "")[:12])

    work = os.path.join(WORK_PARENT, "gmac-repoctl.%d" % os.getpid())
    failure, restart_errors = None, []
    try:
        # Read and verify the backup FIRST, while everything still runs: a
        # corrupt or incomplete backup is refused with no downtime and no
        # safety backup spent. restic restore is read-only on the repository,
        # so it needs no lock.
        begin("reading the backup")
        os.makedirs(work, mode=0o700, exist_ok=True)
        includes = [glob_escape(STAGING)] + [glob_escape(p) for p in FULL_RESTORE_FILES]
        ops.restic_restore(env, full_id, work, includes)
        stage = os.path.join(work, STAGING.lstrip("/"))
        if not os.path.isfile(os.path.join(stage, "MANIFEST.json")):
            raise RepoctlError("the backup has no MANIFEST.json — it cannot be fully restored")
        if not ops.sha_check(stage):
            raise RepoctlError("the restored backup does not match its MANIFEST.sha256 (corrupt?)")
        with open(os.path.join(stage, "MANIFEST.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        plan = full_restore_plan(manifest, work)
        result["ignored"] = plan["ignored"]
        done("checksums verified")

        # Open WebUI stops BEFORE the safety backup: with it running, a chat
        # written while the backup ran (or while the backup was being read)
        # was in neither the safety backup nor the restored database — lost
        # without a trace. Haystack stays up: the safety backup snapshots
        # Qdrant through it (and dumps Postgres via gmac-postgres).
        begin("stopping Open WebUI")
        if ops.running(OPENWEBUI):
            ops.stop(OPENWEBUI)
            stopped.append(OPENWEBUI)
        done(", ".join(stopped) or "it was not running")

        # The safety backup is ops/backup.sh, which takes the backup
        # lock itself — so it runs before this process holds that lock.
        begin("taking a safety backup of the present")
        if request.get("safety_backup") is False:
            state["skipped"] = True
            result["safety_skipped"] = True
            done("skipped — turned off in the panel; there is no way back from this restore")
        else:
            _take_safety(ops, tag, state, result, request_id, done)

        with ops.hold_lock():
            begin("stopping the backend")
            if ops.running(HAYSTACK):
                ops.stop(HAYSTACK)
                stopped.append(HAYSTACK)
            done(", ".join(stopped) or "nothing was running")

            begin("restoring the database")
            orphans = ops.pg_restore(plan["database"], snapshot_id=full_id)
            result["orphans_removed"] = orphans
            done("%d orphaned link row(s) in the backup left out" % orphans if orphans else "")

            begin("restoring the knowledge-base files")
            for f in plan["files"]:
                # Recorded BEFORE the swap: a swap that fails half-way may
                # leave old files in `keep`, and the error must say where.
                state["keep"] = "%s.%s" % (f["path"], tag)
                ops.swap_files(f["restored"], f["path"], state["keep"])
            done("previous files kept in %s" % state["keep"] if state["keep"] else "")

            begin("restoring the vector store")
            if HAYSTACK in stopped or not ops.running(HAYSTACK):
                ops.start(HAYSTACK)
                if HAYSTACK in stopped:
                    stopped.remove(HAYSTACK)
            for c in plan["collections"]:
                ops.qdrant_restore(c["name"], c["snapshot"], c["sha256"])
            # The backend was started above, BEFORE these collections came back,
            # and listed the searchable videos from the old ones: for 5 minutes
            # a restored video was planned as if it did not exist (2026-10-28).
            # Best effort — the backend's own 5-minute refresh is the fallback.
            planner = ops.refresh_planner()
            done("%d collections%s" % (len(plan["collections"]),
                                       "; %s videos searchable" % planner if planner is not None else ""))

            begin("verifying")
            if OPENWEBUI in stopped:
                ops.start(OPENWEBUI)
                stopped.remove(OPENWEBUI)
            result["mismatches"] = verify_full(manifest, ops.qdrant_counts(),
                                               ops.table_counts(REQUIRED_TABLES))
            # Config is NEVER written by a restore: a restored .env could hold
            # a rotated-away secret. Reported so the operator can decide.
            result["config"] = [{"path": c.get("path"),
                                 "matches": ops.config_hash(c.get("path") or "") == c.get("sha256")}
                                for c in plan["config"]]
            result["verified"] = not result["mismatches"]
            steps.append({"step": state["step"], "ok": result["verified"],
                          "detail": "; ".join(result["mismatches"]) or "counts and tables match the backup"})
    except Exception as exc:                           # noqa: BLE001 - report every failure
        failure = exc
    finally:
        # A failed step must never leave the chat or the backend down — this
        # includes a failed safety backup, which now runs with Open WebUI
        # already stopped. (A killed/timed-out unit never reaches here; the
        # unit's ExecStopPost covers that.)
        for c in list(stopped):
            try:
                ops.start(c)
                stopped.remove(c)
            except Exception as exc:                   # noqa: BLE001
                restart_errors.append("%s (%s)" % (c, exc))
        shutil.rmtree(work, ignore_errors=True)
    if failure is not None and state.get("skipped") and not isinstance(failure, BusyError)             and state["step"] not in ("checking the request", "reading the backup", "stopping Open WebUI",
                                      "taking a safety backup of the present",
                                      # one transaction: a failure here rolls back
                                      "restoring the database"):
        steps.append({"step": state["step"], "ok": False, "detail": str(failure)[:300]})
        msg = ("the full restore failed at step '%s': %s. NO safety backup was taken (it was turned "
               "off), so part of the system may already be replaced — restore again from a backup."
               % (state["step"], str(failure)[:300] if isinstance(failure, RepoctlError)
                  else "unexpected %s" % type(failure).__name__))
        if restart_errors:
            msg += " COULD NOT START: %s — start them by hand." % ", ".join(restart_errors)
        raise RestoreFailed(msg, result)
    if failure is not None and not state["safety"]:
        # Failed before the safety backup existed: nothing was replaced, so
        # there is no "way back" to point at — only the reason.
        steps.append({"step": state["step"], "ok": False, "detail": str(failure)[:300]})
        if isinstance(failure, RestoreFailed):
            msg = str(failure)
        else:
            msg = "the full restore failed at step '%s': %s — nothing was replaced" % (
                state["step"], str(failure)[:300] if isinstance(failure, RepoctlError)
                else "unexpected %s" % type(failure).__name__)
        if restart_errors:
            msg += " COULD NOT START: %s — start them by hand." % ", ".join(restart_errors)
        raise RestoreFailed(msg, result)
    if failure is not None:
        steps.append({"step": state["step"], "ok": False, "detail": str(failure)[:300]})
        msg = "the full restore failed at step '%s': %s. The way back is the safety backup %s " \
              "(daily folder, taken just before)." % (state["step"], str(failure)[:300] if isinstance(
                  failure, RepoctlError) else "unexpected %s" % type(failure).__name__, state["safety"])
        if state["keep"]:
            msg += " The previous knowledge-base files are kept in %s." % state["keep"]
        if restart_errors:
            msg += " COULD NOT START: %s — start them by hand." % ", ".join(restart_errors)
        raise RestoreFailed(msg, result)
    if restart_errors:
        result["mismatches"].append("could not start: %s" % ", ".join(restart_errors))
        result["verified"] = False
    return result


# ── dispatch ──────────────────────────────────────────────────────────────

def clear_stale_locks() -> None:
    """`restic unlock` (no --remove-all) removes only STALE locks -- left by a
    restic process that no longer exists -- never a live one. A stale lock
    otherwise fails every later `restic check` (seen 2026-10-26: a killed
    command's lock failed Back Up Now's check). Every folder's repository is
    cleared, so a stale lock in any of the three is fixed."""
    def one(target):
        env = RepoEnv(target).setup()
        try:
            rc, _, err = restic(env.env, "unlock", timeout=120)
            if rc != 0:
                sys.stderr.write("repoctl: unlock %s: %s\n" % (target, (err or "").strip()[:200]))
        except RepoctlError as exc:
            sys.stderr.write("repoctl: %s\n" % exc)
        finally:
            env.close()
    # In parallel: ~6 s each, and this runs before EVERY request (2026-10-28).
    _parallel(one, list(TARGETS))


def _parallel(fn, items: list) -> list:
    """fn over items on threads, results in order. An exception in one is
    re-raised here, as a plain loop would."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(len(items), 1)) as pool:
        return list(pool.map(fn, items))


def run_request(request: dict) -> dict:
    cmd = request["cmd"]
    target = request.get("repo") or "daily"
    clear_stale_locks()
    if cmd == "backup-now":
        return cmd_backup_now(request, target)
    if cmd == "rebuild-catalog":
        # Every repository (all three folders), so no single request's target.
        return cmd_rebuild_catalog(request)
    env = RepoEnv(target).setup()
    try:
        if cmd == "snapshots":
            return cmd_snapshots(env)
        if cmd == "inspect":
            return cmd_inspect(env, request)
        if cmd == "find-deleted":
            return cmd_find_deleted(env, request)
        if cmd == "restore-docs":
            return cmd_restore_docs(env, request)
        if cmd == "find-deleted-chats":
            return cmd_find_deleted_chats(env, request)
        if cmd == "restore-chats":
            return cmd_restore_chats(env, request)
        if cmd == "find-deleted-videos":
            return cmd_find_deleted_videos(env, request)
        if cmd == "restore-videos":
            return cmd_restore_videos(env, request)
        if cmd == "restore-full":
            return cmd_restore_full(env, request)
        return cmd_restore_files(env, request)
    finally:
        env.close()


def main(argv) -> int:
    if "--self-test" in argv:
        return self_test()
    request_file = os.path.join(CTL_DIR, "request.json")
    if not os.path.exists(request_file):
        return 0                                   # nothing to do (spurious start)
    claimed = os.path.join(CTL_DIR, "request.%d.json" % os.getpid())
    try:
        os.rename(request_file, claimed)           # claim: exactly once
    except OSError:
        return 0                                   # a parallel run took it
    request_id = ""
    try:
        request = load_request(claimed)
        request_id = str(request.get("id", ""))
        CURRENT.update(by=str(request.get("by") or "")[:100],
                       requested_at=str(request.get("requested_at") or now_iso())[:40])
        write_status(request["cmd"], "running", request_id, "started")
        # backup-now and restore-full take the backup lock themselves (both run
        # ops/backup.sh, which locks it); holding it here would refuse them.
        if request["cmd"] not in ("backup-now", "restore-full"):
            with JobLock():
                result = run_request(request)
        else:
            result = run_request(request)
        write_status(request["cmd"], "done", request_id, "", result=result)
        sys.stderr.write("repoctl: %s %s done\n" % (request["cmd"], request_id))
        return 0
    except BusyError as exc:
        sys.stderr.write("repoctl: %s %s busy: %s\n" % (_peek_cmd(claimed), request_id, exc))
        write_status(_peek_cmd(claimed), "busy", request_id, "", error=str(exc))
        return 1
    except RepoctlError as exc:
        # The reason went only to status.json, which the next request
        # overwrites; the journal showed systemd's bare "exit-code" and a failed
        # restore could not be diagnosed afterwards (log audit, 2026-10-28).
        sys.stderr.write("repoctl: %s %s FAILED: %s\n" % (_peek_cmd(claimed), request_id, exc))
        write_status(_peek_cmd(claimed), "failed", request_id, "", error=str(exc),
                     result=getattr(exc, "result", None))
        return 1
    except Exception as exc:                       # noqa: BLE001 - never lose the report
        import traceback
        traceback.print_exc()
        write_status(_peek_cmd(claimed), "failed", request_id, "",
                     error="unexpected %s" % type(exc).__name__)
        return 1
    finally:
        try:
            os.remove(claimed)
        except OSError:
            pass


def _peek_cmd(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return str((json.load(fh) or {}).get("cmd") or "unknown")
    except (OSError, ValueError):
        return "unknown"


def cmd_rebuild_catalog(request: dict) -> dict:
    """Rebuild catalog.json from the repositories themselves: every snapshot
    restic still lists, described by its MANIFEST.json. restic is the truth,
    the catalog is only its description — so a missing or corrupt catalog is
    never an error here, and snapshots gone from a repository are marked
    pruned (one folder's failed listing never touches the other folders)."""
    catalog = backup_catalog.read_catalog_file(catalog_path())
    listed, found, problems, envs = [], 0, [], []
    write_status("rebuild-catalog", "running", request.get("id", ""), "reading the backup folders")
    try:
        for target in TARGETS:
            envs.append(RepoEnv(target).setup())    # raises when Drive is signed out
        # The three folders are listed at once: each listing is ~12 s of
        # opening a repository over Drive, and one after another was 39 s.
        listings = _parallel(lambda e: restic(e.env, "snapshots", "--json", timeout=300), envs)
        for target, env, (rc, out, err) in zip(TARGETS, envs, listings):
            if rc != 0:
                problems.append("%s folder: %s" % (target, _restic_why(err, "could not be listed")))
                continue
            snaps = [s for s in (json.loads(out or "[]") if out.strip().startswith("[") else [])
                     if isinstance(s, dict)]
            live = []
            for i, snap in enumerate(snaps):
                snap_id = str(snap.get("id") or snap.get("short_id") or "")
                if not snap_id:
                    continue
                live.append(str(snap.get("short_id") or snap_id))
                tick((TARGETS.index(target) + i / max(len(snaps), 1)) / len(TARGETS))
                # A snapshot never changes once taken, so a record already
                # described from its manifest is kept as it is. Re-dumping every
                # manifest cost ~25 s per snapshot over Drive (3.5 min for 7,
                # 2026-10-28) and threw away corrections made to a record.
                if known_record(catalog, snap_id):
                    found += 1
                    continue
                manifest = {}
                mrc, mout, _ = restic(env.env, "dump",
                                      str(snap.get("short_id") or snap_id), MANIFEST_PATH,
                                      timeout=600)
                if mrc == 0:
                    try:
                        manifest = json.loads(mout or "{}")
                    except ValueError:
                        manifest = {}               # corrupt manifest: record what restic knows
                catalog = backup_catalog.merge_record(catalog, backup_catalog.record_from_manifest(
                    manifest, {"id": snap_id, "short_id": snap.get("short_id") or snap_id[:8],
                               "folder": target, "tags": snap.get("tags") or [],
                               "taken_at": snap.get("time") or "",
                               "size_added_bytes": snap.get("size")
                               or (snap.get("summary") or {}).get("total_bytes_processed") or 0}))
                found += 1
            if live:
                catalog = backup_catalog.prune_records(catalog, live, folder=target)
            listed.append(target)
        backup_catalog.write_catalog_file(catalog_path(), catalog)
        if listed:
            # While envs[0]'s rclone config (the panel's Drive sign-in) still
            # exists: the upload is part of the rebuild, not a later step.
            _catalog_upload(envs[0].env)
    finally:
        for env in envs:
            env.close()
    return {"catalog": catalog_path(), "folders": listed, "snapshots": found,
            "count": len(catalog.get("snapshots") or []), "problems": problems}


# ── self-test (offline: no restic, no docker, no network) ─────────────────

class _FakeOps:
    """Records every side effect cmd_restore_full asks for; touches nothing
    real. `fail` names the one operation that should fail."""

    def __init__(self, fail="", lock_free=True, old_snapshot=False):
        self.calls, self.fail, self.free, self.old = [], fail, lock_free, old_snapshot
        self.up = {FULL_RESTORE_STOP[0], FULL_RESTORE_STOP[1]}

    def _rec(self, *call):
        self.calls.append(call)
        if self.fail == call[0]:
            raise RepoctlError("%s exploded" % call[0])

    def lock_free(self):
        self.calls.append(("lock_free",))
        return self.free

    def hold_lock(self):
        import contextlib
        self.calls.append(("hold_lock",))
        return contextlib.nullcontext()

    def snapshot(self, env, snapshot):
        self.calls.append(("snapshot", snapshot))
        return {"id": "1a2b3c4d" * 8, "short_id": "1a2b3c4d", "time": "2026-10-20T03:03:00Z"}

    def safety_backup(self, tag):
        self.calls.append(("safety_backup", tag))
        if self.fail == "safety_backup":
            return 1, "2026-10-27T10:00:00Z ERROR: drive: Google Drive is full (0 B free)\n", ""
        return 0, "2026-10-27T10:00:00Z backup complete: snapshot 9f8e7d6c\n", ""

    def catalog_safety(self, safety_id, request_id):
        self.calls.append(("catalog_safety", safety_id))

    def restic_restore(self, env, snapshot, work, includes):
        self._rec("restic_restore", snapshot, tuple(includes))
        stage = os.path.join(work, STAGING.lstrip("/"))
        os.makedirs(os.path.join(stage, "postgres"))
        os.makedirs(os.path.join(stage, "qdrant"))
        open(os.path.join(stage, "postgres", "openwebui.dump"), "wb").write(b"PGDMP")
        arts = [{"file": "postgres/openwebui.dump", "size": 5, "sha256": "p" * 64}]
        colls = []
        for i, c in enumerate(FULL_COLLECTIONS):
            open(os.path.join(stage, "qdrant", c + ".snapshot"), "wb").write(b"x")
            arts.append({"file": "qdrant/%s.snapshot" % c, "size": 1, "sha256": "%064d" % i})
            colls.append({"name": c, "points_count": 10 + i})
        os.makedirs(os.path.join(work, "data", "haystack-uploads"))
        open(os.path.join(work, "data", "haystack-uploads", "a.pdf"), "wb").write(b"pdf")
        sources = [{"path": "/data/haystack-uploads", "bytes": 3, "files": 1}]
        if self.old:          # a pre-2026-10-27 snapshot still carries chat uploads
            os.makedirs(os.path.join(work, "data", "openwebui", "uploads"))
            open(os.path.join(work, "data", "openwebui", "uploads", "chat.png"), "wb").write(b"img")
            sources.insert(0, {"path": "/data/openwebui/uploads", "bytes": 3, "files": 1})
        manifest = {"database": {"name": "openwebui", "artifacts": arts},
                    "qdrant": {"collections": colls}, "source_files": sources,
                    "config": [{"path": "/opt/gmac/compose.yml", "sha256": "c" * 64},
                               {"path": "/opt/gmac/.env", "sha256": "e" * 64}]}
        with open(os.path.join(stage, "MANIFEST.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        open(os.path.join(stage, "MANIFEST.sha256"), "w").write("")

    def sha_check(self, stage):
        self.calls.append(("sha_check", stage))
        return self.fail != "sha_check"

    def running(self, container):
        return container in self.up

    def stop(self, container):
        self._rec("stop", container)
        self.up.discard(container)

    def start(self, container):
        self.calls.append(("start", container))
        self.up.add(container)

    def pg_restore(self, dump, snapshot_id=""):
        self._rec("pg_restore", os.path.basename(dump))
        return 0

    def refresh_planner(self):
        self.calls.append(("refresh_planner",))
        return 5

    def qdrant_restore(self, collection, path, sha):
        self._rec("qdrant_restore", collection, sha)

    def swap_files(self, restored, live, keep):
        self._rec("swap_files", live, keep)

    def qdrant_counts(self):
        self.calls.append(("qdrant_counts",))
        return {c: 10 + i for i, c in enumerate(FULL_COLLECTIONS)}

    def table_counts(self, tables):
        self.calls.append(("table_counts",))
        return {t: 3 for t in tables}

    def config_hash(self, path):
        return "c" * 64 if path.endswith("compose.yml") else "z" * 64


def _self_test_swap_rollback(check, tempfile) -> None:
    """The REAL Ops.swap_files on temp dirs, with renames made to fail."""
    real_rename, real_move = os.rename, shutil.move

    def tree(live, restored, keep_parent):
        for d in (live, restored):
            os.makedirs(d)
        for n in ("a", "b", "c"):
            open(os.path.join(live, n), "w").write("old " + n)
        open(os.path.join(restored, "x"), "w").write("new")
        return os.path.join(keep_parent, "live.keep")

    # one entry cannot be renamed back: the rest still are, the stuck one is named
    with tempfile.TemporaryDirectory() as tmp:
        live, restored = os.path.join(tmp, "live"), os.path.join(tmp, "restored")
        keep = tree(live, restored, tmp)

        def bad_move(src, dst):
            raise OSError("disk gone")

        def bad_rename(src, dst):
            if os.path.dirname(dst) == live and os.path.basename(dst) == "b":
                raise OSError("busy")
            return real_rename(src, dst)
        shutil.move, os.rename = bad_move, bad_rename
        try:
            Ops().swap_files(restored, live, keep)
        except RepoctlError as exc:
            msg = str(exc)
            check(sorted(os.listdir(live)) == ["a", "c"] and os.listdir(keep) == ["b"]
                  and keep in msg and "b" in msg,
                  "swap rollback: one failed rename back does not stop the others; it names it and the keep dir")
        else:
            raise AssertionError("a failed swap did not raise")
        finally:
            shutil.move, os.rename = real_move, real_rename

    # the entry in flight left something half-made at live/<name>: removed
    with tempfile.TemporaryDirectory() as tmp:
        live, restored = os.path.join(tmp, "live"), os.path.join(tmp, "restored")
        keep = tree(live, restored, tmp)
        target = os.path.join(tmp, "outside")
        os.makedirs(target)
        open(os.path.join(target, "precious"), "w").write("keep me")

        def half_move(src, dst):
            try:
                os.symlink(target, dst)
            except (OSError, NotImplementedError):
                os.makedirs(os.path.join(dst, "partial"))
            raise OSError("copy failed half-way")
        shutil.move = half_move
        try:
            Ops().swap_files(restored, live, keep)
        except RepoctlError:
            check(sorted(os.listdir(live)) == ["a", "b", "c"]
                  and os.path.exists(os.path.join(target, "precious")),
                  "swap rollback: a half-moved entry (symlink or dir) is removed, a symlink's target untouched")
        else:
            raise AssertionError("a failed swap did not raise")
        finally:
            shutil.move = real_move


def _self_test_restore_full(check, tempfile) -> None:
    global CTL_DIR, WORK_PARENT
    saved = (CTL_DIR, WORK_PARENT)
    _self_test_swap_rollback(check, tempfile)

    # pure planning: an OLD snapshot (still carrying /data/openwebui/uploads,
    # backed up until 2026-10-27) must never write chat uploads back
    with tempfile.TemporaryDirectory() as tmp:
        fake = _FakeOps(old_snapshot=True)
        fake.restic_restore(None, "x", tmp, [])
        manifest = json.load(open(os.path.join(tmp, STAGING.lstrip("/"), "MANIFEST.json")))
        plan = full_restore_plan(manifest, tmp)
        check([f["path"] for f in plan["files"]] == ["/data/haystack-uploads"]
              and "/data/openwebui/uploads" in plan["ignored"],
              "full restore: an old snapshot's chat uploads are ignored, not restored")
        check(sorted(c["name"] for c in plan["collections"]) == sorted(FULL_COLLECTIONS)
              and plan["collections"][0]["sha256"] and plan["database"].endswith("openwebui.dump"),
              "full restore plan: the dump and all six collections with their checksums")
        os.remove(os.path.join(tmp, STAGING.lstrip("/"), "qdrant", FULL_COLLECTIONS[0] + ".snapshot"))
        try:
            full_restore_plan(manifest, tmp)
        except RepoctlError:
            check(True, "full restore plan: a collection snapshot missing from the backup is refused")
        else:
            raise AssertionError("an incomplete backup was planned")

        live = {c["name"]: c["points_count"] for c in manifest["qdrant"]["collections"]}
        check(verify_full(manifest, live, {"user": 2, "chat": 5, "function": 3}) == [],
              "verify: counts equal to the manifest and tables non-empty -> no mismatch")
        live[FULL_COLLECTIONS[1]] -= 1
        bad = verify_full(manifest, live, {"user": 2, "chat": 0, "function": None})
        check(any(FULL_COLLECTIONS[1] in m for m in bad) and any("chat" in m for m in bad)
              and any("function" in m for m in bad) and len(bad) == 3,
              "verify: a point-count mismatch and empty/missing tables are each reported")

    with tempfile.TemporaryDirectory() as tmp:
        CTL_DIR, WORK_PARENT = os.path.join(tmp, "ctl"), tmp
        try:
            # request validation: the typed phrase is exact, "latest" is refused
            p = os.path.join(tmp, "request.json")
            for bad_req in ({"cmd": "restore-full", "snapshot": "1a2b3c4d", "repo": "weekly"},
                            {"cmd": "restore-full", "snapshot": "1a2b3c4d", "confirm": "restore"},
                            {"cmd": "restore-full", "snapshot": "latest", "confirm": "RESTORE"}):
                write_json(p, bad_req)
                try:
                    load_request(p)
                except RepoctlError:
                    check(True, "restore-full refused at load: %s" % sorted(bad_req.items())[0:3])
                else:
                    raise AssertionError("bad restore-full accepted: %r" % bad_req)
            write_json(p, {"cmd": "restore-full", "snapshot": "1a2b3c4d", "repo": "weekly",
                           "confirm": "RESTORE"})
            check(load_request(p)["cmd"] == "restore-full", "a confirmed restore-full request reads back")
            write_json(p, {"cmd": "rebuild-catalog"})
            check(load_request(p)["cmd"] == "rebuild-catalog",
                  "a rebuild-catalog request reads back (the catalog is rebuilt from restic)")
            req = {"id": "r-abc123", "cmd": "restore-full", "snapshot": "1a2b3c4d", "repo": "weekly"}
            fake = _FakeOps()
            try:
                cmd_restore_full(None, dict(req, confirm="nope"), ops=fake)
            except RepoctlError:
                check(fake.calls == [], "restore-full without RESTORE: refused before anything is called")
            else:
                raise AssertionError("unconfirmed restore ran")
            req["confirm"] = "RESTORE"

            # Review focus 1: a scheduled backup holds the lock
            fake = _FakeOps(lock_free=False)
            try:
                cmd_restore_full(None, req, ops=fake)
            except BusyError as exc:
                check("a backup is running" in str(exc)
                      and not any(c[0] in ("stop", "safety_backup") for c in fake.calls),
                      "restore-full while a backup holds the lock: refused, nothing stopped")
            else:
                raise AssertionError("restore ran under a backup's lock")

            # Review focus 3: the safety backup fails (Drive full)
            fake = _FakeOps(fail="safety_backup")
            try:
                cmd_restore_full(None, req, ops=fake)
            except RepoctlError as exc:
                check("Drive is full" in str(exc) and "nothing was replaced" in str(exc)
                      and not any(c[0] in ("pg_restore", "swap_files", "qdrant_restore") for c in fake.calls),
                      "safety backup fails: restore refused with its reason, nothing replaced")
            else:
                raise AssertionError("restore ran without a safety backup")

            # Review focus 4: a step fails after the containers were stopped
            fake = _FakeOps(fail="pg_restore")
            try:
                cmd_restore_full(None, req, ops=fake)
            except RepoctlError as exc:
                msg = str(exc)
                starts = [c[1] for c in fake.calls if c[0] == "start"]
                check(sorted(starts) == sorted(FULL_RESTORE_STOP) and fake.up >= set(FULL_RESTORE_STOP),
                      "a failed step: every container this run stopped is started again")
                check("database" in msg and "9f8e7d6c" in msg,
                      "a failed step: the error names the step and the safety snapshot id")
                check(getattr(exc, "result", {}).get("safety_snapshot") == "9f8e7d6c",
                      "a failed step: the result carries the safety snapshot id")
            else:
                raise AssertionError("a failed pg_restore was reported as success")
            check(not [d for d in os.listdir(tmp) if d.startswith("gmac-repoctl")],
                  "a failed restore leaves no work directory behind")

            # the happy path, against an OLD snapshot (Review focus 2)
            fake = _FakeOps(old_snapshot=True)
            res = cmd_restore_full(None, req, ops=fake)
            names = [c[0] for c in fake.calls]
            check(names.index("lock_free") < names.index("restic_restore") < names.index("safety_backup")
                  < names.index("pg_restore") < names.index("qdrant_restore"),
                  "restore-full order: lock check, read backup, safety backup, database, vectors")
            check(names.index("swap_files") < names.index("qdrant_restore")
                  and ("start", HAYSTACK) in fake.calls[names.index("swap_files"):names.index("qdrant_restore")],
                  "kb files are swapped while haystack is stopped, before it starts for the vectors")
            check(("safety_backup", "pre-restore-r-abc123") in fake.calls,
                  "the safety backup is a milestone tagged with the request id")
            qi = [i for i, c in enumerate(fake.calls) if c[0] == "qdrant_restore"]
            check(("refresh_planner",) in fake.calls
                  and fake.calls.index(("refresh_planner",)) > max(qi),
                  "after the vector store is back, the backend re-lists its searchable videos")

            # The panel's toggle: safety backup off (2026-10-28).
            fk = _FakeOps(old_snapshot=True)
            res_off = cmd_restore_full(None, dict(req, safety_backup=False), ops=fk)
            check(not any(c[0] == "safety_backup" for c in fk.calls) and res_off.get("safety_skipped")
                  and any(c[0] == "pg_restore" for c in fk.calls),
                  "safety backup off: the restore runs without one and says so")
            fk = _FakeOps(fail="qdrant_restore")
            try:
                cmd_restore_full(None, dict(req, safety_backup=False), ops=fk)
            except RestoreFailed as exc:
                check("NO safety backup" in str(exc) and "nothing was replaced" not in str(exc),
                      "safety backup off + a failed later step: the error never claims a way back")
            else:
                raise AssertionError("a failed vector restore was reported as success")
            fk = _FakeOps(fail="pg_restore")
            try:
                cmd_restore_full(None, dict(req, safety_backup=False), ops=fk)
            except RestoreFailed as exc:
                check("nothing was replaced" in str(exc) and "NO safety backup" not in str(exc),
                      "safety backup off + the database step fails: rolled back, nothing replaced")
            else:
                raise AssertionError("a failed pg_restore was reported as success")
            for bad in ("no", 0, None):
                try:
                    check_full_request(dict(req, safety_backup=bad))
                except RepoctlError:
                    pass
                else:
                    raise AssertionError("safety_backup=%r accepted" % (bad,))
            check(True, "safety_backup must be a real true/false")
            check(fake.calls[names.index("restic_restore")][1] == "1a2b3c4d" * 8,
                  "the chosen snapshot is restored by its full id, resolved before the safety backup")
            check(all(c[1] != "/data/openwebui/uploads" for c in fake.calls if c[0] == "swap_files")
                  and not any("openwebui" in str(c) for c in fake.calls if c[0] == "restic_restore"),
                  "an old snapshot's chat uploads are neither read nor written")
            check(len([c for c in fake.calls if c[0] == "qdrant_restore"]) == len(FULL_COLLECTIONS)
                  and all(c[2] for c in fake.calls if c[0] == "qdrant_restore"),
                  "all six collections restored, each with the manifest checksum")
            check(res["verified"] and res["safety_snapshot"] == "9f8e7d6c" and fake.up >= set(FULL_RESTORE_STOP),
                  "restore-full ends verified with every container running")
            cfg = {c["path"]: c["matches"] for c in res["config"]}
            check(cfg == {"/opt/gmac/compose.yml": True, "/opt/gmac/.env": False}
                  and not any(c[0] == "write_config" for c in fake.calls),
                  "config files are hash-compared and reported, never written")
            check(all(s["ok"] for s in res["steps"]) and len(res["steps"]) >= 7,
                  "the result lists every step")
            st = json.load(open(os.path.join(CTL_DIR, "status.json")))
            check(st["cmd"] == "restore-full" and st["state"] == "running",
                  "every step is written to status.json while it runs")

            # The catalog must know the safety snapshot for what it is: the
            # way back from this restore (ops/catalog.py, merged in).
            check(("catalog_safety", "9f8e7d6c") in fake.calls,
                  "the safety snapshot is recorded in the backup catalog")
            catalog_safety_record("9f8e7d6c", "r-abc123")    # what Ops.catalog_safety runs
            path = os.path.join(CTL_DIR, "catalog.json")
            cat = json.load(open(path))
            rec = [r for r in cat["snapshots"] if r["id"] == "9f8e7d6c"][0]
            check(rec["kind"] == "safety" and rec["note"] == "safety backup before restore r-abc123",
                  "the safety record names the restore request it protects")
            # The backup script's own record (written with the manifest) is
            # richer; the belt-and-braces record must never downgrade it.
            backup_catalog.write_catalog_file(path, backup_catalog.merge_record(cat, dict(
                rec, status="verified", verified_at="T",
                contains={"database": {"users": 3}, "collections": {}, "kb_files": 4,
                          "config_files": []})))
            catalog_safety_record("9f8e7d6c", "r-abc123")
            rec2 = [r for r in json.load(open(path))["snapshots"] if r["id"] == "9f8e7d6c"][0]
            check(rec2["status"] == "verified" and rec2["verified_at"] == "T"
                  and rec2["contains"]["kb_files"] == 4,
                  "the safety record never downgrades what the backup run recorded")

            # Silent-loss window: the safety backup is taken only once nobody
            # can write any more — the backup is read and checked first (so a
            # bad backup costs no downtime), then Open WebUI stops, then the
            # safety backup (it needs haystack up for the Qdrant snapshots),
            # then haystack stops.
            seq = fake.calls
            i_read, i_sha = names.index("restic_restore"), names.index("sha_check")
            i_safe = names.index("safety_backup")
            i_owu, i_hay = seq.index(("stop", OPENWEBUI)), seq.index(("stop", HAYSTACK))
            check(i_read < i_sha < i_owu < i_safe < i_hay,
                  "restore-full order: read + verify, stop Open WebUI, safety backup, stop haystack")
            check(names.index("hold_lock") > i_safe,
                  "the backup lock is taken only after the safety backup released it")

            # the safety backup fails AFTER Open WebUI was stopped: it must come back
            fake = _FakeOps(fail="safety_backup")
            try:
                cmd_restore_full(None, req, ops=fake)
            except RepoctlError:
                check(("stop", OPENWEBUI) in fake.calls and ("start", OPENWEBUI) in fake.calls
                      and OPENWEBUI in fake.up and ("stop", HAYSTACK) not in fake.calls,
                      "safety backup fails: Open WebUI is started again, haystack never stopped")
            else:
                raise AssertionError("restore ran without a safety backup")

            # a bad backup is refused before anything stops
            fake = _FakeOps(fail="sha_check")
            try:
                cmd_restore_full(None, req, ops=fake)
            except RepoctlError:
                check(not any(c[0] in ("stop", "safety_backup") for c in fake.calls),
                      "a corrupt backup is refused before any container stops or a safety backup runs")
            else:
                raise AssertionError("a corrupt backup was restored")

            # a swap failure names the keep dir even for the FIRST file swapped
            fake = _FakeOps(fail="swap_files")
            try:
                cmd_restore_full(None, req, ops=fake)
            except RepoctlError as exc:
                check("/data/haystack-uploads.pre-restore-r-abc123" in str(exc),
                      "a failed file swap: the error names the keep directory")
            else:
                raise AssertionError("a failed swap was reported as success")

            # write_json: root writes into a container-writable dir; a symlink
            # planted at the tmp name must never be followed
            target = os.path.join(tmp, "victim")
            open(target, "w").write("keep")
            out = os.path.join(tmp, "status.json")
            try:
                os.symlink(target, "%s.%d.tmp" % (out, os.getpid()))
            except (OSError, NotImplementedError):
                print("skip write_json symlink check (no symlinks on this machine)")
            else:
                write_json(out, {"a": 1})
                check(open(target).read() == "keep" and json.load(open(out)) == {"a": 1}
                      and not os.path.islink(out),
                      "write_json never follows a symlink planted at its tmp path")
        finally:
            CTL_DIR, WORK_PARENT = saved

def self_test() -> int:
    import tempfile
    n = 0

    def check(cond, label):
        nonlocal n
        assert cond, label
        n += 1
        print("ok %s" % label)

    check(valid_snapshot_id("1a2b3c4d") and valid_snapshot_id("latest"), "a recovery point id is an id")
    check(not valid_snapshot_id("--read-data") and not valid_snapshot_id("/etc/passwd")
          and not valid_snapshot_id("1 2") and not valid_snapshot_id(None), "an id is never a flag or a path")

    ok_path = "/data/haystack-uploads/notes.pdf"
    check(safe_restore_path(ok_path), "a file under the knowledge base's storage restores")
    for bad in ("/opt/gmac/.env", "/data/backup-drives/primary.json", "/etc/shadow",
                "/data/haystack-uploads/../openwebui/.env", "notes.pdf", "",
                "/data/haystack-uploads2/x", "/data/haystack-uploads/a\nb"):
        check(not safe_restore_path(bad), "refused: %r" % bad[:40])
    # Chat uploads are no longer backed up (2026-10-27); an old snapshot still
    # holds them, and restoring them would resurrect files nobody owns any more.
    check(not safe_restore_path("/data/openwebui/uploads/x"), "chat uploads are not a restore target")

    check(glob_escape("/data/x/a*b.csv") == "/data/x/a[*]b.csv", "a star in a name matches literally")
    check(glob_escape("/data/x/a[1].csv") == "/data/x/a[[]1].csv", "brackets in a name match literally")

    inv = [{"path": "/data/haystack-uploads/a.pdf", "size": 10},
           {"path": "/data/haystack-uploads/b.pdf", "size": 20}]
    plan = plan_restore(inv, ["/data/haystack-uploads/a.pdf", "/data/haystack-uploads/gone.pdf"])
    check([r["path"] for r in plan["restore"]] == ["/data/haystack-uploads/a.pdf"], "selected files restore")
    check(plan["missing"] == [{"path": "/data/haystack-uploads/gone.pdf"}], "a file not in the backup is reported")

    rows, other = inventory_rows([
        {"path": "/data/haystack-uploads/a.pdf", "size": 10, "type": "file"},
        {"path": "/opt/gmac/.env", "size": 5, "type": "file"},
        {"path": "/data/haystack-uploads", "type": "dir"}])
    check(rows == [{"path": "/data/haystack-uploads/a.pdf", "size": 10}] and other == 1,
          "the inventory offers only restorable files and counts the rest")
    parsed = parse_ndjson('{"message_type":"snapshot"}\n{"path":"/data/x","type":"file"}\n'
                          'not json\n{"broken\n')
    check(len(parsed) == 2 and parsed[0].get("message_type") == "snapshot",
          "restic's line-per-node output parses, bad lines skipped")

    with tempfile.TemporaryDirectory() as tmp:
        rec = {"cmd": "restore-files", "snapshot": "1a2b3c4d",
               "paths": ["/data/haystack-uploads/a.pdf"]}
        p = os.path.join(tmp, "request.json")
        write_json(p, rec)
        check(load_request(p)["cmd"] == "restore-files", "a written request reads back")
        write_json(p, {"cmd": "restore-files", "snapshot": "1a2b3c4d",
                       "paths": ["/opt/gmac/.env"]})
        try:
            load_request(p)
        except RepoctlError:
            check(True, "a request to restore config files is refused at load")
        else:
            raise AssertionError("config restore accepted")
        write_json(p, {"cmd": "nonsense"})
        try:
            load_request(p)
        except RepoctlError:
            check(True, "an unknown command is refused")
        else:
            raise AssertionError("unknown command accepted")

    check(valid_doc({"collection": "documents", "name": "a.csv"}), "a document request is valid")
    check(not valid_doc({"collection": "visual_frames", "name": "a"}) and not valid_doc({"collection":
          "documents", "name": "a\nb"}) and not valid_doc("x"), "other collections / odd names refused")
    check(valid_table("data_dl_es1_test_20261025215407") and not valid_table("users")
          and not valid_table("data_x; drop table users"), "only data_* tables are restorable")
    gone = deleted_docs([{"name": "a.csv"}, {"name": "b.pdf"}], [{"name": "b.pdf"}], "documents")
    check(gone == [{"name": "a.csv", "collection": "documents"}], "deleted = in the backup, not live")
    copy = ("SET x;\nCOPY public.data_catalog (table_name, filename) FROM stdin;\n"
            "data_a\ta.csv\ndata_b\tb.csv\n\\.\n")
    sql = catalog_sql(copy, ["data_a", "evil'; drop"])
    check("COPY restore_catalog (table_name, filename) FROM stdin;" in sql and "data_b\tb.csv" in sql
          and "IN ('data_a')" in sql and "ON CONFLICT (table_name) DO NOTHING" in sql
          and "evil" not in sql, "catalog rows: only the chosen tables, never overwriting")
    check(catalog_sql("SET x;\n", ["data_a"]) == "", "no catalog block -> no SQL")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "request.json")
        write_json(p, {"cmd": "restore-docs", "snapshot": "1a2b3c4d",
                       "docs": [{"collection": "documents", "name": "a.csv"}]})
        check(load_request(p)["docs"][0]["name"] == "a.csv", "a restore-docs request reads back")
        write_json(p, {"cmd": "restore-docs", "snapshot": "1a2b3c4d",
                       "docs": [{"collection": "users", "name": "x"}]})
        try:
            load_request(p)
        except RepoctlError:
            check(True, "a restore-docs request outside the document collections is refused")
        else:
            raise AssertionError("bad collection accepted")
    with tempfile.TemporaryDirectory() as tmp:
        saved = os.environ.get("BACKUP_DRIVE_OVERRIDE")
        os.environ["BACKUP_DRIVE_OVERRIDE"] = os.path.join(tmp, "primary.json")
        open(os.path.join(tmp, "signed-out.json"), "w").write("{}")
        try:
            RepoEnv("daily").setup()
        except RepoctlError as exc:
            check("not connected" in str(exc), "signed out on the page -> every request refuses (no fallback)")
        else:
            raise AssertionError("a signed-out Drive was used")
        if saved is None:
            os.environ.pop("BACKUP_DRIVE_OVERRIDE")
        else:
            os.environ["BACKUP_DRIVE_OVERRIDE"] = saved
    check(status_doc("snapshots", "done", "r-1", result={"count": 2})["result"]["count"] == 2,
          "the status file carries the result")
    # The three backup folders: a request names one (daily when absent), a
    # typo is refused, and a bad folder can never redirect the repository.
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "request.json")
        write_json(p, {"cmd": "snapshots", "repo": "weekly"})
        check(load_request(p)["repo"] == "weekly", "a request names its backup folder")
        write_json(p, {"cmd": "snapshots", "repo": "yearly"})
        try:
            load_request(p)
        except RepoctlError:
            check(True, "an unknown backup folder is refused")
        else:
            raise AssertionError("bad folder accepted")
        write_json(p, {"cmd": "restore-docs", "snapshot": "1a2b3c4d", "repo": "monthly",
                       "docs": [{"collection": "documents", "name": "a.csv"}]})
        check(load_request(p)["repo"] == "monthly", "a restore request names its backup folder")
    old_repo = os.environ.get("RESTIC_REPOSITORY")
    os.environ["RESTIC_REPOSITORY"] = "rclone:gdrive:backups"
    try:
        check(RepoEnv().env["RESTIC_REPOSITORY"] == "rclone:gdrive:backups/daily",
              "no folder given -> the daily folder's repository")
        check(RepoEnv("weekly").env["RESTIC_REPOSITORY"] == "rclone:gdrive:backups/weekly",
              "a request's folder picks that subfolder's repository")
        check(RepoEnv("nonsense").env["RESTIC_REPOSITORY"] == "rclone:gdrive:backups",
              "a bad folder never redirects the repository")
    finally:
        if old_repo is None:
            os.environ.pop("RESTIC_REPOSITORY", None)
        else:
            os.environ["RESTIC_REPOSITORY"] = old_repo
    # ── restore-full (Task 4) ──────────────────────────────────────────────
    _self_test_restore_full(check, tempfile)
    # ── deleted chats / videos: the COPY parser, the never-overwrite SQL ──
    dump_text = (
        "COPY public.user (id, name, email) FROM stdin;\n"
        "u1\tAdmin\t[EMAIL]\n"
        "\\.\n"
        "COPY public.chat (id, user_id, title, updated_at) FROM stdin;\n"
        "c1\tu1\tMy old chat\t1761567000000\n"
        "c2\tu1\tKept chat\t1761568000000\n"
        "\\.\n"
        "COPY public.chat_message (id, chat_id, user_id, role, content) FROM stdin;\n"
        "m1\tc1\tu1\tuser\t{\"text\": \"a\\tb\"}\n"
        "m2\tc1\tu1\tassistant\t{}\n"
        "m3\tc2\tu1\tuser\t{}\n"
        "\\.\n"
        "COPY public.chat_file (id, user_id, chat_id, file_id, message_id, created_at, updated_at) FROM stdin;\n"
        "f1\tu1\tc1\tfile1\tm1\t1\t2\n"
        "\\.\n")
    cols, rows = copy_block(dump_text, "chat")
    check(cols == ["id", "user_id", "title", "updated_at"] and len(rows) == 2
          and rows[0].split("\t")[0] == "c1",
          "copy block: one table exactly — chat never picks up chat_message rows")
    check(row_field(rows[0], cols, "title") == "My old chat"
          and epoch_ms_iso(row_field(rows[0], cols, "updated_at")).startswith("2025-10-"),
          "copy rows: columns read by name; the bigint timestamp becomes readable")
    # A reserved-word table name arrives QUOTED (public."user") — the parser
    # must still find it, or every chat would show 'unknown account'.
    ucols, urows = copy_block('COPY public."user" (id, name, email) FROM stdin;\n'
                              "u1\tAdmin\t[EMAIL]\n\\.\n", "user")
    check(ucols == ["id", "name", "email"] and len(urows) == 1,
          "copy block: a quoted table name (public.\"user\") is found too")
    # Open WebUI rows carry seconds or milliseconds depending on age — both
    # must read as the same real date, never as 1970.
    check(epoch_ms_iso("1761567000")[:10] == epoch_ms_iso("1761567000000")[:10]
          and epoch_ms_iso("1761567000000000")[:10] == "2025-10-27"
          and epoch_ms_iso("0") == "" and epoch_ms_iso("x") == "",
          "timestamps: seconds, milliseconds and microseconds all read correctly")
    mcols, mrows = copy_block(dump_text, "chat_message")
    check(len(mrows) == 3 and mrows[0].split("\t")[4] == '{"text": "a\\tb"}',
          "copy rows: an escaped tab inside a JSON value never splits a row")
    gone = deleted_rows(rows, ["c2"], cols, "id")
    check([row_field(r, cols, "id") for r in gone] == ["c1"],
          "deleted rows: only what the backup holds and the live database lost")
    sql_one = restore_rows_sql("chat", cols, gone, "id")
    check("ON CONFLICT (id) DO NOTHING" in sql_one and 'INSERT INTO "chat"' in sql_one
          and restore_rows_sql("chat", cols, [], "id") == "",
          "restore sql: inserts only missing rows, never overwrites; empty is empty")
    users = {"u1": {"name": "Admin", "email": "[EMAIL]"}}
    s = chat_summary(rows[0], cols, users, {"c1": 2})
    check(s["account"] == "Admin" and s["messages"] == 2
          and s["title"] == "My old chat" and s["when"].startswith("2025-10-"),
          "chat summary: what it was called, whose it was, when, how much is in it")
    tables = {"user": copy_block(dump_text, "user"), "chat": (cols, rows),
              "chat_message": (mcols, mrows), "chat_file": copy_block(dump_text, "chat_file")}
    sql = chat_restore_sql(tables, ["c1"])
    check(sql.index('INSERT INTO "user"') < sql.index('INSERT INTO "chat" (')
          < sql.index('INSERT INTO "chat_message"') < sql.index('INSERT INTO "chat_file"'),
          "chat restore: account first, then the chat, its messages and file links")
    check("m1" in sql and "m2" in sql and "m3" not in sql and "Kept chat" not in sql,
          "chat restore: only the selected chat's rows come back")
    vcols, vrows = copy_block("COPY public.video_catalog (video_id, title) FROM stdin;\n"
                              "v1\tOld video\nv2\tKept\n\\.\n", "video_catalog")
    vsql = video_restore_sql(vcols, [vrows[0]])
    check("v1" in vsql and "v2" not in vsql and "ON CONFLICT (video_id) DO NOTHING" in vsql,
          "video restore: only the selected entries, never overwriting")
    check(valid_item_id("c1") and valid_item_id("9f8c2a1b-dead-beef")
          and not valid_item_id("--x") and not valid_item_id("a\nb") and not valid_item_id("")
          and not valid_item_id(None) and not valid_item_id("x" * 301),
          "item ids are opaque ids, never flags, paths or line breaks")

    # Progress bar (2026-10-28): monotone through a full restore, honest at the ends.
    full = [progress_pct("restore-full", "running", s, f) for s, f in
            [("checking the request", 0), ("reading the backup", 0), ("reading the backup", 0.5),
             ("stopping Open WebUI", 0), ("taking a safety backup of the present", 0.5),
             ("restoring the database", 0), ("verifying", 1.0)]]
    check(full == sorted(full) and full[0] == 0 and full[2] == 18 and full[-1] == 99,
          "progress: a full restore's bar only moves forward and stays below 100 while running")
    check(progress_pct("find-deleted-chats", "running", "reading the backup", 0.5) == 42
          and progress_pct("find-deleted-chats", "running", "comparing with the live database") == 85
          and progress_pct("restore-chats", "running", "restoring 2 chat(s)") == 85,
          "progress: reading the backup is most of a find/restore, the rest follows it")
    check(progress_pct("restore-chats", "done") == 100 and progress_pct("x", "failed") is None
          and progress_pct("restore-full", "running", "a step nobody listed") is None,
          "progress: 100 only when done, no number when failed or unknown")
    check(progress_pct("find-deleted-chats", "running", "started") == 0,
          "progress: a job starts at 0, not at the band its unknown first step falls into")
    cat = {"snapshots": [{"id": "ab" * 32, "short_id": "abababab", "contains": {"database": {}}},
                         {"id": "cd" * 32, "short_id": "cdcdcdcd"}]}
    check(known_record(cat, "ab" * 32) and known_record(cat, "abababab")
          and not known_record(cat, "cd" * 32) and not known_record(cat, "ef" * 32)
          and not known_record({}, "ab"),
          "rebuild: a described snapshot is kept, a bare or unknown one is read again")
    check("status = 'success'" in stale_runs_sql("ab12cd34") and "'ab12cd34'" in stale_runs_sql("ab12cd34")
          and "x';" not in stale_runs_sql("x'; DROP TABLE chat; --"),
          "restore: the restored run's own row becomes a success with its snapshot id, never SQL")
    check("UPDATE public.ops_backup_reports" in stale_runs_sql("ab12cd34"),
          "restore: the run-row update names its schema (the dump empties search_path)")
    k = keep_runs_sql("26\t2026-10-28 13:03:00+00\t\\N\tt\tsuccess\tab\t1\t\\N\t\\N")
    check(keep_runs_sql("") == "" and "COPY _keep_runs" in k and "\\.\n" in k
          and "WHERE NOT EXISTS" in k and "INSERT INTO public.ops_backup_reports" in k,
          "restore: newer backup-history rows are carried across, never duplicated")

    # Orphan-safe database restore (2026-10-28: 4 chat_file rows with no chat
    # made every full restore fail at chat_file_chat_id_fkey).
    post = ("ALTER TABLE ONLY public.chat_file\n    ADD CONSTRAINT chat_file_chat_id_fkey FOREIGN KEY "
            "(chat_id) REFERENCES public.chat(id) ON DELETE CASCADE;\n\nALTER TABLE ONLY public.x\n"
            "    ADD CONSTRAINT x_pkey PRIMARY KEY (id);\nALTER TABLE ONLY public.m\n    ADD CONSTRAINT "
            "m_fk FOREIGN KEY (a, b) REFERENCES public.n(a, b);\n")
    fks = foreign_keys(post)
    check(fks == [("public.chat_file", "chat_file_chat_id_fkey", ["chat_id"], "public.chat", ["id"]),
                  ("public.m", "m_fk", ["a", "b"], "public.n", ["a", "b"])],
          "orphans: every foreign key (one or more columns) is found, primary keys are not")
    sql = orphan_cleanup_sql(fks)
    check("DELETE FROM public.chat_file t WHERE t.chat_id IS NOT NULL AND NOT EXISTS "
          "(SELECT 1 FROM public.chat r WHERE r.id = t.chat_id)" in sql
          and "r.a = t.a AND r.b = t.b" in sql and orphan_cleanup_sql([]) == "",
          "orphans: a row is deleted only when its non-null key matches nothing")
    check(orphans_removed(" ?column? \n-----------\n orphans 4\n(1 row)\n") == 4
          and orphans_removed("") == 0, "orphans: the count psql prints is read back")
    check(_restic_fraction('{"message_type":"status","percent_done":0.25}', 0.1) == 0.25
          and _restic_fraction('{"message_type":"status","percent_done":0.05}', 0.1) == 0.1
          and _restic_fraction("not json", 0.3) == 0.3
          and _restic_fraction('{"percent_done": "x"}', 0.3) == 0.3,
          "progress: restic's percent_done is read, never moves the bar backwards")
    frac = 0.0
    for line in ["2026-10-28T11:52:00Z postgres: dumping openwebui", "noise",
                 "2026-10-28T11:53:00Z qdrant: snapshotting od_frames",
                 "2026-10-28T11:53:30Z restic: backing up to rclone:gdrive:backups/daily",
                 "2026-10-28T11:53:31Z postgres: dumping again?"]:
        frac = backup_stage(line, frac)
    check(frac == 0.55, "progress: the backup script's log lines mark its stage, forward only")

    print("\n%d repoctl self-checks passed" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
