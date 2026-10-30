"""
Google Drive intake (V6) and Drive auto-delete (V7).

Agreed with the client for M3: a local fetcher on his PC bulk-downloads videos and
uploads them to a Google Drive folder; the server picks them up from there and
runs them through the SAME pipeline as every other video (download phase ->
transcription -> visual -> embeddings -> citations). This module is the
server-side half. Nothing here runs on its own:

  - scan    : lists the folder and classifies each file against the video job
              table. Read-only by construction -- ingestion is client-triggered
              only, so a scan that queued would be an unrequested paid run.
  - queue   : (routers/video.py) explicit file ids + confirm=true, catalogues
              each as platform "gdrive" and queues it; the queue worker's
              download phase then copies the file by id via rclone
              (download_for_ingest), so every later phase is unchanged.
  - cleanup : deletes Drive files whose video is fully ingested ("done") and
              finished at least the window ago. Dry run unless the admin toggle
              is ON (the operator's 10-22 promise: auto-delete after 1-2 days, with a
              toggle). A file that cannot be proven safely ingested is kept.

Identity: a Drive file's id is stable across renames and moves, so the video_id
is "gd_" + file id. That makes "is this file already queued/ingested" a primary
key lookup on video_catalog, and keeps Drive videos from ever colliding with an
11-char YouTube id.

rclone is the transport (installed on the VPS host with remote `gdrive:`, used
by ops/backup.sh); compose.yml mounts the host binary and config read-only.
Without them (or without DRIVE_INTAKE_FOLDER_ID), every entry point raises
DriveIntakeNotConfigured, which the router turns into a 503
naming what is missing, instead of a FileNotFoundError 500.

Env (read here, not in config.py, which is infra-owned):
    DRIVE_INTAKE_FOLDER_ID      Drive folder to read (required; no default, so
                                switching to the client's new folder is an env change)
    DRIVE_INTAKE_RCLONE_REMOTE  rclone remote name, default "gdrive:"
    DRIVE_INTAKE_RCLONE_BIN     rclone binary, default "rclone" (on PATH)
    DRIVE_INTAKE_RCLONE_CONFIG  rclone.conf path; optional (rclone's own default
                                lookup applies when unset). Only read: rclone runs
                                on a private copy (see _private_config)
    DRIVE_INTAKE_RCLONE_WORKDIR where that copy lives, default /tmp/rclone-intake
    DRIVE_INTAKE_DRIVE_OVERRIDE the admin panel's Google Drive sign-in
                                (/data/backup-drives/primary.json by default),
                                applied to that copy so every call runs on the
                                connected account; absent -> the server's remote

Self-checks: PYTHONPATH=. python tests/test_drive_intake.py
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional

logger = logging.getLogger("haystack-api.drive-intake")

VIDEO_ID_PREFIX = "gd_"
PLATFORM = "gdrive"

# By extension OR by Drive's mime type: the fetcher may upload a file with no
# extension (Drive still sniffs video/*), and Drive may report a generic
# application/octet-stream for an .mkv. Either signal is enough.
VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi",
                              ".mpg", ".mpeg", ".ts", ".flv", ".wmv", ".3gp"})

# Auto-delete window. the operator told the client "1-2 days" (10-22), so the admin can
# pick inside that range and nothing outside it: a stray "1" would otherwise
# delete source files an hour after ingestion, before anyone could spot-check.
AUTODELETE_DEFAULT_HOURS = 48
AUTODELETE_MIN_HOURS = 24
AUTODELETE_MAX_HOURS = 48

# video_settings keys. Separate from the auto_check_* keys on purpose: the
# panel's existing "Save Settings" POSTs a full VideoSettings, so a drive field
# added there would be reset to its default on every unrelated save.
SETTING_ENABLED = "drive_autodelete_enabled"
SETTING_HOURS = "drive_autodelete_hours"
# Auto-approve (2026-10-26): new videos in the upload folder are queued and
# processed without a click. Default OFF, like auto-delete.
SETTING_AUTOAPPROVE = "drive_autoapprove_enabled"

# The only job status that means "fully ingested" (routers/video.py
# _VALID_STATUSES: new, queued, processing, done, failed).
DONE_STATUS = "done"


class DriveIntakeNotConfigured(RuntimeError):
    """rclone binary, its config, or the folder id is missing -> HTTP 503."""


class DriveCleanupDisabled(RuntimeError):
    """A real (non-dry-run) cleanup was asked for while the toggle is OFF."""


class DriveDeleteRefused(RuntimeError):
    """A delete could not be pinned to exactly one Drive file."""


@dataclass
class DriveFile:
    file_id: str
    name: str
    size: int = 0
    mime_type: str = ""
    mod_time: str = ""


# ── identity ─────────────────────────────────────────────

def video_id_for(file_id: str) -> str:
    return VIDEO_ID_PREFIX + file_id


def file_id_for(video_id: str) -> Optional[str]:
    """None for anything that is not a Drive video -- cleanup keys on this, so a
    YouTube id must never map to a Drive file."""
    if video_id and video_id.startswith(VIDEO_ID_PREFIX) and len(video_id) > len(VIDEO_ID_PREFIX):
        return video_id[len(VIDEO_ID_PREFIX):]
    return None


def drive_view_url(file_id: str) -> str:
    """What a citation links to (citation_formatter uses source_url verbatim for
    every non-YouTube platform). Goes dead once the file is auto-deleted."""
    return "https://drive.google.com/file/d/%s/view" % file_id


def _ext(name: str) -> str:
    return os.path.splitext(name or "")[1].lower()


def is_video_file(f: DriveFile) -> bool:
    return _ext(f.name) in VIDEO_EXTENSIONS or (f.mime_type or "").lower().startswith("video/")


def title_for(f: DriveFile) -> str:
    stem = os.path.splitext(f.name)[0] if _ext(f.name) else f.name
    return stem.strip() or f.file_id


# ── listers ──────────────────────────────────────────────

class DriveLister:
    """What the router needs from Drive. Real: RcloneDriveLister. Tests: FakeDriveLister."""

    folder_id = ""

    def list_files(self) -> List[DriveFile]:
        raise NotImplementedError

    def download(self, file_id: str, dest_path: str) -> str:
        raise NotImplementedError

    def delete(self, file_id: str, name: str) -> None:
        raise NotImplementedError


class FakeDriveLister(DriveLister):
    def __init__(self, files: Iterable[DriveFile] = (), folder_id: str = "FAKE"):
        self.files = list(files)
        self.folder_id = folder_id
        self.downloads = []
        self.deleted = []

    def list_files(self):
        return list(self.files)

    def download(self, file_id, dest_path):
        self.downloads.append((file_id, dest_path))
        return dest_path

    def delete(self, file_id, name):
        self.deleted.append(file_id)
        self.files = [f for f in self.files if f.file_id != file_id]


class RcloneDriveLister(DriveLister):
    """rclone over subprocess. Every command carries --drive-root-folder-id, so
    the folder comes from config (the client's new folder is an env change), not from
    whatever root_folder_id the shared rclone.conf happens to hold for backups."""

    def __init__(self, folder_id: str, remote: str = "gdrive:", binary: str = "rclone",
                 config_path: str = "", runner: Callable = subprocess.run,
                 list_timeout_s: int = 120, transfer_timeout_s: int = 3600):
        self.folder_id = folder_id
        self.remote = remote if remote.endswith(":") else remote + ":"
        self.binary = binary
        self.config_path = config_path
        self.runner = runner
        self.list_timeout_s = list_timeout_s
        self.transfer_timeout_s = transfer_timeout_s

    def _flags(self) -> List[str]:
        flags = ["--drive-root-folder-id", self.folder_id]
        if self.config_path:
            flags += ["--config", self.config_path]
        return flags

    def _run(self, args: List[str], timeout: int) -> str:
        cmd = [self.binary] + args + self._flags()
        r = self.runner(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            # stderr only, never the command env/config: rclone does not echo
            # tokens on error, and the config path is not secret.
            raise RuntimeError("rclone %s failed (exit %s): %s"
                               % (args[0], r.returncode, (r.stderr or "").strip()[:300]))
        return r.stdout or ""

    def list_files(self) -> List[DriveFile]:
        out = self._run(["lsjson", self.remote, "--files-only"], self.list_timeout_s)
        try:
            items = json.loads(out)
        except ValueError:
            # Empty/garbled output must not read as "the folder is empty".
            raise RuntimeError("rclone lsjson returned non-JSON output")
        return [DriveFile(i.get("ID", ""), i.get("Name", ""), int(i.get("Size") or 0),
                          i.get("MimeType", ""), i.get("ModTime", ""))
                for i in items if not i.get("IsDir") and i.get("ID")]

    def download(self, file_id: str, dest_path: str) -> str:
        # copyid copies by ID: Drive allows two files with the same name in one
        # folder, so a name-based copy could ingest the wrong one.
        self._run(["backend", "copyid", self.remote, file_id, dest_path],
                  self.transfer_timeout_s)
        return dest_path

    def delete(self, file_id: str, name: str) -> None:
        # rclone deletefile is name-based. Re-list and require exactly one file
        # with this name AND the expected id; otherwise a duplicate name could
        # delete a different, not-yet-ingested upload. --drive-use-trash=false:
        # deleted PERMANENTLY, not moved to the Drive trash (by request,
        # 2026-10-26 -- an ingested upload must be gone, not recoverable).
        matches = [f for f in self.list_files() if f.name == name]
        if len(matches) != 1 or matches[0].file_id != file_id:
            raise DriveDeleteRefused("%d file(s) named %r in the folder; expected exactly "
                                     "one with id %s" % (len(matches), name, file_id))
        self._run(["deletefile", self.remote + name, "--drive-use-trash=false"], self.list_timeout_s)


# ── the upload folder: chosen in the admin panel, else DRIVE_INTAKE_FOLDER_ID ──
# A small JSON file on the uploads volume (persistent, written only through
# POST /video/drive-intake/folder) rather than video_settings: the worker's
# download phase runs synchronously in a thread and must see the same folder
# the scan and queue saw, without a database round-trip.
FOLDER_FILE_DEFAULT = "/data/haystack-uploads/.drive-intake-folder.json"
_FOLDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,128}$")


def folder_file(environ=None) -> str:
    env = os.environ if environ is None else environ
    return (env.get("DRIVE_INTAKE_FOLDER_FILE") or os.environ.get("DRIVE_INTAKE_FOLDER_FILE")
            or FOLDER_FILE_DEFAULT).strip()


def valid_folder_id(folder_id) -> bool:
    return isinstance(folder_id, str) and bool(_FOLDER_ID_RE.match(folder_id))


def selected_folder(environ=None) -> dict:
    """{'folder_id', 'name', 'source': 'panel'|'env'|'', 'env_folder_id'}."""
    env = os.environ if environ is None else environ
    env_folder = (env.get("DRIVE_INTAKE_FOLDER_ID") or "").strip()
    try:
        with open(folder_file(env), encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        rec = {}
    if isinstance(rec, dict) and valid_folder_id(rec.get("folder_id")):
        return {"folder_id": rec["folder_id"], "name": str(rec.get("name") or ""), "source": "panel",
                "env_folder_id": env_folder}
    return {"folder_id": env_folder, "name": "", "source": "env" if env_folder else "",
            "env_folder_id": env_folder}


def save_selected_folder(folder_id: str, name: str = "", environ=None) -> dict:
    if not valid_folder_id(folder_id):
        raise ValueError("not a Google Drive folder id")
    path = folder_file(environ)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".new"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"folder_id": folder_id, "name": (name or "")[:200],
                   "saved_at": datetime.now(timezone.utc).isoformat()}, fh)
    os.replace(tmp, path)                  # atomic: a scan never reads half a file
    return selected_folder(environ)


def lister_from_env(environ=None, which=shutil.which, exists=os.path.exists,
                    folder_id: str = None) -> RcloneDriveLister:
    env = os.environ if environ is None else environ
    folder = (folder_id if folder_id is not None else selected_folder(env)["folder_id"] or "").strip()
    binary = (env.get("DRIVE_INTAKE_RCLONE_BIN") or "rclone").strip()
    remote = (env.get("DRIVE_INTAKE_RCLONE_REMOTE") or "gdrive:").strip()
    config_path = (env.get("DRIVE_INTAKE_RCLONE_CONFIG") or "").strip()
    missing = []
    if not folder:
        missing.append("DRIVE_INTAKE_FOLDER_ID unset")
    if not which(binary):
        missing.append("rclone binary %r not found in the container" % binary)
    if config_path and not exists(config_path):
        missing.append("rclone config %s not mounted" % config_path)
    if missing:
        raise DriveIntakeNotConfigured("drive intake not configured: " + "; ".join(missing))
    over = override_file(env)
    if over and not os.path.exists(over) and \
            os.path.exists(os.path.join(os.path.dirname(over), "signed-out.json")):
        # Signed out on the admin panel's Drive Backup & Upload Handler: NOT
        # connected. Never fall back to the server's own remote (the previous
        # account) -- scans, downloads, auto-approve and auto-delete stop.
        raise DriveIntakeNotConfigured("Google Drive is not connected (signed out in the admin panel) "
                                       "— sign in with Google to resume uploads")
    if config_path:
        workdir = (env.get("DRIVE_INTAKE_RCLONE_WORKDIR") or _RCLONE_WORKDIR).strip()
        try:
            config_path = _private_config(config_path, workdir,
                                          override=override_file(env),
                                          remote=remote.rstrip(":"))
        except OSError as exc:
            raise DriveIntakeNotConfigured("drive intake not configured: rclone config %s unreadable "
                                           "or %s not writable (%s)" % (config_path, workdir, exc))
    return RcloneDriveLister(folder, remote=remote, binary=binary, config_path=config_path)


# rclone refreshes the Drive access token (hourly) and saves it back to its config
# by renaming a new file over the old one. The host's rclone.conf is bind-mounted
# read-only (it is shared with ops/backup.sh and must never be rewritten from a
# container), and a single-file bind mount cannot be renamed over even when writable:
# measured on the VPS 2026-10-25 with an expired-token copy, every call still listed
# the folder but logged "ERROR : Failed to save config after 10 tries ... rename
# /run/rclone/rclone.conf ...: device or resource busy". rclone therefore runs on a
# private 0600 copy in the container; it is re-copied only when the mounted file
# changes (re-authorised on the host), so the copy's refreshed token is kept between
# calls.
#
# The admin panel's Google Drive sign-in (/data/backup-drives/primary.json -- the
# same override ops/backup_drives.py applies for the nightly backup) replaces the
# [gdrive] token and folder in that copy, so scans, downloads, auto-approve and
# auto-delete all run on the connected account. No sign-in -> the server's own
# remote, unchanged.
_RCLONE_WORKDIR = "/tmp/rclone-intake"
OVERRIDE_FILE_DEFAULT = "/data/backup-drives/primary.json"


def override_file(environ=None) -> str:
    env = os.environ if environ is None else environ
    return (env.get("DRIVE_INTAKE_DRIVE_OVERRIDE") or OVERRIDE_FILE_DEFAULT).strip()


def load_override(path: str) -> dict:
    """The connected account's record, {} when there is none or it is unusable.

    Same validation and refusal as ops/backup_drives.py load_override: the
    values below are written into a config file, so a multi-line value would
    inject settings -- an injecting override is ignored, never applied."""
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not (isinstance(rec, dict) and (rec.get("token") or {}).get("refresh_token")
            and rec.get("root_folder_id")):
        return {}
    for value in (rec["root_folder_id"], json.dumps(rec["token"]),
                  rec.get("client_id", ""), rec.get("client_secret", "")):
        if "\n" in value or "\r" in value:
            return {}
    return rec


def apply_override(text: str, remote: str, rec: dict) -> str:
    """rclone.conf text -> same text with [remote]'s token and root_folder_id
    replaced by the connected account's (and its OAuth client too when the
    record names one: a web-client sign-in must be refreshed as the client that
    issued it, exactly as ops/backup_drives.py does for the backup)."""
    token = json.dumps(rec["token"], separators=(",", ":"))
    out, current, done = [], None, set()
    for raw in text.splitlines():
        m = re.match(r"^\s*\[(.+)\]\s*$", raw)
        if m:
            if current == remote:
                out.extend(_override_lines(done, rec, token))
            current = m.group(1).strip()
            out.append(raw)
            continue
        if current == remote and "=" in raw and not raw.strip().startswith(("#", ";")):
            key = raw.split("=", 1)[0].strip()
            if key == "token":
                out.append("token = %s" % token); done.add("token"); continue
            if key == "root_folder_id":
                out.append("root_folder_id = %s" % rec["root_folder_id"]); done.add("root_folder_id"); continue
            if key in ("client_id", "client_secret") and rec.get(key):
                out.append("%s = %s" % (key, rec[key])); done.add(key); continue
        out.append(raw)
    if current == remote:
        out.extend(_override_lines(done, rec, token))
    return "\n".join(out) + "\n"


def _override_lines(done, rec, token) -> list:
    extra = []
    if "token" not in done:
        extra.append("token = %s" % token)
    if "root_folder_id" not in done:
        extra.append("root_folder_id = %s" % rec["root_folder_id"])
    for key in ("client_id", "client_secret"):
        if key not in done and rec.get(key):
            extra.append("%s = %s" % (key, rec[key]))
    done.update(("token", "root_folder_id", "client_id", "client_secret"))
    return extra


def _private_config(mounted: str, workdir: str, override: str = "", remote: str = "gdrive") -> str:
    """The private copy, with the connected account applied when there is one.

    Re-copied when the mounted config or the panel's override changes. A sidecar
    note (not the copy's mtime) records what it was built from: rclone rewrites
    the copy itself when it refreshes a token, and that must not force a rebuild
    back to the stale mounted token on every call."""
    os.makedirs(workdir, mode=0o700, exist_ok=True)
    copy = os.path.join(workdir, "rclone.conf")
    marker = os.path.join(workdir, "rclone.conf.source")
    want = {"base": os.path.getmtime(mounted),
            "override": os.path.getmtime(override) if override and os.path.exists(override) else 0.0}
    try:
        with open(marker, encoding="utf-8") as fh:
            got = json.load(fh)
    except (OSError, ValueError):
        got = None
    if not os.path.exists(copy) or got != want:
        with open(mounted, encoding="utf-8") as fh:
            text = fh.read()
        rec = load_override(override) if override else {}
        if rec:
            text = apply_override(text, remote, rec)
        tmp = copy + ".new"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, copy)
        with open(marker, "w", encoding="utf-8") as fh:
            json.dump(want, fh)
    return copy


# ── classification ───────────────────────────────────────

async def job_index(db, video_ids: List[str]) -> Dict[str, dict]:
    """{video_id: {"status", "completed_at"}} for catalogued ids, latest job per
    video (video_id is not unique in video_ingest_jobs -- P97). A catalogued
    video with no job has status None."""
    if not video_ids:
        return {}
    rows = await db.fetch(
        "SELECT vc.video_id, j.status, j.completed_at FROM video_catalog vc "
        "LEFT JOIN LATERAL (SELECT status, completed_at FROM video_ingest_jobs "
        "  WHERE video_id = vc.video_id ORDER BY created_at DESC, id DESC LIMIT 1) j ON TRUE "
        "WHERE vc.video_id = ANY($1::text[])",
        list(video_ids))
    return {r["video_id"]: {"status": r.get("status"), "completed_at": r.get("completed_at")}
            for r in rows}


_STATE_FOR_STATUS = {None: "new", "new": "new", "queued": "queued",
                     "processing": "processing", DONE_STATUS: "ingested", "failed": "failed"}


def state_for(index: Dict[str, dict], file_id: str) -> str:
    entry = index.get(video_id_for(file_id))
    if entry is None:
        return "new"
    # An unrecognised status is "unknown", never "new": "new" invites a queue.
    return _STATE_FOR_STATUS.get(entry.get("status"), "unknown")


def _row(f: DriveFile) -> dict:
    return {"file_id": f.file_id, "name": f.name, "size": f.size,
            "mime_type": f.mime_type, "mod_time": f.mod_time}


def classify(files: List[DriveFile], index: Dict[str, dict]) -> dict:
    videos, ignored = [], []
    counts = {s: 0 for s in ("new", "queued", "processing", "ingested", "failed", "unknown")}
    for f in files:
        if not is_video_file(f):
            ignored.append(_row(f))
            continue
        state = state_for(index, f.file_id)
        counts[state] += 1
        entry = index.get(video_id_for(f.file_id)) or {}
        videos.append(dict(_row(f), video_id=video_id_for(f.file_id), state=state,
                           job_status=entry.get("status")))
    return {"videos": videos, "ignored": ignored, "counts": counts}


async def scan(db, lister: DriveLister) -> dict:
    files = lister.list_files()
    index = await job_index(db, [video_id_for(f.file_id) for f in files if is_video_file(f)])
    return dict(classify(files, index), folder_id=lister.folder_id, dry_run=True)


# ── download phase ───────────────────────────────────────

def download_for_ingest(video_id: str, output_dir: str, lister: DriveLister = None) -> str:
    """Called by video_acquisition.download_video for platform "gdrive" -- the
    queue worker's download phase, so a retry re-downloads like any platform.
    Raises (download is always fatal) if the file left the folder."""
    file_id = file_id_for(video_id)
    if file_id is None:
        raise RuntimeError("%s is not a Drive video id" % video_id)
    lister = lister or lister_from_env()
    match = next((f for f in lister.list_files() if f.file_id == file_id), None)
    if match is None:
        raise RuntimeError("drive file %s not found in folder %s (deleted or moved?)"
                           % (file_id, lister.folder_id))
    # No makedirs: ingest_video() already creates work_dir, and rclone's copyto
    # creates missing parents itself.
    # <video_id>.<ext>: extract_audio and the later phases name their outputs
    # after the file stem, same as yt-dlp's "%(id)s.%(ext)s".
    dest = os.path.join(output_dir, video_id + (_ext(match.name) or ".mp4"))
    lister.download(file_id, dest)
    logger.info("drive file %s copied to %s", file_id, dest)
    return dest


# ── settings ─────────────────────────────────────────────

def clamp_hours(value) -> int:
    try:
        hours = int(value)
    except (TypeError, ValueError):
        return AUTODELETE_DEFAULT_HOURS
    return max(AUTODELETE_MIN_HOURS, min(AUTODELETE_MAX_HOURS, hours))


def parse_settings(raw: Dict[str, str]) -> dict:
    """Default OFF: nothing is deleted until an admin turns it on."""
    return {"autodelete_enabled": (raw.get(SETTING_ENABLED) or "false") == "true",
            "autodelete_hours": clamp_hours(raw.get(SETTING_HOURS, AUTODELETE_DEFAULT_HOURS)),
            "autoapprove_enabled": (raw.get(SETTING_AUTOAPPROVE) or "false") == "true"}


def settings_pairs(enabled: bool = None, hours=None, autoapprove: bool = None) -> Dict[str, str]:
    """Only the settings actually given -- so the Video Library's auto-delete
    form never switches auto-approve off, and the reverse."""
    out = {}
    if enabled is not None:
        out[SETTING_ENABLED] = "true" if enabled else "false"
    if hours is not None:
        out[SETTING_HOURS] = str(clamp_hours(hours))
    if autoapprove is not None:
        out[SETTING_AUTOAPPROVE] = "true" if autoapprove else "false"
    return out


# ── cleanup ──────────────────────────────────────────────

def _aware(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    # asyncpg returns timestamptz as aware; a naive value is the DB's UTC.
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def plan_cleanup(files: List[DriveFile], index: Dict[str, dict], window_hours: int,
                 now: datetime) -> dict:
    """Pure selection. Deletable only when: a video file, catalogued as a Drive
    video, latest job "done", completed_at known and >= window ago. Every other
    file is kept with the reason, so the preview explains itself."""
    cutoff = now - timedelta(hours=window_hours)
    would_delete, kept = [], []
    for f in files:
        row = _row(f)
        if not is_video_file(f):
            kept.append(dict(row, reason="not a video file"))
            continue
        entry = index.get(video_id_for(f.file_id))
        if entry is None:
            kept.append(dict(row, reason="not matched to an ingestion job"))
            continue
        status = entry.get("status")
        if status != DONE_STATUS:
            kept.append(dict(row, reason="job status %s" % (status or "not queued")))
            continue
        done_at = _aware(entry.get("completed_at"))
        if done_at is None:
            kept.append(dict(row, reason="ingested, finish time unknown"))
            continue
        if done_at > cutoff:
            kept.append(dict(row, reason="ingested %s, inside the %d h window"
                             % (done_at.isoformat(timespec="minutes"), window_hours)))
            continue
        would_delete.append(dict(row, video_id=video_id_for(f.file_id),
                                 completed_at=done_at.isoformat(timespec="minutes")))
    return {"would_delete": would_delete, "kept": kept}


async def run_cleanup(db, lister: DriveLister, *, enabled: bool, window_hours: int,
                      dry_run: bool = True, now: datetime = None) -> dict:
    if not dry_run and not enabled:
        raise DriveCleanupDisabled("drive auto-delete is OFF in the admin panel; "
                                   "only a dry run is allowed")
    now = now or datetime.now(timezone.utc)
    window_hours = clamp_hours(window_hours)
    # rclone calls block for seconds: off the event loop, like every other
    # subprocess-backed call in routers/video.py.
    files = await asyncio.to_thread(lister.list_files)
    index = await job_index(db, [video_id_for(f.file_id) for f in files if is_video_file(f)])
    plan = plan_cleanup(files, index, window_hours, now)
    deleted, errors = [], []
    if not dry_run:
        for item in plan["would_delete"]:
            try:
                await asyncio.to_thread(lister.delete, item["file_id"], item["name"])
                deleted.append(item)
                logger.info("drive auto-delete: removed %s (%s)", item["file_id"], item["name"])
            except Exception as exc:
                # One bad file must not stop the rest, and must be reported.
                errors.append(dict(item, error=str(exc)[:300]))
                logger.warning("drive auto-delete: kept %s: %s", item["file_id"], exc)
    return dict(plan, deleted=deleted, errors=errors, dry_run=dry_run, enabled=enabled,
                window_hours=window_hours, folder_id=lister.folder_id,
                checked_at=now.isoformat(timespec="seconds"))
