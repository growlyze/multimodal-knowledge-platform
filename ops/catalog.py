#!/usr/bin/env python3
"""catalog.py - the backup catalog (catalog.json): one record per recovery point.

The catalog is the human description of every backup: when it was taken, what
it contains (users, chats, documents, videos, knowledge-base files), what it
does NOT contain, and whether it has been checked. It is what the admin
panel's "Restore backup" list shows; the panel reads the LOCAL copy and never
talks to Google Drive for it.

Layout on Google Drive (next to the three repositories):

    <backup folder>/catalog.json     <- one record per snapshot, all folders
    <backup folder>/daily/           restic repository
    <backup folder>/weekly/          restic repository
    <backup folder>/monthly/         restic repository

Local copy for the panel: /data/backup-drives/ctl/catalog.json.

Written by:
    ops/backup.sh   at the end of every run (and so, after retention)
    ops/repoctl.py        after a full restore's safety backup, and by the
                          rebuild-catalog command when the catalog is missing
                          or corrupt.

Source of truth is ALWAYS restic: every record names a snapshot in one of the
repositories, and `rebuild-catalog` rebuilds the whole catalog from
`restic snapshots --json` + each snapshot's MANIFEST.json. The catalog holds
no secrets (counts, names and times only).

Commands (stdout = JSON, stderr = diagnostics, non-zero exit on failure):
    python3 ops/catalog.py selftest    offline checks (no restic, no network)
    python3 ops/catalog.py record <manifest.json> <facts.json>
                                       one record JSON on stdout
    python3 ops/catalog.py add <catalog.json> <manifest.json> <facts.json>
                                       [<ids-file>|-]   merge the record in,
                                       then prune against the ids file
    python3 ops/catalog.py prune <catalog.json> <ids-file>
                                       mark records restic no longer lists

`facts.json` is what the run knows beyond the manifest:
    {"id", "short_id", "folder", "tags", "kind", "status", "verified_at",
     "size_added_bytes", "note", "taken_at"}
"kind" and "note" are derived from "tags" when absent. `ids-file` holds one
snapshot id (full or short) per line - every record NOT in it is marked
pruned. "-" means "do not prune this time".

Stdlib only.
"""

import json
import os
import sys
from datetime import datetime, timezone

CATALOG_VERSION = 1
# daily / weekly / monthly are the three repositories. The panel's request
# protocol only names these; a record's folder is its repository's subfolder.
TARGETS = ("daily", "weekly", "monthly")
KINDS = ("scheduled", "safety", "manual")
# verified   - `restic check` passed for the run that took it
# unverified - exists, but nothing checked it (a rebuilt record)
# failed-check - a check FAILED; restorable but suspect
# pruned     - restic no longer holds it; never shown as restorable
STATUSES = ("unverified", "verified", "failed-check", "pruned")
# The one thing a new snapshot does not hold. An older snapshot (before
# 2026-10-27) still carries chat uploads and correctly reports "missing: []".
UPLOADS_PATH = "/data/openwebui/uploads"
UPLOADS_NOTE = "Open WebUI chat attachments (not backed up since 2026-10-27)"


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# ── pure helpers (self-tested) ─────────────────────────────────────────────

def kind_from_tags(tags) -> str:
    """safety   - the milestone taken just before a full restore
       manual   - any other milestone run (an explicit human request)
       scheduled- the daily/weekly/monthly runs (Back Up Now included: it is
                  an extra daily recovery point under the same policy)."""
    tags = [str(t) for t in tags or []]
    if any("pre-restore-" in t for t in tags):
        return "safety"
    if any(t == "milestone" or t.startswith("milestone") for t in tags):
        return "manual"
    if any(t in TARGETS for t in tags):
        return "scheduled"
    return "manual"


def note_from_tags(tags) -> str:
    """The pre-restore tag carries the restore request id ('pre-restore-r-1a'
    -> 'safety backup before restore r-1a')."""
    for t in tags or []:
        idx = str(t).find("pre-restore-")
        if idx >= 0:
            return "safety backup before restore %s" % str(t)[idx + len("pre-restore-"):]
    return ""


def record_from_manifest(manifest, facts) -> dict:
    """One catalog record from a run's MANIFEST.json plus what only the run
    knows (facts). Tolerates every older manifest shape: a missing field is
    null/empty, never an error - old backups must stay listable."""
    manifest = manifest if isinstance(manifest, dict) else {}
    facts = facts if isinstance(facts, dict) else {}
    inv = manifest.get("inventory") or {}
    counts = inv.get("counts") or {}
    collections = {}
    for c in (manifest.get("qdrant") or {}).get("collections") or []:
        if isinstance(c, dict) and c.get("name"):
            collections[str(c["name"])] = c.get("points_count")
    sources = [s for s in manifest.get("source_files") or [] if isinstance(s, dict)]
    config_files = []
    for c in manifest.get("config") or []:
        if isinstance(c, dict) and c.get("path"):
            config_files.append(os.path.basename(str(c["path"]).rstrip("/")))
    # Video TITLES, never bare ids: the panel must say what a backup holds.
    videos = [str(v.get("title") or "").strip()
              for v in inv.get("videos") or [] if isinstance(v, dict)]
    tags = [str(t) for t in facts.get("tags") or []]
    kind = facts.get("kind") if facts.get("kind") in KINDS else kind_from_tags(tags)
    note = str(facts.get("note") or "")
    if not note and kind == "safety":
        note = note_from_tags(tags)
    status = facts.get("status") if facts.get("status") in STATUSES else "unverified"
    snap_id = str(facts.get("id") or "")
    record = {
        "id": snap_id,
        "short_id": str(facts.get("short_id") or snap_id[:8]),
        "folder": str(facts.get("folder") or ""),
        "kind": kind,
        "taken_at": str(facts.get("taken_at") or manifest.get("created_at") or ""),
        "status": status,
        "verified_at": str(facts.get("verified_at") or ""),
        "drill_passed_at": "",
        "size_added_bytes": facts.get("size_added_bytes") or 0,
        "contains": {
            "database": {"tables": counts.get("tables"), "users": counts.get("users"),
                         "chats": counts.get("chats")},
            "collections": collections,
            "kb_files": sum(int(s.get("files") or 0) for s in sources),
            "config_files": config_files,
        },
        # Older snapshots carried chat uploads, so they restore them and are
        # missing nothing; new ones say exactly what they cannot restore.
        # No manifest at all -> unknown, and the key is absent: a merge must
        # never replace an old record's answer with an assumption.
        # Videos: PROCESSED ones only (titles). `videos_catalog` is how many
        # the discovery catalog holds in total, so '5 processed (927 in the
        # catalog)' reads honestly instead of listing 927 raw entries.
        "videos": [t for t in videos if t],
        "videos_catalog": counts.get("videos_catalog"),
        "documents_count": len(inv.get("documents") or []),
        "note": note,
    }
    if "source_files" in manifest:
        record["missing"] = [] if any(s.get("path") == UPLOADS_PATH for s in sources) \
            else [UPLOADS_NOTE]
    return record


def _same(a: dict, b: dict) -> bool:
    """Same snapshot? Full ids equal, or one side's short id is a prefix of
    the other's id (restic short ids are unique inside one repository)."""
    ia, ib = str(a.get("id") or ""), str(b.get("id") or "")
    sa, sb = str(a.get("short_id") or ""), str(b.get("short_id") or "")
    if ia and ib and ia == ib:
        return True
    for x, y in ((ia, sb), (ib, sa)):
        if x and y and (x == y or x.startswith(y) or y.startswith(x)):
            return True
    return False


def _merge_status(old, new) -> str:
    """A record's status only ever moves forward with knowledge: a real check
    result (verified / failed-check) wins; 'unverified' (a rebuild) never
    downgrades a checked record but DOES clear 'pruned' (the merge proves the
    snapshot exists right now); 'pruned' is only set by prune_records."""
    old = old if old in STATUSES else "unverified"
    new = new if new in STATUSES else "unverified"
    if new in ("verified", "failed-check", "pruned"):
        return new
    return "unverified" if old == "pruned" else old


def _blank(value) -> bool:
    """'Unknown' in a record: a rebuild without the snapshot's MANIFEST.json
    writes blanks, and a blank must never erase what the taking run recorded.
    (For one snapshot the manifest is immutable, so a blank always means
    'not known here', never 'really zero'.)"""
    return value is None or value == "" or value == [] or value == {} or value == 0


def _merge_value(old, new):
    """New data wins; blanks never erase. Dicts merge key by key, so a record
    with only identity fields cannot wipe the counts another record holds."""
    if isinstance(new, dict):
        out = dict(old) if isinstance(old, dict) else {}
        for key, value in new.items():
            if key in out:
                out[key] = _merge_value(out[key], value)
            elif not _blank(value):
                out[key] = value
        return out
    return new if not _blank(new) else old


def merge_record(catalog, record) -> dict:
    """Upsert one record by id, keeping what a rebuild cannot know
    (verified_at, drill_passed_at, a checked status). Pure: returns a new
    catalog dict, never touches the input."""
    out = {"version": (catalog or {}).get("version") or CATALOG_VERSION,
           "updated_at": (catalog or {}).get("updated_at", ""),
           "snapshots": [dict(r) for r in (catalog or {}).get("snapshots") or []]}
    if not isinstance(record, dict) or not (record.get("id") or record.get("short_id")):
        return out
    idx = next((i for i, r in enumerate(out["snapshots"]) if _same(r, record)), None)
    if idx is None:
        out["snapshots"].append(dict(record))
        return out
    old = out["snapshots"][idx]
    merged = _merge_value(old, record)
    merged["status"] = _merge_status(old.get("status"), record.get("status"))
    merged["verified_at"] = str(record.get("verified_at") or old.get("verified_at") or "")
    merged["drill_passed_at"] = str(record.get("drill_passed_at") or
                                    old.get("drill_passed_at") or "")
    out["snapshots"][idx] = merged
    return out


def prune_records(catalog, live_ids, folder="") -> dict:
    """Mark every record restic no longer lists as 'pruned'. The record stays
    (history: what this system once had), but it is never shown as restorable.
    `live_ids`: full or short ids, as `restic snapshots --json` lists them.
    `folder`: only records of that repository are pruned — one folder's
    failed listing must never mark the other folders' backups gone."""
    live = [str(i) for i in (live_ids or []) if str(i)]
    out = {"version": (catalog or {}).get("version") or CATALOG_VERSION,
           "updated_at": (catalog or {}).get("updated_at", ""),
           "snapshots": []}
    for r in (catalog or {}).get("snapshots") or []:
        rec = dict(r)
        if folder and rec.get("folder") != folder:
            out["snapshots"].append(rec)
            continue
        if live and not any(_same(rec, {"id": i, "short_id": i}) for i in live):
            rec["status"] = "pruned"
        out["snapshots"].append(rec)
    return out


def read_ids(path: str):
    """ids-file -> [ids]; None when the file is '-' or absent (do not prune)."""
    if not path or path == "-":
        return None
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def read_catalog_file(path: str) -> dict:
    """The catalog as written before, or an empty one. A corrupt file is
    rebuilt from restic (rebuild-catalog), never an error here."""
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if isinstance(rec, dict) and isinstance(rec.get("snapshots"), list):
            return rec
    except (OSError, ValueError):
        pass
    return {"version": CATALOG_VERSION, "updated_at": "", "snapshots": []}


def write_catalog_file(path: str, catalog: dict) -> None:
    """Atomic, private, symlink-safe (same discipline as repoctl.write_json):
    this runs as ROOT into /data/backup-drives/ctl, a directory the admin
    panel container can write. O_EXCL + O_NOFOLLOW refuse a symlink planted
    at the predictable tmp name instead of following it."""
    catalog = dict(catalog)
    catalog["updated_at"] = now_iso()
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, mode=0o700, exist_ok=True)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(tmp, flags, 0o600)
    except FileExistsError:
        os.unlink(tmp)
        fd = os.open(tmp, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh)
    os.replace(tmp, path)


# ── commands (thin wrappers: file I/O around the pure helpers) ─────────────

def cmd_record(argv) -> int:
    manifest_path, facts_path = argv[0], argv[1]
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(facts_path, encoding="utf-8") as fh:
        facts = json.load(fh)
    json.dump(record_from_manifest(manifest, facts), sys.stdout)
    sys.stdout.write("\n")
    return 0


def cmd_add(argv) -> int:
    catalog_path, manifest_path, facts_path = argv[0], argv[1], argv[2]
    ids_path = argv[3] if len(argv) > 3 else "-"
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(facts_path, encoding="utf-8") as fh:
        facts = json.load(fh)
    catalog = merge_record(read_catalog_file(catalog_path),
                           record_from_manifest(manifest, facts))
    live = read_ids(ids_path)
    if live is not None:
        # Prune ONLY this run's own folder: `ids` lists one repository's
        # snapshots, and the other folders' records must stay untouched.
        catalog = prune_records(catalog, live, folder=str(facts.get("folder") or ""))
    write_catalog_file(catalog_path, catalog)
    return 0


def cmd_prune(argv) -> int:
    catalog_path, ids_path = argv[0], argv[1]
    live = read_ids(ids_path)
    if live is None:
        sys.stderr.write("catalog: prune needs an ids file (never '-')\n")
        return 2
    write_catalog_file(catalog_path, prune_records(read_catalog_file(catalog_path), live))
    return 0


# ── self-test (offline: no restic, no network) ─────────────────────────────

def selftest() -> int:
    n = 0

    def check(cond, label):
        nonlocal n
        assert cond, label
        n += 1
        print("ok %s" % label)

    # kind and note come from the run's restic tags (the only thing a rebuild
    # sees), so safety backups are recognisable without their shell run.
    check(kind_from_tags(["gmac-backup", "daily"]) == "scheduled",
          "kind: the daily/weekly/monthly runs are scheduled")
    check(kind_from_tags(["gmac-backup", "milestone:pre-restore-r-1a2b"]) == "safety",
          "kind: the pre-restore milestone is a safety backup")
    check(kind_from_tags(["gmac-backup", "milestone:quarter-end"]) == "manual",
          "kind: any other milestone run is manual")
    check(note_from_tags(["gmac-backup", "milestone:pre-restore-r-1a2b"]) ==
          "safety backup before restore r-1a2b",
          "note: the safety note names the restore request it protects")
    check(note_from_tags(["daily"]) == "" and kind_from_tags([]) == "manual",
          "kind/note: no tags -> manual, no note")

    manifest = {
        "created_at": "2026-10-28T03:05:12Z",
        "database": {"name": "openwebui", "size_bytes": 41234567,
                     "artifacts": [{"file": "postgres/openwebui.dump", "size": 5, "sha256": "p" * 64}]},
        "qdrant": {"collections": [{"name": "code_docs", "points_count": 3834},
                                   {"name": "visual_frames", "points_count": 3227}]},
        "source_files": [{"path": "/data/haystack-uploads", "bytes": 9, "files": 412}],
        "config": [{"path": "/opt/gmac/compose.yml"}, {"path": "/opt/gmac/.env"},
                   {"path": "/opt/gmac/tika-config"}],
        "inventory": {
            "documents": [{"name": "a.pdf"}, {"name": "b.pdf"}],
            "videos": [{"video_id": "v00", "title": "Fiber OTE"}, {"video_id": "v01", "title": ""}],
            "counts": {"tables": 123, "users": 3, "chats": 410, "videos_catalog": 927},
        },
    }
    rec = record_from_manifest(manifest, {
        "id": "27053f1a" * 8, "short_id": "27053f1a", "folder": "daily",
        "tags": ["gmac-backup", "milestone:pre-restore-r-1a2b"],
        "status": "verified", "verified_at": "2026-10-28T03:09:00Z",
        "size_added_bytes": 73289021})
    check(rec["kind"] == "safety" and rec["note"] == "safety backup before restore r-1a2b",
          "record: kind and note derived from the facts' tags")
    check(rec["taken_at"] == "2026-10-28T03:05:12Z" and rec["status"] == "verified"
          and rec["verified_at"] == "2026-10-28T03:09:00Z",
          "record: taken_at from the manifest, check state from the run")
    check(rec["contains"]["database"] == {"tables": 123, "users": 3, "chats": 410},
          "record: database counts ride along (users, chats, tables)")
    check(rec["contains"]["collections"] == {"code_docs": 3834, "visual_frames": 3227}
          and rec["contains"]["kb_files"] == 412
          and rec["contains"]["config_files"] == ["compose.yml", ".env", "tika-config"],
          "record: collections, kb files and config names recorded")
    check(rec["documents_count"] == 2 and rec["videos"] == ["Fiber OTE"],
          "record: documents counted, video TITLES listed (never bare ids)")
    check(rec["videos_catalog"] == 927,
          "record: processed videos listed, the catalogued total recorded beside them")
    check(rec["missing"] == [UPLOADS_NOTE],
          "record: a new snapshot says chat attachments are not in it")

    # An OLD snapshot (chat uploads still inside) restores them, so it is
    # missing nothing - and an old manifest without inventory still records.
    old = dict(manifest)
    old["source_files"] = [{"path": "/data/openwebui/uploads", "files": 7},
                           {"path": "/data/haystack-uploads", "files": 3}]
    old.pop("inventory")
    old_rec = record_from_manifest(old, {"id": "aa11bb22", "folder": "weekly",
                                         "tags": ["gmac-backup", "weekly"]})
    check(old_rec["missing"] == [] and old_rec["kind"] == "scheduled",
          "record: a pre-2026-10-27 snapshot holds chat uploads -> missing is empty")
    check("missing" not in record_from_manifest({}, {"id": "x"}),
          "record: no manifest -> missing unknown, never assumed")
    check(old_rec["contains"]["database"] == {"tables": None, "users": None, "chats": None}
          and old_rec["documents_count"] == 0 and old_rec["videos"] == []
          and old_rec["videos_catalog"] is None,
          "record: an old manifest without inventory records nothing false")

    # merge: upsert by id, and knowledge is never lost to a rebuild.
    catalog = {"version": 1, "updated_at": "", "snapshots": []}
    catalog = merge_record(catalog, rec)
    check(len(catalog["snapshots"]) == 1 and catalog["snapshots"][0]["id"] == rec["id"],
          "merge: a new record is added")
    rebuilt = record_from_manifest({}, {"id": rec["id"], "folder": "daily", "tags": []})
    merged = merge_record(catalog, rebuilt)["snapshots"][0]
    check(merged["status"] == "verified" and merged["verified_at"] == "2026-10-28T03:09:00Z",
          "merge: an unverified rebuild never downgrades a checked record")
    check(merged["contains"]["database"]["chats"] == 410,
          "merge: a rebuild without a manifest never wipes recorded counts")
    failed = merge_record(catalog, dict(rec, status="failed-check"))["snapshots"][0]
    check(failed["status"] == "failed-check",
          "merge: a later failed check is recorded (suspect until proven good)")
    drill = merge_record(catalog, dict(rec, drill_passed_at="2026-10-29T00:00:00Z"))["snapshots"][0]
    check(drill["drill_passed_at"] == "2026-10-29T00:00:00Z",
          "merge: a restore drill result upgrades the record")

    # prune: only restic's own list decides what is restorable.
    catalog2 = merge_record(catalog, dict(old_rec))
    pruned = prune_records(catalog2, [rec["short_id"]])
    check([r["status"] for r in pruned["snapshots"]] == ["verified", "pruned"],
          "prune: records restic no longer lists are marked pruned, kept as history")
    back = merge_record(pruned, rebuilt)
    check(back["snapshots"][0]["status"] == "verified",
          "prune: merging a live snapshot clears a stale pruned mark")
    check(prune_records(catalog2, []) == catalog2,
          "prune: an empty id list is ignored (a broken listing never prunes)")
    only_weekly = prune_records(catalog2, ["aaaaaaaa"], folder="weekly")
    check([r["status"] for r in only_weekly["snapshots"]] == ["verified", "pruned"],
          "prune: one folder's listing only prunes that folder's records")

    # The CLI wraps the same helpers on files (what the shell and repoctl use).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cat = os.path.join(td, "ctl", "catalog.json")
        man = os.path.join(td, "MANIFEST.json")
        facts = os.path.join(td, "facts.json")
        ids = os.path.join(td, "ids.txt")
        for path, obj in ((man, manifest), (facts, {"id": "27053f1a" * 8, "short_id": "27053f1a",
                                                    "folder": "daily", "tags": ["gmac-backup", "daily"],
                                                    "status": "verified", "verified_at": "T",
                                                    "size_added_bytes": 7})):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh)
        with open(ids, "w", encoding="utf-8") as fh:
            fh.write("27053f1a\n")
        check(cmd_add([cat, man, facts, ids]) == 0
              and read_catalog_file(cat)["snapshots"][0]["id"] == "27053f1a" * 8
              and read_catalog_file(cat)["snapshots"][0]["status"] == "verified",
              "cli add: writes the record and prunes against the ids file")
        # Pruning is scoped to the record's own folder: one repository's id
        # list must never mark ANOTHER folder's backups gone (a daily run
        # prunes daily; the weekly and monthly records stay untouched).
        with open(man, encoding="utf-8") as fh:
            weekly_manifest = json.load(fh)
        weekly = record_from_manifest(weekly_manifest, {"id": "aa11bb22cc33dd44", "folder": "weekly",
                                                        "tags": ["gmac-backup", "weekly"]})
        cat2 = merge_record(read_catalog_file(cat), weekly)
        write_catalog_file(cat, cat2)
        with open(ids, "w", encoding="utf-8") as fh:
            fh.write("27053f1a\n")
        check(cmd_add([cat, man, facts, ids]) == 0
              and [r["status"] for r in read_catalog_file(cat)["snapshots"]] == ["verified", "unverified"],
              "cli add: a daily run never prunes the weekly or monthly records")
        with open(ids, "w", encoding="utf-8") as fh:
            fh.write("ffffffff\n")
        check(cmd_prune([cat, ids]) == 0
              and read_catalog_file(cat)["snapshots"][0]["status"] == "pruned",
              "cli prune: marks records the repository no longer holds")
        check(read_catalog_file(os.path.join(td, "missing.json")) == {"version": 1, "updated_at": "",
                                                                     "snapshots": []},
              "cli read: a missing or corrupt catalog reads as empty (rebuilt, never an error)")

    print("\n%d catalog self-checks passed" % n)
    return 0


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    try:
        if cmd == "selftest":
            return selftest()
        if cmd == "record" and len(argv) >= 4:
            return cmd_record(argv[2:4])
        if cmd == "add" and len(argv) >= 5:
            return cmd_add(argv[2:])
        if cmd == "prune" and len(argv) >= 4:
            return cmd_prune(argv[2:4])
    except (OSError, ValueError) as exc:
        sys.stderr.write("catalog: %s\n" % exc)
        return 1
    sys.stderr.write("usage: catalog.py selftest | record <manifest> <facts> | "
                     "add <catalog> <manifest> <facts> [ids-file|-] | prune <catalog> <ids-file>\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
