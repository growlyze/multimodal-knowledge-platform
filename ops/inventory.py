#!/usr/bin/env python3
"""inventory.py - the knowledge base's item inventory for MANIFEST.json.

One inventory lists every item the backup covers at item granularity:
documents and spreadsheets (the two document collections in Qdrant, with their
data_* tables), the video catalog, and uploaded chat files. Each backup's
MANIFEST.json records it ("inventory"), so every recovery point remembers the
exact state it captured - which items existed, with what sizes and counts -
alongside the manifest's config checksums and tool versions.

Used by:
    ops/backup.sh   step 4: "inventory" in MANIFEST.json (best effort -
                          an extraction failure never fails a backup)

Commands (stdout = JSON data, stderr = diagnostics, non-zero exit on failure):
    python3 ops/inventory.py live        the live knowledge base's inventory
    python3 ops/inventory.py selftest    offline checks (no docker, no network)

Stdlib only.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

INSTALL_DIR = os.environ.get("INSTALL_DIR", "/opt/gmac")
STACK_NAME = os.environ.get("STACK_NAME", "gmac")
DOCS_HELPER = os.environ.get("DOCS_HELPER", os.path.join(INSTALL_DIR, "ops/qdrant_docs.py"))
HAYSTACK = os.environ.get("HAYSTACK_CONTAINER", STACK_NAME + "-haystack")
POSTGRES = os.environ.get("POSTGRES_CONTAINER", STACK_NAME + "-postgres")
PG_USER = os.environ.get("PG_USER", "openwebui")
PG_DB = os.environ.get("PG_DB", "openwebui")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DOC_COLLECTIONS = ("documents", "code_docs")
# Only what the backup actually holds: Open WebUI chat uploads left the backup
# on 2026-10-27, and listing them here would make every recovery point claim
# files it cannot restore.
UPLOAD_ROOTS = ("/data/haystack-uploads",)
KINDS = ("document", "table", "video", "upload")
SKIP_TABLES = ("data_catalog",)   # the catalog itself, not a spreadsheet's data


# ── pure helpers (self-tested) ─────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def item_name(kind: str, item: dict) -> str:
    """The item's stable name within its kind."""
    return str(item.get("name") or item.get("video_id") or item.get("path") or "")


def parse_count(text):
    """psql -tA 'count(*)' output -> int, or None when it is not one. The
    catalog's database counts are best effort: a None is shown as unknown,
    never as zero (a backup is never described with false numbers)."""
    try:
        return int(str(text or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def video_rows(text) -> list:
    """psql TSV 'video_id<TAB>title' -> [{"video_id","name","title"}]."""
    out = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            continue
        parts = ln.split("\t")
        out.append({"video_id": parts[0], "name": parts[0],
                    "title": parts[1] if len(parts) > 1 else ""})
    return out


def flatten(inv: dict) -> dict:
    """{key: {"kind","name","sig","size","item"}} across one inventory. A key
    is 'kind:name' and is stable across recovery points."""
    out = {}
    for kind in KINDS:
        for item in (inv or {}).get(kind + "s") or []:
            name = item_name(kind, item)
            if not name or (kind == "table" and name in SKIP_TABLES):
                continue
            out["%s:%s" % (kind, name)] = {"kind": kind, "name": name, "item": item}
    return out


# ── live extraction (host side: docker exec into haystack/postgres) ────────

def _docker(*args, stdin=None, timeout=600):
    try:
        return subprocess.run(["docker"] + list(args), input=stdin, capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("docker %s failed (%s)" % (args[0], type(exc).__name__))


def qdocs(*args, timeout=1800):
    with open(DOCS_HELPER, encoding="utf-8") as fh:
        script = fh.read()
    run = _docker("exec", "-i", "-e", "QDRANT_URL=" + QDRANT_URL, HAYSTACK, "python3", "-", *args,
                  stdin=script, timeout=timeout)
    if run.returncode != 0:
        raise RuntimeError((run.stderr or "").strip().splitlines() or ["vector store step failed"])[-1][:300]
    return json.loads(run.stdout or "null")


def _psql(sql: str, timeout=600) -> str:
    run = _docker("exec", "-i", POSTGRES, "psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-At",
                  "-F", "\t", "-U", PG_USER, "-d", PG_DB, stdin=sql, timeout=timeout)
    if run.returncode != 0:
        raise RuntimeError((run.stderr or "").strip().splitlines() or ["database step failed"])[-1][:300]
    return run.stdout or ""


def _uploads() -> list:
    out = []
    for root in UPLOAD_ROOTS:
        for base, _, files in os.walk(root):
            for name in files:
                path = os.path.join(base, name)
                try:
                    out.append({"path": path, "size": os.path.getsize(path)})
                except OSError:
                    continue
    return sorted(out, key=lambda r: r["path"])


# How many things the database held: the catalog's "database" counts. Only
# counts - the catalog is plain JSON shared with the panel and must never
# carry rows, names or secrets.
COUNT_QUERIES = (
    ("tables", "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';"),
    ("users", 'SELECT count(*) FROM "user";'),
    ("chats", "SELECT count(*) FROM chat;"),
    # video_catalog lists every DISCOVERED video (hundreds); only the ones with
    # a finished ingest job have processed content. Both numbers are recorded:
    # the catalog must never describe a backup by undiscovered-quantity.
    ("videos_catalog", "SELECT count(*) FROM video_catalog;"),
)

# PROCESSED videos only (frames, transcripts, sections done). A backup's
# description lists what it can restore meaningfully, not the raw discovery
# list (927 catalog rows vs 5 processed on 2026-10-28).
VIDEO_SQL = ("SELECT v.video_id, v.title FROM video_catalog v "
             "JOIN video_ingest_jobs j ON j.video_id = v.video_id "
             "WHERE j.status = 'done' ORDER BY v.video_id;")


def db_counts() -> dict:
    """{"tables","users","chats"} — each best effort: one broken query costs
    its own count only (recorded as null), never the inventory."""
    counts, notes = {}, []
    for key, sql in COUNT_QUERIES:
        try:
            value = parse_count(_psql(sql))
        except Exception as exc:                      # noqa: BLE001
            value = None
            notes.append("%s count could not be read (%s)" % (key, type(exc).__name__))
        if value is not None:
            counts[key] = value
    return counts, notes


def live_inventory() -> dict:
    """The live knowledge base's items. Each source is best effort: a broken
    source is reported in "notes" and the rest still counts - an inventory is
    for the manifest record, never for failing a backup."""
    notes = []
    documents, tables, videos = [], [], []
    for c in DOC_COLLECTIONS:
        try:
            for d in qdocs("sources", c) or []:
                documents.append(dict(d, collection=c))
        except Exception as exc:                      # noqa: BLE001
            notes.append("documents in %s could not be listed (%s)" % (c, type(exc).__name__))
    try:
        names = [ln.split("\t")[0] for ln in
                 _psql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' "
                       "AND table_name LIKE 'data\\_%' AND table_name <> 'data_catalog' "
                       "ORDER BY table_name;").splitlines() if ln.strip()]
        for name in names:
            rows = _psql("SELECT count(*) FROM %s;" % name).strip()
            tables.append({"name": name, "rows": int(rows or 0)})
    except Exception as exc:                          # noqa: BLE001
        notes.append("spreadsheet tables could not be listed (%s)" % type(exc).__name__)
    try:
        videos = video_rows(_psql(VIDEO_SQL))
    except Exception as exc:                          # noqa: BLE001
        notes.append("the video catalog could not be listed (%s)" % type(exc).__name__)
    counts = {}
    try:
        counts, count_notes = db_counts()
        notes.extend(count_notes)
    except Exception as exc:                          # noqa: BLE001
        notes.append("database counts could not be read (%s)" % type(exc).__name__)
    return {"taken_at": now_iso(), "documents": documents, "tables": tables,
            "videos": videos, "uploads": _uploads(), "counts": counts, "notes": notes}


# ── self-test (offline) ────────────────────────────────────────────────────

def selftest() -> int:
    n = 0

    def check(cond, label):
        nonlocal n
        assert cond, label
        n += 1
        print("ok %s" % label)

    inv = {"documents": [{"name": "doc1.pdf", "source_type": "document", "chunks": 3, "tables": []}],
           "tables": [{"name": "data_a", "rows": 12}, {"name": "data_catalog", "rows": 4}],
           "videos": [{"video_id": "v00", "name": "v00", "title": "t"}],
           "uploads": [{"path": "/data/openwebui/uploads/u1", "size": 10}]}
    flat = flatten(inv)
    check(sorted(flat) == ["document:doc1.pdf", "table:data_a", "upload:/data/openwebui/uploads/u1",
                           "video:v00"],
          "flatten: one stable key per item; data_catalog is not a table item")
    check(flat["video:v00"]["name"] == "v00" and flat["document:doc1.pdf"]["item"]["chunks"] == 3,
          "flatten: names are stable across recovery points, the item rides along")
    check(flatten({}) == {} and flatten(None) == {}, "flatten: empty and missing inventories")
    check(item_name("video", {"video_id": "x"}) == "x" and item_name("upload", {"path": "/p"}) == "/p"
          and item_name("document", {}) == "", "item names come from the kind's own field")
    check(parse_count("410\n") == 410 and parse_count(" 123 ") == 123,
          "parse_count: psql count output becomes an int")
    check(parse_count("") is None and parse_count(None) is None and parse_count("a\nb") is None,
          "parse_count: anything else is None (unknown, never a false zero)")
    check(video_rows("v00\tFiber OTE\nv01\t\n") == [{"video_id": "v00", "name": "v00", "title": "Fiber OTE"},
                                                    {"video_id": "v01", "name": "v01", "title": ""}]
          and video_rows("") == [],
          "video_rows: psql TSV becomes titled video rows")
    print("\n%d inventory self-checks passed" % n)
    return 0


def main(argv) -> int:
    if len(argv) >= 2 and argv[1] == "selftest":
        return selftest()
    if len(argv) >= 2 and argv[1] == "live":
        try:
            print(json.dumps(live_inventory()))
            return 0
        except Exception as exc:                      # noqa: BLE001
            sys.stderr.write("inventory: %s\n" % exc)
            return 1
    sys.stderr.write("usage: inventory.py live|selftest\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
