#!/usr/bin/env python3
"""Qdrant snapshot helper for ops/backup.sh (README.md).

Runs INSIDE the gmac-haystack container, reached over gmac-network — Qdrant's
port is NOT published on the host, which is exactly why the retired M3 script
(`haystack-api/scripts/backup_to_gdrive.sh`) silently backed up zero vectors:
it curled http://localhost:6333 and swallowed the failure behind `|| true`.

The backup script feeds this file to the container's python on stdin:

    docker exec -i -e QDRANT_URL=http://qdrant:6333 gmac-haystack \
        python3 - snap documents < /opt/gmac/ops/qdrant_backup.py

so the file itself never has to be baked into an image or bind-mounted.
Commands (stdout is data, stderr is diagnostics — never the reverse):

    info                          -> JSON {"version": ..., "title": ...}
    list                          -> JSON [{"name", "points_count", "status"}]
    snap <collection>             -> one line: name<TAB>size<TAB>checksum
    get <collection> <name>       -> snapshot bytes, streamed to stdout
    restore <collection> <path> [sha256]
                              -> upload snapshot file <path> (inside this
                                 container) and recover the collection; a
                                 sha256 is passed to Qdrant's checksum check;
                                 used by the restore runbook only
    delete <collection> <name>
                              -> delete snapshot <name> from the server AFTER
                                 the caller verified its download; stops every
                                 nightly run leaving ~250 MB behind in
                                 /data/qdrant/snapshots
    selftest                      -> offline fixture checks, no network

Every command exits non-zero on failure: the backup script treats a missing or
unreadable collection as a failed backup, never as "skipped".
"""

import http.client
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")


def die(msg, code=2):
    sys.stderr.write("qdrant_backup: %s\n" % msg)
    raise SystemExit(code)


def _url(path):
    return QDRANT_URL.rstrip("/") + path


def get_json(path):
    """GET a Qdrant path, return parsed JSON. Never raises urllib traces."""
    try:
        with urllib.request.urlopen(_url(path), timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        die("GET %s -> HTTP %d: %s" % (path, exc.code, exc.read().decode("utf-8", "replace")[:200]))
    except (urllib.error.URLError, OSError) as exc:
        die("GET %s -> %s" % (path, exc))


def snapshot_of(result):
    """Snapshot description from a snapshot response.

    Accept both shapes on purpose: v1.19's `POST /collections/{c}/snapshots`
    returns `result` as a single SnapshotDescription object, while list-style
    endpoints (`GET .../snapshots`, shard snapshots) wrap them in an array.
    Take the first entry of a list; a shape with no "name" is unusable.
    """
    if isinstance(result, list):
        if not result:
            return None
        result = result[0]
    if not isinstance(result, dict) or not result.get("name"):
        return None
    return result


def cmd_info():
    root = get_json("/")
    print(json.dumps({"version": root.get("version"), "title": root.get("title")}))
    return 0


def cmd_list():
    collections = get_json("/collections").get("result", {}).get("collections") or []
    out = []
    for coll in collections:
        name = coll.get("name")
        detail = get_json("/collections/%s" % name).get("result", {})
        out.append({"name": name,
                    "points_count": detail.get("points_count"),
                    "status": detail.get("status")})
    out.sort(key=lambda c: c["name"] or "")
    print(json.dumps(out))
    return 0


def snap_line(snap):
    """name<TAB>size<TAB>checksum for the bash caller (no JSON tool needed there).

    Refuses a description without a size: the caller verifies the downloaded
    file against it, and an unverifiable download must fail the backup rather
    than slip through. A missing checksum is an empty third field (older
    servers omit it).
    """
    if snap.get("size") in (None, ""):
        die("snapshot '%s' has no size to verify the download against" % snap.get("name"))
    return "%s\t%s\t%s" % (snap.get("name"), snap.get("size"), snap.get("checksum") or "")


def _refuse_selftest(collection):
    """The two haystack_*_selftest collections are system-check fixtures —
    rebuildable in seconds, never worth a backup slot. The backup script's
    collection list already omits them; this guard is defense in depth so no
    future caller can snapshot one by accident."""
    if "selftest" in collection:
        die("collection '%s' is a selftest collection — not backed up" % collection)


def cmd_snap(collection):
    _refuse_selftest(collection)
    try:
        # wait=true is the API default, but the download step depends on the
        # snapshot existing immediately after this returns — make it explicit.
        req = urllib.request.Request(_url("/collections/%s/snapshots?wait=true" % collection),
                                     data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        die("create snapshot for '%s' -> HTTP %d: %s"
            % (collection, exc.code, exc.read().decode("utf-8", "replace")[:200]))
    except (urllib.error.URLError, OSError) as exc:
        die("create snapshot for '%s' -> %s" % (collection, exc))
    snap = snapshot_of(body.get("result"))
    if not snap:
        die("create snapshot for '%s' returned no snapshot description" % collection)
    print(snap_line(snap))
    return 0


def cmd_get(collection, name):
    _refuse_selftest(collection)
    try:
        with urllib.request.urlopen(_url("/collections/%s/snapshots/%s" % (collection, name)),
                                   timeout=1800) as resp:
            out = sys.stdout.buffer
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
            out.flush()
    except urllib.error.HTTPError as exc:
        die("download %s/%s -> HTTP %d" % (collection, name, exc.code))
    except (urllib.error.URLError, OSError) as exc:
        die("download %s/%s -> %s" % (collection, name, exc))
    return 0


def cmd_restore(collection, path, checksum=""):
    """POST the snapshot file as multipart/form-data and recover the collection.

    Streams the file in chunks (snapshots can be hundreds of MB), so this needs
    no memory beyond one chunk. `checksum` (optional SHA256 from the backup
    manifest) is passed as the upload API's checksum query parameter, so Qdrant
    itself verifies the snapshot before recovering. Used only by the documented
    restore procedure.
    """
    if not os.path.isfile(path):
        die("restore: no such file inside the container: %s" % path)
    parts = urllib.parse.urlsplit(QDRANT_URL)
    host, port = parts.hostname or "qdrant", parts.port or 80
    boundary = "----gmacrestore%d" % os.getpid()
    head = ("--%s\r\nContent-Disposition: form-data; name=\"snapshot\"; "
            "filename=\"%s\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            % (boundary, os.path.basename(path))).encode()
    tail = ("\r\n--%s--\r\n" % boundary).encode()
    size = os.path.getsize(path)
    query = "?wait=true&priority=snapshot"
    if checksum:
        query += "&checksum=" + urllib.parse.quote(checksum)
    conn = http.client.HTTPConnection(host, port, timeout=1800)
    try:
        # putrequest/putheader (not request()) because the body is streamed in
        # chunks below: request() would need the whole file in memory to set
        # Content-Length itself.
        conn.putrequest("POST", "/collections/%s/snapshots/upload%s" % (collection, query))
        conn.putheader("Content-Type", "multipart/form-data; boundary=%s" % boundary)
        conn.putheader("Content-Length", str(len(head) + size + len(tail)))
        conn.endheaders()
        conn.send(head)
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                conn.send(chunk)
        conn.send(tail)
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", "replace")[:200]
        if resp.status >= 400:
            die("restore '%s' from %s -> HTTP %d: %s" % (collection, path, resp.status, body))
        print(json.dumps({"collection": collection, "restored": True,
                          "size": size, "response": body[:100]}))
        return 0
    except OSError as exc:
        die("restore '%s' from %s -> %s" % (collection, path, exc))
    finally:
        conn.close()


def cmd_delete(collection, name):
    """Delete a server-side snapshot (Qdrant keeps created snapshots under its
    storage dir until removed — see delete_snapshot in the v1.19.0 spec).

    Only ever called for snapshots this backup run created and has already
    downloaded + verified. Not a destructive data operation: it removes a
    temporary copy, never the collection itself."""
    try:
        req = urllib.request.Request(
            _url("/collections/%s/snapshots/%s" % (collection, name)),
            method="DELETE")
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8", "replace")[:100]
    except urllib.error.HTTPError as exc:
        die("delete snapshot %s/%s -> HTTP %d: %s"
            % (collection, name, exc.code, exc.read().decode("utf-8", "replace")[:200]))
    except (urllib.error.URLError, OSError) as exc:
        die("delete snapshot %s/%s -> %s" % (collection, name, exc))
    print(json.dumps({"collection": collection, "deleted": True, "response": body}))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────

def _selftest():
    checks = 0

    def ok(name):
        nonlocal checks
        checks += 1
        print("ok %s" % name)

    # The exact bug in the retired M3 script: list-shaped result.
    snap = snapshot_of([{"name": "documents-2026-10-22.snap", "size": 123, "checksum": "ab"}])
    assert snap["name"] == "documents-2026-10-22.snap", snap
    ok("snapshot_of_list")

    snap = snapshot_of({"name": "one.snap", "size": 1})
    assert snap["name"] == "one.snap", snap
    ok("snapshot_of_dict")

    assert snapshot_of([]) is None
    assert snapshot_of({"size": 1}) is None
    assert snapshot_of("junk") is None
    ok("snapshot_of_bad_shapes")

    assert _url("/collections") == QDRANT_URL.rstrip("/") + "/collections"
    ok("url_join")

    # Selftest collections are never backed up (guard + the script's list).
    for bad in ("haystack_extract_selftest", "haystack_search_selftest"):
        try:
            _refuse_selftest(bad)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("selftest collection %s was allowed" % bad)
    _refuse_selftest("documents")   # real collections pass through
    ok("selftest_collections_refused")

    # The snap output format the bash side parses: name<TAB>size<TAB>checksum.
    line = snap_line({"name": "a.snap", "size": 123, "checksum": "ab"})
    name, size, checksum = line.split("\t")
    assert (name, size, checksum) == ("a.snap", "123", "ab"), line
    ok("snap_line_format")

    # A missing checksum is an empty third field, still TAB-parseable.
    parts = snap_line({"name": "a.snap", "size": 123}).split("\t")
    assert parts == ["a.snap", "123", ""], parts
    ok("snap_line_empty_checksum")

    # No size -> the download could not be verified -> refuse (SystemExit 2),
    # never emit a line the caller would trust.
    try:
        snap_line({"name": "a.snap"})
    except SystemExit as exc:
        assert exc.code == 2
        ok("snap_line_refuses_missing_size")
    else:
        raise AssertionError("snap_line accepted a size-less description")

    # delete is wired to the documented endpoint shape and its dispatch.
    import inspect
    src = inspect.getsource(cmd_delete)
    assert 'snapshots/%s' in src and 'DELETE' in src, src
    for bad in (["x", "delete"], ["x", "delete", "c", "n", "extra"]):
        try:
            main(bad)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("delete dispatch accepted bad args: %r" % (bad,))
    ok("delete_dispatch_requires_collection_and_name")

    print("\n%d qdrant helper self-checks passed" % checks)
    return 0


def main(argv):
    if len(argv) < 2:
        die("usage: qdrant_backup.py info|list|snap|get|restore|selftest [...]")
    cmd, args = argv[1], argv[2:]
    if cmd == "info":
        return cmd_info()
    if cmd == "list":
        return cmd_list()
    if cmd == "snap" and len(args) == 1:
        return cmd_snap(args[0])
    if cmd == "get" and len(args) == 2:
        return cmd_get(args[0], args[1])
    if cmd == "restore" and len(args) in (2, 3):
        return cmd_restore(args[0], args[1], args[2] if len(args) == 3 else "")
    if cmd == "delete" and len(args) == 2:
        return cmd_delete(args[0], args[1])
    if cmd == "selftest":
        return _selftest()
    die("unknown command or wrong arguments: %s" % " ".join(argv[1:]))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
