#!/usr/bin/env python3
"""Bring deleted knowledge-base documents back from a backup's Qdrant snapshot.

Runs INSIDE the gmac-haystack container (Qdrant's port is not published on
the host), fed on stdin exactly like ops/qdrant_backup.py:

    docker exec -i -e QDRANT_URL=http://qdrant:6333 gmac-haystack \
        python3 - <command> ... < /opt/gmac/ops/qdrant_docs.py

The backup's snapshot is NEVER restored over a live collection. It is loaded
into a TEMPORARY collection (name prefix restore_tmp_), documents are read or
copied out of it, and the temporary collection is dropped.

Commands (stdout = JSON data, stderr = diagnostics, non-zero exit on failure):
    load <tmp> <snapshot-path>     create collection <tmp> from a snapshot file
                                   (a path inside this container)
    sources <collection>           -> [{"name","source_type","chunks","tables"}]
                                   one row per document (payload source.filename
                                   or the older plain `source` string)
    copy <tmp> <live> <names-json> copy every point of each named document from
                                   <tmp> into <live>, vectors and payload as they
                                   were; a document that already has ANY point in
                                   <live> is skipped (live data is newer data).
                                   -> {"copied": {name: n}, "skipped": [names],
                                       "missing": [names], "tables": {name: [..]}}
    drop <tmp>                     delete a temporary collection (restore_tmp_* only)
    cleanup                        delete every leftover restore_tmp_* collection
    selftest                       offline checks
"""

import json
import os
import sys
import urllib.error
import urllib.request

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
TMP_PREFIX = "restore_tmp_"
PAGE = 256


def die(msg, code=2):
    sys.stderr.write("qdrant_docs: %s\n" % msg)
    raise SystemExit(code)


def call(method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(QDRANT_URL.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        die("%s %s -> HTTP %d %s" % (method, path, exc.code, exc.read()[:300].decode("utf-8", "replace")))
    except (urllib.error.URLError, OSError) as exc:
        die("%s %s -> %s" % (method, path, exc))


def doc_name(payload):
    """The document a point belongs to, in either payload shape."""
    src = (payload or {}).get("source")
    if isinstance(src, dict):
        return src.get("filename") or ""
    return src if isinstance(src, str) else ""


def summarize(points):
    """[{name, source_type, chunks, tables}] from scrolled points (payload only)."""
    docs = {}
    for p in points:
        payload = p.get("payload") or {}
        name = doc_name(payload)
        if not name:
            continue
        d = docs.setdefault(name, {"name": name, "source_type": payload.get("source_type", ""),
                                   "chunks": 0, "tables": []})
        d["chunks"] += 1
        table = payload.get("table_name")
        if table and table not in d["tables"]:
            d["tables"].append(table)
    return sorted(docs.values(), key=lambda d: d["name"].lower())


def scroll(collection, flt=None, with_vector=False):
    offset = None
    while True:
        body = {"limit": PAGE, "with_payload": True, "with_vector": with_vector}
        if flt:
            body["filter"] = flt
        if offset is not None:
            body["offset"] = offset
        res = call("POST", "/collections/%s/points/scroll" % collection, body)["result"]
        for p in res.get("points") or []:
            yield p
        offset = res.get("next_page_offset")
        if offset is None:
            return


def doc_filter(name):
    return {"should": [{"key": "source.filename", "match": {"value": name}},
                       {"key": "source", "match": {"value": name}}]}


def has_doc(collection, name):
    res = call("POST", "/collections/%s/points/scroll" % collection,
               {"limit": 1, "with_payload": False, "with_vector": False, "filter": doc_filter(name)})
    return bool(res["result"].get("points"))


def is_tmp(name):
    return isinstance(name, str) and name.startswith(TMP_PREFIX) and name.replace("_", "").isalnum()


def cmd_load(tmp, path):
    if not is_tmp(tmp):
        die("refusing to load into %r — only %s* collections" % (tmp, TMP_PREFIX))
    if not os.path.isfile(path):
        die("no snapshot file at %s" % path)
    import httpx                                       # in the haystack image
    with open(path, "rb") as fh:
        resp = httpx.post(QDRANT_URL.rstrip("/") + "/collections/%s/snapshots/upload" % tmp,
                          params={"priority": "snapshot", "wait": "true"},
                          files={"snapshot": (os.path.basename(path), fh)}, timeout=1800)
    if resp.status_code != 200:
        die("loading the snapshot failed: HTTP %d %s" % (resp.status_code, resp.text[:300]))
    info = call("GET", "/collections/%s" % tmp)["result"]
    print(json.dumps({"collection": tmp, "points": info.get("points_count")}))


def cmd_sources(collection):
    print(json.dumps(summarize(scroll(collection))))


def cmd_copy(tmp, live, names_json):
    if not is_tmp(tmp):
        die("copy source must be a %s* collection" % TMP_PREFIX)
    names = json.loads(names_json)          # an argument: the script itself is stdin
    if not (isinstance(names, list) and all(isinstance(n, str) and n for n in names)):
        die("names must be a JSON list of document names")
    out = {"copied": {}, "skipped": [], "missing": [], "tables": {}}
    for name in names:
        if has_doc(live, name):
            out["skipped"].append(name)
            continue
        points = list(scroll(tmp, doc_filter(name), with_vector=True))
        if not points:
            out["missing"].append(name)
            continue
        tables = sorted({(p.get("payload") or {}).get("table_name") for p in points} - {None, ""})
        for i in range(0, len(points), 64):
            batch = [{"id": p["id"], "vector": p.get("vector"), "payload": p.get("payload")}
                     for p in points[i:i + 64]]
            call("PUT", "/collections/%s/points?wait=true" % live, {"points": batch})
        out["copied"][name] = len(points)
        out["tables"][name] = tables
    print(json.dumps(out))


def cmd_drop(tmp):
    if not is_tmp(tmp):
        die("refusing to drop %r — only %s* collections" % (tmp, TMP_PREFIX))
    call("DELETE", "/collections/%s" % tmp)
    print(json.dumps({"dropped": tmp}))


def cmd_cleanup():
    names = [c["name"] for c in call("GET", "/collections")["result"]["collections"]]
    gone = [n for n in names if is_tmp(n)]
    for n in gone:
        call("DELETE", "/collections/%s" % n)
    print(json.dumps({"dropped": gone}))


def selftest():
    pts = [{"payload": {"source": {"filename": "a.csv"}, "source_type": "structured_data", "table_name": "data_a"}},
           {"payload": {"source": {"filename": "a.csv"}, "source_type": "structured_data", "table_name": "data_a2"}},
           {"payload": {"source": "old.pdf", "source_type": "document"}},
           {"payload": {"source": {"filename": "old.pdf"}, "source_type": "document"}},
           {"payload": {"content": "no source"}}]
    s = summarize(pts)
    assert [d["name"] for d in s] == ["a.csv", "old.pdf"], s
    assert s[0]["chunks"] == 2 and s[0]["tables"] == ["data_a", "data_a2"], s
    assert s[1]["chunks"] == 2, "both payload shapes count for one document"
    assert is_tmp("restore_tmp_documents_123") and not is_tmp("documents")
    assert not is_tmp("restore_tmp_../x") and not is_tmp(None)
    assert doc_filter("x.pdf")["should"][0]["key"] == "source.filename"
    print("ok qdrant_docs selftest")


def main(argv):
    if not argv:
        die("usage: load|sources|copy|drop|cleanup|selftest ...")
    cmd, args = argv[0], argv[1:]
    if cmd == "load" and len(args) == 2:
        return cmd_load(*args)
    if cmd == "sources" and len(args) == 1:
        return cmd_sources(*args)
    if cmd == "copy" and len(args) == 3:
        return cmd_copy(*args)
    if cmd == "drop" and len(args) == 1:
        return cmd_drop(*args)
    if cmd == "cleanup" and not args:
        return cmd_cleanup()
    if cmd == "selftest":
        return selftest()
    die("bad arguments for %r" % cmd)


if __name__ == "__main__":
    main(sys.argv[1:])
