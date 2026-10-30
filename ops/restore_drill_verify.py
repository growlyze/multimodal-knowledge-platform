#!/usr/bin/env python3
"""Comparisons for ops/restore-drill.sh: is what came back out of the
backup what MANIFEST.json says went in?

Pure stdlib, host-side (python3 on the VPS). Never reads a secret's value: the
restored .env is only ever hashed, the same way the backup hashed it.

    points  <MANIFEST.json> <restored_list.json> <collection>...
        restored Qdrant point counts == the counts recorded at backup time,
        for every required collection
    files   <MANIFEST.json> <restore_root>
        each backed-up source directory (uploads, Haystack state) exists in the
        restore with at least the files/bytes the manifest inventoried
    config  <MANIFEST.json> <restore_root>
        every config file restores with the sha256 recorded at backup time
        (.env included -- compared by hash only, never printed)
    selftest

Output: one "ok ..." or "FAIL ..." line per item on stdout; exit 1 if any FAIL.
"""

import hashlib
import json
import os
import sys

# Source directories older manifests list but the backup no longer holds
# (Open WebUI chat uploads, dropped 2026-10-27). Drilling an old snapshot
# against today's restore must not report them as lost.
DROPPED_SOURCES = ("/data/openwebui/uploads",)

def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _restored(root, original):
    """/data/x as restored under <root>: restic recreates absolute paths below
    the target, so /data/openwebui/uploads -> <root>/data/openwebui/uploads."""
    return os.path.join(root, original.lstrip("/"))


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_digest(restored_dir, original_dir):
    """Same digest ops/backup.sh writes for tika-config/:
    `find DIR -type f | sort | xargs sha256sum | sha256sum` -- each line is
    '<hex>  <path>', and the paths are the ORIGINAL absolute ones."""
    lines = []
    for base, _dirs, names in os.walk(restored_dir):
        for name in names:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, restored_dir).replace(os.sep, "/")
            lines.append((original_dir.rstrip("/") + "/" + rel, _sha256_file(full)))
    lines.sort()
    text = "".join("%s  %s\n" % (digest, path) for path, digest in lines)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_points(manifest, restored, required):
    out = []
    want = {c.get("name"): c.get("points_count") for c in
            (manifest.get("qdrant") or {}).get("collections") or []}
    got = {c.get("name"): c.get("points_count") for c in restored or []}
    for name in required:
        if name not in want:
            out.append((False, "points %s: not in MANIFEST.json -- the backup did not record it" % name))
        elif name not in got:
            out.append((False, "points %s: missing after restore" % name))
        elif got[name] != want[name]:
            out.append((False, "points %s: restored %s, backup recorded %s (points written during the "
                               "backup run would explain a small difference; investigate anything else)"
                        % (name, got[name], want[name])))
        else:
            out.append((True, "points %s: %s, matches the backup" % (name, got[name])))
    return out


def check_files(manifest, root):
    out = []
    for entry in manifest.get("source_files") or []:
        path, want_files, want_bytes = entry.get("path"), entry.get("files"), entry.get("bytes")
        if want_files is None:
            out.append((True, "files %s: absent at backup time -- nothing to compare" % path))
            continue
        restored = _restored(root, path)
        if not os.path.isdir(restored) and path in DROPPED_SOURCES:
            # A manifest from before 2026-10-27 lists chat uploads; a drill that
            # restores only what is backed up today must not fail on them.
            out.append((True, "files %s: no longer backed up -- not compared" % path))
            continue
        if not os.path.isdir(restored):
            out.append((False, "files %s: not in the restore" % path))
            continue
        files, size = 0, 0
        for base, _dirs, names in os.walk(restored):
            for name in names:
                files += 1
                size += os.path.getsize(os.path.join(base, name))
        # The manifest is written just before restic reads these directories, so
        # an upload arriving in between legitimately makes the restore LARGER.
        # Fewer files than inventoried is the failure that matters.
        if files < want_files:
            out.append((False, "files %s: %d restored, backup inventoried %d" % (path, files, want_files)))
        elif want_bytes is not None and size < want_bytes and files == want_files:
            out.append((False, "files %s: %d bytes restored, backup inventoried %d" % (path, size, want_bytes)))
        else:
            out.append((True, "files %s: %d files / %d bytes restored (backup inventoried %d / %s)"
                        % (path, files, size, want_files, want_bytes)))
    if not manifest.get("source_files"):
        out.append((False, "files: MANIFEST.json lists no source directories"))
    return out


def check_config(manifest, root):
    out = []
    entries = manifest.get("config") or []
    for entry in entries:
        path, want = entry.get("path"), entry.get("sha256")
        restored = _restored(root, path)
        label = os.path.basename(path.rstrip("/"))
        if os.path.isdir(restored):
            got = dir_digest(restored, path)
        elif os.path.isfile(restored):
            got = _sha256_file(restored)
        else:
            out.append((False, "config %s: not in the restore" % label))
            continue
        # Only the verdict is printed -- never content, and not even the digest
        # for .env (a hash of a short secret file is still worth not publishing).
        if got == want:
            out.append((True, "config %s: restored, checksum matches the backup" % label))
        else:
            out.append((False, "config %s: checksum differs from the backup" % label))
    if not entries:
        out.append((False, "config: MANIFEST.json lists no config files"))
    return out


def _report(results):
    bad = 0
    for ok, line in results:
        print(("ok   " if ok else "FAIL ") + line)
        bad += 0 if ok else 1
    return 1 if bad else 0


def _selftest():
    import tempfile
    n = 0

    def check(cond, label):
        nonlocal n
        assert cond, label
        n += 1
        print("ok %s" % label)

    manifest = {
        "qdrant": {"collections": [{"name": "a", "points_count": 10}, {"name": "b", "points_count": 0},
                                   {"name": "x_selftest", "points_count": 2}]},
    }
    res = check_points(manifest, [{"name": "a", "points_count": 10}, {"name": "b", "points_count": 0}], ["a", "b"])
    check(all(ok for ok, _ in res), "points: equal counts pass, including an empty collection")
    res = check_points(manifest, [{"name": "a", "points_count": 9}], ["a", "b"])
    check([ok for ok, _ in res] == [False, False], "points: a short count and a missing collection fail")
    res = check_points({"qdrant": {"collections": []}}, [{"name": "a", "points_count": 1}], ["a"])
    check(not res[0][0] and "not in MANIFEST" in res[0][1], "points: a collection the backup never recorded fails")

    with tempfile.TemporaryDirectory() as root:
        up = os.path.join(root, "data", "uploads")
        os.makedirs(os.path.join(up, "sub"))
        for name, body in (("one.csv", b"abc"), ("sub/two.pdf", b"defgh")):
            with open(os.path.join(up, name), "wb") as fh:
                fh.write(body)
        cfg = os.path.join(root, "opt", "gmac")
        os.makedirs(os.path.join(cfg, "tika-config"))
        with open(os.path.join(cfg, ".env"), "wb") as fh:
            fh.write(b"SECRET=do-not-print\n")
        with open(os.path.join(cfg, "tika-config", "tika-config.xml"), "wb") as fh:
            fh.write(b"<properties/>\n")
        env_sha = hashlib.sha256(b"SECRET=do-not-print\n").hexdigest()
        # The backup's own formula for a directory, over the ORIGINAL paths.
        xml_sha = hashlib.sha256(b"<properties/>\n").hexdigest()
        tika_sha = hashlib.sha256(("%s  /opt/gmac/tika-config/tika-config.xml\n" % xml_sha).encode()).hexdigest()

        m = {"source_files": [{"path": "/data/uploads", "bytes": 8, "files": 2},
                              {"path": "/data/gone", "bytes": None, "files": None}],
             "config": [{"path": "/opt/gmac/.env", "sha256": env_sha},
                        {"path": "/opt/gmac/tika-config", "sha256": tika_sha}]}
        check(all(ok for ok, _ in check_files(m, root)), "files: subfolders counted; an absent-at-backup dir is skipped")
        m_more = dict(m, source_files=[{"path": "/data/uploads", "bytes": 99, "files": 3}])
        check(not check_files(m_more, root)[0][0], "files: fewer files than inventoried fails")
        m_newer = dict(m, source_files=[{"path": "/data/uploads", "bytes": 2, "files": 1}])
        check(check_files(m_newer, root)[0][0], "files: an upload that arrived during the run is not a failure")
        check(not check_files(dict(m, source_files=[{"path": "/data/missing", "bytes": 1, "files": 1}]), root)[0][0],
              "files: a directory missing from the restore fails")
        m_old = dict(m, source_files=[{"path": "/data/openwebui/uploads", "bytes": 5, "files": 1}])
        check(check_files(m_old, root)[0][0], "files: an old manifest's chat-uploads dir is not required")

        res = check_config(m, root)
        check(all(ok for ok, _ in res), "config: file and tika-config dir digest match the backup's formula")
        printed = " ".join(line for _, line in res)
        check("do-not-print" not in printed and env_sha not in printed, "config: .env value and digest never printed")
        m_bad = dict(m, config=[{"path": "/opt/gmac/.env", "sha256": "0" * 64},
                                {"path": "/opt/gmac/Caddyfile", "sha256": "1" * 64}])
        check([ok for ok, _ in check_config(m_bad, root)] == [False, False],
              "config: a changed file and a missing file both fail")
        check(not check_config({"config": []}, root)[0][0], "config: an empty config list fails")
    print("\n%d restore-drill verify self-checks passed" % n)
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "selftest":
        return _selftest()
    if len(argv) >= 4 and argv[1] == "points":
        return _report(check_points(_load(argv[2]), _load(argv[3]), argv[4:]))
    if len(argv) == 4 and argv[1] == "files":
        return _report(check_files(_load(argv[2]), argv[3]))
    if len(argv) == 4 and argv[1] == "config":
        return _report(check_config(_load(argv[2]), argv[3]))
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
