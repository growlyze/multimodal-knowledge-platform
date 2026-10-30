#!/usr/bin/env python3
"""Apply the dashboard's Google Drive sign-in to the backup's rclone remote.

The panel's "Backup drive" page (the dashboard Integrations page) can connect the
backup drive again by signing in with Google; it saves the result as
/data/backup-drives/primary.json  {"token", "root_folder_id", "account", ...}.

Before each backup, ops/backup_drive.sh runs

    python3 ops/backup_drives.py build --base /root/.config/rclone/rclone.conf \
        --override /data/backup-drives/primary.json --remote gdrive --out <tmpfile>

which writes a COPY of the server's rclone config with [gdrive]'s `token` and
`root_folder_id` replaced by the panel's. Everything else in the section (type,
scope, OAuth client) is kept. The repository name does not change
(rclone:gdrive:backups). The server's own rclone.conf is never modified.

Prints `override <account>` or `none` (no usable panel sign-in -> the caller
uses the server's config as it is). The output file holds tokens: 0600, the
caller deletes it.

Commands: build ..., selftest. Stdlib only (host python3).
"""

import json
import os
import re
import sys


def load_override(path):
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ValueError("override %s is unreadable: %s" % (path, type(exc).__name__))
    if not (rec.get("token") or {}).get("refresh_token") or not rec.get("root_folder_id"):
        raise ValueError("override %s has no refresh token or folder" % path)
    for value in (rec["root_folder_id"], json.dumps(rec["token"]),
                  rec.get("client_id", ""), rec.get("client_secret", "")):
        if "\n" in value or "\r" in value:
            raise ValueError("override %s contains a multi-line value" % path)
    return rec


def apply_override(base_text, remote, rec):
    """base rclone.conf text -> same text with [remote]'s token/root_folder_id
    replaced by the panel's (and its OAuth client too when the record names
    one: a web-client sign-in must be refreshed as the client that issued it)."""
    out, current, seen, done = [], None, False, set()
    token = json.dumps(rec["token"], separators=(",", ":"))
    lines = base_text.splitlines()
    for i, raw in enumerate(lines):
        m = re.match(r"^\s*\[(.+)\]\s*$", raw)
        if m:
            if current == remote:
                out[len(out):] = _missing(done, token, rec)
            current = m.group(1).strip()
            seen = seen or current == remote
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
        out[len(out):] = _missing(done, token, rec)
    if not seen:
        raise ValueError("remote [%s] is not in the rclone config" % remote)
    return "\n".join(out) + "\n"


def _missing(done, token, rec):
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


def cmd_build(argv):
    args = dict(zip(argv[::2], argv[1::2]))
    for key in ("--base", "--override", "--remote", "--out"):
        if key not in args:
            sys.stderr.write("missing %s\n" % key)
            return 2
    rec = load_override(args["--override"])
    if rec is None:
        print("none")
        return 0
    with open(args["--base"], encoding="utf-8") as fh:
        text = apply_override(fh.read(), args["--remote"], rec)
    fd = os.open(args["--out"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("override %s" % (rec.get("account") or "panel sign-in"))
    return 0


def _parse(text):
    sections, current = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            current = m.group(1); sections[current] = {}
        elif current and "=" in line:
            k, v = line.split("=", 1); sections[current][k.strip()] = v.strip()
    return sections


def _selftest():
    import contextlib
    import io
    import tempfile
    n = 0

    def check(cond, label):
        nonlocal n
        assert cond, label
        n += 1
        print("ok %s" % label)

    base = ("[gdrive]\ntype = drive\nclient_id = CID\nclient_secret = CS\nscope = drive\n"
            "root_folder_id = OLDFOLDER\ntoken = {\"refresh_token\":\"old\"}\n\n[other]\ntype = local\n")
    rec = {"token": {"access_token": "a", "refresh_token": "new"}, "root_folder_id": "NEWFOLDER",
           "account": "[EMAIL]"}
    sec = _parse(apply_override(base, "gdrive", rec))
    check(json.loads(sec["gdrive"]["token"])["refresh_token"] == "new", "the panel's token replaces the server's")
    check(sec["gdrive"]["root_folder_id"] == "NEWFOLDER", "the panel's folder replaces the server's")
    check(sec["gdrive"]["client_id"] == "CID" and sec["gdrive"]["scope"] == "drive", "client and scope kept")
    check(sec["other"] == {"type": "local"}, "other remotes untouched")
    rec_web = dict(rec, client_id="WEBID", client_secret="WEBSEC")
    sec = _parse(apply_override(base, "gdrive", rec_web))
    check(sec["gdrive"]["client_id"] == "WEBID" and sec["gdrive"]["client_secret"] == "WEBSEC",
          "the panel's web client replaces the remote's own")
    sec = _parse(apply_override("[gdrive]\ntype = drive\n", "gdrive", rec_web))
    check(sec["gdrive"]["client_id"] == "WEBID" and sec["gdrive"]["client_secret"] == "WEBSEC",
          "a section without a client gets the panel's added")
    sec = _parse(apply_override("[gdrive]\ntype = drive\nclient_id = C\nclient_secret = S\n", "gdrive", rec))
    check(sec["gdrive"]["root_folder_id"] == "NEWFOLDER" and "new" in sec["gdrive"]["token"],
          "a section without token/folder gets both added")
    try:
        apply_override("[other]\ntype = local\n", "gdrive", rec)
    except ValueError:
        check(True, "a config without the remote is refused")

    with tempfile.TemporaryDirectory() as d:
        bp, op, out = (os.path.join(d, f) for f in ("rclone.conf", "primary.json", "out.conf"))
        with open(bp, "w") as fh:
            fh.write(base)

        def build():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cmd_build(["--base", bp, "--override", op, "--remote", "gdrive", "--out", out])
            return rc, buf.getvalue().strip()

        check(build() == (0, "none") and not os.path.exists(out), "no panel sign-in -> 'none', nothing written")
        with open(op, "w") as fh:
            json.dump(rec, fh)
        check(build() == (0, "override [EMAIL]"), "a panel sign-in -> override")
        if os.name == "posix":
            check(oct(os.stat(out).st_mode & 0o777) == "0o600", "the generated config is 0600")
        with open(bp) as fh:
            check(fh.read() == base, "the server's own rclone.conf is never modified")
        for bad in ('{"token": {}, "root_folder_id": "F"}', "{broken",
                    json.dumps(dict(rec, root_folder_id="F\nevil = 1")),
                    json.dumps(dict(rec, client_secret="S\nevil = 1"))):
            with open(op, "w") as fh:
                fh.write(bad)
            try:
                cmd_build(["--base", bp, "--override", op, "--remote", "gdrive", "--out", out + "x"])
            except ValueError:
                pass
            else:
                raise AssertionError("bad override accepted: %r" % bad[:30])
        check(not os.path.exists(out + "x"), "an unusable or injecting override is refused, nothing written")
    print("\n%d backup-drive self-checks passed" % n)
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "selftest":
        return _selftest()
    if len(argv) >= 2 and argv[1] == "build":
        try:
            return cmd_build(argv[2:])
        except (OSError, ValueError) as exc:
            sys.stderr.write("backup_drives: %s\n" % exc)
            return 1
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
