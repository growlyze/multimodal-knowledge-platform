# F6b (2026-10-25): offline tests for patch_open_terminal.py's time limits,
# run against a real, patched copy of Open Terminal's own source.
#
# Why these exist: Open Terminal 0.14.0 has no real time limit anywhere. `wait`
# (and OPEN_TERMINAL_EXECUTE_TIMEOUT) only bounds how long the HTTP call blocks
# - the command keeps running - and /files/grep, /files/matches, /files/search
# and /files/glob take no timeout at all. File-search tools were observed to
# run ~10 minutes on large directories. The patch adds a cooperative deadline
# to every walk and a hard kill cap to /execute; these tests drive the patched
# app.
#
# Needs Linux (the PTY runner) and open-terminal==0.14.0 importable - the
# version inside the pinned code-sandbox base image. Skipped otherwise, so a
# Windows `pytest` stays green. Run from WSL/Linux with the package on the path:
#   pip install --target /some/dir open-terminal==0.14.0 pytest
#   PYTHONPATH=/some/dir python3 -m pytest code-execution/tests/test_patch_open_terminal.py -v
# PATCH_OPEN_TERMINAL_SCRIPT=<path> runs them against another copy of the patch
# script (how the RED run against the pre-F6b script was done).

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
PATCH_SCRIPT = os.environ.get(
    "PATCH_OPEN_TERMINAL_SCRIPT", os.path.join(os.path.dirname(_HERE), "patch_open_terminal.py"))
OT_VERSION = "0.14.0"    # what the Dockerfile's pinned :slim digest ships


def _open_terminal_dir():
    if not sys.platform.startswith("linux"):
        return None
    spec = importlib.util.find_spec("open_terminal")
    if spec is None or not spec.submodule_search_locations:
        return None
    try:
        if importlib.metadata.version("open-terminal") != OT_VERSION:
            return None
    except importlib.metadata.PackageNotFoundError:
        return None
    return list(spec.submodule_search_locations)[0]


pytestmark = pytest.mark.skipif(
    _open_terminal_dir() is None,
    reason="needs Linux + open-terminal==%s importable (see module docstring)" % OT_VERSION)


@pytest.fixture(scope="module")
def patched(tmp_path_factory):
    """A patched copy of the installed open_terminal package; returns its root."""
    root = tmp_path_factory.mktemp("ot")
    pkg = root / "open_terminal"
    shutil.copytree(_open_terminal_dir(), pkg, ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ, PATCH_OPEN_TERMINAL_MAIN_PY=str(pkg / "main.py"))
    out = subprocess.run([sys.executable, PATCH_SCRIPT], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return root


# Each scenario runs in its own interpreter: the limits are read from the
# environment at import time, and os.walk is monkeypatched for the slow cases.
_DRIVER = textwrap.dedent(r'''
    import json, os, sys, time
    mode, data_dir = sys.argv[1], sys.argv[2]
    if mode == "endless_walk":
        # A walk that never ends on its own - the shape of grep_search("/") on a
        # real filesystem, minus the 10 minutes. 50 ms per directory.
        def endless_walk(top, *a, **k):
            i = 0
            while True:
                time.sleep(0.05)
                i += 1
                yield (os.path.join(top, "d%d" % i), [], ["f%d.txt" % i])
        os.walk = endless_walk
    import open_terminal.main as m
    from fastapi.testclient import TestClient
    H = {"Authorization": "Bearer k"}
    out = {}
    with TestClient(m.app) as c:
        if mode in ("endless_walk", "normal"):
            calls = {
                "grep": ("/files/grep", {"query": "needle", "path": data_dir}),
                "glob": ("/files/glob", {"pattern": "*.txt", "path": data_dir}),
                "search": ("/files/search", {"query": "needle", "path": data_dir}),
                "matches": ("/files/matches", {"query": "needle", "path": data_dir}),
            }
            for name, (url, params) in calls.items():
                t = time.monotonic()
                r = c.get(url, params=params, headers=H)
                out[name] = {"status": r.status_code, "secs": time.monotonic() - t, "body": r.json()}
        elif mode == "cap":
            t = time.monotonic()
            r = c.post("/execute", params={"wait": 10}, json={"command": "sleep 61.25"}, headers=H)
            out["long"] = {"secs": time.monotonic() - t, "body": r.json()}
            pid = out["long"]["body"]["id"]
            out["long_status"] = c.get("/execute/%s/status" % pid, headers=H).json()
            # outlives its wait: wait=0 returns at once, the cap still kills it
            r = c.post("/execute", params={"wait": 0}, json={"command": "sleep 62.5"}, headers=H)
            out["nowait"] = r.json()
            t = time.monotonic()
            r = c.get("/execute/%s/status" % out["nowait"]["id"], params={"wait": 10}, headers=H)
            out["nowait_status"] = {"secs": time.monotonic() - t, "body": r.json()}
            r = c.post("/execute", params={"wait": 5}, json={"command": "echo hi"}, headers=H)
            out["quick"] = r.json()
            time.sleep(float(os.environ["SANDBOX_EXECUTE_HARD_CAP_S"]) + 0.5)
            out["quick_after_cap"] = c.get("/execute/%s/status" % out["quick"]["id"], headers=H).json()
            ps = os.popen("ps -eo args").read()
            out["survivors"] = [l for l in ps.splitlines() if l.startswith("sleep 6")]
    print("RESULT" + json.dumps(out))
''')


def _drive(patched, tmp_path, mode, **limits):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "needle.txt").write_text("a line with the needle in it\nanother line\n")
    (data / "sub").mkdir(exist_ok=True)
    (data / "sub" / "other.txt").write_text("nothing here\nneedle again\n")
    env = dict(os.environ, HOME=str(home), OPEN_TERMINAL_API_KEY="k",
               PYTHONPATH=os.pathsep.join([str(patched)] + sys.path))
    env.pop("OPEN_TERMINAL_EXECUTE_TIMEOUT", None)
    env.update({k: str(v) for k, v in limits.items()})
    script = tmp_path / "driver.py"
    script.write_text(_DRIVER)
    run = subprocess.run([sys.executable, str(script), mode, str(data)], env=env,
                         capture_output=True, text=True, timeout=90)
    line = [l for l in run.stdout.splitlines() if l.startswith("RESULT")]
    assert line, run.stdout[-2000:] + run.stderr[-3000:]
    return json.loads(line[0][len("RESULT"):])


def test_patch_applies_and_counts_its_blocks(patched):
    main = (patched / "open_terminal" / "main.py").read_text()
    assert main.count("_sandbox_deadline = _SandboxDeadline()") == 4     # grep/matches/search/glob
    assert main.count("**_sandbox_cap_note(background_process)") == 2  # execute + get_status
    assert "_sandbox_arm_hard_cap(background_process)\n" in main


def test_patch_refuses_to_apply_twice(patched, tmp_path):
    # The exact-match guard: a reshaped (here: already patched) source fails the
    # build loudly instead of shipping with the limits silently missing.
    copy = tmp_path / "twice" / "open_terminal"
    shutil.copytree(patched / "open_terminal", copy)
    env = dict(os.environ, PATCH_OPEN_TERMINAL_MAIN_PY=str(copy / "main.py"))
    out = subprocess.run([sys.executable, PATCH_SCRIPT], env=env, capture_output=True, text=True)
    assert out.returncode == 1 and "FATAL" in out.stderr


def test_endless_walks_stop_at_the_deadline_with_a_note(patched, tmp_path):
    res = _drive(patched, tmp_path, "endless_walk", SANDBOX_FILE_SEARCH_DEADLINE_S=1)
    for name in ("grep", "glob", "search", "matches"):
        r = res[name]
        assert r["status"] == 200, (name, r)
        assert r["secs"] < 6, (name, r["secs"])            # was: never returns
        body = r["body"]
        assert body.get("stopped_early") is True, (name, body)
        assert "stopped after 1 s" in body["note"] and "narrow the path" in body["note"].lower()
        if "truncated" in body:
            assert body["truncated"] is True, (name, body)


def test_normal_searches_are_unchanged(patched, tmp_path):
    res = _drive(patched, tmp_path, "normal", SANDBOX_FILE_SEARCH_DEADLINE_S=20)
    for name in ("grep", "glob", "search", "matches"):
        body = res[name]["body"]
        assert res[name]["status"] == 200, (name, body)
        assert "note" not in body and "stopped_early" not in body, (name, body)
    grep = res["grep"]["body"]
    assert grep["truncated"] is False
    assert sorted((os.path.basename(m["file"]), m["line"]) for m in grep["matches"]) == [
        ("needle.txt", 1), ("other.txt", 2)]
    assert sorted(m["path"] for m in res["glob"]["body"]["matches"]) == ["needle.txt", "sub/other.txt"]
    assert [r["name"] for r in res["search"]["body"]["results"]] == ["needle.txt"]
    names = sorted(r["name"] for r in res["matches"]["body"]["results"])
    assert names == ["needle.txt", "other.txt"]


def test_execute_hard_cap_kills_what_outlives_it(patched, tmp_path):
    res = _drive(patched, tmp_path, "cap", SANDBOX_EXECUTE_HARD_CAP_S=2)
    long = res["long"]
    assert long["body"]["status"] == "killed", long        # was: "running" after 10 s
    assert long["secs"] < 6, long["secs"]
    assert "2 s hard limit" in long["body"]["note"]
    assert res["long_status"]["status"] == "killed" and "hard limit" in res["long_status"]["note"]
    assert res["nowait"]["status"] == "running"             # wait=0 returns at once...
    nw = res["nowait_status"]
    assert nw["body"]["status"] == "killed" and nw["secs"] < 6, nw   # ...the cap still kills it
    assert res["quick"]["status"] == "done" and "note" not in res["quick"]
    assert res["quick_after_cap"]["status"] == "done" and "note" not in res["quick_after_cap"]
    assert res["survivors"] == [], res["survivors"]          # the process group is gone
