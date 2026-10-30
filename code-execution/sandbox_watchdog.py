#!/usr/bin/env python3
"""Kill sandbox processes that have outlived any legitimate use.

Why this exists (P45, P68): nothing else in this stack kills a process.

  * `OPEN_TERMINAL_EXECUTE_TIMEOUT` sounds like an execution cap and is not -
    it only sets the default for the `wait` query parameter, i.e. how long an
    HTTP request blocks before returning. The process keeps running after the
    request gives up. Confirmed by reading Open Terminal 0.13.0's source and
    live: a `while True: pass` was still burning a core at t+770s.
  * Client-side execution valves bound only calls made through their own
    tool (P47). The browser's built-in Terminal path - which is what real
    chat turns actually use - has no equivalent.
  * The UI's Stop button cancels the REQUEST, not the process. That is why
    P68's `help()` hang survived it for 30+ minutes.
  * Earlier setup docs described a nightly `docker restart` cron as the de facto
    ceiling. It was never installed, and named two containers that do not
    exist (verified 2026-09-24).

So this is the only thing standing between a stuck process and a human
noticing. Two real hang classes are already on record and neither needs
pathological code: a CPU-bound loop (P45) and a process blocked on stdin that
no amount of waiting will free (P68 - Gemini called `help()`, Python's pager
asked for a keypress, and nobody was there to press one).

Runs INSIDE the sandbox container so it can reach the API on localhost and
read `OPEN_TERMINAL_API_KEY` from its own environment - the key is never
passed on a command line, never written to disk, and never leaves the
container. Same pattern as tests/_sandbox_driver.py.

Usage (from the VPS host):
    docker exec ${STACK_NAME}-code-sandbox-1 python3 /opt/pylib/sandbox_watchdog.py
    docker exec ${STACK_NAME}-code-sandbox-1 python3 /opt/pylib/sandbox_watchdog.py --dry-run

Tuning: SANDBOX_WATCHDOG_MAX_AGE (seconds, default 900). Deliberately well
above any client-side 600s execution deadline so this stays a backstop and
does not race the client for the same process - if this ever kills
something, either the caller's own deadline failed or the request never
went through it.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

BASE_URL = os.environ.get("SANDBOX_WATCHDOG_URL", "http://localhost:8000")
MAX_AGE_SECONDS = int(os.environ.get("SANDBOX_WATCHDOG_MAX_AGE", "900"))


def _request(method, path, timeout=15):
    key = os.environ.get("OPEN_TERMINAL_API_KEY", "")
    req = urllib.request.Request(BASE_URL + path, method=method)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode(errors="replace")
    return json.loads(body) if body.strip() else {}


def age_seconds(process_id, now):
    """Seconds since the process started, derived from its id.

    Open Terminal ids look like `20260924-110546-aaa813` - a UTC timestamp
    plus a random suffix. The list endpoint returns no start_time field at
    all, so the id is the only age signal available without fetching each
    process's log file. Verified UTC by launching a process at a known clock
    time and comparing (2026-09-24).

    Returns None for an id that does not parse, and callers MUST treat that
    as "leave it alone" - killing something because its name looked odd is a
    worse failure than missing one runaway.
    """
    parts = process_id.split("-")
    if len(parts) < 2:
        return None
    try:
        started = datetime.strptime("-".join(parts[:2]), "%Y%m%d-%H%M%S")
    except ValueError:
        return None
    return (now - started.replace(tzinfo=timezone.utc)).total_seconds()


def main(argv):
    dry_run = "--dry-run" in argv
    now = datetime.now(timezone.utc)

    try:
        processes = _request("GET", "/execute")
    except Exception as exc:                      # noqa: BLE001 - see below
        # A watchdog that dies quietly is indistinguishable from a watchdog
        # with nothing to do, which is the exact failure this whole file
        # exists to prevent. Exit non-zero so cron surfaces it.
        print(f"sandbox-watchdog: ERROR listing processes: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if not isinstance(processes, list):
        print(f"sandbox-watchdog: ERROR unexpected list response: "
              f"{str(processes)[:200]}", file=sys.stderr)
        return 2

    failures = 0
    for proc in processes:
        if proc.get("status") != "running":
            continue
        pid = proc.get("id") or ""
        age = age_seconds(pid, now)
        if age is None or age <= MAX_AGE_SECONDS:
            continue

        command = (proc.get("command") or "")[:160]
        if dry_run:
            print(f"sandbox-watchdog: WOULD KILL {pid} age={int(age)}s "
                  f"limit={MAX_AGE_SECONDS}s command={command!r}")
            continue

        try:
            _request("DELETE", f"/execute/{pid}?force=true")
        except Exception as exc:                  # noqa: BLE001
            failures += 1
            print(f"sandbox-watchdog: FAILED to kill {pid} age={int(age)}s: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        # Loud on purpose. Every kill is either a real runaway worth
        # investigating or this threshold being wrong, and both need to be
        # visible to a human - P15's lesson was that a message reaching only
        # a container log reaches nobody.
        print(f"sandbox-watchdog: KILLED {pid} age={int(age)}s "
              f"limit={MAX_AGE_SECONDS}s command={command!r}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
