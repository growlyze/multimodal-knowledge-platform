# Phase 1 of the reliability-verification plan: runtime, execution, timeout
# and cancellation reliability of the code-execution sandbox (P43).
#
# Two tiers of coverage:
#
#   * Offline (always run, no network) - the sandbox_watchdog.py age
#     parsing and threshold rules, because getting them wrong in either
#     direction is costly.
#
#   * Live (skipped automatically unless `ssh` + `docker exec` against the
#     deployed sandbox container actually work from this machine - no API key
#     needed, matching tests/live/conftest.py's OPENWEBUI_API_KEY pattern
#     but gated on SSH/docker reachability instead) - drive the real
#     container the way this session verified the fix. The infinite-loop
#     timeout case additionally requires SANDBOX_LIVE_SLOW_TESTS=1: it runs
#     a real ~600s CPU-bound busy loop against the production sandbox, and
#     would make every default `pytest` invocation take 10+ minutes
#     otherwise. Run explicitly:
#       SANDBOX_LIVE_SLOW_TESTS=1 python -m pytest \
#           code-execution/tests/test_sandbox_runtime.py -k slow -v
#
# Never prints OPEN_TERMINAL_API_KEY: the live helpers read it from the
# container's own environment inside a single `docker exec`, so the value
# never crosses back into this process or its output.

import json
import os
import shlex
import subprocess
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_EXEC_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_CODE_EXEC_ROOT)


# --------------------------------------------------------------------------
# Live: talk to the real sandbox container over ssh + docker exec, via
# the stdlib-only _sandbox_driver.py pushed into the container. No API key
# ever touches this process - the driver reads OPEN_TERMINAL_API_KEY from
# the container's own environment.
# --------------------------------------------------------------------------

_STACK = os.environ.get("STACK_NAME", "gmac")
_CONTAINER = os.environ.get("SANDBOX_CONTAINER", f"{_STACK}-code-sandbox-1")
_SSH_HOST = os.environ.get("SANDBOX_SSH_HOST", _STACK)
_DRIVER_SRC = os.path.join(_TESTS_DIR, "_sandbox_driver.py")
_DRIVER_DST = "/tmp/_sandbox_driver.py"


def _ssh(args, timeout=30, input_bytes=None):
    """Run argv on the VPS, preserving argument boundaries.

    ssh does NOT take an argv the way subprocess does: it joins everything
    after the host into ONE string and hands that to the REMOTE login shell,
    throwing away Python's own argument boundaries. So passing
    ["docker", "exec", "-i", C, "sh", "-c", "cat > /tmp/x"] arrived remotely as

        docker exec -i C sh -c cat > /tmp/x

    - the redirect belonged to the VPS's shell, not the container's, so the
    file was written to the HOST's /tmp and the container never received it.
    _push_driver() did exactly this, silently, returncode 0 throughout.

    It went unnoticed because the sandbox container is long-lived: a copy of
    the driver left over from an earlier manual push kept the live tests
    green, and they only broke when the container was rebuilt for a
    data_tools deploy - i.e. the tests were passing on leftover state, not
    on anything _push_driver did. Same shape as P35 and P47: a green result
    that was never evidence of the thing it claimed.

    shlex.quote per argument makes the remote shell reconstruct the argv we
    actually meant.
    """
    remote = " ".join(shlex.quote(a) for a in args)
    return subprocess.run(
        ["ssh", _SSH_HOST, remote], input=input_bytes,
        capture_output=True, timeout=timeout)


def _sandbox_reachable():
    try:
        r = _ssh(["docker", "exec", _CONTAINER, "true"], timeout=15)
        return r.returncode == 0
    except Exception:
        return False


_LIVE = _sandbox_reachable()
_SLOW_OPT_IN = os.environ.get("SANDBOX_LIVE_SLOW_TESTS") == "1"

live = pytest.mark.skipif(
    not _LIVE,
    reason=f"ssh {_SSH_HOST} / docker exec {_CONTAINER} not reachable from here")
live_slow = pytest.mark.skipif(
    not (_LIVE and _SLOW_OPT_IN),
    reason="opt in with SANDBOX_LIVE_SLOW_TESTS=1 (runs a real ~600s busy "
           "loop against the production sandbox)")


def _push_driver():
    with open(_DRIVER_SRC, "rb") as f:
        src = f.read()
    r = _ssh(["docker", "exec", "-i", _CONTAINER, "sh", "-c",
              f"cat > {_DRIVER_DST}"], timeout=15, input_bytes=src)
    assert r.returncode == 0, r.stderr.decode(errors="replace")


def _driver(*args, timeout=320):
    r = _ssh(["docker", "exec", _CONTAINER, "python3", _DRIVER_DST, *args],
              timeout=timeout)
    assert r.returncode == 0, (
        f"driver {args} failed: {r.stderr.decode(errors='replace')}")
    return json.loads(r.stdout.decode())


def _launch(shell_command, wait=0):
    import base64
    encoded = base64.b64encode(shell_command.encode()).decode()
    return _driver("launch", encoded, str(wait))


def _ps_count(container, pattern):
    r = _ssh(["docker", "exec", container, "ps", "aux"], timeout=15)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    out = r.stdout.decode(errors="replace")
    return sum(1 for line in out.splitlines() if pattern in line)


@live
def test_live_help_does_not_open_a_blocking_pager():
    """A real reported incident, not a hypothetical: a model answered a
    prompt by calling help() on a library function, pydoc picked a terminal
    pager, printed "Press RETURN to continue", and the turn hung 10+
    minutes. The UI's Stop button did not cancel it, because the process
    was alive and blocked on stdin rather than looping - nothing in the
    stack kills a blocked process (P45).

    The Dockerfile sets TERM=dumb/PAGER=cat so pydoc.getpager() returns
    plainpager. This asserts the deployed image actually has that, by
    running the exact call that hung: it must finish, not block.

    Reproduced before the fix (status stayed "running", output was the
    "Press RETURN to continue" prompt) and confirmed fixed after.
    """
    _push_driver()
    launch = _launch('python3 -c "help(print)"', wait=15)
    pid = launch["id"]
    try:
        status = _driver("status", pid, "0")
        assert status.get("status") == "done", (
            "help() did not terminate - the pager is blocking on stdin again. "
            f"status={status.get('status')!r} output={status.get('output')!r}"
        )
        text = "".join(e.get("data", "") for e in status.get("output", []))
        assert "Press RETURN" not in text, text
        assert "Help on built-in function print" in text, text
    finally:
        # force=true so a regression (still blocked on the pager) is cleaned
        # up by the test rather than left running in production.
        _driver("cancel", pid, "true")


@live
def test_live_cancellation_leaves_no_orphan_process():
    """Item 3: cancellation mid-execution must leave no orphaned process.
    tini is PID 1 in the image specifically to reap anything a killed
    process leaves behind (see the Dockerfile comment) - confirm the
    process is actually gone, not just marked "killed" in Open Terminal's
    own bookkeeping.
    """
    _push_driver()
    marker = "sandbox_runtime_cancel_test"
    launch = _launch(f'python3 -c "while True: pass  # {marker}"', wait=0)
    process_id = launch["id"]

    assert _ps_count(_CONTAINER, marker) >= 1, "process did not actually start"

    cancel = _driver("cancel", process_id, "true")
    assert cancel.get("status") == "killed", cancel

    status = _driver("status", process_id, "0")
    assert status.get("status") == "killed", status
    assert status.get("exit_code") in (-9, -15), status  # SIGKILL/SIGTERM

    remaining = _ps_count(_CONTAINER, marker)
    assert remaining == 0, (
        f"expected the cancelled process to be fully reaped, found "
        f"{remaining} matching process(es) still running")


@live
def test_live_concurrent_executions_stay_isolated_with_unique_paths():
    """Item 4, scoped to what this fix guarantees: two concurrent
    executions that each use a uuid-namespaced path do not cross-
    contaminate, even though the sandbox is a single shared container with
    one shared /home/user (the per-user two-container design described in earlier setup docs was
    never deployed - see P43). This does NOT claim the contamination risk is
    gone in general: two pieces of LLM-generated code that both happen to
    pick the same fixed filename can still collide, which is a structural
    property of the deployed single-container topology, not something a
    test in this repo can fix.
    """
    _push_driver()
    marker_a, marker_b = "sbx_isolation_A", "sbx_isolation_B"
    cmd_a = (
        'python3 -c "import uuid,time; p=f\'/tmp/{uuid.uuid4().hex}_A.txt\'; '
        f'open(p,\'w\').write(\'{marker_a}\'); time.sleep(2); '
        'print(open(p).read())"')
    cmd_b = (
        'python3 -c "import uuid,time; p=f\'/tmp/{uuid.uuid4().hex}_B.txt\'; '
        f'open(p,\'w\').write(\'{marker_b}\'); time.sleep(2); '
        'print(open(p).read())"')

    launch_a = _launch(cmd_a, wait=0)
    launch_b = _launch(cmd_b, wait=0)
    assert launch_a["id"] != launch_b["id"]

    status_a = _driver("status", launch_a["id"], "10")
    status_b = _driver("status", launch_b["id"], "10")

    out_a = "".join(e["data"] for e in status_a["output"] if e["type"] == "output")
    out_b = "".join(e["data"] for e in status_b["output"] if e["type"] == "output")

    assert marker_a in out_a and marker_b not in out_a, out_a
    assert marker_b in out_b and marker_a not in out_b, out_b


@live_slow
def test_live_infinite_loop_is_not_autonomously_killed_by_open_terminal():
    """Item 2, ground truth (P45): a genuine CPU-bound `while True: pass` -
    not the memory-bomb case, and not
    `time.sleep(700)` either (a sleeping process yields the CPU
    immediately; it never exercises whatever would kill a loop that pegs a
    core) - is NOT killed by OPEN_TERMINAL_EXECUTE_TIMEOUT. This asserts the
    verified-negative on purpose.

    Read from Open Terminal's own source inside the image
    (`/usr/local/lib/python3.12/site-packages/open_terminal/{env,main}.py`):
    EXECUTE_TIMEOUT only supplies the *default value of the `wait` query
    parameter* when a caller omits it - i.e. how long the HTTP request
    blocks hoping the command finishes before returning "running" anyway.
    It is never passed to anything that kills a process. Grepping every
    kill/terminate/SIGTERM/SIGKILL call site in the package confirms the
    only ways a background /execute process dies are (a) an explicit
    `DELETE /execute/{id}`, or (b) the PTY-session shutdown path used by
    the *interactive* `/api/terminals/user` endpoint - neither fires here
    on its own.

    Confirmed live: launched 16:41:32Z, still "running" through t+770s
    (770s > the compose.yml-configured OPEN_TERMINAL_EXECUTE_TIMEOUT of
    600s with wide margin) with `ps aux` inside the container showing it
    still consuming a full CPU core the entire time; only died when an
    explicit `DELETE ?force=true` was issued by hand. Left unactioned, it
    would have run until the container's own resources (`cpus: 2.0`
    throttles it, doesn't kill it) or the nightly `docker restart` cron (§7)
    intervened - potentially hours.

    This test polls for a much shorter window than that 770s proof (the
    source-level finding means there is no elapsed-time-dependent branch to
    wait out - a short poll demonstrates the same "nothing kills this"
    property without repeatedly burning a full CPU core in production for
    13 minutes on every run) and always force-kills the process itself in a
    finally block, so running this test doesn't leave the exact orphan it's
    documenting.

    F6b (2026-10-25): patch_open_terminal.py now kills an /execute command
    after SANDBOX_EXECUTE_HARD_CAP_S (600 s default). This 90 s window stays
    below that, so the assertion still holds; the cap itself is covered
    offline by tests/test_patch_open_terminal.py.

    If this test ever starts FAILING (status becomes "killed" without our
    intervention), that is good news - it means an Open Terminal upgrade
    added real enforcement - and README.md's sandbox notes + this test both
    need updating to match, not the other way around.
    """
    _push_driver()
    marker = "sbx_infinite_loop_no_autokill_test"
    launch = _launch(f'python3 -c "while True: pass  # {marker}"', wait=5)
    process_id = launch["id"]
    assert launch["status"] == "running", (
        "loop finished immediately - not actually CPU-bound as launched")

    try:
        poll_window_s = 90  # short on purpose; see docstring
        poll_slice = 30
        elapsed = 0
        status = launch
        while status.get("status") == "running" and elapsed < poll_window_s:
            status = _driver("status", process_id, str(poll_slice))
            elapsed += poll_slice

        assert status.get("status") == "running", (
            f"Open Terminal killed the process on its own after {elapsed}s "
            f"with no DELETE call from this test (status={status}) - "
            "P45 may be fixed upstream; update README.md's sandbox notes "
            "and this test's expectation to match")
        assert _ps_count(_CONTAINER, marker) >= 1, (
            "status says running but no matching process found in ps aux")
    finally:
        # Nothing server-side will ever clean this up (that's the bug) -
        # this test must not be the reason a CPU core stays pegged.
        _driver("cancel", process_id, "true")

    remaining = _ps_count(_CONTAINER, marker)
    assert remaining == 0, (
        f"expected our own cleanup DELETE to leave no orphan, found "
        f"{remaining} matching process(es) still running")


# --------------------------------------------------------------------------
# sandbox_watchdog.py - the only thing in this stack that kills a stuck
# process (P45, P68). Offline tests for the age parsing, because getting it
# wrong in either direction is costly: too eager kills a legitimate long
# computation, too lax reproduces the 30-minute hang it exists to prevent.
# --------------------------------------------------------------------------

sys.path.insert(0, _CODE_EXEC_ROOT)
import sandbox_watchdog as wd  # noqa: E402


def test_watchdog_age_parses_open_terminal_ids_as_utc():
    """Ids look like `20260924-110546-aaa813`. The list endpoint returns NO
    start_time, so the id is the only age signal available - verified UTC by
    launching a process at a known clock time and comparing (2026-09-24).
    """
    now = wd.datetime(2026, 9, 24, 11, 20, 46, tzinfo=wd.timezone.utc)
    assert wd.age_seconds("20260924-110546-aaa813", now) == 900.0
    assert wd.age_seconds("20260924-112046-deadbe", now) == 0.0


def test_watchdog_refuses_to_guess_an_age_it_cannot_parse():
    """An unparseable id must return None, and main() must treat that as
    "leave it alone". Killing something because its name looked odd is a
    worse failure than missing one runaway - this pins the contract so a
    later refactor can't quietly turn None into 0 (which would kill
    everything) or into a huge number (same).
    """
    now = wd.datetime(2026, 9, 24, 11, 20, 46, tzinfo=wd.timezone.utc)
    for bad in ("", "nonsense", "notadate-112046-x", "20260924", "2026xx24-110546-a"):
        assert wd.age_seconds(bad, now) is None, bad


def test_watchdog_threshold_stays_a_backstop():
    """The default must sit well clear of any client-side execution deadline
    (600s), so the watchdog is a backstop rather than a competitor racing
    the client to kill the same process. If this ever trips, either that
    deadline failed or the request never went through the tool - both worth
    knowing, and neither is true if the two numbers overlap.
    """
    assert wd.MAX_AGE_SECONDS >= 900
