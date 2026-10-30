"""Patch Open Terminal to kill the subprocess when the client disconnects.

Why this exists: a retest reported pressing Stop mid-computation and finding
the process (plus its workers) still at 92-97% CPU in `ps aux` afterwards.

This took three attempts to get right, each one disproven by an isolated
test rather than assumed - see the session that produced this file for the
full trail. Recorded here so the next person does not retrace it:

  1. First attempt: catch `asyncio.CancelledError` around the existing
     `asyncio.wait_for(asyncio.shield(...))` call, on the theory that Open
     WebUI's Stop (confirmed, via open_webui/tasks.py's stop_task(), to call
     a real `task.cancel()`) would propagate into this server's handler.
     DISPROVEN: a raw TCP socket forcibly closed mid-request left the
     subprocess "running" 8+ seconds later. Starlette/Uvicorn does not
     cancel a handler's task just because the transport closed, unless the
     handler is actively polling the ASGI receive channel - which this one
     was not.
  2. Second attempt: poll Starlette's own `Request.is_disconnected()`
     in a loop instead. This is the officially documented mechanism and
     worked perfectly in an isolated minimal FastAPI app. DISPROVEN against
     Open Terminal's actual app: still never detected the disconnect. Root
     cause, found by bisecting which piece of Open Terminal's app made the
     difference: `normalize_null_query_params`, registered via
     `@app.middleware("http")`, uses Starlette's `BaseHTTPMiddleware` -
     which runs the downstream app against an internally buffered receive
     stream and simply never calls the real `receive()` again once it has
     the body. No amount of polling downstream of it can ever see a
     disconnect message that nothing is asking the transport for.
  3. This version: a raw ASGI middleware (not `@app.middleware("http")`,
     which is the broken BaseHTTPMiddleware kind) added OUTERMOST via
     `app.add_middleware(SandboxDisconnectWatcher)` - Starlette builds its
     middleware stack in reverse registration order, so adding it last
     makes it the first thing a request passes through. It drains the real
     receive() itself, buffers body messages so downstream (including
     BaseHTTPMiddleware and FastAPI's own Pydantic body parsing) sees
     exactly what it would have otherwise, and keeps listening on the real
     transport for `http.disconnect` for the lifetime of the request,
     independent of whether anything downstream ever calls receive() again.
     Verified in an isolated test app that exactly reproduces Open
     Terminal's middleware shape: disconnect detected within one poll
     interval (~0.5-1s) of a real socket close, normal requests unaffected.

Run at Docker build time only (the runtime root filesystem is read_only -
see compose.yml). Every replacement is exact-match and asserts count == 1:
Open Terminal is pulled from a floating `:slim` tag (already drifted 0.13.0
-> 0.14.0 silently) and a future version reshaping these blocks must fail
the build loudly, not silently ship with the patch unapplied.
"""
import os
import sys

# Overridable only so the patch can be dry-run against a copied main.py; the
# Docker build always uses the default (the copy the server actually imports -
# /app/open_terminal in the image is an unused upstream source tree).
MAIN_PY = os.environ.get(
    "PATCH_OPEN_TERMINAL_MAIN_PY",
    "/usr/local/lib/python3.12/site-packages/open_terminal/main.py",
)

MIDDLEWARE_ANCHOR = '''@app.middleware("http")
async def normalize_null_query_params(request: Request, call_next):
    """Strip query parameters whose value is the literal string 'null'."""
    from urllib.parse import urlencode

    raw_params = request.query_params.multi_items()
    cleaned = [(k, v) for k, v in raw_params if v.lower() != "null"]
    if len(cleaned) != len(raw_params):
        request.scope["query_string"] = urlencode(cleaned).encode("utf-8")
    return await call_next(request)
'''

DISCONNECT_WATCHER = '''

class SandboxDisconnectWatcher:
    """Detects a client disconnect even underneath normalize_null_query_params's
    BaseHTTPMiddleware, which never calls receive() again once it has the
    body - see patch_open_terminal.py for why that defeats every simpler
    approach. Must be added LAST (app.add_middleware stacks in reverse, so
    the last one added is outermost) to drain the real transport directly.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        state["sandbox_disconnected"] = False
        buffered = []
        body_done = asyncio.Event()

        async def drain():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    state["sandbox_disconnected"] = True
                    return
                buffered.append(message)
                if not message.get("more_body", False):
                    body_done.set()

        drain_task = asyncio.create_task(drain())

        async def replay_receive():
            if buffered:
                return buffered.pop(0)
            if not body_done.is_set():
                await body_done.wait()
                if buffered:
                    return buffered.pop(0)
            # Body fully delivered and nothing downstream will call
            # receive() again on its own (BaseHTTPMiddleware's behaviour) -
            # block here until a real disconnect arrives, so this watcher
            # is the one thing still listening on the actual transport.
            await drain_task
            return {"type": "http.disconnect"}

        try:
            await self.app(scope, replay_receive, send)
        finally:
            drain_task.cancel()


app.add_middleware(SandboxDisconnectWatcher)

'''

# Two call sites: execute()'s unconditional wait, and get_status()'s which
# also gates on status == "running". Patched separately because they differ.
WAIT_PATTERNS = [
    (
        "execute()",
        '''    if wait is not None:
        try:
            await asyncio.wait_for(
                asyncio.shield(background_process.log_task), timeout=wait
            )
        except asyncio.TimeoutError:
            pass
''',
        '''    if wait is not None:
        await _sandbox_wait_with_disconnect_kill(http_request, background_process, wait)
''',
    ),
    (
        "get_status()",
        '''    if wait is not None and background_process.status == "running":
        try:
            await asyncio.wait_for(
                asyncio.shield(background_process.log_task), timeout=wait
            )
        except asyncio.TimeoutError:
            pass
''',
        '''    if wait is not None and background_process.status == "running":
        await _sandbox_wait_with_disconnect_kill(http_request, background_process, wait)
''',
    ),
]

HELPER_FUNCTION = '''

async def _sandbox_wait_with_disconnect_kill(http_request, background_process, wait):
    """Wait for the process to finish, but kill it the moment the client
    disconnects (e.g. Open WebUI's Stop button) instead of leaving it to run
    unwatched. Polls scope["state"]["sandbox_disconnected"], set live by
    SandboxDisconnectWatcher - NOT Request.is_disconnected(), which (verified)
    never fires here due to normalize_null_query_params's BaseHTTPMiddleware.
    """
    log_wait = asyncio.ensure_future(asyncio.shield(background_process.log_task))
    try:
        while not log_wait.done():
            if http_request.state.sandbox_disconnected:
                if background_process.status == "running":
                    background_process.status = "killed"
                    background_process.runner.kill(force=True)
                    background_process.exit_code = await background_process.runner.wait()
                    background_process.runner.close()
                    background_process.finished_at = time.time()
                return
            done, _ = await asyncio.wait({log_wait}, timeout=min(0.5, wait))
            wait -= 0.5
            if wait <= 0 and not log_wait.done():
                return
    finally:
        if not log_wait.done():
            log_wait.cancel()

'''


SANDBOX_LIMIT_HELPERS = r'''

# F6b (2026-10-25): hard time limits. Neither Open Terminal 0.14.0 limit is
# real: `wait` (and OPEN_TERMINAL_EXECUTE_TIMEOUT) only bounds how long the HTTP
# call blocks - the command keeps running - and list/grep/match/glob/search take
# no timeout at all. File-search tools were observed to run ~10 minutes on
# broad queries: a walk of "/" (P170's walk, minus the OOM). Both limits are
# env-tunable; 0 disables one.
#
# 20 s: a full grep_search-style walk of every upload (/mnt/uploads, 1,226
# files, 3.2 GB) took 5.9 s in the live sandbox (grep -rIl: 4.1 s), 2026-10-25.
_SANDBOX_FILE_SEARCH_DEADLINE_S = float(os.environ.get("SANDBOX_FILE_SEARCH_DEADLINE_S", "20"))
# 600 s: the longest legitimate run in 3,106 completed run_command results (655
# chats in the Open WebUI DB, 2026-10-25) took "ELAPSED: 219.8" printed by the
# script itself; p99 of all completed runs is <= 74 s. 600 is 2.7x that, equal
# to OPEN_TERMINAL_EXECUTE_TIMEOUT and the client tool's 600 s valve, and below
# sandbox_watchdog's 900 s backstop (which stays as the out-of-process net).
_SANDBOX_EXECUTE_HARD_CAP_S = float(os.environ.get("SANDBOX_EXECUTE_HARD_CAP_S", "600"))
_sandbox_cap_tasks = set()     # strong refs: a bare create_task() can be GC'd


class _SandboxDeadline:
    """Cooperative deadline for the file-search walks. They run in a worker
    thread (asyncio.to_thread), which cannot be cancelled from outside - so the
    walk itself checks this and stops, returning what it has found so far."""
    def __init__(self, seconds=None):
        self.seconds = _SANDBOX_FILE_SEARCH_DEADLINE_S if seconds is None else seconds
        self.end = time.monotonic() + self.seconds
        self.hit = False

    def expired(self):
        if self.seconds > 0 and not self.hit and time.monotonic() >= self.end:
            self.hit = True
        return self.hit

    def remaining(self):
        if self.seconds <= 0:
            return None                     # no limit: subprocess timeout=None
        return max(0.05, self.end - time.monotonic())

    def note(self):
        if not self.hit:
            return {}
        return {"stopped_early": True,
                "note": "stopped after %g s - these results are partial. Narrow the "
                        "path (e.g. /mnt/uploads, or one folder) or the pattern and "
                        "search again; do not search /." % self.seconds}


def _sandbox_arm_hard_cap(background_process):
    """Kill the command if it is still running after _SANDBOX_EXECUTE_HARD_CAP_S,
    whatever `wait` the caller used - the same kill kill_process() does."""
    if _SANDBOX_EXECUTE_HARD_CAP_S <= 0:
        return

    async def _cap():
        await asyncio.sleep(_SANDBOX_EXECUTE_HARD_CAP_S)
        if background_process.status != "running":
            return
        logging.getLogger("sandbox").warning(
            "hard cap: killing process %s after %g s: %.200s", background_process.id,
            _SANDBOX_EXECUTE_HARD_CAP_S, background_process.command)
        background_process.sandbox_capped = True
        background_process.status = "killed"
        background_process.runner.kill(force=True)
        background_process.exit_code = await background_process.runner.wait()
        background_process.runner.close()
        background_process.finished_at = time.time()

    task = asyncio.create_task(_cap())
    _sandbox_cap_tasks.add(task)
    task.add_done_callback(_sandbox_cap_tasks.discard)
    if background_process.log_task is not None:
        # finished on its own: drop the timer instead of holding it for 600 s
        background_process.log_task.add_done_callback(lambda _t: task.cancel())


def _sandbox_cap_note(background_process):
    if not getattr(background_process, "sandbox_capped", False):
        return {}
    return {"note": "killed by the sandbox's %g s hard limit - the command was still "
                    "running. Make it faster (smaller input, fewer iterations) "
                    "rather than retrying it unchanged." % _SANDBOX_EXECUTE_HARD_CAP_S}

'''


# P170 (2026-10-20 09:21 and 2026-10-21 18:12, both memcg OOM kills of the
# server at ~12.5 GB): a model called grep_search with path "/". The walk opens
# every name it lists as UTF-8 text and iterates `for line in f` - including
# /dev/zero, whose NUL bytes are valid UTF-8 and never contain a newline, so a
# single "line" grows until the 12 GiB cgroup kills the server (proven in a
# 600 MB-capped child: MemoryError after 1.3 s). It runs in asyncio.to_thread
# inside the server, so the OOM task table showed only open-terminal + tini.
# os.path.isfile() alone is NOT enough: the RED/GREEN run showed /proc/<pid>/
# pagemap is a "regular" file (st_size 0) that reads back as an endless run of
# zero bytes - same MemoryError. So the walk is kept out of the kernel's pseudo
# filesystems entirely, and only regular files elsewhere are opened.
#
# The helper lives in utils/fs.py so ONE change to UserFS.isfile() also closes
# every endpoint that gates on it before a whole-file read (audit 2026-10-21):
# read_file and replace_file_content (both model-callable tools), view/serve,
# compare, skills - each would read /proc/self/pagemap whole. archive_paths
# (zf.write of a char device), match_files (getsize()==0 defeats its 1 MiB cap)
# and notebooks create_session get explicit guards below.
PKG = os.path.dirname(MAIN_PY)

FS_PATCHES = [
    (
        "fs.py helper",
        "import aiofiles\nimport aiofiles.os\n",
        '''import aiofiles
import aiofiles.os
import stat as _sandbox_stat

# P170: os.path.isfile() alone is not enough - /proc/<pid>/pagemap is a
# *regular* file (st_size 0) that streams ~256 GB of NUL bytes with no newline,
# and /dev/zero does the same. Regular files only, never anything that resolves
# into a kernel pseudo-filesystem.
_SANDBOX_PSEUDO_FS = ("/proc", "/sys", "/dev")


def sandbox_in_pseudo_fs(path) -> bool:
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError):
        return True
    return any(real == p or real.startswith(p + "/") for p in _SANDBOX_PSEUDO_FS)


def sandbox_is_plain_file(path) -> bool:
    try:
        if sandbox_in_pseudo_fs(path):
            return False
        return _sandbox_stat.S_ISREG(os.stat(os.path.realpath(path)).st_mode)
    except (OSError, ValueError):
        return False
''',
    ),
    (
        "UserFS.isfile",
        "        return await aiofiles.os.path.isfile(path)\n",
        "        return await asyncio.to_thread(sandbox_is_plain_file, path)\n",
    ),
]

NOTEBOOK_PATCHES = [
    (
        "notebooks import",
        "import aiofiles\nimport aiofiles.os\n",
        "import aiofiles\nimport aiofiles.os\nfrom open_terminal.utils.fs import sandbox_is_plain_file\n",
    ),
    (
        "notebooks create_session",
        "        if not await aiofiles.os.path.isfile(path):\n",
        "        if not await asyncio.to_thread(sandbox_is_plain_file, path):\n",
    ),
]

DEVICE_FILE_PATCHES = [
    (
        "main.py import",
        "from open_terminal.utils.fs import UserFS\n",
        "from open_terminal.utils.fs import UserFS, sandbox_in_pseudo_fs, sandbox_is_plain_file\n",
    ),
    (
        "match_files content scan",
        "                    if os.path.getsize(file_path) > MAX_CONTENT_SEARCH_FILE_SIZE:\n",
        "                    if not sandbox_is_plain_file(file_path) or os.path.getsize(file_path) > MAX_CONTENT_SEARCH_FILE_SIZE:\n",
    ),
    (
        "archive single path",
        "                if os.path.isfile(target):\n                    zf.write(target, os.path.basename(target))\n",
        "                if sandbox_is_plain_file(target):\n                    zf.write(target, os.path.basename(target))\n",
    ),
    (
        "archive walk",
        "                            if not fs.is_path_allowed(full):\n                                continue\n                            arcname",
        "                            if not fs.is_path_allowed(full) or not sandbox_is_plain_file(full):\n                                continue\n                            arcname",
    ),
    (
        "grep_search dir prune",
        '''                dirnames[:] = [
                    d for d in dirnames
                    if fs.is_path_allowed(os.path.join(dirpath, d))
                ]
''',
        '''                dirnames[:] = [
                    d for d in dirnames
                    if fs.is_path_allowed(os.path.join(dirpath, d))
                    and not sandbox_in_pseudo_fs(os.path.join(dirpath, d))
                ]
''',
    ),
    (
        "grep_search file guard",
        '''                    full = os.path.join(dirpath, filename)
                    if not fs.is_path_allowed(full):
                        continue
                    _search_file(full)
''',
        '''                    full = os.path.join(dirpath, filename)
                    if not fs.is_path_allowed(full):
                        continue
                    if not sandbox_is_plain_file(full):
                        continue  # P170: /dev/zero, /proc/*/pagemap
                    _search_file(full)
''',
    ),
]


# F6b: (name, old, new, expected count). Applied AFTER DEVICE_FILE_PATCHES, so
# the grep_search anchors below are the device-guarded text. Every walk checks
# the deadline per directory (and grep per file, plus every 4096 lines inside a
# huge file); git/rg subprocesses get the remaining time as their timeout.
_GIT_LS = '["git", "-C", target, "ls-files", "-co", "--exclude-standard", "-z", "--", "."],'
LIMIT_PATCHES = [
    # ---- /execute hard cap ------------------------------------------------
    ("execute arm cap",
     "    _processes[process_id] = background_process\n",
     "    _processes[process_id] = background_process\n"
     "    _sandbox_arm_hard_cap(background_process)\n", 1),
    # execute() and get_status() build the same response dict; list_processes'
    # is indented deeper and is not matched.
    ("execute/status cap note",
     '        "log_path": background_process.log_path,\n    }\n',
     '        "log_path": background_process.log_path,\n'
     '        **_sandbox_cap_note(background_process),\n    }\n', 2),
    # ---- /files/grep (grep_search) ----------------------------------------
    ("grep deadline",
     "    def _search_sync():\n        def _matches_include(filename: str) -> bool:\n",
     "    _sandbox_deadline = _SandboxDeadline()\n\n"
     "    def _search_sync():\n        def _matches_include(filename: str) -> bool:\n", 1),
    ("grep per-file + per-line check",
     '            nonlocal truncated\n'
     '            if truncated:\n'
     '                return\n'
     '            try:\n'
     '                with open(file_path, "r", encoding="utf-8", errors="strict") as f:\n'
     '                    for line_number, line in enumerate(f, 1):\n',
     '            nonlocal truncated\n'
     '            if truncated or _sandbox_deadline.expired():\n'
     '                return\n'
     '            try:\n'
     '                with open(file_path, "r", encoding="utf-8", errors="strict") as f:\n'
     '                    for line_number, line in enumerate(f, 1):\n'
     '                        if not line_number % 4096 and _sandbox_deadline.expired():\n'
     '                            return\n', 1),
    ("grep walk check",
     "                if truncated:\n                    break\n                for filename in sorted(filenames):\n",
     "                if truncated or _sandbox_deadline.expired():\n                    break\n"
     "                for filename in sorted(filenames):\n", 1),
    ("grep result note",
     '    matches, truncated = await asyncio.to_thread(_search_sync)\n'
     '    return {\n'
     '        "query": query,\n'
     '        "path": target,\n'
     '        "matches": matches,\n'
     '        "truncated": truncated,\n'
     '    }\n',
     '    matches, truncated = await asyncio.to_thread(_search_sync)\n'
     '    return {\n'
     '        "query": query,\n'
     '        "path": target,\n'
     '        "matches": matches,\n'
     '        "truncated": truncated or _sandbox_deadline.hit,\n'
     '        **_sandbox_deadline.note(),\n'
     '    }\n', 1),
    # ---- /files/matches (match_files) -------------------------------------
    ("matches deadline",
     "    def _matches_sync():\n        query_lower = query.lower()\n",
     "    _sandbox_deadline = _SandboxDeadline()\n\n"
     "    def _matches_sync():\n        query_lower = query.lower()\n", 1),
    ("matches git timeout",
     "                git_result = subprocess.run(\n                    " + _GIT_LS + "\n"
     "                    check=False,\n                    capture_output=True,\n                )\n"
     "            except OSError:\n                git_result = None\n",
     "                git_result = subprocess.run(\n                    " + _GIT_LS + "\n"
     "                    check=False,\n                    capture_output=True,\n"
     "                    timeout=_sandbox_deadline.remaining(),\n                )\n"
     "            except subprocess.TimeoutExpired:\n"
     "                _sandbox_deadline.hit = True\n                git_result = None\n"
     "            except OSError:\n                git_result = None\n", 1),
    ("matches walk check",
     '            entries = []\n'
     '            for dirpath, dirnames, filenames in os.walk(target):\n'
     '                dirnames[:] = [\n'
     '                    d\n'
     '                    for d in sorted(dirnames)\n',
     '            entries = []\n'
     '            for dirpath, dirnames, filenames in os.walk(target):\n'
     '                if _sandbox_deadline.expired():\n'
     '                    break\n'
     '                dirnames[:] = [\n'
     '                    d\n'
     '                    for d in sorted(dirnames)\n', 1),
    ("matches rg timeout",
     '                completed = subprocess.run(args, capture_output=True, text=True, check=False)\n'
     '            except OSError:\n'
     '                return None\n',
     '                completed = subprocess.run(args, capture_output=True, text=True, check=False,\n'
     '                                           timeout=_sandbox_deadline.remaining())\n'
     '            except subprocess.TimeoutExpired as exc:\n'
     '                # rg is killed at the deadline; keep the matches it printed\n'
     '                _sandbox_deadline.hit = True\n'
     '                partial = exc.stdout or ""\n'
     '                if isinstance(partial, bytes):\n'
     '                    partial = partial.decode("utf-8", "replace")\n'
     '                completed = subprocess.CompletedProcess(args, 0, partial, "")\n'
     '            except OSError:\n'
     '                return None\n', 1),
    ("matches content-scan check",
     "            matches: dict[str, list[dict]] = {}\n            for file_path in files:\n",
     "            matches: dict[str, list[dict]] = {}\n            for file_path in files:\n"
     "                if _sandbox_deadline.expired():\n                    break\n", 1),
    ("matches result note",
     '            "next_offset": next_offset,\n        }\n',
     '            "next_offset": next_offset,\n            **_sandbox_deadline.note(),\n        }\n', 1),
    # ---- /files/search (search_files) -------------------------------------
    ("search deadline",
     "    def _search_sync():\n        query_lower = query.strip().lower()\n",
     "    _sandbox_deadline = _SandboxDeadline()\n\n"
     "    def _search_sync():\n        query_lower = query.strip().lower()\n", 1),
    ("search git timeout",
     "            git_result = subprocess.run(\n                " + _GIT_LS + "\n"
     "                check=False,\n                capture_output=True,\n            )\n"
     "        except OSError:\n            git_result = None\n",
     "            git_result = subprocess.run(\n                " + _GIT_LS + "\n"
     "                check=False,\n                capture_output=True,\n"
     "                timeout=_sandbox_deadline.remaining(),\n            )\n"
     "        except subprocess.TimeoutExpired:\n"
     "            _sandbox_deadline.hit = True\n            git_result = None\n"
     "        except OSError:\n            git_result = None\n", 1),
    ("search walk check",
     '        else:\n'
     '            for dirpath, dirnames, filenames in os.walk(target):\n'
     '                dirnames[:] = [\n',
     '        else:\n'
     '            for dirpath, dirnames, filenames in os.walk(target):\n'
     '                if _sandbox_deadline.expired():\n'
     '                    break\n'
     '                dirnames[:] = [\n', 1),
    ("search stat check",
     '        for full_path, entry_type in candidates:\n'
     '            if not fs.is_path_allowed(full_path):\n'
     '                continue\n',
     '        for full_path, entry_type in candidates:\n'
     '            if _sandbox_deadline.expired():\n'
     '                break\n'
     '            if not fs.is_path_allowed(full_path):\n'
     '                continue\n', 1),
    ("search result note",
     '    return {"results": await asyncio.to_thread(_search_sync)}\n',
     '    _sandbox_results = await asyncio.to_thread(_search_sync)\n'
     '    return {"results": _sandbox_results, **_sandbox_deadline.note()}\n', 1),
    # ---- /files/glob (glob_search) ----------------------------------------
    ("glob deadline",
     "    def _glob_sync():\n",
     "    _sandbox_deadline = _SandboxDeadline()\n\n    def _glob_sync():\n", 1),
    ("glob walk check",
     "            if truncated:\n                break\n\n            # Prune directories belonging to other users.\n",
     "            if truncated or _sandbox_deadline.expired():\n                break\n\n"
     "            # Prune directories belonging to other users.\n", 1),
    ("glob result note",
     '    matches, truncated = await asyncio.to_thread(_glob_sync)\n'
     '    return {\n'
     '        "pattern": pattern,\n'
     '        "path": target,\n'
     '        "matches": matches,\n'
     '        "truncated": truncated,\n'
     '    }\n',
     '    matches, truncated = await asyncio.to_thread(_glob_sync)\n'
     '    return {\n'
     '        "pattern": pattern,\n'
     '        "path": target,\n'
     '        "matches": matches,\n'
     '        "truncated": truncated or _sandbox_deadline.hit,\n'
     '        **_sandbox_deadline.note(),\n'
     '    }\n', 1),
]


def _apply_counted(content, patches):
    """Like _apply, but each entry states how many occurrences it expects."""
    for name, old, new, expected in patches:
        count = content.count(old)
        if count != expected:
            print(f"patch_open_terminal: FATAL - expected {expected} occurrence(s) "
                  f"of the F6b {name} block, found {count}. Re-verify the file-search "
                  f"deadline / execute hard cap against the new Open Terminal source.",
                  file=sys.stderr)
            sys.exit(1)
        content = content.replace(old, new)
    return content


def _apply(content, patches):
    for name, old, new in patches:
        count = content.count(old)
        if count != 1:
            print(f"patch_open_terminal: FATAL - expected exactly 1 occurrence "
                  f"of the {name} block, found {count}. Re-verify the P170 "
                  f"device-file guard against the new Open Terminal source.",
                  file=sys.stderr)
            sys.exit(1)
        content = content.replace(old, new)
    return content


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    # Every file is computed in memory and every guard checked BEFORE anything
    # is written: main.py imports the fs.py helper, so a half-applied set would
    # ship an ImportError (and `open(p, "w").write(_apply())` truncated p before
    # _apply ran - the 2026-10-21 double-apply dry run wiped utils/fs.py).
    fs_py = os.path.join(PKG, "utils", "fs.py")
    nb_py = os.path.join(PKG, "utils", "notebooks.py")
    new_fs = _apply(_read(fs_py), FS_PATCHES)
    new_nb = _apply(_read(nb_py), NOTEBOOK_PATCHES)

    content = _apply(_read(MAIN_PY), DEVICE_FILE_PATCHES)
    content = _apply_counted(content, LIMIT_PATCHES)

    if content.count(MIDDLEWARE_ANCHOR) != 1:
        print(f"patch_open_terminal: FATAL - expected exactly 1 occurrence of "
              f"the middleware anchor block, found {content.count(MIDDLEWARE_ANCHOR)}.",
              file=sys.stderr)
        sys.exit(1)
    content = content.replace(
        MIDDLEWARE_ANCHOR,
        MIDDLEWARE_ANCHOR + DISCONNECT_WATCHER + HELPER_FUNCTION + SANDBOX_LIMIT_HELPERS, 1)

    for name, old, new in WAIT_PATTERNS:
        count = content.count(old)
        if count != 1:
            print(f"patch_open_terminal: FATAL - expected exactly 1 occurrence "
                  f"of the {name} wait block, found {count}. Open Terminal's "
                  f"source has changed shape (image is on a floating :slim "
                  f"tag - see Dockerfile) and this patch needs re-verifying "
                  f"against the new source before the build can proceed.",
                  file=sys.stderr)
            sys.exit(1)
        content = content.replace(old, new)

    for path, text in ((fs_py, new_fs), (nb_py, new_nb), (MAIN_PY, content)):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    print("patch_open_terminal: applied disconnect watcher + both call sites "
          f"+ {len(FS_PATCHES) + len(NOTEBOOK_PATCHES) + len(DEVICE_FILE_PATCHES)} "
          f"P170 special-file guard(s) + {len(LIMIT_PATCHES)} F6b time-limit patch(es)")


if __name__ == "__main__":
    main()
