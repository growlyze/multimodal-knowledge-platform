"""
Drives Open WebUI's own public API - the same one the browser uses - so a
test can upload a file and complete a chat turn without a human in the loop.

Requires OPENWEBUI_API_KEY (create one: avatar -> Settings -> Account -> API
Keys -> Create). Every call here costs real Anthropic/Gemini spend once it
reaches a model, so run it deliberately against a test deployment.

Deployment-specific settings come from the environment:
    OPENWEBUI_BASE_URL     https://<DOMAIN> of the stack under test (required)
    OPENWEBUI_API_KEY      Open WebUI API key (required)
    OPENWEBUI_TERMINAL_ID  id of the sandbox terminal connection in Open WebUI
                           (Admin -> Settings -> Integrations); needed for tool turns
    SSH_HOST               ssh alias of the VPS, used to reach the sandbox (default: gmac)
    STACK_NAME             compose STACK_NAME, the container-name prefix (default: gmac)

Field names below were verified against a live, authenticated call on
2026-09-23 (`/api/chat/completions` with a real OPENWEBUI_API_KEY):
response top-level keys are exactly `id, choices, created, model, object,
usage` - **there is no `chat_id` in the response, ever**. The endpoint is
a stateless OpenAI-compatible completion call; Open WebUI does not persist
or thread a conversation server-side just because a `chat_id` is sent in
the request. That single wrong assumption meant every multi-turn test
(e.g. the P29 model-switch crash repro) was silently running each "turn"
as an independent, historyless conversation - passing or failing without
ever exercising the code path it claimed to test. Fixed by having THIS
client mint its own chat_id (a uuid4, never expecting one back) and key
`self._history` on it unconditionally.

File attachments: verified against `POST /api/v1/files/`'s real response
(`id`, `filename`, `meta.name`, ...) and `filters/rag_filter.py`'s
`_handle_files()` (rag_filter.py:1097-1113), which reads entries shaped
`{"file": {"id": ..., "filename": ...}}` under `body["metadata"]["files"]`
- NOT the OpenAI-style `{"type": "file", "id": ...}` under a top-level
`files` key this file previously sent, which `_handle_files` never
recognises (`holders` only looks at `metadata.files` and top-level
`body.files`, and a `type`/`id`-only entry has no `filename` for
`route()` to detect a kind from, so it would silently fall through to
"unknown type - leave Open WebUI to it").

Tool execution: verified live, 2026-09-23. Without an explicit `tool_ids`
field, `/api/chat/completions` never registers ANY tool for the model even
though the filters' injected system prompt (FILE_READY_HINT /
DATA_TOOLS hint) unconditionally tells the model it has sandbox
access - open_webui/utils/middleware.py:2983 says so outright: "API
callers don't expect hidden tools; they can explicitly request tools via
tool_ids." Without it, the model doesn't refuse or error - it hallucinates
a plausible-looking `<function_calls><invoke ...>` TEXT block (not a real
tool_use structure) and then confabulates a plausible-but-fabricated
"result" after it, since nothing ever executes and nothing stops its
single-shot generation. This is worse than a crash: it looks like a
successful, grounded answer.

Adding `terminal_id: <OPENWEBUI_TERMINAL_ID>` ("Terminal 1" in the
`terminal_server.connections` config, pointing at the real
`<STACK_NAME>-code-sandbox-1` container) makes the model emit a genuine
`tool_calls` response (`finish_reason: "tool_calls"`, a real `run_command`
function call with real arguments) - confirmed identical whether
`stream` is `false` or `true`. `tool_ids` was also tried with
a registered Open WebUI Tool id but is deliberately left empty by default now - see
`DEFAULT_TOOL_IDS` below for why registering a tool this client cannot
execute made things worse, not better.

But even with both fields, `/api/chat/completions` is a stateless
OpenAI-compatible passthrough: it returns the tool_calls and STOPS. It
does not execute them and continue, even in streaming mode (confirmed:
the SSE stream ends with `finish_reason: "tool_calls"` then `[DONE]`,
nothing more). Open WebUI's own full agentic tool-execution loop
(middleware.py's `execute_tool_call`, ~line 5671) is wired to
`event_emitter`/`event_caller` - the browser's own Socket.IO-driven chat
flow, a different code path entirely from this REST endpoint. Per
Open WebUI's own comment above, that's intentional: API callers are
expected to execute tool calls themselves and continue the conversation,
exactly like any other OpenAI-compatible tool-use client.

The sandbox (`<STACK_NAME>-code-sandbox-1:8000`, an "Open Terminal" API -
`POST /execute` with `{"command": ...}`, `wait=<seconds>` for inline
output) lives on the docker-internal `<STACK_NAME>-network` and is unreachable
from outside the VPS. `chat()` below executes real tool calls by shelling
out over the existing SSH access and proxying the HTTP call through a
container already on that network (`docker exec`, not a new port). The
terminal's bearer key (`terminal_server.connections` in Open WebUI's own
Postgres config) is fetched fresh inside that same remote SSH command and
never touches local disk or this repo - see `_execute_sandbox_command`.
"""

import json
import os
import subprocess
import urllib.parse
import uuid

import httpx

BASE_URL = os.environ.get("OPENWEBUI_BASE_URL", "")
STACK_NAME = os.environ.get("STACK_NAME", "gmac")
SSH_HOST = os.environ.get("SSH_HOST", "gmac")

# DELIBERATELY EMPTY. A Tool registered in Open WebUI
# (`SELECT id, name FROM tool`) is a real Open WebUI Tool - Open WebUI would normally execute it
# server-side. But this client's own chat() loop (below) only knows how to
# execute `run_command` (proxied to the sandbox over SSH); it has no generic
# handler for an arbitrary registered Tool's Python. Registering
# such a tool in tool_ids without being able to execute it just gives
# the model a tool that always bounces with "no local handler" - confirmed
# live 2026-09-23: it was the model's FIRST choice on a no-file query,
# wasting a full round before it fell back to run_command. Leave this empty
# until chat() gains a real per-tool executor; `terminal_id` alone already
# gives the model everything it needs (run_command, which is what every
# passing live test actually uses).
DEFAULT_TOOL_IDS = []

# "Terminal 1" in the `terminal_server.connections` admin config (verified
# live 2026-09-23: `SELECT value FROM config WHERE key='terminal_server.connections'`),
# pointing at the `<STACK_NAME>-code-sandbox-1` container. Not a secret -
# the bearer key that actually authenticates to it is fetched at call time,
# never stored here. Each deployment has its own id: set OPENWEBUI_TERMINAL_ID.
DEFAULT_TERMINAL_ID = os.environ.get("OPENWEBUI_TERMINAL_ID", "")

# Any container already on <STACK_NAME>-network works as the docker-exec proxy to
# reach the sandbox's internal-only address; this one is always present.
_PROXY_CONTAINER = f"{STACK_NAME}-open-webui"
_SANDBOX_URL = f"http://{STACK_NAME}-code-sandbox-1:8000"

# Cached terminal OpenAPI document - see _terminal_spec().
_SPEC_CACHE = None

# Returned only for a tool the terminal server does NOT advertise, i.e. one
# this harness genuinely cannot route. Anything the terminal does advertise is
# executed for real by _call_terminal_tool(); this string used to be returned
# for all 15 non-run_command tools, which is what produced the spurious
# MAX_TOOL_ROUNDS "runaway loop" failures.
_TOOL_STUB = "[LiveClient: tool {name!r} is not advertised by the terminal server]"

# Safety cap on the tool-call continuation loop in chat() - a model that
# keeps calling tools forever (a real failure mode, not hypothetical) must
# not hang a test suite indefinitely.
#
# Raised 5 -> 10 on 2026-09-25. A legitimate upload-and-analyse turn uses 4-5
# rounds on its own (load and inspect the file, run the analysis, re-inspect
# the result's column names, then summarise), so a cap of 5 sat exactly on
# the boundary and criterion 1 failed intermittently with "probable runaway
# loop" while the product had in fact completed the work - instrumenting the
# same turn showed the file loaded, the analysis ran and results came back.
# A cap that fires on correct behaviour teaches you to distrust the alarm.
# 10 still stops a genuine runaway well before it can hang the suite.
MAX_TOOL_ROUNDS = 10


class LiveClientError(RuntimeError):
    pass


def _terminal_request(method: str, path: str, query: dict = None,
                      body: dict = None, wait: int = 90) -> str:
    """Issue one authenticated request to the sandbox's terminal API over SSH
    and return the raw response text.

    The terminal's bearer key is read fresh from Postgres INSIDE this one
    remote SSH command (a single bash one-liner: fetch key, then curl the
    sandbox through a same-network container) - it is never assigned to a
    Python variable here, never written to disk, never logged. Any request
    body travels as JSON over the SSH process's stdin (`--data-binary @-`),
    not interpolated into the remote shell string, so arbitrary content
    (quotes, heredocs, newlines) can't break the quoting or inject into the
    remote shell the way a naive f-string interpolation would (see the
    `int(since_seconds)` fix in tests/live/conftest.py for the same class of
    bug elsewhere in this suite).

    Query values go through `urlencode`, which percent-encodes the single
    quote that would otherwise close the shell quoting around the URL - the
    only untrusted text that reaches the remote command line rather than
    stdin.
    """
    url = f"{_SANDBOX_URL}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(
            {k: v for k, v in query.items() if v is not None})
    parts = [
        f"KEY=$(docker exec -i {STACK_NAME}-postgres psql -U openwebui -d openwebui "
        "-t -A -c \"SELECT value->0->>'key' FROM config "
        "WHERE key='terminal_server.connections'\") && "
        f"docker exec -i {_PROXY_CONTAINER} curl -s -X {method.upper()} "
        f"'{url}' -H \"Authorization: Bearer $KEY\"",
    ]
    if body is not None:
        parts.append("-H 'Content-Type: application/json' --data-binary @-")
    result = subprocess.run(
        ["ssh", SSH_HOST, " ".join(parts)],
        input=json.dumps(body) if body is not None else None,
        capture_output=True,
        # Same non-optional pairing as tests/live/conftest.py: `text=True` alone
        # decodes with the platform default (cp1252 on this project's Windows
        # dev machines), and sandbox output carries bytes cp1252 cannot map. The
        # decode raises inside subprocess's reader THREAD, so stdout comes back
        # None and the caller dies later on json.loads(None) - which reads as a
        # product failure in an acceptance test, not a harness one.
        encoding="utf-8", errors="replace",
        timeout=wait + 30,
    )
    if result.returncode != 0:
        raise LiveClientError(f"terminal request failed: {result.stderr[:300]}")
    return result.stdout


def _terminal_spec() -> dict:
    """The terminal server's own OpenAPI document, fetched once per process.

    Open WebUI builds the model's tool list from exactly this document, so it
    is also the only correct source for which tools a turn can legitimately
    call. Hardcoding the list here would go stale the next time the terminal
    image is bumped - and a harness that silently mishandles a tool the model
    really does have looks exactly like a product bug (see the module-level
    note on `_TOOL_STUB`).
    """
    global _SPEC_CACHE
    if _SPEC_CACHE is None:
        raw = _terminal_request("GET", "/openapi.json", wait=30)
        try:
            _SPEC_CACHE = json.loads(raw)
        except json.JSONDecodeError:
            raise LiveClientError(
                f"terminal /openapi.json returned non-JSON: {raw[:300]}")
    return _SPEC_CACHE


def _resolve_tool(name: str) -> tuple:
    """Map a tool name the model called to (method, path, param_specs).

    Matches on `operationId`, which is what Open WebUI derives the tool name
    from - `grep_search` is `GET /files/grep`, not anything guessable from the
    name alone.
    """
    for path, ops in _terminal_spec().get("paths", {}).items():
        for method, op in ops.items():
            if op.get("operationId") == name:
                return method, path, op.get("parameters", []) or []
    return None, None, None


def _call_terminal_tool(name: str, args: dict) -> str:
    """Execute any tool the terminal server advertises, not just run_command.

    `run_command` is special-cased below because its response is a process
    status object that needs flattening into readable text; every other
    endpoint returns JSON the model can read directly.

    This generality is the point. The terminal advertises 16 operations
    (`list_files`, `read_file`, `grep_search`, `glob_search`, ...) and Open
    WebUI offers all of them to the model, so a model answering a knowledge
    question will reach for `grep_search` before it reaches for a shell. When
    this client handled only `run_command` and fed a stub string back for
    everything else, the model read the stub as a broken tool, tried a
    different one, and burned a round per attempt - exhausting
    MAX_TOOL_ROUNDS and raising "probable runaway loop" for what was purely a
    gap in this harness. That failure was indistinguishable from a real
    product defect in an acceptance test, which is the worst property a test
    harness can have.
    """
    if name == "run_command":
        return _execute_sandbox_command(args.get("command", ""))
    method, path, param_specs = _resolve_tool(name)
    if method is None:
        return _TOOL_STUB.format(name=name)
    query, body = {}, {}
    in_by_name = {p.get("name"): p.get("in") for p in param_specs}
    for key, value in (args or {}).items():
        where = in_by_name.get(key)
        if where == "path":
            path = path.replace("{" + key + "}", urllib.parse.quote(str(value)))
        elif where == "query":
            query[key] = value
        else:
            # Not declared as a path/query parameter, so it belongs in the
            # request body (write_file's `content`, replace_file_content's
            # `old`/`new`). Sending it as a query param instead would 422.
            body[key] = value
    return _terminal_request(method, path, query=query,
                             body=body if body else None)


def _execute_sandbox_command(command: str, wait: int = 90) -> str:
    """Run a shell command in the real code sandbox and return its combined
    stdout/stderr text.
    """
    raw = _terminal_request("POST", "/execute", query={"wait": int(wait)},
                            body={"command": command}, wait=wait)
    try:
        status = json.loads(raw)
    except json.JSONDecodeError:
        raise LiveClientError(f"sandbox returned non-JSON: {raw[:300]}")
    output_text = "".join(
        entry.get("data", "")
        for entry in status.get("output", [])
        if entry.get("type") == "output"
    )
    exit_code = status.get("exit_code")
    if exit_code not in (0, None):
        output_text += f"\n[process exited {exit_code}]"
    if status.get("status") == "running":
        output_text += f"\n[still running after {wait}s wait - output above is partial]"
    return output_text


class LiveClient:
    def __init__(self, api_key=None, base_url=None, timeout=120.0):
        self.api_key = api_key or os.environ.get("OPENWEBUI_API_KEY")
        if not self.api_key:
            raise LiveClientError(
                "OPENWEBUI_API_KEY not set - create one in Open WebUI: "
                "avatar -> Settings -> Account -> API Keys -> Create"
            )
        self.base_url = (base_url or BASE_URL).rstrip("/")
        if not self.base_url:
            raise LiveClientError(
                "OPENWEBUI_BASE_URL not set - point it at the stack under test "
                "(https://<DOMAIN>)")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )
        # Open WebUI (like the real browser client) does not replay prior turns
        # server-side just because chat_id is passed again - the CLIENT is
        # responsible for sending the full message history on every call. This
        # was missing until 2026-09-23 (see docstring on chat() below): a
        # chat_id-only implementation looked plausible but every "later" turn
        # in a conversation was actually sent as a fresh one-message chat with
        # no history, silently defeating any test built around a multi-turn
        # conversation (e.g. the model-switch crash repro in
        # tests/live/test_data_quality_live.py). Keyed by chat_id so several
        # concurrent LiveClient instances/conversations don't cross-contaminate.
        self._history = {}
        self._file_meta = {}  # file_id -> filename, for building metadata.files entries

    def upload_file(self, path: str) -> str:
        """Upload a file exactly as the browser's attachment picker does.
        Returns the file id Open WebUI assigned (what the filter's
        `_sandbox_path()` prefixes onto the filename). Records the filename
        internally - `_handle_files()`'s `route()` call needs it to detect a
        kind (structured/document/archive/...) from the extension; an entry
        with an id but no filename silently falls through as "unknown type".
        """
        import os as _os

        filename = _os.path.basename(path)
        with open(path, "rb") as fh:
            resp = self._client.post("/api/v1/files/", files={"file": (filename, fh)})
        if resp.status_code != 200:
            raise LiveClientError(
                f"upload failed {resp.status_code}: {resp.text[:300]}"
            )
        file_id = resp.json()["id"]
        self._file_meta[file_id] = filename
        return file_id

    def chat(
        self,
        message: str,
        model: str,
        chat_id: str = None,
        file_ids: list = None,
        tool_ids: list = None,
        terminal_id: str = None,
        execute_tools: bool = True,
    ) -> tuple:
        """One turn (possibly several HTTP round-trips if the model calls a
        tool). Returns (chat_id, reply_text) where reply_text is the FINAL
        natural-language answer, not a raw tool-call description.

        `/api/chat/completions` is a stateless OpenAI-compatible endpoint -
        it never returns a chat_id and does not thread history server-side
        (verified live, 2026-09-23; see module docstring). This client mints
        its own chat_id (uuid4) on the first call of a conversation and uses
        it purely as a local key into `self._history` - callers pass it back
        on later turns the same way they would a real server-issued id, but
        no value from the server is ever consulted for this. Every call
        sends the FULL accumulated history plus the new user message, so a
        model switch mid-conversation (call again with the same chat_id and
        a different `model`) still sees prior turns.

        `tool_ids`/`terminal_id` default to this deployment's real
        Tool/terminal (module-level `DEFAULT_TOOL_IDS`/`DEFAULT_TERMINAL_ID`)
        - matching how the filters' injected system prompt unconditionally
        assumes sandbox access is available "for every model, every turn"
        (data_tools_filter.py's own description). Pass `tool_ids=[]` to
        deliberately test the no-tools-registered path (e.g. to reproduce
        the pre-fix hallucination for regression coverage).

        If the model responds with `finish_reason: "tool_calls"`, this
        method executes each `run_command` call for real (see
        `_execute_sandbox_command`) and re-sends the conversation with the
        tool results appended, up to `MAX_TOOL_ROUNDS` times - the same
        loop responsibility Open WebUI's own comment places on API callers
        ("API callers... can explicitly request tools via tool_ids"; it
        does not execute them itself over this endpoint - verified live,
        module docstring). Set `execute_tools=False` to get the raw
        first-round response instead (e.g. to assert on `finish_reason`
        itself without spending the extra sandbox round-trip).

        History stores the real API shape for tool turns - an assistant
        message with `tool_calls` and one `{"role": "tool", "tool_call_id":
        ..., "content": <output>}` per call - not the plain-string shape
        used for ordinary turns, so a later turn in the same chat_id sees
        genuine prior tool activity if the model asks about it.
        """
        if chat_id is None:
            chat_id = str(uuid.uuid4())
        if tool_ids is None:
            tool_ids = DEFAULT_TOOL_IDS
        if terminal_id is None:
            terminal_id = DEFAULT_TERMINAL_ID

        history = self._history.get(chat_id, [])
        user_message = {"role": "user", "content": message}
        messages = history + [user_message]
        files_payload = None
        if file_ids:
            # Open WebUI's backend copies a client-supplied TOP-LEVEL `files`
            # array verbatim into `metadata['files']` before invoking filters
            # (open_webui/main.py: `'files': form_data.get('files', None)`,
            # confirmed by reading the deployed container's source 2026-09-23)
            # - there is no server-side lookup/enrichment from the file id, so
            # the entry itself must carry the filename. A nested `metadata`
            # key set directly by the client is NOT this field (verified live:
            # produced metadata.files=0) - main.py builds `metadata` itself
            # and only reads `files` out of the raw form_data.
            files_payload = [
                {
                    "type": "file",
                    "id": fid,
                    "name": self._file_meta.get(fid, fid),
                    "filename": self._file_meta.get(fid, fid),
                }
                for fid in file_ids
            ]

        for _round in range(MAX_TOOL_ROUNDS):
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "chat_id": chat_id,
            }
            if files_payload:
                payload["files"] = files_payload
            if tool_ids:
                payload["tool_ids"] = tool_ids
            if terminal_id:
                payload["terminal_id"] = terminal_id

            resp = self._client.post("/api/chat/completions", json=payload)
            if resp.status_code != 200:
                raise LiveClientError(
                    f"chat failed {resp.status_code}: {resp.text[:500]}"
                )
            body = resp.json()
            choice = body["choices"][0]
            msg = choice["message"]
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls or not execute_tools:
                reply = msg.get("content") or ""
                assistant_message = {"role": "assistant", "content": reply}
                self._history[chat_id] = messages + [assistant_message]
                return chat_id, reply

            # Real tool call(s) - execute for real and continue the loop,
            # per Open WebUI's own API-caller contract (module docstring).
            assistant_message = {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
            messages = messages + [assistant_message]
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                output = _call_terminal_tool(name, args)
                messages = messages + [
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": output,
                    }
                ]

        raise LiveClientError(
            f"chat_id={chat_id}: model kept calling tools past MAX_TOOL_ROUNDS="
            f"{MAX_TOOL_ROUNDS} without a final answer - probable runaway loop"
        )

    def close(self):
        self._client.close()
