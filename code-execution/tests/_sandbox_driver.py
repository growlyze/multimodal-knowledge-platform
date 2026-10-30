# Tiny stdlib-only driver, pushed into the sandbox container by the live
# tests in test_sandbox_runtime.py (`docker exec -i ... sh -c 'cat > ...'`)
# and run there with `docker exec <container> python3 /tmp/_sandbox_driver.py
# <cmd> ...`.
#
# Exists to keep the pytest side free of nested shell/JSON quoting: every
# earlier attempt at building a curl+JSON one-liner through
# ssh -> bash -> docker exec -> sh -c -> curl -d '<json>' broke the moment
# the payload itself contained a quote character (a single-quoted uuid, a
# double-quoted Python string) - the outer shell's quoting and the JSON
# payload's quoting collide two layers deep. Talking HTTP from *inside* the
# container in Python instead removes every one of those layers: arguments
# arrive as plain argv strings (optionally base64, for the shell command
# text itself), and json.dumps/urllib own escaping end to end.
#
# Talks to Open Terminal 0.13.0's real contract (P43): POST /execute takes {"command": "<shell>"}
# and returns a background job id; GET /execute/{id}/status polls it
# (wait capped at 300s server-side); DELETE /execute/{id}?force=<bool> kills
# it. This is NOT the {"language","code"} contract the tool used to assume.
import base64
import json
import os
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"
KEY = os.environ["OPEN_TERMINAL_API_KEY"]


def _req(method, path, params=None, body=None):
    url = BASE + path
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=310) as resp:
        return json.load(resp)


def main(argv):
    cmd = argv[1]
    if cmd == "launch":
        # argv[2]: base64-encoded shell command. argv[3]: wait seconds (<=300).
        shell_cmd = base64.b64decode(argv[2]).decode()
        wait = argv[3] if len(argv) > 3 else "0"
        print(json.dumps(_req("POST", "/execute", {"wait": wait},
                               {"command": shell_cmd})))
    elif cmd == "status":
        wait = argv[3] if len(argv) > 3 else "0"
        print(json.dumps(_req("GET", f"/execute/{argv[2]}/status", {"wait": wait})))
    elif cmd == "cancel":
        force = argv[3] if len(argv) > 3 else "false"
        print(json.dumps(_req("DELETE", f"/execute/{argv[2]}", {"force": force})))
    else:
        raise SystemExit(f"unknown driver command: {cmd!r}")


if __name__ == "__main__":
    main(sys.argv)
