# tests/live/test_m12_m2_playwright.py
"""Live defense-in-depth verification driven through `live_api_client`.
Despite the filename (kept for traceability with the plan this grew out of),
this drives the deployed system through the live API, not a browser: the live
API is the surface that exercises real tool_ids/terminal_id wiring.

What it checks is product-generic: a turn with no terminal wired up must get
the hard refusal, never a narrated answer.

This is skipped entirely unless LIVE_DEPLOYMENT and OPENWEBUI_API_KEY are
set (conftest.py), so it never runs or spends money by accident.

KNOWN BLOCKER seen on an earlier deployment: every call through this file failed
with `403 You do not have permission to access this resource` from
OPENWEBUI_API_KEY - confirmed not model-specific (reproduced on /api/models
and /api/chat/completions with no model argument at all). If that recurs it
blocks every file in tests/live/, not just this one - it needs the operator's
API-key permissions, not a workaround here.
"""
import os
import subprocess

import pytest

from conftest import SANDBOX_CONTAINER, SSH_HOST

pytestmark = pytest.mark.skipif(
    not os.environ.get("LIVE_DEPLOYMENT"), reason="requires live deployment")


def _code_sandbox_logs(since_seconds: int = 30) -> str:
    """Backend-truth check for whether the sandbox actually ran code, same
    rationale as conftest.py's `fetch_open_webui_logs`: a plausible reply is
    not evidence the code path ran.
    """
    result = subprocess.run(
        ["ssh", SSH_HOST,
         f"docker logs --since {int(since_seconds)}s {SANDBOX_CONTAINER} 2>&1"],
        capture_output=True, timeout=30, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not fetch sandbox logs: {result.stderr[:300]}")
    return result.stdout


def test_terminal_off_hard_refusal(live_client):
    """(CRITICAL) with no terminal wired up for this turn, a code question
    must get the exact hard refusal, never a narrated/fabricated answer.

    NOTE: `terminal_off` in rag_filter.py is computed ONLY from
    `__model__["meta"]["capabilities"]["terminal"] is False` - the per-model
    Admin -> Models -> Capabilities toggle - never from the request's
    `terminal_id`. `terminal_id=""` below omits the terminal from THIS turn's
    payload (so no sandbox call can actually succeed even if the model tries),
    but it does NOT set `terminal_off=True`; for a model whose Admin
    capability has Terminal checked (the normal state), `_route()` still sees
    `terminal_off=False` and injects the SOFT hint, not the hard refusal.
    This test therefore exercises "no terminal available this turn" but not
    the exact `terminal_off` code path. Real coverage of that path needs a
    model with Terminal unchecked in Admin, or a live browser session with
    Terminal toggled off for the conversation. Left in as the closest
    live-API approximation; do not read a pass here as confirming the
    CRITICAL path.
    """
    _, reply = live_client.chat(
        "How many rows are in this file?",
        model="claude-sonnet",
        terminal_id="",
    )
    assert "can't run code" in reply.lower(), (
        f"must refuse with the hard-refusal sentence; got: {reply[:500]}"
    )
    assert "```" not in reply, f"must not include a code block: {reply[:500]}"
