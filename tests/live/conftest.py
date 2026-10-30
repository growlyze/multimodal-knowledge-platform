# tests/live/conftest.py
"""Shared fixtures for the live API test suite. Every test here is SKIPPED
entirely unless both LIVE_DEPLOYMENT and OPENWEBUI_API_KEY are set - never
runs by accident, never spends money on a machine that didn't opt in.

Deployment target is fully environment-driven (generic defaults for a
`gmac`-prefixed stack):
    SSH_HOST              ssh alias of the deployment host   (default: gmac)
    OPENWEBUI_CONTAINER   open-webui container to read logs from
    SANDBOX_CONTAINER     code-sandbox container (see the playwright file)
    OPENWEBUI_TERMINAL_ID sandbox terminal connection id in Open WebUI
    OPENWEBUI_BASE_URL    public base URL of Open WebUI (handed to
                          haystack-api/scripts/live_api_client.py, which
                          defaults to it via the environment)
    STACK_NAME            container-name prefix              (default: gmac)

Only product-generic acceptance behaviors live here; nothing asserts a
particular client's knowledge base content.
"""
import os
import subprocess
import time

import pytest

STACK_NAME = os.environ.get("STACK_NAME", "gmac")
SSH_HOST = os.environ.get("SSH_HOST", "gmac")
OPENWEBUI_CONTAINER = os.environ.get("OPENWEBUI_CONTAINER", f"{STACK_NAME}-open-webui")
SANDBOX_CONTAINER = os.environ.get("SANDBOX_CONTAINER", f"{STACK_NAME}-code-sandbox-1")
# Generic default matching the Caddyfile's {$DOMAIN:localhost}. Exported so
# live_api_client picks it up instead of any baked-in deployment URL.
os.environ.setdefault("OPENWEBUI_BASE_URL", "https://localhost")


def pytest_collection_modifyitems(config, items):
    # Requires a live deployment (SSH + running stack) - see module docstring.
    skip_live = pytest.mark.skipif(
        not os.environ.get("LIVE_DEPLOYMENT"), reason="requires live deployment")
    if not os.environ.get("OPENWEBUI_API_KEY"):
        skip = pytest.mark.skip(reason="OPENWEBUI_API_KEY not set - live suite skipped")
        for item in items:
            item.add_marker(skip)
    for item in items:
        item.add_marker(skip_live)


@pytest.fixture
def live_client():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "haystack-api"))
    from scripts.live_api_client import LiveClient
    client = LiveClient()
    yield client
    client.close()


def fetch_open_webui_logs(since_seconds: int = 60) -> str:
    """The backend-truth half of every live assertion. Uses the SSH access
    already in place - never trust the model's reply text alone: a plausible
    reply is not evidence the filter actually ran.
    """
    result = subprocess.run(
        ["ssh", SSH_HOST,
         f"docker logs --since {int(since_seconds)}s {OPENWEBUI_CONTAINER} 2>&1"],
        capture_output=True, timeout=30,
        # encoding and errors are NOT optional here. `text=True` alone decodes
        # with the platform default, which on this project's Windows dev
        # machines is cp1252 - and container logs carry bytes cp1252 cannot
        # represent (a 0x8f killed a criterion check). The decode raises inside
        # subprocess's reader THREAD, so the failure does not surface as a
        # decode error: stdout comes back None and the caller dies later on
        # `json.loads(None)` with "the JSON object must be str... not
        # NoneType". That reads as a product failure in an acceptance test.
        # Logs are UTF-8 from Docker; errors="replace" keeps one odd byte from
        # taking down a whole run, since this output is only ever searched.
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not fetch deployment logs: {result.stderr[:300]}")
    return result.stdout


def assert_filter_log(pattern: str, since_seconds: int = 60, timeout: float = 15.0):
    """Poll the deployment logs until `pattern` appears, or fail. Filter
    execution is async relative to the HTTP response, so a short poll (not a
    single fetch) avoids a flaky false negative on a slow turn.
    """
    deadline = time.time() + timeout
    last_logs = ""
    while time.time() < deadline:
        last_logs = fetch_open_webui_logs(since_seconds)
        if pattern in last_logs:
            return last_logs
        time.sleep(2)
    # This slice is real production Open WebUI log content and can include a
    # fragment of a real user's prompt (rag_filter.py logs query[:60]). Kept
    # short and test-output-only (this text never leaves the pytest failure
    # report - not committed, not logged elsewhere) as a tradeoff against the
    # debuggability a longer window would give; shrink further if that
    # tradeoff ever looks wrong in practice.
    raise AssertionError(
        f"pattern {pattern!r} did not appear in open-webui logs within {timeout}s.\n"
        f"Last logs fetched:\n{last_logs[-500:]}")
