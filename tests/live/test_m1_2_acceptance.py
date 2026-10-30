# tests/live/test_m1_2_acceptance.py
"""Product acceptance behaviors driven through the live API.

Product-generic behaviors: a pure knowledge question must route to retrieval
only, and two concurrent sessions must not interfere.
"""
# Model names are the LiteLLM aliases in litellm-config.yaml ("claude-haiku",
# "gemini-flash"); override with LIVE_MODEL_A / LIVE_MODEL_B if a deployment
# exposes different ones.
import os

import pytest

from conftest import assert_filter_log, fetch_open_webui_logs

MODEL_A = os.environ.get("LIVE_MODEL_A", "claude-haiku")
MODEL_B = os.environ.get("LIVE_MODEL_B", "gemini-flash")

pytestmark = pytest.mark.skipif(
    not os.environ.get("LIVE_DEPLOYMENT"), reason="requires live deployment")


def test_pure_knowledge_question_no_code_execution(live_client):
    """A pure knowledge question -> RAG only, no sandbox call."""
    _, reply = live_client.chat(
        "what is a firewall in computer networking?", model=MODEL_B)
    logs = assert_filter_log("attachments in request")
    assert "run_command" not in logs.split("attachments in request")[-1][:2000], \
        "a pure knowledge question must trigger NO sandbox execution"
    assert len(reply) > 20, "must actually answer, not refuse"


def test_two_concurrent_sessions_no_interference():
    """Two concurrent chats, different questions, no cross-talk."""
    import sys, os, threading, time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "haystack-api"))
    from scripts.live_api_client import LiveClient
    results = {}

    def run(key, question, model):
        c = LiveClient()
        _, reply = c.chat(question, model=model)
        results[key] = reply
        c.close()

    # Baseline BEFORE starting either thread. "attachments in request" fires on
    # EVERY turn in this file, including the knowledge-question test which runs
    # immediately before this one - an absolute count taken after the join
    # would be contaminated by those and could pass even if only one (or
    # neither) of THIS test's own two threads reached the filter, which is
    # exactly the interference this test exists to catch. A delta over a
    # baseline captured right here is scoped to this test regardless of what
    # ran earlier in the file.
    #
    # `fetch_open_webui_logs`'s `--since Ns` is a ROLLING window relative to
    # "now" at fetch time, not a fixed range - confirmed live: a baseline of 4
    # followed by an after-fetch with the same since_seconds=60 produced
    # delta=1, not >=2, because one of the baseline's own log lines had aged
    # out of the window by the time of the after-fetch (both new lines were
    # genuinely present in the raw log tail). Fixed by widening the
    # after-fetch's window by exactly the elapsed wall-clock time since the
    # baseline was captured (+ a safety margin), so its lower edge never moves
    # past the baseline's - nothing that was counted in `baseline` can age out.
    start = time.time()
    baseline = fetch_open_webui_logs(since_seconds=60).count("attachments in request")

    t1 = threading.Thread(target=run, args=("a", "what is a firewall in computer networking?",
                                           MODEL_A))
    t2 = threading.Thread(target=run, args=("b", "what is photosynthesis?", MODEL_B))
    t1.start(); t2.start()
    t1.join(timeout=60); t2.join(timeout=60)
    # Poll rather than a single fetch: filter logging is async relative to the
    # HTTP response (same reasoning as assert_filter_log in conftest.py), so a
    # one-shot fetch right after join() risks a flaky false negative on a slow
    # turn even though both really did land.
    deadline = time.time() + 15.0
    delta = 0
    logs_after = ""
    while time.time() < deadline:
        window = 60 + int(time.time() - start) + 5
        logs_after = fetch_open_webui_logs(since_seconds=window)
        delta = logs_after.count("attachments in request") - baseline
        if delta >= 2:
            break
        time.sleep(2)
    assert delta >= 2, (
        f"expected both concurrent turns to add >= 2 new 'attachments in request' "
        f"log lines over the baseline ({baseline}); only saw a delta of {delta} "
        f"within 15s:\n{logs_after[-2000:]}")
    assert "a" in results and "b" in results, "both sessions must complete"
    assert "firewall" in results["a"].lower()
    assert "photosynthesis" in results["b"].lower()
    # the real interference check: session B's answer must not talk about
    # session A's topic
    assert "firewall" not in results["b"].lower()
