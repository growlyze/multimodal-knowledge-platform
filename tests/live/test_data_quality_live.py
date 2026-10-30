# tests/live/test_data_quality_live.py
"""Live, API-driven versions of the data-quality and cross-provider tests
that were previously run manually in the browser (one file, three messages,
three models)."""
import os
import pytest

from conftest import assert_filter_log

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("LIVE_DEPLOYMENT"), reason="requires live deployment"),
    pytest.mark.skipif(
        not os.environ.get("ALLOW_LIVE_FILE_TESTS"),
        reason="uploads a file - set ALLOW_LIVE_FILE_TESTS=1 to include"),
]


def test_model_switch_mid_conversation_with_tool_call_does_not_crash(live_client, tmp_path):
    """The exact repro that produced:
    messages.N.content.0.tool_use.id: String should match pattern '^[a-zA-Z0-9_-]+$'
    Claude -> Gemini (forces a tool call) -> Claude, same thread.
    This will be the first fully-conclusive run of this test: _sanitize_tool_call_ids
    now logs when it actually rewrites an id (2026-09-22 commit), so this test
    will be able to tell "no crash because nothing needed fixing" apart from
    "no crash because the sanitiser worked" - which no prior manual run could.
    No API key exists on this machine yet, so this has not actually run live.
    """
    import pandas as pd
    df = pd.DataFrame({
        "Open": [10] * 30, "High": [10.5] * 30, "Low": [9.5] * 30,
        "Close": [10.2] * 30, "Volume": [100] * 30,
    }, index=pd.date_range("2024-01-01", periods=30, freq="1min"))
    path = tmp_path / "switch_test.csv"
    df.to_csv(path)
    file_id = live_client.upload_file(str(path))

    chat_id, reply1 = live_client.chat(
        "how many rows are in this file?", model="claude-haiku",
        file_ids=[file_id])
    assert "30" in reply1, reply1

    chat_id, reply2 = live_client.chat(
        "run a tool to tell me the average close price", model="gemini-flash",
        chat_id=chat_id)
    assert reply2, "gemini turn must produce a tool-using reply"

    # The turn that used to 400 the whole request.
    chat_id, reply3 = live_client.chat(
        "what was the highest high?", model="claude-sonnet", chat_id=chat_id)
    assert "tool_use.id" not in reply3, f"the P29 crash reproduced: {reply3}"
    assert "10.5" in reply3, reply3

    logs = assert_filter_log("attachments in request", since_seconds=120)
    # Whichever it was, it must be recorded - not silent either way (D46-era fix).
    sanitiser_ran = "rewrote" in logs
    print(f"sanitiser rewrote an id this run: {sanitiser_ran}")


def test_large_file_what_is_in_this_file_no_prompt_too_long(live_client):
    """The top reported issue: 'what's in this file?' on a 3.5MB+ CSV used to
    either error 'prompt is too long' (Claude) or hit a real Google 429 quota
    error (Gemini) - the file's raw bytes were being read into the prompt
    instead of routed to load_file(). Uses a SYNTHETIC file at the same size
    class, not a real user data file (those stay local and are never
    committed) - size is what triggers the bug, not the specific data.
    """
    import pandas as pd
    import numpy as np
    n = 60_000  # large enough to be multi-MB as CSV, matching the reported 3.5-11MB range
    idx = pd.date_range("2020-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "Open": np.random.uniform(90, 110, n), "High": np.random.uniform(90, 110, n),
        "Low": np.random.uniform(90, 110, n), "Close": np.random.uniform(90, 110, n),
        "Volume": np.random.randint(100, 10000, n),
    }, index=idx)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name)
        path = f.name
    try:
        file_id = live_client.upload_file(path)

        _, reply = live_client.chat(
            "what's in this file?", model="claude-haiku", file_ids=[file_id])
        assert "prompt is too long" not in reply.lower(), reply
        logs = assert_filter_log("routed as 'structured'", since_seconds=90)
        # Informational only: exact log wording for "the file was routed instead
        # of read into the prompt" isn't pinned down yet (no live run has
        # confirmed it) - not asserted on, so a wording mismatch here doesn't
        # mask the real check above (the reply not containing "prompt is too
        # long"). Whoever runs this live first should tighten this once the
        # actual log line is known.
        print(f"skipped retrieval/flagged present: "
              f"{'skipped retrieval' in logs or 'flagged' in logs}")
    finally:
        os.remove(path)


def test_extra_unknown_columns_via_real_upload(live_client):
    from pathlib import Path
    fixture = Path(__file__).parent.parent.parent / "haystack-api" / "tests" / "fixtures" / "malformed" / "extra_unknown_columns.csv"
    file_id = live_client.upload_file(str(fixture))
    _, reply = live_client.chat(
        "what columns are in this file?", model="claude-haiku",
        file_ids=[file_id])
    # Backend-truth half (P35 lesson): a .csv upload must actually route through
    # rag_filter.py's route() as "structured" (filters/rag_filter.py line ~1115),
    # same pattern already confirmed for the criteria 5/6 uploads in
    # test_m1_2_acceptance.py - a plausible reply alone is not proof the filter ran.
    assert_filter_log("routed as 'structured'", since_seconds=90)
    assert "RSI" in reply or "Custom_Indicator" in reply, \
        f"unknown columns should be visible to the model, not dropped: {reply}"


def test_missing_column_is_reported_to_the_user(live_client):
    """The direct test of Task 3's change: the model must SAY the column was
    missing, not silently answer around it."""
    from pathlib import Path
    fixture = Path(__file__).parent.parent.parent / "haystack-api" / "tests" / "fixtures" / "malformed" / "missing_volume.csv"
    file_id = live_client.upload_file(str(fixture))
    _, reply = live_client.chat(
        "what was the total volume?", model="claude-haiku",
        file_ids=[file_id])
    # Same backend-truth check as test_extra_unknown_columns_via_real_upload -
    # this is a .csv upload too, so it must show the same "routed as 'structured'" line.
    assert_filter_log("routed as 'structured'", since_seconds=90)
    assert "volume" in reply.lower() and (
        "missing" in reply.lower() or "not" in reply.lower()
        or "no volume" in reply.lower() or "doesn't" in reply.lower()), \
        f"model must say Volume was missing, not just fail quietly: {reply}"
