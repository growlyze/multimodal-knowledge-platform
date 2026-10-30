"""Durable download copies (P160): a display_file card is only a REFERENCE to a
sandbox path, and the sandbox's /home/user is tmpfs, so every restart (and the
7-day sandbox cleanup) turns older chats' Download buttons into 404s. outlet()
copies each successfully displayed text file into Open WebUI's own file
storage and appends a link to that copy.

Offline, pure-logic: the sandbox fetch, the Open WebUI store and the file-table
listing are replaced on the Filter instance, the same way the other suites
stub _search. No network, no DB, no open_webui package.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import (
    DURABLE_MAX_BYTES,
    DURABLE_RETENTION_DAYS,
    Filter,
    _display_file_targets,
    _durable_meta,
    _durable_skip_reason,
    _is_durable_copy,
    _select_expired_copies,
)

DAY = 86400


def _display_pair(call_id="call_1", path="/home/user/outputs/trade_ledger.csv",
                  exists=True, name="display_file", source="open_terminal",
                  inline=True):
    """The two output items Open WebUI 0.11.4 stores for one display_file call:
    the function_call, then a function_call_output whose input_text is the JSON
    build_terminal_file_tool_result() produced (utils/middleware.py)."""
    result = {
        "path": path,
        "exists": exists,
        "type": "file",
        "source": source,
        "terminal_selector": "sandbox",
        "terminal_id": "sandbox",
        "session_id": "chat-1",
        "full_path": path,
        "name": path.rsplit("/", 1)[-1],
        "mime_type": "text/csv",
        "content_type": "text/csv",
    }
    if inline:
        result["displayed"] = True
    return [
        {"type": "function_call", "id": "fc_1", "call_id": call_id, "name": name,
         "arguments": json.dumps({"path": path, "inline": inline}), "status": "completed"},
        {"type": "function_call_output", "id": "fco_1", "call_id": call_id,
         "output": [{"type": "input_text", "text": json.dumps(result)}],
         "status": "completed"},
    ]


def _reply_message(output):
    return {"type": "message", "id": "msg_x", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "Here is your ledger."}]}


def _body(output):
    return {
        "chat_id": "chat-1",
        "id": "m-2",
        "messages": [
            {"id": "m-1", "role": "user", "content": "export the trades as csv"},
            {"id": "m-2", "role": "assistant", "content": "Here is your ledger.",
             "output": output + [_reply_message(output)]},
        ],
    }


USER = {"id": "user-1", "email": "u@example.com", "name": "U", "role": "user"}
META = {"chat_id": "chat-1", "terminal_id": "sandbox"}


def _filter(data=b"a,b\n1,2\n"):
    """A Filter whose sandbox fetch returns `data` and whose store records calls."""
    f = Filter()
    calls = {"fetch": [], "store": []}

    async def fetch(target, user, metadata, request, max_bytes):
        calls["fetch"].append((target, max_bytes))
        return data

    async def store(user_id, name, payload, meta):
        calls["store"].append((user_id, name, payload, meta))
        return f"file-{len(calls['store'])}"

    f._fetch_sandbox_file = fetch
    f._store_durable_file = store
    return f, calls


# ── detection ────────────────────────────────────────────────────────────


def test_detects_successful_display_file():
    targets = _display_file_targets(_display_pair())
    assert len(targets) == 1
    t = targets[0]
    assert t["call_id"] == "call_1"
    assert t["path"] == "/home/user/outputs/trade_ledger.csv"
    assert t["name"] == "trade_ledger.csv"
    assert t["terminal_id"] == "sandbox"


def test_ignores_missing_file_other_tools_and_foreign_sources():
    assert _display_file_targets(_display_pair(exists=False)) == []
    assert _display_file_targets(_display_pair(name="read_file")) == []
    assert _display_file_targets(_display_pair(source="something_else")) == []
    # A call with no matching output (still running / failed to record).
    assert _display_file_targets(_display_pair()[:1]) == []
    broken = _display_pair()
    broken[1]["output"][0]["text"] = "not json"
    assert _display_file_targets(broken) == []
    assert _display_file_targets(None) == []


def test_skips_calls_already_copied():
    """outlet() can run twice on one message (continue/regenerate paths); the
    marker on our own appended item stops a second copy of the same call."""
    output = _display_pair() + [{
        "type": "message", "id": "msg_dl", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "x"}],
        rag_filter.DURABLE_MARKER_KEY: ["call_1"],
    }]
    assert _display_file_targets(output) == []


# ── what to copy ─────────────────────────────────────────────────────────


def test_text_formats_only():
    for name in ("a.csv", "b.py", "c.xlsx", "d.json", "e.md", "f.TXT"):
        assert _durable_skip_reason(name, 10) is None, name
    assert _durable_skip_reason("chart.png", 10) is not None
    assert _durable_skip_reason("archive.zip", 10) is not None
    assert _durable_skip_reason("noext", 10) is not None


def test_size_cap():
    assert _durable_skip_reason("a.csv", DURABLE_MAX_BYTES) is None
    assert "MB" in _durable_skip_reason("a.csv", DURABLE_MAX_BYTES + 1)
    assert _durable_skip_reason("a.csv", 0) is not None  # Storage rejects empty


# ── tagging + retention selection ────────────────────────────────────────


def test_tag_identifies_only_our_copies():
    meta = _durable_meta("t.csv", 12, "chat-1", "m-2", "call_1", "/home/user/outputs/t.csv")
    assert _is_durable_copy(meta)
    assert meta["name"] == "t.csv" and meta["size"] == 12
    # A real upload's meta (routers/files.py upload_file_handler shape).
    upload = {"name": "t.csv", "content_type": "text/csv", "size": 12,
              "file_hash": "abc", "data": {}}
    assert not _is_durable_copy(upload)
    # Upload metadata the client controls lands under meta["data"], never at
    # the top level — a user cannot mark their own upload for deletion.
    upload["data"] = {rag_filter.DURABLE_MARKER_KEY: {"kind": "durable_download"}}
    assert not _is_durable_copy(upload)
    assert not _is_durable_copy(None)
    assert not _is_durable_copy({rag_filter.DURABLE_MARKER_KEY: "durable_download"})


def test_retention_selects_old_tagged_copies_only():
    now = 1_000 * DAY
    tagged = _durable_meta("t.csv", 1, "c", "m", "k", "/p")
    rows = [
        {"id": "old-copy", "path": "/u/1", "meta": tagged,
         "created_at": now - (DURABLE_RETENTION_DAYS + 1) * DAY},
        {"id": "new-copy", "path": "/u/2", "meta": tagged,
         "created_at": now - (DURABLE_RETENTION_DAYS - 1) * DAY},
        {"id": "old-upload", "path": "/u/3", "meta": {"name": "x.csv"},
         "created_at": now - 400 * DAY},
        {"id": "no-date", "path": "/u/4", "meta": tagged, "created_at": None},
    ]
    assert [r["id"] for r in _select_expired_copies(rows, now)] == ["old-copy"]


# ── outlet orchestration ─────────────────────────────────────────────────


def _outlet(f, body):
    return asyncio.run(f.outlet(body, __user__=USER, __metadata__=META, __request__=None))


def _assistant(body):
    return body["messages"][-1]


def test_outlet_appends_durable_link_to_output_and_content():
    """The frontend renders `output` and ignores `content` whenever output
    exists (ContentRenderer.svelte), so the link must be an output item;
    content changes too, because the chat:outlet event only patches the
    in-memory message when content differs (Chat.svelte)."""
    f, calls = _filter()
    body = _outlet(f, _body(_display_pair()))
    asst = _assistant(body)
    assert len(calls["store"]) == 1
    user_id, name, payload, meta = calls["store"][0]
    assert (user_id, name, payload) == ("user-1", "trade_ledger.csv", b"a,b\n1,2\n")
    assert _is_durable_copy(meta)
    link = "/api/v1/files/file-1/content?attachment=true"
    last = asst["output"][-1]
    assert last["type"] == "message"
    assert link in last["content"][0]["text"]
    assert last[rag_filter.DURABLE_MARKER_KEY] == ["call_1"]
    assert link in asst["content"]


def test_outlet_is_idempotent():
    f, calls = _filter()
    body = _outlet(f, _body(_display_pair()))
    _outlet(f, body)
    assert len(calls["store"]) == 1
    assert sum(1 for o in _assistant(body)["output"]
               if rag_filter.DURABLE_MARKER_KEY in o) == 1


def test_non_text_file_is_never_fetched():
    f, calls = _filter()
    body = _outlet(f, _body(_display_pair(path="/home/user/outputs/chart.png")))
    assert calls["fetch"] == [] and calls["store"] == []
    assert _assistant(body)["content"] == "Here is your ledger."


def test_oversized_file_is_skipped():
    f, calls = _filter(data=b"x" * (DURABLE_MAX_BYTES + 1))
    body = _outlet(f, _body(_display_pair()))
    assert calls["fetch"][0][1] == DURABLE_MAX_BYTES
    assert calls["store"] == []
    assert _assistant(body)["content"] == "Here is your ledger."


def test_sandbox_miss_or_store_failure_leaves_reply_untouched():
    f, calls = _filter(data=None)
    body = _outlet(f, _body(_display_pair()))
    assert calls["store"] == []
    assert _assistant(body)["content"] == "Here is your ledger."

    f, _ = _filter()

    async def boom(*_a, **_kw):
        raise RuntimeError("storage down")

    f._store_durable_file = boom
    body = _outlet(f, _body(_display_pair()))
    assert _assistant(body)["content"] == "Here is your ledger."


def test_disabled_valve_does_nothing():
    f, calls = _filter()
    f.valves.enable_durable_downloads = False
    _outlet(f, _body(_display_pair()))
    assert calls["fetch"] == [] and calls["store"] == []


# ── retention runner ─────────────────────────────────────────────────────


def test_prune_deletes_only_selected_and_runs_at_most_daily():
    f = Filter()
    now = 1_000 * DAY
    tagged = _durable_meta("t.csv", 1, "c", "m", "k", "/p")
    rows = [
        {"id": "old-copy", "path": "/u/1", "meta": tagged, "created_at": now - 91 * DAY},
        {"id": "old-upload", "path": "/u/3", "meta": {"name": "x"}, "created_at": now - 400 * DAY},
    ]
    listed, deleted = [], []

    async def list_rows(cutoff):
        listed.append(cutoff)
        return rows

    async def delete_row(row):
        deleted.append(row["id"])

    f._list_file_rows = list_rows
    f._delete_file_row = delete_row
    assert asyncio.run(f._maybe_prune_durable(now)) == 1
    assert deleted == ["old-copy"]
    assert listed == [now - DURABLE_RETENTION_DAYS * DAY]
    # Same day: no second listing.
    assert asyncio.run(f._maybe_prune_durable(now + 3600)) == 0
    assert len(listed) == 1
    # A day later it runs again.
    asyncio.run(f._maybe_prune_durable(now + DAY + 1))
    assert len(listed) == 2


def test_prune_failure_is_contained():
    f = Filter()

    async def list_rows(cutoff):
        raise RuntimeError("db down")

    f._list_file_rows = list_rows
    assert asyncio.run(f._maybe_prune_durable(1_000 * DAY)) == 0


# ── Excel workbooks (C32) ────────────────────────────────────────────────
# The export tool writes a real .xlsx; display_file's card reads it
# back from the tmpfs sandbox like any other file, so without a durable copy
# an Excel download dies with the next sandbox restart (P160) while the CSV
# beside it survives. The copy path is bytes end to end (/files/view ->
# Storage.upload_file), so a workbook needs no text decoding - only the
# extension allow-list and the content type had to learn it.

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# A zip local-file header plus bytes that are NOT valid UTF-8: any step that
# decodes/re-encodes as text would change them.
XLSX_BYTES = b"PK\x03\x04\x14\x00\x06\x00\xff\xfe\x80\x81binary\x00tail"


def test_xlsx_is_copied():
    assert _durable_skip_reason("results.xlsx", 10) is None
    assert _durable_skip_reason("REPORT.XLSX", 10) is None
    # Legacy/macro workbooks are not what export_xlsx writes; keep them out.
    assert _durable_skip_reason("old.xls", 10) is not None
    assert _durable_skip_reason("macro.xlsm", 10) is not None


def test_xlsx_copy_is_labelled_as_a_workbook():
    meta = _durable_meta("r.xlsx", 5, "c", "m", "k", "/home/user/outputs/r.xlsx")
    assert meta["content_type"] == XLSX_MIME
    # Existing text types keep their labels exactly.
    assert _durable_meta("t.csv", 1, "c", "m", "k", "/p")["content_type"] == "text/csv"


def test_outlet_copies_an_xlsx_byte_for_byte():
    f, calls = _filter(data=XLSX_BYTES)
    body = _outlet(f, _body(_display_pair(path="/home/user/outputs/results.xlsx")))
    assert len(calls["store"]) == 1
    _, name, payload, meta = calls["store"][0]
    assert name == "results.xlsx"
    assert payload == XLSX_BYTES
    assert meta["content_type"] == XLSX_MIME
    assert "**Download:** [results.xlsx](/api/v1/files/file-1/content?attachment=true)" \
        in _assistant(body)["content"]


def test_csv_copy_is_unchanged_by_the_xlsx_support():
    """The client compares CSV exports byte for byte - the CSV path must hand
    Storage exactly the bytes the sandbox returned, and the link line must
    keep its existing shape."""
    raw = b"entry_time,pnl\r\n2026-01-02 09:30:00,12.5\r\n"
    f, calls = _filter(data=raw)
    body = _outlet(f, _body(_display_pair()))
    assert calls["store"][0][2] == raw
    assert "**Download:** [trade_ledger.csv](/api/v1/files/file-1/content?attachment=true)" \
        in _assistant(body)["content"]
