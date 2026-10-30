"""haystack-api protects /ingest/* and /eval/log-answer with a bearer key
(services/auth.require_admin_or_loader). The filter must send it, or every
chat upload is refused with 401 and never reaches the knowledge base.

Offline: httpx.AsyncClient is replaced by a recorder. No network.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

import rag_filter


class _Response:
    status_code = 200
    text = ""

    def json(self):
        return {"job_id": "job-1"}


def _recording_client(seen):
    class Client:
        def __init__(self, **kw):
            seen.append(kw.get("headers") or {})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, url, **_kw):
            return _Response()

    return Client


def _ingest(f):
    asyncio.run(
        f._post("document", "notes.txt", {"data": {"content": "hello"}}, {"chat_id": "c1"})
    )


def test_key_defaults_to_the_external_loader_key(monkeypatch):
    monkeypatch.setenv("EXTERNAL_LOADER_API_KEY", "loader-123")
    assert rag_filter.Filter().valves.haystack_api_key == "loader-123"


def test_ingest_sends_the_bearer_key(monkeypatch):
    seen = []
    monkeypatch.setattr(httpx, "AsyncClient", _recording_client(seen))
    f = rag_filter.Filter()
    f.valves.haystack_api_key = "loader-123"
    _ingest(f)
    assert seen == [{"Authorization": "Bearer loader-123"}], seen


def test_no_key_sends_no_header(monkeypatch):
    seen = []
    monkeypatch.setattr(httpx, "AsyncClient", _recording_client(seen))
    f = rag_filter.Filter()
    f.valves.haystack_api_key = ""
    _ingest(f)
    assert seen == [{}], seen
