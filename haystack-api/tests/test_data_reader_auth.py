"""DATA_API_KEY: the code sandbox's read-only key.

It must open exactly the two stored-dataset endpoints data_tools.load_dataset()
calls, and nothing else under /admin — the sandbox runs user-submitted code.
"""

import config
from fastapi.testclient import TestClient

from main import app
from routers import admin
from services.db_service import DBError

DATA_KEY = "test-data-key"


class BrokenDB:
    """Any DB call fails: a 503 proves the request got past authentication."""

    async def list_tables(self, filename=None):
        raise DBError("offline")

    async def query(self, *_a, **_kw):
        raise DBError("offline")


def _client(monkeypatch):
    monkeypatch.setattr(config, "DATA_API_KEY", DATA_KEY)
    admin.get_db = lambda: BrokenDB()
    return TestClient(app, raise_server_exceptions=False)


def _auth(key):
    return {"Authorization": "Bearer %s" % key}


def test_data_key_reaches_the_dataset_endpoints(monkeypatch):
    client = _client(monkeypatch)
    for path in ("/admin/data/search?q=sales", "/admin/data/table/data_sales"):
        r = client.get(path, headers=_auth(DATA_KEY))
        assert r.status_code not in (401, 403), (path, r.status_code)


def test_data_key_is_refused_everywhere_else(monkeypatch):
    client = _client(monkeypatch)
    for method, path in (("GET", "/admin/stats"), ("GET", "/admin/documents"),
                         ("DELETE", "/admin/documents/x.pdf"), ("GET", "/admin/tables"),
                         ("POST", "/admin/reindex")):
        r = client.request(method, path, headers=_auth(DATA_KEY))
        assert r.status_code == 401, (method, path, r.status_code)


def test_dataset_endpoints_need_a_key(monkeypatch):
    client = _client(monkeypatch)
    assert client.get("/admin/data/search?q=sales").status_code == 401
    assert client.get("/admin/data/search?q=sales",
                      headers=_auth("wrong")).status_code == 401


def test_admin_key_still_reads_datasets(monkeypatch):
    client = _client(monkeypatch)
    r = client.get("/admin/data/search?q=sales", headers=_auth(config.ADMIN_API_KEY))
    assert r.status_code not in (401, 403), r.status_code


# ── standalone runner (python tests/test_data_reader_auth.py) ──

class _MonkeyPatch:
    """Just enough of pytest's monkeypatch for _client()."""

    def __init__(self):
        self._saved = []

    def setattr(self, target, name, value):
        self._saved.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, value in reversed(self._saved):
            setattr(target, name, value)


def main():
    # Standalone: config reads ADMIN_API_KEY at import time and it is empty
    # here (under pytest, tests/conftest.py sets it). /admin/stats must fail
    # with 401 for the data key — not with the fail-closed 503.
    config.ADMIN_API_KEY = config.ADMIN_API_KEY or "test-admin-key"
    tests = [
        test_data_key_reaches_the_dataset_endpoints,
        test_data_key_is_refused_everywhere_else,
        test_dataset_endpoints_need_a_key,
        test_admin_key_still_reads_datasets,
    ]
    for test in tests:
        patch = _MonkeyPatch()
        try:
            test(patch)
        finally:
            patch.undo()
        print("ok  %s" % test.__name__)
    print("\n%d data-reader auth checks passed" % len(tests))


if __name__ == "__main__":
    main()
