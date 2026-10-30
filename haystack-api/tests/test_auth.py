"""API-auth guards — services/auth.py plus the per-router wiring.

Run: python tests/test_auth.py   (also runs under pytest)

What is pinned here:
  1. fail-closed: no configured key -> 503 {"error": "admin api key not configured"}
  2. the admin key opens the admin-protected endpoints
  3. a wrong or missing bearer token is 401 everywhere protected
  4. ingestion and POST /eval/log-answer accept EITHER the admin key or the
     external loader key (the Open WebUI chat filter sends the loader key)
  5. GET /search and GET /health stay open (the chat pipeline needs them)

Status-code routing is the whole test: auth runs before the handler, so an
empty upload answers 400 ("is empty") only once past the guard — that 400 is
the proof the key was accepted, and no ingestion work is ever queued.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from fastapi.testclient import TestClient

from main import app

ADMIN_KEY = "test-admin-key"
LOADER_KEY = "test-loader-key"
WRONG_KEY = "wrong-key"


def _client() -> TestClient:
    # No `with`: lifespan startup (DB pool, planner warm-up) must not run here.
    return TestClient(app, raise_server_exceptions=False)


def _set_keys(admin="", loader=""):
    """Point the two keys at test values; returns the previous pair."""
    old = config.ADMIN_API_KEY, config.EXTERNAL_LOADER_API_KEY
    config.ADMIN_API_KEY = admin
    config.EXTERNAL_LOADER_API_KEY = loader
    return old


def _restore(old):
    config.ADMIN_API_KEY, config.EXTERNAL_LOADER_API_KEY = old


def _auth(key):
    return {"Authorization": "Bearer %s" % key}


# The two routes every test drives:
#   GET  /eval/system     admin-protected, in-memory only (safe to call)
#   POST /ingest/document the either-key door (empty upload -> 400 past auth)
SYSTEM = "/eval/system"
INGEST = "/ingest/document"
LOG_ANSWER = "/eval/log-answer"
ANSWER_BODY = {"query": "q", "answer": "", "log_id": "x"}


def test_fail_closed_without_any_key():
    old = _set_keys(admin="", loader="")
    try:
        client = _client()
        for method, path, kw in (
            ("GET", SYSTEM, {}),
            ("POST", INGEST, {"files": {"file": ("a.txt", b"", "text/plain")}}),
            ("POST", LOG_ANSWER, {"json": ANSWER_BODY}),
        ):
            r = client.request(method, path, **kw)
            assert r.status_code == 503, (path, r.status_code, r.text)
            assert r.json() == {"error": "admin api key not configured"}, r.text
        # ...and a bearer token changes nothing: nothing is configured to match.
        r = client.get(SYSTEM, headers=_auth(ADMIN_KEY))
        assert r.status_code == 503 and r.json() == {"error": "admin api key not configured"}
    finally:
        _restore(old)


def test_admin_key_passes_and_wrong_key_is_401():
    old = _set_keys(admin=ADMIN_KEY, loader=LOADER_KEY)
    try:
        client = _client()
        r = client.get(SYSTEM, headers=_auth(ADMIN_KEY))
        assert r.status_code == 200, (r.status_code, r.text)

        for headers in (_auth(WRONG_KEY), _auth(""), {}):
            r = client.get(SYSTEM, headers=headers)
            assert r.status_code == 401, (headers, r.status_code, r.text)
    finally:
        _restore(old)


def test_loader_key_is_not_admin():
    """EITHER-key applies to ingestion and log-answer only — not to /eval/system."""
    old = _set_keys(admin=ADMIN_KEY, loader=LOADER_KEY)
    try:
        r = _client().get(SYSTEM, headers=_auth(LOADER_KEY))
        assert r.status_code == 401, (r.status_code, r.text)
    finally:
        _restore(old)


def test_ingestion_accepts_either_key():
    old = _set_keys(admin=ADMIN_KEY, loader=LOADER_KEY)
    try:
        client = _client()
        for key in (ADMIN_KEY, LOADER_KEY):
            r = client.post(INGEST, headers=_auth(key),
                            files={"file": ("a.txt", b"", "text/plain")})
            # 400 = the empty-upload refusal, which the handler raises only
            # after the auth guard has accepted the key.
            assert r.status_code == 400, (key, r.status_code, r.text)

        for headers in (_auth(WRONG_KEY), {}):
            r = client.post(INGEST, headers=headers,
                            files={"file": ("a.txt", b"", "text/plain")})
            assert r.status_code == 401, (headers, r.status_code, r.text)
    finally:
        _restore(old)


def test_log_answer_accepts_either_key():
    old = _set_keys(admin=ADMIN_KEY, loader=LOADER_KEY)
    try:
        client = _client()
        for key in (ADMIN_KEY, LOADER_KEY):
            r = client.post(LOG_ANSWER, headers=_auth(key), json=ANSWER_BODY)
            # An empty answer short-circuits to {"sampled": false} — accepted
            # past auth, with no judge work queued.
            assert r.status_code == 200, (key, r.status_code, r.text)
            assert r.json() == {"sampled": False, "reason": "empty answer"}, r.text

        for headers in (_auth(WRONG_KEY), {}):
            r = client.post(LOG_ANSWER, headers=headers, json=ANSWER_BODY)
            assert r.status_code == 401, (headers, r.status_code, r.text)
    finally:
        _restore(old)


def test_search_and_health_stay_open():
    """The chat pipeline calls these without credentials."""
    import routers.search as search_mod

    async def stub_search(*a, **kw):
        return []

    async def stub_log(*a, **kw):
        return None

    saved = (search_mod.retrieval.search, search_mod._log,
             search_mod.retrieval_cost.schedule)
    search_mod.retrieval.search = stub_search
    search_mod._log = stub_log
    search_mod.retrieval_cost.schedule = lambda *a, **kw: None
    old = _set_keys(admin=ADMIN_KEY, loader=LOADER_KEY)
    try:
        client = _client()
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok", r.text

        # No Authorization header at all — must not be 401/503.
        r = client.get("/search", params={"q": "anything"})
        assert r.status_code == 200, (r.status_code, r.text)
        assert r.json()["results"] == [], r.text

        # A wrong key does not lock the door either: /search is unauthenticated.
        r = client.get("/search", params={"q": "anything"}, headers=_auth(WRONG_KEY))
        assert r.status_code == 200, (r.status_code, r.text)
    finally:
        _restore(old)
        search_mod.retrieval.search, search_mod._log, \
            search_mod.retrieval_cost.schedule = saved


def main():
    tests = [
        test_fail_closed_without_any_key,
        test_admin_key_passes_and_wrong_key_is_401,
        test_loader_key_is_not_admin,
        test_ingestion_accepts_either_key,
        test_log_answer_accepts_either_key,
        test_search_and_health_stay_open,
    ]
    for test in tests:
        test()
        print("ok  %s" % test.__name__)
    print("\n%d auth checks passed" % len(tests))


if __name__ == "__main__":
    main()
