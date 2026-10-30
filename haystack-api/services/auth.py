"""
API authentication — bearer-token guards for the protected endpoints.

Two dependencies, applied per router (routers/*.py):

    require_admin           only ADMIN_API_KEY passes. Used by the admin,
                            video, eval and costs endpoints.

    require_admin_or_loader either ADMIN_API_KEY or EXTERNAL_LOADER_API_KEY
                            passes. Used by the ingestion endpoints and
                            POST /eval/log-answer, which the Open WebUI chat
                            filter calls with the external loader key.

    require_data_reader     either ADMIN_API_KEY or DATA_API_KEY passes.
                            Used only by the read-only stored-dataset
                            endpoints the code sandbox calls (data_tools).

Fail-closed, mirroring routers/extract.py: when no key is configured, the
protected endpoints refuse every request (503, {"error": "admin api key not
configured"}) rather than serving unauthenticated traffic. Key comparison is
constant-time (secrets.compare_digest).
"""

import secrets

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config

_bearer = HTTPBearer(auto_error=False)


class AdminKeyNotConfigured(Exception):
    """Raised when a protected endpoint is hit with no key configured."""


def admin_key_not_configured_handler(request, exc):  # pragma: no cover — trivial
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503,
                        content={"error": "admin api key not configured"})


def _provided(credentials: HTTPAuthorizationCredentials) -> str:
    """The bearer token as a full 'Authorization' header value, or ''."""
    return f"Bearer {credentials.credentials}" if credentials else ""


def _matches(provided: str, expected: str) -> bool:
    """Constant-time bearer comparison; an unset expected key never matches."""
    return bool(expected) and secrets.compare_digest(
        provided, "Bearer %s" % expected)


def _check(credentials: HTTPAuthorizationCredentials, accepted: list) -> None:
    if not any(k for k in accepted):
        raise AdminKeyNotConfigured()
    provided = _provided(credentials)
    if not any(_matches(provided, key) for key in accepted):
        raise HTTPException(status_code=401,
                            detail="Invalid or missing bearer token")


def require_admin(credentials: HTTPAuthorizationCredentials = Security(_bearer)):
    """Pass only with Authorization: Bearer <ADMIN_API_KEY>."""
    _check(credentials, [config.ADMIN_API_KEY])


def require_admin_or_loader(credentials: HTTPAuthorizationCredentials = Security(_bearer)):
    """Pass with either ADMIN_API_KEY or EXTERNAL_LOADER_API_KEY."""
    _check(credentials, [config.ADMIN_API_KEY, config.EXTERNAL_LOADER_API_KEY])


def require_data_reader(credentials: HTTPAuthorizationCredentials = Security(_bearer)):
    """Pass with either ADMIN_API_KEY or DATA_API_KEY."""
    _check(credentials, [config.ADMIN_API_KEY, config.DATA_API_KEY])
