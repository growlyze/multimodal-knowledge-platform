"""Thin shim so haystack-api can call the shared processing/telemetry module
without a hard build-time dependency on it being present.

Why this file exists rather than `from telemetry import ...` directly: the
Dockerfile build context is the repo root (D52), which `COPY`s
`processing/telemetry/` to `./telemetry/` inside the image — but a one-off
local run (`python pipelines/video_ingestion.py` without the matching
PYTHONPATH) or a future build-context change breaking that COPY would make
`telemetry` unimportable. Every function here degrades to an inert no-op
in that case instead of raising ImportError at module load — which would
take the whole API down, violating this project's own "degrade, never
break" rule.

Verified live 2026-10-08: the deployed container logs "tracing to
http://phoenix:4317 ... authenticated=True" at startup, confirming
`_AVAILABLE` is True in production.
"""

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger("haystack-api.telemetry")

try:
    from telemetry import (
        INGESTION_PROJECT, SpanKind, flush as _flush,
        ingestion_session as _ingestion_session, init_tracing as _init_tracing, span as _span,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    # Same string as processing/telemetry/tracing.py's INGESTION_PROJECT --
    # kept in sync by hand since this branch exists specifically for when
    # that module can't be imported.
    INGESTION_PROJECT = "haystack-api-ingestion"

    class SpanKind:  # stand-in so call sites don't need their own availability check
        RETRIEVER = "RETRIEVER"
        TOOL = "TOOL"
        LLM = "LLM"
        CHAIN = "CHAIN"
        EMBEDDING = "EMBEDDING"


class _NoopSpan:
    """set_attribute is the only method every call site uses."""

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        pass


def init_tracing(project_name: str = "haystack-api", endpoint: Optional[str] = None):
    if not _AVAILABLE:
        logger.warning(
            "telemetry module not importable — tracing disabled (processing/telemetry/ "
            "is outside this image's build context; see this file's module docstring)"
        )
        return None
    return _init_tracing(project_name, endpoint=endpoint)


@contextmanager
def span(
    name: str,
    kind: Any,
    *,
    input_value: Optional[str] = None,
    attributes: Optional[dict] = None,
) -> Iterator[Any]:
    if not _AVAILABLE:
        yield _NoopSpan()
        return
    with _span(name, kind, input_value=input_value, attributes=attributes) as sp:
        yield sp


def record_decision(sp: Any, prefix: str, decision: str, reason: str) -> None:
    """Same convention as processing/telemetry/tracing.py — kept local rather than
    imported, since it's a two-line attribute-set and works identically on a real
    span or a _NoopSpan; not worth threading through the availability guard."""
    sp.set_attribute(f"{prefix}.decision", decision)
    sp.set_attribute(f"{prefix}.reason", reason)


def set_output(sp: Any, output_value: str) -> None:
    sp.set_attribute("output.value", output_value)


@contextmanager
def ingestion_session(video_id: str, platform: str = "", title: str = "") -> Iterator[None]:
    if not _AVAILABLE:
        yield
        return
    with _ingestion_session(video_id, platform=platform, title=title):
        yield


def flush(provider: Any, timeout_ms: int = 5000) -> None:
    """Call at shutdown. BatchSpanProcessor holds spans up to schedule_delay_millis
    (5s default) before exporting — without this, whatever's still queued when the
    container stops is silently lost, once per restart/deploy."""
    if _AVAILABLE and provider is not None:
        _flush(provider, timeout_ms=timeout_ms)
