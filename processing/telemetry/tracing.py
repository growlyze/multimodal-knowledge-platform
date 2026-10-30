"""Tracing setup and span helpers — the only place that imports Phoenix/OpenTelemetry.

Everything else in the codebase does `from telemetry import init_tracing, span,
record_decision`. If Phoenix is ever swapped for another OTLP backend, this file
is the whole change.

Degrade, never break the chat: a dead Phoenix or a missing package must never
fail a request. Every entry point here falls back to OpenTelemetry's no-op
tracer, so instrumented code runs unchanged with tracing switched off.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from enum import Enum
from typing import Any, Iterator, Optional

from openinference.instrumentation import dangerously_using_project, using_metadata, using_session
from opentelemetry import trace
from opentelemetry.trace import Span

log = logging.getLogger("telemetry")

# gRPC port on the phoenix container, reachable on the internal stack network only. With no URL
# path and this port, phoenix.otel infers gRPC on its own (checked against
# arize-phoenix-otel 0.17.1: otel.py `_maybe_grpc_endpoint`).
DEFAULT_ENDPOINT = "http://phoenix:4317"
DEFAULT_PROJECT = "multimodal-ai-chat"

# M3 ingestion gets its own Phoenix project so a video's traces aren't mixed
# in with live chat-turn retrieval spans in the "haystack-api" project —
# both run in the same process/tracer, so dangerously_using_project (not a
# second register() call) is the documented way to route a block of spans
# elsewhere. Found via Phoenix docs review 2026-10-12.
INGESTION_PROJECT = "haystack-api-ingestion"

_SPAN_KIND_ATTR = "openinference.span.kind"


class SpanKind(str, Enum):
    """OpenInference span kinds this project uses.

    RETRIEVER: Qdrant search / rerank / confidence gate.
    TOOL:      kb_exec sandbox, Tavily web search, any external call.
    LLM:       an actual Claude/Gemini/Groq model call.
    CHAIN:     the overall request flow that ties the steps together.
    EMBEDDING: a batch embedding call (Voyage, CLIP) that is neither an LLM
               completion nor a retriever lookup — M3's embed phase.
    """

    RETRIEVER = "RETRIEVER"
    TOOL = "TOOL"
    LLM = "LLM"
    CHAIN = "CHAIN"
    EMBEDDING = "EMBEDDING"


def init_tracing(
    project_name: str = DEFAULT_PROJECT,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Register the global tracer provider. Call once at service start-up.

    Returns the provider, or None if tracing could not be set up (the caller
    carries on untraced).

    `batch=True` on purpose: phoenix.otel's default is a synchronous processor, which
    would block a chat turn for the exporter timeout whenever Phoenix is down. The
    batch queue (OTEL_BSP_MAX_QUEUE_SIZE, default 2048) is far above what two users
    can produce; `provider.force_flush()` before a short-lived process exits.

    Phoenix runs with PHOENIX_ENABLE_AUTH=True (see compose.yml) — since that flag
    went on, ingestion 401s without a key. `api_key` defaults to the PHOENIX_API_KEY
    env var (register() reads it itself if we don't pass one); create the key in the
    Phoenix UI under Settings once logged in, then put it in this service's .env.
    """
    endpoint = endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or DEFAULT_ENDPOINT
    api_key = api_key or os.environ.get("PHOENIX_API_KEY")
    try:
        from phoenix.otel import register

        # auto_instrument=False: instrumentation is explicit via span(), so a
        # library upgrade can't silently start emitting spans we didn't ask for.
        provider = register(
            project_name=project_name,
            endpoint=endpoint,
            api_key=api_key,
            batch=True,
            auto_instrument=False,
            set_global_tracer_provider=True,
        )
    except Exception:  # noqa: BLE001 — tracing must never take a service down
        log.exception("tracing disabled: could not register with %s", endpoint)
        return None
    log.info("tracing to %s (project=%s, authenticated=%s)", endpoint, project_name, bool(api_key))
    return provider


@contextmanager
def span(
    name: str,
    kind: SpanKind,
    *,
    input_value: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Iterator[Span]:
    """Open a span of the given OpenInference kind; the current span for the block.

    An exception leaving the block is recorded on the span and re-raised —
    so a failed tool call shows up as an errored TOOL span, not a gap.
    """
    tracer = trace.get_tracer("multimodal-ai-chat")
    with tracer.start_as_current_span(name) as sp:
        sp.set_attribute(_SPAN_KIND_ATTR, SpanKind(kind).value)
        if input_value is not None:
            sp.set_attribute("input.value", input_value)
        for key, value in (attributes or {}).items():
            sp.set_attribute(key, value)
        yield sp


def record_decision(sp: Span, prefix: str, decision: str, reason: str) -> None:
    """Write `<prefix>.decision` and `<prefix>.reason` on a span.

    Every branch point logs *why* it went the way it did, not just the outcome:
    `record_decision(sp, "retrieval", "declined", f"top score {s:.2f} below threshold {t}")`.
    Put concrete numbers in `reason` — "low score" can't be audited later.
    """
    sp.set_attribute(f"{prefix}.decision", decision)
    sp.set_attribute(f"{prefix}.reason", reason)


def set_output(sp: Span, output_value: str) -> None:
    sp.set_attribute("output.value", output_value)


@contextmanager
def ingestion_session(video_id: str, platform: str = "", title: str = "") -> Iterator[None]:
    """Group one video's ingestion spans (including resume_interrupted retries)
    under a single Phoenix session, routed to INGESTION_PROJECT, with
    platform/title attached as filterable metadata — not a dashboard, just
    makes a video's history findable as one unit in the Phoenix UI instead
    of one disconnected trace per attempt.

    session_id = video_id on purpose: a resumed ingestion reruns
    ingest_video() with the same video_id, so it lands in the same session.
    """
    with dangerously_using_project(project_name=INGESTION_PROJECT), \
            using_session(session_id=video_id), \
            using_metadata({"platform": platform, "title": title}):
        yield


def flush(provider, timeout_ms: int = 5000) -> None:
    """Push queued spans out now. For scripts and workers that exit right after."""
    if provider is not None:
        provider.force_flush(timeout_millis=timeout_ms)
