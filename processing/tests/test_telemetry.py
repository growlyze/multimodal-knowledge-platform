"""Offline checks for processing/telemetry — no Phoenix, no network.

Run from processing/:  python tests/test_telemetry.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from telemetry import SpanKind, init_tracing, record_decision, set_output, span

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


def _last():
    return _exporter.get_finished_spans()[-1]


def check_kind_and_io():
    _exporter.clear()
    with span("qdrant.search", SpanKind.RETRIEVER, input_value="q") as sp:
        set_output(sp, "3 hits")
    a = _last().attributes
    assert a["openinference.span.kind"] == "RETRIEVER"
    assert a["input.value"] == "q" and a["output.value"] == "3 hits"
    print("ok kind_and_io")


def check_decision_and_reason():
    _exporter.clear()
    with span("retrieval.gate", SpanKind.RETRIEVER) as sp:
        record_decision(sp, "retrieval", "declined", "top score 0.34 below threshold 0.50")
    a = _last().attributes
    assert a["retrieval.decision"] == "declined"
    assert a["retrieval.reason"] == "top score 0.34 below threshold 0.50"
    print("ok decision_and_reason")


def check_exception_is_recorded_and_reraised():
    _exporter.clear()
    try:
        with span("kb_exec", SpanKind.TOOL) as sp:
            sp.set_attribute("exec.success", False)
            raise ValueError("boom")
    except ValueError:
        pass
    else:
        raise AssertionError("span() swallowed the exception")
    s = _last()
    assert s.status.status_code == trace.StatusCode.ERROR
    assert any(e.name == "exception" for e in s.events)
    print("ok exception_recorded_and_reraised")


def check_nesting():
    _exporter.clear()
    with span("chat", SpanKind.CHAIN):
        with span("retrieve", SpanKind.RETRIEVER):
            pass
    inner, outer = _exporter.get_finished_spans()
    assert inner.parent.span_id == outer.context.span_id
    print("ok nesting")


def check_embedding_kind():
    # M3 ingestion's embed phase (CLIP + Voyage) is neither a retriever nor an
    # LLM call — OpenInference has a dedicated EMBEDDING kind for this.
    _exporter.clear()
    with span("ingest.embed", SpanKind.EMBEDDING, input_value="42 items") as sp:
        set_output(sp, "42 vectors")
    a = _last().attributes
    assert a["openinference.span.kind"] == "EMBEDDING"
    print("ok embedding_kind")


def check_init_tracing_degrades():
    # Unreachable endpoint must not raise; batch mode queues and returns.
    init_tracing("telemetry-test", endpoint="http://127.0.0.1:1")
    print("ok init_tracing_degrades")


def main():
    checks = [
        check_kind_and_io,
        check_decision_and_reason,
        check_exception_is_recorded_and_reraised,
        check_nesting,
        check_embedding_kind,
        check_init_tracing_degrades,
    ]
    for c in checks:
        c()
    print(f"{len(checks)} checks passed")


if __name__ == "__main__":
    main()
