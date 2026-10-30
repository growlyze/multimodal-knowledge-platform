"""
System health counters — in process, flushed to PostgreSQL hourly.

Recording is synchronous and lock-guarded so any thread can call it from any
context (the filter, a background task, a thread pool), and it never touches the
network. The hourly flush is the only thing that talks to the database.

    metrics = get_metrics()
    metrics.record_search_latency(140)
    metrics.record_voyage_429()
    metrics.get_current()      # live numbers for the dashboard

The flush task starts itself the first time get_metrics() runs inside an event
loop, so INFRA does not have to add anything to main.py's lifespan.
"""

import asyncio
import logging
import math
import os
import threading
import time
from collections import deque

from evaluation.eval_db import EvalDB, get_eval_db

logger = logging.getLogger("haystack-api.evaluation")

FLUSH_INTERVAL = float(os.getenv("EVAL_FLUSH_INTERVAL", "3600"))
# ponytail: a bounded window, not every sample ever. 10K latencies is ~80KB and
# more than an hour of traffic at any volume this box will see.
MAX_SAMPLES = 10_000


def percentile(values: list, pct: float):
    """Nearest-rank percentile. None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(pct / 100.0 * len(ordered)) - 1
    return ordered[min(len(ordered) - 1, max(0, index))]


class SystemMetrics:
    """In-process counters. Thread-safe. Flushed to DB hourly."""

    def __init__(self, db: EvalDB = None):
        self.db = db or get_eval_db()
        self._lock = threading.Lock()
        self._started = time.time()
        self._flush_task = None
        self._reset()

    def _reset(self):
        self.search_latencies = deque(maxlen=MAX_SAMPLES)
        self.embedding_latencies = deque(maxlen=MAX_SAMPLES)
        self.error_count = 0
        self.query_count = 0
        self.empty_result_count = 0
        self.voyage_429_count = 0
        # M3 video ingestion counters
        self.videos_processed = 0
        self.frames_extracted = 0
        self.keyframes_detected = 0
        self.asr_segments = 0
        self.ocr_labels = 0
        self.od_detections = 0
        self.voyage_embedding_calls = 0
        self.clip_embedding_calls = 0

    # ── recording ───────────────────────────────────────

    def record_search_latency(self, ms: int):
        with self._lock:
            self.search_latencies.append(int(ms))

    def record_embedding_latency(self, ms: int):
        with self._lock:
            self.embedding_latencies.append(int(ms))

    def record_error(self):
        with self._lock:
            self.error_count += 1

    def record_query(self):
        with self._lock:
            self.query_count += 1

    def record_empty_result(self):
        with self._lock:
            self.empty_result_count += 1

    def record_voyage_429(self):
        with self._lock:
            self.voyage_429_count += 1

    def record_video_processed(self):
        with self._lock:
            self.videos_processed += 1

    def record_frames(self, extracted: int = 0, keyframes: int = 0):
        with self._lock:
            self.frames_extracted += extracted
            self.keyframes_detected += keyframes

    def record_asr_segments(self, count: int):
        with self._lock:
            self.asr_segments += count

    def record_ocr_labels(self, count: int):
        with self._lock:
            self.ocr_labels += count

    def record_od_detections(self, count: int):
        with self._lock:
            self.od_detections += count

    def record_voyage_call(self):
        with self._lock:
            self.voyage_embedding_calls += 1

    def record_clip_call(self):
        with self._lock:
            self.clip_embedding_calls += 1

    # ── reading ─────────────────────────────────────────

    def _snapshot_locked(self) -> dict:
        """Caller holds self._lock. threading.Lock is not reentrant — keep it that way."""
        search = list(self.search_latencies)
        embedding = list(self.embedding_latencies)
        return {
            "search_latency_p50": percentile(search, 50),
            "search_latency_p95": percentile(search, 95),
            "embedding_latency_avg": (round(sum(embedding) / len(embedding))
                                      if embedding else None),
            "error_count": self.error_count,
            "query_count": self.query_count,
            "empty_result_count": self.empty_result_count,
            "voyage_429_count": self.voyage_429_count,
            "samples": len(search),
            "videos_processed": self.videos_processed,
            "frames_extracted": self.frames_extracted,
            "keyframes_detected": self.keyframes_detected,
            "asr_segments": self.asr_segments,
            "ocr_labels": self.ocr_labels,
            "od_detections": self.od_detections,
            "voyage_embedding_calls": self.voyage_embedding_calls,
            "clip_embedding_calls": self.clip_embedding_calls,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    def get_current(self) -> dict:
        """Current counter values (for live dashboard)."""
        current = self.snapshot()
        current["uptime_seconds"] = round(time.time() - self._started)
        current["error_rate"] = (round(current["error_count"] / current["query_count"], 4)
                                 if current["query_count"] else 0.0)
        return current

    # ── flushing ────────────────────────────────────────

    async def flush_to_db(self, period: str = "hourly") -> dict:
        """Aggregate, store, reset. Called by the background timer."""
        with self._lock:
            row = self._snapshot_locked()
            self._reset()
        if not row["query_count"] and row["search_latency_p50"] is None:
            return row          # an idle hour is not worth a row
        await self.db.execute(
            """
            INSERT INTO eval_system_metrics
                (period, search_latency_p50, search_latency_p95, embedding_latency_avg,
                 error_count, query_count, empty_result_count, voyage_429_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            period, row["search_latency_p50"], row["search_latency_p95"],
            row["embedding_latency_avg"], row["error_count"], row["query_count"],
            row["empty_result_count"], row["voyage_429_count"],
        )
        logger.info("system metrics flushed: %s", row)
        return row

    async def history(self, limit: int = 24) -> list:
        return await self.db.fetch(
            "SELECT * FROM eval_system_metrics ORDER BY timestamp DESC LIMIT $1",
            max(1, int(limit)),
        )

    def ensure_flush_task(self, interval: float = None):
        """Start the hourly flush if an event loop is running. Idempotent."""
        if self._flush_task is not None and not self._flush_task.done():
            return self._flush_task
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return None         # no loop yet — tests, or import time
        self._flush_task = asyncio.create_task(self._flush_forever(interval or FLUSH_INTERVAL))
        return self._flush_task

    async def _flush_forever(self, interval: float):
        while True:
            await asyncio.sleep(interval)
            try:
                await self.flush_to_db()
            except Exception as exc:        # a metrics flush may never kill the loop
                logger.warning("metrics flush failed: %s: %s", type(exc).__name__, exc)


_metrics = None


def get_metrics() -> SystemMetrics:
    """Shared instance — one set of counters for the process."""
    global _metrics
    if _metrics is None:
        _metrics = SystemMetrics()
    _metrics.ensure_flush_task()
    return _metrics


if __name__ == "__main__":
    m = SystemMetrics()
    for ms in range(1, 101):
        m.record_search_latency(ms)
    m.record_query()
    m.record_error()
    assert m.get_current()["search_latency_p50"] == 50, m.get_current()
    assert m.get_current()["search_latency_p95"] == 95, m.get_current()
    assert m.get_current()["error_rate"] == 1.0
    assert percentile([], 50) is None and percentile([7], 95) == 7
    m.record_video_processed()
    m.record_frames(extracted=100, keyframes=15)
    m.record_asr_segments(42)
    m.record_ocr_labels(8)
    m.record_od_detections(3)
    m.record_voyage_call()
    m.record_clip_call()
    cur = m.get_current()
    assert cur["videos_processed"] == 1
    assert cur["frames_extracted"] == 100
    assert cur["keyframes_detected"] == 15
    assert cur["asr_segments"] == 42
    print("system metrics self-check ok")
