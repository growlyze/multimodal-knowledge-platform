"""
M3 video ingestion orchestrator — sequential per-video pipeline.

Wires the Phase A foundation components (acquisition, ASR, visual extraction,
OCR, embedding) into a six-phase pipeline that processes one video at a time:

    Download → ASR → Visual → Embedding → Qdrant upsert → Cleanup

Each phase checkpoints to eval_db so a crash mid-pipeline can resume from
the last completed phase instead of re-downloading and re-transcribing.

Sequential processing is deliberate: MAX_CONCURRENT_VIDEOS=1 because WhisperX
(~3 GB) and CLIP (~400 MB) cannot coexist in RAM on the 8 GB VPS, and because
yt-dlp rate limits compound when concurrent downloads hit the same channel.

Design decisions
----------------
- ASR (Groq, D65) and visual/text embedding (Voyage multimodal, D66) are both
  cloud-hosted — no local model load/unload ordering to manage, unlike the
  original WhisperX+CLIP design this replaced.
- eval_db is optional: when None, checkpoints are skipped and the pipeline runs
  fire-and-forget. This keeps _self_test() offline and lets a manual one-off
  ingest run without a database.
- OD (object detection) track is a real vision-model description (Gemini,
  D67/pipelines/vision_description.py), not the earlier "chart_frame"
  placeholder — matches what D59 actually built the od_frames collection for
  (Voyage TEXT embeddings, because OD output was always meant to be text).
- Non-critical failures (OCR on one frame, vision description on one frame)
  log and continue. Only download and ASR failures are fatal — without
  content there is nothing to index.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from pipelines import usage_meter
from pipelines.ingestion_quality import (
    check_asr_quality, check_chart_descriptions, check_chart_talk, check_embedding_consistency,
    check_frame_coverage, check_transcript_coverage, check_visual_quality,
)
from telemetry_compat import SpanKind, ingestion_session, record_decision, set_output, span

logger = logging.getLogger("haystack-api.video_ingestion")

# job.error_message prefix when the vision provider is unusable before a video
# starts. routers/video.py's queue worker stops on it (like its download-failure
# breaker), so the rest of the queue stays queued instead of each video failing.
VISION_UNAVAILABLE_PREFIX = "vision provider unavailable"


def _vision_preflight():
    """One tiny Gemini call with the ingestion vision model, before any paid phase
    runs (gemini_down.md: "ingest nothing until it's back" was a written rule only).
    Returns a provider_health.HealthResult. Tests replace this."""
    from pipelines import provider_health
    return provider_health.probe_gemini(model=getattr(config, "VISION_MODEL", None))

# Phase names in execution order. Used by resume_interrupted() to determine
# which phases to skip and by _self_test() to verify ordering.
PHASE_ORDER = ("download", "asr", "visual", "embed", "index", "cleanup")

# Share of the whole run each phase takes, for the admin panel's single
# progress bar (overall_percent). Rough wall-clock shares on the 1:53
# checkpoint video: the visual phase (frame decode + Gemini descriptions)
# dominates, so a per-phase bar alone sat at "0%" for most of the run after
# download/ASR had visibly finished. Keyed in PHASE_ORDER order; sums to 100.
PHASE_WEIGHTS = {"download": 10, "asr": 15, "visual": 50, "embed": 12, "index": 10, "cleanup": 3}


def overall_percent(phase, phase_progress, status) -> int:
    """Whole-run progress 0-100 from a job row's phase/phase_progress/status.

    Phases before `phase` count fully, `phase` itself counts weight x
    phase_progress (clamped to 0..1). Only status "done" reads 100: a job
    still processing caps at 99, so the bar never says finished while the
    worker has not recorded the run yet. Queued/new/unknown phase -> 0; a
    failed or cancelled job shows where it stopped.
    """
    if status == "done":
        return 100
    if phase not in PHASE_WEIGHTS:
        return 0
    try:
        frac = float(phase_progress or 0.0)
    except (TypeError, ValueError):
        frac = 0.0
    frac = min(1.0, max(0.0, frac))
    before = sum(PHASE_WEIGHTS[p] for p in PHASE_ORDER[:PHASE_ORDER.index(phase)])
    pct = int(round(before + PHASE_WEIGHTS[phase] * frac))
    return min(pct, 99)


# ── Cooperative cancel (POST /video/cancel) ─────────────────────────
# A run cannot be killed from outside: it is a sync function on a worker
# thread, mid-way through paid calls whose spend must still be recorded. The
# router passes should_stop() into ingest_video; the run checks it at cheap
# points -- before each phase, before each ASR chunk, per sampled frame in the
# visual phase, before each embedding batch, before the overview call -- and
# unwinds with JobCancelled. A call already in flight always finishes.

class JobCancelled(Exception):
    """Raised at a checkpoint once should_stop() says the user cancelled.

    Every phase's broad `except Exception` re-raises it: the visual phase in
    particular turns any error into a "degraded" ASR-only run, which for a
    cancel would go on to pay for embedding and re-indexing."""

    def __init__(self, phase: str):
        super().__init__(f"cancelled by user during {phase}")
        self.phase = phase


# phase_progress writes are throttled to one per this many seconds per
# phase (the visual phase's own 15 s throttle predates this constant).
PROGRESS_WRITE_INTERVAL_S = 15.0


class _RunControl:
    """Per-run cancel check + throttled phase_progress writer.

    Lives in a thread-local (see _run_control) rather than being threaded
    through every phase function's signature: the phase functions are
    replaced wholesale by fakes in the self-tests and by nothing else, and
    one ingest_video runs per worker thread."""

    def __init__(self, video_id, eval_db, main_loop, should_stop):
        self.video_id = video_id
        self.eval_db = eval_db
        self.main_loop = main_loop
        self.should_stop = should_stop or (lambda: False)
        # The phase the user would see as running. Starts at "download": a
        # stop seen before anything ran is reported as during download.
        self.phase = "download"
        self.video_path = None
        self.result = None
        self._last_write = float("-inf")

    def check(self):
        try:
            stop = bool(self.should_stop())
        except Exception as exc:
            # A broken stop flag must not kill a paid run half-way.
            logger.warning("[%s] should_stop() raised, ignoring: %s", self.video_id, exc)
            stop = False
        if stop:
            raise JobCancelled(self.phase)

    def progress(self, fraction: float):
        now = time.monotonic()
        if now - self._last_write >= PROGRESS_WRITE_INTERVAL_S:
            self._last_write = now
            _update_job_status(self.eval_db, self.video_id, "processing", phase=self.phase,
                               progress=fraction, main_loop=self.main_loop)

    def enter(self, phase: str):
        """Checkpoint between phases: a stop requested during the previous
        phase is reported against that phase, then the new one begins."""
        self.check()
        self.phase = phase
        self._last_write = float("-inf")

    def on_chunk(self, done: int, total: int):
        self.check()
        if total:
            self.progress(done / total)


_RUN = threading.local()
_NO_CONTROL = _RunControl("", None, None, None)


def _run_ctl() -> _RunControl:
    """The current run's control, or an inert one (never stops, writes to no
    database) when a phase function is called outside ingest_video."""
    return getattr(_RUN, "ctl", None) or _NO_CONTROL


@contextmanager
def _run_control(video_id, eval_db, main_loop, should_stop):
    ctl = _RunControl(video_id, eval_db, main_loop, should_stop)
    prev = getattr(_RUN, "ctl", None)
    _RUN.ctl = ctl
    try:
        yield ctl
    finally:
        _RUN.ctl = prev


def _stop_point():
    """Raise JobCancelled if the current run has been asked to stop."""
    _run_ctl().check()


# ── Result dataclass ────────────────────────────────────────────────

@dataclass
class IngestResult:
    """Outcome of a single video ingestion run."""
    video_id: str
    status: str                           # "done", "failed" or "cancelled"
    phases_completed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    # Measured API usage + list-price cost (usage_meter.UsageMeter.summary()).
    # Filled even when a phase fails: the paid phases before it still spent.
    usage: Dict[str, Any] = field(default_factory=dict)
    # Set when status == "cancelled": the phase that was running when the
    # stop was seen. The router decides from it whether Qdrant was touched
    # (index/cleanup) or a previous index is still intact.
    phase: Optional[str] = None


@contextmanager
def _timed(result: "IngestResult", phase: str):
    """Exact wall-clock seconds for one phase into result.stats["phase_seconds"].
    Checkpoint rows only carry completed_at (started_at was never written), so
    per-stage times could only be inferred from gaps between them -- not good
    enough for the stage timings reported to the client."""
    t = time.monotonic()
    try:
        yield
    finally:
        result.stats.setdefault("phase_seconds", {})[phase] = round(time.monotonic() - t, 2)


# ── Memory guard ────────────────────────────────────────────────────

def check_memory_watermark() -> bool:
    """True when available RAM exceeds MEMORY_WATERMARK_GB.

    Checked before starting a new video — not between phases, because partial
    progress is more valuable than aborting mid-pipeline. If memory is tight
    between phases the OS will swap, which is slow but recoverable; starting
    a NEW video when memory is already low risks OOM-killing the container
    and losing all progress on the current video.
    """
    import psutil
    watermark = getattr(config, "MEMORY_WATERMARK_GB", 4)
    avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    ok = avail_gb >= watermark
    if not ok:
        logger.warning(
            "memory watermark not met: %.1f GB available, need %d GB",
            avail_gb, watermark,
        )
    return ok


# ── Checkpoint helpers ──────────────────────────────────────────────
# eval_db is asyncpg-based (async def execute / fetch / fetchrow).
# The orchestrator is sync, so DB calls go through asyncio.run() (or
# get_event_loop if one is already running). When eval_db is None the
# helpers are no-ops — the pipeline still works, just without recovery.

def _run_async(coro, main_loop=None):
    """Run an async coroutine from sync context.

    `main_loop`, when given, MUST be used instead of spinning up a new event
    loop -- P95 (2026-10-12): ingest_video() runs inside asyncio.to_thread(),
    and every eval_db call this makes shares the module-level EvalDB
    singleton's asyncpg connection pool, which is bound to whichever event
    loop first created it (the FastAPI app's main loop, warmed up by an
    earlier request). asyncpg pools/connections are NOT safe to use from a
    second, independently-created event loop -- doing so produced
    "another operation is in progress" and, under real concurrent load
    during the very first live ingestion attempt, crashed the whole
    container with a silent SIGSEGV (root-caused, not reproduced in
    isolation -- see 14-problems-bugs.md P95). run_coroutine_threadsafe
    schedules the coroutine onto the SAME loop/thread the pool belongs to,
    instead of creating a second, colliding one.

    Without `main_loop` (self-test, or a direct non-threaded call), falls
    back to the old behavior: handles both "no event loop" and "loop
    already running" cases, the latter arising when called from an async
    FastAPI endpoint directly rather than via asyncio.to_thread.
    """
    if main_loop is not None:
        return asyncio.run_coroutine_threadsafe(coro, main_loop).result(timeout=30)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We are inside an async context (e.g. FastAPI). Create a new
        # loop in a thread would deadlock. Use nest_asyncio if available,
        # otherwise fall back to a new event loop in a thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=30)
    else:
        return asyncio.run(coro)


def _save_checkpoint(
    eval_db,
    video_id: str,
    phase: str,
    status: str,
    output_path: str = "",
    item_count: int = 0,
    main_loop=None,
) -> None:
    """Write or update a phase checkpoint row. No-op when eval_db is None.

    `main_loop` -- see _run_async's docstring (P95): must be the event loop
    eval_db's asyncpg pool was created on, when this runs inside
    asyncio.to_thread().
    """
    if eval_db is None:
        return
    sql = """
        INSERT INTO video_phase_checkpoints
            (video_id, phase, status, output_path, item_count, completed_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        ON CONFLICT (video_id, phase) DO UPDATE
            SET status = $3, output_path = $4, item_count = $5, completed_at = NOW()
    """
    try:
        _run_async(eval_db.execute(sql, video_id, phase, status, output_path, item_count),
                  main_loop=main_loop)
    except Exception as exc:
        # Checkpoint failure must not kill the pipeline — the work is done,
        # we just cannot resume from it if a later phase crashes.
        logger.warning("checkpoint save failed for %s/%s: %s", video_id, phase, exc)


def _update_job_status(
    eval_db,
    video_id: str,
    status: str,
    phase: str = "",
    error_message: str = "",
    main_loop=None,
    progress: float = 0.0,
) -> None:
    """Update video_ingest_jobs for progress tracking. No-op when eval_db is None.

    `progress` is the fraction of the current phase done (phase_progress, read
    by the admin panel's progress bar and ETA); 0.0 at the start of a phase.

    `main_loop` -- see _run_async's docstring (P95): must be the event loop
    eval_db's asyncpg pool was created on, when this runs inside
    asyncio.to_thread().
    """
    if eval_db is None:
        return
    sql = """
        UPDATE video_ingest_jobs
        SET status = $2, phase = $3, error_message = $4, phase_progress = $5, updated_at = NOW()
        WHERE video_id = $1
    """
    try:
        _run_async(eval_db.execute(sql, video_id, status, phase, error_message, float(progress)),
                   main_loop=main_loop)
    except Exception as exc:
        logger.warning("job status update failed for %s: %s", video_id, exc)


def _get_checkpoints(eval_db, video_id: str) -> Dict[str, str]:
    """Return {phase: status} for all completed checkpoints.

    Returns empty dict when eval_db is None (no checkpoints to read).
    """
    if eval_db is None:
        return {}
    sql = """
        SELECT phase, status FROM video_phase_checkpoints
        WHERE video_id = $1
    """
    try:
        rows = _run_async(eval_db.fetch(sql, video_id))
        return {row["phase"]: row["status"] for row in rows}
    except Exception as exc:
        logger.warning("checkpoint read failed for %s: %s", video_id, exc)
        return {}


# ── Qdrant upsert ──────────────────────────────────────────────────
# Uses the same httpx-based QdrantService as the rest of haystack-api.
# Video collections store a single dense vector per point (no sparse
# text vector) because the payloads carry structured metadata, not
# free-text chunks. Sparse BM25 on OCR text is a Phase D optimisation.

def _upsert_to_qdrant(
    collection: str,
    ids: List[str],
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    dimensions: int,
) -> int:
    """Upsert points to a Qdrant collection, creating it if needed.

    Returns the number of points upserted. Raises on failure — the caller
    decides whether to treat an upsert failure as fatal.
    """
    from services.qdrant_service import QdrantService

    async def _do_upsert():
        qs = QdrantService()
        try:
            await qs.ensure_collection(collection, dimensions, enable_sparse=False)
            await qs.upsert(collection, ids, vectors, payloads)
            return len(ids)
        finally:
            await qs.aclose()

    return _run_async(_do_upsert())


def _delete_video_points(video_id: str) -> None:
    """Remove every point this video has in the four video collections.

    Point ids are positional ({video_id}:asr:{i}, ...:visual:{i}, ...), so a
    re-index that yields fewer points than last time -- transcript windows
    vs the old per-segment index (P152), or a different keyframe count --
    would otherwise leave the old tail behind as stale search results.
    """
    from services.qdrant_service import QdrantService

    async def _do_delete():
        qs = QdrantService()
        try:
            for collection in (config.VIDEO_VISUAL_COLLECTION, config.VIDEO_OCR_COLLECTION,
                               config.VIDEO_OD_COLLECTION, config.VIDEO_TRANSCRIPT_COLLECTION):
                await qs.delete_by_filter(collection, {"video_id": video_id})
        finally:
            await qs.aclose()

    _run_async(_do_delete())


# P152: the transcript track is indexed as overlapping ~60 s windows, not as
# Whisper's ~7 s segments. A lone segment is often just a question ("And why
# does the system even need a rate limit?") with its answer in the next
# segments; it reranked 0.537 against a real user question and fell under the
# 0.70 floor, so chat answered nothing about any video. The 60 s window holding
# the answer scored 0.751 (90 s: 0.808), measured 2026-10-20.
TRANSCRIPT_WINDOW_S = 60.0
TRANSCRIPT_STRIDE_S = 45.0     # a new window every 45 s -> ~15 s overlap


@dataclass
class TranscriptWindow:
    text: str
    start_time: float
    end_time: float
    confidence: float          # mean of the segments' avg_logprob
    words: List[dict] = field(default_factory=list)


def transcript_windows(segments) -> List[TranscriptWindow]:
    """Group consecutive non-empty segments into windows of at most
    TRANSCRIPT_WINDOW_S (a single longer segment stays its own window), a new
    window starting at the first segment at least TRANSCRIPT_STRIDE_S after
    the previous window's start. Every segment lands in at least one window;
    word timestamps are kept so citations can still point inside a window."""
    segs = [s for s in segments if (s.text or "").strip()]
    windows, i = [], 0
    while i < len(segs):
        start = segs[i].start_time
        j = i
        while j + 1 < len(segs) and segs[j + 1].end_time - start <= TRANSCRIPT_WINDOW_S:
            j += 1
        group = segs[i:j + 1]
        windows.append(TranscriptWindow(
            text=" ".join(s.text.strip() for s in group),
            start_time=start,
            end_time=group[-1].end_time,
            confidence=sum(s.confidence for s in group) / len(group),
            words=[w for s in group for w in (s.words or [])],
        ))
        if j + 1 >= len(segs):
            break
        nxt = next((k for k in range(i + 1, j + 1) if segs[k].start_time >= start + TRANSCRIPT_STRIDE_S), None)
        i = nxt if nxt is not None else j + 1
    return windows


# P152 stage B: "summarize this video" can't be answered by passage
# similarity -- no single passage scores high against an overview question
# (0 results on the pilot's 1:53 checkpoint video). One overview per video,
# written at index time from the same transcript windows, answers it.
# Minute marks (the client 3:29, P180): when the prompt showed clock times, the
# overview's marks were Gemini's own picks/estimates, not grounded in a passage.
# Now the model only cites a passage NUMBER ([#n]); _ground_overview_timestamps()
# replaces each one with that passage's real start time, so every mark in a stored
# overview is the start of an actual transcript window. Output format unchanged:
# "[h:mm:ss]".
OVERVIEW_SYSTEM_INSTRUCTION = (
    "You summarise a transcribed video (a lecture, talk, tutorial or presentation). "
    "List the 5-8 main concepts it teaches, in the order taught. For each: a short heading, "
    "two or three sentences in plain words explaining what the instructor says, and the "
    "passage number [#n] of the passage where it is first taught, written exactly as it "
    "appears in the transcript (for example [#12]). If the concept is explained most fully "
    "in a different, later passage, add that passage number too (for example [#12] [#40]); "
    "never more than two passage numbers per concept. Never write a clock time or "
    "timestamp yourself. Use only the transcript."
)
# Up to two marks (2026-10-26): with one, "deployment urgency" in the March 27
# overview pointed at its first mention (1:15:38) while the client looked for the
# full explanation at 1:33:51. _ground_overview_timestamps() enforces the cap.
_MAX_MARKS_PER_CONCEPT = 2


def _hms(seconds: float) -> str:
    s = int(seconds or 0)
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def _valid_time(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v == v and v >= 0.0          # not NaN, not negative


def _overview_windows(windows) -> list:
    """The windows the overview prompt numbers, in time order. A window without a
    usable start time is left out: it has no minute mark to give."""
    return sorted((w for w in windows if _valid_time(w.start_time)), key=lambda w: float(w.start_time))


def _overview_prompt(title: str, windows) -> str:
    lines = ["[#%d] %s" % (i, w.text) for i, w in enumerate(_overview_windows(windows), 1)]
    return "Video title: %s\n\nTranscript:\n%s" % (title, "\n".join(lines))


_PASSAGE_REF = re.compile(r"\[#\s*(\d{1,5})\s*\]")
# A clock time the model wrote itself, bracketed or parenthesised: [1:07:55], (12:30).
# Only these shapes -- a bare "1:2" risk-reward ratio in prose is not touched.
_MODEL_TIMESTAMP = re.compile(r"[\[(]\s*\d{1,2}:[0-5]\d(?::[0-5]\d)?\s*[\])]")


# A run of adjacent grounded marks, "[0:01:40] [0:05:00]" or "[0:00:10], [0:05:00]".
_MARK_RUN = re.compile(r"\[\d+:\d\d:\d\d\](?:[ 	]*,?[ 	]*\[\d+:\d\d:\d\d\])+")


def _tidy_mark_run(m) -> str:
    """One concept's marks: deduplicated (the model repeated a passage live,
    "[0:07:18] [0:07:18]"), capped at _MAX_MARKS_PER_CONCEPT in the order the
    model gave them (first-taught, then fullest), then shown in time order."""
    marks = []
    for t in re.findall(r"\d+:\d\d:\d\d", m.group(0)):
        if t not in marks:
            marks.append(t)
    marks = marks[:_MAX_MARKS_PER_CONCEPT]
    marks.sort(key=lambda t: [int(x) for x in t.split(":")])
    return " ".join("[%s]" % t for t in marks)


def _ground_overview_timestamps(text: str, numbered: list) -> str:
    """[#n] -> "[h:mm:ss]" of passage n's real start. A reference to a passage that
    does not exist is removed rather than guessed, and so is any clock time the
    model wrote on its own."""
    invented = len(_MODEL_TIMESTAMP.findall(text))
    text = _MODEL_TIMESTAMP.sub("", text)
    bad = []

    def _sub(m):
        n = int(m.group(1))
        if 1 <= n <= len(numbered):
            return "[%s]" % _hms(float(numbered[n - 1].start_time))
        bad.append(n)
        return ""

    text = _PASSAGE_REF.sub(_sub, text)
    text = _MARK_RUN.sub(_tidy_mark_run, text)
    if invented or bad:
        logger.warning("video overview: removed %d model-written timestamp(s) and %d reference(s) "
                       "to passages that do not exist (%s); %d passages available",
                       invented, len(bad), bad[:5], len(numbered))
    # Tidy what a removed mark leaves behind ("taught  .", "( )").
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _overview_generate(prompt: str) -> str:
    """The network call (Gemini). Usage is metered as ingestion spend."""
    from google import genai
    from google.genai import types

    # Bound to a name: an unbound genai.Client() is garbage-collected mid-call
    # ("Cannot send a request, as the client has been closed", live check
    # 2026-10-20 -- the faked test could not see it).
    client = genai.Client()
    response = client.models.generate_content(
        model=getattr(config, "VISION_MODEL", "gemini-3.6-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=OVERVIEW_SYSTEM_INSTRUCTION,
            # Thinking tokens come out of this budget (P138): at 1,500 the live
            # overview stopped after 303 chars, mid-sentence.
            max_output_tokens=6000,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    usage_meter.record_gemini(getattr(response, "usage_metadata", None))
    candidates = getattr(response, "candidates", None) or []
    if candidates and "MAX_TOKENS" in str(getattr(candidates[0], "finish_reason", "")):
        logger.warning("video overview hit max_output_tokens -- discarded rather than stored truncated")
        return ""
    return (response.text or "").strip()


def _index_video_overview(video_id: str, title: str, windows, platform: str = "",
                          source_url: str = "") -> int:
    """Write the video's overview point ({video_id}:overview) into the
    transcript collection. Non-fatal: any failure writes nothing, returns 0.
    Stored as a transcript_segment flagged overview=True so it is cited like
    any transcript passage (linked to the video's start)."""
    if not windows:
        return 0
    try:
        text = _overview_generate(_overview_prompt(title, windows))
        if text:
            text = _ground_overview_timestamps(text, _overview_windows(windows))
        if not text:
            return 0
        from pipelines.video_embedder import embed_texts_voyage
        vector = embed_texts_voyage([text])
        payload = {
            "video_id": video_id, "title": title, "text": text, "overview": True,
            "start_time": 0.0, "end_time": max(w.end_time for w in windows), "confidence": 0.0,
            "words": [], "topic_id": -1, "source_type": "transcript_segment",
            "platform": platform, "source_url": source_url,
        }
        return _upsert_to_qdrant(config.VIDEO_TRANSCRIPT_COLLECTION, ["%s:overview" % video_id],
                                 vector, [payload], config.VOYAGE_EMBEDDING_DIM)
    except Exception as exc:
        logger.warning("[%s] video overview skipped: %s", video_id, exc)
        return 0


# ── Phase implementations ──────────────────────────────────────────

def _phase_download(
    video_id: str, work_dir: str,
    platform: str = "youtube", source_url: str = "", duration: float = 0.0,
) -> dict:
    """Phase 1: download video and extract audio.

    platform/source_url come from video_catalog — YouTube downloads are
    re-derived from video_id, but every other platform has no synthesizable
    watch URL, so it has to download from the URL recorded at discovery time.

    duration : video_catalog's recorded length, or 0.0 if unknown. Some
        platforms' yt-dlp extractors never expose duration in metadata-only
        mode (confirmed for Instagram — P104), so an unknown value is
        backfilled here via ffprobe on the file we just downloaded, which
        works regardless of which platform supplied (or failed to supply)
        it. A value already known from metadata is trusted as-is — no
        reason to pay for an extra subprocess call.

    Returns {"video_path": str, "audio_path": str, "duration": float}.
    Raises on failure — download is always fatal.
    """
    from pipelines.video_acquisition import download_video, extract_audio, probe_duration_seconds

    logger.info("[%s] phase 1/6: download", video_id)
    video_path = download_video(video_id, work_dir, platform=platform, source_url=source_url)
    audio_path = extract_audio(video_path, work_dir)
    if duration <= 0:
        duration = probe_duration_seconds(video_path)
        logger.info("[%s] duration missing from metadata, ffprobe reports %.1fs",
                    video_id, duration)
    logger.info("[%s] download complete: video=%s audio=%s",
                video_id, video_path, audio_path)
    return {"video_path": video_path, "audio_path": audio_path, "duration": duration}


def _phase_asr(audio_path: str, video_id: str) -> list:
    """Phase 2: speech-to-text via WhisperX.

    Runs before visual extraction so WhisperX's ~3 GB is freed first.
    Returns list of TranscriptSegment. Raises on failure — ASR is fatal.
    """
    from pipelines.asr_track import transcribe_video

    logger.info("[%s] phase 2/6: ASR (WhisperX)", video_id)
    # on_chunk: progress = chunks done / total, and a cancel checkpoint
    # before each paid Groq request of a chunked (long) video.
    segments = transcribe_video(audio_path, on_chunk=_run_ctl().on_chunk)
    logger.info("[%s] ASR complete: %d segments", video_id, len(segments))
    return segments


# P139: measured 2026-10-19 -- keyframes were held at full resolution until
# the whole video had been read (~7 MB resident each) at ~50-60 keyframes
# per minute: 20+ GB for one hour of video, against ~30 GB free on the VPS.
# Each keyframe is now classified/OCR'd/described as it's captured, and only
# a copy this wide is kept for the embedding phase (Voyage multimodal).
KEEP_FRAME_MAX_WIDTH = 768
# Vision descriptions ran one at a time (~3.7s each), making the visual phase
# ~2.8x the video's length. They now run this many at once while decoding
# continues. Kept modest: describe_chart_frame() returns "" on any error, so
# rate-limit failures would degrade descriptions silently rather than loudly.
VISION_WORKERS = 4
# At most this many full-resolution frames may wait for a description; past
# it the reader blocks on the oldest one, so a slow vision API can't
# re-create the unbounded backlog this change removed.
VISION_MAX_PENDING = VISION_WORKERS * 3


# ── Cost levers (2026-10-25), all DEFAULT OFF ──────────────────────────
# Read with os.environ at call time rather than config.py (an infra file the
# sprint lead owns); unset = the pipeline behaves exactly as before (self-
# test 22a). Measured baseline they act on, z6AxPV2NQ9U (1:53): 2,031
# keyframes (change 624, burst 971, safety 420, cue 16), $3.5787 total, of
# which Gemini descriptions $3.0019 and Voyage image embeddings ~$0.28.

def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _vision_workers() -> int:
    """VIDEO_VISION_WORKERS: concurrent Gemini description calls. Flex-tier
    requests (VIDEO_DESC_SERVICE_TIER=flex) may queue for minutes each, so
    the default 4 would stretch one long video's visual phase to many hours.
    Each pending call holds a full-resolution frame (~7 MB, P139) and the
    backlog is capped at 3x this, so raise it with the memory headroom in
    mind (16 workers -> ~48 frames -> ~340 MB)."""
    raw = os.environ.get("VIDEO_VISION_WORKERS")
    if not raw:
        return VISION_WORKERS
    try:
        n = int(raw)
    except ValueError:
        n = 0
    if n < 1:
        logger.warning("VIDEO_VISION_WORKERS=%r invalid, using %d", raw, VISION_WORKERS)
        return VISION_WORKERS
    return n


def _dedupe_threshold():
    """VIDEO_KEYFRAME_DEDUPE_SSIM: SSIM similarity in (0, 1] at or above
    which a burst/safety/cue keyframe counts as a near-duplicate of the last
    KEPT keyframe and is dropped before OCR, Gemini and embedding. None =
    off. An invalid value turns the lever off (logged), never on: a bad
    threshold like 0 would drop every frame that is not a detected change."""
    raw = os.environ.get("VIDEO_KEYFRAME_DEDUPE_SSIM")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0.0 < value <= 1.0:
        logger.warning("VIDEO_KEYFRAME_DEDUPE_SSIM=%r not in (0, 1], dedupe off", raw)
        return None
    return value


# Above this share of Flex frames falling back to Standard, the video gets a
# warning: the run still completes at full quality, but its cost is closer to
# Standard than the Flex estimate (user decision 2026-10-26: carry on, flag it).
FLEX_FALLBACK_WARN_SHARE = 0.30


def flex_fallback_warning(gemini: dict) -> str:
    """Warning text when too many chart descriptions fell back from Flex to
    Standard, else "". `gemini` is UsageMeter.summary()["gemini"]."""
    fallbacks = gemini.get("flex_fallbacks", 0)
    if not fallbacks:
        return ""
    tried = gemini.get("flex_calls", 0) + fallbacks
    share = fallbacks / tried
    if share <= FLEX_FALLBACK_WARN_SHARE and not gemini.get("flex_breaker_tripped"):
        return ""
    text = ("Flex tier busy: %d of %d chart descriptions (%.0f%%) fell back to standard price"
            % (fallbacks, tried, share * 100))
    if gemini.get("flex_breaker_tripped"):
        text += "; flex switched off for the rest of this video"
    return text


def _shrink_for_embedding(frame):
    import cv2
    h, w = frame.shape[:2]
    if w <= KEEP_FRAME_MAX_WIDTH:
        return frame
    return cv2.resize(frame, (KEEP_FRAME_MAX_WIDTH, int(round(h * KEEP_FRAME_MAX_WIDTH / w))),
                      interpolation=cv2.INTER_AREA)


def _assess_ocr(video_id: str, ocr_raw: list, labels_to_text) -> tuple:
    """Run the OCR precision checks (pipelines/ocr_precision.py) over every
    chart keyframe of the video, in order; return (ocr_texts, ocr_labels).

    ocr_texts keeps the old flat-text shape for embedding/search, with the
    " [uncertain]" marker after every flagged reading (marker contract in
    ocr_precision's docstring -- rag_filter relies on it). A failed frame
    gives "" and []. Degrade, never break ingestion: if the checks themselves
    raise, fall back to the unmarked pre-2026-10-28 text and log it -- the
    video still indexes, without flags and without ocr_labels, so the absence
    is visible rather than a silent "all ok".
    """
    from pipelines import ocr_precision as op
    idx = [i for i, item in enumerate(ocr_raw) if item is not None]
    frames = [ocr_raw[i] for i in idx]
    texts = [""] * len(ocr_raw)
    try:
        out_labels = [[] for _ in ocr_raw]
        assessed = [op.assess_frame(l, w, h) for l, w, h in frames]
        w, h = (frames[-1][1], frames[-1][2]) if frames else (None, None)
        op.assess_sequence(assessed, height=h, width=w)
        for i, rs in zip(idx, assessed):
            texts[i] = op.render_text(rs)
            out_labels[i] = op.payload_labels(rs)
        flagged = sum(1 for rs in assessed for r in rs if r["status"] in op.FLAG_STATUSES)
        priced = sum(1 for rs in assessed for r in rs if r["values"])
        logger.info("[%s] OCR precision: %d/%d price readings flagged %s",
                    video_id, flagged, priced, op.MARKER)
    except Exception as exc:
        logger.warning("[%s] OCR precision checks failed, indexing unflagged text: %s",
                       video_id, exc, exc_info=True)
        for i, (labels, _, _) in zip(idx, frames):
            texts[i] = labels_to_text(labels)
        out_labels = None
    return texts, out_labels


def _phase_visual(video_path: str, video_id: str, segments: list = None,
                  on_progress=None) -> dict:
    """Phase 3: sample frames, detect keyframes, classify, OCR.

    `segments` (from the ASR phase, optional) feeds audio-cue-triggered
    capture — a frame grabbed near a transcript timestamp where the
    presenter says something like "look here" (pipelines/audio_cue_capture.py).
    Keyframes also now include periodic safety frames and post-change burst
    frames (pipelines/adaptive_capture.py), not just SSIM-detected changes —
    a long static narration-only stretch no longer produces zero keyframes.

    Returns {
        "keyframes": [(frame_bgr, frame_ms), ...],
        "chart_flags": [bool, ...],   -- parallel to keyframes
        "ocr_texts": [str, ...],      -- per chart keyframe; flagged prices
                                         carry " [uncertain]" (ocr_precision)
        "ocr_labels": [[{...}], ...], -- per chart keyframe: every reading's
                                         text/value/values/conf/bbox/status
        "od_descriptions": [str, ...], -- per chart keyframe (Gemini, D67)
        "capture_reasons": [str, ...], -- parallel to keyframes, for visibility
    }
    Non-critical failures (single frame OCR) are logged and skipped.

    `on_progress(fraction)` is called after each sampled frame with how far
    through the video the read is (0..1); the caller throttles its own writes.
    """
    import cv2
    from pipelines.adaptive_capture import AdaptiveCaptureController
    from pipelines.annotation_change_detector import AnnotationChangeDetector
    from pipelines.audio_cue_capture import find_cue_timestamps
    from pipelines.chart_classifier import is_chart_frame
    from pipelines.ocr_track import extract_labels, labels_to_text
    from pipelines.vision_description import describe_chart_frame

    logger.info("[%s] phase 3/6: visual extraction", video_id)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Sample at 1 fps — every fps-th frame. For a 30 fps video this reads
    # 1 frame per second, which is enough to catch annotation changes that
    # persist for at least one second (on-screen annotations typically stay 2-10s).
    sample_interval = max(1, int(round(fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    detector = AnnotationChangeDetector(ssim_threshold=config.SSIM_THRESHOLD)
    cue_timestamps_s = find_cue_timestamps(segments) if segments else []
    capture = AdaptiveCaptureController(
        safety_interval_s=config.SAFETY_FRAME_INTERVAL_S,
        active_safety_interval_s=config.ACTIVE_SAFETY_FRAME_INTERVAL_S,
        active_window_s=config.ACTIVE_WINDOW_S,
        active_change_threshold=config.ACTIVE_CHANGE_THRESHOLD,
        burst_frame_count=config.BURST_FRAME_COUNT,
        cue_timestamps_s=cue_timestamps_s,
        cue_tolerance_s=config.AUDIO_CUE_TOLERANCE_S,
    )
    from concurrent.futures import ThreadPoolExecutor

    # Cost lever: drop near-duplicate keyframes (see _dedupe_threshold).
    dedupe_threshold = _dedupe_threshold()
    dedupe = None
    if dedupe_threshold is not None:
        from pipelines.annotation_change_detector import NearDuplicateFilter
        dedupe = NearDuplicateFilter(dedupe_threshold)
    deduped_frames = 0
    vision_workers = _vision_workers()
    vision_max_pending = vision_workers * 3     # same 3x ratio as VISION_MAX_PENDING

    keyframes = []          # (frame shrunk for embedding, timestamp_ms) -- P139
    capture_reasons = []    # parallel to keyframes — "change"/"burst"/"cue"/"safety"
    chart_flags = []        # parallel to keyframes
    ocr_raw = []            # (labels, w, h) per chart keyframe -- None = OCR failed
    od_futures = []         # one per chart keyframe, in order (resolved below)
    frame_index = 0
    total_sampled = 0

    pool = ThreadPoolExecutor(max_workers=vision_workers)
    cancelled = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % sample_interval != 0:
                frame_index += 1
                continue
            # Cancel checkpoint once per sampled frame (1/s of video): cheap,
            # and it sits before both the OCR and the Gemini submission below.
            _stop_point()

            frame_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            total_sampled += 1
            if on_progress is not None and total_frames:
                on_progress(min(1.0, (frame_index + 1) / total_frames))

            # AnnotationChangeDetector handles its own 640x360 resize internally
            changed = detector.process_frame(frame, frame_ms)
            reason = capture.decide(frame_ms, changed)
            if reason is not None and dedupe is not None:
                # A detected change is never dropped (adaptive_capture's own
                # rule); only the extra burst/safety/cue captures are judged.
                # capture.decide() has already restarted its safety timer, so
                # a dropped frame does not make the next one fire sooner.
                if reason != "change" and dedupe.is_near_duplicate(frame):
                    deduped_frames += 1
                    reason = None
                else:
                    dedupe.keep(frame)
            if reason is not None:
                try:
                    is_chart = bool(is_chart_frame(frame))
                except Exception as exc:
                    logger.warning("[%s] chart classification failed on frame: %s", video_id, exc)
                    is_chart = False
                if is_chart:
                    # OCR and the vision model read the FULL-resolution frame.
                    # Raw readings are kept, not flattened here: the precision
                    # checks after the loop need confidence, bbox and the
                    # neighbouring keyframes to flag a misread price.
                    try:
                        # P5: tag labels crossed by a chart bar while the frame is in memory
                        from pipelines.ocr_precision import mark_occluded
                        ocr_raw.append((mark_occluded(extract_labels(frame), frame),
                                        frame.shape[1], frame.shape[0]))
                    except Exception as exc:
                        logger.warning("[%s] OCR failed on keyframe at %dms: %s",
                                       video_id, frame_ms, exc)
                        ocr_raw.append(None)
                    # D67: real vision-model description. describe_chart_frame()
                    # never raises — a missing description degrades one frame.
                    pending = [f for f in od_futures if not f.done()]
                    if len(pending) >= vision_max_pending:
                        pending[0].result()     # backpressure -- see VISION_MAX_PENDING
                    od_futures.append(pool.submit(describe_chart_frame, frame))
                keyframes.append((_shrink_for_embedding(frame), frame_ms))
                capture_reasons.append(reason)
                chart_flags.append(is_chart)

            frame_index += 1
    except JobCancelled:
        cancelled = True
        raise
    finally:
        cap.release()
        if cancelled:
            # Queued descriptions are dropped before they reach Gemini (no
            # charge); only the <= vision_workers calls already in flight
            # finish, in the background -- not the whole backlog (up to 3x
            # workers) the normal path waits for.
            for f in od_futures:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
        else:
            od_descriptions = [f.result() for f in od_futures]
            pool.shutdown(wait=True)

    ocr_texts, ocr_labels = _assess_ocr(video_id, ocr_raw, labels_to_text)

    chart_count = sum(chart_flags)
    reason_counts = {r: capture_reasons.count(r) for r in set(capture_reasons)}
    logger.info(
        "[%s] visual complete: %d sampled, %d keyframes (%s), %d charts, %d OCR texts",
        video_id, total_sampled, len(keyframes), reason_counts, chart_count, len(ocr_texts),
    )
    out = {
        "keyframes": keyframes,
        "chart_flags": chart_flags,
        "ocr_texts": ocr_texts,
        "ocr_labels": ocr_labels,
        "od_descriptions": od_descriptions,
        "capture_reasons": capture_reasons,
        # Frames actually decoded (1/s): the frame-coverage check (P137).
        "sampled_frames": total_sampled,
    }
    if dedupe is not None:
        # Only when the lever is on, so the output is unchanged when it is off.
        logger.info("[%s] near-duplicate dedupe (ssim >= %.3f) dropped %d keyframes",
                    video_id, dedupe_threshold, deduped_frames)
        out["deduped_frames"] = deduped_frames
    return out


# embed_frames_visual's default batch size -- used only to size the embed
# phase's progress bar; the call itself keeps its own default.
_FRAME_EMBED_BATCH = 96


def _phase_embed(
    keyframes: list,
    chart_flags: list,
    ocr_texts: list,
    od_descriptions: list,
    segments: list,
    video_id: str,
) -> dict:
    """Phase 4: embed all tracks via CLIP and Voyage.

    Returns {
        "clip_vectors": [...],          -- one per keyframe
        "ocr_vectors": [...],           -- one per chart keyframe with OCR text
        "od_vectors": [...],            -- one per chart keyframe with OD desc
        "transcript_vectors": [...],    -- one per ASR segment
    }
    """
    from pipelines.video_embedder import embed_frames_visual, embed_texts_voyage

    logger.info("[%s] phase 4/6: embedding", video_id)

    # Progress = Voyage batches done / total over all four tracks, and a
    # cancel checkpoint before each batch. The totals mirror the calls below
    # (frames keep embed_frames_visual's own default batch size).
    import math
    ctl = _run_ctl()
    text_bs = config.VOYAGE_BATCH_SIZE
    total_batches = [0]
    done_batches = [0]

    def _count(n, bs):
        total_batches[0] += math.ceil(n / bs) if n else 0

    def _on_batch():
        ctl.check()
        if total_batches[0]:
            ctl.progress(done_batches[0] / total_batches[0])
        done_batches[0] += 1

    # CLIP: embed all keyframes (chart and non-chart alike — visual similarity
    # search should find any frame, not only charts)
    kf_frames = [kf[0] for kf in keyframes]
    # Cost lever VIDEO_SKIP_IMAGE_EMBED_WHEN_DESCRIBED (default off): a chart
    # keyframe with a non-empty Gemini description is already searchable as
    # text in od_frames, so its Voyage image embedding is skipped (~$0.28 of
    # z6AxPV2NQ9U's $3.58). The model stays Voyage multimodal (locked, D66);
    # only which frames reach it changes. A frame whose description FAILED
    # ("") keeps its image vector, or it would be in no visual track at all.
    clip_indices = None
    if _env_flag("VIDEO_SKIP_IMAGE_EMBED_WHEN_DESCRIBED"):
        described_kf = set()
        chart_kf_indices = [i for i, is_c in enumerate(chart_flags) if is_c]
        for desc, kf_idx in zip(od_descriptions, chart_kf_indices):
            if desc.strip():
                described_kf.add(kf_idx)
        clip_indices = [i for i in range(len(keyframes)) if i not in described_kf]
        kf_frames = [kf_frames[i] for i in clip_indices]
        logger.info("[%s] image embedding skipped for %d described keyframes, %d embedded",
                    video_id, len(described_kf), len(kf_frames))
    # Still named "clip_vectors" below (D66 moved this to Voyage multimodal,
    # not CLIP, but renaming every local variable touching this value isn't
    # worth the diff — see video_embedder.py's embed_frames_visual).
    non_empty = [t for t in (ocr_texts or []) if t.strip()]
    non_empty_od = [t for t in (od_descriptions or []) if t.strip()]
    seg_texts = [w.text for w in transcript_windows(segments)] if segments else []
    _count(len(kf_frames), _FRAME_EMBED_BATCH)
    _count(len(non_empty), text_bs)
    _count(len(non_empty_od), text_bs)
    _count(len(seg_texts), text_bs)

    clip_vectors = embed_frames_visual(kf_frames, on_batch=_on_batch) if kf_frames else []

    # Voyage OCR: only non-empty OCR texts
    ocr_vectors = []
    if non_empty:
        ocr_vectors = embed_texts_voyage(non_empty, batch_size=text_bs, on_batch=_on_batch)

    # Voyage OD: real Gemini descriptions (D67). Only non-empty ones — same
    # filter the OCR block above already has. describe_chart_frame() returns
    # "" on failure by design (one frame should degrade, not fail the whole
    # video); without this filter, a single empty string in the batch would
    # make the entire embed call fail, since Voyage's API rejects blanks —
    # found in code review 2026-10-09.
    od_vectors = []
    if non_empty_od:
        od_vectors = embed_texts_voyage(non_empty_od, batch_size=text_bs, on_batch=_on_batch)

    # Voyage ASR: transcript segment texts
    transcript_vectors = []
    # Same windows _phase_index builds (deterministic), P152.
    if seg_texts:
        transcript_vectors = embed_texts_voyage(seg_texts, batch_size=text_bs, on_batch=_on_batch)

    total = len(clip_vectors) + len(ocr_vectors) + len(od_vectors) + len(transcript_vectors)
    logger.info(
        "[%s] embedding complete: %d CLIP, %d OCR, %d OD, %d ASR (%d total)",
        video_id, len(clip_vectors), len(ocr_vectors), len(od_vectors),
        len(transcript_vectors), total,
    )
    out = {
        "clip_vectors": clip_vectors,
        "ocr_vectors": ocr_vectors,
        "od_vectors": od_vectors,
        "transcript_vectors": transcript_vectors,
    }
    if clip_indices is not None:
        # Keyframe index of each clip vector; _phase_index needs it to stamp
        # each vector with its own frame_ms. Absent when the lever is off.
        out["clip_indices"] = clip_indices
    return out


def _expected_embed_counts(keyframes, ocr_texts, od_descriptions, segments, emb) -> dict:
    """Eligible items per track for check_embedding_consistency. "clip" is
    the frames that were MEANT to be image-embedded: with the skip-image
    lever on that is len(clip_indices), not every keyframe, or each lever
    run would read as an incomplete embedding and hide real drops."""
    clip_indices = emb.get("clip_indices")
    return {
        "clip": len(keyframes) if clip_indices is None else len(clip_indices),
        "ocr": sum(1 for t in ocr_texts if t.strip()),
        # Non-empty count, not len() — a batch of silent vision-
        # description failures (describe_chart_frame() -> "")
        # must count as missing, or this quality gate can never
        # catch it. Found in code review 2026-10-09.
        "od": sum(1 for t in od_descriptions if t.strip()),
        "asr": sum(1 for s in segments if s.text.strip()),
    }


def _phase_index(
    video_id: str,
    title: str,
    keyframes: list,
    chart_flags: list,
    ocr_texts: list,
    od_descriptions: list,
    segments: list,
    clip_vectors: list,
    ocr_vectors: list,
    od_vectors: list,
    transcript_vectors: list,
    platform: str = "youtube",
    source_url: str = "",
    clip_indices: list = None,
    ocr_labels: list = None,
) -> dict:
    """Phase 5: upsert all vectors to the four Qdrant collections.

    clip_indices (cost lever, _phase_embed): keyframe index of each
    clip_vector when some keyframes were not image-embedded; None = one
    vector per keyframe, in order (the default, unchanged).

    platform/source_url (P103) are written into every payload so
    citation_formatter.py can build a correct watch link later --
    YouTube's is re-derived from video_id, but Instagram/TikTok/Twitter/X
    have no synthesizable URL and would otherwise only ever get a broken
    youtube.com link.

    Returns {"total_points": int, "overview_written": bool | None} -- None when
    there was no transcript, so no overview was due.
    """
    from pipelines.video_topic_segmentation import segment_by_topic

    logger.info("[%s] phase 5/6: Qdrant upsert", video_id)
    _delete_video_points(video_id)
    total = 0
    overview_written = None

    # -- Visual frames: one point per keyframe, CLIP 512d --
    if clip_vectors:
        ids = [f"{video_id}:visual:{i}" for i in range(len(clip_vectors))]
        payloads = []
        visual_kf = range(len(keyframes)) if clip_indices is None else clip_indices
        for i in visual_kf:
            _, frame_ms = keyframes[i]
            payloads.append({
                "video_id": video_id,
                "title": title,
                "frame_ms": frame_ms,
                "is_chart": chart_flags[i] if i < len(chart_flags) else False,
                "source_type": "visual_frame",
                "platform": platform,
                "source_url": source_url,
            })
        total += _upsert_to_qdrant(
            config.VIDEO_VISUAL_COLLECTION, ids, clip_vectors, payloads,
            config.CLIP_EMBEDDING_DIM,
        )

    # -- OCR frames: one point per chart keyframe with non-empty OCR text --
    if ocr_vectors:
        # Map back to chart keyframe indices for metadata
        chart_kf_indices = [i for i, is_c in enumerate(chart_flags) if is_c]
        non_empty_ocr = [(t, idx, k) for k, (t, idx) in enumerate(zip(ocr_texts, chart_kf_indices))
                         if t.strip()]
        ids = [f"{video_id}:ocr:{i}" for i in range(len(ocr_vectors))]
        payloads = []
        for j, (text, kf_idx, k) in enumerate(non_empty_ocr):
            _, frame_ms = keyframes[kf_idx]
            payload = {
                "video_id": video_id,
                "title": title,
                "frame_ms": frame_ms,
                "ocr_text": text,
                "source_type": "ocr_frame",
                "platform": platform,
                "source_url": source_url,
            }
            # Additive (2026-10-28): per-reading value/conf/bbox/status so a
            # consumer can tell a confirmed price from a flagged one without
            # parsing the marker. Absent on points indexed before this field
            # existed -- absence means "not assessed", never "all ok".
            if ocr_labels is not None and k < len(ocr_labels):
                payload["ocr_labels"] = ocr_labels[k]
            payloads.append(payload)
        total += _upsert_to_qdrant(
            config.VIDEO_OCR_COLLECTION, ids, ocr_vectors, payloads,
            config.VOYAGE_EMBEDDING_DIM,
        )

    # -- OD frames: one point per chart keyframe with a non-empty description --
    # Same non-empty-pairing pattern the OCR block above uses — od_vectors
    # only has one entry per non-empty description now (see _phase_embed's
    # filter), so it can no longer be zipped 1:1 against every chart
    # keyframe index without the same re-pairing.
    if od_vectors:
        chart_kf_indices = [i for i, is_c in enumerate(chart_flags) if is_c]
        non_empty_od = [(t, idx) for t, idx in zip(od_descriptions, chart_kf_indices) if t.strip()]
        ids = [f"{video_id}:od:{i}" for i in range(len(od_vectors))]
        payloads = []
        for j, (text, kf_idx) in enumerate(non_empty_od):
            _, frame_ms = keyframes[kf_idx]
            payloads.append({
                "video_id": video_id,
                "title": title,
                "frame_ms": frame_ms,
                "od_detections": text,
                "source_type": "od_frame",
                "platform": platform,
                "source_url": source_url,
            })
        total += _upsert_to_qdrant(
            config.VIDEO_OD_COLLECTION, ids, od_vectors, payloads,
            config.VOYAGE_EMBEDDING_DIM,
        )

    # -- Transcript: one point per ~60 s window (P152), not per ASR segment --
    if transcript_vectors:
        non_empty_segs = transcript_windows(segments)
        ids = [f"{video_id}:asr:{i}" for i in range(len(transcript_vectors))]
        # Semantic segmentation by topic/concept (client spec item): a text
        # topic-shift confirmed by a nearby real screen change, not a fixed
        # time window. Reuses transcript_vectors already computed above —
        # no second embedding call just to find boundaries.
        topic_ids = segment_by_topic(
            non_empty_segs, transcript_vectors,
            keyframe_timestamps_ms=[frame_ms for _, frame_ms in keyframes],
        )
        payloads = []
        for j, seg in enumerate(non_empty_segs):
            payloads.append({
                "video_id": video_id,
                "title": title,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
                "confidence": seg.confidence,
                "words": seg.words,
                "topic_id": topic_ids[j],
                "source_type": "transcript_segment",
                "platform": platform,
                "source_url": source_url,
            })
        total += _upsert_to_qdrant(
            config.VIDEO_TRANSCRIPT_COLLECTION, ids, transcript_vectors, payloads,
            config.VOYAGE_EMBEDDING_DIM,
        )
        # Cancel checkpoint before the overview's paid Gemini call. The caller
        # deletes this video's (now partial) points on JobCancelled.
        _stop_point()
        written = _index_video_overview(video_id, title, non_empty_segs,
                                        platform=platform, source_url=source_url)
        total += written
        overview_written = bool(written)

    logger.info("[%s] indexed %d total points across 4 collections", video_id, total)
    return {"total_points": total, "overview_written": overview_written}


def _phase_cleanup(video_path: str, video_id: str) -> None:
    """Phase 6: delete downloaded video file (keep audio for re-processing).

    Audio is kept because re-running ASR with a better model or different
    parameters is common during tuning, and re-downloading from YouTube
    risks rate-limit bans.
    """
    logger.info("[%s] phase 6/6: cleanup", video_id)
    try:
        if os.path.isfile(video_path):
            os.remove(video_path)
            logger.info("[%s] deleted video file: %s", video_id, video_path)
    except OSError as exc:
        # Cleanup failure is non-fatal — the file wastes disk but the
        # indexed content is already in Qdrant.
        logger.warning("[%s] failed to delete video file: %s", video_id, exc)


# ── Main orchestrator ───────────────────────────────────────────────

def ingest_video(video_id: str, *args, should_stop=None, **kwargs) -> "IngestResult":
    """Run _ingest_video_phases() under a usage meter (see usage_meter.py).

    Every Groq/Gemini/Voyage response during the run is metered from its own
    reported usage; the summary lands on result.usage whether the run finished,
    failed or was cancelled part-way. Same arguments as _ingest_video_phases(),
    plus:

    should_stop : optional no-argument callable; True = the user cancelled
        (POST /video/cancel). Checked at the cooperative checkpoints listed
        above JobCancelled. The run then returns status="cancelled" with
        .phase set -- see _finish_cancelled for what is cleaned up.
    """
    meter = usage_meter.start(video_id)
    try:
        with _run_control(video_id, None, None, should_stop) as ctl:
            try:
                result = _ingest_video_phases(video_id, *args, **kwargs)
            except JobCancelled as exc:
                result = _finish_cancelled(video_id, ctl, exc)
    finally:
        usage_meter.stop()
    result.usage = meter.summary()
    logger.info("[%s] measured usage: %s", video_id, result.usage)
    return result


def _finish_cancelled(video_id: str, ctl: "_RunControl", exc: JobCancelled) -> "IngestResult":
    """Turn a JobCancelled into the run's result and undo what must not stay.

    - Nothing is upserted before the index phase, so a stop before it leaves
      Qdrant alone: a forced re-ingest of a done video keeps its old, good
      index (the router then puts that job back to 'done').
    - Once the index phase began, _phase_index has already deleted the old
      points and written some new ones: the partial set is removed, and the
      index checkpoint is marked so nothing later mistakes it for complete.
    - The downloaded video file is removed like a normal cleanup phase would
      (the audio is kept, as always), so a cancel never leaks the download.
    The job row itself is written by the router's worker, which knows
    whether a previous index exists.
    """
    result = ctl.result or IngestResult(video_id=video_id, status="cancelled")
    result.status = "cancelled"
    result.phase = exc.phase
    result.error = str(exc)
    logger.warning("[%s] %s", video_id, result.error)
    if exc.phase in ("index", "cleanup"):
        try:
            _delete_video_points(video_id)
        except Exception as del_exc:
            # Qdrant unreachable: say so on the job instead of failing the
            # cancel; a re-ingest deletes this video's points first anyway.
            logger.error("[%s] could not delete partial points after cancel: %s",
                         video_id, del_exc)
            result.stats.setdefault("warnings", []).append(
                f"partial index could not be removed after cancel: {del_exc}")
        _save_checkpoint(ctl.eval_db, video_id, "index", "cancelled", main_loop=ctl.main_loop)
    if ctl.video_path:
        _phase_cleanup(ctl.video_path, video_id)
    return result


def _ingest_video_phases(
    video_id: str,
    work_dir: str,
    eval_db=None,
    title: str = "",
    source_url: str = "",
    platform: str = "youtube",
    duration: float = 0.0,
    main_loop=None,
) -> IngestResult:
    """Run the full six-phase pipeline for one video.

    Parameters
    ----------
    video_id   : the platform's native video/post ID.
    work_dir   : directory for temporary files (video, audio). Created if needed.
    eval_db    : optional EvalDB instance for checkpoint persistence.
    title      : video title for Qdrant payloads. Fetched from metadata if empty.
    source_url : the URL this video was discovered/pasted from. Required for
                 any non-YouTube platform — see video_acquisition._resolve_download_url.
    platform   : "youtube" (default) or whatever video_catalog.platform recorded.
    duration   : video length in seconds, from video_catalog — used only to judge
                 ASR/visual quality (segments and keyframes per unit time), not
                 passed to any phase function.
    main_loop  : the event loop eval_db's asyncpg pool was created on. REQUIRED
                 whenever this function runs inside asyncio.to_thread() (the
                 production call site, routers/video.py's _process_queue) --
                 every _update_job_status/_save_checkpoint call below uses it
                 to schedule back onto that loop instead of creating a second,
                 colliding one. See _run_async's docstring and P95.

    Returns IngestResult with status "done" or "failed".
    """
    os.makedirs(work_dir, exist_ok=True)
    started = time.monotonic()
    result = IngestResult(video_id=video_id, status="failed")
    # Visible problems (admin panel), not just span labels: P137 found a video
    # that read 0 frames and still reported "done".
    warnings: List[str] = []
    result.stats["warnings"] = warnings
    sampled_frames = 0
    descriptions_missing = ""     # set when every chart frame's description failed
    # Cancel control (set up by ingest_video): give it this run's DB handle
    # and result, so progress writes and a cancel's partial result use them.
    ctl = _run_ctl()
    ctl.eval_db, ctl.main_loop, ctl.result = eval_db, main_loop, result
    ctl.check()

    with ingestion_session(video_id, platform=platform, title=title), \
            span("ingest_video", SpanKind.CHAIN, input_value=video_id,
                 attributes={"ingest.platform": platform,
                             "pipeline.phase": "ingestion"}) as root_sp:

        # Pre-flight: memory check
        if not check_memory_watermark():
            watermark = getattr(config, "MEMORY_WATERMARK_GB", 4)
            result.error = (
                f"insufficient memory: need {watermark} GB free"
            )
            record_decision(root_sp, "ingest", "aborted", result.error)
            _update_job_status(eval_db, video_id, "failed", error_message=result.error, main_loop=main_loop)
            return result

        # Pre-flight: the vision provider (chart descriptions + overview). A
        # persistent failure -- empty credit, bad key, model gone, no key -- stops
        # here, before download/Groq/Voyage spend, and is visible on the job. A
        # transient one (timeout, 429, 5xx) only warns: the run may recover, and the
        # chart-description check below catches it if it doesn't.
        if getattr(config, "INGEST_VISION_PREFLIGHT", True):
            from pipelines.provider_health import PERSISTENT
            pre = _vision_preflight()
            record_decision(root_sp, "ingest.vision_preflight", pre.category,
                            f"{pre.provider}: {pre.detail} ({pre.latency_ms}ms)")
            if pre.category in PERSISTENT:
                result.error = f"{VISION_UNAVAILABLE_PREFIX} ({pre.category}): {pre.provider} -- {pre.detail}"
                logger.error("[%s] %s", video_id, result.error)
                record_decision(root_sp, "ingest", "aborted", result.error)
                _update_job_status(eval_db, video_id, "failed", error_message=result.error, main_loop=main_loop)
                return result
            if not pre.ok:
                warnings.append(f"vision provider check before start: {pre.category} ({pre.detail}); "
                                f"chart descriptions may be incomplete")

        ctl.enter("download")
        _update_job_status(eval_db, video_id, "processing", phase="download", main_loop=main_loop)

        # ── Phase 1: Download ──
        with _timed(result, "download"), span("ingest.download", SpanKind.CHAIN, input_value=video_id,
                 attributes={"ingest.platform": platform}) as sp:
            try:
                dl = _phase_download(video_id, work_dir, platform=platform,
                                      source_url=source_url, duration=duration)
                video_path = dl["video_path"]
                audio_path = dl["audio_path"]
                ctl.video_path = video_path     # a cancel from here on still removes it
                # P104: metadata-blind platforms (Instagram confirmed) get their
                # duration backfilled via ffprobe inside _phase_download -- use
                # that corrected value for every quality check below, not the
                # possibly-zero one video_catalog recorded at discovery time.
                duration = dl["duration"]
                result.stats["duration_s"] = duration
                set_output(sp, f"video={video_path} audio={audio_path}")
                _save_checkpoint(eval_db, video_id, "download", "done", output_path=video_path, main_loop=main_loop)
                result.phases_completed.append("download")
            except JobCancelled:
                raise       # a cancel is not a phase failure (see JobCancelled)
            except Exception as exc:
                result.error = f"download failed: {exc}"
                logger.error("[%s] %s", video_id, result.error, exc_info=True)
                record_decision(sp, "ingest", "failed", result.error)
                record_decision(root_sp, "ingest", "failed", result.error)
                _update_job_status(eval_db, video_id, "failed", phase="download",
                                   error_message=result.error, main_loop=main_loop)
                return result

        # ── Phase 2: ASR ──
        ctl.enter("asr")
        _update_job_status(eval_db, video_id, "processing", phase="asr", main_loop=main_loop)
        with _timed(result, "asr"), span("ingest.asr", SpanKind.TOOL, input_value=audio_path) as sp:
            try:
                segments = _phase_asr(audio_path, video_id)
                sample = segments[0].text[:80] if segments else ""
                sp.set_attribute("asr.segment_count", len(segments))
                # .confidence is WhisperX's avg_logprob (a log probability, usually
                # negative) — NOT a 0-1 score. Named accordingly so a trace reader
                # doesn't mistake -0.3 for "30% confident".
                qc = check_asr_quality(segments, duration_s=duration)
                sp.set_attribute("asr.avg_logprob", round(qc.score, 3))
                record_decision(sp, "asr.quality", qc.label, qc.reason)
                set_output(sp, f"{len(segments)} segments, avg_logprob={qc.score:.2f}, "
                               f"sample={sample!r}")
                _save_checkpoint(eval_db, video_id, "asr", "done", item_count=len(segments),
                                main_loop=main_loop)
                result.phases_completed.append("asr")
                result.stats["transcript_segments"] = len(segments)
                tqc = check_transcript_coverage(segments, duration_s=duration)
                record_decision(sp, "asr.coverage", tqc.label, tqc.reason)
                if tqc.label in ("truncated", "gap"):
                    warnings.append(tqc.reason)
            except JobCancelled:
                raise       # a cancel is not a phase failure (see JobCancelled)
            except Exception as exc:
                result.error = f"ASR failed: {exc}"
                logger.error("[%s] %s", video_id, result.error, exc_info=True)
                record_decision(sp, "ingest", "failed", result.error)
                record_decision(root_sp, "ingest", "failed", result.error)
                _update_job_status(eval_db, video_id, "failed", phase="asr",
                                   error_message=result.error, main_loop=main_loop)
                return result

        # ── Phase 3: Visual extraction ──
        ctl.enter("visual")
        _update_job_status(eval_db, video_id, "processing", phase="visual", main_loop=main_loop)
        with _timed(result, "visual"), span("ingest.visual", SpanKind.CHAIN, input_value=video_path) as sp:
            try:
                # Throttled progress writes: the admin panel shows % and time
                # left from phase_progress; one DB write per ~15 s, not per frame.
                last_write = [0.0]

                def _visual_progress(fraction):
                    now = time.monotonic()
                    if now - last_write[0] >= 15.0:
                        last_write[0] = now
                        _update_job_status(eval_db, video_id, "processing", phase="visual",
                                           progress=fraction, main_loop=main_loop)

                vis = _phase_visual(video_path, video_id, segments=segments,
                                    on_progress=_visual_progress)
                keyframes = vis["keyframes"]
                chart_flags = vis["chart_flags"]
                ocr_texts = vis["ocr_texts"]
                ocr_labels = vis.get("ocr_labels")
                od_descriptions = vis["od_descriptions"]
                capture_reasons = vis.get("capture_reasons", [])

                chart_count = sum(chart_flags)
                non_empty_ocr = sum(1 for t in ocr_texts if t.strip())
                sp.set_attribute("visual.keyframe_count", len(keyframes))
                sp.set_attribute("visual.chart_count", chart_count)
                sp.set_attribute("visual.ocr_non_empty_count", non_empty_ocr)
                for reason in ("change", "burst", "cue", "safety"):
                    sp.set_attribute(f"visual.capture_count.{reason}",
                                     capture_reasons.count(reason))
                qc = check_visual_quality(len(keyframes), duration_s=duration)
                sp.set_attribute("visual.keyframes_per_hour", round(qc.score, 1))
                record_decision(sp, "visual.quality", qc.label, qc.reason)
                set_output(sp, f"{len(keyframes)} keyframes, {chart_count} charts, "
                               f"{non_empty_ocr}/{len(ocr_texts)} OCR texts non-empty")

                _save_checkpoint(eval_db, video_id, "visual", "done",
                                 item_count=len(keyframes), main_loop=main_loop)
                result.phases_completed.append("visual")
                result.stats["frames_extracted"] = len(keyframes)
                result.stats["keyframes"] = len(keyframes)
                if "deduped_frames" in vis:
                    # Cost lever on: how many near-duplicates never reached
                    # OCR/Gemini/embedding -- the trial's main number.
                    result.stats["keyframes_deduped"] = vis["deduped_frames"]
                result.stats["ocr_labels"] = len(ocr_texts)
                sampled_frames = vis.get("sampled_frames", 0)
                cqc = check_chart_talk(segments, chart_count, len(keyframes), duration_s=duration)
                record_decision(sp, "visual.chart_talk", cqc.label, cqc.reason)
                if cqc.label == "charts_missed":
                    warnings.append(cqc.reason)
                # describe_chart_frame() turns every failure into "": count them.
                described = sum(1 for t in od_descriptions if t.strip())
                dqc = check_chart_descriptions(chart_count, described)
                record_decision(sp, "visual.chart_descriptions", dqc.label, dqc.reason)
                result.stats["chart_descriptions"] = "%d/%d" % (described, chart_count)
                gem = usage_meter.current_gemini_summary()
                if "flex_fallbacks" in gem:
                    result.stats["flex_fallbacks"] = gem["flex_fallbacks"]
                fbw = flex_fallback_warning(gem)
                if fbw:
                    warnings.append(fbw)
                if dqc.label == "partial":
                    warnings.append(dqc.reason)
                elif dqc.label == "missing":
                    descriptions_missing = dqc.reason
            except JobCancelled:
                raise       # a cancel is not a phase failure (see JobCancelled)
            except Exception as exc:
                # Visual failure is non-fatal — we still have transcript to index.
                # Log the error but continue with empty visual data.
                logger.error("[%s] visual extraction failed (continuing with ASR only): %s",
                             video_id, exc, exc_info=True)
                record_decision(sp, "ingest", "degraded",
                                f"visual extraction failed, continuing ASR-only: {exc}")
                _save_checkpoint(eval_db, video_id, "visual", "failed", main_loop=main_loop)
                keyframes = []
                chart_flags = []
                ocr_texts = []
                ocr_labels = None
                od_descriptions = []
                sampled_frames = 0
                result.stats["frames_extracted"] = 0
                result.stats["keyframes"] = 0
                result.stats["ocr_labels"] = 0

        # ── Phase 4: Embedding ──
        ctl.enter("embed")
        _update_job_status(eval_db, video_id, "processing", phase="embed", main_loop=main_loop)
        with _timed(result, "embed"), span("ingest.embed", SpanKind.EMBEDDING,
                 input_value=f"{len(keyframes)} frames, {len(segments)} segments") as sp:
            try:
                emb = _phase_embed(
                    keyframes, chart_flags, ocr_texts, od_descriptions,
                    segments, video_id,
                )
                clip_vectors = emb["clip_vectors"]
                ocr_vectors = emb["ocr_vectors"]
                od_vectors = emb["od_vectors"]
                transcript_vectors = emb["transcript_vectors"]
                total_emb = (len(clip_vectors) + len(ocr_vectors)
                             + len(od_vectors) + len(transcript_vectors))
                expected = _expected_embed_counts(keyframes, ocr_texts, od_descriptions,
                                                  segments, emb)
                clip_indices = emb.get("clip_indices")
                qc = check_embedding_consistency(expected, total_emb)
                record_decision(sp, "embed.quality", qc.label, qc.reason)
                set_output(sp, f"{total_emb} vectors (clip={len(clip_vectors)} "
                               f"ocr={len(ocr_vectors)} od={len(od_vectors)} "
                               f"asr={len(transcript_vectors)})")
                _save_checkpoint(eval_db, video_id, "embed", "done", item_count=total_emb, main_loop=main_loop)
                result.phases_completed.append("embed")
                result.stats["embeddings_created"] = total_emb
                # Reused by routers/video.py's cost ledger (D68) — without
                # this, the OD text-embedding calls were invisible to spend
                # tracking entirely. Found in code review 2026-10-09.
                result.stats["od_descriptions"] = expected["od"]
                result.stats["text_items_embedded"] = expected["ocr"] + expected["od"] + expected["asr"]
            except JobCancelled:
                raise       # a cancel is not a phase failure (see JobCancelled)
            except Exception as exc:
                result.error = f"embedding failed: {exc}"
                logger.error("[%s] %s", video_id, result.error, exc_info=True)
                record_decision(sp, "ingest", "failed", result.error)
                record_decision(root_sp, "ingest", "failed", result.error)
                _update_job_status(eval_db, video_id, "failed", phase="embed",
                                   error_message=result.error, main_loop=main_loop)
                return result

        # ── Phase 5: Qdrant upsert ──
        ctl.enter("index")
        _update_job_status(eval_db, video_id, "processing", phase="index", main_loop=main_loop)
        with _timed(result, "index"), span("ingest.index", SpanKind.CHAIN, input_value=f"{total_emb} vectors") as sp:
            try:
                idx = _phase_index(
                    video_id, title, keyframes, chart_flags,
                    ocr_texts, od_descriptions, segments,
                    clip_vectors, ocr_vectors, od_vectors, transcript_vectors,
                    platform=platform, source_url=source_url,
                    # Passed only when the skip-image lever produced it, so
                    # the default call is exactly the pre-lever one.
                    **({"clip_indices": clip_indices} if clip_indices is not None else {}),
                    # Same rule: only when the visual phase produced labels.
                    **({"ocr_labels": ocr_labels} if ocr_labels else {}),
                )
                set_output(sp, f"{idx['total_points']} points indexed across 4 collections")
                _save_checkpoint(eval_db, video_id, "index", "done",
                                 item_count=idx["total_points"], main_loop=main_loop)
                result.phases_completed.append("index")
                # Optional output, but never silent: without it "summarize this
                # video" is answered from passages only (gemini_down.md item B).
                if idx.get("overview_written") is False:
                    reason = ("no video overview was written (generation failed or returned "
                              "nothing -- see 'video overview' in the logs); summary questions "
                              "fall back to passages")
                    warnings.append(reason)
                    record_decision(sp, "index.overview", "missing", reason)
                result.stats["overview_written"] = idx.get("overview_written")
            except JobCancelled:
                raise       # a cancel is not a phase failure (see JobCancelled)
            except Exception as exc:
                result.error = f"indexing failed: {exc}"
                # _phase_index deletes the old points first: whatever is left
                # is partial, so the old "index done" checkpoint no longer
                # describes Qdrant (the router reads it to decide whether a
                # cancelled re-ingest kept a complete previous index).
                _save_checkpoint(eval_db, video_id, "index", "failed", main_loop=main_loop)
                logger.error("[%s] %s", video_id, result.error, exc_info=True)
                record_decision(sp, "ingest", "failed", result.error)
                record_decision(root_sp, "ingest", "failed", result.error)
                _update_job_status(eval_db, video_id, "failed", phase="index",
                                   error_message=result.error, main_loop=main_loop)
                return result

        # ── Phase 6: Cleanup ──
        with _timed(result, "cleanup"), span("ingest.cleanup", SpanKind.CHAIN, input_value=video_path) as sp:
            _phase_cleanup(video_path, video_id)
            set_output(sp, "video file removed, audio kept")
            _save_checkpoint(eval_db, video_id, "cleanup", "done", main_loop=main_loop)
            result.phases_completed.append("cleanup")

        # ── Done (or failed: nothing could be decoded) ──
        elapsed = time.monotonic() - started
        fqc = check_frame_coverage(sampled_frames, duration_s=duration)
        record_decision(root_sp, "visual.coverage", fqc.label, fqc.reason)
        if fqc.label == "unreadable":
            # The transcript is indexed, but the video's frames never were: a
            # "done" here is the silent failure P137 describes. Retry re-indexes.
            result.status = "failed"
            result.error = "visual: " + fqc.reason
            logger.error("[%s] %s", video_id, result.error)
            _update_job_status(eval_db, video_id, "failed", phase="visual",
                               error_message=result.error, main_loop=main_loop)
            return result
        if fqc.label == "partial":
            warnings.append(fqc.reason)
        if descriptions_missing:
            # Same shape as "unreadable" above: the transcript is indexed, but a
            # chart-heavy video has no chart descriptions at all -- a provider
            # outage, not a finished video. Retry re-indexes (fixed point ids,
            # _delete_video_points first), so it cannot duplicate anything.
            result.status = "failed"
            result.error = "visual: " + descriptions_missing
            logger.error("[%s] %s", video_id, result.error)
            record_decision(root_sp, "ingest", "failed", result.error)
            _update_job_status(eval_db, video_id, "failed", phase="visual",
                               error_message=result.error, main_loop=main_loop)
            return result
        result.status = "done"
        _update_job_status(eval_db, video_id, "done", phase="cleanup", main_loop=main_loop)
        record_decision(root_sp, "ingest", "done",
                        f"{len(result.phases_completed)} phases completed in {elapsed:.1f}s")
        set_output(root_sp, f"{idx['total_points']} points indexed, stats={result.stats}")
        logger.info("[%s] ingestion complete in %.1fs: %s", video_id, elapsed, result.stats)
        return result


# ── Crash recovery ──────────────────────────────────────────────────

def resume_interrupted(
    video_id: str,
    work_dir: str,
    eval_db=None,
    title: str = "",
    source_url: str = "",
    platform: str = "youtube",
    duration: float = 0.0,
    main_loop=None,
) -> IngestResult:
    """Resume a crashed pipeline from the last completed checkpoint.

    Reads checkpoints from eval_db to determine which phases completed
    successfully, then re-runs from the first incomplete phase. When
    eval_db is None, behaves identically to ingest_video() (starts
    from scratch) because there are no checkpoints to read.

    title/source_url/platform/duration are forwarded to ingest_video() the
    same way video_id/work_dir/eval_db already were — found missing in code
    review: a resumed video was silently losing its title and source_url,
    which breaks any non-YouTube download (video_acquisition requires
    source_url) and strips the title from every Qdrant payload.

    main_loop -- see ingest_video()'s docstring (P95): required whenever
    this itself runs inside asyncio.to_thread().
    """
    checkpoints = _get_checkpoints(eval_db, video_id)
    if not checkpoints:
        logger.info("[%s] no checkpoints found, starting from scratch", video_id)
        return ingest_video(
            video_id, work_dir, eval_db=eval_db, title=title,
            source_url=source_url, platform=platform, duration=duration,
            main_loop=main_loop,
        )

    # Find the first incomplete phase
    completed = {p for p, s in checkpoints.items() if s == "done"}
    resume_from = None
    for phase in PHASE_ORDER:
        if phase not in completed:
            resume_from = phase
            break

    if resume_from is None:
        # All phases completed — nothing to resume
        logger.info("[%s] all phases already completed", video_id)
        return IngestResult(
            video_id=video_id,
            status="done",
            phases_completed=list(PHASE_ORDER),
        )

    logger.info("[%s] resuming from phase '%s' (completed: %s)",
                video_id, resume_from, sorted(completed))

    # For now, resume restarts from the first incomplete phase.
    # Intermediate artifacts (audio, keyframes) are ephemeral and not
    # serialised to disk between phases, so we re-run from scratch but
    # skip phases that have completed. A future optimisation could
    # serialise intermediate state to work_dir.
    return ingest_video(
        video_id, work_dir, eval_db=eval_db, title=title,
        source_url=source_url, platform=platform, duration=duration,
        main_loop=main_loop,
    )


# ── Offline self-test ───────────────────────────────────────────────

def _self_test():
    """Numbered offline checks — no network, no database, no video files."""
    global check_memory_watermark, _phase_download, _phase_asr, _phase_visual
    global _phase_embed, _phase_index, _phase_cleanup, ingest_video, _update_job_status
    global _vision_preflight
    import tempfile

    # Offline: the vision preflight (one real Gemini call) answers "ok" for every
    # check below except 20a-20e, which set their own outcome.
    from pipelines.provider_health import HealthResult as _HR
    saved_preflight = _vision_preflight
    _vision_preflight = lambda: _HR("gemini:test", "ok", 200, 1, "answered")

    # 1. IngestResult dataclass construction
    r = IngestResult(video_id="abc123", status="done")
    assert r.video_id == "abc123"
    assert r.status == "done"
    assert r.phases_completed == []
    assert r.error is None
    assert r.stats == {}
    print("ok 1 — IngestResult dataclass construction")

    # 2. IngestResult with all fields
    r2 = IngestResult(
        video_id="xyz", status="failed",
        phases_completed=["download", "asr"],
        error="embedding failed",
        stats={"keyframes": 42, "transcript_segments": 100},
    )
    assert r2.error == "embedding failed"
    assert r2.stats["keyframes"] == 42
    assert len(r2.phases_completed) == 2
    print("ok 2 — IngestResult with all fields populated")

    # 3. check_memory_watermark with mock
    import types
    fake_psutil = types.ModuleType("psutil")

    class FakeVMem:
        def __init__(self, avail):
            self.available = avail
    # Monkey-patch: temporarily replace the import
    import sys
    sys.modules["psutil"] = fake_psutil

    # 5 GB available, watermark is 4 GB → should pass
    fake_psutil.virtual_memory = lambda: FakeVMem(5 * 1024 ** 3)
    assert check_memory_watermark() is True
    print("ok 3 — memory watermark passes when enough RAM")

    # 4. check_memory_watermark fails when low
    fake_psutil.virtual_memory = lambda: FakeVMem(2 * 1024 ** 3)
    assert check_memory_watermark() is False
    print("ok 4 — memory watermark fails when RAM too low")

    # Clean up mock
    del sys.modules["psutil"]

    # 5. _get_checkpoints returns empty dict when eval_db is None
    cp = _get_checkpoints(None, "test_video")
    assert cp == {}
    print("ok 5 — _get_checkpoints returns {} when eval_db is None")

    # 6. _save_checkpoint is no-op when eval_db is None (no exception)
    _save_checkpoint(None, "test_video", "download", "done")
    print("ok 6 — _save_checkpoint no-op when eval_db is None")

    # 7. PHASE_ORDER has correct phases in correct order
    assert PHASE_ORDER == ("download", "asr", "visual", "embed", "index", "cleanup")
    assert len(PHASE_ORDER) == 6
    print("ok 7 — PHASE_ORDER has 6 phases in correct sequence")

    # 8. Phase ordering: ASR before visual (WhisperX memory constraint)
    asr_idx = PHASE_ORDER.index("asr")
    vis_idx = PHASE_ORDER.index("visual")
    assert asr_idx < vis_idx, "ASR must run before visual to free WhisperX memory"
    print("ok 8 — ASR phase ordered before visual (memory constraint)")

    # 9. Cleanup removes video file (with mock)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
        f.write(b"fake video content")
    assert os.path.isfile(tmp_path), "temp file should exist before cleanup"
    _phase_cleanup(tmp_path, "test_cleanup")
    assert not os.path.isfile(tmp_path), "temp file should be deleted after cleanup"
    print("ok 9 — _phase_cleanup deletes video file")

    # 10. Cleanup handles missing file gracefully
    _phase_cleanup("/nonexistent/path/video.mp4", "test_missing")
    print("ok 10 — _phase_cleanup handles missing file without error")

    # 11. _update_job_status is no-op when eval_db is None
    _update_job_status(None, "test", "processing", phase="download")
    print("ok 11 — _update_job_status no-op when eval_db is None")

    # 12. IngestResult default factory isolation (mutable defaults)
    r_a = IngestResult(video_id="a", status="done")
    r_b = IngestResult(video_id="b", status="done")
    r_a.phases_completed.append("download")
    r_a.stats["keyframes"] = 10
    assert r_b.phases_completed == [], "mutable default must not leak between instances"
    assert r_b.stats == {}, "mutable default must not leak between instances"
    print("ok 12 — IngestResult mutable defaults are isolated")

    # 13. _phase_download forwards platform/source_url so a non-YouTube video
    # downloads from its recorded URL instead of a synthesized YouTube one.
    import types as _types
    fake_acq = _types.ModuleType("pipelines.video_acquisition")
    captured = {}

    def fake_download_video(video_id, work_dir, platform="youtube", source_url=""):
        captured["video_id"] = video_id
        captured["platform"] = platform
        captured["source_url"] = source_url
        return f"{work_dir}/{video_id}.mp4"

    def fake_extract_audio(video_path, work_dir=None):
        return video_path.replace(".mp4", ".wav")

    probe_calls = []

    def fake_probe_duration_seconds(video_path):
        probe_calls.append(video_path)
        return 63.5

    fake_acq.download_video = fake_download_video
    fake_acq.extract_audio = fake_extract_audio
    fake_acq.probe_duration_seconds = fake_probe_duration_seconds
    import sys as _sys
    _sys.modules["pipelines.video_acquisition"] = fake_acq
    try:
        result = _phase_download(
            "1234567890", "/tmp/work",
            platform="twitter", source_url="https://twitter.com/u/status/1234567890")
    finally:
        del _sys.modules["pipelines.video_acquisition"]

    assert captured["platform"] == "twitter", captured
    assert captured["source_url"] == "https://twitter.com/u/status/1234567890", captured
    assert result["video_path"] == "/tmp/work/1234567890.mp4"
    print("ok 13 — _phase_download forwards platform/source_url to download_video")

    # 13b. P104: when video_catalog's duration is unknown (0 — confirmed true
    # for every Instagram reel, yt-dlp never returns it in metadata-only mode),
    # _phase_download backfills it via ffprobe on the file it just downloaded.
    probe_calls.clear()
    _sys.modules["pipelines.video_acquisition"] = fake_acq
    try:
        result13b = _phase_download("ig1", "/tmp/work", platform="instagram",
                                     source_url="https://www.instagram.com/reel/ig1/",
                                     duration=0.0)
    finally:
        del _sys.modules["pipelines.video_acquisition"]
    assert probe_calls == ["/tmp/work/ig1.mp4"], probe_calls
    assert result13b["duration"] == 63.5, result13b
    print("ok 13b — _phase_download backfills an unknown duration via ffprobe (P104)")

    # 13c. A duration already known from metadata (the YouTube/normal case)
    # must NOT trigger an extra ffprobe call — only an actually-missing
    # value is worth the subprocess cost.
    probe_calls.clear()
    _sys.modules["pipelines.video_acquisition"] = fake_acq
    try:
        result13c = _phase_download("yt1", "/tmp/work", platform="youtube",
                                     duration=600.0)
    finally:
        del _sys.modules["pipelines.video_acquisition"]
    assert probe_calls == [], probe_calls
    assert result13c["duration"] == 600.0, result13c
    print("ok 13c — _phase_download trusts an already-known duration, skipping ffprobe")

    # 14. Full happy-path ingest_video() with every phase mocked — the one check
    # that actually calls ingest_video() itself, so a control-flow mistake in the
    # span-wrapping refactor (wrong indentation, a variable now out of scope,
    # a swallowed return) fails here instead of shipping unnoticed.
    saved_fns = (check_memory_watermark, _phase_download, _phase_asr,
                _phase_visual, _phase_embed, _phase_index, _phase_cleanup)

    class FakeSegment:
        def __init__(self, text, confidence):
            self.text = text
            self.confidence = confidence
            self.words = []

    def fake_phase_download(video_id, work_dir, platform="youtube", source_url="", duration=0.0):
        return {"video_path": f"{work_dir}/{video_id}.mp4",
                "audio_path": f"{work_dir}/{video_id}.wav",
                "duration": duration}

    def fake_phase_asr(audio_path, video_id):
        from pipelines import usage_meter
        usage_meter.record_groq(42.0)   # a real phase's API call, metered
        return [FakeSegment("service level agreement forms here", 0.92)]

    def fake_phase_visual(video_path, video_id, segments=None, on_progress=None):
        return {"keyframes": [("frame0", 1000)], "chart_flags": [True],
                "ocr_texts": ["SLA 5021.25"], "od_descriptions": ["chart_frame"],
                "capture_reasons": ["change"]}

    def fake_phase_embed(keyframes, chart_flags, ocr_texts, od_descriptions, segments, video_id):
        return {"clip_vectors": [[0.1]], "ocr_vectors": [[0.2]],
                "od_vectors": [[0.3]], "transcript_vectors": [[0.4]]}

    def fake_phase_index(video_id, title, keyframes, chart_flags, ocr_texts,
                         od_descriptions, segments, clip_v, ocr_v, od_v, transcript_v,
                         platform="youtube", source_url=""):
        return {"total_points": 4}

    def fake_phase_cleanup(video_path, video_id):
        pass

    check_memory_watermark = lambda: True
    _phase_download = fake_phase_download
    _phase_asr = fake_phase_asr
    _phase_visual = fake_phase_visual
    _phase_embed = fake_phase_embed
    _phase_index = fake_phase_index
    _phase_cleanup = fake_phase_cleanup
    try:
        result14 = ingest_video("vid14", "/tmp/work14", title="Test Video")
    finally:
        (check_memory_watermark, _phase_download, _phase_asr,
         _phase_visual, _phase_embed, _phase_index, _phase_cleanup) = saved_fns

    assert result14.status == "done", result14
    assert result14.phases_completed == list(PHASE_ORDER), result14.phases_completed
    assert result14.stats["transcript_segments"] == 1, result14.stats
    assert result14.stats["keyframes"] == 1, result14.stats
    assert result14.stats["embeddings_created"] == 4, result14.stats
    assert result14.error is None, result14.error
    print("ok 14 — ingest_video() happy path completes all 6 phases with spans wrapping each")

    # 14b. Exact wall-clock per stage and measured API usage travel with the
    # result -- the checkpoint promised to the client (2026-10-19) needs both, and
    # the meter must be closed afterwards so chat search is never billed.
    from pipelines import usage_meter as _um
    assert set(result14.stats["phase_seconds"]) == set(PHASE_ORDER), result14.stats
    assert all(v >= 0 for v in result14.stats["phase_seconds"].values())
    assert result14.usage["video_id"] == "vid14", result14.usage
    assert result14.usage["groq"]["audio_seconds"] == 42.0, result14.usage
    assert _um.stop() is None, "meter left active after ingest_video"
    print("ok 14b — per-stage wall-clock and metered usage returned; meter closed")

    # 14c. Every job status the pipeline writes is one the router and admin
    # panel know. It wrote "running", which nothing else recognises: the video
    # being ingested vanished from every admin tab and from the router's
    # "already queued/processing" guard (2026-10-19).
    saved_ujs = _update_job_status
    written = []
    _update_job_status = lambda db, vid, status, phase="", error_message="", main_loop=None, progress=0.0:         written.append((status, phase, progress))
    check_memory_watermark = lambda: True
    (_phase_download, _phase_asr, _phase_visual, _phase_embed, _phase_index,
     _phase_cleanup) = (fake_phase_download, fake_phase_asr, fake_phase_visual,
                        fake_phase_embed, fake_phase_index, fake_phase_cleanup)
    try:
        ingest_video("vid14c", "/tmp/work14c", eval_db=object(), title="T")
    finally:
        _update_job_status = saved_ujs
        (check_memory_watermark, _phase_download, _phase_asr,
         _phase_visual, _phase_embed, _phase_index, _phase_cleanup) = saved_fns
    statuses = {s for s, _, _ in written}
    assert statuses <= {"processing", "done", "failed"}, written
    assert [p for s, p, _ in written if s == "processing"] == ["download", "asr", "visual", "embed", "index"], written
    print("ok 14c — pipeline writes only statuses the router/admin know ('processing', not 'running')")

    # 14d. P137 and the transcript/chart checks surface as a FAILED job or as
    # warnings, never as a silent "done". 10-minute video:
    #  - 0 frames read -> failed, "no video frames could be read";
    #  - frames read, transcript stopping at 1:40 -> done + a warning.
    class Seg14d:
        def __init__(self, t):
            self.text, self.start_time, self.end_time, self.confidence, self.words = "talk", t, t + 5, -0.2, []

    def visual_with(sampled):
        def _v(video_path, video_id, segments=None, on_progress=None):
            return {"keyframes": [("frame0", 1000)] if sampled else [], "chart_flags": [True] if sampled else [],
                    "ocr_texts": ["SLA"] if sampled else [], "od_descriptions": ["desc"] if sampled else [],
                    "capture_reasons": ["change"] if sampled else [], "sampled_frames": sampled}
        return _v

    def run14d(sampled, seg_times):
        global check_memory_watermark, _phase_download, _phase_asr, _phase_visual, _phase_embed
        global _phase_index, _phase_cleanup
        check_memory_watermark = lambda: True
        _phase_download = lambda video_id, work_dir, platform="youtube", source_url="", duration=0.0: {
            "video_path": "/tmp/v.mp4", "audio_path": "/tmp/v.wav", "duration": 600.0}
        _phase_asr = lambda audio_path, video_id: [Seg14d(t) for t in seg_times]
        _phase_visual = visual_with(sampled)
        _phase_embed = fake_phase_embed
        _phase_index = fake_phase_index
        _phase_cleanup = fake_phase_cleanup
        try:
            return ingest_video("vid14d", "/tmp/work14d", title="T", duration=600.0)
        finally:
            (check_memory_watermark, _phase_download, _phase_asr,
             _phase_visual, _phase_embed, _phase_index, _phase_cleanup) = saved_fns

    unreadable = run14d(0, [i * 60.0 for i in range(10)])
    assert unreadable.status == "failed", unreadable
    assert "no video frames could be read" in (unreadable.error or ""), unreadable.error
    short = run14d(600, [0.0, 50.0, 95.0])
    assert short.status == "done", short
    assert any("transcript stops at" in w for w in short.stats.get("warnings", [])), short.stats
    clean = run14d(600, [i * 60.0 for i in range(10)])
    assert clean.status == "done" and clean.stats.get("warnings") == [], clean.stats
    print("ok 14d — unreadable video -> failed; early-ending transcript -> done with a warning")

    # 15. _phase_index itself (not mocked, unlike check 14's wholesale fake)
    # threads topic_id into transcript payloads correctly — a bug in the
    # new segment_by_topic wiring would otherwise slip past every other
    # check in this file.
    global _upsert_to_qdrant, _delete_video_points, _index_video_overview, _overview_generate
    saved_upsert = _upsert_to_qdrant
    saved_delete = _delete_video_points
    saved_overview = _index_video_overview
    overview_calls = []
    _index_video_overview = lambda vid, title, wins, platform="", source_url="": (
        overview_calls.append((vid, len(wins))) or 0)
    captured_payloads = {}
    call_order = []

    def fake_delete(video_id):
        call_order.append(("delete", video_id))
    _delete_video_points = fake_delete

    def fake_upsert(collection, ids, vectors, payloads, dimensions):
        call_order.append(("upsert", collection))
        captured_payloads[collection] = payloads
        return len(ids)

    _upsert_to_qdrant = fake_upsert

    class TopicSeg:
        def __init__(self, text, start_time, confidence=-0.2):
            self.text, self.start_time, self.end_time = text, start_time, start_time + 2
            self.confidence, self.words = confidence, []

    # 100 s apart: far enough to land in separate ~60 s transcript windows (P152)
    segs15 = [TopicSeg("discussing service level agreements", 0.0),
              TopicSeg("now the load balancer setup", 100.0)]
    try:
        idx_result = _phase_index(
            "vid15", "Topic Test", keyframes=[("f", 99800)], chart_flags=[False],
            ocr_texts=[], od_descriptions=[], segments=segs15,
            clip_vectors=[], ocr_vectors=[], od_vectors=[],
            transcript_vectors=[[1.0, 0.0], [0.0, 1.0]],  # orthogonal -> topic shift
            platform="instagram", source_url="https://www.instagram.com/reel/vid15/",
        )
    finally:
        _upsert_to_qdrant = saved_upsert
        _delete_video_points = saved_delete
        _index_video_overview = saved_overview

    transcript_payloads = captured_payloads[config.VIDEO_TRANSCRIPT_COLLECTION]
    assert transcript_payloads[0]["topic_id"] == 0, transcript_payloads
    assert transcript_payloads[1]["topic_id"] == 1, transcript_payloads
    assert idx_result["total_points"] == 2, idx_result
    # P103: platform/source_url must reach the payload, or citation_formatter
    # has nothing but video_id to build a link from and falls back to a
    # youtube.com URL that never existed for a non-YouTube video.
    assert transcript_payloads[0]["platform"] == "instagram", transcript_payloads
    assert transcript_payloads[0]["source_url"] == "https://www.instagram.com/reel/vid15/", \
        transcript_payloads
    print("ok 15 — _phase_index assigns topic_id via segment_by_topic, confirmed by a nearby keyframe, "
          "and carries platform/source_url into every payload (P103)")

    # 15b. A re-index produces fewer transcript points (windows) than the old
    # per-segment index did; ids are positional ({video_id}:asr:{i}), so the
    # old tail would linger as stale points. The video's points go first.
    assert call_order[0] == ("delete", "vid15"), call_order
    assert all(op == "upsert" for op, _ in call_order[1:]), call_order
    print("ok 15b — _phase_index deletes the video's old points before upserting")

    # 15c. P152: transcripts are indexed as ~60 s windows, not ~7 s Whisper
    # segments. A lone segment ("And why does the system even need a rate
    # limit?") reranked 0.537 against the user's question and fell under
    # the 0.70 floor; the 60 s window holding its answer scored 0.751.
    class WSeg:
        def __init__(self, i):
            self.text, self.start_time, self.end_time = "s%d" % i, 10.0 * i, 10.0 * i + 10
            self.confidence, self.words = -0.1 * i, [{"word": "s%d" % i, "start": 10.0 * i, "end": 10.0 * i + 1}]
    wsegs = [WSeg(i) for i in range(10)]                 # 0..100 s
    wins = transcript_windows(wsegs + [type("E", (), {"text": "  ", "start_time": 200.0, "end_time": 201.0,
                                                       "confidence": 0.0, "words": []})()])
    assert len(wins) >= 2, wins
    assert all(w.end_time - w.start_time <= TRANSCRIPT_WINDOW_S for w in wins), [(w.start_time, w.end_time) for w in wins]
    covered = {t for w in wins for t in w.text.split()}
    assert covered == {"s%d" % i for i in range(10)}, covered          # every segment, blank one dropped
    assert wins[0].text.startswith("s0 s1") and wins[0].start_time == 0.0
    assert wins[1].start_time < wins[0].end_time, "consecutive windows overlap"
    assert [x["word"] for x in wins[0].words][:2] == ["s0", "s1"]       # word timestamps kept
    long_seg = WSeg(0); long_seg.end_time = 90.0                       # one segment longer than a window
    assert len(transcript_windows([long_seg])) == 1
    print("ok 15c — transcript_windows: <=60 s overlapping windows covering every segment, words kept")

    # 15d. P152 stage B: _phase_index writes a per-video overview from the
    # same windows ("summarize this video" can't be answered by passage
    # similarity). Prompt lines are numbered passages in time order; the real
    # times are put back afterwards (gemini_down.md item C, check 20f).
    assert overview_calls == [("vid15", 2)], overview_calls
    prompt = _overview_prompt("Episode 12", [TranscriptWindow("sla talk", 4075.0, 4133.0, -0.1),
                                             TranscriptWindow("intro", 5.0, 60.0, -0.1)])
    assert "Episode 12" in prompt and "[#1] intro" in prompt and "[#2] sla talk" in prompt, prompt
    print("ok 15d — _phase_index writes the overview; its prompt numbers the passages in time order")

    # 15e. The overview point: id {video_id}:overview, flagged overview=True,
    # cited like a transcript passage; no text -> nothing written (non-fatal).
    import types as _t15e
    saved_gen = _overview_generate
    upserts15e = []
    fake_emb15e = _t15e.ModuleType("pipelines.video_embedder")
    fake_emb15e.embed_texts_voyage = lambda texts, batch_size=96: [[0.5] * 4 for _ in texts]
    sys.modules["pipelines.video_embedder"] = fake_emb15e
    _upsert_to_qdrant = lambda coll, ids, vecs, payloads, dim: upserts15e.append((coll, ids, payloads)) or len(ids)
    try:
        _overview_generate = lambda p: "Main concepts: 1. SLA at 1:07:55"
        n_ok = _index_video_overview("vidE", "Episode 12", [TranscriptWindow("x", 0.0, 50.0, 0.0)],
                                     platform="youtube", source_url="")
        _overview_generate = lambda p: ""
        n_empty = _index_video_overview("vidE", "Episode 12", [TranscriptWindow("x", 0.0, 50.0, 0.0)])
    finally:
        _overview_generate = saved_gen
        _upsert_to_qdrant = saved_upsert
        del sys.modules["pipelines.video_embedder"]
    assert (n_ok, n_empty) == (1, 0), (n_ok, n_empty)
    coll, ids, payloads = upserts15e[0]
    assert coll == config.VIDEO_TRANSCRIPT_COLLECTION and ids == ["vidE:overview"], upserts15e
    assert payloads[0]["overview"] is True and payloads[0]["source_type"] == "transcript_segment"
    assert payloads[0]["text"].startswith("Main concepts") and payloads[0]["start_time"] == 0.0
    print("ok 15e — overview point id/flags/text; empty overview writes nothing")

    # 15f. A truncated overview is worse than none: gemini-3.6-flash spends
    # thinking tokens out of max_output_tokens, and at 1,500 the live overview
    # stopped after 303 chars mid-sentence (finish_reason MAX_TOKENS,
    # 2026-10-20). MAX_TOKENS -> "" (no overview, passages answer instead).
    fake_genai15f = _t15e.ModuleType("google.genai")
    fake_types15f = _t15e.ModuleType("google.genai.types")
    for name in ("GenerateContentConfig", "ThinkingConfig"):
        setattr(fake_types15f, name, lambda **kw: kw)
    captured15f = {}

    def _client_for(finish):
        class _Models:
            def generate_content(self, model, contents, config):
                captured15f["config"] = config
                return _t15e.SimpleNamespace(
                    text="### 1. Concept (truncated", usage_metadata=None,
                    candidates=[_t15e.SimpleNamespace(finish_reason=finish)])
        return lambda: _t15e.SimpleNamespace(models=_Models())

    fake_google15f = _t15e.ModuleType("google")
    fake_google15f.genai = fake_genai15f
    saved_mods = {k: sys.modules.get(k) for k in ("google", "google.genai", "google.genai.types")}
    sys.modules.update({"google": fake_google15f, "google.genai": fake_genai15f, "google.genai.types": fake_types15f})
    try:
        fake_genai15f.Client = _client_for("FinishReason.MAX_TOKENS")
        truncated = _overview_generate("p")
        fake_genai15f.Client = _client_for("FinishReason.STOP")
        complete = _overview_generate("p")
    finally:
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert truncated == "" and complete.startswith("### 1."), (truncated, complete)
    assert captured15f["config"]["max_output_tokens"] >= 6000, captured15f
    print("ok 15f — a MAX_TOKENS (truncated) overview is discarded; budget >= 6000 tokens")

    # 16. _phase_embed filters empty OD descriptions before embedding, same
    # as the OCR block right above it — a bug found in code review: a single
    # failed Gemini vision call (describe_chart_frame() returns "" on
    # failure, by design) used to send Voyage a batch containing "", which
    # would fail the WHOLE video's embedding phase instead of degrading
    # just that one frame.
    import types as _types2
    fake_embedder = _types2.ModuleType("pipelines.video_embedder")
    captured_od_call = {}

    def fake_embed_frames_visual(frames, on_batch=None):
        return [[0.1]] * len(frames)

    def fake_embed_texts_voyage(texts, batch_size=96, on_batch=None):
        captured_od_call.setdefault("calls", []).append(list(texts))
        return [[0.2]] * len(texts)

    fake_embedder.embed_frames_visual = fake_embed_frames_visual
    fake_embedder.embed_texts_voyage = fake_embed_texts_voyage
    sys.modules["pipelines.video_embedder"] = fake_embedder
    try:
        emb_result = _phase_embed(
            keyframes=[("f1", 0), ("f2", 1000)],
            chart_flags=[True, True],
            ocr_texts=["SLA here", "LB here"],
            od_descriptions=["a real description", ""],  # second one "failed"
            segments=[],
            video_id="vid16",
        )
    finally:
        del sys.modules["pipelines.video_embedder"]

    # Two embed_texts_voyage calls happen: one for OCR, one for OD. The OD
    # call must not include the empty string.
    od_calls = [c for c in captured_od_call["calls"] if c != ["SLA here", "LB here"]]
    assert od_calls == [["a real description"]], captured_od_call
    assert len(emb_result["od_vectors"]) == 1, emb_result
    print("ok 16 — _phase_embed filters empty OD descriptions before embedding")

    # 16b. P152: the transcript track embeds the same windows _phase_index
    # stores -- one vector per window, never per raw segment, or vectors and
    # payloads would pair up wrongly.
    captured_od_call.clear()
    segs16 = [WSeg(i) for i in range(10)]
    sys.modules["pipelines.video_embedder"] = fake_embedder
    try:
        emb16 = _phase_embed(keyframes=[], chart_flags=[], ocr_texts=[], od_descriptions=[],
                             segments=segs16, video_id="vid16b")
    finally:
        del sys.modules["pipelines.video_embedder"]
    assert captured_od_call["calls"] == [[w.text for w in transcript_windows(segs16)]], captured_od_call
    assert len(emb16["transcript_vectors"]) == len(transcript_windows(segs16))
    print("ok 16b — transcript embedding uses the windows, one vector per window")

    # 17. resume_interrupted forwards title/source_url/platform/duration to
    # ingest_video, same as it already forwards video_id/work_dir/eval_db —
    # a bug found in code review: without this, a resumed video would lose
    # its title/platform/source_url, breaking non-YouTube downloads and
    # Qdrant payload titles. Dormant today (nothing calls resume_interrupted
    # yet) but real once wired up.
    captured_resume_call = {}

    class FakeEvalDBResume:
        async def fetch(self, *a, **kw):
            return [{"phase": "download", "status": "done"}]

    def fake_ingest_video(video_id, work_dir, eval_db=None, title="", source_url="",
                           platform="youtube", duration=0.0, main_loop=None):
        captured_resume_call.update(
            video_id=video_id, work_dir=work_dir, title=title,
            source_url=source_url, platform=platform, duration=duration,
        )
        return IngestResult(video_id=video_id, status="done")

    saved_ingest_video = ingest_video
    ingest_video = fake_ingest_video
    try:
        resume_interrupted(
            "vid17", "/tmp/work", eval_db=FakeEvalDBResume(),
            title="Real Title", source_url="https://example.com/v17",
            platform="tiktok", duration=123.0,
        )
    finally:
        ingest_video = saved_ingest_video

    assert captured_resume_call["title"] == "Real Title", captured_resume_call
    assert captured_resume_call["source_url"] == "https://example.com/v17", captured_resume_call
    assert captured_resume_call["platform"] == "tiktok", captured_resume_call
    assert captured_resume_call["duration"] == 123.0, captured_resume_call
    print("ok 17 — resume_interrupted forwards title/source_url/platform/duration")

    # 18. _run_async(coro, main_loop=...) executes on the GIVEN loop's thread
    # (via run_coroutine_threadsafe) when called from a different thread,
    # instead of spinning up a brand-new event loop. This is the fix for a
    # real production crash (P95): the old behavior always created a fresh
    # event loop in whatever thread called it, but _update_job_status's/
    # _save_checkpoint's eval_db has its asyncpg connection pool bound to
    # the MAIN event loop (created there on first use) -- using that same
    # pool from a second, independently-created event loop is unsupported
    # by asyncpg and produced "another operation is in progress", logged
    # immediately before a container segfault during the first real
    # ingestion attempt.
    import threading as _threading

    main_loop_ready = _threading.Event()
    main_loop_holder = {}

    def _run_main_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        main_loop_holder["loop"] = loop
        main_loop_holder["thread_id"] = _threading.get_ident()
        main_loop_ready.set()
        loop.run_forever()

    main_thread = _threading.Thread(target=_run_main_loop, daemon=True)
    main_thread.start()
    main_loop_ready.wait(timeout=5)
    main_loop = main_loop_holder["loop"]

    async def _record_thread_id():
        return _threading.get_ident()

    # Called from THIS thread (not the main_loop's thread) -- simulates
    # ingest_video() running inside asyncio.to_thread().
    try:
        executed_on = _run_async(_record_thread_id(), main_loop=main_loop)
        assert executed_on == main_loop_holder["thread_id"], (
            f"coroutine ran on thread {executed_on}, expected the main loop's "
            f"thread {main_loop_holder['thread_id']} -- a fresh event loop was "
            "created instead of reusing the given one"
        )
    finally:
        main_loop.call_soon_threadsafe(main_loop.stop)
        main_thread.join(timeout=5)
        # close() only requires the loop to be stopped, not to be called from
        # its own thread -- join() above guarantees that. Without this, the
        # loop's self-pipe fd gets torn down with the thread, and __del__
        # raises "Invalid file descriptor: -1" at GC time -- harmless
        # (doesn't affect the exit code) but noisy.
        main_loop.close()
    print("ok 18 — _run_async(coro, main_loop=...) runs the coroutine on the given loop's thread")

    # 19. P139: _phase_visual must not hold every keyframe at full resolution.
    # Measured 2026-10-19: ~7 MB resident per keyframe and ~50-60 keyframes
    # per minute, i.e. 20+ GB for one hour of video. Kept frames are shrunk
    # for embedding, while OCR and the vision description still see the full
    # frame -- and descriptions come back in keyframe order even though they
    # now run concurrently.
    import tempfile as _tf19
    import cv2 as _cv2
    import numpy as np
    import pipelines.chart_classifier as _cc
    import pipelines.ocr_track as _ocr
    import pipelines.vision_description as _vd

    with _tf19.TemporaryDirectory() as _d19:
        vpath = os.path.join(_d19, "synthetic.mp4")
        writer = _cv2.VideoWriter(vpath, _cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
        for sec in range(12):
            img = np.full((720, 1280, 3), 30 + sec * 18, dtype=np.uint8)
            _cv2.putText(img, "frame %d" % sec, (200, 360), _cv2.FONT_HERSHEY_SIMPLEX, 4,
                         (255, 255, 255), 8)
            for _ in range(10):
                writer.write(img)
        writer.release()

        seen_widths = []

        def _fake_describe(frame):
            seen_widths.append(frame.shape[1])
            time.sleep(0.05)
            return "desc@%d" % int(frame.mean())

        saved19 = (_cc.is_chart_frame, _ocr.extract_labels, _vd.describe_chart_frame)
        _cc.is_chart_frame = lambda f: True
        _ocr.extract_labels = lambda f: [{"text": "w%d" % f.shape[1], "score": 0.9,
                                          "bbox": [0, 0, 1, 1]}]
        _vd.describe_chart_frame = _fake_describe
        progress19 = []
        try:
            vis = _phase_visual(vpath, "synthetic", on_progress=progress19.append)
        finally:
            _cc.is_chart_frame, _ocr.extract_labels, _vd.describe_chart_frame = saved19

    kfs = vis["keyframes"]
    assert kfs, "no keyframes captured from the synthetic video"
    assert all(f.shape[1] <= KEEP_FRAME_MAX_WIDTH for f, _ in kfs), [f.shape for f, _ in kfs]
    assert set(seen_widths) == {1280}, seen_widths            # Gemini saw full resolution
    assert all(t == "w1280" for t in vis["ocr_texts"]), vis["ocr_texts"]  # so did OCR
    assert len(vis["od_descriptions"]) == sum(vis["chart_flags"]) == len(kfs)
    # order preserved: frames get brighter over time, so the descriptions must too
    brightness = [int(d.split("@")[1]) for d in vis["od_descriptions"]]
    assert brightness == sorted(brightness), brightness
    print("ok 19 — keyframes kept small, OCR/vision see full frames, descriptions stay in order")

    # 19b. The visual phase reports how far through the video it is, so the
    # admin panel can show % and time left (it showed nothing for 27 min on
    # the 1:53 checkpoint video, 2026-10-19).
    assert progress19, "no progress reported"
    assert progress19 == sorted(progress19), progress19
    assert 0.0 <= progress19[0] and progress19[-1] >= 0.9 and progress19[-1] <= 1.0, progress19
    print("ok 19b — visual phase reports monotonic progress up to ~100%")

    # 25. OCR precision (the client 2026-10-27: "a wrong number presented as
    # certain is worse than a missing one"). The visual phase keeps every
    # reading's value/confidence/status, marks a misread with the exact
    # "[uncertain]" contract in the flat text, and _phase_index stores the
    # structured list as `ocr_labels` next to the unchanged `ocr_text`.
    def _axis25(f):
        labs = [{"text": "{:,.2f}".format(28460 - 20 * k), "score": 0.999,
                 "bbox": [1200, 40 + 26 * k, 1270, 56 + 26 * k]} for k in range(24)]
        # the real 1080p misread: 28,220.00 read as 20,220.00, at its height
        labs.append({"text": "20,220.00", "score": 0.93, "bbox": [1201, 352, 1270, 368]})
        return labs

    saved25 = (_cc.is_chart_frame, _ocr.extract_labels, _vd.describe_chart_frame)
    _cc.is_chart_frame = lambda f: True
    _ocr.extract_labels = _axis25
    _vd.describe_chart_frame = lambda f: "d"
    try:
        with _tf19.TemporaryDirectory() as _d25:
            vpath25 = os.path.join(_d25, "s.mp4")
            writer = _cv2.VideoWriter(vpath25, _cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
            for sec in range(6):
                img = np.full((720, 1280, 3), 30 + sec * 30, dtype=np.uint8)
                for _ in range(10):
                    writer.write(img)
            writer.release()
            vis25 = _phase_visual(vpath25, "vid25")
    finally:
        _cc.is_chart_frame, _ocr.extract_labels, _vd.describe_chart_frame = saved25
    assert vis25["ocr_texts"] and len(vis25["ocr_labels"]) == len(vis25["ocr_texts"])
    assert all("20,220.00 [uncertain]" in t for t in vis25["ocr_texts"]), vis25["ocr_texts"][0]
    assert all(t.count("[uncertain]") == 1 for t in vis25["ocr_texts"])
    lab25 = {l["text"]: l for l in vis25["ocr_labels"][0]}
    assert lab25["20,220.00"]["status"] == "out_of_range" and lab25["20,220.00"]["value"] == 20220.0
    assert lab25["28,460.00"]["conf"] == 0.999 and lab25["28,460.00"]["bbox"] == [1200, 40, 1270, 56]
    print("ok 25 — visual phase keeps per-reading value/conf/status and marks the misread")

    upserts25 = []
    saved_up25 = (_upsert_to_qdrant, _delete_video_points)
    globals()["_upsert_to_qdrant"] = lambda coll, ids, vecs, payloads, dim: (
        upserts25.append((coll, payloads)) or len(ids))
    globals()["_delete_video_points"] = lambda vid: None
    try:
        _phase_index("vid25", "t", [("f", 0), ("f", 1000), ("f", 2000)], [True, False, True],
                     ["", "A [uncertain]"], [], [], [], [[0.2]], [], [],
                     ocr_labels=[[], [{"text": "A", "value": None, "values": [], "conf": 0.5,
                                       "bbox": [0, 0, 1, 1], "status": "uncertain"}]])
    finally:
        globals()["_upsert_to_qdrant"], globals()["_delete_video_points"] = saved_up25
    ocr_pl = [p for c, ps in upserts25 if c == config.VIDEO_OCR_COLLECTION for p in ps]
    assert len(ocr_pl) == 1 and ocr_pl[0]["frame_ms"] == 2000, ocr_pl
    assert ocr_pl[0]["ocr_text"] == "A [uncertain]"
    assert ocr_pl[0]["ocr_labels"][0]["status"] == "uncertain", ocr_pl[0]
    print("ok 25b — _phase_index stores ocr_labels on the matching OCR point (additive)")

    _self_test_cost_levers(fake_embedder, captured_od_call)

    _self_test_provider_outage(_HR, fake_phase_download, fake_phase_embed, fake_phase_cleanup, saved_fns)
    _self_test_cancel(fake_phase_download, fake_phase_visual, fake_phase_embed, fake_phase_index,
                      saved_fns)
    _vision_preflight = saved_preflight

    # 24. Flex fallback share: warn above 30% (or once the breaker trips),
    # stay silent at or below it and on runs that never used Flex.
    assert flex_fallback_warning({}) == ""
    assert flex_fallback_warning({"flex_calls": 10}) == ""
    assert flex_fallback_warning({"flex_calls": 7, "flex_fallbacks": 3}) == ""      # exactly 30%
    w = flex_fallback_warning({"flex_calls": 6, "flex_fallbacks": 4})
    assert "4 of 10" in w and "40%" in w and "switched off" not in w, w
    w2 = flex_fallback_warning({"flex_calls": 95, "flex_fallbacks": 5, "flex_breaker_tripped": True})
    assert "switched off" in w2, w2
    print("ok 24 — flex fallback warning above 30% or when the breaker trips")

    print("\n22 checks passed (+ 20a-20g gemini_down/overview checks, 22a-22f cost-lever checks, "
          "23a-23i cancel/progress checks)")


def _self_test_cancel(fake_phase_download, fake_phase_visual, fake_phase_embed, fake_phase_index,
                      saved_fns):
    """23a-23i: cooperative cancel (POST /video/cancel) and overall progress.

    A cancel must stop paid work at the next checkpoint, keep the spend that
    was already made on the result, never delete a good index that a forced
    re-ingest has not touched yet, and clear partial points once indexing
    has started."""
    global check_memory_watermark, _phase_download, _phase_asr, _phase_visual
    global _phase_embed, _phase_index, _phase_cleanup, _delete_video_points, _save_checkpoint
    global _update_job_status

    # 23a. overall_percent: weights download 10, asr 15, visual 50, embed 12,
    # index 10, cleanup 3; finished phases count fully, the current one by its
    # phase_progress; only a DONE job reads 100.
    assert sum(PHASE_WEIGHTS.values()) == 100 and tuple(PHASE_WEIGHTS) == PHASE_ORDER, PHASE_WEIGHTS
    cases = [
        (("download", 0.0, "processing"), 0),
        (("asr", 0.5, "processing"), 18),         # 10 + 7.5
        (("visual", 0.5, "processing"), 50),      # 10 + 15 + 25
        (("embed", 0.0, "processing"), 75),
        (("index", 1.0, "processing"), 97),
        (("cleanup", 1.0, "processing"), 99),     # never 100 until done
        (("cleanup", 0.0, "done"), 100),
        ((None, None, "done"), 100),
        ((None, None, "queued"), 0),
        ((None, None, None), 0),
        (("visual", 0.4, "cancelled"), 45),       # where it stopped
        (("visual", 7.0, "processing"), 75),      # progress clamped to 1
        (("bogus", 0.5, "processing"), 0),
    ]
    for args, want in cases:
        got = overall_percent(*args)
        assert got == want and isinstance(got, int), (args, got, want)
    print("ok 23a — overall_percent weights phases 10/15/50/12/10/3, done=100, processing caps at 99")

    class Seg:
        def __init__(self, t):
            self.text, self.start_time, self.end_time, self.confidence, self.words = "talk", t, t + 5, -0.2, []

    deleted, cleaned, checkpoints, statuses = [], [], [], []
    saved_extra = (_delete_video_points, _save_checkpoint, _update_job_status)

    def run(should_stop, asr=None, visual=None, embed=None, index=None, eval_db=None):
        global check_memory_watermark, _phase_download, _phase_asr, _phase_visual
        global _phase_embed, _phase_index, _phase_cleanup, _delete_video_points, _save_checkpoint
        global _update_job_status
        for lst in (deleted, cleaned, checkpoints, statuses):
            lst.clear()
        check_memory_watermark = lambda: True
        _phase_download = lambda video_id, work_dir, platform="youtube", source_url="", duration=0.0: {
            "video_path": "/tmp/c.mp4", "audio_path": "/tmp/c.wav", "duration": 600.0}
        _phase_asr = asr or (lambda audio_path, video_id: [Seg(i * 60.0) for i in range(10)])
        _phase_visual = visual or (lambda video_path, video_id, segments=None, on_progress=None: {
            "keyframes": [("f", 1000)], "chart_flags": [True], "ocr_texts": ["SLA"],
            "od_descriptions": ["desc"], "capture_reasons": ["change"], "sampled_frames": 600})
        _phase_embed = embed or fake_phase_embed
        _phase_index = index or fake_phase_index
        _phase_cleanup = lambda video_path, video_id: cleaned.append(video_path)
        _delete_video_points = lambda video_id: deleted.append(video_id)
        _save_checkpoint = lambda db, vid, phase, status, output_path="", item_count=0, main_loop=None: \
            checkpoints.append((phase, status))
        _update_job_status = lambda db, vid, status, phase="", error_message="", main_loop=None, progress=0.0: \
            statuses.append((status, phase, round(progress, 3)))
        try:
            return ingest_video("c1", "/tmp/wc", eval_db=eval_db, title="T", duration=600.0,
                                should_stop=should_stop)
        finally:
            (check_memory_watermark, _phase_download, _phase_asr,
             _phase_visual, _phase_embed, _phase_index, _phase_cleanup) = saved_fns
            _delete_video_points, _save_checkpoint, _update_job_status = saved_extra

    # 23b. Stop requested before anything ran: no phase is paid for, the
    # result says cancelled during download, nothing in Qdrant is touched.
    r = run(lambda: True)
    assert r.status == "cancelled" and r.phase == "download", (r.status, r.phase)
    assert r.phases_completed == [] and deleted == [], (r.phases_completed, deleted)
    assert r.error == "cancelled by user during download", r.error
    assert r.usage.get("video_id") == "c1", r.usage          # usage summary still filled
    print("ok 23b — a stop before download runs nothing and returns status=cancelled")

    # 23c. Mid-ASR: the real _phase_asr's per-chunk hook raises, the paid
    # chunks before it are kept on the usage summary, chunks after it are never
    # sent, existing points are NOT deleted (a forced re-ingest must keep its
    # old index), and the downloaded video file is still cleaned up.
    import sys as _sys
    import types as _types
    stop = {"now": False}
    sent_chunks = []
    fake_asr_mod = _types.ModuleType("pipelines.asr_track")

    def _fake_transcribe(audio_path, model=None, on_chunk=None):
        for i in range(3):
            if on_chunk is not None:
                on_chunk(i, 3)
            sent_chunks.append(i)
            usage_meter.record_groq(600.0)
            stop["now"] = True          # the user clicks Cancel during chunk 1
        return [Seg(0.0)]

    fake_asr_mod.transcribe_video = _fake_transcribe
    saved_asr_mod = _sys.modules.get("pipelines.asr_track")
    _sys.modules["pipelines.asr_track"] = fake_asr_mod
    real_phase_asr = saved_fns[2]
    try:
        r = run(lambda: stop["now"], asr=real_phase_asr, eval_db=object())
    finally:
        if saved_asr_mod is not None:
            _sys.modules["pipelines.asr_track"] = saved_asr_mod
        else:
            del _sys.modules["pipelines.asr_track"]
    assert r.status == "cancelled" and r.phase == "asr", (r.status, r.phase, r.error)
    assert sent_chunks == [0], sent_chunks
    assert r.usage["groq"]["audio_seconds"] == 600.0, r.usage    # partial spend kept
    assert deleted == [], deleted
    assert cleaned == ["/tmp/c.mp4"], cleaned
    assert r.phases_completed == ["download"], r.phases_completed
    print("ok 23c — cancel mid-ASR stops before the next chunk, keeps partial spend, "
          "deletes no points, still removes the video file")

    # 23d. ASR progress = chunks done / total, written while the phase runs.
    stop["now"] = False
    sent_chunks.clear()

    def _fake_transcribe_all(audio_path, model=None, on_chunk=None):
        for i in range(4):
            on_chunk(i, 4)
        return [Seg(i * 60.0) for i in range(10)]

    fake_asr_mod.transcribe_video = _fake_transcribe_all
    _sys.modules["pipelines.asr_track"] = fake_asr_mod
    global PROGRESS_WRITE_INTERVAL_S
    saved_interval = PROGRESS_WRITE_INTERVAL_S
    PROGRESS_WRITE_INTERVAL_S = 0.0
    try:
        r = run(lambda: False, asr=real_phase_asr, eval_db=object())
    finally:
        PROGRESS_WRITE_INTERVAL_S = saved_interval
        if saved_asr_mod is not None:
            _sys.modules["pipelines.asr_track"] = saved_asr_mod
        else:
            del _sys.modules["pipelines.asr_track"]
    assert r.status == "done", (r.status, r.error)
    asr_progress = [p for s, ph, p in statuses if ph == "asr" and p > 0]
    assert asr_progress == [0.25, 0.5, 0.75], statuses
    print("ok 23d — ASR writes phase_progress = chunks done / total")

    # 23e. Mid-visual: the visual phase swallows its own errors as "degraded"
    # (continue ASR-only) -- a cancel must NOT be swallowed that way, or the
    # run would go on to pay for embedding and re-index.
    stop["now"] = False
    embedded = []

    def _visual_cancelled(video_path, video_id, segments=None, on_progress=None):
        stop["now"] = True
        _stop_point()
        raise AssertionError("_stop_point must raise once a stop is requested")

    r = run(lambda: stop["now"], visual=_visual_cancelled,
            embed=lambda *a, **k: embedded.append(1) or fake_phase_embed(*a, **k))
    assert r.status == "cancelled" and r.phase == "visual", (r.status, r.phase, r.error)
    assert embedded == [] and deleted == [], (embedded, deleted)
    assert ("visual", "failed") not in checkpoints, checkpoints
    print("ok 23e — cancel mid-visual is not swallowed as 'degraded'; no embed, no delete")

    # 23f. Cancel once indexing has started: the half-written points are
    # deleted and the index checkpoint is marked so a later run does not
    # believe a complete index is still there.
    stop["now"] = False

    def _index_cancelled(*a, **k):
        stop["now"] = True
        _stop_point()
        return {"total_points": 4}

    r = run(lambda: stop["now"], index=_index_cancelled)
    assert r.status == "cancelled" and r.phase == "index", (r.status, r.phase)
    assert deleted == ["c1"], deleted
    assert ("index", "cancelled") in checkpoints, checkpoints
    assert cleaned == ["/tmp/c.mp4"], cleaned
    print("ok 23f — cancel during index deletes the partial points and marks the index checkpoint")

    # 23g. A stop requested after indexing finished does not undo a complete
    # index: there is no checkpoint before cleanup, the run ends done.
    stop["now"] = False
    r = run(lambda: stop["now"],
            index=lambda *a, **k: (stop.__setitem__("now", True) or {"total_points": 4}))
    assert r.status == "done" and deleted == [], (r.status, deleted)
    print("ok 23g — a stop after the index phase finished leaves a done run alone")

    # 23h. The real _phase_visual: a cancel stops submitting frames to the
    # vision model, cancels the queued ones (no paid call for them) and does
    # not wait for the backlog -- at most the in-flight calls finish.
    import tempfile as _tf
    import cv2 as _cv2
    import numpy as np
    import pipelines.chart_classifier as _cc
    import pipelines.ocr_track as _ocr
    import pipelines.vision_description as _vd
    with _tf.TemporaryDirectory() as d:
        vpath = os.path.join(d, "cancel.mp4")
        writer = _cv2.VideoWriter(vpath, _cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 240))
        for sec in range(20):
            img = np.full((240, 320, 3), 10 + sec * 12, dtype=np.uint8)
            _cv2.putText(img, "f%d" % sec, (60, 140), _cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            for _ in range(5):
                writer.write(img)
        writer.release()

        described, classified = [], []

        def _slow_describe(frame):
            time.sleep(0.3)
            described.append(1)
            return "d"

        def _classify(frame):
            classified.append(1)
            return True

        saved_v = (_cc.is_chart_frame, _ocr.extract_labels, _vd.describe_chart_frame)
        _cc.is_chart_frame, _ocr.extract_labels, _vd.describe_chart_frame = (
            _classify, (lambda f: []), _slow_describe)
        saved_env = os.environ.get("VIDEO_VISION_WORKERS")
        os.environ["VIDEO_VISION_WORKERS"] = "1"
        t0 = time.monotonic()
        try:
            with _run_control("c1", None, None, lambda: len(classified) >= 4):
                try:
                    _phase_visual(vpath, "c1")
                    raise AssertionError("expected JobCancelled from _phase_visual")
                except JobCancelled:
                    pass
            elapsed = time.monotonic() - t0
            time.sleep(0.7)        # let the one in-flight call finish
        finally:
            _cc.is_chart_frame, _ocr.extract_labels, _vd.describe_chart_frame = saved_v
            if saved_env is None:
                os.environ.pop("VIDEO_VISION_WORKERS", None)
            else:
                os.environ["VIDEO_VISION_WORKERS"] = saved_env
    assert len(classified) == 4, classified           # no frame read after the stop
    assert len(described) < 4, described               # queued descriptions never ran
    assert elapsed < 0.9, elapsed                       # did not wait for the backlog
    print("ok 23h — cancel in the visual pool cancels queued vision calls; only in-flight ones finish")

    # 23i. Embed progress = batches done / total across all four tracks, and
    # a cancel between batches stops the next Voyage request.
    fake_emb = _types.ModuleType("pipelines.video_embedder")
    batches = []

    def _batched(items, batch_size=96, on_batch=None, **kw):
        out = []
        for i in range(0, len(items), batch_size):
            on_batch()
            batches.append(len(items[i:i + batch_size]))
            out.extend([[0.1]] * len(items[i:i + batch_size]))
        return out

    fake_emb.embed_frames_visual = _batched
    fake_emb.embed_texts_voyage = _batched
    saved_emb = _sys.modules.get("pipelines.video_embedder")
    _sys.modules["pipelines.video_embedder"] = fake_emb
    saved_bs = config.VOYAGE_BATCH_SIZE
    config.VOYAGE_BATCH_SIZE = 2
    PROGRESS_WRITE_INTERVAL_S = 0.0
    emb_writes = []
    _update_job_status = lambda db, vid, status, phase="", error_message="", main_loop=None, progress=0.0: \
        emb_writes.append((phase, round(progress, 3)))
    kfs = [("f", 1000 * i) for i in range(3)]
    try:
        with _run_control("c1", object(), None, lambda: False):
            _run_ctl().phase = "embed"
            _phase_embed(kfs, [True] * 3, ["a", "b", "c"], ["x", "y", "z"], [Seg(0.0)], "c1")
        # 3 frames -> 1 batch (embed_frames_visual keeps its own default of
        # 96), 3 OCR -> 2, 3 OD -> 2 (VOYAGE_BATCH_SIZE=2), 1 window -> 1 = 6
        assert [p for ph, p in emb_writes if ph == "embed"] == [round(i / 6, 3) for i in range(6)], emb_writes
        batches.clear()

        def _stop_after_two():
            return len(batches) >= 2

        with _run_control("c1", None, None, _stop_after_two):
            _run_ctl().phase = "embed"
            try:
                _phase_embed(kfs, [True] * 3, ["a", "b", "c"], ["x", "y", "z"], [Seg(0.0)], "c1")
                raise AssertionError("expected JobCancelled from _phase_embed")
            except JobCancelled as exc:
                assert exc.phase == "embed", exc.phase
        assert len(batches) == 2, batches
    finally:
        PROGRESS_WRITE_INTERVAL_S = saved_interval
        config.VOYAGE_BATCH_SIZE = saved_bs
        _update_job_status = saved_extra[2]
        if saved_emb is not None:
            _sys.modules["pipelines.video_embedder"] = saved_emb
        else:
            del _sys.modules["pipelines.video_embedder"]
    print("ok 23i — embed writes phase_progress = batches done / total; a cancel stops the next batch")


def _self_test_cost_levers(fake_embedder, captured_embed_calls):
    """22a-22f: the opt-in ingestion cost levers (2026-10-25). Each is env-
    driven and DEFAULT OFF; 22a pins that with nothing set the visual and
    embed phases return exactly what they returned before the levers."""
    import sys
    import tempfile
    import cv2
    import numpy as np
    import pipelines.chart_classifier as _cc
    import pipelines.ocr_track as _ocr
    import pipelines.vision_description as _vd

    lever_env = ("VIDEO_KEYFRAME_DEDUPE_SSIM", "VIDEO_SKIP_IMAGE_EMBED_WHEN_DESCRIBED",
                 "VIDEO_VISION_WORKERS")

    def with_env(env, fn):
        saved = {k: os.environ.pop(k, None) for k in lever_env}
        os.environ.update(env)
        try:
            return fn()
        finally:
            for k in lever_env:
                os.environ.pop(k, None)
            os.environ.update({k: v for k, v in saved.items() if v is not None})

    # A static chart for 10 s, then a different static chart for 10 s, at
    # 5 fps: two real changes, and everything else burst/safety frames over
    # an unchanged picture -- the near-duplicates the lever exists to drop.
    with tempfile.TemporaryDirectory() as d:
        vpath = os.path.join(d, "static.mp4")
        writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 360))
        a = np.full((360, 640, 3), 30, dtype=np.uint8)
        cv2.rectangle(a, (60, 60), (260, 200), (200, 200, 200), -1)
        b = np.full((360, 640, 3), 220, dtype=np.uint8)
        cv2.rectangle(b, (300, 100), (600, 330), (0, 0, 180), -1)
        for img in (a, b):
            for _ in range(10 * 5):
                writer.write(img)
        writer.release()

        described = []

        def fake_describe(frame):
            described.append(int(frame.mean()))
            return "desc"

        saved_fns = (_cc.is_chart_frame, _ocr.extract_labels, _vd.describe_chart_frame)
        saved_cfg = (config.SAFETY_FRAME_INTERVAL_S, config.BURST_FRAME_COUNT)
        _cc.is_chart_frame = lambda f: True
        _ocr.extract_labels = lambda f: []
        _vd.describe_chart_frame = fake_describe
        config.SAFETY_FRAME_INTERVAL_S, config.BURST_FRAME_COUNT = 3.0, 2
        try:
            base = with_env({}, lambda: _phase_visual(vpath, "lever"))
            base_described = len(described)
            described.clear()
            dedup = with_env({"VIDEO_KEYFRAME_DEDUPE_SSIM": "0.97"},
                             lambda: _phase_visual(vpath, "lever"))
            dedup_described = len(described)
            described.clear()
            bad = with_env({"VIDEO_KEYFRAME_DEDUPE_SSIM": "1.5"},
                           lambda: _phase_visual(vpath, "lever"))
        finally:
            _cc.is_chart_frame, _ocr.extract_labels, _vd.describe_chart_frame = saved_fns
            config.SAFETY_FRAME_INTERVAL_S, config.BURST_FRAME_COUNT = saved_cfg

    # 22a. Off: same keys as before the lever, burst + safety frames all kept.
    # ocr_labels: always present since the OCR precision checks (2026-10-28),
    # independent of any cost lever.
    assert set(base) == {"keyframes", "chart_flags", "ocr_texts", "ocr_labels", "od_descriptions",
                         "capture_reasons", "sampled_frames"}, sorted(base)
    assert base["capture_reasons"].count("change") == 2, base["capture_reasons"]
    assert {"burst", "safety"} <= set(base["capture_reasons"]), base["capture_reasons"]
    assert base_described == len(base["keyframes"]), (base_described, len(base["keyframes"]))
    print("ok 22a — no lever env: visual phase output unchanged, bursts/safety frames kept")

    # 22b. On: only the two real changes survive; every dropped frame is
    # counted (and skipped OCR, Gemini and the image embedding); a real
    # detected change is never dropped, same rule as adaptive_capture.
    assert dedup["capture_reasons"] == ["change", "change"], dedup["capture_reasons"]
    assert dedup_described == 2 and len(dedup["od_descriptions"]) == 2, dedup_described
    assert dedup["deduped_frames"] == len(base["keyframes"]) - 2, dedup["deduped_frames"]
    assert len(dedup["keyframes"]) == len(dedup["chart_flags"]) == 2
    # An out-of-range threshold is ignored (logged), never "drop everything".
    assert len(bad["keyframes"]) == len(base["keyframes"]) and "deduped_frames" not in bad
    print("ok 22b — VIDEO_KEYFRAME_DEDUPE_SSIM drops near-duplicate keyframes, keeps changes")

    # 22c. Vision concurrency: Flex requests can queue for minutes, so 4
    # workers would turn a 2 h video into a day. Env-tunable, default 4.
    assert with_env({}, _vision_workers) == VISION_WORKERS == 4
    assert with_env({"VIDEO_VISION_WORKERS": "16"}, _vision_workers) == 16
    assert with_env({"VIDEO_VISION_WORKERS": "0"}, _vision_workers) == 4
    assert with_env({"VIDEO_VISION_WORKERS": "lots"}, _vision_workers) == 4
    print("ok 22c — VIDEO_VISION_WORKERS overrides vision concurrency, invalid -> 4")

    # 22d. Skip the image embedding for keyframes whose chart description is
    # non-empty (their content is already searchable in od_frames). Off:
    # every keyframe embedded, no new key.
    kfs = [("f-described", 1000), ("f-desc-failed", 2000), ("f-not-chart", 3000)]
    args = dict(keyframes=kfs, chart_flags=[True, True, False], ocr_texts=["", ""],
                od_descriptions=["a load balancer", ""], segments=[], video_id="vid22")
    embedded = []
    real_visual = fake_embedder.embed_frames_visual
    fake_embedder.embed_frames_visual = lambda frames, on_batch=None: (embedded.append(list(frames))
                                                        or [[0.1]] * len(frames))
    sys.modules["pipelines.video_embedder"] = fake_embedder
    try:
        off = with_env({}, lambda: _phase_embed(**args))
        on = with_env({"VIDEO_SKIP_IMAGE_EMBED_WHEN_DESCRIBED": "1"}, lambda: _phase_embed(**args))
    finally:
        del sys.modules["pipelines.video_embedder"]
        fake_embedder.embed_frames_visual = real_visual
    assert "clip_indices" not in off and len(off["clip_vectors"]) == 3, off
    assert embedded[0] == ["f-described", "f-desc-failed", "f-not-chart"], embedded
    # A failed description ("") keeps its image embedding: otherwise that
    # frame would be in no visual collection at all.
    assert embedded[1] == ["f-desc-failed", "f-not-chart"], embedded
    assert on["clip_indices"] == [1, 2] and len(on["clip_vectors"]) == 2, on
    assert len(on["od_vectors"]) == 1, on   # the description itself is still embedded
    print("ok 22d — VIDEO_SKIP_IMAGE_EMBED_WHEN_DESCRIBED skips described frames only")

    # 22e. _phase_index pairs each visual vector with ITS keyframe's
    # timestamp via clip_indices -- positional pairing would stamp the
    # 2000 ms vector with 1000 ms and cite the wrong moment.
    global _upsert_to_qdrant, _delete_video_points
    saved_up, saved_del = _upsert_to_qdrant, _delete_video_points
    captured = {}
    _upsert_to_qdrant = lambda coll, ids, vecs, payloads, dim: (
        captured.__setitem__(coll, (ids, payloads)) or len(ids))
    _delete_video_points = lambda vid: None
    try:
        _phase_index("vid22", "t", kfs, [True, True, False], ["", ""], ["a load balancer", ""], [],
                     on["clip_vectors"], [], on["od_vectors"], [], clip_indices=on["clip_indices"])
        vis_on = captured.pop(config.VIDEO_VISUAL_COLLECTION)
        _phase_index("vid22", "t", kfs, [True, True, False], ["", ""], ["a load balancer", ""], [],
                     off["clip_vectors"], [], off["od_vectors"], [])
        vis_off = captured.pop(config.VIDEO_VISUAL_COLLECTION)
    finally:
        _upsert_to_qdrant, _delete_video_points = saved_up, saved_del
    assert [p["frame_ms"] for p in vis_on[1]] == [2000, 3000], vis_on
    assert [p["is_chart"] for p in vis_on[1]] == [True, False], vis_on
    assert vis_on[0] == ["vid22:visual:0", "vid22:visual:1"], vis_on
    assert [p["frame_ms"] for p in vis_off[1]] == [1000, 2000, 3000], vis_off
    print("ok 22e — visual payloads pair with their own keyframe when frames are skipped")

    # 22f. The embedding-completeness gate expects only the frames that were
    # meant to be image-embedded; otherwise every lever run reads as
    # "incomplete" and real drops become invisible in the noise.
    class _S:
        text = "words"
    assert _expected_embed_counts(kfs, ["x"], ["d", ""], [_S()], off) == \
        {"clip": 3, "ocr": 1, "od": 1, "asr": 1}
    assert _expected_embed_counts(kfs, ["x"], ["d", ""], [_S()], on)["clip"] == 2
    print("ok 22f — embedding completeness counts only frames meant for image embedding")


def _self_test_provider_outage(_HR, fake_phase_download, fake_phase_embed, fake_phase_cleanup, saved_fns):
    """gemini_down.md items B, C and E: an unavailable vision provider is refused
    up front, a run whose chart descriptions all failed is FAILED, partial
    descriptions and a missing overview are visible warnings, and overview minute
    marks come only from real passage times."""
    global check_memory_watermark, _phase_download, _phase_asr, _phase_visual
    global _phase_embed, _phase_index, _phase_cleanup, _vision_preflight

    class Seg:
        def __init__(self, t):
            self.text, self.start_time, self.end_time, self.confidence, self.words = "talk", t, t + 5, -0.2, []

    calls = []

    def run(preflight, od_descriptions, overview_written=True):
        global check_memory_watermark, _phase_download, _phase_asr, _phase_visual
        global _phase_embed, _phase_index, _phase_cleanup, _vision_preflight
        calls.clear()
        n = len(od_descriptions)

        def _download(*a, **kw):
            calls.append("download")
            return fake_phase_download("v20", "/tmp/w20", duration=600.0)

        def _visual(video_path, video_id, segments=None, on_progress=None):
            return {"keyframes": [("f", 1000 * i) for i in range(n)], "chart_flags": [True] * n,
                    "ocr_texts": ["SLA"] * n, "od_descriptions": list(od_descriptions),
                    "capture_reasons": ["change"] * n, "sampled_frames": 600}

        def _index(*a, **kw):
            return {"total_points": 4, "overview_written": overview_written}

        saved_pre = _vision_preflight
        check_memory_watermark = lambda: True
        _vision_preflight = lambda: preflight
        _phase_download, _phase_asr, _phase_visual = _download, (lambda a, v: [Seg(i * 60.0) for i in range(10)]), _visual
        _phase_embed, _phase_index, _phase_cleanup = fake_phase_embed, _index, fake_phase_cleanup
        try:
            return ingest_video("v20", "/tmp/w20", title="T", duration=600.0)
        finally:
            _vision_preflight = saved_pre
            (check_memory_watermark, _phase_download, _phase_asr,
             _phase_visual, _phase_embed, _phase_index, _phase_cleanup) = saved_fns

    up = _HR("gemini:test", "ok", 200, 5, "answered")
    full = ["a chart description"] * 5

    # 20a. Empty credit: refused before the download -- nothing paid for, job
    # failed with a prefix the queue worker stops on.
    r = run(_HR("gemini:gemini-3.6-flash", "billing", 402, 90, "Your prepayment credits are depleted"), full)
    assert r.status == "failed" and r.error.startswith(VISION_UNAVAILABLE_PREFIX), r
    assert "(billing)" in r.error and calls == [], (r.error, calls)
    for cat in ("auth", "model_unavailable", "not_configured"):
        assert run(_HR("gemini:x", cat, 0, 1, "d"), full).error.startswith(VISION_UNAVAILABLE_PREFIX), cat
    print("ok 20a — persistent vision-provider failure refuses the video before any paid phase")

    # 20b. A transient preflight failure proceeds, visibly.
    r = run(_HR("gemini:x", "network", 0, 15000, "timed out"), full)
    assert r.status == "done" and calls == ["download"], (r, calls)
    assert any("vision provider check before start: network" in w for w in r.stats["warnings"]), r.stats
    print("ok 20b — transient preflight failure proceeds with a warning")

    # 20c. Every chart description blank (the provider died mid-run) -> FAILED,
    # with the counts in the error; not a silent "done".
    r = run(up, [""] * 5)
    assert r.status == "failed" and "0 of 5 chart keyframes" in (r.error or ""), r
    assert r.stats["chart_descriptions"] == "0/5" and "index" in r.phases_completed, r
    print("ok 20c — all chart descriptions failed -> job failed (transcript indexed, retry re-indexes)")

    # 20d. Some descriptions blank -> done + warning; overview missing -> done + warning.
    r = run(up, ["d", "", "d", "d", ""])
    assert r.status == "done" and r.stats["chart_descriptions"] == "3/5", r
    assert any("2 of 5 chart keyframes have no chart description" in w for w in r.stats["warnings"]), r.stats
    r = run(up, full, overview_written=False)
    assert r.status == "done" and any("no video overview was written" in w for w in r.stats["warnings"]), r.stats
    assert r.stats["overview_written"] is False
    print("ok 20d — partial descriptions and a missing overview are visible warnings")

    # 20e. Recovery: provider back, same video -> done, no warnings.
    r = run(up, full)
    assert r.status == "done" and r.stats["warnings"] == [] and r.stats["chart_descriptions"] == "5/5", r.stats
    print("ok 20e — provider back: the same video ingests cleanly")

    # 20f. Overview minute marks come from the passages, never from the model.
    wins = [TranscriptWindow("why the sla forms", 4075.0, 4133.0, -0.1),
            TranscriptWindow("intro", 5.0, 60.0, -0.1),
            TranscriptWindow("broken timing", float("nan"), 10.0, -0.1),
            TranscriptWindow("negative", -3.0, 10.0, -0.1)]
    prompt = _overview_prompt("Episode 12", wins)
    assert "[#1] intro" in prompt and "[#2] why the sla forms" in prompt, prompt
    assert "broken timing" not in prompt and "negative" not in prompt, prompt
    assert not _MODEL_TIMESTAMP.search(prompt), "no clock time may reach the model: %r" % prompt
    numbered = _overview_windows(wins)
    out = _ground_overview_timestamps(
        "### SLA\nWhy it forms [#2].\n### Intro\nOpening [#1], also [1:23:45] and (9:99) and [#7].", numbered)
    assert "Why it forms [1:07:55]." in out and "Opening [0:00:05]" in out, out
    assert "1:23:45" not in out and "[#7]" not in out and "[#" not in out, out
    assert "(9:99)" in out, "only real clock shapes are stripped"   # 9:99 is not a time
    assert _ground_overview_timestamps("No marks at all.", numbered) == "No marks at all."
    print("ok 20f — overview minute marks are the cited passages' real start times; "
          "invented times and dangling refs removed")

    # 20g. Up to two marks per concept (the client 3:29 follow-up, 2026-10-26): one
    # mark pinned "deployment urgency" to 1:15:38 (first mention) while the client looked
    # for the full explanation at 1:33:51. The prompt now asks for the first
    # passage AND the fullest one; the grounding keeps at most two marks per
    # run, drops a repeated one ("[0:07:18] [0:07:18]" seen live) and keeps time order.
    assert "two" in OVERVIEW_SYSTEM_INSTRUCTION and "most fully" in OVERVIEW_SYSTEM_INSTRUCTION, \
        OVERVIEW_SYSTEM_INSTRUCTION
    wins = [TranscriptWindow("a", 10.0, 60.0, -0.1), TranscriptWindow("b", 100.0, 160.0, -0.1),
            TranscriptWindow("c", 200.0, 260.0, -0.1), TranscriptWindow("d", 300.0, 360.0, -0.1)]
    numbered = _overview_windows(wins)
    out = _ground_overview_timestamps("Urgency [#4] [#2].\nRepeat [#3] [#3].\nMany [#1] [#2] [#3].\nPair [#1], [#4].", numbered)
    assert "Urgency [0:01:40] [0:05:00]." in out, out          # time order, both kept
    assert "Repeat [0:03:20]." in out, out                        # duplicate dropped
    assert "Many [0:00:10] [0:01:40]." in out, out                # capped at two
    assert "Pair [0:00:10] [0:05:00]." in out, out                # ", " separated pair is one run
    print("ok 20g — up to two grounded marks per concept, deduplicated, in time order")


if __name__ == "__main__":
    _self_test()
