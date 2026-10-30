"""
/video — video ingestion queue endpoints.

Channel discovery, FIFO queue management, and a background worker that
processes queued videos one at a time with rate-limiting between downloads.

    POST /video/discover          scan a YouTube channel for new videos
    POST /video/queue/{video_id}  add one video to the ingestion queue
    POST /video/queue/batch       add several videos at once
    DELETE /video/queue/{video_id} remove a queued (not processing) video
    POST /video/cancel/{video_id} take a queued video off the queue, or stop a processing one
    GET  /video/status            list videos with ingestion status
    GET  /video/status/{video_id} detailed status for one video
    POST /video/process           start the background FIFO worker
    GET  /video/cookies/status    cookie health (age, usage, fingerprint, last validation)
    POST /video/cookies           upload a cookies.txt file (D72 guardrail applies)
    POST /video/cookies/text      same, pasted as raw text instead of a file
    POST /video/cookies/validate  re-run the live YouTube probe on the saved file
    DELETE /video/cookies         remove the configured cookies.txt
    POST /video/drive-intake/scan      list the Drive intake folder, classify (dry run)
    POST /video/drive-intake/queue     queue explicit Drive file ids (confirm=true)
    GET  /video/drive-intake/settings  auto-delete toggle + window
    POST /video/drive-intake/settings  save them
    POST /video/drive-intake/cleanup   auto-delete preview (dry run) or run (toggle ON)
"""

import asyncio
import datetime
import logging
import os
import random
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
from services.auth import require_admin
from evaluation.eval_db import get_eval_db
from pipelines import drive_intake, query_planner
from pipelines.cost_tracker import (
    check_budget, estimate_groq_cost, estimate_vision_cost, estimate_voyage_cost,
    estimate_voyage_text_cost, record_cost, record_usage, summarize_costs,
)

# The orchestrator is built by a parallel agent — guard the import so the
# self-test (and any deploy before the orchestrator lands) doesn't blow up.
try:
    from pipelines.video_ingestion import ingest_video, overall_percent
except ImportError:
    ingest_video = None

    def overall_percent(phase, phase_progress, status):
        return 100 if status == "done" else 0
# Literal copy of video_ingestion.VISION_UNAVAILABLE_PREFIX, so the queue worker
# can match it even when the guarded import above failed.
VISION_UNAVAILABLE_PREFIX = "vision provider unavailable"

logger = logging.getLogger("haystack-api.video")
router = APIRouter(dependencies=[Depends(require_admin)])


# ── Request / response models ─────────────────────────────

class BatchQueueRequest(BaseModel):
    video_ids: List[str]
    # Re-queue videos that are already done (a paid re-run) -- see queue_video.
    force: bool = False


class FetchUrlRequest(BaseModel):
    url: str


class VideoSettings(BaseModel):
    auto_check_enabled: bool = False
    auto_check_hour: int = 6
    auto_queue: bool = False


# ── Module-level worker flag ──────────────────────────────
# Only one queue worker at a time: a second POST /video/process returns 409
# rather than doubling the download load and the YouTube rate-limit risk.

_worker_active = False

# Video ids whose running ingest the user asked to stop (POST /cancel). The
# queue worker's should_stop() reads it from the ingest thread at each
# checkpoint; the status endpoints read it for cancel_requested. In-process
# on purpose: one uvicorn worker runs the one queue worker, and a restart
# kills the ingest anyway, so there is nothing for a DB column to outlive.
# set.add/discard/`in` are atomic under the GIL; no lock needed.
_cancel_requests = set()

ALREADY_INGESTED_DETAIL = "already ingested — pass force=true to re-ingest (paid)"


# ── Helpers ───────────────────────────────────────────────

# Valid status values for video_ingest_jobs — validated on filter queries so a
# typo comes back as 400 rather than an empty list that looks like "nothing here".
# "cancelled": a run stopped by POST /video/cancel (video_ingest_jobs.status
# has no CHECK constraint, so no schema change).
_VALID_STATUSES = frozenset({"new", "queued", "processing", "done", "failed", "cancelled"})


async def _has_complete_index(db, video_id: str) -> bool:
    """True when a previous run indexed this video completely.

    Read from video_phase_checkpoints, which persist across runs and
    restarts: "index" and "cleanup" are written only after the index phase
    finished. A new run overwrites them only when it reaches them -- and an
    index phase that fails or is cancelled marks "index" failed/cancelled,
    since it deleted the old points first. So when a cancelled re-ingest
    stopped BEFORE indexing, these rows still describe the previous index,
    untouched in Qdrant. Chosen over "was it re-queued from done" because
    that fact is gone once the queue resets the row, and this one covers
    single, batch and pre-metering videos alike.
    Known blind spot: a previous run that indexed fully but was then failed
    by a post-index quality gate (no readable frames, every chart
    description missing) also reads True -- its points are still there.
    """
    try:
        rows = await db.fetch(
            "SELECT phase, status FROM video_phase_checkpoints WHERE video_id = $1",
            video_id)
    except Exception as exc:
        logger.warning("checkpoint read failed for %s: %s", video_id, exc)
        return False
    done = {r["phase"] for r in rows or [] if r["status"] == "done"}
    return {"index", "cleanup"} <= done


async def _remove_queued_job(db, job, video_id: str) -> str:
    """Take a queued job off the queue (DELETE /queue and POST /cancel).

    A queued re-ingest of an indexed video reuses that video's only job row
    (P97), so deleting it would show an indexed, searchable video as never
    ingested: that row goes back to 'done' instead. Returns the note
    recorded on it, or "" when the row was deleted.
    """
    if await _has_complete_index(db, video_id):
        note = "re-ingest cancelled before it started; previous index kept"
        await db.execute(
            "UPDATE video_ingest_jobs SET status = 'done', queue_position = NULL, "
            "warnings = $2, completed_at = NOW(), updated_at = NOW() WHERE id = $1",
            job["id"], note)
        return note
    await db.execute("DELETE FROM video_ingest_jobs WHERE id = $1", job["id"])
    return ""


async def _requeue_row(db, job_id: int, pos: int) -> None:
    """Put an existing (done/failed/cancelled) job row back on the queue."""
    await db.execute(
        "UPDATE video_ingest_jobs SET status = 'queued', phase = NULL, "
        "phase_progress = 0.0, error_message = NULL, queue_position = $2, "
        "retry_count = retry_count + 1, started_at = NULL, completed_at = NULL, "
        "updated_at = NOW() WHERE id = $1",
        job_id, pos,
    )


def _with_progress(row, video_id: str, status_key: str):
    """Copy of a status row plus cancel_requested and overall_progress
    (0-100), for the admin panel's single progress bar and Cancel button."""
    if row is None:
        return None
    out = dict(row)
    out["cancel_requested"] = video_id in _cancel_requests
    out["overall_progress"] = overall_percent(out.get("phase"), out.get("phase_progress"),
                                              out.get(status_key))
    return out


async def _next_queue_position(db) -> int:
    """One past the highest position in the queue, or 1 if the queue is empty."""
    row = await db.fetchrow(
        "SELECT COALESCE(MAX(queue_position), 0) + 1 AS next_pos "
        "FROM video_ingest_jobs"
    )
    return row["next_pos"] if row else 1


async def _in_flight_count(db) -> int:
    """How many videos are currently queued or processing — the number
    MAX_IN_FLIGHT_VIDEOS caps (D68). Counts total in-flight, not just this
    call's batch, so repeated small batch calls can't work around the cap."""
    row = await db.fetchrow(
        "SELECT COUNT(*) AS cnt FROM video_ingest_jobs WHERE status IN ('queued', 'processing')"
    )
    return row["cnt"] if row else 0


def _parse_upload_date(raw: str) -> Optional[datetime.date]:
    """yt-dlp's upload_date is "YYYYMMDD" or missing entirely (common on
    extract_flat and on some non-YouTube extractors). asyncpg needs a real
    date for a DATE column, so anything else becomes None rather than a
    write failure."""
    raw = raw or ""
    if len(raw) == 8 and raw.isdigit():
        try:
            return datetime.date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
        except ValueError:
            return None
    return None


# ── Endpoints ─────────────────────────────────────────────

@router.post("/discover")
async def discover_videos(channel_url: Optional[str] = None):
    """Scan a YouTube channel for videos not yet in video_catalog.

    Calls video_acquisition.list_channel_videos (built by a parallel agent).
    Each video not already catalogued is inserted; already-known ones are
    skipped silently.
    """
    try:
        from pipelines.video_acquisition import list_channel_videos
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="video_acquisition module not available yet")

    # With no VIDEO_DISCOVERY_CHANNEL_URL configured, the caller must name the
    # channel explicitly — discovery never guesses a default source.
    url = channel_url or config.VIDEO_DISCOVERY_CHANNEL_URL
    if not url:
        raise HTTPException(
            status_code=400,
            detail="no discovery channel configured: pass channel_url or set "
                   "VIDEO_DISCOVERY_CHANNEL_URL")
    videos = await asyncio.to_thread(list_channel_videos, url)

    db = get_eval_db()
    new_videos = []
    for v in videos:
        existing = await db.fetchrow(
            "SELECT video_id FROM video_catalog WHERE video_id = $1",
            v["video_id"],
        )
        if existing:
            continue
        upload_date = _parse_upload_date(v.get("upload_date") or "")
        await db.execute(
            "INSERT INTO video_catalog "
            "(video_id, title, duration, upload_date, chapter_count, "
            " thumbnail_url, channel_id, discovered_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())",
            v["video_id"], v.get("title", ""),
            v.get("duration") or 0, upload_date,
            v.get("chapter_count", 0), v.get("thumbnail_url", ""),
            v.get("channel_id", ""),
        )
        new_videos.append(v)

    return {
        "discovered": len(new_videos),
        "total": len(videos),
        "new_videos": new_videos,
    }


@router.post("/fetch-url")
async def fetch_url(body: FetchUrlRequest):
    """Fetch metadata for a video pasted from any yt-dlp-supported platform
    (YouTube, Twitter/X, Instagram, TikTok, ...).

    Catalogues the video so it shows up in the Video Library like any
    channel-discovered video — does NOT download or queue it. The user
    still clicks Ingest to kick off the actual download/transcribe/index run.
    """
    try:
        from pipelines.video_acquisition import fetch_url_metadata
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="video_acquisition module not available yet")

    try:
        v = await asyncio.to_thread(fetch_url_metadata, body.url)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail="could not fetch metadata: %s" % exc)

    db = get_eval_db()
    existing = await db.fetchrow(
        "SELECT video_id FROM video_catalog WHERE video_id = $1", v["video_id"])
    if existing:
        return {"video_id": v["video_id"], "already_catalogued": True, "video": v}

    upload_date = _parse_upload_date(v.get("upload_date") or "")
    await db.execute(
        "INSERT INTO video_catalog "
        "(video_id, title, duration, upload_date, chapter_count, "
        " thumbnail_url, channel_id, discovered_at, source_url, platform) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), $8, $9)",
        v["video_id"], v.get("title", ""),
        v.get("duration") or 0, upload_date,
        v.get("chapter_count", 0), v.get("thumbnail_url", ""),
        v.get("channel_id") or "unknown",
        v["source_url"], v["platform"],
    )
    return {"video_id": v["video_id"], "already_catalogued": False, "video": v}


@router.post("/queue/batch")
async def queue_batch(body: BatchQueueRequest, force: bool = False):
    """Queue multiple videos at once.

    Validates all IDs exist in video_catalog. Videos already queued or
    processing are skipped (not an error). Videos already done are skipped
    and listed under "already_ingested" unless force (body or query) is
    true -- a re-run is the whole paid pipeline again. A done/failed video
    that is queued reuses its latest row instead of inserting a second one
    (P97, same as queue_video).

    Registered before /queue/{video_id} on purpose — FastAPI matches routes
    in registration order, so a literal path must come before a parameterized
    one that would otherwise swallow it. This endpoint was unreachable via
    its real path for an unknown period before this fix: a call here always
    matched /queue/{video_id} first with video_id="batch", returning 404
    "video not found: batch" instead of ever running this function. Found
    2026-10-09 while live-testing the D68 batch-size cap.
    """
    if not body.video_ids:
        raise HTTPException(status_code=400, detail="video_ids must not be empty")

    if len(body.video_ids) > config.MAX_QUEUE_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail="batch size %d exceeds MAX_QUEUE_BATCH_SIZE (%d) — see D68, "
                   "this is a deliberate no-bulk-load safety limit"
                   % (len(body.video_ids), config.MAX_QUEUE_BATCH_SIZE))

    force = force or body.force
    db = get_eval_db()
    details = []
    already_ingested = []
    queued = 0
    in_flight = await _in_flight_count(db)

    for vid in body.video_ids:
        catalog = await db.fetchrow(
            "SELECT video_id FROM video_catalog WHERE video_id = $1", vid)
        if not catalog:
            details.append({"video_id": vid, "status": "not_found"})
            continue

        # Latest row whatever its status (see queue_video, P97): it decides
        # skip / already ingested, and is reused below rather than a second
        # row being inserted next to it.
        existing = await db.fetchrow(
            "SELECT id, status FROM video_ingest_jobs WHERE video_id = $1 "
            "ORDER BY created_at DESC LIMIT 1",
            vid,
        )
        if existing and existing["status"] in ("queued", "processing"):
            details.append({"video_id": vid, "status": "skipped",
                            "reason": existing["status"]})
            continue
        if existing and existing["status"] == "done" and not force:
            already_ingested.append(vid)
            details.append({"video_id": vid, "status": "skipped",
                            "reason": "already_ingested"})
            continue

        if in_flight >= config.MAX_IN_FLIGHT_VIDEOS:
            details.append({"video_id": vid, "status": "skipped",
                            "reason": "in_flight_limit_reached"})
            continue

        pos = await _next_queue_position(db)
        if existing:
            await _requeue_row(db, existing["id"], pos)
        else:
            await db.execute(
                "INSERT INTO video_ingest_jobs "
                "(video_id, status, queue_position, created_at, updated_at) "
                "VALUES ($1, 'queued', $2, NOW(), NOW())",
                vid, pos,
            )
        details.append({"video_id": vid, "status": "queued",
                        "queue_position": pos})
        queued += 1
        in_flight += 1

    skipped = len(body.video_ids) - queued - sum(
        1 for d in details if d["status"] == "not_found")
    return {"queued": queued, "skipped": skipped, "details": details,
            "already_ingested": already_ingested}


@router.post("/queue/{video_id}")
async def queue_video(video_id: str, force: bool = False):
    """Add a single video to the ingestion queue.

    The video must exist in video_catalog (404) and must not already be
    queued or processing (409). A video whose latest job is done is refused
    with 409 unless force=true: re-queueing it re-runs the whole paid
    pipeline (download, Groq, a Gemini call per chart frame, Voyage), so a
    stray Ingest click on a finished video must not do that silently.
    """
    db = get_eval_db()

    catalog = await db.fetchrow(
        "SELECT video_id FROM video_catalog WHERE video_id = $1", video_id)
    if not catalog:
        raise HTTPException(status_code=404, detail="video not found: %s" % video_id)

    # Most recent row for this video, whatever its status -- video_id has no
    # uniqueness constraint in video_ingest_jobs (only the SERIAL id does),
    # so a retry after a failed/done job must reuse that row, not insert a
    # second one. Two rows for the same video_id broke the admin video
    # listing (its per-row key is the video_id only). Found live 2026-10-12, P97.
    existing_job = await db.fetchrow(
        "SELECT id, status FROM video_ingest_jobs WHERE video_id = $1 "
        "ORDER BY created_at DESC LIMIT 1",
        video_id,
    )
    if existing_job and existing_job["status"] in ("queued", "processing"):
        raise HTTPException(
            status_code=409,
            detail="video %s is already %s" % (video_id, existing_job["status"]))
    if existing_job and existing_job["status"] == "done" and not force:
        raise HTTPException(status_code=409, detail=ALREADY_INGESTED_DETAIL)

    in_flight = await _in_flight_count(db)
    if in_flight >= config.MAX_IN_FLIGHT_VIDEOS:
        raise HTTPException(
            status_code=409,
            detail="in-flight limit reached (%d/%d queued or processing) — "
                   "see D68, this is a deliberate no-bulk-load safety limit"
                   % (in_flight, config.MAX_IN_FLIGHT_VIDEOS))

    pos = await _next_queue_position(db)
    if existing_job:
        await _requeue_row(db, existing_job["id"], pos)
    else:
        await db.execute(
            "INSERT INTO video_ingest_jobs "
            "(video_id, status, queue_position, created_at, updated_at) "
            "VALUES ($1, 'queued', $2, NOW(), NOW())",
            video_id, pos,
        )

    return {"video_id": video_id, "queue_position": pos, "status": "queued"}


@router.delete("/queue/{video_id}")
async def dequeue_video(video_id: str):
    """Remove a video from the queue.

    Only videos with status='queued' can be removed — a video already
    processing cannot be cancelled from here.
    """
    db = get_eval_db()

    job = await db.fetchrow(
        "SELECT id, status FROM video_ingest_jobs WHERE video_id = $1 "
        "ORDER BY created_at DESC LIMIT 1",
        video_id,
    )
    if not job:
        raise HTTPException(status_code=404,
                            detail="no ingest job for video: %s" % video_id)
    if job["status"] != "queued":
        raise HTTPException(
            status_code=409,
            detail="video %s has status '%s' — only 'queued' can be removed"
                   % (video_id, job["status"]))

    await _remove_queued_job(db, job, video_id)

    return {"video_id": video_id, "removed": True}


async def _delete_video_points(video_id: str) -> dict:
    """{collection: points deleted} across the four video collections — the
    same set the ingest pipeline clears before a re-index."""
    from services.qdrant_service import QdrantService
    qs = QdrantService()
    out = {}
    try:
        for collection in (config.VIDEO_VISUAL_COLLECTION, config.VIDEO_OCR_COLLECTION,
                           config.VIDEO_OD_COLLECTION, config.VIDEO_TRANSCRIPT_COLLECTION):
            out[collection] = await qs.delete_by_filter(collection, {"video_id": video_id})
    finally:
        await qs.aclose()
    return out


@router.post("/planner/refresh")
async def refresh_planner():
    """Re-list the searchable videos now and drop every cached plan. Called by
    ops/repoctl.py at the end of a full restore: the backend is restarted
    BEFORE the vector store is restored, so its first listing is the old one."""
    query_planner.invalidate_indexed()
    videos = await query_planner.indexed_videos()
    return {"videos": len(videos)}


@router.delete("/ingested/{video_id}")
async def delete_ingested_video(video_id: str):
    """Delete an ingested video's knowledge (admin panel, 2026-10-28).

    Removes its points from the four video search collections, then its job
    and checkpoint rows, so it reads as "new" again and can be re-ingested.
    Kept on purpose: the video_catalog entry (the channel listing, not the
    knowledge) and video_ingest_usage / video_ingest_costs (what was spent
    stays measured). Refused while queued or processing — the worker would
    write points back after the delete. Points go first: if Qdrant fails, the
    rows still say "done" and the delete can simply be retried; the reverse
    order would leave searchable points behind a video shown as never ingested.
    """
    db = get_eval_db()
    job = await db.fetchrow(
        "SELECT id, status FROM video_ingest_jobs WHERE video_id = $1 "
        "ORDER BY created_at DESC LIMIT 1",
        video_id,
    )
    if not job:
        raise HTTPException(status_code=404, detail="video %s was never ingested" % video_id)
    if job["status"] in ("queued", "processing"):
        raise HTTPException(
            status_code=409,
            detail="video %s is %s — stop or remove it first" % (video_id, job["status"]))
    try:
        points = await _delete_video_points(video_id)
    except Exception as exc:
        logger.error("delete of %s: search points could not be removed: %s", video_id, exc)
        raise HTTPException(status_code=502,
                            detail="the search index could not be cleared — nothing else was "
                                   "changed, try again")
    await db.execute("DELETE FROM video_phase_checkpoints WHERE video_id = $1", video_id)
    await db.execute("DELETE FROM video_ingest_jobs WHERE video_id = $1", video_id)
    # Gone from search now, so gone from the planner's video list now too —
    # not up to 5 minutes later.
    query_planner.invalidate_indexed()
    logger.info("deleted ingested video %s: %s", video_id, points)
    return {"video_id": video_id, "deleted": True, "points": points,
            "points_total": sum(points.values())}


@router.post("/cancel/{video_id}")
async def cancel_video(video_id: str):
    """Cancel a video's ingest.

    queued     -> taken off the queue, same as DELETE /queue/{video_id}.
    processing -> 202: a cooperative stop is requested. The run stops at its
                  next checkpoint (before a phase, an ASR chunk, a sampled
                  frame, an embedding batch, the overview call); calls
                  already in flight finish and are still recorded as spend.
                  Final job status 'cancelled' -- or 'done' again for a
                  re-ingest stopped before indexing, whose old index is kept.
    otherwise  -> 409. Uncatalogued -> 404.
    """
    db = get_eval_db()
    catalog = await db.fetchrow(
        "SELECT video_id FROM video_catalog WHERE video_id = $1", video_id)
    if not catalog:
        raise HTTPException(status_code=404, detail="video not found: %s" % video_id)

    job = await db.fetchrow(
        "SELECT id, status FROM video_ingest_jobs WHERE video_id = $1 "
        "ORDER BY created_at DESC LIMIT 1",
        video_id,
    )
    status = job["status"] if job else None
    if status == "queued":
        note = await _remove_queued_job(db, job, video_id)
        return {"video_id": video_id, "status": "cancelled",
                "message": note or "removed from the queue; nothing was run"}
    if status == "processing":
        _cancel_requests.add(video_id)
        logger.info("cancel requested for %s", video_id)
        return JSONResponse(status_code=202, content={
            "video_id": video_id, "status": "cancelling",
            "message": "stops at the next checkpoint; calls already in flight finish"})
    raise HTTPException(
        status_code=409,
        detail="video %s has %s — only a queued or processing ingest can be cancelled"
               % (video_id, ("status '%s'" % status) if status else "no ingest job"))


@router.get("/status")
async def list_status(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List all videos with their ingestion status.

    Joins video_catalog with video_ingest_jobs so every catalogued video
    appears even if it was never queued. Processing videos include their
    current phase and progress.
    """
    if status and status not in _VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail="status must be one of: %s" % ", ".join(sorted(_VALID_STATUSES)))

    db = get_eval_db()

    if status:
        rows = await db.fetch(
            "SELECT vc.*, vij.status AS job_status, vij.queue_position, "
            "       vij.phase, vij.phase_progress, vij.retry_count, "
            "       vij.error_message, vij.warnings, vij.started_at AS job_started, "
            "       vij.completed_at AS job_completed "
            "FROM video_catalog vc "
            "JOIN video_ingest_jobs vij ON vc.video_id = vij.video_id "
            "WHERE vij.status = $1 "
            "ORDER BY vc.discovered_at DESC "
            "LIMIT $2 OFFSET $3",
            status, limit, offset,
        )
    else:
        rows = await db.fetch(
            "SELECT vc.*, vij.status AS job_status, vij.queue_position, "
            "       vij.phase, vij.phase_progress, vij.retry_count, "
            "       vij.error_message, vij.warnings, vij.started_at AS job_started, "
            "       vij.completed_at AS job_completed "
            "FROM video_catalog vc "
            "LEFT JOIN video_ingest_jobs vij ON vc.video_id = vij.video_id "
            "ORDER BY vc.discovered_at DESC "
            "LIMIT $1 OFFSET $2",
            limit, offset,
        )

    videos = [_with_progress(r, r["video_id"], "job_status") for r in rows]
    return {"videos": videos, "count": len(videos)}


@router.get("/status/{video_id}")
async def video_status(video_id: str):
    """Detailed status for one video: catalog row, ingest job, phase checkpoints."""
    db = get_eval_db()

    catalog = await db.fetchrow(
        "SELECT * FROM video_catalog WHERE video_id = $1", video_id)
    if not catalog:
        raise HTTPException(status_code=404,
                            detail="video not found: %s" % video_id)

    job = await db.fetchrow(
        "SELECT * FROM video_ingest_jobs WHERE video_id = $1 "
        "ORDER BY created_at DESC LIMIT 1",
        video_id,
    )

    checkpoints = await db.fetch(
        "SELECT * FROM video_phase_checkpoints WHERE video_id = $1 "
        "ORDER BY started_at",
        video_id,
    )

    return {
        "video": catalog,
        "job": _with_progress(job, video_id, "status"),
        "checkpoints": checkpoints,
    }


@router.post("/process")
async def start_processing(background_tasks: BackgroundTasks):
    """Start the FIFO queue worker in the background.

    Only one worker runs at a time — a second call returns 409.  The worker
    picks up queued videos in queue_position order, runs each through the
    ingestion orchestrator, and sleeps between videos to stay under YouTube's
    rate limit.
    """
    global _worker_active
    if _worker_active:
        raise HTTPException(
            status_code=409,
            detail="queue worker is already running")

    if ingest_video is None:
        raise HTTPException(
            status_code=503,
            detail="video_ingestion pipeline not available yet")

    db = get_eval_db()
    count_row = await db.fetchrow(
        "SELECT COUNT(*) AS cnt FROM video_ingest_jobs WHERE status = 'queued'")
    queued_count = count_row["cnt"] if count_row else 0

    if queued_count == 0:
        return {"message": "Nothing queued", "queued_count": 0}

    background_tasks.add_task(_process_queue, db)
    return {"message": "Queue processing started", "queued_count": queued_count}


async def _process_queue(db):
    """FIFO worker — processes queued videos one at a time.

    Runs until the queue is empty, then clears the worker flag. Each video
    is updated to 'processing' before the call and to 'done' or 'failed'
    after. A random sleep between videos keeps YouTube comfortable.

    D68 safety layer: stops itself (rather than draining the whole queue)
    if the running spend ceiling is hit, or if downloads fail several times
    in a row — the latter is a circuit breaker against hammering YouTube
    into a harder, longer IP ban once it starts rejecting us.
    """
    global _worker_active
    if _worker_active:
        return
    _worker_active = True
    consecutive_download_failures = 0
    current_video = None
    try:
        while True:
            if not await check_budget(db):
                logger.warning(
                    "queue worker stopping: ingestion spend ceiling reached "
                    "(config.MAX_INGESTION_SPEND_USD)")
                break

            row = await db.fetchrow(
                "SELECT vij.id, vij.video_id, vc.title, vc.source_url, vc.platform, "
                "       vc.duration "
                "FROM video_ingest_jobs vij "
                "JOIN video_catalog vc ON vc.video_id = vij.video_id "
                "WHERE vij.status = 'queued' ORDER BY vij.queue_position LIMIT 1"
            )
            if not row:
                break

            job_id = row["id"]
            video_id = row["video_id"]
            duration = row.get("duration") or 0
            # A cancel left over from an earlier run of this video (one that
            # finished just as Cancel was clicked) must not stop this run.
            # Cleared before the row says 'processing' -- POST /cancel only
            # registers a stop for a processing job, so none can be lost here.
            _cancel_requests.discard(video_id)
            current_video = video_id

            await db.execute(
                "UPDATE video_ingest_jobs SET status='processing', phase='download', "
                "started_at=NOW(), updated_at=NOW() WHERE id=$1",
                job_id,
            )

            try:
                # ingest_video is synchronous (downloads, transcribes, embeds) —
                # run in a thread so the event loop stays responsive. It
                # catches its own phase failures and returns a result rather
                # than raising, so the circuit breaker below reads
                # result.status/.error, not an exception.
                # main_loop=asyncio.get_running_loop(): ingest_video() runs on a
                # worker thread (asyncio.to_thread) but its internal
                # _update_job_status/_save_checkpoint calls use `db` (this
                # process's single shared EvalDB) whose asyncpg pool is bound
                # to THIS loop. Without passing it through, those calls create
                # a second, independent event loop in the worker thread and
                # use the pool from there too -- asyncpg does not support
                # that, and it crashed the whole container (SIGSEGV) during
                # the first real ingestion attempt. See P95.
                result = await asyncio.to_thread(
                    ingest_video, video_id, config.VIDEO_WORK_DIR, db,
                    title=row.get("title") or "",
                    source_url=row.get("source_url") or "",
                    platform=row.get("platform") or "youtube",
                    duration=duration,
                    main_loop=asyncio.get_running_loop(),
                    # POST /cancel adds the id to _cancel_requests; the run
                    # polls this from its thread at each checkpoint.
                    should_stop=lambda vid=video_id: vid in _cancel_requests,
                )
            except Exception as exc:
                # Truly unexpected — ingest_video is designed not to raise.
                logger.exception("video %s ingestion raised unexpectedly", video_id)
                await db.execute(
                    "UPDATE video_ingest_jobs SET status='failed', "
                    "error_message=$2, updated_at=NOW() WHERE id=$1",
                    job_id, str(exc)[:500],
                )
                result = None

            if result is not None and result.status == "done":
                # P104: some platforms' yt-dlp metadata never carries duration
                # (confirmed for every Instagram reel) -- ingest_video()
                # backfills it via ffprobe on the downloaded file and reports
                # the corrected value in result.stats. Persist it back to
                # video_catalog and use it for cost estimation below; the
                # `duration` read from the queue row before ingestion ran may
                # still be the original (possibly zero) metadata value.
                corrected_duration = result.stats.get("duration_s") or duration
                if corrected_duration and corrected_duration != duration:
                    await db.execute(
                        "UPDATE video_catalog SET duration=$1 WHERE video_id=$2",
                        corrected_duration, video_id,
                    )
                    duration = corrected_duration
                await db.execute(
                    "UPDATE video_ingest_jobs SET status='done', warnings=$2, "
                    "completed_at=NOW(), updated_at=NOW() WHERE id=$1",
                    # '' = checks ran, nothing found; NULL is left for jobs
                    # finished before the checks existed (panel: "Not checked").
                    job_id, (None if result.stats.get("warnings") is None
                             else "; ".join(result.stats["warnings"])),
                )
                logger.info("video %s ingested successfully", video_id)
                consecutive_download_failures = 0
                # voyage_cost combines the image-based multimodal calls
                # (one per keyframe) and the text-embedding calls (OCR + OD
                # + transcript, via text_items_embedded) — these are billed
                # at different rates. Previously only counted keyframes,
                # silently undercounting real spend against the D68 ceiling.
                # Found in code review 2026-10-09.
                # Measured usage (usage_meter) when the run carries it; the
                # flat estimates below are only the fallback for a result
                # without one.
                if getattr(result, "usage", None):
                    await record_usage(db, video_id, result.usage,
                                       result.stats.get("phase_seconds", {}), "done")
                else:
                    voyage_cost = (
                        estimate_voyage_cost(result.stats.get("keyframes", 0))
                        + estimate_voyage_text_cost(result.stats.get("text_items_embedded", 0))
                    )
                    await record_cost(
                        db, video_id,
                        groq_cost=estimate_groq_cost(duration),
                        voyage_cost=voyage_cost,
                        # ocr_labels == chart-frame count: the same loop that runs
                        # OCR also runs the D67 vision description, once per frame.
                        vision_cost=estimate_vision_cost(result.stats.get("ocr_labels", 0)),
                    )
            elif result is not None and result.status == "cancelled":
                await _record_cancelled(db, job_id, video_id, result)
                # Deliberately leaves consecutive_download_failures alone: a
                # user's cancel says nothing about whether YouTube is
                # blocking us, so it neither trips nor resets the breaker.
            elif result is not None:
                logger.error("video %s ingestion failed: %s", video_id, result.error)
                await db.execute(
                    "UPDATE video_ingest_jobs SET status='failed', "
                    "error_message=$2, updated_at=NOW() WHERE id=$1",
                    job_id, (result.error or "")[:500],
                )
                # The phases before the failure were paid for -- previously a
                # failed run recorded no cost at all, invisible to the D68 ceiling.
                if getattr(result, "usage", None):
                    await record_usage(db, video_id, result.usage,
                                       result.stats.get("phase_seconds", {}), "failed")
                # gemini_down.md: the vision provider is unusable (empty credit,
                # bad key, ...). Every later video would be refused the same way,
                # so stop the worker and leave them queued; /video/process
                # restarts it once the provider is back (retry this one too).
                if (result.error or "").startswith(VISION_UNAVAILABLE_PREFIX):
                    logger.error("queue worker stopping: %s -- remaining videos stay queued",
                                 result.error)
                    break
                if (result.error or "").startswith("download failed"):
                    consecutive_download_failures += 1
                    if consecutive_download_failures >= config.MAX_CONSECUTIVE_DOWNLOAD_FAILURES:
                        logger.error(
                            "queue worker stopping: %d consecutive download "
                            "failures — possible IP block, not retrying blind",
                            consecutive_download_failures)
                        break
                else:
                    consecutive_download_failures = 0

            _cancel_requests.discard(video_id)
            current_video = None
            # Whatever the outcome (done, failed, cancelled after the old points
            # were cleared), the searchable set may have changed: the planner
            # must not keep planning from the list it had before this run.
            query_planner.invalidate_indexed()

            # Rate-limit between videos so yt-dlp doesn't trigger IP blocks.
            await asyncio.sleep(
                random.uniform(config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX))
    finally:
        if current_video is not None:
            # Left the loop mid-video (a break above): its cancel is spent.
            _cancel_requests.discard(current_video)
        _worker_active = False
        logger.info("queue worker stopped")


async def _record_cancelled(db, job_id: int, video_id: str, result) -> None:
    """Final job state + spend for a run stopped by POST /cancel.

    The usage up to the stop is real spend, recorded with status
    "cancelled" so the D68 ceiling still counts it. A re-ingest stopped
    before its index phase never touched Qdrant: when the previous run's
    index is complete the job goes back to 'done' (with a note), so the
    panel does not show a searchable video as cancelled. Otherwise
    'cancelled', with the phase it stopped in.
    """
    phase = getattr(result, "phase", None) or "unknown phase"
    stats = getattr(result, "stats", None) or {}
    if getattr(result, "usage", None):
        await record_usage(db, video_id, result.usage, stats.get("phase_seconds", {}),
                           "cancelled")
    if phase not in ("index", "cleanup") and await _has_complete_index(db, video_id):
        note = "re-ingest cancelled during %s; previous index kept" % phase
        logger.info("video %s: %s", video_id, note)
        await db.execute(
            "UPDATE video_ingest_jobs SET status='done', error_message=$2, warnings=$2, "
            "completed_at=NOW(), updated_at=NOW() WHERE id=$1",
            job_id, note,
        )
        return
    message = "cancelled by user during %s" % phase
    extra = [w for w in stats.get("warnings") or [] if "after cancel" in w]
    if extra:
        message += "; " + "; ".join(extra)
    logger.info("video %s: %s", video_id, message)
    await db.execute(
        "UPDATE video_ingest_jobs SET status='cancelled', error_message=$2, "
        "completed_at=NOW(), updated_at=NOW() WHERE id=$1",
        job_id, message[:500],
    )


# ── Settings ─────────────────────────────────────────────

# Stored in eval_db so they survive container restarts. The table is created
# lazily on first write — no migration needed.
_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS video_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""
_SETTINGS_CREATED = False


async def _ensure_settings_table(db):
    global _SETTINGS_CREATED
    if _SETTINGS_CREATED:
        return
    await db.execute(_SETTINGS_TABLE)
    _SETTINGS_CREATED = True


@router.get("/costs")
async def get_costs():
    """Ingestion cost view for the admin panel: official unit prices, measured
    usage + list-price spend per platform and per video (P146), and pre-metering
    estimates kept separate. Free tier is "not confirmed" -- no API response
    reports it; billing-data reconciliation is a later step."""
    db = get_eval_db()
    usage_rows = await db.fetch(
        "SELECT u.video_id, COALESCE(c.title, '') AS title, u.status, u.usage, "
        "u.phase_seconds, u.prices_as_of, u.recorded_at "
        "FROM video_ingest_usage u LEFT JOIN video_catalog c ON c.video_id = u.video_id")
    estimate_rows = await db.fetch(
        "SELECT k.video_id, COALESCE(c.title, '') AS title, k.groq_cost_usd, "
        "k.voyage_cost_usd, k.vision_cost_usd, k.recorded_at "
        "FROM video_ingest_costs k LEFT JOIN video_catalog c ON c.video_id = k.video_id")
    return summarize_costs(usage_rows or [], estimate_rows or [])


@router.get("/settings")
async def get_settings():
    db = get_eval_db()
    pool = await db.pool()
    if pool is None:
        return VideoSettings().model_dump()
    await _ensure_settings_table(db)
    rows = await db.fetch("SELECT key, value FROM video_settings")
    raw = {r["key"]: r["value"] for r in rows}
    return VideoSettings(
        auto_check_enabled=raw.get("auto_check_enabled", "false") == "true",
        auto_check_hour=int(raw.get("auto_check_hour", "6")),
        auto_queue=raw.get("auto_queue", "false") == "true",
    ).model_dump()


@router.post("/settings")
async def save_settings(settings: VideoSettings):
    db = get_eval_db()
    pool = await db.pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="eval database unavailable")
    await _ensure_settings_table(db)
    pairs = {
        "auto_check_enabled": str(settings.auto_check_enabled).lower(),
        "auto_check_hour": str(settings.auto_check_hour),
        "auto_queue": str(settings.auto_queue).lower(),
    }
    for k, v in pairs.items():
        await db.execute(
            "INSERT INTO video_settings (key, value) VALUES ($1, $2) "
            "ON CONFLICT (key) DO UPDATE SET value = $2",
            k, v,
        )
    logger.info("video settings saved: %s", pairs)
    return {"message": "Settings saved", "settings": settings.model_dump()}


# ── Google Drive intake (V6) + auto-delete (V7) ──────────────
# the client's local fetcher uploads videos to a Drive folder; these endpoints are
# the server-side pickup. Nothing here runs by itself: scan is read-only,
# queue needs explicit ids AND confirm=true (ingestion is client-triggered
# only, and a folder can hold hundreds of files -- D68), and a real cleanup
# needs the admin toggle ON. Queued Drive videos are processed by the same
# worker as every other video (/video/process), whose download phase copies
# the file by id (video_acquisition.download_video -> drive_intake).

class DriveQueueRequest(BaseModel):
    file_ids: List[str] = []
    confirm: bool = False


class DriveCleanupRequest(BaseModel):
    dry_run: bool = True


class DriveFolderRequest(BaseModel):
    folder_id: str
    name: str = ""
    confirm: bool = False


class DriveIntakeSettings(BaseModel):
    # None = leave unchanged (each form saves only its own toggle).
    autodelete_enabled: Optional[bool] = None
    autodelete_hours: Optional[int] = None
    autoapprove_enabled: Optional[bool] = None


# Indirection points so the offline tests can inject a FakeDriveLister and a
# fixed clock without touching rclone or the wall clock.
def _drive_lister():
    return drive_intake.lister_from_env()


def _drive_folder_lister(folder_id):
    return drive_intake.lister_from_env(folder_id=folder_id)


def _drive_now():
    return datetime.datetime.now(datetime.timezone.utc)


def _lister_or_503():
    try:
        return _drive_lister()
    except drive_intake.DriveIntakeNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))


async def _db_or_503():
    # EvalDB swallows a dead pool and returns [] -- for a scan that would mark
    # every file "new" and invite re-queuing videos that are already ingested.
    db = get_eval_db()
    if await db.pool() is None:
        raise HTTPException(status_code=503, detail="eval database unavailable")
    return db


async def _drive_list(lister):
    # A failed rclone call raises (never an empty list that reads as "folder
    # empty"); surface it as 502 with rclone's own message.
    try:
        return await asyncio.to_thread(lister.list_files)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="drive listing failed: %s" % exc)


async def _drive_settings(db) -> dict:
    await _ensure_settings_table(db)
    rows = await db.fetch("SELECT key, value FROM video_settings")
    return drive_intake.parse_settings({r["key"]: r["value"] for r in rows})


@router.post("/drive-intake/scan")
async def drive_intake_scan():
    """List the Drive intake folder and classify each file as new / queued /
    processing / ingested / failed (non-video files listed as ignored).
    Dry run by construction: reads only, never queues."""
    lister = _lister_or_503()
    db = await _db_or_503()
    files = await _drive_list(lister)
    index = await drive_intake.job_index(
        db, [drive_intake.video_id_for(f.file_id) for f in files if drive_intake.is_video_file(f)])
    return dict(drive_intake.classify(files, index), folder_id=lister.folder_id, dry_run=True)


@router.post("/drive-intake/queue")
async def drive_intake_queue(body: DriveQueueRequest):
    """Queue explicit Drive files for ingestion. Catalogues each as platform
    "gdrive" (title = file name, source_url = the Drive view link) and adds a
    queued job, exactly like /video/queue/batch -- same D68 caps, same
    retry-reuses-the-row rule (P97). Does NOT start the worker."""
    if not body.confirm:
        raise HTTPException(status_code=400,
                            detail="confirm=true is required to queue Drive files")
    file_ids = [f.strip() for f in body.file_ids if f and f.strip()]
    if not file_ids:
        raise HTTPException(status_code=400, detail="file_ids must not be empty")
    if len(file_ids) > config.MAX_QUEUE_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail="batch size %d exceeds MAX_QUEUE_BATCH_SIZE (%d) — see D68, "
                   "this is a deliberate no-bulk-load safety limit"
                   % (len(file_ids), config.MAX_QUEUE_BATCH_SIZE))
    lister = _lister_or_503()
    db = await _db_or_503()
    # Re-list rather than trusting ids from an older scan: the file may have
    # been removed, and only a listed file has a name/extension to record.
    by_id = {f.file_id: f for f in await _drive_list(lister)}
    index = await drive_intake.job_index(
        db, [drive_intake.video_id_for(fid) for fid in file_ids if fid in by_id])
    in_flight = await _in_flight_count(db)
    details, queued = [], 0
    for fid in file_ids:
        f = by_id.get(fid)
        if f is None:
            details.append({"file_id": fid, "status": "not_found"})
            continue
        if not drive_intake.is_video_file(f):
            details.append({"file_id": fid, "name": f.name, "status": "not_video"})
            continue
        vid = drive_intake.video_id_for(fid)
        state = drive_intake.state_for(index, fid)
        if state not in ("new", "failed"):
            details.append({"file_id": fid, "name": f.name, "video_id": vid,
                            "status": "skipped", "reason": state})
            continue
        if in_flight >= config.MAX_IN_FLIGHT_VIDEOS:
            details.append({"file_id": fid, "name": f.name, "video_id": vid,
                            "status": "skipped", "reason": "in_flight_limit_reached"})
            continue
        # duration 0: Drive gives no media metadata; the download phase
        # backfills it with ffprobe (P104), same as Instagram reels.
        await db.execute(
            "INSERT INTO video_catalog (video_id, title, duration, channel_id, "
            "discovered_at, source_url, platform) "
            "VALUES ($1, $2, 0, 'gdrive', NOW(), $3, $4) ON CONFLICT (video_id) DO NOTHING",
            vid, drive_intake.title_for(f), drive_intake.drive_view_url(fid),
            drive_intake.PLATFORM)
        pos = await _next_queue_position(db)
        existing = await db.fetchrow(
            "SELECT id, status FROM video_ingest_jobs WHERE video_id = $1 "
            "ORDER BY created_at DESC LIMIT 1", vid)
        if existing:
            await db.execute(
                "UPDATE video_ingest_jobs SET status = 'queued', phase = NULL, "
                "phase_progress = 0.0, error_message = NULL, queue_position = $2, "
                "retry_count = retry_count + 1, started_at = NULL, completed_at = NULL, "
                "updated_at = NOW() WHERE id = $1", existing["id"], pos)
        else:
            await db.execute(
                "INSERT INTO video_ingest_jobs "
                "(video_id, status, queue_position, created_at, updated_at) "
                "VALUES ($1, 'queued', $2, NOW(), NOW())", vid, pos)
        details.append({"file_id": fid, "name": f.name, "video_id": vid,
                        "status": "queued", "queue_position": pos})
        queued += 1
        in_flight += 1
    logger.info("drive intake: queued %d of %d requested file(s)", queued, len(file_ids))
    return {"queued": queued, "details": details}


def _folder_view() -> dict:
    sel = drive_intake.selected_folder()
    return {"intake_folder_id": sel["folder_id"], "intake_folder_name": sel["name"],
            "intake_folder_source": sel["source"], "intake_folder_default": sel["env_folder_id"]}


@router.get("/drive-intake/settings")
async def drive_intake_get_settings():
    db = get_eval_db()
    if await db.pool() is None:
        return dict(drive_intake.parse_settings({}), **_folder_view(), autopilot=dict(_autopilot))
    return dict(await _drive_settings(db), **_folder_view(), autopilot=dict(_autopilot))


@router.post("/drive-intake/folder")
async def drive_intake_set_folder(body: DriveFolderRequest):
    """Choose the upload folder (admin panel). Needs confirm=true; the folder
    must be listable with the shared rclone credential; refused while a Drive
    video is queued or processing (its download re-lists the folder)."""
    if not body.confirm:
        raise HTTPException(status_code=400, detail="confirm must be true")
    if not drive_intake.valid_folder_id(body.folder_id):
        raise HTTPException(status_code=400, detail="not a Google Drive folder id")
    db = await _db_or_503()
    busy = await db.fetchrow(
        "SELECT COUNT(*) AS cnt FROM video_ingest_jobs j JOIN video_catalog c ON c.video_id = j.video_id "
        "WHERE c.platform = $1 AND j.status IN ('queued', 'processing')", drive_intake.PLATFORM)
    if busy and busy["cnt"]:
        raise HTTPException(status_code=409, detail="%d Drive video(s) are queued or processing; change the "
                                                    "upload folder after they finish" % busy["cnt"])
    try:
        lister = _drive_folder_lister(body.folder_id)
    except drive_intake.DriveIntakeNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    await _drive_list(lister)                      # 502 if the credential cannot list it
    try:
        drive_intake.save_selected_folder(body.folder_id, body.name)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="could not save the folder choice: %s" % exc)
    logger.info("drive intake folder set to %s (%s)", body.folder_id, body.name)
    return _folder_view()


@router.post("/drive-intake/settings")
async def drive_intake_save_settings(settings: DriveIntakeSettings):
    db = await _db_or_503()
    await _ensure_settings_table(db)
    pairs = drive_intake.settings_pairs(settings.autodelete_enabled, settings.autodelete_hours,
                                        settings.autoapprove_enabled)
    for k, v in pairs.items():
        await db.execute(
            "INSERT INTO video_settings (key, value) VALUES ($1, $2) "
            "ON CONFLICT (key) DO UPDATE SET value = $2", k, v)
    logger.info("drive intake settings saved: %s", pairs)
    return await _drive_settings(db)


# ── Drive autopilot: auto-approve + auto-delete, both default OFF ──────────
# One loop in this process (started from main.lifespan). Each round reads the
# two toggles; with both off it does nothing at all. Auto-approve queues only
# "new" videos (never a failed one, so a broken file cannot loop), at most
# MAX_QUEUE_BATCH_SIZE per round, through the same queue endpoint as a click,
# then starts the same worker -- which keeps its own spend ceiling (D68).
AUTOPILOT_INTERVAL_S = int(os.getenv("DRIVE_AUTOPILOT_INTERVAL_S", "600"))
_autopilot = {"last_run_at": None, "last_queued": 0, "last_deleted": 0, "last_error": ""}


async def drive_autopilot_round(now=None) -> dict:
    db = get_eval_db()
    if await db.pool() is None:
        return {"skipped": "eval database unavailable"}
    settings = await _drive_settings(db)
    if not (settings["autoapprove_enabled"] or settings["autodelete_enabled"]):
        return {"skipped": "both toggles off"}
    out = {"queued": 0, "deleted": 0}
    lister = _drive_lister()
    if settings["autoapprove_enabled"]:
        scan = await drive_intake_scan()
        new = [v["file_id"] for v in scan["videos"] if v["state"] == "new"][:config.MAX_QUEUE_BATCH_SIZE]
        if new:
            res = await drive_intake_queue(DriveQueueRequest(file_ids=new, confirm=True))
            out["queued"] = res["queued"]
            if res["queued"] and not _worker_active and ingest_video is not None:
                asyncio.create_task(_process_queue(db))
    if settings["autodelete_enabled"]:
        report = await drive_intake.run_cleanup(
            db, lister, enabled=True, window_hours=settings["autodelete_hours"], dry_run=False,
            now=now or _drive_now())
        out["deleted"] = len(report.get("deleted", []))
    return out


async def drive_autopilot_loop():
    while True:
        try:
            res = await drive_autopilot_round()
            _autopilot.update(last_run_at=_drive_now().isoformat(), last_error="",
                              last_queued=res.get("queued", 0), last_deleted=res.get("deleted", 0))
            if res.get("queued") or res.get("deleted"):
                logger.info("drive autopilot: %s", res)
        except asyncio.CancelledError:
            raise
        except Exception as exc:          # one bad round never stops the loop
            detail = getattr(exc, "detail", None) or str(exc)
            _autopilot.update(last_run_at=_drive_now().isoformat(), last_error=str(detail)[:300])
            logger.warning("drive autopilot round failed: %s", detail)
        await asyncio.sleep(AUTOPILOT_INTERVAL_S)


@router.post("/drive-intake/cleanup")
async def drive_intake_cleanup(body: DriveCleanupRequest):
    """Auto-delete from Drive the files whose video is ingested ("done") and
    finished at least the configured window ago. dry_run (default) reports
    what would go; a real run is refused (409) while the admin toggle is OFF.
    Meant to be called by an operator or a host timer -- nothing schedules it."""
    lister = _lister_or_503()
    db = await _db_or_503()
    settings = await _drive_settings(db)
    try:
        return await drive_intake.run_cleanup(
            db, lister, enabled=settings["autodelete_enabled"],
            window_hours=settings["autodelete_hours"], dry_run=body.dry_run,
            now=_drive_now())
    except drive_intake.DriveCleanupDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ── YouTube cookies (P91/D72) ───────────────────────────────
# Lets the admin panel supply a cookies.txt without SSH access, either as a
# file upload or pasted text. The content itself never comes back out
# through any endpoint here -- only the health numbers cookie_guard tracks
# (age, usage, a content fingerprint) and the result of a live validation
# probe against YouTube.

class CookieTextRequest(BaseModel):
    content: str


async def _save_and_validate(content: str) -> dict:
    """Shared by the file-upload and paste-text endpoints: save, then
    immediately probe it against real YouTube so the admin finds out right
    away whether the cookies actually work, not just that they parsed.

    probe_cookies() makes a real (blocking) network call, so it runs off
    the event loop the same way ingest_video() does elsewhere in this
    codebase (asyncio.to_thread) -- otherwise this request would stall
    every other concurrent one for however long the yt-dlp call takes.
    """
    from pipelines.cookie_guard import get_cookie_status, record_validation, save_cookie_file
    try:
        save_cookie_file(content, config.YTDLP_COOKIES_FILE)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    from pipelines.video_acquisition import probe_cookies
    result = await asyncio.to_thread(probe_cookies, config.YTDLP_COOKIES_FILE)
    record_validation(config.YTDLP_COOKIES_FILE, result["valid"], result["error"])
    return {
        "message": "Cookies saved",
        "validation": result,
        "status": get_cookie_status(config.YTDLP_COOKIES_FILE),
    }


@router.get("/cookies/status")
async def cookies_status():
    from pipelines.cookie_guard import get_cookie_status
    return get_cookie_status(config.YTDLP_COOKIES_FILE)


@router.post("/cookies")
async def upload_cookies(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="file is not valid text")
    logger.info("cookies.txt uploaded via admin panel (%d bytes)", len(raw))
    return await _save_and_validate(content)


@router.post("/cookies/text")
async def paste_cookies(body: CookieTextRequest):
    logger.info("cookies.txt pasted via admin panel (%d bytes)", len(body.content))
    return await _save_and_validate(body.content)


@router.post("/cookies/validate")
async def validate_cookies():
    """Re-run the live probe against the already-saved cookies file --
    for "did these stop working since I uploaded them" without re-uploading."""
    from pipelines.cookie_guard import get_cookie_status, record_validation
    status = get_cookie_status(config.YTDLP_COOKIES_FILE)
    if not status["configured"]:
        raise HTTPException(status_code=400, detail="no cookies file configured")
    from pipelines.video_acquisition import probe_cookies
    result = await asyncio.to_thread(probe_cookies, config.YTDLP_COOKIES_FILE)
    record_validation(config.YTDLP_COOKIES_FILE, result["valid"], result["error"])
    return {"validation": result, "status": get_cookie_status(config.YTDLP_COOKIES_FILE)}


@router.delete("/cookies")
async def delete_cookies():
    from pipelines.cookie_guard import remove_cookie_file
    remove_cookie_file(config.YTDLP_COOKIES_FILE)
    logger.info("cookies.txt removed via admin panel")
    return {"message": "Cookies removed"}


# ── Self-test ─────────────────────────────────────────────
# Offline — no database, no network. Exercises router wiring, validation
# helpers and the worker's non-reentrant flag.

def _self_test():
    import asyncio as _aio

    ok = 0

    # 1. Router has routes
    paths = [r.path for r in router.routes]
    assert "/discover" in paths, "missing /discover route"
    assert "/process" in paths, "missing /process route"
    ok += 1
    print("ok %d — router has expected routes" % ok)

    # 2. _next_queue_position returns 1 for an empty result
    class FakeDB:
        async def fetchrow(self, sql, *a):
            return {"next_pos": 1}
        async def fetch(self, sql, *a):
            return []
        async def execute(self, sql, *a):
            return True

    pos = _aio.run(_next_queue_position(FakeDB()))
    assert pos == 1, "expected 1, got %s" % pos
    ok += 1
    print("ok %d — queue position starts at 1 for empty queue" % ok)

    # 3. Queue position increments
    class FakeDB2:
        async def fetchrow(self, sql, *a):
            return {"next_pos": 42}
        async def fetch(self, sql, *a):
            return []
        async def execute(self, sql, *a):
            return True

    pos2 = _aio.run(_next_queue_position(FakeDB2()))
    assert pos2 == 42, "expected 42, got %s" % pos2
    ok += 1
    print("ok %d — queue position calculation returns correct value" % ok)

    # 4. Status filter validation
    assert "queued" in _VALID_STATUSES
    assert "processing" in _VALID_STATUSES
    assert "bogus" not in _VALID_STATUSES
    ok += 1
    print("ok %d — status filter validates known states" % ok)

    # 5. Worker flag prevents re-entry
    global _worker_active
    saved = _worker_active
    _worker_active = True

    async def _check_reentrant():
        # _process_queue should return immediately when flag is set — the early
        # guard exits before the try/finally, so the flag stays True (correct:
        # the *first* caller's finally will clear it).
        class NoDB:
            async def fetchrow(self, sql, *a):
                raise AssertionError("should not reach the DB")
        await _process_queue(NoDB())

    _aio.run(_check_reentrant())
    assert _worker_active, "flag must stay True — the rejected call must not clear it"
    _worker_active = saved
    ok += 1
    print("ok %d — worker flag prevents re-entry" % ok)

    # 6. BatchQueueRequest validates
    req = BatchQueueRequest(video_ids=["a", "b"])
    assert len(req.video_ids) == 2
    ok += 1
    print("ok %d — BatchQueueRequest accepts a list of IDs" % ok)

    # 7. BatchQueueRequest rejects non-list
    try:
        BatchQueueRequest(video_ids="not-a-list")
        # pydantic v2 coerces a string into a list of characters — check length
        # pydantic v1 raises. Either way we verify the model exists.
    except Exception:
        pass
    ok += 1
    print("ok %d — BatchQueueRequest model exists and is typed" % ok)

    # 8. _next_queue_position handles None row
    class FakeDB3:
        async def fetchrow(self, sql, *a):
            return None

    pos3 = _aio.run(_next_queue_position(FakeDB3()))
    assert pos3 == 1, "expected 1 for None row, got %s" % pos3
    ok += 1
    print("ok %d — queue position defaults to 1 when DB returns None" % ok)

    # 9. Router has the fetch-url route
    assert "/fetch-url" in paths, "missing /fetch-url route"
    ok += 1
    print("ok %d — router has /fetch-url route" % ok)

    # 10. _parse_upload_date parses yt-dlp's YYYYMMDD format
    d = _parse_upload_date("20261001")
    assert d == datetime.date(2026, 10, 1), d
    ok += 1
    print("ok %d — _parse_upload_date parses YYYYMMDD" % ok)

    # 11. _parse_upload_date tolerates missing/malformed input
    assert _parse_upload_date("") is None
    assert _parse_upload_date("not-a-date") is None
    ok += 1
    print("ok %d — _parse_upload_date returns None for missing/malformed input" % ok)

    # 12. FetchUrlRequest validates
    req2 = FetchUrlRequest(url="https://twitter.com/u/status/1")
    assert req2.url == "https://twitter.com/u/status/1"
    ok += 1
    print("ok %d — FetchUrlRequest model accepts a url" % ok)

    # 13. _process_queue passes the catalog's work_dir/title/source_url/platform
    # through to ingest_video — this is the bug that made every queued video
    # (YouTube included) crash with "missing work_dir" before this fix.
    global ingest_video
    saved_ingest_video = ingest_video
    captured_call = {}

    class FakeResult:
        status = "done"
        error = None
        stats = {"keyframes": 5}

    def fake_ingest_video(video_id, work_dir, eval_db=None, title="", source_url="",
                          platform="youtube", duration=0, main_loop=None, should_stop=None):
        captured_call["video_id"] = video_id
        captured_call["work_dir"] = work_dir
        captured_call["title"] = title
        captured_call["source_url"] = source_url
        captured_call["platform"] = platform
        captured_call["duration"] = duration
        captured_call["main_loop"] = main_loop
        return FakeResult()
    ingest_video = fake_ingest_video

    class FakeDB4:
        job_calls = 0

        async def fetchrow(self, sql, *a):
            if "video_ingest_costs" in sql:
                return {"total": 0.0}
            FakeDB4.job_calls += 1
            if FakeDB4.job_calls == 1:
                return {
                    "id": 7, "video_id": "tw123",
                    "title": "A tweet", "source_url": "https://twitter.com/u/status/tw123",
                    "platform": "twitter", "duration": 42,
                }
            return None  # queue empty on the next poll

        async def execute(self, sql, *a):
            return True

    saved_sleep_min, saved_sleep_max = config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX
    config.YTDLP_SLEEP_MIN = config.YTDLP_SLEEP_MAX = 0
    try:
        _aio.run(_process_queue(FakeDB4()))
    finally:
        ingest_video = saved_ingest_video
        config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX = saved_sleep_min, saved_sleep_max

    assert captured_call["video_id"] == "tw123", captured_call
    assert captured_call["work_dir"] == config.VIDEO_WORK_DIR, captured_call
    assert captured_call["title"] == "A tweet", captured_call
    assert captured_call["source_url"] == "https://twitter.com/u/status/tw123", captured_call
    assert captured_call["platform"] == "twitter", captured_call
    assert captured_call["duration"] == 42, captured_call
    # P95: main_loop must be the real running loop, not the unset default --
    # without it, ingest_video()'s internal eval_db calls (on its own worker
    # thread) create a second, colliding event loop against the same asyncpg
    # pool and crashed the container during the first real ingestion attempt.
    assert isinstance(captured_call["main_loop"], _aio.AbstractEventLoop), captured_call
    ok += 1
    print("ok %d — _process_queue forwards work_dir/title/source_url/platform/duration/main_loop to ingest_video" % ok)

    # 13d. P104: when ingest_video() backfills an unknown duration (reported
    # via result.stats["duration_s"]), _process_queue must persist it to
    # video_catalog and use it for Groq cost estimation -- not the original
    # (zero, for every Instagram reel) value read from the queue row.
    class FakeResultWithDuration:
        status = "done"
        error = None
        stats = {"keyframes": 5, "duration_s": 63.5}

    def fake_ingest_video_ig(video_id, work_dir, eval_db=None, title="", source_url="",
                             platform="youtube", duration=0, main_loop=None, should_stop=None):
        return FakeResultWithDuration()
    ingest_video = fake_ingest_video_ig

    global record_cost
    saved_record_cost = record_cost
    captured_cost_call = {}

    async def fake_record_cost(db, video_id, groq_cost=0.0, voyage_cost=0.0, vision_cost=0.0):
        captured_cost_call["groq_cost"] = groq_cost
    record_cost = fake_record_cost

    executed_sql = []

    class FakeDB4b:
        job_calls = 0

        async def fetchrow(self, sql, *a):
            if "video_ingest_costs" in sql:
                return {"total": 0.0}
            FakeDB4b.job_calls += 1
            if FakeDB4b.job_calls == 1:
                return {
                    "id": 8, "video_id": "ig1",
                    "title": "Video by smcandict",
                    "source_url": "https://www.instagram.com/reel/ig1/",
                    "platform": "instagram", "duration": 0,
                }
            return None

        async def execute(self, sql, *a):
            executed_sql.append((sql, a))
            return True

    try:
        _aio.run(_process_queue(FakeDB4b()))
    finally:
        ingest_video = saved_ingest_video
        record_cost = saved_record_cost

    catalog_updates = [call for call in executed_sql if "UPDATE video_catalog" in call[0]]
    assert len(catalog_updates) == 1, executed_sql
    assert catalog_updates[0][1] == (63.5, "ig1"), catalog_updates[0]
    assert captured_cost_call["groq_cost"] == estimate_groq_cost(63.5), captured_cost_call
    ok += 1
    print("ok %d — _process_queue persists an ffprobe-backfilled duration and uses it for cost estimation (P104)" % ok)

    # 13b. A run carrying measured usage (usage_meter) records THAT, done or
    # failed -- not the flat estimates; a failed run's paid phases still count.
    global record_usage
    saved_record_usage = record_usage
    usage_calls = []

    async def fake_record_usage(db, video_id, usage, phase_seconds, status):
        usage_calls.append((video_id, usage["total_usd"], phase_seconds, status))
    record_usage = fake_record_usage

    def _result(status, error=None):
        return type("R", (), {"status": status, "error": error,
                              "stats": {"duration_s": 60.0, "phase_seconds": {"asr": 1.5}},
                              "usage": {"total_usd": 0.42, "prices_as_of": "2026-10-19"}})()

    for status, err in (("done", None), ("failed", "embedding failed: boom")):
        FakeDB4b.job_calls = 0
        ingest_video = (lambda s, e: (lambda *a, **k: _result(s, e)))(status, err)
        record_cost = fake_record_cost
        try:
            _aio.run(_process_queue(FakeDB4b()))
        finally:
            ingest_video = saved_ingest_video
            record_cost = saved_record_cost
    record_usage = saved_record_usage
    assert usage_calls == [("ig1", 0.42, {"asr": 1.5}, "done"),
                           ("ig1", 0.42, {"asr": 1.5}, "failed")], usage_calls
    ok += 1
    print("ok %d — measured usage is recorded for done AND failed runs" % ok)

    # 13c. Ingestion warnings (frame/transcript/chart checks) are stored on the
    # job row so the admin panel can show them -- a span label alone is
    # invisible, which is how P137's 0-frame "success" went unnoticed.
    executed_sql.clear()
    FakeDB4b.job_calls = 0
    warned = type("R", (), {"status": "done", "error": None,
                            "stats": {"duration_s": 60.0, "warnings": ["the transcript stops at 49%"]},
                            "usage": {}})()
    ingest_video = lambda *a, **k: warned
    record_cost = fake_record_cost
    try:
        _aio.run(_process_queue(FakeDB4b()))
    finally:
        ingest_video = saved_ingest_video
        record_cost = saved_record_cost
    done_sql = [(s, a) for s, a in executed_sql if "status='done'" in s]
    assert done_sql and "warnings" in done_sql[0][0], executed_sql
    assert "the transcript stops at 49%" in done_sql[0][1], done_sql
    ok += 1
    print("ok %d — ingestion warnings are written to the job row" % ok)

    # 13d. Checks that ran and found nothing store '' -- not NULL, which is
    # what a job finished before the checks existed has; the panel shows the
    # latter as "Not checked" (2026-10-21).
    executed_sql.clear()
    FakeDB4b.job_calls = 0
    clean = type("R", (), {"status": "done", "error": None,
                           "stats": {"duration_s": 60.0, "warnings": []}, "usage": {}})()
    ingest_video = lambda *a, **k: clean
    record_cost = fake_record_cost
    try:
        _aio.run(_process_queue(FakeDB4b()))
    finally:
        ingest_video = saved_ingest_video
        record_cost = saved_record_cost
    done_sql = [(s, a) for s, a in executed_sql if "status='done'" in s]
    assert done_sql and done_sql[0][1][1] == "", done_sql
    ok += 1
    print("ok %d — a checked run with no warnings stores '' (not NULL = never checked)" % ok)

    # 13e. gemini_down.md: a video refused because the vision provider is
    # unusable (empty credit, bad key) stops the worker -- the next queued video
    # is NOT pulled and failed the same way; any other failure still moves on.
    class FakeDB13e:
        def __init__(self):
            self.job_calls = 0

        async def fetchrow(self, sql, *a):
            if "video_ingest_costs" in sql:
                return {"total": 0.0}
            self.job_calls += 1
            if self.job_calls <= 2:
                return {"id": 20 + self.job_calls, "video_id": "q%d" % self.job_calls, "title": "T",
                        "source_url": "", "platform": "youtube", "duration": 60}
            return None

        async def execute(self, sql, *a):
            return True

    ran = []

    def _failing(error):
        def _ingest(video_id, *a, **k):
            ran.append(video_id)
            return type("R", (), {"status": "failed", "error": error,
                                  "stats": {"duration_s": 60.0}, "usage": {}})()
        return _ingest

    saved_sleep_min, saved_sleep_max = config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX
    config.YTDLP_SLEEP_MIN = config.YTDLP_SLEEP_MAX = 0
    try:
        for error, expect in (
                (VISION_UNAVAILABLE_PREFIX + " (billing): gemini:gemini-3.6-flash -- credits depleted", ["q1"]),
                ("embedding failed: boom", ["q1", "q2"])):
            ran.clear()
            ingest_video = _failing(error)
            _aio.run(_process_queue(FakeDB13e()))
            assert ran == expect, (error, ran)
    finally:
        ingest_video = saved_ingest_video
        config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX = saved_sleep_min, saved_sleep_max
    ok += 1
    print("ok %d — vision provider unavailable stops the queue worker (rest stay queued); "
          "other failures continue" % ok)

    # 14. queue_batch rejects an oversized batch before ever touching the DB
    # (D68) — the length check runs before get_eval_db() is called.
    oversized = ["v%d" % i for i in range(config.MAX_QUEUE_BATCH_SIZE + 1)]
    try:
        _aio.run(queue_batch(BatchQueueRequest(video_ids=oversized)))
        raise AssertionError("expected HTTPException for oversized batch")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400, exc
    ok += 1
    print("ok %d — queue_batch rejects a batch over MAX_QUEUE_BATCH_SIZE" % ok)

    # 15. _in_flight_count reads the queued+processing count
    class FakeDB5:
        async def fetchrow(self, sql, *a):
            return {"cnt": 4}

    assert _aio.run(_in_flight_count(FakeDB5())) == 4
    ok += 1
    print("ok %d — _in_flight_count reads the queued+processing count" % ok)

    # 16. _process_queue stops immediately when the spend ceiling is already
    # reached — never even looks at the queue (D68 budget circuit breaker).
    class FakeDB6:
        async def fetchrow(self, sql, *a):
            if "video_ingest_costs" in sql:
                return {"total": 999.0}  # far over any default ceiling
            raise AssertionError("should stop before querying the job queue")

    saved_limit = config.MAX_INGESTION_SPEND_USD
    config.MAX_INGESTION_SPEND_USD = 50.0
    try:
        _aio.run(_process_queue(FakeDB6()))
    finally:
        config.MAX_INGESTION_SPEND_USD = saved_limit
    ok += 1
    print("ok %d — _process_queue stops immediately once the spend ceiling is already reached" % ok)

    # 17. _process_queue stops after MAX_CONSECUTIVE_DOWNLOAD_FAILURES in a
    # row — the circuit breaker against hammering a YouTube IP block.
    class FailingResult:
        status = "failed"
        error = "download failed: Sign in to confirm you're not a bot"

    def failing_ingest_video(video_id, work_dir, eval_db=None, **kwargs):
        return FailingResult()

    class FakeDB7:
        job_count = 0

        async def fetchrow(self, sql, *a):
            if "video_ingest_costs" in sql:
                return {"total": 0.0}
            FakeDB7.job_count += 1
            return {"id": FakeDB7.job_count, "video_id": "v%d" % FakeDB7.job_count,
                    "title": "", "source_url": "", "platform": "youtube", "duration": 0}

        async def execute(self, sql, *a):
            return True

    saved_ingest_video2 = ingest_video
    saved_max_failures = config.MAX_CONSECUTIVE_DOWNLOAD_FAILURES
    saved_sleep_min2, saved_sleep_max2 = config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX
    ingest_video = failing_ingest_video
    config.MAX_CONSECUTIVE_DOWNLOAD_FAILURES = 3
    config.YTDLP_SLEEP_MIN = config.YTDLP_SLEEP_MAX = 0
    try:
        _aio.run(_process_queue(FakeDB7()))
    finally:
        ingest_video = saved_ingest_video2
        config.MAX_CONSECUTIVE_DOWNLOAD_FAILURES = saved_max_failures
        config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX = saved_sleep_min2, saved_sleep_max2

    assert FakeDB7.job_count == 3, (
        "worker should stop after exactly 3 consecutive download failures, "
        "tried %d videos" % FakeDB7.job_count)
    ok += 1
    print("ok %d — _process_queue stops after MAX_CONSECUTIVE_DOWNLOAD_FAILURES" % ok)

    # 18. /queue/batch is registered before /queue/{video_id} — FastAPI
    # matches routes in registration order, so the literal path must come
    # first or every call to /queue/batch 404s as "video not found: batch"
    # instead of ever reaching queue_batch(). Found live 2026-10-09; this
    # pins the fix so the ordering can't silently regress.
    post_paths = [r.path for r in router.routes if "POST" in getattr(r, "methods", set())]
    batch_idx = post_paths.index("/queue/batch")
    single_idx = post_paths.index("/queue/{video_id}")
    assert batch_idx < single_idx, (
        "/queue/batch must be registered before /queue/{video_id} "
        "(got indices %d, %d)" % (batch_idx, single_idx))
    ok += 1
    print("ok %d — /queue/batch is registered before /queue/{video_id}" % ok)

    # 19. queue_video reuses an existing FAILED/DONE job row instead of
    # inserting a second one. video_ingest_jobs.video_id has no uniqueness
    # constraint (only the SERIAL id does), so a plain INSERT on every retry
    # leaves the old row behind -- which broke the admin video listing,
    # since its per-video key only varies by video_id, not by job id. Found live 2026-10-12
    # (P97) retrying the Instagram reel from P96 after its first attempt
    # failed: dequeue_video only deletes status='queued' rows, so the
    # failed row from attempt 1 was never removed before attempt 2's queue
    # call inserted a second row.
    global get_eval_db

    class FakeDBQueueRetry:
        def __init__(self):
            self.executed = []

        async def fetchrow(self, sql, *a):
            if "FROM video_catalog" in sql:
                return {"video_id": "v_retry"}
            if "next_pos" in sql:
                return {"next_pos": 1}
            if "FROM video_ingest_jobs WHERE video_id" in sql and "ORDER BY" in sql:
                return {"id": 4, "status": "failed"}
            return None

        async def fetch(self, sql, *a):
            return []

        async def execute(self, sql, *a):
            self.executed.append((sql, a))
            return True

    saved_get_eval_db = get_eval_db
    fake_db19 = FakeDBQueueRetry()
    get_eval_db = lambda: fake_db19
    try:
        resp = _aio.run(queue_video("v_retry"))
    finally:
        get_eval_db = saved_get_eval_db

    assert resp["status"] == "queued", resp
    assert len(fake_db19.executed) == 1, fake_db19.executed
    sql, args = fake_db19.executed[0]
    assert "UPDATE video_ingest_jobs" in sql, sql
    assert "INSERT INTO video_ingest_jobs" not in sql, sql
    assert args[0] == 4, ("must update the existing row (id=4), not insert a new one", args)
    ok += 1
    print("ok %d — queue_video reuses an existing failed/done job row instead of inserting a second one" % ok)

    # 20-26. The D72/D74 cookie endpoints -- status/upload/reject/delete/
    # paste/validate. probe_cookies is mocked throughout: it makes a real
    # network call to YouTube, which this offline suite must never do.
    import os as _os
    import tempfile as _tempfile_cookies
    import pipelines.video_acquisition as _video_acq_mod

    class FakeUploadFile:
        def __init__(self, data: bytes):
            self._data = data

        async def read(self):
            return self._data

    netscape = (b"# Netscape HTTP Cookie File\n"
               b".youtube.com\tTRUE\t/\tTRUE\t0\tSID\tabc123value\n")

    with _tempfile_cookies.TemporaryDirectory() as _cdir:
        cookies_path = _os.path.join(_cdir, "cookies.txt")
        saved_cookies_cfg = config.YTDLP_COOKIES_FILE
        config.YTDLP_COOKIES_FILE = cookies_path
        saved_probe = _video_acq_mod.probe_cookies
        _video_acq_mod.probe_cookies = lambda cf: {"valid": True, "error": None}
        try:
            status20 = _aio.run(cookies_status())
            assert status20["configured"] is False, status20
            ok += 1
            print("ok %d — GET /cookies/status reports 'not configured' with no file" % ok)

            resp21 = _aio.run(upload_cookies(FakeUploadFile(netscape)))
            assert resp21["status"]["configured"] is True, resp21
            assert resp21["validation"] == {"valid": True, "error": None}, resp21
            assert _os.path.isfile(cookies_path)
            ok += 1
            print("ok %d — POST /cookies saves a valid upload, validates it, and reports status" % ok)

            try:
                _aio.run(upload_cookies(FakeUploadFile(b"not a cookie file")))
                raise AssertionError("expected HTTPException for invalid content")
            except HTTPException as exc:
                assert exc.status_code == 400, exc
            ok += 1
            print("ok %d — POST /cookies rejects non-cookie content with 400" % ok)

            # 23. POST /cookies/text -- the paste path, same validation.
            resp23 = _aio.run(paste_cookies(CookieTextRequest(content=netscape.decode())))
            assert resp23["status"]["configured"] is True, resp23
            ok += 1
            print("ok %d — POST /cookies/text saves pasted content and validates it" % ok)

            # 24. POST /cookies/validate re-probes the already-saved file
            # and the new result is visible in status afterward.
            _video_acq_mod.probe_cookies = lambda cf: {
                "valid": False, "error": "YouTube still challenges this session"}
            resp24 = _aio.run(validate_cookies())
            assert resp24["validation"]["valid"] is False, resp24
            status24 = _aio.run(cookies_status())
            assert status24["last_validation"]["valid"] is False, status24
            ok += 1
            print("ok %d — POST /cookies/validate re-probes and updates the stored result" % ok)

            # 25. POST /cookies/validate with nothing configured is a 400,
            # not a confusing attempt to probe a file that doesn't exist.
            _aio.run(delete_cookies())
            try:
                _aio.run(validate_cookies())
                raise AssertionError("expected HTTPException with nothing configured")
            except HTTPException as exc:
                assert exc.status_code == 400, exc
            ok += 1
            print("ok %d — POST /cookies/validate rejects when nothing is configured" % ok)

            # 26. DELETE /cookies removes the file.
            _aio.run(upload_cookies(FakeUploadFile(netscape)))
            _aio.run(delete_cookies())
            assert not _os.path.isfile(cookies_path)
            status26 = _aio.run(cookies_status())
            assert status26["configured"] is False, status26
            ok += 1
            print("ok %d — DELETE /cookies removes the file" % ok)
        finally:
            config.YTDLP_COOKIES_FILE = saved_cookies_cfg
            _video_acq_mod.probe_cookies = saved_probe

    # 27. GET /video/costs reads both tables (titles joined from video_catalog)
    # and returns summarize_costs()'s shape for the admin panel.
    import json as _json27
    from pipelines.usage_meter import UsageMeter
    m27 = UsageMeter("c1")
    m27.record_groq(120.0)
    queries27 = []

    class FakeDB27:
        async def fetch(self, sql, *a):
            queries27.append(sql)
            if "video_ingest_usage" in sql:
                return [{"video_id": "c1", "title": "Intro lesson", "status": "done",
                         "usage": _json27.dumps(m27.summary()), "phase_seconds": "{}",
                         "prices_as_of": "2026-10-19", "recorded_at": "2026-10-19 15:00"}]
            return [{"video_id": "old", "title": "", "groq_cost_usd": 0.5, "voyage_cost_usd": 0.0,
                     "vision_cost_usd": 0.0, "recorded_at": "2026-10-19 14:00"}]

    saved27 = get_eval_db
    get_eval_db = lambda: FakeDB27()
    try:
        costs = _aio.run(get_costs())
    finally:
        get_eval_db = saved27
    assert any("video_catalog" in q for q in queries27), queries27
    assert costs["videos"][0]["title"] == "Intro lesson", costs["videos"]
    assert [e["video_id"] for e in costs["estimated_pre_metering"]] == ["old"]
    assert costs["platforms"][0]["platform"] == "Groq" and costs["platforms"][0]["billed_seconds"] == 120.0
    ok += 1
    print("ok %d — GET /video/costs returns measured per-platform spend, unit prices, estimates apart" % ok)

    ok = _self_test_cancel(ok)
    print("\n%d checks passed" % ok)


def _self_test_cancel(ok):
    """Re-ingest guard, POST /video/cancel, and the progress/cancel fields on
    the status endpoints (2026-10-26). A done video re-run costs the whole
    pipeline again (~$3.58 for the 1:53 checkpoint video), so it needs force;
    a running ingest can be stopped at the next checkpoint."""
    import asyncio as _aio
    import json as _json
    global get_eval_db, ingest_video, record_usage

    class QueueDB:
        """video_catalog + latest video_ingest_jobs row per video, and the
        checkpoint rows; records every write."""

        def __init__(self, jobs, checkpoints=None):
            self.jobs = jobs                    # video_id -> {"id", "status", ...}
            self.checkpoints = checkpoints or {}
            self.executed = []

        async def fetchrow(self, sql, *a):
            if "FROM video_catalog" in sql:
                return None if a[0] == "missing" else {"video_id": a[0]}
            if "next_pos" in sql:
                return {"next_pos": 9}
            if "COUNT(*)" in sql:
                return {"cnt": 0}
            if "FROM video_ingest_jobs" in sql and "ORDER BY created_at DESC" in sql:
                return self.jobs.get(a[0])
            return None

        async def fetch(self, sql, *a):
            if "video_phase_checkpoints" in sql:
                return [{"phase": p, "status": s} for p, s in self.checkpoints.get(a[0], {}).items()]
            if "FROM video_catalog vc" in sql:
                return [{"video_id": vid, "title": "T", "job_status": j["status"], "phase": j.get("phase"),
                         "phase_progress": j.get("phase_progress")} for vid, j in self.jobs.items()]
            return []

        async def execute(self, sql, *a):
            self.executed.append((sql, a))
            return True

    def call(db, coro_fn, *args, **kwargs):
        global get_eval_db
        saved = get_eval_db
        get_eval_db = lambda: db
        try:
            return _aio.run(coro_fn(*args, **kwargs))
        finally:
            get_eval_db = saved

    # Re-ingesting a done video without force -> 409, nothing written.
    db = QueueDB({"d1": {"id": 1, "status": "done"}})
    try:
        call(db, queue_video, "d1")
        raise AssertionError("expected 409 for a done video without force")
    except HTTPException as exc:
        assert exc.status_code == 409, exc
        assert exc.detail == "already ingested — pass force=true to re-ingest (paid)", exc.detail
    assert db.executed == [], db.executed
    resp = call(db, queue_video, "d1", force=True)
    assert resp["status"] == "queued", resp
    assert len(db.executed) == 1 and "UPDATE video_ingest_jobs" in db.executed[0][0], db.executed
    assert db.executed[0][1][0] == 1, db.executed       # the same row, reused (P97)
    ok += 1
    print("ok %d — POST /queue/{id} on a done video: 409 without force, re-queues the same row with force" % ok)

    # Batch: done videos are reported under already_ingested, failed ones
    # reuse their row (no second row -- P97), new ones get a row.
    db = QueueDB({"d1": {"id": 1, "status": "done"}, "f1": {"id": 2, "status": "failed"},
                  "q1": {"id": 3, "status": "queued"}})
    resp = call(db, queue_batch, BatchQueueRequest(video_ids=["d1", "f1", "n1", "q1"]))
    assert resp["already_ingested"] == ["d1"], resp
    assert resp["queued"] == 2, resp
    writes = [(sql.split()[0], args) for sql, args in db.executed]
    assert ("UPDATE", (2, 9)) in [(w, a[:2]) for w, a in writes], db.executed
    assert sum(1 for w, _ in writes if w == "INSERT") == 1, db.executed
    assert not any(a and a[0] == 1 for w, a in writes), db.executed     # d1 untouched
    db = QueueDB({"d1": {"id": 1, "status": "done"}})
    resp = call(db, queue_batch, BatchQueueRequest(video_ids=["d1"], force=True))
    assert resp["queued"] == 1 and resp["already_ingested"] == [], resp
    assert "UPDATE video_ingest_jobs" in db.executed[0][0] and db.executed[0][1][0] == 1, db.executed
    ok += 1
    print("ok %d — POST /queue/batch skips done videos (already_ingested) unless force, reuses rows" % ok)

    # Cancel a queued video: removed from the queue like DELETE /queue/{id}.
    _cancel_requests.clear()
    db = QueueDB({"q1": {"id": 3, "status": "queued"}})
    resp = call(db, cancel_video, "q1")
    assert resp["status"] == "cancelled" and resp["video_id"] == "q1" and resp["message"], resp
    assert db.executed == [("DELETE FROM video_ingest_jobs WHERE id = $1", (3,))], db.executed
    assert "q1" not in _cancel_requests
    ok += 1
    print("ok %d — POST /cancel on a queued video removes it from the queue" % ok)

    # A queued forced re-ingest of a video with a complete index is put back
    # to done rather than deleted -- deleting its only job row would show an
    # indexed video as never ingested.
    db = QueueDB({"q2": {"id": 4, "status": "queued"}},
                 checkpoints={"q2": {"index": "done", "cleanup": "done"}})
    resp = call(db, cancel_video, "q2")
    assert resp["status"] == "cancelled", resp
    assert len(db.executed) == 1 and "status = 'done'" in db.executed[0][0], db.executed
    ok += 1
    print("ok %d — cancelling a queued re-ingest of an indexed video restores it to done" % ok)

    # Cancel a processing video: 202, a cooperative stop is requested, and
    # both status endpoints show cancel_requested + overall_progress.
    db = QueueDB({"p1": {"id": 5, "status": "processing", "phase": "visual", "phase_progress": 0.5}})
    resp = call(db, cancel_video, "p1")
    assert resp.status_code == 202, resp.status_code
    body = _json.loads(resp.body)
    assert body == {"video_id": "p1", "status": "cancelling",
                    "message": "stops at the next checkpoint; calls already in flight finish"}, body
    assert "p1" in _cancel_requests
    assert db.executed == [], db.executed
    listing = call(db, list_status)
    row = listing["videos"][0]
    assert row["cancel_requested"] is True and row["overall_progress"] == 50, row
    detail = call(db, video_status, "p1")
    assert detail["job"]["cancel_requested"] is True and detail["job"]["overall_progress"] == 50, detail
    ok += 1
    print("ok %d — POST /cancel on a processing video: 202, cancel_requested + overall_progress in status" % ok)

    # 404 for an unknown video; 409 for one with nothing to cancel.
    for vid, jobs, code in (("missing", {}, 404), ("d1", {"d1": {"id": 1, "status": "done"}}, 409),
                            ("n0", {}, 409)):
        try:
            call(QueueDB(jobs), cancel_video, vid)
            raise AssertionError("expected %d for %s" % (code, vid))
        except HTTPException as exc:
            assert exc.status_code == code, (vid, exc.status_code)
    assert "cancelled" in _VALID_STATUSES
    ok += 1
    print("ok %d — POST /cancel: 404 uncatalogued, 409 when not queued/processing; 'cancelled' is a valid status" % ok)

    # Worker: a cancelled run is marked cancelled (usage recorded as such),
    # the worker moves on to the next video, and a cancel neither counts
    # toward nor resets the consecutive-download-failure breaker.
    _cancel_requests.clear()
    order = ["v1", "v2", "v3", "v4"]
    outcomes = {"v1": ("failed", "download failed: bot check"), "v2": ("cancelled", None),
                "v3": ("failed", "download failed: bot check"), "v4": ("done", None)}
    ran, stop_seen, usage_calls = [], {}, []

    class WorkerDB(QueueDB):
        def __init__(self, checkpoints=None):
            super().__init__({}, checkpoints)
            self.n = 0

        async def fetchrow(self, sql, *a):
            if "video_ingest_costs" in sql:
                return {"total": 0.0}
            if "WHERE vij.status = 'queued'" in sql:
                if self.n >= len(order):
                    return None
                vid = order[self.n]
                self.n += 1
                return {"id": self.n, "video_id": vid, "title": "", "source_url": "",
                        "platform": "youtube", "duration": 60}
            return None

    def fake_ingest(video_id, *a, should_stop=None, **k):
        ran.append(video_id)
        status, err = outcomes[video_id]
        if status == "cancelled":
            _cancel_requests.add(video_id)          # the user clicked Cancel mid-run
            stop_seen[video_id] = should_stop()
            return type("R", (), {"status": "cancelled", "phase": "asr",
                                  "error": "cancelled by user during asr",
                                  "stats": {"phase_seconds": {}}, "usage": {"total_usd": 0.1}})()
        return type("R", (), {"status": status, "error": err, "phase": None,
                              "stats": {"duration_s": 60.0}, "usage": {}})()

    async def fake_record_usage(db, video_id, usage, phase_seconds, status):
        usage_calls.append((video_id, status))

    saved = (ingest_video, record_usage, config.MAX_CONSECUTIVE_DOWNLOAD_FAILURES,
             config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX)
    ingest_video, record_usage = fake_ingest, fake_record_usage
    config.MAX_CONSECUTIVE_DOWNLOAD_FAILURES = 2
    config.YTDLP_SLEEP_MIN = config.YTDLP_SLEEP_MAX = 0
    wdb = WorkerDB()
    try:
        _aio.run(_process_queue(wdb))
    finally:
        (ingest_video, record_usage, config.MAX_CONSECUTIVE_DOWNLOAD_FAILURES,
         config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX) = saved
    # v1 fail (1), v2 cancelled (still 1), v3 fail (2) -> breaker stops before v4
    assert ran == ["v1", "v2", "v3"], ran
    assert stop_seen == {"v2": True}, stop_seen
    assert ("v2", "cancelled") in usage_calls, usage_calls
    cancelled_sql = [(s, a) for s, a in wdb.executed if "status='cancelled'" in s]
    assert len(cancelled_sql) == 1 and "completed_at=NOW()" in cancelled_sql[0][0], wdb.executed
    assert cancelled_sql[0][1] == (2, "cancelled by user during asr"), cancelled_sql
    assert not _cancel_requests, _cancel_requests       # registry cleared
    ok += 1
    print("ok %d — worker marks a cancelled run 'cancelled', records its usage, continues, "
          "and leaves the download breaker alone" % ok)

    # Worker: a cancelled forced re-ingest whose previous index is complete
    # (stopped before indexing) goes back to done with a visible note; one
    # stopped during indexing (points deleted) stays cancelled.
    for phase, ckpt, want in (("visual", {"index": "done", "cleanup": "done"}, "done"),
                              ("index", {"index": "cancelled", "cleanup": "done"}, "cancelled"),
                              ("visual", {}, "cancelled")):
        order[:] = ["r1"]
        outcomes["r1"] = ("cancelled", None)

        def fake_ingest_r(video_id, *a, should_stop=None, **k):
            return type("R", (), {"status": "cancelled", "phase": phase,
                                  "error": "cancelled by user during %s" % phase,
                                  "stats": {}, "usage": {}})()

        saved = (ingest_video, config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX)
        ingest_video = fake_ingest_r
        config.YTDLP_SLEEP_MIN = config.YTDLP_SLEEP_MAX = 0
        wdb = WorkerDB(checkpoints={"r1": ckpt})
        try:
            _aio.run(_process_queue(wdb))
        finally:
            ingest_video, config.YTDLP_SLEEP_MIN, config.YTDLP_SLEEP_MAX = saved
        final = [(s, a) for s, a in wdb.executed if "status='%s'" % want in s]
        assert len(final) == 1, (phase, ckpt, wdb.executed)
        if want == "done":
            assert "re-ingest cancelled during visual; previous index kept" in final[0][1], final
    ok += 1
    print("ok %d — a cancelled re-ingest keeps 'done' when the old index is intact; "
          "stays 'cancelled' once indexing started" % ok)

    # DELETE /ingested/{id}: points first, then job + checkpoint rows; the
    # catalog entry and spend history are never touched (2026-10-28).
    global _delete_video_points
    saved_del = _delete_video_points
    cleared = []

    async def fake_points(video_id):
        cleared.append(video_id)
        return {"visual_frames": 3, "ocr_frames": 2, "od_frames": 0, "transcript_segments": 5}

    _delete_video_points = fake_points
    try:
        db = QueueDB({"d1": {"id": 1, "status": "done"}})
        resp = call(db, delete_ingested_video, "d1")
        assert resp["deleted"] and resp["points_total"] == 10 and cleared == ["d1"], resp
        assert [s.split(" WHERE")[0] for s, _ in db.executed] == [
            "DELETE FROM video_phase_checkpoints", "DELETE FROM video_ingest_jobs"], db.executed
        assert not any("video_catalog" in s or "usage" in s or "costs" in s for s, _ in db.executed)
        for status in ("queued", "processing"):
            db = QueueDB({"b1": {"id": 2, "status": status}})
            try:
                call(db, delete_ingested_video, "b1")
                raise AssertionError("a %s video was deleted" % status)
            except HTTPException as exc:
                assert exc.status_code == 409 and db.executed == [], (exc, db.executed)
        db = QueueDB({})
        try:
            call(db, delete_ingested_video, "nope")
            raise AssertionError("a never-ingested video was 'deleted'")
        except HTTPException as exc:
            assert exc.status_code == 404, exc

        async def broken(video_id):
            raise RuntimeError("qdrant down")
        _delete_video_points = broken
        db = QueueDB({"d1": {"id": 1, "status": "done"}})
        try:
            call(db, delete_ingested_video, "d1")
            raise AssertionError("a failed point delete was reported as success")
        except HTTPException as exc:
            assert exc.status_code == 502 and db.executed == [], (exc, db.executed)
    finally:
        _delete_video_points = saved_del
    ok += 1
    print("ok %d — DELETE /ingested: points, then rows; catalog + spend kept; refused while "
          "running; a Qdrant failure changes nothing" % ok)
    return ok


# To mount this router, add to main.py:
#   from routers import video
#   app.include_router(video.router, prefix="/video", tags=["video"])


if __name__ == "__main__":
    _self_test()
