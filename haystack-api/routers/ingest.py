"""
/ingest — async ingestion endpoints.

Called by the Open WebUI filter (fire and forget) and by the admin panel.
Every POST does the same three things: read the upload, register a job, hand the
work to BackgroundTasks. The response comes back before any chunking, embedding
or Qdrant traffic happens, so the chat is never blocked by ingestion.

    POST /ingest/document    PDF, DOCX, TXT, MD  (Tika via the extractors)
    POST /ingest/code        source files
    POST /ingest/transcript  pre-transcribed text with timestamps
    POST /ingest/media       images / audio / video
    POST /ingest/structured  CSV, TSV, XLSX, JSON, Parquet (pandas -> PostgreSQL)
    POST /ingest/archive     ZIP / GZ / tar: unpacked safely, each file inside
                             is ingested through the pipeline its type already has
    GET  /ingest/status       recent jobs (drives the admin panel's Jobs page)
    GET  /ingest/status/{id}  job status and quality report

Every POST accepts either the admin API key or the external loader key
(services/auth.py): the admin panel and the chat filter both ingest here.
"""

import asyncio
import json
import logging
import re

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

import config
from extractors import archive
from pipelines import archive_ingestion, ingestion, structured
from services.auth import require_admin_or_loader

logger = logging.getLogger("haystack-api.ingest")
router = APIRouter()


# Architecture decision #8: chat attachments are indexed, chat text is NOT — our own
# answers fed back in start outranking the source material. The filter stopped sending
# these in v1.1 (rag_filter 70a7f6b), but a deployed filter lives in Open WebUI's
# database rather than in this repo, so the refusal belongs here, at the one door every
# /ingest route goes through. Attachments are unaffected: they never carry either mark.
_CONVERSATION_FILE = re.compile(r"^conversation_.*_\d+\.txt$", re.I)

# The states JobStore ever writes. Validated on the way in so a typo in a filter
# comes back as a 400 rather than an empty list that reads as "nothing failed".
# `deleted` is written by DELETE /admin/documents (P123), not by a job itself.
JOB_STATES = frozenset({"queued", "processing", "done", "failed", "deleted"})


def is_conversation(filename: str, source_type: str) -> bool:
    return source_type == "conversation" or bool(_CONVERSATION_FILE.match(filename or ""))


assert is_conversation("conversation_4cc31daa_2.txt", "document")
assert is_conversation("whatever.txt", "conversation")
assert not is_conversation("report.pdf", "document")     # chat attachments still index
assert not is_conversation("conversation_notes.txt", "document")   # a real user file


async def _queue(background_tasks: BackgroundTasks, file: UploadFile, source_type: str,
                 collection: str = None, metadata: str = None) -> dict:
    """Shared body of every POST endpoint."""
    filename = file.filename or "unknown"
    if is_conversation(filename, source_type):
        logger.warning("refused conversation ingest: %s (source_type=%s) — "
                       "chat text is not indexed", filename, source_type)
        raise HTTPException(
            status_code=400,
            detail="chat text is not indexed (architecture decision #8); "
                   "chat file attachments are")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="%s is empty" % filename)

    extra = _parse_metadata(metadata)
    job = await ingestion.create_job(filename, source_type, collection)
    background_tasks.add_task(
        ingestion.run_job, job["job_id"], filename, file_bytes,
        source_type, job["collection"], extra,
    )
    return {**job, "bytes": len(file_bytes),
            "message": "Queued for ingestion. Searchable shortly."}


def _parse_metadata(raw: str) -> dict:
    """Optional JSON blob of extra payload fields (video_id, page, author, ...)."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="metadata is not valid JSON: %s" % exc)
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    return parsed


@router.post("/document", dependencies=[Depends(require_admin_or_loader)])
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_type: str = Form("document"),
    collection: str = Form(None),
    metadata: str = Form(None),
):
    """PDF, DOCX, TXT, Markdown. Recursive chunking, 400-500 tokens."""
    return await _queue(background_tasks, file, source_type, collection, metadata)


@router.post("/code", dependencies=[Depends(require_admin_or_loader)])
async def ingest_code(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_type: str = Form("code"),
    collection: str = Form(None),
    metadata: str = Form(None),
):
    """Source code and docs. Function-boundary chunking, 200-400 tokens.
    Defaults to the code_docs collection."""
    return await _queue(background_tasks, file, source_type, collection, metadata)


@router.post("/transcript", dependencies=[Depends(require_admin_or_loader)])
async def ingest_transcript(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_type: str = Form("transcript"),
    collection: str = Form(None),
    metadata: str = Form(None),
):
    """Pre-transcribed text with timestamps. Semantic chunking, timestamps kept.

    Accepts `[HH:MM:SS] text`, `MM:SS text`, SRT and WebVTT cue lines.
    """
    return await _queue(background_tasks, file, source_type, collection, metadata)


@router.post("/media", dependencies=[Depends(require_admin_or_loader)])
async def ingest_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_type: str = Form("media"),
    collection: str = Form(None),
    metadata: str = Form(None),
):
    """Images, audio, video.

    ponytail: the description comes from the existing media extractor, which is
    a placeholder — there is no vision provider configured yet. Everything after
    that (chunk, embed, upsert, status) is the real path, so wiring a vision
    call in later is a change to extraction only.
    """
    return await _queue(background_tasks, file, source_type, collection, metadata)


@router.post("/structured", dependencies=[Depends(require_admin_or_loader)])
async def ingest_structured(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_type: str = Form("structured_data"),
    metadata: str = Form(None),
):
    """CSV / TSV / XLSX / JSON / Parquet -> PostgreSQL, plus one summary in Qdrant.

    The work is pipelines/structured.py's; this is the same queue-and-return
    shape as every other endpoint here, so the filter's fire-and-forget POST and
    the admin panel's upload both get a job_id back and neither waits on pandas.

    Two differences from _queue(), which is why this does not use it:

    * There is no collection to choose. A data file is rows in PostgreSQL and a
      single summary chunk, and the summary always goes to the documents
      collection — so the form field is not offered rather than ignored.
    * The extension is checked before the job is created. Everything else here
      discovers a bad file inside the background task, where the only trace is a
      failed job; an unsupported extension is knowable now, and a 400 says so
      while the caller is still listening.
    """
    filename = file.filename or "unknown"
    if is_conversation(filename, source_type):
        logger.warning("refused conversation ingest: %s (source_type=%s) — "
                       "chat text is not indexed", filename, source_type)
        raise HTTPException(
            status_code=400,
            detail="chat text is not indexed (architecture decision #8); "
                   "chat file attachments are")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="%s is empty" % filename)

    try:
        fmt = structured.detect_format(filename)
    except structured.StructuredError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    extra = _parse_metadata(metadata)
    job = await ingestion.create_job(filename, structured.SOURCE_TYPE,
                                     config.COLLECTION_DOCUMENTS)
    background_tasks.add_task(structured.run_structured_job, job["job_id"],
                              filename, file_bytes, extra)
    return {**job, "bytes": len(file_bytes), "format": fmt,
            "message": "Queued. Loading into the database; searchable shortly."}


@router.post("/archive", dependencies=[Depends(require_admin_or_loader)])
async def ingest_archive(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form(None),
):
    """ZIP / GZ / tar upload -> every eligible file inside it, ingested by type.

    Same queue-and-return shape as every other endpoint here. The unpacking is
    extractors/archive.py's bounded, traversal-checked layer - the archive is
    never handed to Tika, which decompresses without a ceiling - and each
    member then goes through the pipeline its own extension already has
    (structured -> PostgreSQL, document/code -> chunk+embed). Unsupported
    members are skipped with a reason in the job's quality report.

    The extension is checked before the job exists, like /structured: a
    non-compressed file here is knowable now, and a 400 says so while the
    caller is still listening.
    """
    filename = file.filename or "unknown"
    if is_conversation(filename, "archive"):
        logger.warning("refused conversation ingest: %s (source_type=archive) — "
                       "chat text is not indexed", filename)
        raise HTTPException(
            status_code=400,
            detail="chat text is not indexed (architecture decision #8); "
                   "chat file attachments are")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="%s is empty" % filename)

    if archive.compression_suffix(filename) is None:
        raise HTTPException(
            status_code=400,
            detail="%s is not a compressed file - upload it to /ingest/<its "
                   "type> instead" % filename)

    extra = _parse_metadata(metadata)
    job = await ingestion.create_job(filename, "archive", config.COLLECTION_DOCUMENTS)
    background_tasks.add_task(archive_ingestion.run_archive_job, job["job_id"],
                              filename, file_bytes, extra)
    return {**job, "bytes": len(file_bytes),
            "message": "Queued for ingestion. Unpacking; each file inside is "
                       "indexed on its own."}


@router.get("/status")
async def ingestion_jobs(limit: int = ingestion.DEFAULT_JOB_LIST, status: str = None):
    """Recent ingestion jobs, newest first — the admin panel's Jobs page.

    A separate route from /status/{job_id} rather than one with an optional
    segment, because they answer different questions: "how is my upload doing"
    and "what has this service been asked to do lately".

    The panel has always called this and fallen back to the uploads started in
    the current browser session when it 404s. That fallback is why a CSV
    uploaded in chat never appeared there — the filter posts it from inside
    Open WebUI, so no panel session is holding its job id, and a failed
    ingestion was indistinguishable from one that never happened.
    """
    if status and status not in JOB_STATES:
        raise HTTPException(
            status_code=400,
            detail="status must be one of: %s" % ", ".join(sorted(JOB_STATES)))
    jobs = await ingestion.get_store().list(limit=limit, status=status)
    return {"jobs": jobs, "count": len(jobs)}


@router.get("/status/{job_id}")
async def ingestion_status(job_id: str):
    """Job status: queued, processing, done or failed.

    A finished job carries chunks_created and the Layer 1 quality report.
    """
    job = await ingestion.get_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such ingestion job: %s" % job_id)
    return job

