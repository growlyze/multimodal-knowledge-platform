"""
/ingest/archive — ingest a compressed upload member by member.

One job per archive. The archive is unpacked by extractors/archive.py (the
same bounded, traversal-checked layer /extract uses), and every eligible
member goes through the exact pipeline its own type already has:

    structured member  -> pipelines/structured  (pandas -> PostgreSQL + summary)
    document member    -> pipelines/ingestion    (chunk -> embed -> Qdrant)
    code member        -> pipelines/ingestion    (code chunker -> Qdrant)

Media members, nested archives, empty files and unsupported types are skipped
with a per-member reason in the job's quality report — never silently dropped.
A member that fails does not fail its siblings; the job itself only fails when
the archive cannot be read at all (or nothing in it could be processed).

Member identity is `archive:member` (e.g. "backup.zip:sales.csv") so two archives
carrying the same inner filename are two datasets, and re-uploading the same
archive replaces its own members — the same replace-on-reupload contract
direct uploads have. A single-stream .csv.gz is one file, not a container: its
identity is the file inside it, exactly like POST /ingest/structured.
"""

import logging

from extractors import archive
from extractors.router import classify_member
from pipelines import ingestion, structured

logger = logging.getLogger("haystack-api.archive-ingest")

# The quality report is read by humans polling a status endpoint. Counts are
# always exact; the name lists are trimmed so one 500-file archive cannot make
# the job row enormous.
_MAX_LISTED = 50


def _trim(entries: list) -> dict:
    return {"count": len(entries),
            "items": entries[:_MAX_LISTED],
            "truncated": len(entries) > _MAX_LISTED}


def _member_identity(archive_name: str, member: archive.Member) -> str:
    if member.single:
        return member.name
    return "%s:%s" % (archive_name, member.name)


async def run_archive_job(job_id: str, filename: str, file_bytes: bytes,
                          metadata: dict = None, db=None) -> dict:
    """The background half of POST /ingest/archive. Never raises."""
    store = ingestion.get_store()
    name = archive.basename(filename)
    report = {"kind": "archive",
              "format": archive.compression_suffix(name),
              "members_total": 0}
    indexed, failed, skipped = [], [], []

    await store.update(job_id, status="processing")
    try:
        members = archive.read_members(name, file_bytes)
    except archive.ArchiveError as exc:
        logger.warning("archive job %s refused: %s", job_id, exc)
        report["error"] = str(exc)
        await store.update(job_id, status="failed", error=str(exc),
                           quality=report)
        return {"job_id": job_id, "status": "failed", "error": str(exc)}

    report["members_total"] = len(members)
    chunks = 0
    for member in members:
        identity = _member_identity(name, member)
        if member.status != "ok":
            skipped.append({"name": identity, "reason": member.reason})
            continue
        kind, reason = classify_member(member.name, member.data)
        if kind is None:
            skipped.append({"name": identity, "reason": reason})
            continue
        member_meta = {**(metadata or {}), "archive": name, "member": member.name}
        try:
            if kind == "structured":
                result = await structured.ingest_structured(
                    identity, member.data, member_meta, db=db)
                if result.get("status") != "done":
                    raise RuntimeError(result.get("error") or "structured load failed")
                created = result.get("indexed", 0)
            else:
                text = await ingestion.extract(identity, member.data, kind)
                result = await ingestion.ingest_text(
                    text, identity, kind, metadata=member_meta)
                created = result["chunks_created"]
            chunks += created
            entry = {"name": identity, "kind": kind, "indexed": created}
            # A dedup skip (pipelines/dedup.py) indexes 0 on purpose; without
            # the reason an all-duplicate archive reads "done, 0 chunks".
            if result.get("skipped"):
                entry["skipped"] = result["skipped"]
                if result.get("duplicate_of"):
                    entry["duplicate_of"] = result["duplicate_of"]
            indexed.append(entry)
        except Exception as exc:
            logger.warning("archive member %s failed: %s", identity, exc)
            failed.append({"name": identity,
                           "error": "%s: %s" % (type(exc).__name__, exc)})

    report.update(indexed=_trim(indexed), skipped=_trim(skipped),
                  failed=_trim(failed), chunks_created=chunks)
    attempted = len(indexed) + len(failed)

    if attempted and not indexed:
        error = "no file in the archive could be processed (%s)" % (
            failed[0]["error"] if failed else "nothing eligible")
        logger.warning("archive job %s failed: %s", job_id, error)
        report["error"] = error
        await store.update(job_id, status="failed", error=error,
                           chunks_created=chunks, quality=report)
        return {"job_id": job_id, "status": "failed", "error": error}

    await store.update(job_id, status="done", chunks_created=chunks,
                       quality=report)
    logger.info("archive job %s done: %d of %d file(s) indexed, %d chunk(s)",
                job_id, len(indexed), len(members), chunks)
    return {"job_id": job_id, "status": "done", "chunks_created": chunks,
            "quality": report}
