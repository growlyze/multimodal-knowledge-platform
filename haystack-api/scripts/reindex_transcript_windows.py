"""
Re-index already-ingested videos' transcripts as ~60 s windows (P152).

Videos ingested before P152 have one Qdrant point per Whisper segment (~7 s),
which the reranker scores too low to ever pass the 0.70 floor. This rebuilds
the windows from the segments already stored in Qdrant -- no download, no
re-transcription, no Gemini -- re-embeds them (Voyage text, cents per video)
and replaces ONLY that video's transcript points. Frame/OCR/description
points are not touched.

    # dry run (read-only): what would change, per video
    docker compose exec -e PYTHONPATH=/app haystack-api python scripts/reindex_transcript_windows.py
    # apply to every video in the transcript collection, or to named ones
    docker compose exec -e PYTHONPATH=/app haystack-api python scripts/reindex_transcript_windows.py --apply
    docker compose exec -e PYTHONPATH=/app haystack-api python scripts/reindex_transcript_windows.py --apply <video_id>
    # skip regenerating the per-video overview (P152 stage B)
    docker compose exec -e PYTHONPATH=/app haystack-api python scripts/reindex_transcript_windows.py --apply --no-overview

Also (re)writes each video's overview point ({video_id}:overview) from the new
windows, so this one command leaves a video fully current. The overview point
is excluded when rebuilding windows (it isn't a transcript segment). The run's
Voyage + Gemini usage is metered and ADDED to the video's video_ingest_usage row
(cost_tracker.add_usage), so the admin panel's Ingestion Costs include it. Until
2026-10-21 it was only printed: the 2026-10-20 re-index (~$0.05 Gemini, Voyage
unmetered) is missing from the panel's totals.
PYTHONPATH=/app is required: Python puts only scripts/ on the path, so `import config`
fails without it (found on the first live run, 2026-10-20).
"""
import asyncio
import sys
from types import SimpleNamespace

import config
from pipelines.video_embedder import embed_texts_voyage
from evaluation.eval_db import get_eval_db
from pipelines import cost_tracker, usage_meter
from pipelines.video_ingestion import _index_video_overview, _upsert_to_qdrant, transcript_windows
from pipelines.video_topic_segmentation import segment_by_topic
from services.qdrant_service import QdrantService

COLL = config.VIDEO_TRANSCRIPT_COLLECTION


async def _scroll(qs, collection, video_id=None, fields=None):
    flt = {"must": [{"key": "video_id", "match": {"value": video_id}}]} if video_id else None
    points, offset = [], None
    while True:
        body = {"limit": 1000, "offset": offset, "with_vector": False,
                "with_payload": fields if fields else True}
        if flt:
            body["filter"] = flt
        r = await qs._request("POST", f"/collections/{collection}/points/scroll", json=body)
        points += r["points"]
        offset = r.get("next_page_offset")
        if offset is None:
            return points


async def main():
    apply = "--apply" in sys.argv
    overview = "--no-overview" not in sys.argv
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    qs = QdrantService()
    try:
        ids = wanted or sorted({p["payload"]["video_id"]
                                for p in await _scroll(qs, COLL, fields=["video_id"])})
        for vid in ids:
            pts = [p["payload"] for p in await _scroll(qs, COLL, vid) if not p["payload"].get("overview")]
            if not pts:
                print("%s: no transcript points, skipped" % vid)
                continue
            segs = sorted((SimpleNamespace(text=p.get("text") or "", start_time=p["start_time"],
                                           end_time=p.get("end_time", p["start_time"]),
                                           confidence=p.get("confidence") or 0.0,
                                           words=p.get("words") or [])
                           for p in pts), key=lambda s: s.start_time)
            wins = transcript_windows(segs)
            meta = pts[0]
            print("%s: %d stored points -> %d windows%s" % (vid, len(pts), len(wins), "" if apply else " (dry run)"))
            if not apply:
                continue
            # Metered from here: the window embeddings (Voyage) and the overview (Gemini).
            meter = usage_meter.start("reindex:" + vid)
            try:
                vectors = embed_texts_voyage([w.text for w in wins], batch_size=config.VOYAGE_BATCH_SIZE)
            except BaseException:
                usage_meter.stop()
                raise
            frames = await _scroll(qs, config.VIDEO_VISUAL_COLLECTION, vid, fields=["frame_ms"])
            topic_ids = segment_by_topic(wins, vectors,
                                         keyframe_timestamps_ms=[f["payload"].get("frame_ms") or 0 for f in frames])
            payloads = [{
                "video_id": vid, "title": meta.get("title", ""),
                "start_time": w.start_time, "end_time": w.end_time, "text": w.text,
                "confidence": w.confidence, "words": w.words, "topic_id": topic_ids[j],
                "source_type": "transcript_segment",
                "platform": meta.get("platform", "youtube"), "source_url": meta.get("source_url", ""),
            } for j, w in enumerate(wins)]
            await qs.delete_by_filter(COLL, {"video_id": vid})
            n = _upsert_to_qdrant(COLL, ["%s:asr:%d" % (vid, i) for i in range(len(wins))],
                                  vectors, payloads, config.VOYAGE_EMBEDDING_DIM)
            print("   replaced with %d window points" % n)
            try:
                if overview:
                    wrote = _index_video_overview(vid, meta.get("title", ""), wins,
                                                  platform=meta.get("platform", "youtube"),
                                                  source_url=meta.get("source_url", ""))
                    print("   overview: %s" % ("written" if wrote else "NOT written"))
            finally:
                usage_meter.stop()
            spent = meter.summary()
            spent["video_id"] = vid
            await cost_tracker.add_usage(get_eval_db(), vid, spent)
            print("   metered: Voyage %d text tokens, Gemini %d in / %d out tokens, $%.4f at list price"
                  " -- added to this video's ingestion cost"
                  % (spent["voyage"]["text_tokens"], spent["gemini"]["input_tokens"],
                     spent["gemini"]["output_tokens"] + spent["gemini"]["thought_tokens"],
                     spent["total_usd"]))
    finally:
        await qs.aclose()


if __name__ == "__main__":
    asyncio.run(main())
