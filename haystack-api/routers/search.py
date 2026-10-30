"""
/search — Retrieval endpoint.

Called by the Open WebUI filter before every LLM call.
Returns relevant chunks with citations.

Every query is logged here rather than in the filter. The filter runs inside Open
WebUI's container and cannot import `evaluation` — it ships in this one — so this
route is the only place that sees a query *and* can reach the eval database. The
answer is the other half and only exists in the filter, which posts it back to
POST /eval/log-answer once the model has replied.
"""

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from evaluation.coverage_analyzer import CoverageAnalyzer
from evaluation.eval_db import get_eval_db
from evaluation.query_logger import QueryLogger
from evaluation.system_metrics import get_metrics
from pipelines import retrieval, retrieval_cost, usage_meter
from pipelines.retrieval import parse_timestamp
from services.citation_formatter import format_citation, format_sources_list

logger = logging.getLogger("haystack-api.search")
router = APIRouter()


class SearchResult(BaseModel):
    content: str
    source: str
    source_type: str  # "transcript", "document", "ocr", "visual", "code"
    score: float
    # Ready to paste into an answer: "[Quarterly Report — Page 4](url)".
    # Formatted here rather than in the filter, which cannot import this package.
    citation: str = ""
    # The namespaced source object (url, title, page_number, ...) also stays in
    # `metadata["source"]`, so API consumers get the rich fields either way.
    metadata: dict = {}


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    collections_searched: List[str]
    # Pre-formatted "Sources:" block over the deduplicated results.
    sources: str = ""
    # The eval_query_log row this search created. The filter hands it back to
    # POST /eval/log-answer so the answer score links to the query that caused it.
    log_id: Optional[str] = None


# Payload keys promoted to top-level SearchResult fields; the rest become metadata,
# so video timestamps and anything INGESTION adds later flow through untouched.
# "source" is deliberately not promoted — the object has to reach the caller whole.
_PROMOTED = {"content", "source_type"}


def _payload(hit: dict) -> dict:
    """The chunk payload, with the collection filled in for the formatter."""
    payload = hit.get("payload") or {}
    return {**payload, "collection": payload.get("collection") or hit.get("collection")}


def _display_source(payload: dict) -> str:
    """The plain string callers have always had under `source`."""
    source = payload.get("source")
    if isinstance(source, dict):
        return str(source.get("original_filename") or source.get("filename")
                   or source.get("title") or source.get("url") or "unknown")
    return str(source or "unknown")


# P105: document chunks carry their text under "content"; M3's video
# payloads never do -- a transcript_segment uses "text", an ocr_frame uses
# "ocr_text", an od_frame uses "od_detections" (see retrieval.py's
# _RERANK_TEXT_KEYS, which already knows this for reranking -- this endpoint
# didn't). Without this fallback, _to_result() always returned "" for every
# video hit: rag_filter.py's <context> block got a correct citation LINK and
# an empty passage body, so the model had a source to point at but nothing
# to ground an answer in. A visual_frame hit has no text at all -- that's a
# real empty passage (pure image similarity), not a bug.
_TEXT_FALLBACK_KEYS = ("content", "text", "ocr_text", "od_detections")


def _text(payload: dict) -> str:
    for key in _TEXT_FALLBACK_KEYS:
        value = payload.get(key)
        if value:
            return value
    return ""


def _to_result(hit: dict) -> SearchResult:
    payload = _payload(hit)
    return SearchResult(
        content=_text(payload),
        source=_display_source(payload),
        source_type=payload.get("source_type") or "document",
        score=round(float(hit.get("score", 0.0)), 6),
        citation=format_citation(payload),
        metadata={
            **{k: v for k, v in payload.items() if k not in _PROMOTED},
            "collection": hit.get("collection"),
            "chunk_id": hit.get("id"),
        },
    )


async def _log(query: str, results: list, collections: list, latency_ms: int,
               user_id: Optional[str], had_code_execution: bool,
               had_file_attachment: bool,
               in_scope: bool = True) -> Optional[str]:
    """Record the query for the evaluation dashboard. Returns its log id.

    Safe on the hot path: EvalDB degrades to a no-op with a cooldown when
    PostgreSQL is unreachable, so a dead eval database costs a log line a minute,
    not a slow chat. The catch-all is belt and braces on top of that — search must
    behave identically whether or not evaluation is working.
    """
    try:
        metrics = get_metrics()
        metrics.record_query()
        metrics.record_search_latency(latency_ms)
        if not results:
            metrics.record_empty_result()

        # QueryLogger hands back an id whether or not the row was actually written.
        # Without this guard a dead eval database still yields a log_id, the filter
        # posts an answer against it, and every sampled answer buys an LLM judge call
        # whose verdict has no query row to attach to. Cheap check: pool() short
        # circuits on both the ready and the cooling-down paths.
        if await get_eval_db().pool() is None:
            return None

        log_id = await QueryLogger().log_query(
            query=query,
            results_count=len(results),
            # max(), not results[0]: correct whatever order the reranker returns.
            top_score=max((r.score for r in results), default=None),
            collections_searched=collections,
            latency_ms=latency_ms,
            user_id=user_id,
            had_code_execution=had_code_execution,
            had_file_attachment=had_file_attachment,
        )
        # An empty result is the coverage signal — a question the corpus can't
        # answer. Only when it is a question the corpus was SUPPOSED to answer,
        # though: rag_filter now answers general questions from the model's own
        # knowledge instead of refusing, and every one of those still misses
        # here. Filing them turns the top_gaps panel, which is a list of things
        # to go ingest, into a list of off-topic questions nobody will ever
        # ingest for. The filter is the only side that knows, so it tells us.
        if not results and in_scope:
            await CoverageAnalyzer().record_empty_query(query)
        return log_id
    except Exception:
        logger.warning("eval logging failed — search unaffected", exc_info=True)
        return None


@router.get("", response_model=SearchResponse)
async def search_knowledge(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
    collections: Optional[str] = Query(
        None, description="Comma-separated collection names. Default: search all."
    ),
    source_type: Optional[str] = Query(
        None, description="Filter by source type (transcript, document, etc.)"
    ),
    # Context only the caller knows. Optional so /search stays usable by hand and
    # by external tools; they simply log as an anonymous query with no flags set.
    user_id: Optional[str] = Query(None, description="Open WebUI user id, for the query log"),
    had_code_execution: bool = Query(False, description="Query was routed to code execution"),
    had_file_attachment: bool = Query(False, description="Query arrived with a data file"),
    in_scope: bool = Query(
        True,
        description="Whether the corpus is expected to cover this question at "
                    "all. False suppresses the coverage-gap record on a miss.",
    ),
):
    """
    Search the knowledge base using hybrid retrieval + reranking.

    Pipeline:
    1. Embed query with configured embedding model
    2. Dense search on Qdrant (top RETRIEVAL_TOP_K per collection)
    3. BM25 sparse search on Qdrant (same)
    4. RRF fusion -> merged top 20
    5. Rerank with cross-encoder -> at most top_k results
    6. Drop anything the reranker scored below MIN_RERANK_SCORE — `results` and
       the `sources` block are built from the same survivors, so nothing is ever
       cited that was not retrieved. An empty list is a valid answer.
    7. Log the query for evaluation, and return with citations
    """
    explicit = (
        [c.strip() for c in collections.split(",") if c.strip()]
        if collections
        else None
    )
    # A copy: resolve_collections() returns the module-level lists themselves,
    # and the timestamp branch below appends to this one. Without list(), one
    # source_type=document query naming a time permanently added the video
    # collections to retrieval.DEFAULT_COLLECTIONS for the whole process.
    search_collections = list(retrieval.resolve_collections(
        source_type=source_type, explicit=explicit,
    ))
    # source_type is used for collection routing above; only pass it as a
    # payload filter when it names a specific document-level type, not a
    # collection-routing category like "video" or "document".
    _routing_types = {"video", "document"}
    filters = ({"source_type": source_type}
               if source_type and source_type not in _routing_types
               else None)

    # Timestamp queries ("at 15:30", "1:22:45") add video transcript collection
    # and a time-range filter so the search prioritises the matching segment.
    target_ts = parse_timestamp(q)
    if target_ts is not None and not explicit:
        video_cols = retrieval.VIDEO_COLLECTIONS
        for vc in video_cols:
            if vc not in search_collections:
                search_collections.append(vc)
        ts_window = 30
        filters = {
            **(filters or {}),
            "start_time": {"$lte": target_ts + ts_window},
            "end_time": {"$gte": target_ts - ts_window},
        }

    started = time.perf_counter()
    # P161: every paid call this search makes (Voyage query embed, video
    # embed_query, rerank, Gemini planner) records into this one meter. It is
    # a ContextVar scope, not a global, because searches run concurrently;
    # recording is in-memory, and the row is written by a background task
    # (retrieval_cost.schedule), so metering adds nothing to the latency.
    with usage_meter.search_scope() as meter:
        try:
            hits = await retrieval.search(q, top_n=top_k, collections=search_collections,
                                          filters=filters)
        except Exception as exc:
            # An empty result set reads as "nothing relevant" — a failed embedding
            # call must not look like that. Surface it. Whatever was already
            # spent before the failure is still stored.
            get_metrics().record_error()
            logger.exception("search failed for '%s'", q[:80])
            retrieval_cost.schedule(meter, None, user_id)
            raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000)

    results = [_to_result(h) for h in hits]
    log_id = await _log(q, results, search_collections, latency_ms,
                        user_id, had_code_execution, had_file_attachment,
                        in_scope)
    retrieval_cost.schedule(meter, log_id, user_id)

    return SearchResponse(
        query=q,
        results=results,
        collections_searched=search_collections,
        sources=format_sources_list([_payload(h) for h in hits]),
        log_id=log_id,
    )
