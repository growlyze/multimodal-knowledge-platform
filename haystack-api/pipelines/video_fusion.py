"""
Multi-track video moment fusion retrieval for M3.

Pipeline:
  visual (CLIP) ──┐
  OCR  (text)  ──┤
  OD   (objs)  ──┤ → WRRF → top-k merged moments
  ASR  (text)  ──┘

WRRF (Weighted Reciprocal Rank Fusion):
  score(d) = Σ  w_i / (RRF_K + rank_i)
  summed over every track i where document d appears.
  rank_i is 1-indexed; w_i are per-track weights.

Default weights: equal, 0.25 each (P145/D80, 2026-10-19 -- supersedes D59's
clip=0.25, ocr=0.30, od=0.15, asr=0.30). Track ids never coincide, so fusion
interleaves tracks by weight rather than merging moments; unequal weights let
the heaviest tracks take every slot. Equal weights hand the reranker a mix and
let it choose. Measured on 10 real client questions: D59 4/10, equal 8/10.

Four Qdrant collections (payloads carry at minimum frame_ts/start_time and video_id):
  visual_frames        — CLIP embeddings of video frames
  ocr_frames           — text embeddings of OCR-extracted frame text
  od_frames            — dense embeddings + pattern_type metadata for visual (OD) frame descriptions
  transcript_segments  — Voyage embeddings of ASR transcript segments
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

try:
    from haystack.core.component import component
    from haystack.dataclasses import Document
except ImportError:
    # Haystack not installed (dev machine, CI without full deps). Provide
    # enough for the self-test and type hints. Any real usage (the VPS
    # container) has Haystack; this fallback only matters offline.
    from dataclasses import dataclass as _dc

    def component(cls):
        """No-op decorator when Haystack is not installed."""
        return cls

    class _OutputTypes:
        def __call__(self, **kwargs):
            def _deco(fn):
                return fn
            return _deco
    component.output_types = _OutputTypes()

    @_dc
    class Document:
        id: str = ""
        content: str = ""
        meta: dict = None
        def __post_init__(self):
            if self.meta is None:
                self.meta = {}

logger = logging.getLogger("haystack-api.video_fusion")

# Same constant as the text-search RRF in pipelines/retrieval.py.
# Changing it here without matching there produces inconsistent score scales
# across search modes. The value 60 is the de-facto standard in the literature.
RRF_K = 60

_TRACK_VISUAL = "visual"
_TRACK_OCR = "ocr"
_TRACK_OD = "od"
_TRACK_ASR = "asr"


@component
class MultiTrackFusionRetriever:
    """Merge ranked result lists from four video retrieval tracks into one.

    Implements Weighted Reciprocal Rank Fusion (WRRF):
        score(d) = Σ  w_i / (RRF_K + rank_i)
    summed over every track where d appears (1-indexed ranks).

    Inputs
    ------
    visual_results : List[Document]  — ranked CLIP hits
    ocr_results    : List[Document]  — ranked OCR text hits
    od_results     : List[Document]  — ranked OD metadata hits
    asr_results    : List[Document]  — ranked ASR transcript segment hits

    Output
    ------
    merged_results : List[Document]  — up to top_k documents, sorted by wrrf_score
        Each merged document's meta carries:
          frame_ts          int   milliseconds since video start (frame tracks)
          start_time        float seconds (ASR track)
          video_id          str   source video identifier
          timestamp_ms      int   alias of frame_ts for downstream convenience
          retrieval_sources list  which tracks contributed (["visual", "ocr", ...])
          ocr_text          str   OCR content (from ocr_frames track, if present)
          od_detections     Any   OD payload field (from od_frames track, if present)
          asr_text          str   transcript text (from ASR track, if present)
          wrrf_score        float fused score

    Dedup key
    ---------
    Uses Document.id when set (Haystack stores Qdrant's UUID there). Falls back to
    "{video_id}_{frame_ts}" composite so frames arriving without an explicit id
    from both CLIP and OCR tracks still collapse to a single merged document.
    ASR segments use "{video_id}_{start_time}" since they carry start_time, not frame_ts.
    """

    def __init__(
        self,
        w_clip: float = 0.25,
        w_ocr: float = 0.25,
        w_od: float = 0.25,
        w_asr: float = 0.25,
        top_k: int = 10,
    ) -> None:
        total = w_clip + w_ocr + w_od + w_asr
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Track weights must sum to 1.0; got {total:.6f}. "
                "Non-unit weights shift the absolute score range and make "
                "threshold comparisons across queries meaningless."
            )
        self.weights = {
            _TRACK_VISUAL: w_clip,
            _TRACK_OCR: w_ocr,
            _TRACK_OD: w_od,
            _TRACK_ASR: w_asr,
        }
        self.top_k = top_k

    @component.output_types(merged_results=List[Document])
    def run(
        self,
        visual_results: List[Document],
        ocr_results: List[Document],
        od_results: List[Document],
        asr_results: Optional[List[Document]] = None,
    ) -> Dict[str, List[Document]]:
        track_lists = {
            _TRACK_VISUAL: visual_results or [],
            _TRACK_OCR: ocr_results or [],
            _TRACK_OD: od_results or [],
            _TRACK_ASR: asr_results or [],
        }

        # Accumulators keyed by dedup key
        wrrf_scores: Dict[str, float] = {}
        canonical: Dict[str, Document] = {}       # first-seen doc per key
        sources: Dict[str, List[str]] = {}         # which tracks contributed
        ocr_texts: Dict[str, str] = {}             # harvested from ocr track
        od_det: Dict[str, Any] = {}                # harvested from od track
        asr_texts: Dict[str, str] = {}             # harvested from asr track

        for track, docs in track_lists.items():
            weight = self.weights[track]
            for rank_0, doc in enumerate(docs):
                # rank_i is 1-indexed in the WRRF formula.
                rank_1 = rank_0 + 1
                key = self._dedup_key(doc)

                wrrf_scores[key] = wrrf_scores.get(key, 0.0) + weight / (RRF_K + rank_1)

                if key not in canonical:
                    canonical[key] = doc
                    sources[key] = []
                if track not in sources[key]:
                    sources[key].append(track)

                # Harvest track-specific payload: only the track that carries
                # the field is authoritative for it. Overwriting with a later
                # rank from the same track would degrade quality.
                if track == _TRACK_OCR and key not in ocr_texts:
                    text = doc.meta.get("ocr_text") or doc.content or ""
                    if text:
                        ocr_texts[key] = text
                elif track == _TRACK_OD and key not in od_det:
                    detections = doc.meta.get("od_detections")
                    if detections is not None:
                        od_det[key] = detections
                elif track == _TRACK_ASR and key not in asr_texts:
                    text = doc.meta.get("text") or doc.content or ""
                    if text:
                        asr_texts[key] = text

        # Sort descending by score then slice.
        ordered_keys = sorted(wrrf_scores, key=wrrf_scores.__getitem__, reverse=True)

        merged: List[Document] = []
        for key in ordered_keys[: self.top_k]:
            base = canonical[key]
            meta = dict(base.meta)
            frame_ts = meta.get("frame_ts")
            meta.update(
                {
                    "wrrf_score": round(wrrf_scores[key], 6),
                    "retrieval_sources": sources[key],
                    "retrieval_source": ",".join(sources[key]),  # flat string for span attrs
                    "ocr_text": ocr_texts.get(key, meta.get("ocr_text", "")),
                    "od_detections": od_det.get(key, meta.get("od_detections")),
                    "asr_text": asr_texts.get(key, meta.get("text", "")),
                    "frame_ts": frame_ts,
                    "timestamp_ms": frame_ts,  # alias — downstream expects this name
                    "video_id": meta.get("video_id"),
                }
            )
            merged.append(
                Document(
                    id=base.id,
                    content=base.content,
                    meta=meta,
                )
            )

        logger.debug(
            "WRRF fusion: visual=%d ocr=%d od=%d asr=%d → %d merged (top_k=%d, weights=%s)",
            len(visual_results),
            len(ocr_results),
            len(od_results),
            len(asr_results or []),
            len(merged),
            self.top_k,
            {k: round(v, 2) for k, v in self.weights.items()},
        )
        return {"merged_results": merged}

    @staticmethod
    def _dedup_key(doc: Document) -> str:
        """Stable identity across tracks.

        Prefers Document.id (populated from Qdrant's point UUID by Haystack).
        Falls back to composite key: frame tracks use frame_ts, ASR uses
        start_time. Both shapes collapse correctly when the same moment
        arrives from multiple tracks.
        """
        if doc.id:
            return doc.id
        video_id = doc.meta.get("video_id", "")
        ts = doc.meta.get("frame_ts") or doc.meta.get("start_time", "")
        return f"{video_id}_{ts}"


class MomentSearchRetriever:
    """Four-track Qdrant retrieval + WRRF fusion for video moment search.

    Queries all four collections via services/qdrant_service.py (this
    project's own httpx-based Qdrant client — see that module's docstring)
    concurrently via asyncio.gather, then feeds the results into
    MultiTrackFusionRetriever. Rewritten 2026-10-12: the original
    implementation depended on haystack_integrations.document_stores.qdrant,
    which this project doesn't install anywhere (requirements.txt comments
    it out: "using httpx directly, not qdrant-client") — it would have
    raised ImportError the instant anything actually called it in production.

    The visual collection is queried by the Voyage multimodal embedding
    (D66). OCR, OD and ASR collections are queried by Voyage text embedding.
    OD accepts an optional filter on the `pattern_type` payload field, using
    the same filter shape services/qdrant_service.py's build_filter() expects
    everywhere else in this codebase.

    Constructor
    -----------
    clip_collection_name    str
    ocr_collection_name     str
    od_collection_name      str
    asr_collection_name     str
    top_k                   int  candidates per track (default 20)
    fusion_weights          dict optional override: {"visual": 0.25, "ocr": 0.25, "od": 0.25, "asr": 0.25}
    fusion_top_k            int  documents returned after fusion (default 10)

    run() inputs
    ------------
    query_embedding      List[float]  Voyage-multimodal-space vector (visual track, D66)
    ocr_query_embedding  List[float]  Voyage text embedding (OCR, OD and ASR tracks)
    od_pattern_type      str | None   optional filter on od_frames payload field

    run() output
    ------------
    {"documents": List[Document]} where each Document.meta carries:
        timestamp_ms, video_id, ocr_text, od_detections, asr_text,
        retrieval_sources, retrieval_source, wrrf_score, frame_ts
    """

    def __init__(
        self,
        clip_collection_name: str,
        ocr_collection_name: str,
        od_collection_name: str,
        asr_collection_name: str,
        top_k: int = 20,
        fusion_weights: Optional[Dict[str, float]] = None,
        fusion_top_k: int = 10,
    ) -> None:
        self._collection = {
            _TRACK_VISUAL: clip_collection_name,
            _TRACK_OCR: ocr_collection_name,
            _TRACK_OD: od_collection_name,
            _TRACK_ASR: asr_collection_name,
        }
        self.top_k = top_k

        fw = fusion_weights or {}
        self._fusion = MultiTrackFusionRetriever(
            w_clip=fw.get("visual", 0.25),
            w_ocr=fw.get("ocr", 0.25),
            w_od=fw.get("od", 0.25),
            w_asr=fw.get("asr", 0.25),
            top_k=fusion_top_k,
        )

    @staticmethod
    def _to_documents(hits: List[dict]) -> List[Document]:
        """qdrant_service.search_dense() hits -> the Document shape
        MultiTrackFusionRetriever already expects. content is left empty —
        the fusion harvesting logic reads payload fields (ocr_text,
        od_detections, text) directly, falling back to content only if a
        field is absent, which never happens for these payload shapes."""
        return [
            Document(id=str(hit.get("id", "")), content="", meta=dict(hit.get("payload") or {}))
            for hit in hits
        ]

    async def run(
        self,
        query_embedding: List[float],
        ocr_query_embedding: List[float],
        od_pattern_type: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query all four tracks concurrently, then fuse.

        query_embedding is the Voyage multimodal vector for the visual track.
        ocr_query_embedding is the Voyage text vector, shared by OCR, OD and
        ASR tracks (all three search over text-space embeddings).

        od_pattern_type filters OD results to frames that contain a specific
        visual pattern type (e.g. "diagram"). None means "all OD frames".

        Track failures degrade gracefully: a failed track is treated as an
        empty list and the remaining tracks produce the result. The failure
        is logged so it surfaces in Phoenix traces, not silently hidden.
        """
        from services.qdrant_service import get_qdrant

        qdrant = get_qdrant()
        # P152: a question naming one ingested video searches only that video
        # (every track), so its passages aren't crowded out by the others'.
        base = {"video_id": video_id} if video_id else {}
        od_filters = dict(base, **({"pattern_type": od_pattern_type} if od_pattern_type is not None else {})) or None
        track_filters = base or None

        async def _track(name: str, vector: List[float], filters=None):
            try:
                hits = await qdrant.search_dense(self._collection[name], vector, self.top_k, filters)
                return name, self._to_documents(hits)
            except Exception as exc:
                logger.warning(
                    "track '%s' retrieval failed (%s) — treating as empty for fusion",
                    name, exc,
                )
                return name, []

        pairs = await asyncio.gather(
            _track(_TRACK_VISUAL, query_embedding, track_filters),
            _track(_TRACK_OCR, ocr_query_embedding, track_filters),
            _track(_TRACK_OD, ocr_query_embedding, od_filters),
            _track(_TRACK_ASR, ocr_query_embedding, track_filters),
        )
        results = dict(pairs)

        logger.debug(
            "parallel retrieval complete: visual=%d ocr=%d od=%d asr=%d (top_k=%d, od_filter=%r)",
            len(results[_TRACK_VISUAL]), len(results[_TRACK_OCR]),
            len(results[_TRACK_OD]), len(results[_TRACK_ASR]),
            self.top_k, od_pattern_type,
        )

        fusion_output = self._fusion.run(
            visual_results=results[_TRACK_VISUAL],
            ocr_results=results[_TRACK_OCR],
            od_results=results[_TRACK_OD],
            asr_results=results[_TRACK_ASR],
        )
        # asr_documents: the transcript track's own ranking, so the caller can keep
        # its top passages when the frame tracks crowd them out of the fused list.
        # od_documents: the same for the frame-description track -- a visual
        # question's answer is a description that fusion can rank out (M3 E-class,
        # 2026-10-23: the settings-window frames sat at od rank 25+).
        return {"documents": fusion_output["merged_results"], "asr_documents": results[_TRACK_ASR],
                "od_documents": results[_TRACK_OD]}


# ── Offline self-test ──────────────────────────────────────────────

def _self_test():
    """Numbered offline checks — no Qdrant, no Haystack integration."""
    ok = 0

    def _doc(id_: str, **meta) -> Document:
        return Document(id=id_, content=meta.get("content", ""), meta=meta)

    # 1. 4-track weights must sum to 1.0
    fuser = MultiTrackFusionRetriever(w_clip=0.25, w_ocr=0.30, w_od=0.15, w_asr=0.30)
    assert len(fuser.weights) == 4
    assert abs(sum(fuser.weights.values()) - 1.0) < 1e-6
    ok += 1
    print("ok %d — 4-track weights sum to 1.0" % ok)

    # 2. Non-unit weights raise
    try:
        MultiTrackFusionRetriever(w_clip=0.5, w_ocr=0.5, w_od=0.5, w_asr=0.5)
        assert False, "should have raised"
    except ValueError:
        pass
    ok += 1
    print("ok %d — non-unit weights raise ValueError" % ok)

    # 3. Basic 4-track fusion
    v = [_doc("v1", video_id="X", frame_ts=1000)]
    o = [_doc("v1", video_id="X", frame_ts=1000, ocr_text="SLA")]
    d = [_doc("v1", video_id="X", frame_ts=1000, od_detections="chart_frame")]
    a = [_doc("a1", video_id="X", start_time=1.0, text="service level agreement")]
    result = fuser.run(v, o, d, a)
    merged = result["merged_results"]
    assert len(merged) >= 1
    ok += 1
    print("ok %d — 4-track fusion produces merged results" % ok)

    # 4. WRRF score is higher when more tracks contribute
    single_track = MultiTrackFusionRetriever(w_clip=0.25, w_ocr=0.30, w_od=0.15, w_asr=0.30)
    r1 = single_track.run([_doc("s1", video_id="Y", frame_ts=0)], [], [], [])
    r4 = single_track.run(
        [_doc("s1", video_id="Y", frame_ts=0)],
        [_doc("s1", video_id="Y", frame_ts=0)],
        [_doc("s1", video_id="Y", frame_ts=0)],
        [_doc("s1", video_id="Y", frame_ts=0)],
    )
    score_1 = r1["merged_results"][0].meta["wrrf_score"]
    score_4 = r4["merged_results"][0].meta["wrrf_score"]
    assert score_4 > score_1
    ok += 1
    print("ok %d — more tracks -> higher WRRF score" % ok)

    # 5. ASR text harvested into merged meta
    asr_only = [_doc("t1", video_id="Z", start_time=5.0, text="load balancer")]
    r = fuser.run([], [], [], asr_only)
    assert r["merged_results"][0].meta.get("asr_text") == "load balancer"
    ok += 1
    print("ok %d — ASR text harvested into merged document" % ok)

    # 6. retrieval_sources includes 'asr'
    assert "asr" in r["merged_results"][0].meta["retrieval_sources"]
    ok += 1
    print("ok %d — retrieval_sources includes 'asr'" % ok)

    # 7. Dedup key falls back to start_time for ASR
    asr_doc = _doc(None, video_id="A", start_time=42.5)
    key = MultiTrackFusionRetriever._dedup_key(asr_doc)
    assert key == "A_42.5"
    ok += 1
    print("ok %d — dedup key uses start_time for ASR docs" % ok)

    # 8. Empty all tracks produces empty result
    r = fuser.run([], [], [], [])
    assert r["merged_results"] == []
    ok += 1
    print("ok %d — empty tracks -> empty result" % ok)

    # 9. top_k respected
    many_docs = [_doc(f"m{i}", video_id="M", frame_ts=i * 1000) for i in range(20)]
    small = MultiTrackFusionRetriever(w_clip=0.25, w_ocr=0.30, w_od=0.15, w_asr=0.30, top_k=3)
    r = small.run(many_docs, [], [], [])
    assert len(r["merged_results"]) == 3
    ok += 1
    print("ok %d — top_k=3 limits output to 3" % ok)

    # 10. ASR-only query (visual/OCR/OD empty) works
    asr_segs = [
        _doc("seg1", video_id="V", start_time=10.0, text="displacement"),
        _doc("seg2", video_id="V", start_time=20.0, text="capacity sweep"),
    ]
    r = fuser.run([], [], [], asr_segs)
    assert len(r["merged_results"]) == 2
    ok += 1
    print("ok %d — ASR-only query returns results" % ok)

    # 11. No track monopolises the fused list. Track point ids never coincide
    # ({video_id}:ocr:{i} vs {video_id}:od:{i}), so fusion never merges a
    # moment across tracks -- it interleaves them by weight, and with D59's
    # unequal weights the top two tracks took all 10 slots: the Gemini chart
    # descriptions (od, 0.15) could never reach the reranker. Measured on 10 real
    # client questions (2026-10-19): D59 4/10, od-heaviest 5/10, equal 8/10 (P145).
    defaults = MultiTrackFusionRetriever()
    tracks = [[_doc("%s%d" % (t, i), video_id="V", frame_ts=i) for i in range(10)]
              for t in ("v", "o", "d", "a")]
    merged = defaults.run(*tracks)["merged_results"]
    contributing = {d.id[0] for d in merged}
    assert contributing == {"v", "o", "d", "a"}, "tracks in fused top-k: %s" % contributing
    ok += 1
    print("ok %d — every track with hits reaches the fused top-k (equal weights, P145)" % ok)

    # 12. Track constants defined
    assert _TRACK_ASR == "asr"
    assert len({_TRACK_VISUAL, _TRACK_OCR, _TRACK_OD, _TRACK_ASR}) == 4
    ok += 1
    print("ok %d — all 4 track constants are distinct" % ok)

    # 13. MomentSearchRetriever queries services/qdrant_service.py (this
    # project's real httpx-based Qdrant client) and fuses results -- rewritten
    # 2026-10-12. The old implementation depended on
    # haystack_integrations.document_stores.qdrant, which isn't installed
    # anywhere in this project (requirements.txt comments it out: "using
    # httpx directly, not qdrant-client") -- it would have raised ImportError
    # the instant anything actually called it. Found live when wiring this
    # retriever into pipelines/retrieval.py for the first time.
    import asyncio as _asyncio13
    import sys as _sys13
    import types as _types13

    class FakeQdrantForMoment:
        def __init__(self, hits_by_collection):
            self.hits_by_collection = hits_by_collection
            self.calls = []

        async def search_dense(self, collection, vector, top_k=50, filters=None):
            self.calls.append((collection, vector, top_k, filters))
            return self.hits_by_collection.get(collection, [])

    fake_hits = {
        "visual_frames": [{"id": "v1", "payload": {"video_id": "X", "frame_ts": 1000}}],
        "ocr_frames": [{"id": "o1", "payload": {"video_id": "X", "frame_ts": 1000, "ocr_text": "SLA"}}],
        "od_frames": [{"id": "d1", "payload": {"video_id": "X", "frame_ts": 1000,
                                               "od_detections": "bullish LB"}}],
        "transcript_segments": [{"id": "a1", "payload": {"video_id": "X", "start_time": 1.0,
                                                         "text": "service level agreement"}}],
    }
    fake_qdrant_svc = FakeQdrantForMoment(fake_hits)
    fake_qdrant_module = _types13.ModuleType("services.qdrant_service")
    fake_qdrant_module.get_qdrant = lambda: fake_qdrant_svc
    _sys13.modules["services.qdrant_service"] = fake_qdrant_module
    try:
        retriever = MomentSearchRetriever(
            clip_collection_name="visual_frames", ocr_collection_name="ocr_frames",
            od_collection_name="od_frames", asr_collection_name="transcript_segments",
        )
        out = _asyncio13.run(retriever.run(
            query_embedding=[0.1] * 1024, ocr_query_embedding=[0.2] * 1024,
        ))
    finally:
        del _sys13.modules["services.qdrant_service"]

    assert len(out["documents"]) >= 1, out
    assert {d.meta.get("video_id") for d in out["documents"]} == {"X"}
    assert fake_qdrant_svc.calls[0][2] == 20, fake_qdrant_svc.calls  # default top_k
    ok += 1
    print("ok %d — MomentSearchRetriever queries qdrant_service directly and fuses results" % ok)

    # 14. A failed track degrades to empty instead of crashing the whole search.
    class FlakyQdrant:
        async def search_dense(self, collection, vector, top_k=50, filters=None):
            if collection == "od_frames":
                raise RuntimeError("qdrant unreachable")
            return []

    fake_qdrant_module2 = _types13.ModuleType("services.qdrant_service")
    fake_qdrant_module2.get_qdrant = lambda: FlakyQdrant()
    _sys13.modules["services.qdrant_service"] = fake_qdrant_module2
    try:
        retriever2 = MomentSearchRetriever(
            clip_collection_name="visual_frames", ocr_collection_name="ocr_frames",
            od_collection_name="od_frames", asr_collection_name="transcript_segments",
        )
        out2 = _asyncio13.run(retriever2.run(
            query_embedding=[0.1] * 1024, ocr_query_embedding=[0.2] * 1024,
        ))
    finally:
        del _sys13.modules["services.qdrant_service"]
    assert out2["documents"] == [], out2
    ok += 1
    print("ok %d — a failed track degrades to empty instead of crashing the whole search" % ok)

    # 15. P152: a question naming one ingested video searches only that video.
    # Every track gets the video_id filter (merged with OD's pattern filter);
    # without video_id the filters are exactly what they were.
    svc15 = FakeQdrantForMoment(fake_hits)
    fake_qdrant_module.get_qdrant = lambda: svc15
    _sys13.modules["services.qdrant_service"] = fake_qdrant_module
    try:
        r15 = MomentSearchRetriever(
            clip_collection_name="visual_frames", ocr_collection_name="ocr_frames",
            od_collection_name="od_frames", asr_collection_name="transcript_segments",
        )
        _asyncio13.run(r15.run(query_embedding=[0.1] * 1024, ocr_query_embedding=[0.2] * 1024,
                               od_pattern_type="SLA", video_id="V1"))
        filtered = {c: f for c, _, _, f in svc15.calls}
        svc15.calls.clear()
        _asyncio13.run(r15.run(query_embedding=[0.1] * 1024, ocr_query_embedding=[0.2] * 1024))
        unfiltered = {c: f for c, _, _, f in svc15.calls}
    finally:
        del _sys13.modules["services.qdrant_service"]
    assert filtered == {"visual_frames": {"video_id": "V1"}, "ocr_frames": {"video_id": "V1"},
                        "od_frames": {"pattern_type": "SLA", "video_id": "V1"},
                        "transcript_segments": {"video_id": "V1"}}, filtered
    assert unfiltered == {"visual_frames": None, "ocr_frames": None, "od_frames": None,
                          "transcript_segments": None}, unfiltered
    ok += 1
    print("ok %d — video_id filters every track to one video; without it filters are unchanged" % ok)

    print("\n%d checks passed" % ok)


if __name__ == "__main__":
    _self_test()
