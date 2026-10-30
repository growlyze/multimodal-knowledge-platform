"""
Hybrid retrieval: embed -> dense + sparse search -> RRF -> cross-encoder rerank.

    query
      ├── dense search  (top RETRIEVAL_TOP_K per collection)
      ├── sparse search (top RETRIEVAL_TOP_K per collection)   ← catches abbreviations (SLA, RTO, KPI)
      ├── reciprocal rank fusion over all four lists           → top RRF_CANDIDATES
      ├── rerank                                               → top_n
      ├── drop everything below MIN_RERANK_SCORE               → may be empty
      └── expand each winner to its sibling chunks             → EXPANSION_MAX_CHUNKS

The reranker decides *which* topics matter; expansion decides how much of each
one the model gets to read. Matching chunk 3 of a 10-chunk explanation used to
send chunk 3 alone.

top_n caps how many results come back; the floor decides how many deserve to.
An off-topic query returns nothing rather than the five least-bad chunks.
"""

import asyncio
import logging
import re
import time

import config
from services.embedding_service import create_embedding_provider
from services.qdrant_service import chunk_index, get_qdrant, source_of
from services.reranker_service import create_reranker
from telemetry_compat import SpanKind, record_decision, set_output, span

logger = logging.getLogger("haystack-api.retrieval")

RRF_K = 60           # standard constant
RRF_CANDIDATES = 20  # how many survive fusion into the reranker
VIDEO_FUSION_TOP_K = 40  # video candidates per sub-query into its own rerank (P152)
VIDEO_TRANSCRIPT_GUARANTEE = 15  # transcript-track passages always reranked too (the client 2026-10-22)
# Frame-description (od) track entries also reranked, for a VISUAL question only.
# Fusion keeps roughly the first 10 of each track; "Is there a chart where a
# app settings window covers the price?" had its four right frames at od
# ranks 25/36/45/59, behind dozens of near-identical "Symbol Search" pop-ups from
# another video (M3 E-class, 2026-10-23). 30 reaches the first; each extra
# description is ~250 rerank tokens, which Voyage bills at a negligible rate.
VIDEO_FRAME_GUARANTEE = 30
# A safety net only, for a hung service: rag_filter's search_timeout valve
# (default 20 s from rag_filter 1.21, 2026-10-20) drops EVERYTHING, text included, when
# it fires, so this must stay below it. A normal video leg takes ~2-3 s. At
# 4.0 s under the old 5 s valve, real questions were cut to 0 results (planner
# alone 1.6-3.2 s). KEEP THIS BELOW THE VALVE.
VIDEO_LEG_DEADLINE_S = 15.0  # below rag_filter 1.21's search_timeout default of 20 s
# How long the video leg waits for the planner before answering with the
# search it already ran on the original question (started in parallel).
PLANNER_BUDGET_S = 3.0
EXPANSION_MAX_CHUNKS = 50  # ceiling on rerank winners + siblings handed back

# Relevance floor on reranker scores. RERANK_TOP_N is a ceiling, not a quota:
# without this, a query the corpus cannot answer still returned exactly
# RERANK_TOP_N chunks, and the "Sources:" block cites whatever /search returns.
# Reported live as a code-reference answer citing three unrelated PDFs and a
# CSV.
#
# 0.70 is measured, not chosen. Voyage rerank-2.5 against this corpus (VPS,
# 2026-09-08) put genuine matches at 0.8789..0.7695 and off-topic queries at
# 0.6289..0.3008 — an empty band between 0.63 and 0.77, and this sits in the
# middle of it. The first value here was 0.30, read off Voyage's published
# example where irrelevant documents score ~0.26; this corpus scores its noise
# twice that high, so 'hi' cleared the floor and cited three manual pages.
# Calibration is per corpus and per reranker model: re-measure before trusting a
# number from anyone's documentation, including Voyage's. tests/test_retrieval.py
# pins both edges against the real scores.
#
# Deliberately stricter than the filter's own min_relevance_score (0.3), which
# this makes a no-op — two layers, and the tighter one wins.
MIN_RERANK_SCORE = 0.70

# The video leg's own floor (M3 I-4, user decision 2026-10-22: one threshold for
# text documents, a separate one for video). Spoken, machine-transcribed passages
# that answer a paraphrased question rerank lower than written documents do, even
# when exactly right: in the M3 Layer 1 run the expected passage was among the
# candidates, often rank 1-3, at 0.61-0.66 ("which stops did he target" #1 0.629,
# "sold all five at the same price?" #1 0.658), and 0.70 dropped it.
# 0.55 is measured (trace of all 92 M3 cases, live index, 2026-10-23): the five
# not-covered negatives' best video candidate across every sub-query is
# 0.296-0.475 (Wyckoff 0.319, Bollinger 0.386, Ichimoku 0.423, Gamma Reversal
# Block 0.416, "2023 Mentorship Episode 5" breakers 0.475). Simulated Layer 1:
# 0.70 -> 61/76 scorable, 0.60-0.65 -> 65, 0.50-0.55 -> 68; 0.55 keeps the
# gain with 0.075 of headroom over the worst negative. Re-measure per reranker.
VIDEO_MIN_RERANK_SCORE = 0.55

# A NAMED video's best passages are what the user means even below the video
# floor ("Which trades break the partial-taking rules from Episode 36?": the
# right window #1 at 0.609). The top NAMED_VIDEO_KEEP of that video are kept down
# to NAMED_VIDEO_MIN_SCORE -- an absolute minimum, so a video that simply doesn't
# cover the question (the octant grid asked of Episode 36: best 0.31) still
# returns nothing and the unfiltered fallback (I-6) answers instead.
NAMED_VIDEO_KEEP = 2
NAMED_VIDEO_MIN_SCORE = 0.45

# Spoken explanation before chart description (M3 I-5). For a "why does X come
# back to Y?" question five chart descriptions reranked 0.72-0.79 and the
# lecturer's own explanation 0.47-0.59: the reranker rewards a description that
# repeats the question's words over the passage that answers it. Outside visual
# questions a transcript passage is ordered as if it scored this much higher --
# ordering only: the floor and the reported score stay the reranker's own.
# Modest on purpose: in the trace m3-002's right passage scored 0.41-0.50 (rank
# 25, under the floor), which no ordering fix reaches -- that one needs the
# planner's "why does ... create <concept>" query (0.89). 0-0.30 simulated
# identically on the 92 cases; 0.15 lifts a near-tie without overriding a
# clearly better frame.
SPOKEN_PREFERENCE = 0.15

# "What happens after X?" (M3 I-7): the answer is the window AFTER the one that
# matches X. Windows are <= 60 s at a 45 s stride, so the next two start within
# ~90 s of the matched one; FOLLOWUP_SPAN_S bounds the lookup.
FOLLOWUP_WINDOWS = 2
FOLLOWUP_SPAN_S = 150

DEFAULT_COLLECTIONS = [config.COLLECTION_DOCUMENTS, config.COLLECTION_CODE]

VIDEO_COLLECTIONS = [
    config.VIDEO_VISUAL_COLLECTION,
    config.VIDEO_OCR_COLLECTION,
    config.VIDEO_OD_COLLECTION,
    config.VIDEO_TRANSCRIPT_COLLECTION,
]

ALL_COLLECTIONS = DEFAULT_COLLECTIONS + VIDEO_COLLECTIONS


def resolve_collections(
    source_type: str = None,
    collection: str = None,
    explicit: list[str] = None,
) -> list[str]:
    """Determine which collections to search.

    explicit     caller-provided list (from ?collections=a,b)  — wins outright
    collection   single collection name                        — just that one
    source_type  'video' | 'document' | None                   — category routing

    source_type=None merges text + video (ALL_COLLECTIONS) rather than
    defaulting to text only — a chat question that never names a source_type
    (which is every question rag_filter.py sends) must still be able to
    reach video content. Fixed 2026-10-12: this used to return text-only,
    which was the entire reason M3 video content was unreachable from the
    live chat UI no matter how a question was phrased.
    """
    if explicit:
        return explicit
    if collection:
        return [collection]
    if source_type == "video":
        return VIDEO_COLLECTIONS
    if source_type == "document":
        return DEFAULT_COLLECTIONS
    return ALL_COLLECTIONS


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Merge ranked lists. score = sum over lists of 1 / (k + rank)."""
    scores: dict = {}
    docs: dict = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            key = (item.get("collection"), item["id"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(key, item)

    ordered = sorted(scores, key=scores.get, reverse=True)
    return [{**docs[key], "rrf_score": scores[key]} for key in ordered]


# A comparison question ("what is the difference between X and Y?") asks about
# two entities at once, and no single chunk covers both - so the reranker scores
# every candidate as half-relevant and the whole page falls under the floor.
# Measured on the live corpus 2026-09-25:
#   "what is the difference between strategy.entry and strategy.order?" -> 0.5742, 0/5 kept
#   "strategy.entry strategy.order difference"                          -> 0.7891, kept
# Same content, same reranker; only the query framing differs. Lowering
# MIN_RERANK_SCORE is NOT the fix - 0.5742 sits below the measured noise band
# (0.63..0.77), so any floor that admits it also admits the junk that once let
# 'hi' cite three manual pages.
_COMPARISON_PATTERNS = [
    re.compile(r"difference[s]?\s+between\s+(.+?)\s+and\s+(.+?)\s*[?.!]*$", re.I),
    re.compile(r"^(?:how\s+do(?:es)?\s+)?(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s*[?.!]*$", re.I),
    re.compile(r"compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)\s*[?.!]*$", re.I),
]


# A video position written as a clock: "15:30", "1:22:45". The look-arounds
# keep ratios, prices and longer digit runs out ("1:2 RR", "12:30:45:10"); the
# [0-5]\d fields reject "1:75".
_TIMESTAMP_PATTERN = re.compile(
    r"(?<![\d:.])(\d{1,2}):([0-5]\d)(?::([0-5]\d))?(?![\d:])",
)

# I-2: positions written in words (M3 question set, 2026-10-22: "around 25
# minutes in" named a moment, but only clocks parsed, so the moment lookup
# never ran). Two units ("1 hour 8 minutes", "1h08m", "44m30s") read as a
# position on their own; ONE unit needs a position cue around it, because
# "15 minute", "1h", "4 hour" are far more often chart timeframes ("the 5 minute
# chart", "a 4 hour bar") and "for 20 minutes" a duration.
_H = r"(?:hours?|hrs?|h)"
_M = r"(?:minutes?|mins?|m)"
_S = r"(?:seconds?|secs?|s)"
_HOUR_MIN_PATTERN = re.compile(
    r"\b(\d{1,2})\s*%s\s*(?:and\s+)?(\d{1,2})\s*%s(?:\s*(?:and\s+)?(\d{1,2})\s*%s)?\b" % (_H, _M, _S), re.I)
_MIN_SEC_PATTERN = re.compile(r"\b(\d{1,3})\s*%s\s*(?:and\s+)?(\d{1,2})\s*%s\b" % (_M, _S), re.I)
_MINUTES_PATTERN = re.compile(r"\b(\d{1,3})(?:\s*-\s*|\s*)%s\b" % _M, re.I)
_HOURS_PATTERN = re.compile(r"\b(\d{1,2})(?:\s*-\s*|\s*)%s\b" % _H, re.I)
# "around minute 44", "the 44th minute": the word order alone says position.
_MINUTE_N_PATTERN = re.compile(r"\bminute\s+(\d{1,3})\b|\b(\d{1,3})(?:st|nd|rd|th)\s+minute\b", re.I)
# Relative positions: resolved against the named video's duration.
_HALFWAY_PATTERN = re.compile(
    r"\b(?:half\s*-?\s*way|midway)\b|\bmiddle\s+of\s+(?:the\s+|this\s+|that\s+)?"
    r"(?:video|episode|stream|commentary|lecture|clip|post)\b", re.I)

# One-unit forms count only next to one of these ("at 44 min", "25 minutes in",
# "the 25 minute mark").
_POSITION_BEFORE = frozenset({"at", "around", "about", "near", "approximately", "approx", "roughly", "by"})
_POSITION_AFTER = frozenset({"in", "into", "mark", "point", "timestamp"})
# "for 20 minutes", "the last 10 minutes", "the first 5 minutes after the open".
_DURATION_BEFORE = frozenset({"for", "every", "each", "last", "first", "next", "past", "within", "over",
                              "than", "per", "took", "takes", "lasted"})
# "halfway into the gap" is a location on a chart, not a time.
_PRICE_WORDS = frozenset({"gap", "gaps", "fvg", "fvgs", "range", "move", "leg", "candle", "candles", "block",
                          "blocks", "ob", "retracement", "level", "levels", "zone", "zones", "array", "arrays",
                          "wick", "body", "bar", "imbalance", "swing", "rally", "drop", "trend", "target",
                          "day", "week", "session", "order", "high", "low", "premium", "discount"})
# I-3: clock times of day. "In Episode 36, what happened after the 9:30 open?"
# read as 9 min 30 s into the video and answered from the end of the 9:11
# episode (2026-10-22). A time with any of these within three words either side
# is a time of day, not a video position.
_MARKET_WORDS = frozenset({"open", "opens", "opening", "close", "closes", "closing", "candle", "candles",
                           "bar", "bars", "killzone", "killzones", "kill", "session", "sessions", "ny", "nyc",
                           "york", "est", "edt", "et", "cst", "cdt", "pst", "utc", "gmt", "bst", "cet",
                           "london", "asia", "asian", "eastern", "central", "pacific", "clock", "am", "pm",
                           "macro", "macros", "rth", "eth", "bell", "premarket", "settlement"})
_AM_PM = re.compile(r"^\s*[ap]\.?\s?m\b", re.I)
_TOKEN = re.compile(r"[a-z]+|\d+")
_CLAUSE_END = re.compile(r"[,;?!]|\.(?=\s|$)")
_CONTEXT_WORDS = 3


def _context(query: str, start: int, end: int):
    """The words of the same clause just before and just after query[start:end]."""
    before = _CLAUSE_END.split(query[:start])[-1]
    after_raw = query[end:]
    after = _CLAUSE_END.split(after_raw)[0]
    return (_TOKEN.findall(before.lower())[-_CONTEXT_WORDS:],
            _TOKEN.findall(after.lower())[:_CONTEXT_WORDS],
            bool(_AM_PM.match(after_raw)))


def _candidates(query: str):
    """(start, end, kind, seconds) for every time-like span, in text order;
    a span inside a longer one ("8 minutes" in "1 hour 8 minutes") is dropped."""
    found = []
    for m in _TIMESTAMP_PATTERN.finditer(query):
        a, b, c = m.groups()
        secs = int(a) * 3600 + int(b) * 60 + int(c) if c is not None else int(a) * 60 + int(b)
        found.append((m.start(), m.end(), "clock", secs))
    for m in _HOUR_MIN_PATTERN.finditer(query):
        found.append((m.start(), m.end(), "multi",
                      int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)))
    for m in _MIN_SEC_PATTERN.finditer(query):
        found.append((m.start(), m.end(), "multi", int(m.group(1)) * 60 + int(m.group(2))))
    for m in _MINUTE_N_PATTERN.finditer(query):
        found.append((m.start(), m.end(), "minute_n", int(m.group(1) or m.group(2)) * 60))
    for m in _MINUTES_PATTERN.finditer(query):
        found.append((m.start(), m.end(), "single", int(m.group(1)) * 60))
    for m in _HOURS_PATTERN.finditer(query):
        found.append((m.start(), m.end(), "single", int(m.group(1)) * 3600))
    for m in _HALFWAY_PATTERN.finditer(query):
        found.append((m.start(), m.end(), "halfway", None))
    found.sort(key=lambda f: (f[0], -(f[1] - f[0])))
    kept, reach = [], -1
    for f in found:
        if f[0] >= reach:
            kept.append(f)
            reach = f[1]
    return kept


def parse_timestamp(query: str, duration: float | None = None) -> int | None:
    """The video position a query names, as total seconds, or None.

    "at 15:30" -> 930, "1:22:45" -> 4965, "around 25 minutes in" -> 1500,
    "1h08m" -> 4080, "the 25 minute mark" -> 1500, "what is SLA" -> None.
    Clock times of day are not positions: "the 9:30 open", "9:30 AM", "8:30 NY
    time", "the 10:00 candle" -> None (I-3). "halfway through" is half of
    `duration` (the named video's length), None without it. Rule for a bare
    "9:30": a video position -- unless `duration` is known and shorter, when it
    can't be one. The LAST time that qualifies wins: rag_filter 1.33 searches a
    follow-up naming no video as "<previous message> <follow-up>", so "...around
    25 minutes in, what is he forecasting? what about at 44:30?" asks about
    44:30 -- the latest message comes last. Skipped times don't compete: "after
    the 9:30 open, what does he say at 25:00" is still 1500.
    """
    found = None
    for start, end, kind, secs in _positions(query, duration):
        if duration and secs > duration:
            continue
        found = secs
    return found


def _positions(query: str, duration: float | None):
    """(start, end, kind, seconds) for each time in `query` that reads as a video
    position -- parse_timestamp's rules minus the past-the-end check, which its
    two callers apply in opposite directions."""
    for start, end, kind, secs in _candidates(query or ""):
        before, after, am_pm = _context(query, start, end)
        if am_pm or _MARKET_WORDS & set(before + after):
            continue
        if kind == "single":
            if not ((before and before[-1] in _POSITION_BEFORE) or (after and after[0] in _POSITION_AFTER)):
                continue
        if kind in ("single", "multi") and before and before[-1] in _DURATION_BEFORE:
            continue
        if kind == "halfway":
            if not query[start:end].lower().startswith("middle"):
                if not after or after[0] not in ("through", "into", "in", "point", "mark", "of"):
                    continue
                rest = _TOKEN.findall(_CLAUSE_END.split(query[end:])[0].lower())[1:5]
                if _PRICE_WORDS & set(rest):
                    continue
            if not duration:
                continue
            secs = int(duration / 2)
        yield start, end, kind, secs


def past_end_position(query: str, duration: float | None):
    """(asked, reread) when the first video position `query` names lies past the
    end of a `duration`-long video and none inside it does, else None.

    Round 3, issue 5: "In Episode 36, what happens at 3:10:00?" -- Episode 36 is
    9:11 long, parse_timestamp rightly returned None, and nothing told either
    model the length, so both answered vaguely. `reread` is the same digits read
    one unit smaller when that lands inside the video ("3:10:00" -> 3:10 = 190 s:
    users type a trailing :00 onto a m:ss time), else None."""
    if not duration or parse_timestamp(query, duration=duration) is not None:
        return None
    for start, end, kind, secs in _positions(query, duration):
        if secs <= duration:
            continue
        reread = None
        parts = query[start:end].strip().split(":")
        if kind == "clock" and len(parts) == 3 and int(parts[2]) == 0:
            reread = int(parts[0]) * 60 + int(parts[1])
            if reread > duration:
                reread = None
        return secs, reread
    return None


# Payload fields only video transcript segments carry. routers/search.py adds a
# window on them for any query naming a clock time ("the 9:30 open"), but text
# chunks have neither field, so on a text collection the window can only ever
# exclude everything -- and as build_filter() renders the {"$lte": n} operator
# dicts, Qdrant rejects the request with a 400 that qdrant_service turns into a
# silent []. Every timestamp query lost the whole document KB that
# way (found live 2026-10-16; the range-operator half is P100).
_VIDEO_ONLY_FILTER_KEYS = frozenset({"start_time", "end_time"})


def _text_filters(filters: dict | None) -> dict | None:
    """`filters` minus the keys no text-collection payload can satisfy."""
    if not filters:
        return filters
    kept = {k: v for k, v in filters.items() if k not in _VIDEO_ONLY_FILTER_KEYS}
    return kept or None


# Questions about what is ON the chart, where a frame description is the answer
# and must keep winning over speech (M3 E-class: labels, zones, "what does he
# draw", "what was on screen"). Deliberately narrow: a false negative only drops
# the spoken preference, a false positive denies it to a concept question.
_VISUAL_RE = re.compile(
    r"\b(?:charts?|labels?|labell?ed|draws?|drawn|drawing|marks?|marked|screen|shows?|shown|showing|"
    r"zones?|highlight\w*|boxes|box|visible|window)\b", re.I)

# "What happens after / next / then", "and then what": a sequence question.
# "... after the cache is cleared?" inside a definition is NOT one --
# the verb must be happen/come/do, asked as what-next.
_WHAT_NEXT_RE = re.compile(
    r"\bwhat\s+(?:happens|happened|comes|came|(?:does|did)\s+\w+\s+do|is|was)\s+(?:next|after|afterwards|then)\b"
    r"|\band\s+then\s+what\b"
    # rag_filter 1.34 searches a follow-up as "<anchor> <previous> <message>", so the
    # bare "then what?" / "what next?" / "after that?" now reach here (2026-10-23).
    r"|\bthen\s+what\b|\bwhat\s+next\b|\bafter\s+that\s*\?", re.I)


def _is_visual_question(question: str) -> bool:
    return bool(_VISUAL_RE.search(question or ""))


def _asks_what_next(question: str) -> bool:
    return bool(_WHAT_NEXT_RE.search(question or ""))


def comparison_terms(query: str) -> str | None:
    """Rewrite a comparison question into the keyword form that retrieves.

    Returns None when the query is not a comparison, so the caller leaves an
    ordinary query completely untouched.
    """
    for pattern in _COMPARISON_PATTERNS:
        match = pattern.search(query.strip())
        if match:
            left, right = (g.strip(" 	?.,") for g in match.groups()[:2])
            if left and right:
                return f"{left} {right} difference"
    return None


async def search(query: str, top_n: int = None, collections: list[str] = None,
                 filters: dict = None, _allow_rewrite: bool = True) -> list[dict]:
    """Run the full pipeline. Returns ranked [{id, score, payload, collection}, ...].

    Raises on embedding failure — an empty list would silently look like
    "nothing relevant found", which is worse than a visible error.

    One RETRIEVER span per call (a no-op if telemetry_compat can't reach the
    shared telemetry module — see its docstring). It carries the confidence-gate
    decision the "Sources:" block depends on: *why* this search answered or
    declined, not just the count already in the log line above.
    """
    top_n = top_n or config.RERANK_TOP_N
    collections = collections or DEFAULT_COLLECTIONS

    # Video collections are handled by _search_video() (MomentSearchRetriever,
    # D59 4-track WRRF fusion) instead of the generic single-embedder fan-out
    # below -- visual_frames holds Voyage MULTIMODAL embeddings (D66) while
    # every text collection holds Voyage TEXT embeddings, two different
    # models' vector spaces despite both being 1024d. Wired in 2026-10-12;
    # previously visual_frames was excluded outright (P94) since this
    # retriever didn't work in production (it depended on
    # haystack_integrations, never installed here -- see video_fusion.py).
    video_collections = [c for c in collections if c in VIDEO_COLLECTIONS]
    text_collections = [c for c in collections if c not in VIDEO_COLLECTIONS]
    started = time.perf_counter()

    with span("retrieval.search", SpanKind.RETRIEVER, input_value=query,
             attributes={"retrieval.collections": ",".join(collections),
                         "retrieval.top_n": top_n}) as sp:
        if not text_collections and not video_collections:
            set_output(sp, "0 results")
            return []

        # P152: the video leg plans its own queries (one Gemini call) and runs
        # concurrently with the text leg; its results are merged after the
        # text leg is reranked and floored, so text results are unchanged.
        video_task = (asyncio.create_task(_video_results(query, top_n, sp))
                      if video_collections else None)

        try:
            text_part = await _text_leg(query, top_n, text_collections, filters, sp,
                                        _allow_rewrite)
        except BaseException:
            # A failed or cancelled text leg must not leave the video leg's
            # Gemini/Voyage calls running unobserved (P152 review).
            if video_task is not None:
                video_task.cancel()
            raise
        fused, reranked, rescued = text_part

        video_groups = await _await_video_leg(video_task, started, sp)
        if not fused and not any(video_groups):
            logger.info("no candidates for '%s' in %s", query[:80], collections)
            record_decision(sp, "retrieval", "declined", "no candidates after fusion")
            set_output(sp, "0 results")
            return []
        text_primaries = len(reranked)
        if config.ENABLE_TOPIC_EXPANSION and reranked and not rescued:
            reranked = await expand_results(reranked)
        # Video is appended after the text results (and their topic expansion),
        # never in place of them: measured 2026-10-20, letting video compete for
        # the top-5 cut the KB answer for "how is an SLA measured across three
        # tiers" from 14 text passages to 2.
        reranked = _merge(reranked, video_groups,
                          max(sum(1 for g in video_groups if g), top_n - text_primaries))
        logger.info("search '%s' -> %d results in %.0fms (%d candidates fused)",
                    query[:80], len(reranked), (time.perf_counter() - started) * 1000, len(fused))
        set_output(sp, f"{len(reranked)} results")
        return reranked


async def _await_video_leg(video_task, started: float, sp) -> list:
    """The video leg's results, or [] if it misses VIDEO_LEG_DEADLINE_S."""
    if video_task is None:
        return []
    remaining = VIDEO_LEG_DEADLINE_S - (time.perf_counter() - started)
    try:
        return await asyncio.wait_for(video_task, timeout=max(0.05, remaining))
    except asyncio.TimeoutError:
        logger.warning("video leg cut at the %.1fs deadline -- answering with text only",
                       VIDEO_LEG_DEADLINE_S)
        record_decision(sp, "retrieval.video", "degraded",
                        f"video leg exceeded {VIDEO_LEG_DEADLINE_S}s deadline")
        return []


async def _text_leg(query: str, top_n: int, text_collections: list, filters, sp,
                    allow_rewrite: bool):
    """The text collections' search, exactly as before P152: dense (+ sparse)
    per collection, fused, reranked, floored; plus the comparison-terms rescue
    (C9) when the floor empties the page. Returns (fused, results, rescued)."""
    tasks = []
    if text_collections:
        embedder = create_embedding_provider()
        qdrant = get_qdrant()
        vector = (await embedder.embed([query], input_type="query"))[0]
        text_filters = _text_filters(filters)
        for name in text_collections:
            tasks.append(qdrant.search_dense(name, vector, config.RETRIEVAL_TOP_K, text_filters))
            if config.ENABLE_HYBRID_SEARCH:
                tasks.append(qdrant.search_sparse(name, query, config.RETRIEVAL_TOP_K, text_filters))

    result_lists = list(await asyncio.gather(*tasks)) if tasks else []

    # Never hand the reranker fewer candidates than the caller asked results for.
    fused = reciprocal_rank_fusion(result_lists)[:max(RRF_CANDIDATES, top_n)]
    sp.set_attribute("retrieval.candidate_count", len(fused))
    if not fused:
        return fused, [], False
    reranked, calibrated = await _rerank(query, fused, top_n)
    sp.set_attribute("retrieval.rerank_scores",
                     [round(float(c.get("score") or 0.0), 4) for c in reranked])
    if calibrated:
        reranked = _drop_irrelevant(query, reranked, sp)
    else:
        # Same guard as the floor itself: RRF-fallback and NoopReranker scores
        # are not relevance, so nothing was filtered on them — say so on the span.
        record_decision(sp, "retrieval", "answered" if reranked else "declined",
                        "reranker uncalibrated (RRF fallback or missing "
                        "RERANKER_API_KEY) — relevance floor not applied")
    # Only when the floor emptied the page, and only once (allow_rewrite).
    # A comparison question that retrieves nothing is the one case where the
    # corpus almost certainly does cover the answer - see comparison_terms().
    # Gated on `not reranked` so a query that already returns results is
    # never rewritten: this can add recall, never remove it. Text leg only:
    # a passing video passage must not suppress it, and the video leg already
    # ran on the original message (P152 review).
    if not reranked and allow_rewrite:
        rewritten = comparison_terms(query)
        if rewritten:
            logger.info("comparison query '%s' returned nothing above the "
                        "floor - retrying as '%s'", query[:80], rewritten)
            record_decision(sp, "retrieval", "declined",
                            f"empty; retrying as comparison terms {rewritten!r}")
            rescued = await search(rewritten, top_n=top_n, collections=text_collections,
                                   filters=filters, _allow_rewrite=False)
            return fused, rescued, True
    return fused, reranked, False


# Priority order of payload keys that hold a candidate's rerankable text.
# "content" covers the M1/M2 document/code collections; the rest are M3's
# video payload shapes (pipelines/video_ingestion.py's _phase_index) --
# transcript segments use "text", OCR frames "ocr_text", vision-description
# (OD) frames "od_detections". "title" is the last resort for visual_frames,
# which has no text field at all (pure image similarity) -- every M3 payload
# sets it, so this is what keeps a visual_frame candidate from reranking as
# "" once visual_frames started flowing through this path (2026-10-12,
# MomentSearchRetriever wiring): even one empty string makes Voyage's rerank
# API reject the WHOLE batch (P99), not just degrade that one candidate.
_RERANK_TEXT_KEYS = ("content", "text", "ocr_text", "od_detections", "title")


def _rerank_text(payload: dict) -> str:
    """The text to show the reranker for one candidate. Checking only
    "content" silently reranked every M3 result as an empty string, and
    Voyage's rerank API rejects a batch containing one -- found live
    2026-10-12, first real retrieval test against real M3 content."""
    for key in _RERANK_TEXT_KEYS:
        value = payload.get(key)
        if value:
            return value
    return ""


# Maps _phase_index's source_type payload field back to the Qdrant
# collection it lives in -- MomentSearchRetriever's fused output doesn't
# carry the collection name directly (fusion merges across all 4), but
# downstream code (RRF dedup keys, the response's metadata.collection field)
# needs it the same way every text-collection candidate already has it.
_SOURCE_TYPE_TO_VIDEO_COLLECTION = {
    "visual_frame": config.VIDEO_VISUAL_COLLECTION,
    "ocr_frame": config.VIDEO_OCR_COLLECTION,
    "od_frame": config.VIDEO_OD_COLLECTION,
    "transcript_segment": config.VIDEO_TRANSCRIPT_COLLECTION,
}


async def _search_video(query: str, sp=None, video_id: str = None,
                        frame_guarantee: int = 0) -> list[dict]:
    """Run the 4-track MomentSearchRetriever and shape its output as one
    more ranked list for reciprocal_rank_fusion() to combine with the text
    lists.

    Degrades to an empty list (never raises) on any failure -- the Voyage
    multimodal/text query embedding calls are a second point of failure
    beyond the main text embedder, and a dead video path must not take down
    a search that still has perfectly good text results. `sp` is optional
    (only used to record the decision on the caller's span) and, like
    retrieval.py's other span-taking helpers, works identically when it's a
    real span or telemetry_compat's _NoopSpan.
    """
    try:
        from pipelines.video_embedder import embed_query
        from pipelines.video_fusion import MomentSearchRetriever

        query_emb = await asyncio.to_thread(embed_query, query)
        retriever = MomentSearchRetriever(
            clip_collection_name=config.VIDEO_VISUAL_COLLECTION,
            ocr_collection_name=config.VIDEO_OCR_COLLECTION,
            od_collection_name=config.VIDEO_OD_COLLECTION,
            asr_collection_name=config.VIDEO_TRANSCRIPT_COLLECTION,
            top_k=config.RETRIEVAL_TOP_K,
            # P152: 10 across 4 tracks let only ~2-3 transcript candidates
            # reach the reranker; the right window sat at rank 11-25.
            fusion_top_k=VIDEO_FUSION_TOP_K,
            fusion_weights={"visual": config.WRRF_W_CLIP, "ocr": config.WRRF_W_OCR,
                            "od": config.WRRF_W_OD, "asr": config.WRRF_W_ASR},
        )
        result = await retriever.run(
            query_embedding=query_emb.visual_embedding,
            ocr_query_embedding=query_emb.voyage_embedding,
            **({"video_id": video_id} if video_id else {}),
        )
    except Exception as exc:
        logger.warning("video search failed (%s) — continuing with text results only", exc)
        if sp is not None:
            record_decision(sp, "retrieval.video", "degraded", f"video search failed: {exc}")
        return []

    candidates = []
    for rank, doc in enumerate(result["documents"]):
        meta = dict(doc.meta)
        if meta.get("overview"):
            # Reachable only via a summary question (_fetch_overview): as an
            # ordinary candidate the 5-8-concept prose wins a passage query's
            # guaranteed slot and is cited at 0:00:00 (P152 review).
            continue
        collection = _SOURCE_TYPE_TO_VIDEO_COLLECTION.get(meta.get("source_type"), "video")
        # rrf_score: _rerank's fallback (reranker down) orders by it; video
        # candidates used to get it from the shared text fusion (P152).
        candidates.append({"id": doc.id, "collection": collection, "payload": meta,
                           "rrf_score": 1.0 / (RRF_K + rank + 1)})
    # The transcript track's own top passages always reach the reranker. A frame-
    # heavy video (2,031 keyframes vs ~500) filled the 40 fused slots with frames,
    # and Episode 12's key explanation -- 31st in the transcript track --
    # never got reranked (the client, 2026-10-22). The reranker reads the words, so
    # giving it these costs nothing but a few more rerank tokens.
    seen = {str(c["id"]) for c in candidates}
    for rank, doc in enumerate((result.get("asr_documents") or [])[:VIDEO_TRANSCRIPT_GUARANTEE]):
        meta = dict(doc.meta)
        if meta.get("overview") or str(doc.id) in seen:
            continue
        seen.add(str(doc.id))
        candidates.append({"id": doc.id, "collection": config.VIDEO_TRANSCRIPT_COLLECTION,
                           "payload": meta, "rrf_score": 1.0 / (RRF_K + VIDEO_FUSION_TOP_K + rank + 1)})
    # Same rescue for frame descriptions, when the caller asks (visual questions,
    # VIDEO_FRAME_GUARANTEE): there the description is the answer.
    for rank, doc in enumerate((result.get("od_documents") or [])[:frame_guarantee]):
        meta = dict(doc.meta)
        if meta.get("overview") or str(doc.id) in seen:
            continue
        seen.add(str(doc.id))
        candidates.append({"id": doc.id, "collection": config.VIDEO_OD_COLLECTION,
                           "payload": meta, "rrf_score": 1.0 / (RRF_K + VIDEO_FUSION_TOP_K + rank + 1)})
    return candidates


async def _video_results(question: str, top_n: int, sp=None) -> list:
    """P152: the video leg, planned. One list of floored results per sub-query.

    The chat message is rewritten into clean sub-queries (query_planner); a
    named, indexed video narrows every track to it; each sub-query's
    candidates are reranked against that sub-query, not the whole message,
    and floored like the text leg. Never raises: any failure means no video
    results, never a broken chat turn.
    """
    from pipelines import query_planner

    spoken_first = not _is_visual_question(question)
    # Only a visual question asks for the frame-description rescue; passed only
    # then, so every other question's search is exactly what it was.
    frames_kw = {} if spoken_first else {"frame_guarantee": VIDEO_FRAME_GUARANTEE}

    async def _rank(sub_query, cands, named=False):
        if not cands:
            return []
        # Every candidate is reranked (Voyage bills the documents sent, not
        # top_n), so the spoken preference can lift a transcript passage that
        # sat below the top_n frames; the cut to top_n comes after it.
        ranked, calibrated = await _rerank(sub_query, cands, len(cands))
        if not calibrated:
            return ranked[:top_n]
        return _video_floor(ranked, top_n, named=named, spoken_first=spoken_first)

    async def _raw():
        # The original question, searched at once in parallel with the
        # planner: if the planner is slow or fails, the answer still has
        # video passages instead of none (a live question got 0 results when
        # the whole leg waited on the planner, 2026-10-20).
        try:
            return await _rank(question, await _search_video(question, sp, **frames_kw))
        except Exception as exc:
            logger.warning("raw-question video search failed (%s)", exc)
            return []

    raw_task = asyncio.create_task(_raw())
    keep_raw = False   # set when the planned queries only ADD to the raw question's search
    try:
        try:
            plan = await asyncio.wait_for(
                asyncio.to_thread(query_planner.plan_video_queries, question), PLANNER_BUDGET_S)
        except asyncio.TimeoutError:
            logger.warning("query planner over %.1fs -- planning from the question itself",
                           PLANNER_BUDGET_S)
            plan = None
        if plan is None or (plan.queries == [question] and plan.intent == "passage"):
            # Round 3, issue 2 (live 2026-10-23 12:11 UTC, six parallel chats): the
            # planner ran over budget and "Summarize the video he posted on X." was
            # searched as-is across every video -- no passage resembles it, 0
            # results, and the reply said there is no X video. The reference ("posted
            # on X", "Episode 36", "March 27") and the "summarize" cue are in the
            # question; resolving them needs no model, so a missing plan no longer
            # loses the named-video filter or the overview. Nothing named -> the
            # raw search IS the plan, exactly as before.
            plan = await _unplanned(question)
            if plan is None:
                # H2: a definition question with no plan still gets the lecturer's
                # explanation searched beside the raw question (lecture_queries).
                lecture = query_planner.lecture_queries(question)
                if not lecture:
                    return [await raw_task]
                plan = query_planner.VideoQueryPlan(lecture, None, "passage")
                keep_raw = True
        video_id, queries = None, plan.queries
        catalog = await query_planner.indexed_videos()
        # Round 4: "<X question> and in the other video?" names X only to exclude it.
        # Enforced here whatever the planner put in video_name: every video is
        # searched and X's results dropped (_one), never X alone.
        excluded = query_planner.excluded_video_id(question, catalog, plan.video_name)
        others = [vid for vid, _ in catalog if vid != excluded]
        if excluded:
            queries = query_planner.strip_video_references(plan.queries)
            if plan.intent == "summary" and len(others) == 1:
                # One video left: it is "the other one" -- its overview answers.
                video_id = others[0]
        elif plan.video_name and len(query_planner.referenced_video_ids(question, catalog)) > 1:
            # m3-053 (sprint 2026-10-25): "Compare how the load balancer is explained in
            # Episode 12 with the March 27 commentary" -- the planner copied one of the
            # two into video_name (which one varied run to run) and the other half of
            # the comparison was never searched. Two videos referenced: search them all.
            video_id = None
        elif plan.video_name:
            video_id = query_planner.resolve_video_id(plan.video_name, catalog)
        else:
            # The planner names a video only when the user calls it one: "Summarize
            # the video he posted on X" came back intent=summary, video_name=null
            # (no overview, nothing returned), and "Was the March 27 trade
            # profitable?" searched every video, leaving the right window (0.53)
            # under the unnamed floor (M3 J/D, 2026-10-23). A date or platform
            # reference that resolves to exactly one video names it.
            video_id = query_planner.implied_video_id(question, catalog)
            if video_id:
                # The reference is the filter now, and noise inside the video: "March 27
                # trade result" ranked garbled OCR and title cards that carry the date
                # (0.62) above the spoken outcome (0.40), 2026-10-23.
                queries = query_planner.strip_video_references(plan.queries)
        if plan.intent == "passage" and not keep_raw:
            # H2: whatever the planner wrote, a definition question also searches
            # "why does the market create <concept>" (query_planner.lecture_queries).
            queries = query_planner.with_lecture_queries(question, queries)
        if sp is not None:
            sp.set_attribute("retrieval.video.plan",
                             "queries=%r video=%r->%r intent=%s"
                             % (queries, plan.video_name, video_id, plan.intent))
            if excluded:
                sp.set_attribute("retrieval.video.excluded", excluded)
            # Which model wrote the plan (gemini/claude): evaluations must be able
            # to tell a Gemini-planned result from a Claude-planned one.
            sp.set_attribute("retrieval.video.plan_provider", plan.provider or "none")

        async def _one(sub_query, vid=video_id, keep_best=False):
            cands = await (_search_video(sub_query, sp, video_id=vid, **frames_kw) if vid
                           else _search_video(sub_query, sp, **frames_kw))
            # Dropped before the rerank: no tokens spent on the video the user excluded.
            cands = _without_video(cands, excluded)
            return await _rank(sub_query, cands, named=bool(vid) or keep_best)

        # P100: "what does he explain at 1:07:55 in episode 12" names a moment;
        # answer from the passages covering it, not from similarity search
        # (live it returned frames at 26:10, 2:17, 25:50 -- 2026-10-21). Only
        # with a resolved video: alone, "the 9:30 open" is as often a time of day
        # as a position in a video. Nothing at that time -> the usual search.
        # I-2: "halfway through" and a time past the video's end need its duration.
        duration = query_planner.video_duration(video_id) if video_id else None
        at = parse_timestamp(question, duration=duration)
        if video_id and at is not None and plan.intent == "passage":
            moment = await _passages_at(video_id, at)
            if moment:
                # "...what about at 44:30? then what?" (2026-10-23): the moment alone
                # repeated 44:25 and the user got "nothing after 45:24". A what-next
                # question continues from the moment into the windows after it.
                if _asks_what_next(question):
                    return await _with_following_windows([moment])
                return [moment]
        # Round 3, issue 5: a time past the named video's end ("In Episode 36, what
        # happens at 3:10:00?" -- it is 9:11 long). Both models answered vaguely:
        # nothing told them the length. The note says so; when the digits also
        # read as m:ss inside the video, the passages there follow it. Only with a
        # resolved video and a known duration -- no named video, no note.
        note = None
        past = past_end_position(question, duration) if video_id else None
        if past:
            asked, reread = past
            title = dict(await query_planner.indexed_videos()).get(video_id) or "This video"
            note = _past_end_note(video_id, title, duration, asked, reread)
            if sp is not None:
                record_decision(sp, "retrieval.video", "past_end",
                                f"{video_id}: {asked}s asked, video {duration:.0f}s, reread {reread}")
            moment = await _passages_at(video_id, reread) if reread is not None else []
            if moment:
                return [[note] + moment]

        # "How many videos do you have?" asks about the collection, not about
        # any passage -- no transcript line resembles it (0 results live,
        # 2026-10-20). Answer it from the indexed videos and their overviews.
        if plan.intent == "library":
            raw_task.cancel()
            # The collection answer does NOT replace the content search: "do any of
            # the videos show STH/STL labels?" was planned as library and got only
            # the video list, while Episode 12's frames carry STH hundreds of times
            # (the client, 2026-10-22). Passages that clear the floor come back with it;
            # "how many videos" matches no passage, so it's unchanged.
            library, passage_groups = await asyncio.gather(
                _library_result(), asyncio.gather(*(_one(q) for q in queries)))
            return ([[library]] if library else []) + [g for g in passage_groups if g]

        groups = list(await asyncio.gather(*(_one(q) for q in queries)))
        if keep_raw:
            groups = [await raw_task] + groups
        if video_id and not _clears_video_floor(groups):
            # I-6: the named video has nothing relevant ("In the wrong video,
            # Episode 36, where does he explain the octant grid?" -- the grid is
            # in the X video; Episode 36's best passage reranked 0.31). The same
            # sub-queries run across every video, appended after whatever the
            # named video kept, so the answer can say "not in Episode 36, but
            # in <video> at <time>" instead of nothing.
            # The widened results keep their best like a named video's: the X video's
            # octant passages rerank 0.52-0.57, so the 0.55 floor made the fallback a
            # coin toss and the question often returned nothing (2026-10-23). The user
            # named a video, so the topic is expected to exist somewhere.
            wider = list(await asyncio.gather(*(_one(q, None, keep_best=True) for q in queries)))
            if any(wider):
                logger.info("named video %s had nothing above %.2f -- widened to all videos",
                            video_id, VIDEO_MIN_RERANK_SCORE)
                if sp is not None:
                    record_decision(sp, "retrieval.video", "widened",
                                    f"named video {video_id} below the video floor; searched all videos")
                groups = groups + wider
        if not any(groups):
            # The planned searches found nothing above the floor: the original
            # question's results are better than an empty answer.
            groups = [_without_video(await raw_task, excluded)]
        else:
            raw_task.cancel()
        if _asks_what_next(question):
            groups = await _with_following_windows(groups)
        # Stage B: a whole-video summary is answered by the stored overview,
        # first; passage results still follow. No overview -> passages only.
        if plan.intent == "summary" and video_id:
            overview = await _fetch_overview(video_id)
            if overview:
                groups.insert(0, [overview])
        elif plan.intent == "summary" and excluded:
            # "Summarize X. and the other video?" with several others: which one is
            # the user's call. The library minus X lets the model list them and ask.
            library = await _library_result(exclude=excluded)
            if library:
                groups.insert(0, [library])
        if video_id and plan.intent != "summary" and _PRICE_QUESTION_RE.search(question):
            readings = await _price_readings(video_id)
            if readings:
                groups.insert(0, readings)
        if note:
            groups.insert(0, [note])
        return groups
    except Exception as exc:
        logger.warning("planned video search failed (%s) -- using the original question's results", exc)
        try:
            return [await raw_task]
        except Exception:
            return []
    finally:
        if not raw_task.done():
            raw_task.cancel()


# "Summarize the video he posted on X": the planner's summary intent, read off
# the question when there is no plan (round 3, issue 2).
_SUMMARY_CUE_RE = re.compile(r"\b(?:summari[sz]e|summary|overview|recap|main (?:concepts|points|ideas|takeaways))\b",
                             re.I)


async def _unplanned(question: str):
    """A plan made without a model, for when the planner is over budget or
    failed: the original question, filtered to the one video it references
    (query_planner.implied_video_id), summary intent on a summary cue. None when
    the question references no single indexed video."""
    from pipelines import query_planner
    try:
        catalog = await query_planner.indexed_videos()
        video_id = query_planner.implied_video_id(question, catalog)
        # "Summarize X. and in the other video?" -- X is excluded, not the filter;
        # still worth a plan so _video_results drops it (round 4).
        excluded = query_planner.excluded_video_id(question, catalog)
        if excluded:
            video_id = None
    except Exception as exc:
        logger.warning("video reference lookup failed (%s)", exc)
        return None
    if not video_id and not excluded:
        return None
    # "<full title> tell me about this" asks for the whole video: its overview,
    # not the keyframes whose title text happens to match best (2026-10-29).
    intent = ("summary" if _SUMMARY_CUE_RE.search(question)
              or query_planner.asks_about_titled_video(question, catalog) else "passage")
    return query_planner.VideoQueryPlan([question], None, intent)


def _without_video(candidates: list, video_id) -> list:
    """candidates minus the excluded video's ("the other video", round 4)."""
    if not video_id:
        return candidates
    return [c for c in candidates if (c.get("payload") or {}).get("video_id") != video_id]


def _clock(seconds) -> str:
    """9:11 under an hour, 1:07:55 over -- how titles and users write lengths."""
    s = int(seconds or 0)
    return "%d:%02d" % (s // 60, s % 60) if s < 3600 else _hms(s)


def _past_end_note(video_id: str, title: str, duration: float, asked: int, reread) -> dict:
    """The note result for a time past the end of a named video. Shaped like the
    library result (source_type video_library): rag_filter renders that as a
    video hit, so the turn stays strict -- a new source_type it does not know
    would not (VIDEO_SOURCE_TYPES in rag_filter.py)."""
    text = "%s is %s long; %s is past the end of it." % (title, _clock(duration), _hms(asked))
    if reread is not None:
        text += (" Read as %s (minutes:seconds), it falls inside the video; the passages there follow."
                 % _clock(reread))
    return {"id": "video-note-%s" % video_id, "collection": config.VIDEO_TRANSCRIPT_COLLECTION,
            "score": 1.0, "rrf_score": 1.0,
            "payload": {"text": text, "source_type": "video_library", "video_id": video_id,
                        "title": "%s -- length %s" % (title, _clock(duration))}}


def _score(c) -> float:
    return float(c.get("score") or 0.0)


def _order_score(c) -> float:
    """What video results are ORDERED by: the reranker's score, or the spoken-
    preference score _video_floor set. Never used for the floor."""
    return float(c.get("rank_score", _score(c)))


def _video_floor(ranked: list, top_n: int, named: bool = False, spoken_first: bool = False) -> list:
    """One sub-query's reranked video candidates (best first) -> what it returns.

    Kept: everything at or above VIDEO_MIN_RERANK_SCORE; for a named video also
    its top NAMED_VIDEO_KEEP down to NAMED_VIDEO_MIN_SCORE. Then, unless the
    question is visual, transcript passages are ordered SPOKEN_PREFERENCE higher
    (a copy carries rank_score; "score" stays the reranker's, which is what
    /search reports and rag_filter's own min_relevance_score reads). Cut to top_n."""
    kept = [c for c in ranked if _score(c) >= VIDEO_MIN_RERANK_SCORE]
    if named:
        kept += [c for c in ranked[:NAMED_VIDEO_KEEP]
                 if NAMED_VIDEO_MIN_SCORE <= _score(c) < VIDEO_MIN_RERANK_SCORE]
    if spoken_first:
        kept = [({**c, "rank_score": _score(c) + SPOKEN_PREFERENCE}
                 if (c.get("payload") or {}).get("source_type") == "transcript_segment" else c)
                for c in kept]
        kept.sort(key=_order_score, reverse=True)
    return kept[:top_n]


def _clears_video_floor(groups: list) -> bool:
    """Whether any result is relevant on its own merits -- named-video keeps
    below the floor don't count (they are what I-6's fallback adds to)."""
    return any(_score(c) >= VIDEO_MIN_RERANK_SCORE for g in groups for c in g)


async def _with_following_windows(groups: list) -> list:
    """I-7: the best transcript passage among the results, followed by the next
    FOLLOWUP_WINDOWS windows of the same video (within FOLLOWUP_SPAN_S). They
    take the matched passage's score -- they are looked up because of it, not
    ranked -- and sit right after it in its group. Lookup failure or no
    transcript passage: groups unchanged."""
    best, where = None, None
    for gi, g in enumerate(groups):
        for ci, c in enumerate(g):
            pl = c.get("payload") or {}
            if pl.get("source_type") != "transcript_segment" or pl.get("start_time") is None \
                    or not pl.get("video_id") or pl.get("overview"):
                continue
            if best is None or _score(c) > _score(best):
                best, where = c, (gi, ci)
    if best is None:
        return groups
    pl = best["payload"]
    start = float(pl["start_time"])
    try:
        points, _ = await get_qdrant().scroll(
            config.VIDEO_TRANSCRIPT_COLLECTION, limit=20,
            filters={"video_id": pl["video_id"], "start_time": {"$gt": start, "$lte": start + FOLLOWUP_SPAN_S}})
    except Exception as exc:
        logger.warning("follow-up window lookup failed for %s @%ss (%s)", pl["video_id"], start, exc)
        return groups
    have = {(str((c.get("payload") or {}).get("video_id")), (c.get("payload") or {}).get("start_time"))
            for g in groups for c in g}
    points = sorted((p for p in points if not (p.get("payload") or {}).get("overview")),
                    key=lambda p: float(p["payload"].get("start_time") or 0))
    nxt = [{"id": p["id"], "collection": config.VIDEO_TRANSCRIPT_COLLECTION, "payload": p["payload"],
            "score": _score(best), "rank_score": _order_score(best), "rrf_score": 1.0}
           for p in points[:FOLLOWUP_WINDOWS]
           if (str(p["payload"].get("video_id")), p["payload"].get("start_time")) not in have]
    if not nxt:
        return groups
    gi, ci = where
    groups = [list(g) for g in groups]
    groups[gi][ci + 1:ci + 1] = nxt
    return groups


LIBRARY_LIST_MAX = 25


def _hms(seconds) -> str:
    s = int(seconds or 0)
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


# Overview headings come in whatever list style Gemini picked for that video
# (all three seen live, 2026-10-20): "### 1. Heading", "* **Heading:** ...",
# "1. **Heading** [0:08:28]".
_HEADING_RES = (
    re.compile(r"^#{2,4}\s*(?:\d+[.)]\s*)?(.+?)\s*$"),
    re.compile(r"^(?:\d+[.)]|[*-])\s*\*\*(.+?)\*\*"),
)


def _overview_headings(text: str) -> list:
    heads = []
    for line in text.splitlines():
        line = line.strip()
        for rx in _HEADING_RES:
            m = rx.match(line)
            if m:
                heads.append(re.sub(r"\s*\[[0-9:]+\]\s*$", "", m.group(1)).strip(" :*"))
                break
    return [h for h in heads if h]


def _library_text(videos: list, total: int, excluding: bool = False) -> str:
    """The library answer: every indexed video (up to LIBRARY_LIST_MAX) with
    its duration and the main-concept headings of its overview. excluding: the
    list leaves out the video the user already asked about ("the other video")."""
    lines = ["The video knowledge base contains %d ingested video%s (their transcripts, chart frames "
             "and chart descriptions are searchable):" % (total, "" if total == 1 else "s")]
    if excluding:
        # Total stays the real count: "I only have one video" was the bug (round 4).
        lines = ["The video knowledge base contains %d ingested videos. The %d other than the one "
                 "already asked about (ask the user which one they mean):" % (total, len(videos))]
    for n, v in enumerate(videos[:LIBRARY_LIST_MAX], 1):
        heads = _overview_headings(v.get("text") or "")
        line = "%d. %s (%s)" % (n, v.get("title") or "untitled", _hms(v.get("end_time")))
        if heads:
            line += " - main concepts: " + "; ".join(heads)
        lines.append(line)
    if len(videos) > LIBRARY_LIST_MAX:
        lines.append("...and %d more" % (len(videos) - LIBRARY_LIST_MAX))
    return "\n".join(lines)


async def _library_result(exclude: str = None):
    """One result describing the ingested video collection, or None when no
    video is indexed. Score 1.0: it answers a question about the collection
    itself, so passage relevance doesn't apply. exclude: a video_id left out of
    the list ("the other video")."""
    from pipelines import query_planner
    try:
        indexed = await query_planner.indexed_videos()
        if not indexed:
            return None
        overviews, offset = {}, None
        while True:
            points, offset = await get_qdrant().scroll(config.VIDEO_TRANSCRIPT_COLLECTION, limit=256,
                                                       offset=offset, filters={"overview": True})
            for p in points:
                overviews[p["payload"].get("video_id")] = p["payload"]
            if offset is None:
                break
    except Exception as exc:
        logger.warning("video library lookup failed (%s)", exc)
        return None
    videos = [dict(overviews.get(vid) or {}, title=title) for vid, title in indexed if vid != exclude]
    if not videos:
        return None
    excluding = len(videos) < len(indexed)
    title = ("Video library (%d other video%s)" % (len(videos), "" if len(videos) == 1 else "s") if excluding
             else "Video library (%d video%s)" % (len(indexed), "" if len(indexed) == 1 else "s"))
    return {"id": "video-library", "collection": config.VIDEO_TRANSCRIPT_COLLECTION, "score": 1.0,
            "rrf_score": 1.0,
            "payload": {"text": _library_text(videos, len(indexed), excluding), "source_type": "video_library",
                        "title": title}}


# Half-width of the moment a timestamp names; same as routers/search.py's ts_window.
MOMENT_WINDOW_S = 30
MOMENT_MAX_PASSAGES = 3


async def _passages_at(video_id: str, at_seconds: int) -> list:
    """The transcript windows of one video that overlap at_seconds +/-
    MOMENT_WINDOW_S, in time order (closest-starting first when capped), or []
    on failure / nothing there. Score 1.0 like the overview: the user named the
    moment, so passage similarity and the relevance floor don't apply."""
    lo, hi = at_seconds - MOMENT_WINDOW_S, at_seconds + MOMENT_WINDOW_S
    try:
        points, _ = await get_qdrant().scroll(
            config.VIDEO_TRANSCRIPT_COLLECTION, limit=50,
            filters={"video_id": video_id, "start_time": {"$lte": hi}, "end_time": {"$gte": lo}})
    except Exception as exc:
        logger.warning("moment lookup failed for %s @%ss (%s)", video_id, at_seconds, exc)
        return []
    # The overview spans the whole video, so the range matches it too.
    points = [p for p in points if not (p.get("payload") or {}).get("overview")]
    points.sort(key=lambda p: abs(float(p["payload"].get("start_time") or 0) - at_seconds))
    points = sorted(points[:MOMENT_MAX_PASSAGES], key=lambda p: float(p["payload"].get("start_time") or 0))
    out = [{"id": p["id"], "collection": config.VIDEO_TRANSCRIPT_COLLECTION,
            "payload": p["payload"], "score": 1.0, "rrf_score": 1.0} for p in points]
    return out + await _frames_at(video_id, at_seconds)


MOMENT_MAX_FRAMES = 3


async def _frames_at(video_id: str, at_seconds: int) -> list:
    """Chart descriptions AND on-screen text of one video within at_seconds +/-
    MOMENT_WINDOW_S, closest first, at most MOMENT_MAX_FRAMES of each. "What does
    he draw on the chart around 1:08:00?" is answered by the chart, not the
    transcript -- the moment lookup returned only transcript windows (the client,
    2026-10-22). The OCR text is what carries exact numbers: "the entry price
    shown in the order window at 0:20" got only the description ("an order
    window") and the answer said the price wasn't in the knowledge base, while
    the frame's text read "Short 3 @ 4 223.75" (2026-10-29). Frames store
    frame_ms (milliseconds); transcripts store seconds."""
    lo, hi = (at_seconds - MOMENT_WINDOW_S) * 1000, (at_seconds + MOMENT_WINDOW_S) * 1000
    out = []
    # od_detections holds the chart description, ocr_text the screen text
    # (video_ingestion.py writes those keys).
    for collection, key in ((config.VIDEO_OD_COLLECTION, "od_detections"),
                            (config.VIDEO_OCR_COLLECTION, "ocr_text")):
        try:
            points, _ = await get_qdrant().scroll(
                collection, limit=200,
                filters={"video_id": video_id, "frame_ms": {"$gte": lo, "$lte": hi}})
        except Exception as exc:
            logger.warning("frame lookup failed for %s @%ss in %s (%s)", video_id, at_seconds, collection, exc)
            continue
        points = [p for p in points if ((p.get("payload") or {}).get(key) or "").strip()]
        points.sort(key=lambda p: abs(float(p["payload"].get("frame_ms") or 0) / 1000 - at_seconds))
        points = _distinct_text(points, key)[:MOMENT_MAX_FRAMES]
        points.sort(key=lambda p: float(p["payload"].get("frame_ms") or 0))
        out += [{"id": p["id"], "collection": collection,
                 "payload": p["payload"], "score": 1.0, "rrf_score": 1.0} for p in points]
    return out


def _distinct_text(points: list, key: str) -> list:
    """points minus repeats of the same text (consecutive keyframes of one
    screen read identically), keeping the first of each."""
    seen, out = set(), []
    for p in points:
        text = " ".join(str(p["payload"].get(key) or "").split())
        if text not in seen:
            seen.add(text)
            out.append(p)
    return out


# "Which price levels are shown", "the exact entry price": a question about the
# numbers on screen. Those live in the OCR text; similarity search ranked the
# chart descriptions ("a high near 4227.50") above it, so the answer gave only
# approximate levels for a video whose frames read 4223.75 exactly (2026-10-29).
_PRICE_QUESTION_RE = re.compile(
    r"\b(?:price|prices|level|levels|entry|exit|fill|stop|target|value|values|number|numbers|"
    r"profit|loss|p&l|pnl|high|low)\b", re.I)
_PRICE_TOKEN_RE = re.compile(r"\d[\d ,]*\.\d{2}\b")
PRICE_READINGS_MAX = 8


async def _price_readings(video_id: str) -> list:
    """The named video's on-screen text that carries prices, distinct screens
    only, spread across the video in time order, at most PRICE_READINGS_MAX.
    Score 1.0 like a named moment: the user asked for this video's numbers."""
    try:
        points, _ = await get_qdrant().scroll(config.VIDEO_OCR_COLLECTION, limit=500,
                                              filters={"video_id": video_id})
    except Exception as exc:
        logger.warning("price readings lookup failed for %s (%s)", video_id, exc)
        return []
    points = [p for p in points if _PRICE_TOKEN_RE.search((p.get("payload") or {}).get("ocr_text") or "")]
    points.sort(key=lambda p: float(p["payload"].get("frame_ms") or 0))
    points = _distinct_text(points, "ocr_text")
    if len(points) > PRICE_READINGS_MAX:
        step = len(points) / PRICE_READINGS_MAX
        points = [points[int(i * step)] for i in range(PRICE_READINGS_MAX)]
    return [{"id": p["id"], "collection": config.VIDEO_OCR_COLLECTION,
             "payload": p["payload"], "score": 1.0, "rrf_score": 1.0} for p in points]


async def _fetch_overview(video_id: str):
    """The video's overview point (P152 stage B), or None. Score 1.0: it is
    looked up because the user asked for this video's summary, not ranked by
    similarity, so the relevance floor doesn't apply to it."""
    try:
        points, _ = await get_qdrant().scroll(config.VIDEO_TRANSCRIPT_COLLECTION, limit=1,
                                              filters={"video_id": video_id, "overview": True})
    except Exception as exc:
        logger.warning("overview lookup failed for %s (%s)", video_id, exc)
        return None
    if not points:
        return None
    p = points[0]
    return {"id": p["id"], "collection": config.VIDEO_TRANSCRIPT_COLLECTION,
            "payload": p["payload"], "score": 1.0, "rrf_score": 1.0}


def _pinned(c) -> bool:
    """A result _video_results puts first on purpose: a video's overview, the
    library list, the past-end note (source_type video_library)."""
    pl = c.get("payload") or {}
    return bool(pl.get("overview")) or pl.get("source_type") == "video_library"


def _merge(text_results: list, video_groups: list, video_cap: int) -> list:
    """Text results unchanged and first, then up to video_cap video results:
    the best of each sub-query first (so topic 3 of a 3-part question isn't
    crowded out by topics 1-2), strongest of those first, then the rest by
    score; de-duplicated."""
    out = list(text_results)
    seen = {(c.get("collection"), str(c.get("id"))) for c in out}
    added = 0

    def _add(c):
        nonlocal added
        key = (c.get("collection"), str(c.get("id")))
        if key not in seen and added < video_cap:
            seen.add(key)
            out.append(c)
            added += 1

    # The heads go strongest first, not in sub-query order (H2, 2026-10-25): "what
    # is an SLA" planned ["what is an SLA", "why does the market
    # create an SLA"]; the first query's best was a March 27 passage at
    # 0.70, the second's Episode 12 1:07:55 at 0.89, and sub-query order put March
    # 27 on top -- in one of two chats the model never cited Episode 12. A pinned
    # result (overview, library list, past-end note) still leads: those are inserted
    # first on purpose, and a passage's rank_score (spoken preference) can exceed 1.0.
    heads = [g[0] for g in video_groups if g]
    heads.sort(key=lambda c: (not _pinned(c), -_order_score(c)))
    for c in heads:
        _add(c)
    for c in sorted((c for g in video_groups for c in g[1:]), key=_order_score, reverse=True):
        _add(c)
    return out


async def _rerank(query: str, candidates: list[dict],
                  top_n: int) -> tuple[list[dict], bool]:
    """Reranking is a quality boost, not a correctness requirement — if the
    reranker API fails, fall back to RRF order rather than returning nothing.

    Returns (results, scores_are_relevance). The flag is what lets the caller
    apply MIN_RERANK_SCORE: RRF scores are ~1/RRF_K and NoopReranker's are
    positions, so thresholding either would empty the knowledge base on exactly
    the two paths that exist to keep it working.
    """
    texts = [_rerank_text(c["payload"]) for c in candidates]
    reranker = create_reranker()
    try:
        ranking = await reranker.rerank(query, texts, top_n)
    except Exception as exc:
        logger.warning("reranker failed (%s) — falling back to RRF order", exc)
        return [{**c, "score": c["rrf_score"]} for c in candidates[:top_n]], False

    return ([{**candidates[i], "score": score} for i, score in ranking
             if i < len(candidates)],
            getattr(reranker, "calibrated", False))


def _drop_irrelevant(query: str, reranked: list[dict], sp=None) -> list[dict]:
    """Keep only chunks the reranker judged relevant.

    Returning nothing is a legitimate answer: the filter says so out loud
    ("No relevant information was found in the knowledge base") and /search logs
    the empty result as a coverage gap. Citing an unrelated document instead is
    strictly worse — the answer looks sourced when it is not.

    `sp` is optional and only used to record the *.decision/*.reason attributes
    on the caller's span — keeps this function testable/callable without a span,
    same as before this existed.
    """
    kept = [c for c in reranked if float(c.get("score") or 0.0) >= MIN_RERANK_SCORE]
    if len(kept) != len(reranked):
        logger.info("relevance floor: kept %d/%d results >= %.2f for '%s'",
                    len(kept), len(reranked), MIN_RERANK_SCORE, query[:80])
    if sp is not None:
        reason = (f"kept {len(kept)}/{len(reranked)} results >= {MIN_RERANK_SCORE:.2f}"
                  if reranked else "reranker returned no candidates to filter")
        record_decision(sp, "retrieval", "answered" if kept else "declined", reason)
    return kept


async def expand_results(reranked_chunks: list[dict], collection: str = None,
                         max_expansion: int = EXPANSION_MAX_CHUNKS) -> list[dict]:
    """Rerank winners plus every sibling chunk from the same source documents.

    Winners keep their rerank score and are marked `is_primary` in their payload
    so citation formatting can tell what the reranker actually chose. The result
    is grouped by source — best-ranked source first — and ordered by chunk_index
    inside each group, so the model reads each topic in document order.

    `collection` overrides where siblings are looked up; by default each chunk is
    expanded inside the collection it came from, which is what a search across
    both collections needs.

    Siblings inherit their primary's score rather than the 0 the spec asked for:
    rag_filter drops every hit below min_relevance_score (0.3) before it builds
    the context block, so zero-scored siblings would be fetched and then thrown
    away without ever reaching the model.

    max_expansion caps the total; the filter's _fit_context does the real token
    accounting afterwards. Over the cap, every primary is kept and the remaining
    slots go to the siblings nearest a primary by chunk_index.
    """
    if not reranked_chunks:
        return reranked_chunks
    started = time.perf_counter()
    reranked_chunks = reranked_chunks[:max_expansion]

    primaries = []  # [(group key, hit)]
    groups = {}     # (collection, source) -> {rank, score, indexes}
    for hit in reranked_chunks:
        payload = {**(hit.get("payload") or {}), "is_primary": True}
        key = (collection or hit.get("collection"), source_of(payload))
        primaries.append((key, {**hit, "payload": payload}))
        if not all(key):
            continue  # nothing to expand against — a chunk with no source
        group = groups.setdefault(key, {"rank": len(groups), "indexes": [],
                                        "score": float(hit.get("score") or 0.0)})
        group["indexes"].append(chunk_index(payload))

    ordered = [hit for _, hit in primaries]
    if not groups:
        return ordered

    by_collection = {}
    for coll, source in groups:
        by_collection.setdefault(coll, []).append(source)

    qdrant = get_qdrant()
    seen = [str(hit["id"]) for _, hit in primaries]
    try:
        pages = await asyncio.gather(*[
            qdrant.expand_by_source(coll, sources, exclude_ids=seen)
            for coll, sources in by_collection.items()
        ])
    except Exception as exc:
        # Expansion is a completeness boost, not a correctness requirement —
        # same deal as the reranker. The winners alone still answer the question.
        logger.warning("expansion failed (%s) — returning reranked chunks only", exc)
        return ordered

    siblings = []
    for point in [p for page in pages for p in page]:
        key = (point["collection"], source_of(point["payload"]))
        group = groups.get(key)
        if group is None:
            continue
        index = chunk_index(point["payload"])
        distance = min(abs(index - i) for i in group["indexes"])
        siblings.append((distance, group["rank"], index, key, {
            "id": point["id"],
            "collection": point["collection"],
            "score": group["score"],
            "payload": {**point["payload"], "is_primary": False},
        }))
    siblings.sort(key=lambda s: s[:3])

    entries = [(groups[key]["rank"] if key in groups else len(groups),
                chunk_index(hit["payload"]), hit) for key, hit in primaries]
    entries += [(rank, index, hit)
                for _, rank, index, _, hit in siblings[:max_expansion - len(primaries)]]
    entries.sort(key=lambda e: e[:2])

    expanded = [hit for _, _, hit in entries]
    logger.info("expanded %d chunks to %d chunks from %d sources in %.0fms",
                len(primaries), len(expanded), len(groups),
                (time.perf_counter() - started) * 1000)
    return expanded
