"""Turn a chat message into clean video-search queries (P152).

Chat messages arrive as full instructions -- "In the ingested video, find the
section where ... and at what timestamp? Use
only the ingested content" -- and the video leg of retrieval used that whole
text as its query. Measured 2026-10-20 on real questions: the
passage that answers the definition question ranked #25 in vector search with the
full message (#11 with a short query) and reranked 0.654 (0.819 short), so
every one of five real questions returned nothing.

One cheap Gemini call rewrites the message into 1-4 standalone queries (one per
distinct question/topic, instructions stripped), copies the video title the
user names, and says whether the ask is a whole-video summary. Any failure
falls back to the original message as the single query -- today's behaviour.
"""

from __future__ import annotations

import difflib
import contextvars
import logging
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger("haystack-api.query_planner")

MAX_QUERIES = 4
# Flash-Lite: median 1.04 s vs 1.78 s for gemini-3.6-flash on the same four
# real questions, same plans (split compounds, named video, library intent),
# measured 2026-10-20. The planner is on every chat turn, so its latency is
# the chat's latency.
PLANNER_MODEL = "gemini-3.5-flash-lite"
_CACHE_SIZE = 256

# Time budget inside retrieval.PLANNER_BUDGET_S (3.0 s): retrieval stops waiting
# at 3.0 s, so the planner must finish -- Gemini AND the Claude fallback -- before
# that, or the fallback never gets a turn. The genai SDK cannot enforce this itself
# (10 s is the API's minimum deadline, see _gemini_response), so Gemini runs in a
# worker thread and is abandoned after GEMINI_WAIT_S. 1.5 s is ~1.5x Flash-Lite's
# measured median (1.04 s); a Gemini call slower than that is treated as a failure
# and Claude gets what remains of PLANNER_DEADLINE_S.
PLANNER_DEADLINE_S = 2.9
GEMINI_WAIT_S = 1.5
CLAUDE_MIN_S = 0.5            # never send Claude a request with less time than this

# Circuit breaker (2026-10-22 credit outage: every turn first spent a failed Gemini
# call before Claude ran). A billing/auth error opens it at once -- it will not fix
# itself in seconds; any other failure (5xx, SDK errors) opens it after
# GEMINI_TRIP_AFTER consecutive failures. A timeout never counts (round 3: under
# concurrent load it is slowness, not an outage -- see record_failure). While open, Gemini is skipped and Claude gets the whole
# deadline. It closes on its own: once GEMINI_COOLDOWN_S has passed the next turn
# tries Gemini again, and a success resets it. Gemini is never disabled for good.
GEMINI_COOLDOWN_S = 120.0
GEMINI_TRIP_AFTER = 2
_HARD_FAILURE_MARKERS = ("402", "RESOURCE_EXHAUSTED", "401", "403", "PERMISSION_DENIED",
                         "UNAUTHENTICATED", "API_KEY_INVALID")

# Gemini calls abandoned after GEMINI_WAIT_S keep running to the SDK's own 10 s
# deadline. The pool bounds how many can pile up -- but GEMINI_WAIT_S starts at
# submission, so a turn queued behind busy workers spends its whole wait without
# its call ever being sent. Two workers did exactly that with six parallel chats
# (live 2026-10-23 12:11 UTC): four turns "timed out" in the queue, the circuit
# opened, and the planner ran over budget. 16 covers a burst of concurrent turns
# plus the abandoned calls still running out their 10 s; idle threads cost nothing.
_gemini_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="planner-gemini")


@dataclass
class VideoQueryPlan:
    queries: List[str] = field(default_factory=list)
    video_name: Optional[str] = None
    intent: str = "passage"          # "passage" | "summary" | "library"
    # Which model produced the plan: "gemini" | "claude" | "" (no plan -- the
    # original question). Not part of equality: two identical plans are the same
    # plan whoever wrote them; this is for tracing and evaluation only.
    provider: str = field(default="", compare=False)


class _GeminiBreaker:
    """Tracks whether Gemini should be tried. Thread-safe: plan_video_queries runs
    in asyncio.to_thread, so concurrent chat turns share this."""

    def __init__(self, cooldown_s=GEMINI_COOLDOWN_S, trip_after=GEMINI_TRIP_AFTER,
                 clock=time.monotonic):
        self._lock = threading.Lock()
        self._clock = clock
        self.cooldown_s = cooldown_s
        self.trip_after = trip_after
        self.failures = 0
        self.open_until = 0.0
        self.reason = ""

    def allow(self) -> bool:
        with self._lock:
            return self._clock() >= self.open_until

    def record_success(self) -> None:
        with self._lock:
            if self.open_until or self.failures:
                logger.info("query planner: Gemini answered again -- circuit closed")
            self.failures, self.open_until, self.reason = 0, 0.0, ""

    def record_failure(self, exc) -> None:
        # A timeout is slowness, not an outage. Live 2026-10-23 12:11 UTC, six
        # parallel chats: four Gemini waits ran out in two seconds, two "in a row"
        # opened the circuit for 120 s, and every turn after that planned with
        # Claude alone -- while Gemini was answering, just slower than GEMINI_WAIT_S
        # under load. The turn that timed out still gets Claude; the next turn tries
        # Gemini again. Billing/auth errors and real failures (5xx, refusals,
        # exceptions from the SDK) still count.
        if isinstance(exc, TimeoutError):
            return
        text = "%s: %s" % (type(exc).__name__, exc)
        hard = any(m in text for m in _HARD_FAILURE_MARKERS)
        with self._lock:
            self.failures += 1
            if hard or self.failures >= self.trip_after:
                self.open_until = self._clock() + self.cooldown_s
                self.reason = text[:200]
                logger.warning("query planner: Gemini circuit open for %.0fs after %d failure(s) "
                               "(%s) -- planning with Claude until then",
                               self.cooldown_s, self.failures, self.reason)

    def reset(self) -> None:
        with self._lock:
            self.failures, self.open_until, self.reason = 0, 0.0, ""


_breaker = _GeminiBreaker()


def _setting(name: str, default: bool) -> bool:
    import config
    return bool(getattr(config, name, default))


SYSTEM_INSTRUCTION = (
    "You prepare search queries for a library of transcribed instructional videos. "
    "Rewrite the user's chat message into short standalone search queries: one per distinct "
    "question or topic they ask about, at most 4, phrased the way the concept would be spoken "
    "in a lecture. Split a question that asks about two things into two queries "
    "(\"why does X form and how does X work\" -> \"why X forms\" and \"how X works\"). "
    "When they ask what a concept is or what it means, add a query in the way a lecturer "
    "explains it -- why it forms, or how it comes about (\"what is a trend\" -> "
    "\"what is a trend\" and \"why does the market create a trend\"): "
    "these videos explain concepts rather than define them. The same holds when they ask why it "
    "returns to a concept or what happens when it reaches or rebalances it: add \"why does the market "
    "create <the concept>\". "
    # H3 (sprint 2026-10-25), measured on Gemini 2 runs each. m3-042 "In Episode 12, what
    # happens after traffic rebalances the load balancer?" was searched as "what happens after
    # ..." -- those words match every narration, the explanation it follows (1:07:55) ranked
    # 10th in the video, and the what-next windows followed the wrong passage. retrieval adds
    # the following windows itself, so the event alone is the query. m3-089 lost "first
    # presented SLA" to "the service level agreement": the specific passage reranks 0.52 for the general
    # term and 0.84-0.87 with the user's own words. Expanding an abbreviation stays allowed:
    # "RTO" alone (instead of "RTO target") lost m3-063's Episode 12 passage, 3/3 runs.
    "When they ask what happens or happened after something, query that something itself, not the words "
    "\"what happens after\": the search adds the passages that follow it. "
    "Keep the user's own specific names, labels and numbers word for word in the queries (\"the first "
    "observed spike\", \"step 3 above\") -- never replace them with a more general term; adding the "
    "full term after an abbreviation is fine. "
    "When they ask whether the videos show, mention or contain something (on-screen labels, "
    "a diagram, a term), query that thing itself, e.g. \"SLA KPI table labels\". "
    "Remove instructions about format, timestamps, quoting or sourcing. "
    "When they want work done with what the video teaches -- data analysis, code, checking "
    "their file or data -- drop the task and query only the rule or method from the video, "
    "phrased as a plain question about what he did or said on the video, not as a rule, label or "
    "definition (\"analyse my data with the method he teaches\" -> "
    "\"where he explains the first step\" and \"where he explains the exception\"; \"which of my "
    "records break his stated rules\" -> \"when and why he says to stop\"). "
    "When they state a claim and ask you to confirm it (\"he says X, right?\", \"didn't he?\", "
    "\"correct?\"), query the topic in neutral words, not the claim, so the passage is found "
    "whether the claim is true or false (\"he always starts at 9:30, right?\" -> "
    "\"when he starts\" and \"what he says about 9:30\"). "
    "If the user names a specific video, copy its title exactly as they wrote it into "
    "video_name, otherwise null. When they ask about the other video, another video or a "
    "different video, the video they named before it is the one they do NOT mean: never put "
    "it in video_name (null unless they name the other one too). Set intent to \"summary\" only when they ask for an overview, "
    "summary or the main concepts of a whole video. Set intent to \"library\" only when they ask "
    "about the video collection itself -- which or how many videos or transcripts there are, "
    "what the videos cover, or whether there is any knowledge in the videos at all -- not "
    "when they ask whether the videos show or say something specific. "
    "Otherwise \"passage\"."
)


def _generate(prompt: str) -> Optional[dict]:
    """The one network call: Gemini with a structured response. Raises or
    returns None on failure; plan_video_queries() turns both into a fallback.

    Its token usage is recorded for the chat-retrieval cost (P161) -- only
    here, so a cache hit in plan_video_queries() is never billed. Runs in
    asyncio.to_thread, which carries the /search meter's context with it."""
    from pipelines import usage_meter

    response = _gemini_response(prompt)
    usage_meter.record_search_gemini(getattr(response, "usage_metadata", None))
    parsed = response.parsed
    return parsed.model_dump() if parsed is not None else None


_claude_client = None


def _generate_fallback(prompt: str, timeout: Optional[float] = None) -> Optional[dict]:
    """Claude answers the planner prompt when Gemini can't. Raises on failure.

    Not configured (no PLANNER_ANTHROPIC_API_KEY) -> raises, so the caller
    degrades to plain search exactly as before. Usage is metered as its own
    retrieval line (claude_planner). `timeout` is what is left of the planner's
    deadline; None keeps the client's own 2.5 s."""
    from pipelines import usage_meter

    response = _claude_response(prompt, timeout=timeout)
    usage_meter.record_search_claude(getattr(response, "usage", None))
    if getattr(response, "stop_reason", None) == "refusal":
        raise RuntimeError("Claude planner refused")
    parsed = getattr(response, "parsed_output", None)
    return parsed.model_dump() if parsed is not None else None


def _claude_response(prompt: str, timeout: Optional[float] = None):
    """The raw Claude response (tests replace this boundary)."""
    import anthropic
    from pydantic import BaseModel

    import config

    if not config.PLANNER_ANTHROPIC_API_KEY:
        raise RuntimeError("no PLANNER_ANTHROPIC_API_KEY configured")

    class _PlanSchema(BaseModel):
        queries: List[str]
        video_name: Optional[str] = None
        intent: str = "passage"

    # The planner runs inside PLANNER_BUDGET_S (3 s) on every chat turn: a short
    # timeout and no SDK retries, so a slow fallback can't hold the video leg. One
    # client for the process: the first live call took 4.2 s (import + TLS setup),
    # later ones 1.5-1.9 s -- reuse keeps the connection warm.
    global _claude_client
    if _claude_client is None:
        _claude_client = anthropic.Anthropic(api_key=config.PLANNER_ANTHROPIC_API_KEY, timeout=2.5, max_retries=0)
    extra = {"timeout": timeout} if timeout is not None else {}
    return _claude_client.messages.parse(
        model=config.PLANNER_FALLBACK_MODEL,
        max_tokens=400,
        system=SYSTEM_INSTRUCTION,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "low"},
        output_format=_PlanSchema,
        **extra,
    )


def _gemini_response(prompt: str):
    """The raw Gemini response (tests replace this boundary)."""
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    import config

    class _PlanSchema(BaseModel):
        queries: List[str]
        video_name: Optional[str] = None
        intent: str = "passage"

    # Cap a stuck planner so it can't hold a chat turn. 10 s is the Gemini
    # API's minimum deadline: 8 s was rejected outright ("Manually set
    # deadline 8s is too short", 400) on every call, live check 2026-10-20.
    client = genai.Client(http_options=types.HttpOptions(timeout=10000))
    response = client.models.generate_content(
        model=PLANNER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_PlanSchema,
            max_output_tokens=400,
            # Same question -> same plan. At the default temperature it varied
            # between one combined query (answer reranked 0.677, dropped) and a
            # split pair (0.814), live 2026-10-20.
            temperature=0.0,
        ),
    )
    return response


# Task words a lecture transcript never contains. The planner is told to drop
# them, but a query that keeps "... on my file" (M3 Layer 1, 2026-10-22) only
# drags the embedding away from the passage, so they are also removed here.
# Narrow on purpose: "data" or "file" alone can be lecture words ("the data
# shows..."), so only the user-side phrases go.
_TASK_RE = re.compile(
    r"\b(?:(?:in|on|from|against|with|using|of)\s+)?"
    r"(?:my|the user'?s)\s+(?:uploaded\s+|own\s+)?(?:\w+\s+)?(?:files?|data(?:set)?|csv|spreadsheet|records?|logs?)\b",
    re.IGNORECASE)
_TRAILING_RE = re.compile(r"(?:\s+\b(?:on|in|with|from|using|against|of|for|to|and)\b)+$", re.IGNORECASE)


def _clean_query(query: str) -> str:
    q = " ".join(_TASK_RE.sub(" ", query).split())
    return _TRAILING_RE.sub("", q).strip()


_cache: "OrderedDict[str, VideoQueryPlan]" = OrderedDict()
# Plans are made in worker threads (asyncio.to_thread), so concurrent chat turns share this
# cache: without the lock, an evict between get() and move_to_end() raises KeyError and the
# turn loses its plan.
_cache_lock = threading.Lock()


def _call_generate(prompt: str):
    # Looks _generate up at call time, so tests that replace it are honoured.
    return _generate(prompt)


def _generate_bounded(prompt: str, wait_s: float) -> Optional[dict]:
    """_generate, abandoned after wait_s. Raises TimeoutError (or whatever
    _generate raised). The copied context carries the /search usage meter into
    the worker thread, exactly as asyncio.to_thread does for the caller."""
    ctx = contextvars.copy_context()
    future = _gemini_pool.submit(ctx.run, _call_generate, prompt)
    try:
        return future.result(timeout=wait_s)
    except _FutureTimeout:
        raise TimeoutError("Gemini planner exceeded %.1fs" % wait_s) from None


def _cached(question: str) -> Optional[VideoQueryPlan]:
    if not _setting("PLANNER_CACHE_ENABLED", True):
        return None
    with _cache_lock:
        plan = _cache.get(question)
    if plan is None:
        return None
    # A plan Claude wrote while Gemini was down is not reused once Gemini can be
    # tried again: the question gets Gemini's plan, as in normal operation, and
    # that replaces the cached one. While Gemini is still out, Claude's plan stands.
    if plan.provider != "gemini" and _breaker.allow():
        return None
    with _cache_lock:
        if question in _cache:
            _cache.move_to_end(question)
    return plan


def plan_video_queries(question: str) -> VideoQueryPlan:
    """Plan the video leg for one chat message. Never raises.

    Gemini first (bounded to GEMINI_WAIT_S), Claude with whatever is left of
    PLANNER_DEADLINE_S, else the original question. The plan records which
    provider wrote it."""
    hit = _cached(question)
    if hit is not None:
        return hit
    started = time.monotonic()
    fallback = VideoQueryPlan([question], None, "passage")
    raw, provider, gemini_error = None, "", None
    if _breaker.allow():
        try:
            raw = _generate_bounded(question, GEMINI_WAIT_S)
            provider = "gemini"
            _breaker.record_success()
        except Exception as exc:
            gemini_error = exc
            _breaker.record_failure(exc)
            # 2026-10-22: Google's prepaid credit ran out at 11:04 UTC and every video
            # question on every chat model silently lost its plan. Claude answers the
            # same prompt instead (the chats' own Anthropic key), so one provider being
            # down no longer degrades retrieval. Gemini stays primary: same plans as
            # before in normal operation, one consistent search for all four models.
            logger.warning("query planner (Gemini) failed (%s) -- trying the Claude fallback", exc)
    else:
        gemini_error = RuntimeError("Gemini circuit open (%s)" % _breaker.reason)

    if provider != "gemini":
        if not _setting("PLANNER_FALLBACK_ENABLED", True):
            # Evaluation mode: only Gemini may plan, so a Gemini failure shows up
            # as an unplanned search instead of being covered by Claude.
            logger.warning("query planner failed (%s; fallback disabled) -- searching with the "
                           "original message", gemini_error)
            return fallback
        remaining = PLANNER_DEADLINE_S - (time.monotonic() - started)
        try:
            raw = _generate_fallback(question, timeout=max(CLAUDE_MIN_S, remaining))
            provider = "claude"
        except Exception as exc2:
            logger.warning("query planner failed (%s; fallback: %s) -- searching with the original message",
                           gemini_error, exc2)
            return fallback
    if not raw:
        return fallback
    queries = []
    for x in raw.get("queries") or []:
        q = _clean_query(str(x))
        # Cleaning can turn two queries into one ("alert above the SLA" and "...
        # on my sales file"): searching it twice is a wasted search and rerank.
        if q and q.lower() not in (s.lower() for s in queries):
            queries.append(q)
    queries = queries[:MAX_QUERIES]
    if not queries:
        return fallback
    intent = raw.get("intent") if raw.get("intent") in ("passage", "summary", "library") else "passage"
    plan = VideoQueryPlan(queries, (raw.get("video_name") or "").strip() or None, intent, provider)
    if _setting("PLANNER_CACHE_ENABLED", True):
        with _cache_lock:
            _cache[question] = plan               # only real plans: a transient failure isn't remembered
            _cache.move_to_end(question)
            if len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)
    return plan


# ── Offline self-test ──────────────────────────────────────

_NON_WORD = re.compile(r"[^a-z0-9]+")
# "ep12" -> "ep 12": users glue the abbreviation to the number.
_ALNUM_EDGE = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
# Words a user adds around a title ("the ... video", "the video he posted on X")
# that identify no title.
_FILLER = frozenset({"the", "a", "an", "video", "videos", "ingested", "in", "of", "on", "from", "lesson",
                     "lecture", "clip", "post", "posts", "posted", "he", "his", "him", "this",
                     "that", "one", "about", "where", "with", "s"})
_ID_RE = re.compile(r"[A-Za-z0-9_-]{6,}")
# I-1: "Ep 12" / "Ep. 12" / "ep12" never matched "Episode 12" -- titles spell it
# out, and whole-word containment saw "ep" as a word no title has.
_ABBREVIATIONS = {"ep": "episode", "eps": "episode", "epi": "episode", "epis": "episode"}
# I-1: "the X post" / "the twitter video" name a video by where it was posted.
# Kept out of title matching: the X video's title contains "YouTube" ("I don't
# have time for YouTube to compress..."), so "the youtube video" used to resolve
# to the one video that is NOT on YouTube.
_PLATFORM_WORDS = {"x": "twitter", "twitter": "twitter", "tweet": "twitter", "tweets": "twitter",
                   "tweeted": "twitter", "youtube": "youtube", "yt": "youtube"}
# Typos: "episde" / "comentary" / "twiter". 0.8 accepts one dropped or swapped
# letter in a word of 5+ ("episde"~"episode" 0.92) and rejects different words
# that share letters ("march"~"market" 0.55, "maker"~"market" 0.73).
_FUZZY_MIN = 0.8
_FUZZY_MIN_LEN = 5


def _words(text: str) -> List[str]:
    return [w for w in _NON_WORD.split(_ALNUM_EDGE.sub(" ", (text or "").lower())) if w]


def _similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() >= _FUZZY_MIN


def _word_in(word: str, title_words: set) -> bool:
    """`word` is one of `title_words`, allowing a small typo -- never in a number
    (Episode 12 must never match Episode 13, 2025 never 2026)."""
    if word in title_words:
        return True
    if word.isdigit() or len(word) < _FUZZY_MIN_LEN:
        return False
    return any(len(t) >= _FUZZY_MIN_LEN and not t.isdigit() and _similar(word, t) for t in title_words)


def _platform_of(word: str) -> Optional[str]:
    if word in _PLATFORM_WORDS:
        return _PLATFORM_WORDS[word]
    if len(word) >= _FUZZY_MIN_LEN:
        for alias in ("twitter", "youtube"):
            if _similar(word, alias):
                return alias
    return None


def resolve_video_id(video_name: Optional[str], catalog: List[Tuple[str, str]],
                     platforms: Optional[dict] = None) -> Optional[str]:
    """The indexed video a user named, or None when it isn't clear.

    A wrong filter hides the right video entirely, so this is conservative:
    1. a video_id appearing in the text (a pasted URL or id) wins;
    2. platform words ("the X post", "the twitter video") narrow the catalog to
       that platform's videos (`platforms`: video_id -> platform, default the
       map indexed_videos() cached). Named by platform alone, the video is the
       platform's ONLY one; two on it, or no platform data, is None;
    3. the titles containing every other word the user wrote, as whole words
       ("Episode 1" never matches "Episode 12"; "ep" reads as "episode"; a
       one-letter typo in a 5+ letter word still counts) -- exactly one wins,
       more than one is ambiguous and resolves to None;
    4. extra words that are in no title ("March 27 trade", "the March 27
       stream"): the title matching the most of the user's words wins if it is
       the only best one AND, with no number written, matches at least two
       words and more than it misses ("Acme weekly maker model" matches
       "acme"+"weekly" of the March 27 title and misses two: None);
    5. otherwise the closest title by similarity, accepted only at >= 0.6 and
       >= 0.1 ahead of the runner-up. (A ratio margin alone can't separate
       "Episode 12" from "Episode 36" -- 0.93 apart -- which is why whole-word
       containment comes first.)
    In 4 and 5 a number the user wrote that a title lacks rules that title out.
    """
    if not video_name or not catalog:
        return None
    for token in _ID_RE.findall(video_name):
        for vid, _ in catalog:
            if token == vid:
                return vid
    platform, name_words = None, []
    for w in _words(video_name):
        if w in _FILLER:
            continue
        p = _platform_of(w)
        if p:
            platform = platform or p
        else:
            name_words.append(_ABBREVIATIONS.get(w, w))
    if platform:
        if platforms is None:
            platforms = _indexed_cache.get("platforms") or {}
        if platforms:
            catalog = [(vid, t) for vid, t in catalog if platforms.get(vid) == platform]
            if not name_words:
                return catalog[0][0] if len(catalog) == 1 else None
        # No platform data: a title alongside the platform word still resolves
        # by title; the platform word alone can't say which video.
    if not name_words or not catalog:
        return None
    titled = [(vid, set(_words(title))) for vid, title in catalog]
    contained = [vid for vid, tw in titled if all(_word_in(w, tw) for w in name_words)]
    if len(contained) == 1:
        return contained[0]
    if len(contained) > 1:
        return None
    # A number the user wrote that a title lacks rules that title out: "March
    # 20" fuzzy-matched the ingested "March 27" at ~0.93, "2025" matched "2026"
    # (P152 review) -- a dated or numbered video that isn't ingested.
    digits = {w for w in name_words if w.isdigit()}
    others = [w for w in name_words if not w.isdigit()]
    eligible = [(vid, tw) for vid, tw in titled if digits <= tw]
    hits = sorted(((sum(_word_in(w, tw) for w in others), vid) for vid, tw in eligible), reverse=True)
    if hits:
        best, vid = hits[0]
        runner_up = hits[1][0] if len(hits) > 1 else -1
        if best > runner_up and (digits or (best >= 2 and best > len(others) - best)):
            return vid
    name = " ".join(name_words)
    scored = sorted(((difflib.SequenceMatcher(None, name, " ".join(_words(title))).ratio(), vid)
                     for vid, title in catalog if digits <= set(_words(title))), reverse=True)
    if not scored:
        return None
    best, vid = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    return vid if best >= 0.6 and best - runner_up >= 0.1 else None


# References to one video inside the question itself. The planner copies a
# video_name only when the user calls something a video: "Summarize the video he
# posted on X" came back video_name=null (no overview fetched, 0 results) and
# "Was the March 27 trade profitable?" searched all four videos, leaving the
# right window (0.53) under the unnamed floor (M3 Layer 1, 2026-10-23). A dated
# stream, an episode number and a platform post each identify a video. Bare "X"
# never counts ("he says X, right?"): only "posted on X", "the X post/video".
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
          r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_REFERENCE_RES = (
    re.compile(r"\b%s\.?\s+\d{1,2}(?:st|nd|rd|th)?\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?%s\b"
               % (_MONTH, _MONTH), re.I),
    re.compile(r"\b(?:episode|ep)\.?\s*\d{1,3}\b", re.I),
    re.compile(r"\b(?:posted|post|video|clip|tweeted)\s+(?:on|to|from)\s+(?:x|twitter)\b"
               r"|\b(?:x|twitter)\s+(?:post|video|clip)\b|\btwitter\b|\btweet(?:ed|s)?\b", re.I),
)


# A video named by its TITLE (2026-10-29): "* **Pattern Recognition - Fiber OTE
# NYO and Asian Session [aQrd75xwBS4]** tell me about this" referenced no date,
# episode or platform, so when the planner ran over budget nothing filtered to
# the video and the chat answered from five bare keyframes. Titles are matched
# without markdown, punctuation or the "2017-10-10 - " date prefix.
_TITLE_NOISE_RE = re.compile(r"[^a-z0-9]+")
_TITLE_DATE_PREFIX_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*-\s*")
_BRACKET_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{6,20})\]")
_MIN_TITLE_CHARS = 15      # shorter titles ("Episode 12") collide with ordinary words


def _norm_title(text: str) -> str:
    return " %s " % _TITLE_NOISE_RE.sub(" ", (text or "").lower()).strip()


def titled_video_id(question: str, catalog: List[Tuple[str, str]]) -> Optional[str]:
    """The one indexed video whose title (or bracketed id, "[aQrd75xwBS4]") the
    question contains; None when none or more than one match."""
    q = _norm_title(question)
    ids = {m.group(1) for m in _BRACKET_ID_RE.finditer(question or "")}
    found = set()
    for vid, title in catalog:
        title = title or ""
        if ids & {m.group(1) for m in _BRACKET_ID_RE.finditer(title)} or vid in ids:
            found.add(vid)
            continue
        for variant in (title, _TITLE_DATE_PREFIX_RE.sub("", title), _BRACKET_ID_RE.sub("", title),
                        _BRACKET_ID_RE.sub("", _TITLE_DATE_PREFIX_RE.sub("", title))):
            v = _norm_title(variant)
            if len(v.strip()) >= _MIN_TITLE_CHARS and v in q:
                found.add(vid)
                break
    return found.pop() if len(found) == 1 else None


# What is left of "<title> tell me about this" once the title is removed: a
# request for the whole video, answered by its overview.
_ABOUT_THIS_RE = re.compile(
    r"^(?:(?:can you |please |could you )?(?:tell me|talk) (?:more )?about (?:this|it|that)(?: video)?"
    r"|what(?:'s| is) (?:this|it|that)(?: video)? about|explain (?:this|it|that)(?: video)?"
    r"|(?:give me )?(?:a )?(?:summary|overview|rundown)(?: of (?:this|it))?|what does (?:he|it) (?:cover|teach)"
    r"|)\??$", re.I)


def asks_about_titled_video(question: str, catalog: List[Tuple[str, str]]) -> bool:
    """True when the question names one video by title and asks nothing more
    specific than "tell me about this"."""
    vid = titled_video_id(question, catalog)
    if not vid:
        return False
    title = dict(catalog).get(vid) or ""
    rest = _norm_title(question)
    for variant in (title, _TITLE_DATE_PREFIX_RE.sub("", title)):
        rest = rest.replace(_norm_title(_BRACKET_ID_RE.sub("", variant)), " ")
        rest = rest.replace(_norm_title(variant), " ")
    for m in _BRACKET_ID_RE.finditer(question or ""):
        rest = rest.replace(_norm_title(m.group(1)), " ")
    return bool(_ABOUT_THIS_RE.match(" ".join(rest.split())))


def implied_video_id(question: str, catalog: List[Tuple[str, str]],
                     platforms: Optional[dict] = None) -> Optional[str]:
    """The one indexed video the question refers to by date, episode, platform or
    title, for when the planner named none. Every reference must resolve (an
    un-ingested "Episode 5" beside "March 27" is not a filter on March 27) and to
    the same video: "Episode 12 with the March 27 commentary" compares two -> None."""
    titled = titled_video_id(question, catalog)
    found = {titled} if titled else set()
    for pattern in _REFERENCE_RES:
        for m in pattern.finditer(question or ""):
            vid = resolve_video_id(m.group(0), catalog, platforms)
            if vid is None:
                return None
            found.add(vid)
    return found.pop() if len(found) == 1 else None


def referenced_video_ids(question: str, catalog: List[Tuple[str, str]],
                         platforms: Optional[dict] = None) -> set:
    """Every indexed video the question references by date, episode or platform
    (unresolvable references ignored). Two or more = a cross-video question: the
    planner's video_name must not narrow it to one (m3-053)."""
    found = set()
    for pattern in _REFERENCE_RES:
        for m in pattern.finditer(question or ""):
            vid = resolve_video_id(m.group(0), catalog, platforms)
            if vid is not None:
                found.add(vid)
    return found


_DANGLING_RE = re.compile(r"\s*\b(?:in|on|from|of|during|for)(?:\s+the)?\s*$", re.I)


def strip_video_references(queries: List[str]) -> List[str]:
    """Sub-queries with the date/episode/platform references removed, for a video
    implied_video_id() already turned into a filter. A query that would be left
    empty is kept whole; duplicates left behind are dropped."""
    out = []
    for q in queries:
        stripped = q
        for pattern in _REFERENCE_RES:
            stripped = pattern.sub(" ", stripped)
        # "service level agreement in Episode 12" must not leave "... agreement in".
        stripped = _DANGLING_RE.sub("", " ".join(stripped.split())).strip() or q
        if stripped.lower() not in (o.lower() for o in out):
            out.append(stripped)
    return out


# H2 (sprint 2026-10-25): lecture videos explain a concept -- why it comes about --
# rather than define it, so "what is <concept>" never reached the lecturer's
# explanation in a measured case while the "why does ... create <concept>" query
# ranked it 1st (0.89; a "why does price come back to <concept>" query 25th at 0.47).
# The planner prompt asks for that query, but a prompt is followed most of the time,
# not always, and an over-budget planner (live 2026-10-25 18:50, a definition
# question) yields no plan at all. lecture_queries() adds it in code.
# Narrow on purpose: an article-led noun phrase after "what is" ("what is he
# forecasting", "what is on the chart" stay untouched), at most _LECTURE_MAX_WORDS
# words, cut at the clause end or a sourcing tail ("according to the videos").
_CONCEPT = r"((?:an?|the)\s+[a-z][a-z0-9'/\-]*(?:\s+[a-z0-9'/\-]+)*?)"
_CONCEPT_END = (r"(?=\s*(?:$|[?,.;:!]|\s(?:and|or|but|according|in\s+(?:the\s+)?(?:videos?|lectures?)"
                r"|from\s+the|as\s+(?:he|she|they)|that|which|when|if)\b))")
_LECTURE_RES = (
    re.compile(r"\bwhat(?:'s|\s+is|\s+exactly\s+is)\s+" + _CONCEPT + _CONCEPT_END, re.I),
    re.compile(r"\b(?:define|definition\s+of|meaning\s+of)\s+" + _CONCEPT + _CONCEPT_END, re.I),
    re.compile(r"\bwhy\s+(?:does|do|would|will)\s+price\s+(?:come\s+back|return|retrace|go\s+back)\s+(?:in)?to\s+"
               + _CONCEPT + _CONCEPT_END, re.I),
)
_LECTURE_MAX_WORDS = 6
# "what is the difference between ...", "what is the price at ...": questions about a
# fact or a comparison, not about a concept the lecturer explains.
_NOT_CONCEPTS = frozenset({"difference", "differences", "point", "purpose", "reason", "way", "best", "time", "date",
                           "price", "prices", "level", "video", "videos", "chart", "speaker", "setup", "trade",
                           "result", "timestamp", "name", "title", "range", "number", "percentage", "same",
                           "first", "last", "next", "main", "other", "rule", "rules"})


def lecture_queries(question: str) -> List[str]:
    """For a question asking what a concept is (or why price returns to it): the
    query in the way the lecturer explains it, "why does the market create <it>".
    [] for anything else."""
    out = []
    for pattern in _LECTURE_RES:
        for m in pattern.finditer(question or ""):
            concept = " ".join(m.group(1).lower().split())
            if concept.split()[1] in _NOT_CONCEPTS:
                continue
            if concept.startswith("the "):
                rest = concept[4:]
                concept = ("an " if rest[0] in "aeiou" else "a ") + rest
            if len(concept.split()) > _LECTURE_MAX_WORDS:
                continue
            q = "why does the market create " + concept
            if q not in out:
                out.append(q)
    return out


def with_lecture_queries(question: str, queries: List[str]) -> List[str]:
    """queries plus lecture_queries(question) not already among them, within
    MAX_QUERIES (the planner's own queries give way from the end)."""
    have = {q.lower() for q in queries}
    extra = [q for q in lecture_queries(question) if q.lower() not in have][:MAX_QUERIES - 1]
    return list(queries[:max(1, MAX_QUERIES - len(extra))]) + extra


# Round 4 (live 2026-10-23): rag_filter 1.33 searches a follow-up as "<previous
# message> <follow-up>", so "and in the other video?" arrived as "Summarize the video
# he posted on X. and in the other video?" -- the X reference resolved, the search
# was filtered to the X video, and both Claude models said "I only have one video".
# Narrow on purpose: "the other side", "on the other hand", "another load balancer",
# "another one" (an example, a step) are lecture phrasing, not a second video. "the other
# one" is kept: it only matters when a video was named before it (excluded_video_id).
_OTHER_VIDEO_RE = re.compile(
    r"\b(?:the\s+other|another|a\s+different)\s+(?:videos?|episodes?|clips?|streams?|posts?|recordings?)\b"
    r"|\bthe\s+other\s+one\b", re.I)


def other_video_reference(question: str):
    """The "the other video / another video / a different video / the other one"
    match in the question, or None."""
    return _OTHER_VIDEO_RE.search(question or "")


def excluded_video_id(question: str, catalog: List[Tuple[str, str]], video_name: Optional[str] = None,
                      platforms: Optional[dict] = None) -> Optional[str]:
    """The video the user does NOT mean: the one referenced before "the other
    video" ("Compare Episode 12 with the other video" -> Episode 12). Read off the
    text before the phrase first, then the planner's video_name (the planner may
    copy the excluded video there despite its rule). None without the phrase or
    without a resolvable video before it."""
    m = other_video_reference(question)
    if not m:
        return None
    vid = implied_video_id(question[:m.start()], catalog, platforms)
    if vid is None and video_name:
        vid = resolve_video_id(video_name, catalog, platforms)
    return vid


_indexed_cache = {"at": 0.0, "videos": [], "platforms": {}, "durations": {}}
_INDEXED_TTL_S = 300.0


async def indexed_videos() -> List[Tuple[str, str]]:
    """(video_id, title) of every video with transcript points, cached 5 min.
    The same scroll caches each video's platform (video_platforms()) and
    duration (video_duration()): every point carries `platform`, and the
    largest `end_time` is where the transcript ends -- no second query.
    [] on any Qdrant error -- no filter is the safe failure."""
    import time

    import config
    from services.qdrant_service import get_qdrant

    if time.monotonic() - _indexed_cache["at"] < _INDEXED_TTL_S and _indexed_cache["videos"]:
        return _indexed_cache["videos"]
    try:
        qs, seen, plats, ends, offset = get_qdrant(), {}, {}, {}, None
        while True:
            body = {"limit": 1000, "offset": offset, "with_vector": False,
                    "with_payload": ["video_id", "title", "platform", "end_time"]}
            r = await qs._request("POST", "/collections/%s/points/scroll"
                                  % config.VIDEO_TRANSCRIPT_COLLECTION, json=body)
            for pt in r.get("points", []):
                pl = pt.get("payload") or {}
                vid = pl.get("video_id")
                if not vid:
                    continue
                seen.setdefault(vid, pl.get("title") or "")
                if pl.get("platform"):
                    plats.setdefault(vid, str(pl["platform"]).lower())
                try:
                    ends[vid] = max(ends.get(vid, 0.0), float(pl.get("end_time") or 0.0))
                except (TypeError, ValueError):
                    pass
            offset = r.get("next_page_offset")
            if offset is None:
                break
    except Exception as exc:
        logger.warning("could not list indexed videos (%s) -- no video filter", exc)
        return []
    new = sorted(seen.items())
    if {v for v, _ in new} != {v for v, _ in _indexed_cache["videos"]}:
        # Plans are cached by question text and name the videos they search;
        # one made from the old list would keep steering that question to it.
        with _cache_lock:
            _cache.clear()
    _indexed_cache.update(at=time.monotonic(), videos=new, platforms=plats,
                          durations={v: e for v, e in ends.items() if e > 0})
    return _indexed_cache["videos"]


def invalidate_indexed() -> None:
    """Forget the indexed-video list and every plan made from it.

    Called when the set of searchable videos changes in this process (a video
    deleted, an ingest finished) and by POST /video/planner/refresh after a full
    restore. 2026-10-28: the backend restarted mid-restore, listed the videos
    while the vector store was still the pre-restore one (4 of 5), and for the
    next 5 minutes a question about the 5th was planned as if it did not exist —
    the chat answered from bare keyframes while its transcript sat restored."""
    _indexed_cache.update(at=0.0, videos=[], platforms={}, durations={})
    with _cache_lock:
        _cache.clear()


# Warm start (P181 follow-up, approved 2026-10-25): the first planner call after a
# restart builds the SDK clients -- imports plus TLS -- and took 3.1-4.2 s, over
# retrieval.PLANNER_BUDGET_S, so the first video question ran unplanned (live
# 2026-10-25 18:50, a definition question never reached the named episode).
# main.py's lifespan runs warm_up() in a background task. It calls the providers'
# boundaries directly, not plan_video_queries: a warm-up must not open the circuit
# (a billing error here is not a turn) nor leave a cached plan. Two tiny calls per
# restart, unmetered (no /search meter is active at startup).
WARM_PROMPT = "what is a trend"
WARM_CLAUDE_TIMEOUT_S = 10.0    # the cold call took 4.2 s; nobody waits on this one


async def warm_up() -> dict:
    """List the indexed videos and call each configured planner provider once.
    Never raises; returns what happened per step, for the startup log."""
    import asyncio
    import os

    import config

    result = {"videos": len(await indexed_videos())}
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        try:
            await asyncio.to_thread(_call_generate, WARM_PROMPT)
            result["gemini"] = "ok"
        except Exception as exc:
            result["gemini"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
    else:
        result["gemini"] = "not configured"
    if getattr(config, "PLANNER_ANTHROPIC_API_KEY", ""):
        try:
            await asyncio.to_thread(_generate_fallback, WARM_PROMPT, WARM_CLAUDE_TIMEOUT_S)
            result["claude"] = "ok"
        except Exception as exc:
            result["claude"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
    else:
        result["claude"] = "not configured"
    return result


def video_platforms() -> dict:
    """video_id -> platform from the last indexed_videos() scroll ({} before one)."""
    return dict(_indexed_cache.get("platforms") or {})


def video_duration(video_id: Optional[str]) -> Optional[float]:
    """Seconds of transcript a video has, from the last indexed_videos() scroll;
    None when unknown -- callers must then skip anything that needs it."""
    return (_indexed_cache.get("durations") or {}).get(video_id) if video_id else None


def _self_test():
    global _generate
    ok = 0
    q = ("In the ingested video 2022 Acme Mentorship Episode 12, find the section where the speaker "
         "explains why a Service Level Agreement is needed. What does the speaker say, and at what timestamp?")
    saved = _generate
    calls = []

    def fake(result):
        def _g(prompt):
            calls.append(prompt)
            if isinstance(result, Exception):
                raise result
            return result
        return _g

    try:
        # 1. A well-formed plan is used as returned.
        _generate = fake({"queries": ["why does the market create a service level agreement",
                                      "how price rebalances after a service level agreement"],
                          "video_name": "2022 Acme Mentorship Episode 12", "intent": "passage"})
        p = plan_video_queries(q)
        assert p == VideoQueryPlan(["why does the market create a service level agreement",
                                    "how price rebalances after a service level agreement"],
                                   "2022 Acme Mentorship Episode 12", "passage"), p
        assert q in calls[0], "the question must reach the model"
        ok += 1
        print("ok %d — a well-formed plan is used as returned" % ok)

        # 2-3. No plan / an exception -> the original message, unfiltered (today's behaviour).
        for label, result in (("None", None), ("exception", RuntimeError("timeout"))):
            _generate = fake(result)
            assert plan_video_queries(q + label) == VideoQueryPlan([q + label], None, "passage")
            ok += 1
            print("ok %d — planner %s falls back to the original question" % (ok, label))

        # 4. Clamping and cleaning. (2-3 failed Gemini on purpose: reset the
        # circuit breaker so these plans come from the faked Gemini again.)
        _breaker.reset()
        _generate = fake({"queries": [], "video_name": "", "intent": "chat"})
        assert plan_video_queries(q + "e") == VideoQueryPlan([q + "e"], None, "passage")
        _generate = fake({"queries": ["a", "  ", "b", "c", "d", "e", "f"], "video_name": None, "intent": "summary"})
        assert plan_video_queries(q + "m") == VideoQueryPlan(["a", "b", "c", "d"], None, "summary")
        # Library questions ("how many videos do you have?") keep their intent.
        _generate = fake({"queries": ["ingested videos"], "video_name": None, "intent": "library"})
        assert plan_video_queries(q + "lib").intent == "library"
        ok += 1
        print("ok %d — empty queries fall back, >4 clamped, blanks dropped, unknown intent -> passage" % ok)

        # 5. Same question twice -> one model call (in-process cache).
        calls.clear()
        _generate = fake({"queries": ["x"], "video_name": None, "intent": "passage"})
        plan_video_queries("cache me")
        plan_video_queries("cache me")
        assert len(calls) == 1, calls
        ok += 1
        print("ok %d — repeated question served from the cache" % ok)
    finally:
        _generate = saved

    # 6. Named video -> id, conservatively: a wrong filter hides the right
    # video entirely, so anything ambiguous or weak resolves to no filter.
    catalog = [("8GkQfdAXZP0", "2022 Acme Mentorship Episode 12"),
               ("z6AxPV2NQ9U", "Acme 2026 Weekly Review \\ March 27, 2026"),
               ("pWnMpdN_g98", "2022 Acme Mentorship Episode 36")]
    assert resolve_video_id("2022 Acme Mentorship Episode 12", catalog) == "8GkQfdAXZP0"
    assert resolve_video_id("**\u201c2022 Acme Mentorship Episode 12,\u201d**", catalog) == "8GkQfdAXZP0"
    assert resolve_video_id("Acme 2026 Weekly Review", catalog) == "z6AxPV2NQ9U"
    assert resolve_video_id("Episode 1", catalog) is None
    assert resolve_video_id("Golden Path lecture", catalog) is None
    assert resolve_video_id(None, catalog) is None
    assert resolve_video_id("https://www.youtube.com/watch?v=z6AxPV2NQ9U&utm_source=x", catalog) == "z6AxPV2NQ9U"
    assert resolve_video_id("2022 Acme Mentorship Episode 12", []) is None
    # Review finding 3: a dated/numbered video that ISN'T ingested must not
    # fuzzy-match a sibling (March 20 vs the ingested March 27 scored ~0.93).
    assert resolve_video_id("Acme 2026 Weekly Review March 20", catalog) is None
    assert resolve_video_id("Acme 2025 Weekly Review", catalog) is None
    assert resolve_video_id("the Acme 2026 Weekly Review video", catalog) == "z6AxPV2NQ9U"
    ok += 1
    print("ok %d — resolve_video_id: exact/quoted/partial titles and pasted ids; ambiguous or weak -> None" % ok)

    # 7. indexed_videos pages through the transcript collection's scroll API
    # (real response shape: points[].payload, next_page_offset) and degrades
    # to [] -- no filter -- when Qdrant fails.
    import asyncio
    import services.qdrant_service as _qs

    pages = [{"points": [{"id": 1, "payload": {"video_id": "B", "title": "Bee"}},
                         {"id": 2, "payload": {"video_id": "A", "title": "Ay"}}], "next_page_offset": 2},
             {"points": [{"id": 3, "payload": {"video_id": "A", "title": "Ay"}}], "next_page_offset": None}]

    class _FakeQ:
        def __init__(self, fail=False):
            self.fail, self.i = fail, 0

        async def _request(self, method, path, **kw):
            if self.fail:
                raise RuntimeError("qdrant down")
            page = pages[self.i]
            self.i += 1
            return page

    saved_get = _qs.get_qdrant
    try:
        _indexed_cache.update(at=0.0, videos=[])
        _qs.get_qdrant = lambda: _FakeQ()
        assert asyncio.run(indexed_videos()) == [("A", "Ay"), ("B", "Bee")]
        _indexed_cache.update(at=0.0, videos=[])
        _qs.get_qdrant = lambda: _FakeQ(fail=True)
        assert asyncio.run(indexed_videos()) == []
    finally:
        _qs.get_qdrant = saved_get
        _indexed_cache.update(at=0.0, videos=[])
    ok += 1
    print("ok %d — indexed_videos pages the scroll API, de-duplicates, and returns [] on failure" % ok)

    # 7b. Stale video list (2026-10-28 restore): a plan made from the old list
    # must not outlive a change of the searchable set.
    saved_get = _qs.get_qdrant
    try:
        _indexed_cache.update(at=0.0, videos=[("A", "Ay")])        # the old list: B missing
        _cache.clear()
        _cache["tell me about Bee"] = "plan made without B"
        _qs.get_qdrant = lambda: _FakeQ()
        assert asyncio.run(indexed_videos()) == [("A", "Ay"), ("B", "Bee")]
        assert "tell me about Bee" not in _cache, "a changed video set must drop cached plans"
        _cache["q"] = "plan"
        invalidate_indexed()
        assert _cache == {} if isinstance(_cache, dict) else len(_cache) == 0
        assert _indexed_cache["videos"] == [] and _indexed_cache["at"] == 0.0
    finally:
        _qs.get_qdrant = saved_get
        _indexed_cache.update(at=0.0, videos=[])
        _cache.clear()
    ok += 1
    print("ok %d — a changed video set, or invalidate_indexed(), drops the list and every cached plan" % ok)

    # 8. Deterministic and splitting: the same question got two different
    # plans live (2026-10-20) -- "why X forms and how it rebalances" as ONE
    # query reranked the answer 0.677 (dropped), split it scored 0.814. The
    # request must pin temperature 0 and tell the model to split compounds.
    import types as _t8
    seen8 = {}

    class _Models8:
        def generate_content(self, model, contents, config):
            seen8.update(config)
            return _t8.SimpleNamespace(parsed=None)

    fake_genai8 = _t8.ModuleType("google.genai")
    fake_genai8.Client = lambda **kw: _t8.SimpleNamespace(models=_Models8())
    fake_types8 = _t8.ModuleType("google.genai.types")
    fake_types8.GenerateContentConfig = lambda **kw: kw
    fake_types8.ThinkingConfig = lambda **kw: kw
    fake_types8.HttpOptions = lambda **kw: kw
    fake_google8 = _t8.ModuleType("google")
    fake_google8.genai = fake_genai8
    import sys as _sys8
    saved8 = {k: _sys8.modules.get(k) for k in ("google", "google.genai", "google.genai.types")}
    _sys8.modules.update({"google": fake_google8, "google.genai": fake_genai8, "google.genai.types": fake_types8})
    try:
        saved_generate8 = _generate
        assert saved_generate8("q") is None
    finally:
        for k, v in saved8.items():
            if v is None:
                _sys8.modules.pop(k, None)
            else:
                _sys8.modules[k] = v
    assert seen8.get("temperature") == 0.0, seen8
    assert "thinking_config" not in seen8, "Flash-Lite plans without thinking: %r" % seen8
    assert "split" in seen8.get("system_instruction", "").lower(), seen8.get("system_instruction")
    ok += 1
    print("ok %d — planner request pins temperature 0 and asks to split compound questions" % ok)

    # 9. Gemini down -> the same prompt goes to Claude (2026-10-22: Google's
    # prepaid credit ran out at 11:04 UTC and every video question silently fell
    # back to plain search, on every chat model). Both down -> plain search, and
    # neither failure is cached.
    import sys as _sys9
    _mod9 = _sys9.modules[__name__]      # patch the module globals plan_video_queries reads
    saved9 = (_mod9._generate, getattr(_mod9, "_generate_fallback", None))
    calls9 = []

    def _gemini_402(q):
        calls9.append("gemini")
        raise RuntimeError("402 RESOURCE_EXHAUSTED")

    def _claude_ok(q, timeout=None):
        calls9.append("claude")
        return {"queries": ["why does the market create a service level agreement"], "video_name": None, "intent": "passage"}

    def _claude_down(q, timeout=None):
        calls9.append("claude")
        raise RuntimeError("529 overloaded")

    try:
        _cache.clear()
        _breaker.reset()
        _mod9._generate, _mod9._generate_fallback = _gemini_402, _claude_ok
        p9 = plan_video_queries("what is a service level agreement")
        assert calls9 == ["gemini", "claude"], calls9
        assert p9.queries == ["why does the market create a service level agreement"], p9
        _cache.clear(); calls9.clear(); _breaker.reset()
        _mod9._generate, _mod9._generate_fallback = _gemini_402, _claude_down
        p9b = plan_video_queries("what is a service level agreement")
        assert p9b.queries == ["what is a service level agreement"] and p9b.intent == "passage", p9b
        assert "what is a service level agreement" not in _cache, "a double failure must not be cached"
    finally:
        _mod9._generate, _mod9._generate_fallback = saved9
        _cache.clear()
        _breaker.reset()
    ok += 1
    print("ok %d — Gemini failure falls back to Claude; both failing falls back to plain search, uncached" % ok)

    # 10. I-1: the ways users actually name a video (M3 question set, measured
    # 2026-10-22: "Ep 12", "episde 12", "March 27 demo" all resolved to None, so
    # the named video's filter, overview and moment lookup were all lost).
    full = catalog + [("2081940656545505280", "The Acme Channel - I don't have time for YouTube "
                                              "to compress and all... I am falling asl...")]
    plats = {"8GkQfdAXZP0": "youtube", "pWnMpdN_g98": "youtube", "z6AxPV2NQ9U": "youtube",
             "2081940656545505280": "twitter"}
    E12, E36, M27, XP = "8GkQfdAXZP0", "pWnMpdN_g98", "z6AxPV2NQ9U", "2081940656545505280"
    for name, want in (("Ep 12", E12), ("Ep. 12", E12), ("ep12", E12), ("EP 36", E36), ("episode 36", E36),
                       ("Acme Mentorship Ep 12", E12), ("Episode #12", E12),
                       ("episde 12", E12), ("Episod 36", E36), ("Mentorhsip Episode 12", E12),
                       ("the March 27 reviw", M27), ("Acme 2026 Weekly Reviw", M27),
                       ("March 27 demo", M27), ("the March 27 stream", M27), ("Episode 12 SLA lesson", E12),
                       ("Episode 36", E36), ("episode 12", E12), ("March 27", M27),
                       ("the March 27 review", M27), ("Acme 2026 Weekly Review", M27)):
        got = resolve_video_id(name, full, plats)
        assert got == want, (name, got, want)
    ok += 1
    print("ok %d — resolve_video_id: Ep/Ep./epN abbreviations, small typos and extra non-title words" % ok)

    # 11. Negative: the digit guard survives fuzzy matching, and anything
    # ambiguous or weak stays unfiltered (a wrong filter hides the right video).
    for name in ("Episode 13", "Episode 21", "Episode 1", "Ep 3", "Acme 2025 Weekly Review",
                 "March 20 demo", "2023 Mentorship Episode 12", "Acme Mentorship", "Mentorship Ep",
                 "the 2022 episode", "2022 golden path", "Acme weekly maker model", "Golden Path lecture",
                 "the youtube video about time", "weekly structure", "the stream"):
        got = resolve_video_id(name, full, plats)
        assert got is None, (name, got)
    assert resolve_video_id("Episode 12", full, plats) != E36
    assert resolve_video_id("Episode 36", full, plats) != E12
    ok += 1
    print("ok %d — resolve_video_id: Episode 12 never matches 36, 2025 never 2026, ambiguous/weak -> None" % ok)

    # 12. Platform references ("the X post") name the one video on that platform;
    # two videos on it, an unknown platform map, or YouTube (3 videos) -> None.
    for name in ("the X post", "X post video", "the video he posted on X", "the twitter video",
                 "the twiter video", "his tweet video", "the video on X", "X video"):
        assert resolve_video_id(name, full, plats) == XP, (name, resolve_video_id(name, full, plats))
    assert resolve_video_id("the youtube video", full, plats) is None
    assert resolve_video_id("the X post", full, None) is None, "no platform data -> no platform match"
    two_x = full + [("999", "Another clip")]
    assert resolve_video_id("the X post", two_x, dict(plats, **{"999": "twitter"})) is None
    # A platform word alongside a title narrows to that platform's videos.
    assert resolve_video_id("the Episode 12 YouTube video", full, plats) == E12
    assert resolve_video_id("the Episode 12 X post", full, plats) is None
    # The default platform map is the one indexed_videos() cached.
    saved_c = dict(_indexed_cache)
    try:
        _indexed_cache.update(platforms=plats)
        assert resolve_video_id("the X post", full) == XP
    finally:
        _indexed_cache.clear()
        _indexed_cache.update(saved_c)
    ok += 1
    print("ok %d — resolve_video_id: a platform reference resolves only to that platform's single video" % ok)

    # 14. A video referenced only inside the question (M3, 2026-10-23): the planner
    # leaves video_name null for "the video he posted on X" / "on March 27". One
    # resolvable reference names it; two different videos, an unresolvable one,
    # or none -> None.
    try:
        _indexed_cache.update(platforms=plats)
        for q, want in (("Summarize the video he posted on X.", XP),
                        ("What date is the traffic spike he projects levels from in the X post?", XP),
                        ("Was the March 27 demo successful in the end?", M27),
                        ("Why was he reluctant to deploy on March 27th?", M27),
                        ("In Ep 12, what is an aggressive rollout?", E12),
                        ("Compare the load balancer in Episode 12 with the March 27 review.", None),
                        ("What does the 2023 Mentorship Episode 5 say about breakers?", None),
                        ("What happened on March 20?", None),
                        ("What is a service level agreement?", None),
                        ("he says X, right?", None)):
            got = implied_video_id(q, full)
            assert got == want, (q, got, want)
    finally:
        _indexed_cache.clear()
        _indexed_cache.update(saved_c)
    assert strip_video_references(["March 27 demo result", "what happened to the last step on the March 27 demo",
                                   "March 27", "Episode 12 service level agreement", "service level agreement in Episode 12"]) == [
        "demo result", "what happened to the last step on the demo", "March 27", "service level agreement"]
    ok += 1
    print("ok %d — implied_video_id: one date/episode/platform reference in the question names the video" % ok)

    # 14b. A video named by its TITLE (2026-10-29, live): with the planner over
    # budget, "* **<title>** tell me about this" got five bare keyframes.
    titled = [("gd_fiber", "2017-10-10 - Pattern Recognition - Cache RTO Sizing and Regional Rollout [aQrd75xwBS4]"),
              ("pWnMpdN_g98", "2022 Acme Mentorship Episode 36"),
              ("gd_intro", "Acme Mentorship 2022 Introduction"),
              ("z6AxPV2NQ9U", "Acme 2026 Weekly Review \\ March 27, 2026")]
    live_q = "*   **Pattern Recognition - Cache RTO Sizing and Regional Rollout [aQrd75xwBS4]** tell me about this"
    for q, want in ((live_q, "gd_fiber"),
                    ("Pattern Recognition - Cache RTO Sizing and Regional Rollout tell me about this", "gd_fiber"),
                    ("what is [aQrd75xwBS4] about?", "gd_fiber"),
                    ("2017-10-10 - Pattern Recognition - Cache RTO Sizing and Regional Rollout [aQrd75xwBS4]", "gd_fiber"),
                    ("In Acme Mentorship 2022 Introduction what is a weekly maker model?", "gd_intro"),
                    ("What is pattern recognition in practice?", None),          # words, not the title
                    ("What is an RTO?", None)):
        got = titled_video_id(q, titled)
        assert got == want, (q, got, want)
    assert implied_video_id(live_q, titled) == "gd_fiber"
    for q, want in ((live_q, True),
                    ("Pattern Recognition - Cache RTO Sizing and Regional Rollout [aQrd75xwBS4] what is this video about?", True),
                    ("Pattern Recognition - Cache RTO Sizing and Regional Rollout: summary", True),
                    ("In Pattern Recognition - Cache RTO Sizing and Regional Rollout, where is the RTO drawn?", False),
                    ("What is pattern recognition?", False)):
        got = asks_about_titled_video(q, titled)
        assert got is want, (q, got, want)
    ok += 1
    print("ok %d — a video named by its title or [id] is found without the planner; "
          "'tell me about this' asks for its overview" % ok)

    # 14b. "and in the other video?" (round 4, live 2026-10-23): rag_filter 1.33
    # searches a follow-up as "<previous message> <follow-up>", so the previous
    # message's video is named again -- and implied_video_id filtered to it, the
    # reply said "I only have one video". The video named before "the other
    # video" is the one the user does NOT mean. "other" in lecture phrasing
    # ("the other side", "on the other hand", "another service level agreement") is not it.
    for q in ("and in the other video?", "what about the other one?", "In the other episode?",
              "Is it explained in a different video?", "show me another video on it",
              "Compare Episode 12 with the other video."):
        assert other_video_reference(q), q
    for q in ("price runs to the other side of the range", "on the other hand he waits",
              "another service level agreement forms", "a different entry model", "the other load balancer",
              "Summarize the video he posted on X.", "one gap and then another one"):
        assert not other_video_reference(q), q
    try:
        _indexed_cache.update(platforms=plats)
        for q, name, want in (("Summarize the video he posted on X. and in the other video?", None, XP),
                              ("Summarize the video he posted on X. and in the other video?",
                               "the video he posted on X", XP),
                              ("In Episode 36, what is a service level agreement? and in the other video?", "Episode 36", E36),
                              ("In Episode 36, what is a service level agreement? and in the other video?", None, E36),
                              ("Compare Episode 12 with the other video.", "Episode 12", E12),
                              ("Compare Episode 12 with the other video.", None, E12),
                              ("Summarize the video he posted on X.", None, None),
                              ("what is in the other video?", None, None),
                              ("In Episode 36, what is a service level agreement?", "Episode 36", None)):
            got = excluded_video_id(q, full, name)
            assert got == want, (q, name, got, want)
    finally:
        _indexed_cache.clear()
        _indexed_cache.update(saved_c)
    low14 = SYSTEM_INSTRUCTION.lower()
    assert "the other video" in low14 and "do not mean" in low14, "prompt lost the other-video rule"
    ok += 1
    print("ok %d — 'the other video': the video named before it is excluded, never the filter" % ok)

    # 13. indexed_videos also caches each video's platform and duration (the
    # largest end_time of its transcript points) -- one scroll, no new client.
    pages13 = [{"points": [{"id": 1, "payload": {"video_id": "X1", "title": "Post", "platform": "twitter",
                                                 "end_time": 1064.0}},
                           {"id": 2, "payload": {"video_id": "X1", "title": "Post", "platform": "twitter",
                                                 "end_time": 500.0}},
                           {"id": 3, "payload": {"video_id": "Y1", "title": "Yt", "platform": "youtube",
                                                 "end_time": 551.0}}], "next_page_offset": None}]
    seen13 = []

    class _FakeQ13:
        async def _request(self, method, path, **kw):
            seen13.append(kw.get("json") or {})
            return pages13[0]

    try:
        _indexed_cache.update(at=0.0, videos=[])
        _qs.get_qdrant = lambda: _FakeQ13()
        assert asyncio.run(indexed_videos()) == [("X1", "Post"), ("Y1", "Yt")]
        assert set(seen13[0]["with_payload"]) >= {"video_id", "title", "platform", "end_time"}, seen13
        assert video_platforms() == {"X1": "twitter", "Y1": "youtube"}, video_platforms()
        assert video_duration("X1") == 1064.0 and video_duration("Y1") == 551.0
        assert video_duration("nope") is None and video_duration(None) is None
    finally:
        _qs.get_qdrant = saved_get
        _indexed_cache.update(at=0.0, videos=[], platforms={}, durations={})
    ok += 1
    print("ok %d — indexed_videos caches platform and duration per video" % ok)

    # 10. M3 Layer 1 (2026-10-22): cross-milestone questions kept the task in the
    # query ("... on my uploaded file") and false-premise questions queried the
    # claim itself ("Episode 36 shipped all five releases at the same time") -- both
    # reranked the real passage out. The shared prompt (Gemini and Claude) must
    # carry both rules, and the P180 rules must survive the edit.
    low = SYSTEM_INSTRUCTION.lower()
    for needle in ("drop the task", "data analysis", "code", "what he did or said", "right?", "neutral",
                   "split", "why it forms", "video collection itself", "query that thing itself",
                   "video_name", "at most 4"):
        assert needle in low, "prompt lost %r" % needle
    assert MAX_QUERIES == 4
    ok += 1
    print("ok %d — prompt strips the task part, queries false premises neutrally, keeps the P180 rules" % ok)

    # 11. Safety net when the model leaves the task words in anyway: they never
    # occur in a lecture transcript, so they only dilute the query's embedding.
    assert _clean_query("the alert from the first presented SLA on my sales file") \
        == "the alert from the first presented SLA", _clean_query("the alert from the first presented SLA on my sales file")
    assert _clean_query("partial rollout rules in my records") == "partial rollout rules"
    assert _clean_query("does my uploaded file show a service level agreement") == "does show a service level agreement"
    assert _clean_query("why does the market create a service level agreement") == "why does the market create a service level agreement"
    assert _clean_query("on my file") == ""
    saved11 = _generate
    try:
        _generate = fake({"queries": ["on my file", "alert one step above the first presented SLA",
                                      "alert one step above the first presented SLA on my sales file"],
                          "video_name": "March 27", "intent": "passage"})
        p11 = plan_video_queries("clean me")
        assert p11.queries == ["alert one step above the first presented SLA"], p11
        _generate = fake({"queries": ["my data"], "video_name": None, "intent": "passage"})
        assert plan_video_queries("clean me 2").queries == ["clean me 2"], "all queries emptied -> original"
    finally:
        _generate = saved11
        _cache.clear()
    ok += 1
    print("ok %d — task words stripped from planned queries; emptied and duplicate queries dropped" % ok)

    ok = _self_test_resilience(ok)
    print("\n%d checks passed" % ok)


def _self_test_resilience(ok: int) -> int:
    """gemini_down.md item A/G: bounded fallback, circuit breaker, recovery, and a
    cache that never passes one provider's plan off as another's."""
    import sys as _sys
    _mod = _sys.modules[__name__]
    saved = (_mod._generate, _mod._generate_fallback, _mod._breaker, _mod.GEMINI_WAIT_S)
    calls = []
    now = [1000.0]
    plan_g = {"queries": ["gemini plan"], "video_name": None, "intent": "passage"}
    plan_c = {"queries": ["claude plan"], "video_name": None, "intent": "passage"}

    def gemini_ok(q):
        calls.append("gemini")
        return plan_g

    def gemini_slow(q):
        calls.append("gemini")
        time.sleep(1.2)                     # longer than the shortened wait below
        return plan_g

    def gemini_raises(msg):
        def _g(q):
            calls.append("gemini")
            raise RuntimeError(msg)
        return _g

    def claude_ok(q, timeout=None):
        calls.append(("claude", timeout))
        return plan_c

    def fresh(**kw):
        _cache.clear()
        calls.clear()
        _mod._breaker = _GeminiBreaker(clock=lambda: now[0], **kw)

    try:
        # 10. A slow Gemini is abandoned after GEMINI_WAIT_S and Claude gets the
        # rest of the deadline -- the planner returns inside PLANNER_DEADLINE_S
        # instead of waiting out the SDK's 10 s.
        fresh()
        _mod.GEMINI_WAIT_S = 0.3
        _mod._generate, _mod._generate_fallback = gemini_slow, claude_ok
        t = time.monotonic()
        p = plan_video_queries("slow gemini")
        took = time.monotonic() - t
        _mod.GEMINI_WAIT_S = saved[3]
        assert p.queries == ["claude plan"] and p.provider == "claude", p
        assert took < 1.0, "planner waited for the slow Gemini call: %.2fs" % took
        claude_timeout = calls[-1][1]
        assert CLAUDE_MIN_S <= claude_timeout <= PLANNER_DEADLINE_S - 0.25, claude_timeout
        ok += 1
        print("ok %d — slow Gemini abandoned at GEMINI_WAIT_S; Claude gets the remaining budget (%.2fs)"
              % (ok, claude_timeout))

        # 11. A billing/auth error opens the circuit at once: the next turn goes
        # straight to Claude with the whole deadline, no Gemini call.
        fresh()
        _mod._generate, _mod._generate_fallback = gemini_raises("402 RESOURCE_EXHAUSTED"), claude_ok
        plan_video_queries("q11a")
        assert calls == ["gemini", ("claude", calls[1][1])], calls
        calls.clear()
        p11 = plan_video_queries("q11b")
        assert calls == [("claude", calls[0][1])], "Gemini must be skipped while open: %r" % calls
        assert calls[0][1] >= PLANNER_DEADLINE_S - 0.1, calls
        assert p11.provider == "claude"
        ok += 1
        print("ok %d — a 402 opens the circuit immediately; next turn skips Gemini" % ok)

        # 12. A transient failure alone does not open it; GEMINI_TRIP_AFTER in a row does.
        fresh()
        _mod._generate = gemini_raises("503 UNAVAILABLE")
        plan_video_queries("q12a")
        assert _mod._breaker.allow(), "one transient failure must not open the circuit"
        plan_video_queries("q12b")
        assert not _mod._breaker.allow(), "two in a row must"
        ok += 1
        print("ok %d — transient failures open the circuit only after %d in a row" % (ok, GEMINI_TRIP_AFTER))

        # 13. Recovery: after the cooldown Gemini is tried again; a success closes
        # the circuit, and a cached Claude plan is replaced by Gemini's.
        fresh()
        _mod._generate, _mod._generate_fallback = gemini_raises("402 RESOURCE_EXHAUSTED"), claude_ok
        assert plan_video_queries("same question").provider == "claude"
        calls.clear()
        assert plan_video_queries("same question").provider == "claude"     # still open: cache stands
        assert calls == [], "a Claude plan is reused while Gemini is still out: %r" % calls
        now[0] += GEMINI_COOLDOWN_S + 1
        _mod._generate = gemini_ok
        p13 = plan_video_queries("same question")
        assert p13.provider == "gemini" and p13.queries == ["gemini plan"], p13
        assert calls == ["gemini"], calls
        assert _mod._breaker.allow() and _mod._breaker.failures == 0
        assert _cache["same question"].provider == "gemini"
        ok += 1
        print("ok %d — circuit recovers after the cooldown; Gemini's plan replaces the cached Claude plan" % ok)

        # 14. A cached Gemini plan is served from the cache (no calls); with the
        # cache disabled nothing is stored or reused.
        import config as _cfg
        fresh()
        _mod._generate = gemini_ok
        plan_video_queries("cached")
        calls.clear()
        assert plan_video_queries("cached").provider == "gemini" and calls == []
        saved_cache = getattr(_cfg, "PLANNER_CACHE_ENABLED", True)
        try:
            _cfg.PLANNER_CACHE_ENABLED = False
            _cache.clear()
            plan_video_queries("uncached")
            plan_video_queries("uncached")
            assert calls == ["gemini", "gemini"] and "uncached" not in _cache, (calls, list(_cache))
        finally:
            _cfg.PLANNER_CACHE_ENABLED = saved_cache
        ok += 1
        print("ok %d — cache serves Gemini plans; PLANNER_CACHE_ENABLED=false stores and reuses nothing" % ok)

        # 15. Evaluation mode: with the fallback disabled a Gemini failure is an
        # unplanned search, never a Claude plan scored as Gemini's.
        fresh()
        saved_fb = getattr(_cfg, "PLANNER_FALLBACK_ENABLED", True)
        try:
            _cfg.PLANNER_FALLBACK_ENABLED = False
            _mod._generate, _mod._generate_fallback = gemini_raises("402 RESOURCE_EXHAUSTED"), claude_ok
            p15 = plan_video_queries("eval question")
            assert p15 == VideoQueryPlan(["eval question"], None, "passage") and p15.provider == "", p15
            assert calls == ["gemini"], calls
        finally:
            _cfg.PLANNER_FALLBACK_ENABLED = saved_fb
        ok += 1
        print("ok %d — PLANNER_FALLBACK_ENABLED=false: Gemini failure never becomes a Claude plan" % ok)

        # 16. The worker thread sees the caller's context (the /search usage meter
        # is a contextvar -- without copy_context Gemini spend would go unmetered).
        probe = contextvars.ContextVar("probe", default="unset")
        seen = []

        def gemini_ctx(q):
            seen.append(probe.get())
            return plan_g

        fresh()
        _mod._generate = gemini_ctx
        token = probe.set("caller")
        try:
            plan_video_queries("ctx")
        finally:
            probe.reset(token)
        assert seen == ["caller"], seen
        ok += 1
        print("ok %d — Gemini worker inherits the caller's context (usage metering)" % ok)

        # 18. Round 3 (live 2026-10-23 12:11 UTC, six parallel chats): four Gemini
        # waits ran out, the circuit opened for 120 s, and every turn after that
        # planned with Claude alone. A timeout is slowness, not an outage: it never
        # counts toward opening the circuit (a 402 or a 503 still does, 11-12).
        def gemini_sleeps(s):
            def _g(q):
                calls.append("gemini")
                time.sleep(s)
                return plan_g
            return _g

        fresh()
        _mod.GEMINI_WAIT_S = 0.1
        _mod._generate, _mod._generate_fallback = gemini_sleeps(0.4), claude_ok
        try:
            for i in range(GEMINI_TRIP_AFTER + 2):
                assert plan_video_queries("timeout %d" % i).provider == "claude"
        finally:
            _mod.GEMINI_WAIT_S = saved[3]
        assert _mod._breaker.allow() and _mod._breaker.failures == 0, \
            "timeouts opened the circuit (%d failures)" % _mod._breaker.failures
        time.sleep(0.5)                     # let the abandoned calls free their workers
        ok += 1
        print("ok %d — Gemini timeouts never open the circuit" % ok)

        # 19. Concurrent turns are not serialised behind each other: six plans at
        # once, each Gemini call well inside the wait, all come back from Gemini.
        # With two pool workers the later submissions queued behind the first ones,
        # ran out their wait without being sent, and tripped the circuit.
        fresh()
        _mod.GEMINI_WAIT_S = 0.8
        _mod._generate, _mod._generate_fallback = gemini_sleeps(0.3), claude_ok
        results = [None] * 6

        def _turn(i):
            results[i] = plan_video_queries("concurrent %d" % i).provider

        threads = [threading.Thread(target=_turn, args=(i,)) for i in range(6)]
        try:
            for th in threads:
                th.start()
            for th in threads:
                th.join()
        finally:
            _mod.GEMINI_WAIT_S = saved[3]
        assert results == ["gemini"] * 6, results
        assert _mod._breaker.allow() and _mod._breaker.failures == 0
        ok += 1
        print("ok %d — six concurrent plans all answered by Gemini; circuit stays closed" % ok)

        # 16b. H2: a definition question gets the lecturer's "why does the market
        # create <concept>" query in code; anything that defines nothing gets none.
        why = "why does the market create a service level agreement"
        for asked in ("According to the videos, what is a service level agreement?", "what is the service level agreement",
                      "What is a service level agreement in the videos? Two sentences, cite your sources.",
                      "Why does price come back to a service level agreement?", "define a service level agreement"):
            assert lecture_queries(asked) == [why], (asked, lecture_queries(asked))
        assert lecture_queries("what is the load balancer?") == ["why does the market create a load balancer"]
        for asked in ("In episode 12, what is he forecasting?", "In the X post video, what is on the chart at 14:32?",
                      "What's the difference between a service level agreement and an nested service level agreement?",
                      "What is a standard bullish PD array after a service level agreement fails to invert?",
                      "What price level did he give for the 50% equilibrium in Episode 36?", "How many videos do you have?"):
            assert lecture_queries(asked) == [], (asked, lecture_queries(asked))
        assert with_lecture_queries("what is a service level agreement", ["what is a service level agreement"]) == \
            ["what is a service level agreement", why]
        assert with_lecture_queries("what is a service level agreement", [why.upper()]) == [why.upper()]
        assert with_lecture_queries("what is a service level agreement", ["a", "b", "c", "d"]) == ["a", "b", "c", why]
        ok += 1
        print("ok %d — definition questions add the lecturer's why-query; others unchanged" % ok)

        # 17. The planner's own deadline sits inside retrieval's wait.
        from pipelines import retrieval as _r
        assert GEMINI_WAIT_S + CLAUDE_MIN_S <= PLANNER_DEADLINE_S < _r.PLANNER_BUDGET_S, \
            (GEMINI_WAIT_S, PLANNER_DEADLINE_S, _r.PLANNER_BUDGET_S)
        ok += 1
        print("ok %d — GEMINI_WAIT_S + CLAUDE_MIN_S <= PLANNER_DEADLINE_S < retrieval.PLANNER_BUDGET_S" % ok)
    finally:
        _mod._generate, _mod._generate_fallback, _mod._breaker, _mod.GEMINI_WAIT_S = saved
        _cache.clear()
    return ok


if __name__ == "__main__":
    _self_test()
