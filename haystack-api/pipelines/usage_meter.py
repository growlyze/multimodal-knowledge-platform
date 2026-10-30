"""Measured API usage for one video's ingestion — tokens, pixels, audio seconds.

cost_tracker.py's estimate_* functions multiply counts by flat per-item rates
(e.g. $0.002 per Gemini frame, whatever the frame actually cost). That cannot
answer "what did this video actually cost per stage, in tokens", which is what
the client was promised for the pilot checkpoint (2026-10-19). Every provider
already reports exact usage on each response; this module keeps it:

  Gemini   response.usage_metadata -> prompt / candidates / thoughts tokens
  Voyage   result.total_tokens (text); text_tokens + image_pixels (multimodal)
  Groq     verbose_json duration -> billed seconds (10 s minimum per request)

Dollars are usage x PRICES, a dated table copied from each provider's official
pricing page. Usage is measured; prices are published list rates and change,
so PRICES carries its as-of date and the summary repeats it.

One meter is active at a time (start() / stop()), matching the single ingestion
worker (routers/video.py enforces one). record_* calls are no-ops when no meter
is active, so query-time embedding calls made by chat search are never counted.

Chat retrieval is metered separately (P161 item 3): search_scope() opens a
SearchUsageMeter for ONE /search call and the record_search_* functions write
into it. That meter is a contextvars.ContextVar, NOT a module global like the
ingestion meter: searches run concurrently in one event loop, and a global
"current meter" would hand search B's tokens to search A. A ContextVar is
per-task and is copied into asyncio.create_task() and asyncio.to_thread(), which
is exactly where retrieval makes its paid calls (video leg task, planner and
video embed_query threads). Inside a search scope the ingestion record_*
functions are no-ops, so a chat question asked during an ingestion run is never
billed to the video.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Optional

# Official list prices, checked 2026-10-19:
#   https://groq.com/pricing                           (whisper-large-v3)
#   https://ai.google.dev/gemini-api/docs/pricing      (gemini-3.6-flash, paid tier)
#   https://docs.voyageai.com/docs/pricing             (voyage-4-large, voyage-multimodal-3.5)
# Gemini 3.6 Flash doubles to $1.50 / $7.50 from 2027-01-01 (same page).
# Voyage's first 200M tokens / 150B pixels per model are free, so the invoice
# can be $0 while list cost is not -- the summary reports list cost.
PRICES = {
    "as_of": "2026-10-19",
    "groq_per_audio_hour": 0.111,
    "groq_min_seconds_per_request": 10.0,
    "gemini_input_per_m_tokens": 0.75,
    "gemini_output_per_m_tokens": 3.75,      # output INCLUDING thinking tokens
    "voyage_text_per_m_tokens": 0.12,        # voyage-4-large
    "voyage_mm_per_m_tokens": 0.12,          # voyage-multimodal-3.5 text part
    "voyage_mm_per_b_pixels": 0.60,          # voyage-multimodal-3.5 image part
    # Retrieval-only rates, added 2026-10-21 (P161). Read off the same official
    # pages via a search of them that day, not by opening the pages directly:
    #   rerank-2.5: $0.05 / 1M tokens, where the tokens are query tokens x number
    #     of documents + all document tokens (Voyage reports them as
    #     usage.total_tokens on every rerank response).
    #   gemini-3.5-flash-lite (query_planner.PLANNER_MODEL), standard paid tier:
    #     $0.30 / 1M input, $2.50 / 1M output (thinking tokens bill as output).
    "retrieval_as_of": "2026-10-21",
    "voyage_rerank_per_m_tokens": 0.05,
    "gemini_lite_input_per_m_tokens": 0.30,
    "gemini_lite_output_per_m_tokens": 2.50,
    # claude-haiku-5-5 (query_planner fallback when Gemini fails, 2026-10-22):
    #   $0.10 / 1M input, $0.50 / 1M output (prompts up to 100K tokens) --
    #   Anthropic's model table as cached in the claude-api skill (2026-10-06).
    "claude_haiku_input_per_m_tokens": 0.10,
    "claude_haiku_output_per_m_tokens": 0.50,
    # Gemini Flex inference tier (cost lever VIDEO_DESC_SERVICE_TIER=flex,
    # vision_description.py): "50% of Standard pricing (billed per token)",
    # gemini-3.6-flash listed as supported -- ai.google.dev/gemini-api/docs/
    # flex-inference and /optimization, read via Context7 2026-10-25. Applied
    # to input, output and thinking tokens alike.
    "flex_as_of": "2026-10-25",
    "gemini_flex_multiplier": 0.5,
}


_SOURCES = {
    "Groq": "https://groq.com/pricing",
    "Gemini": "https://ai.google.dev/gemini-api/docs/pricing",
    "Voyage": "https://docs.voyageai.com/docs/pricing",
    "Anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
}


def unit_prices(retrieval: bool = False) -> list:
    """PRICES as display rows (admin panel): platform, model, unit, $/unit,
    as-of date and the official page it was copied from. The ingestion page's
    rows are unchanged by default; retrieval=True adds the chat-search rates."""
    p = PRICES
    rows = [
        ("Groq", "whisper-large-v3", "audio hour (10 s minimum per request)", p["groq_per_audio_hour"]),
        ("Gemini", "gemini-3.6-flash", "1M input tokens", p["gemini_input_per_m_tokens"]),
        ("Gemini", "gemini-3.6-flash", "1M output tokens (incl. thinking)", p["gemini_output_per_m_tokens"]),
        ("Voyage", "voyage-4-large", "1M text tokens", p["voyage_text_per_m_tokens"]),
        ("Voyage", "voyage-multimodal-3.5", "1M text tokens", p["voyage_mm_per_m_tokens"]),
        ("Voyage", "voyage-multimodal-3.5", "1B image pixels", p["voyage_mm_per_b_pixels"]),
    ]
    if retrieval:
        rows += [
            ("Voyage", "rerank-2.5", "1M tokens (query x docs + docs)", p["voyage_rerank_per_m_tokens"]),
            ("Gemini", "gemini-3.5-flash-lite", "1M input tokens", p["gemini_lite_input_per_m_tokens"]),
            ("Gemini", "gemini-3.5-flash-lite", "1M output tokens (incl. thinking)",
             p["gemini_lite_output_per_m_tokens"]),
            ("Anthropic", "claude-haiku-5-5", "1M input tokens (planner fallback)",
             p["claude_haiku_input_per_m_tokens"]),
            ("Anthropic", "claude-haiku-5-5", "1M output tokens (planner fallback)",
             p["claude_haiku_output_per_m_tokens"]),
        ]
    return [{"platform": pl, "model": model, "unit": unit, "usd": usd,
             "as_of": p["as_of"], "source": _SOURCES[pl]} for pl, model, unit, usd in rows]


class UsageMeter:
    def __init__(self, video_id: str):
        self.video_id = video_id
        self._lock = threading.Lock()
        self.groq_requests = 0
        self.groq_audio_seconds = 0.0
        self.groq_billed_seconds = 0.0
        self.gemini_calls = 0
        self.gemini_calls_without_usage = 0
        self.gemini_input_tokens = 0
        self.gemini_candidate_tokens = 0
        self.gemini_thought_tokens = 0
        # Subset of the gemini_* counters above that ran on the Flex tier
        # (half price). Kept as a subset, not a second set of totals, so every
        # existing reader of input/output/thought tokens still sees all calls.
        self.gemini_flex_calls = 0
        self.gemini_flex_input_tokens = 0
        self.gemini_flex_candidate_tokens = 0
        self.gemini_flex_thought_tokens = 0
        # Frames whose Flex call failed after its retries and went to Standard
        # (vision_description). With successful flex_calls this gives the
        # measured fallback share -- no published Flex shed rate exists.
        self.gemini_flex_fallbacks = 0
        # Consecutive Flex failures; at the breaker threshold the rest of this
        # video skips Flex. Kept per video so the next one tries Flex again.
        self.gemini_flex_consecutive_failures = 0
        self.gemini_flex_breaker_tripped = False
        self.voyage_text_tokens = 0
        self.voyage_text_requests = 0
        self.voyage_mm_text_tokens = 0
        self.voyage_mm_pixels = 0
        self.voyage_mm_images = 0
        self.voyage_mm_requests = 0

    def record_groq(self, audio_seconds: float) -> None:
        seconds = float(audio_seconds or 0.0)
        with self._lock:
            self.groq_requests += 1
            self.groq_audio_seconds += seconds
            self.groq_billed_seconds += max(seconds, PRICES["groq_min_seconds_per_request"])

    def record_gemini(self, usage, service_tier: Optional[str] = None) -> None:
        """service_tier is what the caller ASKED for ("flex" or None). The
        response's own usage.traffic_type, when the API fills it, overrides
        it: that is what was actually billed (Flex never silently upgrades to
        Standard per the docs, but measuring beats trusting that)."""
        with self._lock:
            self.gemini_calls += 1
            if usage is None:
                self.gemini_calls_without_usage += 1
                return
            inp = getattr(usage, "prompt_token_count", None) or 0
            cand = getattr(usage, "candidates_token_count", None) or 0
            thought = getattr(usage, "thoughts_token_count", None) or 0
            self.gemini_input_tokens += inp
            self.gemini_candidate_tokens += cand
            self.gemini_thought_tokens += thought
            if _is_flex(getattr(usage, "traffic_type", None), service_tier):
                self.gemini_flex_calls += 1
                self.gemini_flex_input_tokens += inp
                self.gemini_flex_candidate_tokens += cand
                self.gemini_flex_thought_tokens += thought
                self.gemini_flex_consecutive_failures = 0

    def record_flex_fallback(self, breaker_threshold: int) -> None:
        with self._lock:
            self.gemini_flex_fallbacks += 1
            self.gemini_flex_consecutive_failures += 1
            if breaker_threshold > 0 and self.gemini_flex_consecutive_failures >= breaker_threshold:
                self.gemini_flex_breaker_tripped = True

    def flex_breaker_tripped(self) -> bool:
        with self._lock:
            return self.gemini_flex_breaker_tripped

    def _flex_fields(self) -> dict:
        # Only present once a flex call ran or fell back: the summary is
        # persisted as JSON (video_ingest_usage), and a standard-only run must
        # keep its old shape.
        if not (self.gemini_flex_calls or self.gemini_flex_fallbacks):
            return {}
        fields = {"flex_calls": self.gemini_flex_calls,
                  "flex_input_tokens": self.gemini_flex_input_tokens,
                  "flex_output_tokens": self.gemini_flex_candidate_tokens,
                  "flex_thought_tokens": self.gemini_flex_thought_tokens,
                  "flex_prices_as_of": PRICES["flex_as_of"]}
        if self.gemini_flex_fallbacks:
            fields["flex_fallbacks"] = self.gemini_flex_fallbacks
            fields["flex_breaker_tripped"] = self.gemini_flex_breaker_tripped
        return fields

    def record_voyage_text(self, total_tokens: int) -> None:
        with self._lock:
            self.voyage_text_requests += 1
            self.voyage_text_tokens += int(total_tokens or 0)

    def record_voyage_multimodal(self, text_tokens: int, image_pixels: int, images: int) -> None:
        with self._lock:
            self.voyage_mm_requests += 1
            self.voyage_mm_text_tokens += int(text_tokens or 0)
            self.voyage_mm_pixels += int(image_pixels or 0)
            self.voyage_mm_images += int(images or 0)

    def summary(self) -> dict:
        p = PRICES
        with self._lock:
            groq_usd = self.groq_billed_seconds / 3600.0 * p["groq_per_audio_hour"]
            gemini_output = self.gemini_candidate_tokens + self.gemini_thought_tokens
            gemini_usd = (self.gemini_input_tokens * p["gemini_input_per_m_tokens"]
                          + gemini_output * p["gemini_output_per_m_tokens"]) / 1e6
            if self.gemini_flex_calls:
                # Flex tokens were priced at full rate above; take back the
                # discounted share. Skipped entirely when no flex call ran,
                # so a standard-only run computes exactly what it always did.
                flex_full = (self.gemini_flex_input_tokens * p["gemini_input_per_m_tokens"]
                             + (self.gemini_flex_candidate_tokens + self.gemini_flex_thought_tokens)
                             * p["gemini_output_per_m_tokens"]) / 1e6
                gemini_usd -= flex_full * (1.0 - p["gemini_flex_multiplier"])
            voyage_text_usd = self.voyage_text_tokens * p["voyage_text_per_m_tokens"] / 1e6
            voyage_mm_usd = (self.voyage_mm_text_tokens * p["voyage_mm_per_m_tokens"] / 1e6
                             + self.voyage_mm_pixels * p["voyage_mm_per_b_pixels"] / 1e9)
            return {
                "video_id": self.video_id,
                "prices_as_of": p["as_of"],
                "groq": {
                    "requests": self.groq_requests,
                    "audio_seconds": self.groq_audio_seconds,
                    "billed_seconds": self.groq_billed_seconds,
                    "usd": groq_usd,
                },
                "gemini": {
                    "calls": self.gemini_calls,
                    "calls_without_usage": self.gemini_calls_without_usage,
                    "input_tokens": self.gemini_input_tokens,
                    "output_tokens": self.gemini_candidate_tokens,
                    "thought_tokens": self.gemini_thought_tokens,
                    "usd": gemini_usd,
                    **self._flex_fields(),
                },
                "voyage": {
                    "text_requests": self.voyage_text_requests,
                    "text_tokens": self.voyage_text_tokens,
                    "text_usd": voyage_text_usd,
                    "mm_requests": self.voyage_mm_requests,
                    "mm_text_tokens": self.voyage_mm_text_tokens,
                    "mm_pixels": self.voyage_mm_pixels,
                    "mm_images": self.voyage_mm_images,
                    "mm_usd": voyage_mm_usd,
                    "usd": voyage_text_usd + voyage_mm_usd,
                },
                "total_usd": groq_usd + gemini_usd + voyage_text_usd + voyage_mm_usd,
            }


_active: Optional[UsageMeter] = None
_active_lock = threading.Lock()


def start(video_id: str) -> UsageMeter:
    """Begin metering one video. Replaces any meter left by a crashed job."""
    global _active
    with _active_lock:
        _active = UsageMeter(video_id)
        return _active


def stop() -> Optional[UsageMeter]:
    """End metering and return the meter (None if none was active)."""
    global _active
    with _active_lock:
        meter, _active = _active, None
        return meter


def _current() -> Optional[UsageMeter]:
    # Inside a chat search the ingestion meter is never charged: its record_*
    # calls are global, and a question asked while a video ingests would
    # otherwise land on that video's bill.
    if _search_meter.get() is not None:
        return None
    with _active_lock:
        return _active


# ── Chat retrieval (one meter per /search call) ─────────

SEARCH_SERVICES = ("voyage_query_embed", "voyage_video_query", "voyage_rerank", "gemini_planner",
                   "claude_planner")


class SearchUsageMeter:
    """Usage of one /search call, per paid service. In-memory counters only:
    recording is a lock and three additions, never I/O, so metering adds no
    latency to the search (persisting it happens off the request path)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._services = {name: {"requests": 0, "tokens": 0, "usd": 0.0} for name in SEARCH_SERVICES}
        self.gemini_calls_without_usage = 0

    def record(self, service: str, tokens: int, usd: float) -> None:
        with self._lock:
            row = self._services.setdefault(service, {"requests": 0, "tokens": 0, "usd": 0.0})
            row["requests"] += 1
            row["tokens"] += int(tokens or 0)
            row["usd"] += float(usd or 0.0)

    def record_gemini_without_usage(self) -> None:
        with self._lock:
            self.gemini_calls_without_usage += 1
            self._services["gemini_planner"]["requests"] += 1

    def summary(self) -> dict:
        with self._lock:
            services = {k: dict(v) for k, v in self._services.items()}
            services["gemini_planner"]["calls_without_usage"] = self.gemini_calls_without_usage
        return {"prices_as_of": PRICES["retrieval_as_of"], "services": services,
                "total_usd": sum(v["usd"] for v in services.values())}


_search_meter: "contextvars.ContextVar[Optional[SearchUsageMeter]]" = \
    contextvars.ContextVar("search_usage_meter", default=None)


@contextmanager
def search_scope():
    """Meter every paid call made while this block runs, in this task and in
    any task/thread it starts. Re-entrant: the comparison-terms rescue calls
    search() again inside the first one, and its calls belong to the same
    question, so a nested scope yields the outer meter."""
    existing = _search_meter.get()
    if existing is not None:
        yield existing
        return
    meter = SearchUsageMeter()
    token = _search_meter.set(meter)
    try:
        yield meter
    finally:
        _search_meter.reset(token)


def record_search_voyage_text(service: str, total_tokens: int) -> None:
    m = _search_meter.get()
    if m is not None:
        m.record(service, total_tokens, int(total_tokens or 0) * PRICES["voyage_text_per_m_tokens"] / 1e6)


def record_search_voyage_multimodal(service: str, text_tokens: int, image_pixels: int = 0) -> None:
    m = _search_meter.get()
    if m is not None:
        m.record(service, text_tokens,
                 int(text_tokens or 0) * PRICES["voyage_mm_per_m_tokens"] / 1e6
                 + int(image_pixels or 0) * PRICES["voyage_mm_per_b_pixels"] / 1e9)


def record_search_rerank(total_tokens: int) -> None:
    m = _search_meter.get()
    if m is not None:
        m.record("voyage_rerank", total_tokens,
                 int(total_tokens or 0) * PRICES["voyage_rerank_per_m_tokens"] / 1e6)


def record_search_gemini(usage) -> None:
    """The query planner's Gemini call (Flash-Lite rates). None = the call
    returned no usage: counted as a request with 0 tokens, not dropped."""
    m = _search_meter.get()
    if m is None:
        return
    if usage is None:
        m.record_gemini_without_usage()
        return
    inp = getattr(usage, "prompt_token_count", None) or 0
    out = (getattr(usage, "candidates_token_count", None) or 0) + \
          (getattr(usage, "thoughts_token_count", None) or 0)
    m.record("gemini_planner", inp + out,
             (inp * PRICES["gemini_lite_input_per_m_tokens"]
              + out * PRICES["gemini_lite_output_per_m_tokens"]) / 1e6)


def record_search_claude(usage) -> None:
    """The query planner's Claude fallback call (Haiku rates). None = no usage
    returned: counted as a request with 0 tokens."""
    m = _search_meter.get()
    if m is None:
        return
    inp = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
    out = int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
    m.record("claude_planner", inp + out,
             (inp * PRICES["claude_haiku_input_per_m_tokens"]
              + out * PRICES["claude_haiku_output_per_m_tokens"]) / 1e6)


def record_groq(audio_seconds: float) -> None:
    m = _current()
    if m is not None:
        m.record_groq(audio_seconds)


def _is_flex(traffic_type, requested: Optional[str]) -> bool:
    """traffic_type is a google-genai TrafficType enum (2.29.0 values include
    ON_DEMAND and ON_DEMAND_FLEX), a plain string, or None when the API does
    not fill it -- then the requested tier is the only evidence there is."""
    if traffic_type is not None:
        name = getattr(traffic_type, "name", None) or str(traffic_type)
        return "FLEX" in name.upper()
    return (requested or "").lower() == "flex"


def record_gemini(usage, service_tier: Optional[str] = None) -> None:
    m = _current()
    if m is not None:
        m.record_gemini(usage, service_tier=service_tier)


def record_flex_fallback(breaker_threshold: int) -> None:
    m = _current()
    if m is not None:
        m.record_flex_fallback(breaker_threshold)


def current_gemini_summary() -> dict:
    """The running video's Gemini counters so far ({} when not metering)."""
    m = _current()
    return m.summary()["gemini"] if m is not None else {}


def flex_breaker_tripped() -> bool:
    m = _current()
    return m.flex_breaker_tripped() if m is not None else False


def record_voyage_text(total_tokens: int) -> None:
    m = _current()
    if m is not None:
        m.record_voyage_text(total_tokens)


def record_voyage_multimodal(text_tokens: int, image_pixels: int, images: int) -> None:
    m = _current()
    if m is not None:
        m.record_voyage_multimodal(text_tokens, image_pixels, images)


# ── Offline self-test ──────────────────────────────────────

def _self_test():
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    ok = 0

    # 1. Groq bills a 10 s minimum per request: a 3 s chunk costs 10 s.
    m = UsageMeter("v")
    m.record_groq(600.0)
    m.record_groq(3.0)
    s = m.summary()
    assert s["groq"]["requests"] == 2
    assert s["groq"]["audio_seconds"] == 603.0
    assert s["groq"]["billed_seconds"] == 610.0
    assert abs(s["groq"]["usd"] - 610.0 / 3600 * 0.111) < 1e-12
    ok += 1
    print("ok %d — groq: 10 s minimum per request, billed seconds x hourly rate" % ok)

    # 2. Gemini: input = prompt tokens; output = candidates + THOUGHTS (P138:
    # thinking tokens are billed at the output rate and were invisible before).
    m = UsageMeter("v")
    m.record_gemini(SimpleNamespace(prompt_token_count=1300, candidates_token_count=90,
                                    thoughts_token_count=60))
    m.record_gemini(SimpleNamespace(prompt_token_count=1300, candidates_token_count=110,
                                    thoughts_token_count=None))
    g = m.summary()["gemini"]
    assert (g["calls"], g["input_tokens"], g["output_tokens"], g["thought_tokens"]) == (2, 2600, 200, 60)
    assert abs(g["usd"] - (2600 * 0.75 + 260 * 3.75) / 1e6) < 1e-12
    ok += 1
    print("ok %d — gemini: thinking tokens billed as output" % ok)

    # 3. A call that failed before returning usage counts as a call, 0 tokens.
    m = UsageMeter("v")
    m.record_gemini(None)
    assert m.summary()["gemini"]["calls_without_usage"] == 1
    ok += 1
    print("ok %d — gemini: missing usage is counted, not silently dropped" % ok)

    # 4. Voyage text + multimodal priced separately; pixels per billion.
    m = UsageMeter("v")
    m.record_voyage_text(5000)
    m.record_voyage_multimodal(text_tokens=0, image_pixels=331_776 * 96, images=96)
    v = m.summary()["voyage"]
    assert v["text_tokens"] == 5000 and v["mm_images"] == 96 and v["mm_pixels"] == 331_776 * 96
    assert abs(v["usd"] - (5000 * 0.12 / 1e6 + 331_776 * 96 * 0.60 / 1e9)) < 1e-12
    ok += 1
    print("ok %d — voyage: text tokens + multimodal pixels" % ok)

    # 5. Totals and the price date travel with the numbers.
    s = m.summary()
    assert s["video_id"] == "v" and s["prices_as_of"] == PRICES["as_of"]
    assert abs(s["total_usd"] - (s["groq"]["usd"] + s["gemini"]["usd"] + s["voyage"]["usd"])) < 1e-12
    ok += 1
    print("ok %d — summary carries total and prices_as_of" % ok)

    # 6. Module-level recorders are no-ops with no active meter (chat search
    # embeds queries in the same process and must not be billed to a video).
    assert stop() is None
    record_voyage_text(999)
    meter = start("vid1")
    record_voyage_text(10)
    record_gemini(SimpleNamespace(prompt_token_count=1, candidates_token_count=1,
                                  thoughts_token_count=0))
    assert stop() is meter and stop() is None
    record_voyage_text(999)
    assert meter.summary()["voyage"]["text_tokens"] == 10
    ok += 1
    print("ok %d — recorders only count while a meter is active" % ok)

    # 7. Thread-safe: describe_chart_frame runs on a 4-worker pool.
    meter = start("vid2")
    usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=1, thoughts_token_count=1)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: record_gemini(usage), range(2000)))
    stop()
    assert meter.summary()["gemini"]["input_tokens"] == 2000
    ok += 1
    print("ok %d — concurrent recording loses nothing" % ok)

    # 8. Search recorders: no-ops outside a search scope; inside one, each
    # service is priced from PRICES. The planner is Flash-Lite, not 3.6 Flash.
    record_search_voyage_text("voyage_query_embed", 999)
    with search_scope() as sm:
        record_search_voyage_text("voyage_query_embed", 10)
        record_search_voyage_text("voyage_video_query", 8)
        record_search_voyage_multimodal("voyage_video_query", text_tokens=6, image_pixels=0)
        record_search_rerank(20_000)
        record_search_gemini(SimpleNamespace(prompt_token_count=400, candidates_token_count=30,
                                             thoughts_token_count=10))
        record_search_gemini(None)
    s = sm.summary()
    sv = s["services"]
    assert sv["voyage_query_embed"] == {"requests": 1, "tokens": 10,
                                        "usd": 10 * PRICES["voyage_text_per_m_tokens"] / 1e6}, sv
    assert sv["voyage_video_query"]["requests"] == 2 and sv["voyage_video_query"]["tokens"] == 14
    assert abs(sv["voyage_rerank"]["usd"] - 20_000 * PRICES["voyage_rerank_per_m_tokens"] / 1e6) < 1e-15
    g = sv["gemini_planner"]
    assert (g["requests"], g["tokens"], g["calls_without_usage"]) == (2, 440, 1), g
    assert abs(g["usd"] - (400 * PRICES["gemini_lite_input_per_m_tokens"]
                           + 40 * PRICES["gemini_lite_output_per_m_tokens"]) / 1e6) < 1e-15
    assert abs(s["total_usd"] - sum(v["usd"] for v in sv.values())) < 1e-15
    assert s["prices_as_of"] == PRICES["retrieval_as_of"] and set(sv) == set(SEARCH_SERVICES)
    ok += 1
    print("ok %d — search recorders: no-op outside a scope, priced per service inside" % ok)

    # 9. A nested search (retrieval's comparison-terms rescue calls search()
    # again) records into the outer scope, not a fresh one that nobody stores.
    with search_scope() as outer:
        with search_scope() as inner:
            record_search_rerank(5)
    assert inner is outer and outer.summary()["services"]["voyage_rerank"]["tokens"] == 5
    ok += 1
    print("ok %d — nested search scope shares the outer meter" % ok)

    # 10. Concurrent searches never mis-attribute, including calls made in
    # asyncio.to_thread (the planner, video embed_query). A module-global
    # "current meter" like start()/stop() would hand B's tokens to A here. And
    # a video ingestion running at the same time is never billed for them.
    import asyncio

    async def one_search(tokens):
        with search_scope() as m:
            for _ in range(20):
                await asyncio.sleep(0)                 # interleave with the other search
                await asyncio.to_thread(record_search_voyage_text, "voyage_query_embed", tokens)
                record_search_rerank(tokens)
            # a sub-task (retrieval's video leg is asyncio.create_task) inherits the scope
            await asyncio.create_task(asyncio.to_thread(record_search_rerank, tokens))
            return m

    async def both():
        return await asyncio.gather(one_search(1), one_search(1000))

    video = start("ingesting-now")
    ma, mb = asyncio.run(both())
    stop()
    a, b = ma.summary()["services"], mb.summary()["services"]
    assert a["voyage_query_embed"]["tokens"] == 20 and a["voyage_rerank"]["tokens"] == 21, a
    assert b["voyage_query_embed"]["tokens"] == 20_000 and b["voyage_rerank"]["tokens"] == 21_000, b
    assert video.summary()["voyage"]["text_tokens"] == 0
    ok += 1
    print("ok %d — two concurrent searches each keep only their own usage" % ok)

    # 11. Inside a search, the ingestion recorders never charge the video
    # being ingested in the same process (chat search during an ingestion run).
    video = start("ingesting-now")
    with search_scope():
        record_voyage_text(50)
        record_gemini(SimpleNamespace(prompt_token_count=5, candidates_token_count=5,
                                      thoughts_token_count=0))
    record_voyage_text(7)
    stop()
    vs = video.summary()
    assert vs["voyage"]["text_tokens"] == 7 and vs["gemini"]["calls"] == 0, vs
    ok += 1
    print("ok %d — a search never bills the video being ingested" % ok)

    # 12. Flex tier (cost lever, VIDEO_DESC_SERVICE_TIER=flex): its tokens are
    # billed at half the standard rate. Without this the meter would report
    # a flex run at full list price and the trial would look like no saving.
    m = UsageMeter("v")
    m.record_gemini(SimpleNamespace(prompt_token_count=1000, candidates_token_count=100,
                                    thoughts_token_count=0))
    std_only = m.summary()["gemini"]
    # No flex call yet -> the summary shape is exactly what it was before the
    # lever existed (it is persisted as JSON in video_ingest_usage).
    assert "flex_calls" not in std_only, std_only
    m.record_gemini(SimpleNamespace(prompt_token_count=1000, candidates_token_count=100,
                                    thoughts_token_count=20), service_tier="flex")
    # The response's own traffic_type wins when present (measured, not assumed).
    m.record_gemini(SimpleNamespace(prompt_token_count=1000, candidates_token_count=100,
                                    thoughts_token_count=0,
                                    traffic_type=SimpleNamespace(name="ON_DEMAND_FLEX")))
    g = m.summary()["gemini"]
    std_usd = (1000 * 0.75 + 100 * 3.75) / 1e6
    flex_usd = PRICES["gemini_flex_multiplier"] * ((2000 * 0.75 + 220 * 3.75) / 1e6)
    assert PRICES["gemini_flex_multiplier"] == 0.5
    assert (g["calls"], g["input_tokens"], g["output_tokens"], g["thought_tokens"]) == (3, 3000, 300, 20), g
    assert (g["flex_calls"], g["flex_input_tokens"], g["flex_output_tokens"],
            g["flex_thought_tokens"]) == (2, 2000, 200, 20), g
    assert abs(g["usd"] - (std_usd + flex_usd)) < 1e-12, g
    # A standard call whose traffic_type says ON_DEMAND stays at full price
    # even if the caller asked for flex: the response is what was billed.
    m2 = UsageMeter("v")
    m2.record_gemini(SimpleNamespace(prompt_token_count=1000, candidates_token_count=0,
                                     thoughts_token_count=0, traffic_type="ON_DEMAND"),
                     service_tier="flex")
    assert abs(m2.summary()["gemini"]["usd"] - 1000 * 0.75 / 1e6) < 1e-15
    # Module-level recorder passes the tier through.
    video = start("flex-video")
    record_gemini(SimpleNamespace(prompt_token_count=10, candidates_token_count=0,
                                  thoughts_token_count=0), service_tier="flex")
    stop()
    assert video.summary()["gemini"]["flex_calls"] == 1
    ok += 1
    print("ok %d — gemini flex tier billed at the flex multiplier, standard unchanged" % ok)

    print("%d checks passed" % ok)


if __name__ == "__main__":
    _self_test()
