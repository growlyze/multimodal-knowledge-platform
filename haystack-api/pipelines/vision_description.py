"""Vision-capable frame description via Gemini (D67).

Replaces the YOLO-placeholder OD track (video_ingestion.py's old
`od_descriptions.append("chart_frame")`) with a real natural-language
description of what the frame shows (slide, diagram, chart, UI, document...). D59 chose Voyage TEXT
embeddings for the od_frames collection specifically because OD output was
meant to be text, not images — this module is what actually produces that
text; the placeholder never matched its own collection's design.

Gemini 3.6 Flash per D67: scores far higher than Flash-Lite specifically on
visual *reasoning* (77.7% vs 35.1%, Roboflow Reasoning category), the exact
skill needed to read labels, numbers and highlighted areas off a dense
annotated slide or diagram rather than just describe colors and shapes.
"""

import logging
import os

import config

logger = logging.getLogger("haystack-api.vision_description")

# A vision model with no context describes "boxes and lines"; naming the kinds
# of visuals a video frame usually carries (and the domain terms configured in
# DOMAIN_JARGON, the same list asr_track.py primes Whisper with) gets a
# description a retriever can actually match against a question.
SYSTEM_INSTRUCTION = (
    "You are describing a single frame from an educational, business or "
    "presentation video. The frame may show a slide, a diagram, a chart or "
    "graph, a dashboard or software UI, a document or spreadsheet, a "
    "whiteboard, or people and a scene, often with annotations or highlights "
    "added by the presenter. In 1-3 concise sentences, describe: what kind "
    "of visual it is, its title or main heading if legible, the key text, "
    "labels, numbers or data shown, any highlighted, circled or annotated "
    "areas, and what the presenter appears to be pointing out. Use the exact "
    "terms written on screen where legible. If the frame has no meaningful "
    "visual content (a blank screen, a transition, only a talking head), say "
    "so briefly instead of guessing at content that isn't there."
)


# ── Cost levers (2026-10-25, all DEFAULT OFF) ───────────────────────────
# Gemini frame descriptions were 84% of the measured z6AxPV2NQ9U pilot cost
# ($3.0019 of $3.5787: 2.43M input tokens ~1,250/frame, 165,658 output,
# 148,849 thinking). These env knobs cut that per call. They are read at call
# time with os.environ, not config.py (infra file, lead-owned), and with
# none set the request is byte-for-byte the pre-lever one (self-test 4).
# Values/semantics checked against google-genai 2.29.0 (the pinned SDK, by
# introspecting the running container's types) and ai.google.dev docs via
# Context7 on 2026-10-25.

# Allowed thinking levels for gemini-3.6-flash. "minimal" is the floor: the
# docs say Gemini 3 Flash models cannot turn thinking fully off, and the
# legacy thinking_budget=0 knob is not recommended for 3.x and 400s when sent
# together with thinking_level. Default stays "low" (P138).
_THINKING_LEVELS = ("minimal", "low", "medium", "high")
_DEFAULT_THINKING_LEVEL = "low"
# Gemini 3 per-image token cap: default 1120, high 1120, medium 560, low 280.
_MEDIA_RESOLUTIONS = {"low": "MEDIA_RESOLUTION_LOW",
                      "medium": "MEDIA_RESOLUTION_MEDIUM",
                      "high": "MEDIA_RESOLUTION_HIGH"}
# Flex: 50% price, synchronous, but sheddable -- 503 at capacity, 429 at
# rate limit, no server-side fallback to Standard, and requests may queue,
# so the docs recommend client timeouts of 10 min+ and client-side retries.
_FLEX_DEFAULT_TIMEOUT_S = 900
_FLEX_DEFAULT_ATTEMPTS = 4
_FLEX_RETRY_STATUS = [429, 503]
# Consecutive Flex failures before the rest of a video goes straight to
# Standard. Shedding comes in waves (discuss.ai.google.dev 2026-05: 100% of
# Flex requests 503'd for a week); without a breaker every frame would wait
# out its Flex retries first and a video would take hours longer for nothing.
_FLEX_DEFAULT_BREAKER = 5


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip().lower()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        logger.warning("%s=%r is not an integer, using %d", name, os.environ.get(name), default)
        return default


def _thinking_level() -> str:
    level = _env("VIDEO_DESC_THINKING_LEVEL")
    if not level:
        return _DEFAULT_THINKING_LEVEL
    if level not in _THINKING_LEVELS:
        # An invalid level would 400 every frame, and describe_chart_frame()
        # turns errors into "" -- a whole video silently undescribed.
        logger.warning("VIDEO_DESC_THINKING_LEVEL=%r not one of %s, using %r",
                       level, _THINKING_LEVELS, _DEFAULT_THINKING_LEVEL)
        return _DEFAULT_THINKING_LEVEL
    return level


def _service_tier():
    tier = _env("VIDEO_DESC_SERVICE_TIER")
    if not tier:
        return None
    if tier != "flex":
        # Only flex is wired: it is synchronous, so it fits this per-frame
        # call. The Batch API (also 50%) is asynchronous with up to 24 h
        # turnaround and would need a different pipeline shape.
        logger.warning("VIDEO_DESC_SERVICE_TIER=%r unsupported (only 'flex'), using standard", tier)
        return None
    return tier


def _flex_fallback_enabled() -> bool:
    # On by default: a frame left undescribed because Flex was busy is a
    # quality loss, and Standard for that one frame costs only the other half.
    return _env("VIDEO_DESC_FLEX_FALLBACK") not in ("false", "0", "no", "off")


def _lever_config_kwargs(types, tier="__env__") -> dict:
    """Extra GenerateContentConfig kwargs from the cost-lever env vars; {}
    when none is set. `tier` overrides the configured tier (None = Standard)
    so the Flex fallback sends the identical request without the tier."""
    if tier == "__env__":
        tier = _service_tier()
    extra = {}
    res = _env("VIDEO_DESC_MEDIA_RESOLUTION")
    if res:
        if res in _MEDIA_RESOLUTIONS:
            extra["media_resolution"] = getattr(types.MediaResolution, _MEDIA_RESOLUTIONS[res])
        else:
            logger.warning("VIDEO_DESC_MEDIA_RESOLUTION=%r not one of %s, not sent",
                           res, sorted(_MEDIA_RESOLUTIONS))
    if tier == "flex":
        extra["service_tier"] = "flex"
        extra["http_options"] = types.HttpOptions(
            timeout=_env_int("VIDEO_DESC_FLEX_TIMEOUT_S", _FLEX_DEFAULT_TIMEOUT_S) * 1000,
            retry_options=types.HttpRetryOptions(
                attempts=_env_int("VIDEO_DESC_FLEX_ATTEMPTS", _FLEX_DEFAULT_ATTEMPTS),
                http_status_codes=list(_FLEX_RETRY_STATUS)),
        )
    return extra


def describe_chart_frame(frame_bgr, model: str = None) -> str:
    """Describe one visual keyframe using Gemini vision.

    Returns a short natural-language description, or "" on any failure —
    this must never raise. A missing description degrades the OD track for
    one frame (same tolerance the OCR step next to it already has); it must
    not fail the whole video's ingestion.
    """
    if model is None:
        model = getattr(config, "VISION_MODEL", "gemini-3.6-flash")

    try:
        import cv2
        from google import genai
        from google.genai import types

        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            logger.warning("could not JPEG-encode frame for vision description")
            return ""

        from pipelines import usage_meter
        client = genai.Client()

        def _call(tier):
            response = client.models.generate_content(
                model=model,
                contents=[
                    "Describe this video frame.",
                    types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"),
                ],
                # P138: gemini-3.6-flash spends "thinking" tokens out of
                # max_output_tokens. At 150 with default thinking it used 144 to
                # think and returned 2 visible tokens ("This frame is") -- every
                # frame description was truncated. thinking_level="low" keeps a
                # little reasoning for reading the frame (~60 tokens observed) and
                # 400 leaves the description room to finish (finish_reason STOP).
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=400,
                    thinking_config=types.ThinkingConfig(thinking_level=_thinking_level()),
                    **_lever_config_kwargs(types, tier),
                ),
            )
            # Exact tokens incl. thinking, billed at the output rate; the flat
            # per-frame estimate in cost_tracker can't see either. The tier is
            # passed so a flex call is billed at the flex price (usage_meter).
            usage_meter.record_gemini(getattr(response, "usage_metadata", None),
                                      service_tier=tier)
            return (response.text or "").strip()

        tier = _service_tier()
        if tier != "flex" or usage_meter.flex_breaker_tripped():
            return _call(None)
        try:
            return _call("flex")
        except Exception as exc:
            # Flex is sheddable and its 429/503 retries are already spent.
            # Re-send the identical request on Standard so the frame keeps a
            # full-quality description; only this frame pays full price.
            if not _flex_fallback_enabled():
                raise
            usage_meter.record_flex_fallback(
                _env_int("VIDEO_DESC_FLEX_BREAKER", _FLEX_DEFAULT_BREAKER))
            logger.warning("flex description failed, retrying on standard: %s", exc)
            return _call(None)
    except Exception as exc:
        logger.warning("vision description failed, continuing without it: %s", exc)
        return ""


def _self_test():
    import sys
    import types as pytypes

    # 1. describe_chart_frame returns the model's text on success
    fake_cv2 = pytypes.ModuleType("cv2")
    fake_cv2.imencode = lambda ext, frame: (True, pytypes.SimpleNamespace(
        tobytes=lambda: b"fake-jpeg-bytes"))

    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai_pkg = pytypes.ModuleType("google")
    fake_genai_pkg.genai = fake_genai
    fake_types_mod = pytypes.ModuleType("google.genai.types")

    captured_calls = []

    class FakePart:
        @staticmethod
        def from_bytes(data, mime_type):
            return {"data": data, "mime_type": mime_type}

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeResponse:
        text = "A slide titled Quarterly Revenue shows a bar chart with the Q3 bar circled in red."
        # Real GenerateContentResponse shape (google-genai 2.29.0).
        usage_metadata = pytypes.SimpleNamespace(
            prompt_token_count=1290, candidates_token_count=85, thoughts_token_count=58)

    # Tiers whose calls raise, to simulate Flex shedding (None = Standard).
    fail_tiers = set()

    class FakeModels:
        def generate_content(self, model, contents, config):
            captured_calls.append({"model": model, "contents": contents, "config": config})
            if config.kwargs.get("service_tier") in fail_tiers:
                raise RuntimeError("503 UNAVAILABLE (fake)")
            return FakeResponse()

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    class FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeKw:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMediaResolution:
        # Real enum member names, google-genai 2.29.0 (types.MediaResolution).
        MEDIA_RESOLUTION_LOW = "MEDIA_RESOLUTION_LOW"
        MEDIA_RESOLUTION_MEDIUM = "MEDIA_RESOLUTION_MEDIUM"
        MEDIA_RESOLUTION_HIGH = "MEDIA_RESOLUTION_HIGH"

    fake_genai.Client = FakeClient
    fake_types_mod.Part = FakePart
    fake_types_mod.GenerateContentConfig = FakeConfig
    fake_types_mod.ThinkingConfig = FakeThinkingConfig
    fake_types_mod.MediaResolution = FakeMediaResolution
    fake_types_mod.HttpOptions = type("FakeHttpOptions", (FakeKw,), {})
    fake_types_mod.HttpRetryOptions = type("FakeHttpRetryOptions", (FakeKw,), {})

    sys.modules["cv2"] = fake_cv2
    sys.modules["google"] = fake_genai_pkg
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = fake_types_mod
    from pipelines import usage_meter
    meter = usage_meter.start("vision-test")
    try:
        result = describe_chart_frame("fake-frame-array")
    finally:
        usage_meter.stop()
        for mod in ("cv2", "google", "google.genai", "google.genai.types"):
            del sys.modules[mod]

    assert "Quarterly Revenue" in result, result
    assert captured_calls[0]["model"] == "gemini-3.6-flash", captured_calls
    assert captured_calls[0]["config"].kwargs["system_instruction"] == SYSTEM_INSTRUCTION
    # P138: gemini-3.6-flash thinks inside max_output_tokens -- at 150 it
    # spent 144 tokens thinking and returned 2 visible ("This frame is").
    cfg = captured_calls[0]["config"].kwargs
    assert cfg["thinking_config"].kwargs == {"thinking_level": "low"}, cfg
    assert cfg["max_output_tokens"] >= 400, cfg
    print("ok 1 — describe_chart_frame returns the model's description and uses D67's model")

    g = meter.summary()["gemini"]
    assert (g["calls"], g["input_tokens"], g["output_tokens"], g["thought_tokens"]) == (1, 1290, 85, 58), g
    print("ok 1b — the call's usage_metadata (incl. thinking tokens) is metered")

    # 2. describe_chart_frame degrades to "" rather than raising on any failure
    sys.modules["cv2"] = pytypes.ModuleType("cv2")  # no imencode attribute -> AttributeError inside
    try:
        result2 = describe_chart_frame("fake-frame-array")
    finally:
        del sys.modules["cv2"]
    assert result2 == "", result2
    print("ok 2 — describe_chart_frame degrades to '' rather than raising")

    # 3. SYSTEM_INSTRUCTION names the kinds of visual a description must cover
    for term in ["slide", "diagram", "chart", "dashboard", "document", "people"]:
        assert term in SYSTEM_INSTRUCTION, f"{term} missing from SYSTEM_INSTRUCTION"
    print("ok 3 — SYSTEM_INSTRUCTION names the kinds of visual to describe")

    # 4-7: cost levers. Every lever is env-driven and DEFAULT OFF; with no
    # env set the request must be exactly the pre-lever one (check 4), or an
    # unrelated deploy would silently change every description and its price.
    import os
    lever_env = ("VIDEO_DESC_THINKING_LEVEL", "VIDEO_DESC_MEDIA_RESOLUTION",
                 "VIDEO_DESC_SERVICE_TIER", "VIDEO_DESC_FLEX_TIMEOUT_S",
                 "VIDEO_DESC_FLEX_ATTEMPTS", "VIDEO_DESC_FLEX_FALLBACK",
                 "VIDEO_DESC_FLEX_BREAKER")

    def run_with(env):
        saved_env = {k: os.environ.pop(k, None) for k in lever_env}
        os.environ.update(env)
        sys.modules["cv2"] = fake_cv2
        sys.modules["google"] = fake_genai_pkg
        sys.modules["google.genai"] = fake_genai
        sys.modules["google.genai.types"] = fake_types_mod
        captured_calls.clear()
        m = usage_meter.start("vision-lever-test")
        try:
            text = describe_chart_frame("fake-frame-array")
        finally:
            usage_meter.stop()
            for mod in ("cv2", "google", "google.genai", "google.genai.types"):
                del sys.modules[mod]
            for k in lever_env:
                os.environ.pop(k, None)
            os.environ.update({k: v for k, v in saved_env.items() if v is not None})
        assert captured_calls, "describe_chart_frame made no API call: %r" % text
        return text, captured_calls[0]["config"].kwargs, m.summary()["gemini"]

    _, cfg4, g4 = run_with({})
    assert set(cfg4) == {"system_instruction", "max_output_tokens", "thinking_config"}, cfg4
    assert cfg4["thinking_config"].kwargs == {"thinking_level": "low"}, cfg4
    assert cfg4["max_output_tokens"] == 400 and "flex_calls" not in g4, (cfg4, g4)
    print("ok 4 — no lever env set: request identical to the pre-lever one")

    # 5. Thinking: gemini-3.6-flash cannot turn thinking fully off; "minimal"
    # is the documented floor (thinking_budget=0 is the legacy 2.5 knob and
    # 400s when mixed with thinking_level). A typo must not reach the API
    # as an invalid level -- that would 400 every frame into "".
    _, cfg5, _ = run_with({"VIDEO_DESC_THINKING_LEVEL": "minimal"})
    assert cfg5["thinking_config"].kwargs == {"thinking_level": "minimal"}, cfg5
    _, cfg5b, _ = run_with({"VIDEO_DESC_THINKING_LEVEL": "nothink"})
    assert cfg5b["thinking_config"].kwargs == {"thinking_level": "low"}, cfg5b
    print("ok 5 — VIDEO_DESC_THINKING_LEVEL=minimal applied; invalid value falls back to low")

    # 6. Media resolution: per-image token cap (Gemini 3 docs: default 1120,
    # medium 560, low 280). Invalid value -> not sent at all (default).
    _, cfg6, _ = run_with({"VIDEO_DESC_MEDIA_RESOLUTION": "low"})
    assert cfg6["media_resolution"] == "MEDIA_RESOLUTION_LOW", cfg6
    _, cfg6b, _ = run_with({"VIDEO_DESC_MEDIA_RESOLUTION": "Medium"})
    assert cfg6b["media_resolution"] == "MEDIA_RESOLUTION_MEDIUM", cfg6b
    _, cfg6c, _ = run_with({"VIDEO_DESC_MEDIA_RESOLUTION": "tiny"})
    assert "media_resolution" not in cfg6c, cfg6c
    print("ok 6 — VIDEO_DESC_MEDIA_RESOLUTION low/medium applied; invalid value not sent")

    # 7. Flex tier: half price, synchronous, but sheddable (503/429) and may
    # queue for minutes. Needs a long timeout and client-side retries, or
    # most frames would fail into "" -- and the meter must bill it as flex.
    _, cfg7, g7 = run_with({"VIDEO_DESC_SERVICE_TIER": "flex"})
    assert cfg7["service_tier"] == "flex", cfg7
    http = cfg7["http_options"].kwargs
    assert http["timeout"] == 900_000, http   # ms; docs recommend >= 10 min
    retry = http["retry_options"].kwargs
    assert retry["attempts"] == 4 and retry["http_status_codes"] == [429, 503], retry
    assert g7["flex_calls"] == 1, g7
    _, cfg7b, _ = run_with({"VIDEO_DESC_SERVICE_TIER": "flex", "VIDEO_DESC_FLEX_TIMEOUT_S": "120",
                            "VIDEO_DESC_FLEX_ATTEMPTS": "6"})
    assert cfg7b["http_options"].kwargs["timeout"] == 120_000, cfg7b
    assert cfg7b["http_options"].kwargs["retry_options"].kwargs["attempts"] == 6, cfg7b
    _, cfg7c, g7c = run_with({"VIDEO_DESC_SERVICE_TIER": "batch"})
    assert "service_tier" not in cfg7c and "http_options" not in cfg7c and "flex_calls" not in g7c
    print("ok 7 — VIDEO_DESC_SERVICE_TIER=flex: tier, timeout, retries, metered as flex")

    # 8. Flex still failing after its retries falls back to ONE Standard call
    # with the identical request minus tier/http_options: same model, prompt,
    # image and settings, so the frame keeps a full-quality description
    # instead of "" (2026-10-26, user decision). The Standard call is billed
    # at full price; the shed Flex call produced no tokens, so nothing else.
    fail_tiers.clear(); fail_tiers.add("flex")
    text8, _, g8 = run_with({"VIDEO_DESC_SERVICE_TIER": "flex"})
    calls8 = [c["config"].kwargs for c in captured_calls]
    assert "Quarterly Revenue" in text8, text8
    assert len(calls8) == 2 and calls8[0]["service_tier"] == "flex", calls8
    assert "service_tier" not in calls8[1] and "http_options" not in calls8[1], calls8[1]
    def _plain(cfg):
        # Fake config objects are rebuilt per call; compare what they hold.
        return {k: getattr(v, "kwargs", v) for k, v in cfg.items()
                if k not in ("service_tier", "http_options")}
    assert _plain(calls8[0]) == _plain(calls8[1]), \
        "fallback request must match the Flex one apart from the tier"
    assert captured_calls[0]["contents"] == captured_calls[1]["contents"]
    assert captured_calls[0]["model"] == captured_calls[1]["model"]
    assert g8["flex_fallbacks"] == 1 and g8["flex_calls"] == 0 and g8["calls"] == 1, g8
    print("ok 8 — Flex failure falls back to an identical Standard call, metered as a fallback")

    # 8b. Fallback can be switched off (pure-Flex trial): the frame degrades to "".
    text8b, _, g8b = run_with({"VIDEO_DESC_SERVICE_TIER": "flex", "VIDEO_DESC_FLEX_FALLBACK": "false"})
    assert text8b == "" and len(captured_calls) == 1 and "flex_fallbacks" not in g8b, (text8b, g8b)
    print("ok 8b — VIDEO_DESC_FLEX_FALLBACK=false: no Standard call, frame left undescribed")

    # 8c. Both tiers down: still never raises, still counted as a fallback.
    fail_tiers.add(None)
    text8c, _, g8c = run_with({"VIDEO_DESC_SERVICE_TIER": "flex"})
    assert text8c == "" and len(captured_calls) == 2 and g8c["flex_fallbacks"] == 1, (text8c, g8c)
    print("ok 8c — Flex and Standard both failing: '' and no exception")

    # 8d. Standard tier failing is never retried as anything else.
    fail_tiers.clear(); fail_tiers.add(None)
    text8d, _, _ = run_with({})
    assert text8d == "" and len(captured_calls) == 1, captured_calls
    fail_tiers.clear()
    print("ok 8d — no Flex configured: a failure is not retried on another tier")

    # 8e. Circuit breaker: Flex shedding comes in waves (a forum report had
    # 100% failures for a week), so after N consecutive Flex failures in one
    # video, the rest of that video goes straight to Standard instead of
    # waiting out Flex retries per frame. Breaker state lives on the video's
    # usage meter, so the next video tries Flex again.
    fail_tiers.add("flex")
    os.environ.update({"VIDEO_DESC_SERVICE_TIER": "flex", "VIDEO_DESC_FLEX_BREAKER": "3"})
    sys.modules["cv2"] = fake_cv2
    sys.modules["google"] = fake_genai_pkg
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = fake_types_mod
    captured_calls.clear()
    m8e = usage_meter.start("vision-breaker-test")
    try:
        texts = [describe_chart_frame("fake-frame-array") for _ in range(5)]
    finally:
        usage_meter.stop()
        for mod in ("cv2", "google", "google.genai", "google.genai.types"):
            del sys.modules[mod]
        for k in lever_env:
            os.environ.pop(k, None)
        fail_tiers.clear()
    tiers = [c["config"].kwargs.get("service_tier") for c in captured_calls]
    # 3 frames try Flex then Standard; frames 4-5 skip Flex entirely.
    assert tiers == ["flex", None, "flex", None, "flex", None, None, None], tiers
    assert all("Quarterly Revenue" in t for t in texts), texts
    g8e = m8e.summary()["gemini"]
    assert g8e["flex_fallbacks"] == 3 and g8e["flex_breaker_tripped"] is True, g8e
    assert g8e["calls"] == 5 and g8e["flex_calls"] == 0, g8e
    print("ok 8e — 3 consecutive Flex failures trip the breaker: rest of the video goes to Standard")

    print("\n12 checks passed")


if __name__ == "__main__":
    _self_test()
