"""Provider health: classify a provider failure, probe a provider, read the logs.

docs/gemini_down.md: the 2026-10-22 Gemini credit outage was found by accident,
nothing alerted on it, and the documented "is it back?" log check silently stopped
detecting it once the planner gained a Claude fallback. This module is the one
place that knows how to tell those situations apart:

    classify(...)            -> "ok" | "billing" | "auth" | "rate_limited" |
                                "model_unavailable" | "network" | "server_error" |
                                "not_configured" | "unknown"
    probe_gemini/voyage/anthropic/groq()  one bounded, near-free call each
    scan_logs(lines)         counts the failure signatures in container logs

No key value is ever read into a message, logged, or returned: keys are passed to
the SDK/HTTP client only, and every detail string goes through redact().

Not wired into ops/storage_monitor.sh or the admin dashboard yet (that schema is
storage-only: bytes and percent thresholds). Run on demand:
    docker compose exec -e PYTHONPATH=/app -w /app haystack-api python scripts/provider_health.py probe
"""

import os
import re
import time
from dataclasses import asdict, dataclass

CATEGORIES = ("ok", "billing", "auth", "rate_limited", "model_unavailable", "network",
              "server_error", "not_configured", "unknown")

# Gemini answers an empty prepaid balance with 402 RESOURCE_EXHAUSTED and "Your
# prepayment credits are depleted" (2026-10-22). A plain quota/rate limit is 429
# RESOURCE_EXHAUSTED -- same status word, different problem, different fix.
_BILLING_WORDS = ("prepayment", "credit", "billing", "insufficient_quota", "insufficient funds",
                  "payment required", "credit balance is too low")
_AUTH_WORDS = ("api_key_invalid", "api key not valid", "invalid api key", "invalid x-api-key",
               "permission_denied", "unauthenticated", "authentication_error", "invalid_api_key")
_MODEL_WORDS = ("not_found", "model not found", "is not found", "does not exist", "not supported")


@dataclass
class HealthResult:
    provider: str
    category: str
    http_status: int = 0
    latency_ms: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.category == "ok"

    def to_dict(self) -> dict:
        return asdict(self)


_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),                 # Google API key
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),                  # Anthropic/OpenAI-style
    re.compile(r"gsk_[A-Za-z0-9]{16,}"),                    # Groq
    re.compile(r"pa-[A-Za-z0-9_\-]{16,}"),                  # Voyage
    re.compile(r"(?i)(key|token|authorization|bearer)([=:\s\"']+)[A-Za-z0-9_.\-]{12,}"),
)


def redact(text) -> str:
    out = str(text or "")
    for pat in _SECRET_PATTERNS:
        out = pat.sub(lambda m: (m.group(1) + m.group(2) + "[REDACTED]") if m.lastindex else "[REDACTED]", out)
    return out[:300]


def classify(status=None, text: str = "", exc: BaseException = None) -> str:
    """One category for a provider outcome. `status` is the HTTP status (or the
    SDK error's code), `text` the error body/message, `exc` the exception raised."""
    if exc is not None:
        status = status or getattr(exc, "code", None) or getattr(exc, "status_code", None)
        text = "%s %s %s" % (text, type(exc).__name__, exc)
        name = type(exc).__name__.lower()
        if any(k in name for k in ("timeout", "connect", "network", "remoteprotocol")):
            return "network"
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    if status is not None and 200 <= status <= 299:
        return "ok"                     # a healthy answer that mentions "credit" is still healthy
    low = (text or "").lower()
    if status == 402 or any(w in low for w in _BILLING_WORDS):
        return "billing"
    if status in (401, 403) or any(w in low for w in _AUTH_WORDS):
        return "auth"
    if status == 429 or "resource_exhausted" in low or "rate limit" in low or "rate_limit" in low:
        return "rate_limited"
    if status == 404 or any(w in low for w in _MODEL_WORDS):
        return "model_unavailable"
    if status is not None and 500 <= status <= 599:
        return "server_error"
    if "timed out" in low or "timeout" in low or "connection" in low:
        return "network"
    return "unknown"


# Outages that a retry seconds later will not fix. Ingestion refuses to start a
# video on these (it would spend Groq/Voyage money and store a video with no chart
# descriptions); anything else is treated as transient and only warned about.
PERSISTENT = frozenset({"billing", "auth", "model_unavailable", "not_configured"})


# ── probes ───────────────────────────────────────────────────────────────────

PROBE_TIMEOUT_S = 15.0
PROBE_PROMPT = "Reply with: OK"


def _timed(provider, fn) -> HealthResult:
    t = time.monotonic()
    try:
        status, detail = fn()
        cat = classify(status, detail)
    except Exception as exc:  # noqa: BLE001 -- a probe reports, it never raises
        status, detail, cat = getattr(exc, "code", None) or 0, "%s: %s" % (type(exc).__name__, exc), None
        cat = classify(exc=exc)
    try:
        status = int(status or 0)
    except (TypeError, ValueError):
        status = 0
    return HealthResult(provider, cat, status, int((time.monotonic() - t) * 1000), redact(detail))


def probe_gemini(model: str = None, timeout_s: float = PROBE_TIMEOUT_S) -> HealthResult:
    """One text generation, a few output tokens (~$0.00001). Billing only shows
    on a generation call -- a model listing succeeds with an empty balance."""
    import config
    model = model or getattr(config, "VISION_MODEL", "gemini-3.6-flash")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        return HealthResult("gemini:%s" % model, "not_configured", detail="GOOGLE_API_KEY not set")

    def _call():
        from google import genai
        from google.genai import types
        client = genai.Client(http_options=types.HttpOptions(timeout=int(max(timeout_s, 10) * 1000)))
        response = client.models.generate_content(
            model=model, contents=PROBE_PROMPT,
            config=types.GenerateContentConfig(max_output_tokens=32,
                                               thinking_config=types.ThinkingConfig(thinking_level="low")),
        )
        return 200, "answered: %r" % ((response.text or "").strip()[:40])

    return _timed("gemini:%s" % model, _call)


def _http(provider, method, url, headers, json_body=None, timeout_s=PROBE_TIMEOUT_S) -> HealthResult:
    def _call():
        import httpx
        with httpx.Client(timeout=timeout_s) as client:
            r = client.request(method, url, headers=headers, json=json_body)
        return r.status_code, ("HTTP %d" % r.status_code) if r.status_code < 300 else r.text[:300]
    return _timed(provider, _call)


def probe_voyage(timeout_s: float = PROBE_TIMEOUT_S) -> HealthResult:
    import config
    key = getattr(config, "EMBEDDING_API_KEY", "")
    if not key:
        return HealthResult("voyage", "not_configured", detail="EMBEDDING_API_KEY not set")
    return _http("voyage", "POST", "%s/embeddings" % config.VOYAGE_BASE_URL,
                 {"Authorization": "Bearer %s" % key},
                 {"input": ["ok"], "model": config.EMBEDDING_MODEL}, timeout_s)


def probe_anthropic(timeout_s: float = PROBE_TIMEOUT_S) -> HealthResult:
    """A 1-token message: an exhausted credit balance only shows on a message call."""
    import config
    key = getattr(config, "PLANNER_ANTHROPIC_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return HealthResult("anthropic", "not_configured", detail="no Anthropic key set")
    return _http("anthropic", "POST", "https://api.anthropic.com/v1/messages",
                 {"x-api-key": key, "anthropic-version": "2023-06-01"},
                 {"model": config.PLANNER_FALLBACK_MODEL, "max_tokens": 1,
                  "messages": [{"role": "user", "content": PROBE_PROMPT}]}, timeout_s)


def probe_groq(timeout_s: float = PROBE_TIMEOUT_S) -> HealthResult:
    """Model listing: proves the key and the network. Groq bills per audio second
    and has no free call that exercises billing, so a credit problem shows only on
    a real transcription -- the result says so."""
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return HealthResult("groq", "not_configured", detail="GROQ_API_KEY not set")
    res = _http("groq", "GET", "https://api.groq.com/openai/v1/models",
                {"Authorization": "Bearer %s" % key}, None, timeout_s)
    if res.ok:
        res.detail = "key and network OK (billing not checkable without a transcription)"
    return res


PROBES = {"gemini": probe_gemini, "voyage": probe_voyage, "anthropic": probe_anthropic, "groq": probe_groq}


# ── logs ─────────────────────────────────────────────────────────────────────

# Exact substrings the code writes (query_planner.py, vision_description.py,
# video_ingestion.py) or that LiteLLM relays from Google. Kept literal so a test
# can feed representative lines and a renamed log line breaks a test, not the check.
LOG_SIGNATURES = {
    "litellm_billing": ("prepayment credits are depleted",),
    "planner_gemini_failed": ("query planner (Gemini) failed",),
    "planner_circuit_open": ("Gemini circuit open",),
    "planner_total_failure": ("query planner failed (",),
    "planner_gemini_recovered": ("Gemini answered again",),
    "vision_failed": ("vision description failed",),
    "overview_skipped": ("video overview skipped",),
}


def scan_logs(lines) -> dict:
    counts = {k: 0 for k in LOG_SIGNATURES}
    for line in lines:
        for key, needles in LOG_SIGNATURES.items():
            if any(n in line for n in needles):
                counts[key] += 1
    gemini_failing = (counts["litellm_billing"] + counts["planner_gemini_failed"]
                      + counts["planner_circuit_open"] + counts["vision_failed"])
    if gemini_failing:
        verdict = "GEMINI FAILING in these logs"
    else:
        # The whole point of this function: zero planner failures does NOT mean
        # Gemini is up -- with the Claude fallback, and with no traffic, it can be
        # down and still log nothing here.
        verdict = ("no Gemini failure lines found -- NOT proof Gemini is healthy; "
                   "run `scripts/provider_health.py probe gemini`")
    return {"counts": counts, "verdict": verdict}


def _self_test():
    ok = 0

    # 1. classification of every documented outcome
    cases = [
        ((402, "Your prepayment credits are depleted"), "billing"),
        ((429, '{"error":{"status":"RESOURCE_EXHAUSTED","message":"Your prepayment credits are depleted"}}'),
         "billing"),
        ((429, '{"error":{"status":"RESOURCE_EXHAUSTED","message":"Quota exceeded for requests per minute"}}'),
         "rate_limited"),
        ((400, "API key not valid. Please pass a valid API key. API_KEY_INVALID"), "auth"),
        ((401, '{"type":"authentication_error","message":"invalid x-api-key"}'), "auth"),
        ((403, "PERMISSION_DENIED"), "auth"),
        ((404, "models/gemini-9 is not found for API version v1beta"), "model_unavailable"),
        ((503, "UNAVAILABLE: The model is overloaded"), "server_error"),
        ((200, "HTTP 200"), "ok"),
        ((None, ""), "unknown"),
    ]
    for (status, text), want in cases:
        got = classify(status, text)
        assert got == want, (status, text, got, want)
    ok += 1
    print("ok %d — classify: billing vs rate limit vs auth vs model vs server vs ok" % ok)

    # 2. exceptions: SDK-style errors carry .code; timeouts/connect errors are network
    class APIError(Exception):
        def __init__(self, code, msg):
            super().__init__("%d %s" % (code, msg))
            self.code = code

    class ConnectTimeout(Exception):
        pass

    assert classify(exc=APIError(402, "RESOURCE_EXHAUSTED prepayment credits are depleted")) == "billing"
    assert classify(exc=APIError(429, "RESOURCE_EXHAUSTED quota")) == "rate_limited"
    assert classify(exc=ConnectTimeout("timed out")) == "network"
    assert classify(exc=TimeoutError("Gemini planner exceeded 1.5s")) == "network"
    ok += 1
    print("ok %d — classify(exc=...): SDK codes, timeouts and connect errors" % ok)

    # 3. secrets never survive into a detail string
    leaky = ("bad key AIzaSyA1234567890abcdefghijklmnop, header Bearer sk-ant-api03-abcdefghijklmnopqrstuvwxyz "
             "gsk_ABCDEFGHIJKLMNOPQRSTUV key=pa-abcdefghijklmnopqrstu")
    clean = redact(leaky)
    for secret in ("AIzaSyA1234567890", "sk-ant-api03-abc", "gsk_ABCDEFGH", "pa-abcdefgh"):
        assert secret not in clean, (secret, clean)
    assert "[REDACTED]" in clean
    ok += 1
    print("ok %d — redact() strips Google/Anthropic/Groq/Voyage keys and bearer values" % ok)

    # 4. probes: missing key -> not_configured, without any network call
    saved = {k: os.environ.pop(k, None) for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")}
    try:
        assert probe_gemini().category == "not_configured"
        assert probe_groq().category == "not_configured"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    ok += 1
    print("ok %d — probes report not_configured when a key is missing (no network call)" % ok)

    # 5. a probe that raises is reported, never re-raised, and redacted
    def boom():
        raise APIError(402, "prepayment credits are depleted for key AIzaSyA1234567890abcdefghijklmnop")
    res = _timed("gemini:test", boom)
    assert res.category == "billing" and res.http_status == 402, res
    assert "AIzaSy" not in res.detail, res.detail
    ok += 1
    print("ok %d — a failing probe returns a classified, redacted HealthResult" % ok)

    # 6. log scan: representative lines from each container
    litellm = [
        "2026-10-22T11:05:01Z litellm.exceptions.APIError: GeminiException - {\"error\": {\"code\": 402, "
        "\"message\": \"Your prepayment credits are depleted.\", \"status\": \"RESOURCE_EXHAUSTED\"}}",
        "2026-10-22T11:05:09Z INFO: 172.18.0.5 - \"POST /v1/chat/completions HTTP/1.1\" 200 OK",
    ]
    haystack_fallback = [
        "2026-10-22 12:01:00 | WARNING | haystack-api.query_planner | query planner (Gemini) failed "
        "(402 RESOURCE_EXHAUSTED) -- trying the Claude fallback",
        "2026-10-22 12:01:00 | WARNING | haystack-api.query_planner | query planner: Gemini circuit open for "
        "120s after 1 failure(s) (ClientError: 402) -- planning with Claude until then",
    ]
    r1 = scan_logs(litellm)
    assert r1["counts"]["litellm_billing"] == 1 and r1["verdict"].startswith("GEMINI FAILING"), r1
    r2 = scan_logs(haystack_fallback)
    # The documented check `grep -c "query planner failed"` counts ZERO here,
    # while Gemini is plainly down -- the bug gemini_down.md item D describes.
    assert sum("query planner failed" in l for l in haystack_fallback) == 0
    assert r2["counts"]["planner_gemini_failed"] == 1 and r2["counts"]["planner_circuit_open"] == 1, r2
    assert r2["counts"]["planner_total_failure"] == 0
    assert r2["verdict"].startswith("GEMINI FAILING"), r2
    r3 = scan_logs(["query planner failed (circuit open; fallback: no key) -- searching with the original message"])
    assert r3["counts"]["planner_total_failure"] == 1, r3
    quiet = scan_logs(["2026-10-23 INFO search 'sla' -> 5 results"])
    assert "NOT proof" in quiet["verdict"], quiet
    ok += 1
    print("ok %d — scan_logs: Gemini failures found even when the old 'query planner failed' "
          "grep counts 0; silence is never reported as healthy" % ok)

    print("\n%d checks passed" % ok)


if __name__ == "__main__":
    _self_test()
