"""
Dashboard support endpoints — read-only, non-secret deployment status for the
Next.js dashboard (dashboard/, served at /kb).

    GET /dashboard/config            what is configured: booleans for every
                                     credential, plus non-secret settings
                                     (model names, collection names, limits)
    GET /dashboard/providers/health  one bounded, near-free probe per AI
                                     provider (pipelines/provider_health.py)

Every endpoint requires Authorization: Bearer <ADMIN_API_KEY>
(services/auth.require_admin). No key, token or password value is ever
returned: credentials are reported as true/false only, and provider probe
details pass through provider_health.redact().
"""

import asyncio
import os

from fastapi import APIRouter, Depends

import config
from services.auth import require_admin

router = APIRouter(prefix="/dashboard", tags=["dashboard"],
                   dependencies=[Depends(require_admin)])

# The provider probes are each bounded by their own timeout; this caps the
# whole request so a hung SDK can never pin the dashboard.
PROBE_DEADLINE_S = 30.0


def _set(value) -> bool:
    """True when a credential/setting is present (never returns the value)."""
    return bool(str(value or "").strip())


def _env(name: str) -> bool:
    return _set(os.getenv(name, ""))


def _drive_override_present() -> bool:
    path = os.getenv("DRIVE_INTAKE_DRIVE_OVERRIDE", "/data/backup-drives/primary.json")
    try:
        return bool(path) and os.path.isfile(path)
    except OSError:
        return False


@router.get("/config")
async def dashboard_config():
    """Non-secret configuration snapshot. Credentials are booleans only."""
    return {
        "credentials": {
            "admin_api_key": _set(config.ADMIN_API_KEY),
            "external_loader_api_key": _set(config.EXTERNAL_LOADER_API_KEY),
            "embedding_api_key": _set(config.EMBEDDING_API_KEY),
            "reranker_api_key": _set(config.RERANKER_API_KEY),
            "google_api_key": _env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY"),
            "anthropic_api_key": _env("ANTHROPIC_API_KEY") or _set(config.PLANNER_ANTHROPIC_API_KEY),
            "groq_api_key": _env("GROQ_API_KEY"),
            "voyage_api_key": _env("VOYAGE_API_KEY"),
        },
        "models": {
            "embedding_provider": config.EMBEDDING_PROVIDER,
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dimensions": config.EMBEDDING_DIMENSIONS,
            "reranker_provider": config.RERANKER_PROVIDER,
            "reranker_model": config.RERANKER_MODEL,
            "vision_model": config.VISION_MODEL,
            "transcription_model": config.GROQ_WHISPER_MODEL,
            "planner_fallback_model": config.PLANNER_FALLBACK_MODEL,
        },
        "collections": {
            "documents": config.COLLECTION_DOCUMENTS,
            "code": config.COLLECTION_CODE,
            "video_transcripts": config.VIDEO_TRANSCRIPT_COLLECTION,
            "video_visual": config.VIDEO_VISUAL_COLLECTION,
            "video_ocr": config.VIDEO_OCR_COLLECTION,
        },
        "retrieval": {
            "hybrid_search": config.ENABLE_HYBRID_SEARCH,
            "top_k": config.RETRIEVAL_TOP_K,
            "rerank_top_n": config.RERANK_TOP_N,
        },
        "integrations": {
            # Optional: Drive intake is off unless a folder is configured or
            # chosen in the dashboard (POST /video/drive-intake/folder).
            "drive_intake_folder_configured": _env("DRIVE_INTAKE_FOLDER_ID"),
            "drive_connected_account": _drive_override_present(),
            "video_discovery_channel_configured": _set(config.VIDEO_DISCOVERY_CHANNEL_URL),
            "video_discovery_channel_url": config.VIDEO_DISCOVERY_CHANNEL_URL or "",
            # Phoenix runs with auth on: spans are only accepted with an API key.
            "tracing_configured": _env("PHOENIX_API_KEY"),
            "cors_origins": len(config.CORS_ORIGINS),
        },
        "limits": {
            "max_queue_batch_size": config.MAX_QUEUE_BATCH_SIZE,
            "max_in_flight_videos": config.MAX_IN_FLIGHT_VIDEOS,
            "max_ingestion_spend_usd": config.MAX_INGESTION_SPEND_USD,
        },
    }


@router.get("/providers/health")
async def providers_health():
    """Probe each AI provider once (a few tokens at most). Missing keys report
    'not_configured' without any network call."""
    from pipelines import provider_health

    async def _probe(name, fn):
        try:
            result = await asyncio.wait_for(asyncio.to_thread(fn), PROBE_DEADLINE_S)
            return result.to_dict()
        except asyncio.TimeoutError:
            return {"provider": name, "category": "network", "http_status": 0,
                    "latency_ms": int(PROBE_DEADLINE_S * 1000), "detail": "probe timed out"}
        except Exception as exc:  # noqa: BLE001 -- a probe reports, it never raises
            return {"provider": name, "category": "unknown", "http_status": 0,
                    "latency_ms": 0, "detail": provider_health.redact(type(exc).__name__)}

    results = await asyncio.gather(*(_probe(n, f) for n, f in provider_health.PROBES.items()))
    return {"providers": list(results),
            "ok": all(r.get("category") in ("ok", "not_configured") for r in results)}
