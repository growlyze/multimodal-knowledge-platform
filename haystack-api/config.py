"""
Central configuration — every tunable knob in one place.
Swap embedding model, vector DB, or extraction behavior by changing env vars.
"""

import os


# ── Server ──────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ── Database ────────────────────────────────────────────
# Default carries no credentials: a local dev postgres with trust/peer auth,
# or set DATABASE_URL explicitly (postgresql://user:pass@host:5432/openwebui)
# for a real deployment. Never commit a URL with credentials.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres@localhost:5432/openwebui",
)

# ── Qdrant ──────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_DOCUMENTS = os.getenv("COLLECTION_DOCUMENTS", "documents")
COLLECTION_CODE = os.getenv("COLLECTION_CODE", "code_docs")

# ── Embedding (swappable) ───────────────────────────────
# Options: "voyage", "openai", "jina", "cohere", "local"
# Voyage is served through MongoDB Atlas — Atlas keys 403 on api.voyageai.com,
# and old voyageai.com keys 401 here. Base URL must match where the key was issued.
VOYAGE_BASE_URL = os.getenv("VOYAGE_BASE_URL", "https://ai.mongodb.com/v1").rstrip("/")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "voyage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-4-large")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# ── Reranker (swappable) ────────────────────────────────
# Options: "voyage", "cohere", "jina", "none"
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "voyage")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "rerank-2.5")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "")

# ── Retrieval ───────────────────────────────────────────
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "50"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
# After reranking, pull in every sibling chunk of each winning source so the model
# reads whole topics instead of the one paragraph that matched.
ENABLE_TOPIC_EXPANSION = os.getenv("ENABLE_TOPIC_EXPANSION", "true").lower() == "true"

# ── Chunking defaults ───────────────────────────────────
CHUNK_SIZE_TEXT = int(os.getenv("CHUNK_SIZE_TEXT", "500"))
CHUNK_OVERLAP_TEXT = int(os.getenv("CHUNK_OVERLAP_TEXT", "100"))
CHUNK_SIZE_CODE = int(os.getenv("CHUNK_SIZE_CODE", "300"))
CHUNK_OVERLAP_CODE = int(os.getenv("CHUNK_OVERLAP_CODE", "50"))
CHUNK_SIZE_TRANSCRIPT = int(os.getenv("CHUNK_SIZE_TRANSCRIPT", "500"))
CHUNK_OVERLAP_TRANSCRIPT = int(os.getenv("CHUNK_OVERLAP_TRANSCRIPT", "75"))

# ── Tika (used internally for PDF/DOCX extraction) ─────
TIKA_URL = os.getenv("TIKA_URL", "http://tika:9998")

# Compressed uploads (B1: .zip / .gz preprocessing) -- shared safe-extraction
# limits for extractors/archive.py. Every one is a decompression-bomb or
# resource-exhaustion guard, and every one has a shadow limit already in the
# stack: Caddy caps uploads at 100 MB, the largest legitimate member seen so
# far is a ~60 MB CSV out of an 11 MB .csv.gz, and the biggest real archive in
# use holds ~96 files. Defaults sit well clear of those; a legitimate file that
# trips one gets a message naming the knob to raise.
ARCHIVE_MAX_MEMBERS = int(os.getenv("ARCHIVE_MAX_MEMBERS", "512"))
ARCHIVE_MAX_MEMBER_BYTES = int(os.getenv("ARCHIVE_MAX_MEMBER_BYTES",
                                         str(128 * 1024 * 1024)))
ARCHIVE_MAX_TOTAL_BYTES = int(os.getenv("ARCHIVE_MAX_TOTAL_BYTES",
                                        str(512 * 1024 * 1024)))
# Expanded:compressed ceiling (the zip-bomb shape). Not enforced below
# ARCHIVE_RATIO_FLOOR_BYTES -- a 1 KB member compressing 1000:1 is harmless.
ARCHIVE_MAX_RATIO = float(os.getenv("ARCHIVE_MAX_RATIO", "200"))
ARCHIVE_RATIO_FLOOR_BYTES = int(os.getenv("ARCHIVE_RATIO_FLOOR_BYTES",
                                          str(1024 * 1024)))
# Total characters of member text /extract returns for a container. The filter
# injects at most max_attachment_tokens (~80k chars) of it and strips archives
# from prompts entirely; this bounds the stored copy too.
ARCHIVE_MAX_TEXT_CHARS = int(os.getenv("ARCHIVE_MAX_TEXT_CHARS", "1000000"))
# Cap on Tika output for any extraction. Tika decompresses and concatenates
# package formats without a bound of its own (B1's "cap the Tika fallback");
# ordinary PDFs are far under this and see no change.
EXTRACT_MAX_TEXT_CHARS = int(os.getenv("EXTRACT_MAX_TEXT_CHARS", "5000000"))

# ── External extraction auth ────────────────────────────
# Must match Open WebUI's Admin → Documents → External API Key.
# Open WebUI only uses the external engine when its key is non-empty —
# a blank key there silently falls back to Tika.
EXTERNAL_LOADER_API_KEY = os.getenv("EXTERNAL_LOADER_API_KEY", "")

# ── API auth (services/auth.py) ─────────────────────────
# Bearer key for the admin, video, eval and costs endpoints, and one of the
# two keys accepted by the ingestion endpoints and POST /eval/log-answer
# (the chat filter sends EXTERNAL_LOADER_API_KEY). Empty = fail closed: the
# protected endpoints answer 503 rather than serving unauthenticated traffic.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
# Read-only key for the two stored-dataset endpoints (GET /admin/data/search,
# GET /admin/data/table/*). The code sandbox runs user-submitted code, so it
# holds this key only — never ADMIN_API_KEY or EXTERNAL_LOADER_API_KEY.
DATA_API_KEY = os.getenv("DATA_API_KEY", "")

# ── CORS ────────────────────────────────────────────────
# Comma-separated allow-list of browser origins for the admin panel.
# Example: CORS_ORIGINS="https://panel.example.com,http://localhost:3000"
CORS_ORIGINS = [o.strip() for o in
                os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
                if o.strip()]

# ── File storage ────────────────────────────────────────
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/data/haystack-uploads")

# ── M3: Video ingestion ─────────────────────────────────
VIDEO_VISUAL_COLLECTION = os.getenv("VIDEO_VISUAL_COLLECTION", "visual_frames")
VIDEO_OCR_COLLECTION = os.getenv("VIDEO_OCR_COLLECTION", "ocr_frames")
VIDEO_OD_COLLECTION = os.getenv("VIDEO_OD_COLLECTION", "od_frames")
VIDEO_TRANSCRIPT_COLLECTION = os.getenv("VIDEO_TRANSCRIPT_COLLECTION", "transcript_segments")

# Adaptive capture (client spec item, pipelines/adaptive_capture.py):
# periodic safety frame + burst capture on top of the SSIM detector below.
# 45s/2s split the middle of the client's stated 30-60s static / 1-4s active
# ranges. 3 changes within 10s is the "active stretch" trigger.
SAFETY_FRAME_INTERVAL_S = float(os.getenv("SAFETY_FRAME_INTERVAL_S", "45"))
ACTIVE_SAFETY_FRAME_INTERVAL_S = float(os.getenv("ACTIVE_SAFETY_FRAME_INTERVAL_S", "2"))
ACTIVE_WINDOW_S = float(os.getenv("ACTIVE_WINDOW_S", "10"))
ACTIVE_CHANGE_THRESHOLD = int(os.getenv("ACTIVE_CHANGE_THRESHOLD", "3"))
BURST_FRAME_COUNT = int(os.getenv("BURST_FRAME_COUNT", "2"))
AUDIO_CUE_TOLERANCE_S = float(os.getenv("AUDIO_CUE_TOLERANCE_S", "1.0"))

SSIM_THRESHOLD = float(os.getenv("SSIM_THRESHOLD", "0.15"))
CANONICAL_WIDTH = int(os.getenv("CANONICAL_WIDTH", "640"))
CANONICAL_HEIGHT = int(os.getenv("CANONICAL_HEIGHT", "360"))

CHART_DARK_RATIO = float(os.getenv("CHART_DARK_RATIO", "0.40"))
CHART_EDGE_RATIO = float(os.getenv("CHART_EDGE_RATIO", "0.015"))

# D66: visual_frames moved from local CLIP (512d) to Voyage multimodal-3.5,
# same 1024d space as the other 3 collections. Kept under its old env var
# name to avoid touching every deployment's config — the value changed, the
# knob didn't. Confirm 1024 against the first real multimodal_embed() call
# (logged by embed_frames_visual) before trusting this blind; Voyage's docs
# don't explicitly confirm multimodal-3.5's un-truncated default dimension.
CLIP_EMBEDDING_DIM = int(os.getenv("CLIP_EMBEDDING_DIM", "1024"))
VOYAGE_EMBEDDING_DIM = int(os.getenv("VOYAGE_EMBEDDING_DIM", "1024"))
VOYAGE_BATCH_SIZE = int(os.getenv("VOYAGE_BATCH_SIZE", "96"))
VOYAGE_RPM_LIMIT = int(os.getenv("VOYAGE_RPM_LIMIT", "200"))
VOYAGE_MULTIMODAL_MODEL = os.getenv("VOYAGE_MULTIMODAL_MODEL", "voyage-multimodal-3.5")

# D65: ASR moved from local WhisperX to Groq's hosted Whisper — full
# large-v3 (not Turbo), client's explicit choice for maximum quality.
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")

# D67: vision-capable chart description. Not Flash-Lite — scores far higher
# on visual reasoning specifically (77.7% vs 35.1%, Roboflow Reasoning
# category), the skill this feature actually needs. GOOGLE_API_KEY already
# exists in the VPS .env (picked up automatically by google-genai).
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.6-flash")

# Query-planner fallback when Gemini fails (2026-10-22: Google's prepaid credit ran
# out and every video question lost its plan, on every chat model). The key is the
# SAME Anthropic key Open WebUI uses for Claude chats (user decision 2026-10-22), so
# planner spend lands on the chat account. Unset -> no fallback (plain search, as before).
PLANNER_ANTHROPIC_API_KEY = os.getenv("PLANNER_ANTHROPIC_API_KEY", "")
PLANNER_FALLBACK_MODEL = os.getenv("PLANNER_FALLBACK_MODEL", "claude-haiku-5-5")
# Evaluation switches (both default to normal operation). A provider-specific
# evaluation (e.g. "M3 Layer 1 with the Gemini planner") sets
# PLANNER_FALLBACK_ENABLED=false so Claude can never write a plan that gets scored
# as Gemini's, and PLANNER_CACHE_ENABLED=false so no earlier plan is reused.
PLANNER_CACHE_ENABLED = os.getenv("PLANNER_CACHE_ENABLED", "true").lower() == "true"
PLANNER_FALLBACK_ENABLED = os.getenv("PLANNER_FALLBACK_ENABLED", "true").lower() == "true"
# One tiny Gemini call (VISION_MODEL) before a video's paid phases start; a
# billing/auth/missing-model failure refuses the video instead of indexing it
# without chart descriptions (gemini_down.md). "false" restores the old behaviour.
INGEST_VISION_PREFLIGHT = os.getenv("INGEST_VISION_PREFLIGHT", "true").lower() == "true"

MAX_CONCURRENT_VIDEOS = int(os.getenv("MAX_CONCURRENT_VIDEOS", "1"))
MEMORY_WATERMARK_GB = int(os.getenv("MEMORY_WATERMARK_GB", "4"))
YTDLP_SLEEP_MIN = int(os.getenv("YTDLP_SLEEP_MIN", "15"))
YTDLP_SLEEP_MAX = int(os.getenv("YTDLP_SLEEP_MAX", "45"))
YTDLP_MAX_RETRIES = int(os.getenv("YTDLP_MAX_RETRIES", "3"))

# D68: safety/guardrail layer — formalizes the pre-M3 "3-5 test videos, no
# bulk load" policy as an enforced limit rather than a norm a batch API call
# can silently bypass. MAX_IN_FLIGHT_VIDEOS caps total queued+processing at
# once (not just per-call); MAX_INGESTION_SPEND_USD is a running-total
# circuit breaker on Groq/Voyage/vision spend; MAX_CONSECUTIVE_DOWNLOAD_FAILURES
# stops the worker rather than hammering YouTube into a harder IP ban.
MAX_QUEUE_BATCH_SIZE = int(os.getenv("MAX_QUEUE_BATCH_SIZE", "15"))
MAX_IN_FLIGHT_VIDEOS = int(os.getenv("MAX_IN_FLIGHT_VIDEOS", "15"))
MAX_INGESTION_SPEND_USD = float(os.getenv("MAX_INGESTION_SPEND_USD", "50.0"))
MAX_CONSECUTIVE_DOWNLOAD_FAILURES = int(os.getenv("MAX_CONSECUTIVE_DOWNLOAD_FAILURES", "3"))

# P91's still-open gap (cookies.txt never supplied) plus a usage guardrail,
# researched 2026-10-14. No official YouTube rate limit is published;
# community/vendor guides put guest-session throttling around ~300
# videos/hour and logged-in-account throttling around ~2000/hour, with the
# explicit, repeated warning that using a real account this way risks it
# being rate-limited, forced through verification, or banned. This project
# ingests a handful of pilot videos at a time, not bulk, so these defaults
# sit an order of magnitude under even the most conservative published
# ceiling rather than trusting an unverifiable number. A failed check never
# blocks the download itself — only whether cookies are attached to it; the
# existing impersonate/PO-token path (P91) still runs either way.
YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY = int(os.getenv("YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY", "20"))
YTDLP_COOKIES_MIN_DELAY_SECONDS = int(os.getenv("YTDLP_COOKIES_MIN_DELAY_SECONDS", "30"))
YTDLP_COOKIES_WARN_THRESHOLD_PCT = float(os.getenv("YTDLP_COOKIES_WARN_THRESHOLD_PCT", "0.8"))
# A session cookie file silently goes stale on its own (YouTube rotates
# session cookies in open browser tabs as a security measure) long before
# any count-based budget would catch it — this forces a fresh, verified
# re-export on a schedule instead of trusting a file indefinitely.
YTDLP_COOKIES_MAX_AGE_DAYS = int(os.getenv("YTDLP_COOKIES_MAX_AGE_DAYS", "14"))

# D65/D66/D67 cost-per-unit estimates, used only to total up spend against
# MAX_INGESTION_SPEND_USD — approximate, not an invoice. Groq: full
# large-v3 $0.111/hr (D65). Voyage multimodal: ~$0.0006/frame (C19/D66
# estimate, 1024px-ish keyframes). Gemini 3.6 Flash vision description:
# ~$0.002/frame (D67/C22 estimate).
GROQ_COST_PER_HOUR_USD = float(os.getenv("GROQ_COST_PER_HOUR_USD", "0.111"))
VOYAGE_COST_PER_FRAME_USD = float(os.getenv("VOYAGE_COST_PER_FRAME_USD", "0.0006"))
# Separate, much cheaper rate for Voyage TEXT embeddings (OCR text, OD
# descriptions, transcript segments — all via embed_texts_voyage(), not the
# image-based multimodal calls VOYAGE_COST_PER_FRAME_USD covers). Added in
# code review 2026-10-09 — the cost ledger previously only counted
# keyframe/image embeddings, undercounting real spend against the D68
# safety ceiling. ~50 tokens/item x $0.12/1M tokens (voyage text pricing).
VOYAGE_TEXT_COST_PER_ITEM_USD = float(os.getenv("VOYAGE_TEXT_COST_PER_ITEM_USD", "0.00001"))
VISION_COST_PER_FRAME_USD = float(os.getenv("VISION_COST_PER_FRAME_USD", "0.002"))

# Video fusion track weights. Equal by default (P145/D80): track point ids never
# coincide, so fusion interleaves tracks by weight instead of merging moments,
# and D59's unequal 0.30/0.30/0.25/0.15 let OCR + transcript take every fused
# slot -- the Gemini chart descriptions never reached the reranker. These were
# also never read by anything until P145; retrieval._search_video now passes them.
WRRF_W_ASR = float(os.getenv("WRRF_W_ASR", "0.25"))
WRRF_W_OCR = float(os.getenv("WRRF_W_OCR", "0.25"))
WRRF_W_CLIP = float(os.getenv("WRRF_W_CLIP", "0.25"))
WRRF_W_OD = float(os.getenv("WRRF_W_OD", "0.25"))

VIDEO_DISCOVERY_CHANNEL_URL = os.getenv("VIDEO_DISCOVERY_CHANNEL_URL", "")
VIDEO_WORK_DIR = os.getenv("VIDEO_WORK_DIR", "/data/video-ingest-work")

# P91/C21 bot-detection fix: disguise the TLS/HTTP fingerprint as a real
# browser (requires curl_cffi — see requirements.txt) rather than changing
# the IP. Empty string disables impersonation entirely. Cookies file is
# optional and separate — exported from a real logged-in YouTube session,
# placed on the VPS outside the repo (never committed).
YTDLP_IMPERSONATE_TARGET = os.getenv("YTDLP_IMPERSONATE_TARGET", "chrome")
# Defaults under UPLOAD_DIR (already a persistent host-mounted volume, per
# compose.yml) so a dashboard upload survives a container rebuild without
# a new volume mount. Never committed; never logged; never returned by any
# API response (see cookie_guard.get_cookie_status).
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE",
                               os.path.join(UPLOAD_DIR, "ytdlp-cookies", "cookies.txt"))

# PO-token provider sidecar (compose.yml `bgutil-provider`, port 4416,
# internal-only/unauthenticated by its own design — never expose this).
# "bgutil-provider" resolves via Docker's internal DNS on the compose network.
# Empty string disables it (requires the bgutil-ytdlp-pot-provider pip
# plugin — see requirements.txt — and falls back to no PO token, not an
# error, if the plugin or server aren't reachable).
YTDLP_POT_PROVIDER_URL = os.getenv("YTDLP_POT_PROVIDER_URL", "http://bgutil-provider:4416")
