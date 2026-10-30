# Growlyze Multimodal AI Chat

A self-hosted AI knowledge platform. Users chat with an assistant that answers
from **your own knowledge base** — documents, spreadsheets, code, images, audio
and video — and cites its sources. Operators manage everything from a built-in
admin dashboard.

Repository: https://github.com/growlyze/multimodal-knowledge-platform

The product is generic: one codebase serves any client or industry. Each client
runs an independent copy with its own configuration, branding, data, keys and
backups — no code changes needed.

> **Before you rely on it:** two retrieval behaviours still need validation —
> visual retrieval and summary-based retrieval. See
> [section 15](#15-known-limitations-and-features-requiring-further-validation).

**Contents**

1. [Features](#1-features)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Installation — step by step](#4-installation--step-by-step)
5. [Open WebUI setup](#5-open-webui-setup)
6. [Admin dashboard](#6-admin-dashboard)
7. [API keys and models](#7-api-keys-and-models)
8. [Configuration reference](#8-configuration-reference)
9. [Customizing for a client](#9-customizing-for-a-client)
10. [Integrations](#10-integrations)
11. [Backups and restore](#11-backups-and-restore)
12. [Backend API](#12-backend-api)
13. [Development and tests](#13-development-and-tests)
14. [Troubleshooting](#14-troubleshooting)
15. [Known limitations and features requiring further validation](#15-known-limitations-and-features-requiring-further-validation)

---

## 1. Features

- **Chat with citations** — Open WebUI frontend; answers grounded in the
  knowledge base with numbered sources.
- **Multimodal ingestion** — PDF, Word, HTML, Markdown, text, code,
  spreadsheets (CSV/XLSX become queryable tables), archives (zip/tar/gz),
  images (OCR + visual description), audio and video (transcription, keyframes,
  on-screen text, visual descriptions).
- **Hybrid retrieval** — dense embeddings + keyword search in Qdrant, reranking,
  query planning, deduplication.
- **Data analysis sandbox** — the model writes and runs Python in an isolated
  container with the `data_tools` library (load files and stored datasets,
  summaries, charts, Excel export, code validation).
- **Multiple model providers** — Gemini, Claude, Groq and Voyage through LiteLLM;
  users switch model per chat.
- **Admin dashboard** (Next.js) — overview, knowledge base, uploads, jobs, video
  library, backups and restore, costs, integrations, administration.
- **Optional integrations** — Google Drive (backups, document intake), video
  channel discovery, Phoenix tracing.
- **Encrypted backups** — restic snapshots to Google Drive with
  daily/weekly/monthly retention and point-in-time restore.

---

## 2. Architecture

```
                         https://DOMAIN
                               │
                           ┌───┴───┐
                           │ Caddy │  HTTPS, routing
                           └───┬───┘
            ┌──────────────────┼───────────────────┐
            │ /                │ /kb/              │ /traces/
      ┌─────┴──────┐    ┌──────┴──────┐      ┌─────┴─────┐
      │ Open WebUI │    │  Dashboard  │      │  Phoenix  │
      │  (chat)    │    │  (Next.js)  │      │ (tracing) │
      └──┬──────┬──┘    └──────┬──────┘      └───────────┘
         │      │              │ server-side, ADMIN_API_KEY
         │      │       ┌──────┴───────┐
         │      └──────►│ haystack-api │ ingestion · retrieval · admin API
         │   filters    │   (FastAPI)  │
         │              └─┬───┬────┬───┘
   ┌─────┴─────┐          │   │    │
   │  LiteLLM  │      Postgres Qdrant Tika
   │ (models)  │
   └───────────┘   code-sandbox (Python, data_tools) ◄── Open WebUI
```

### Services (`compose.yml`)

Only Caddy publishes ports (80/443). Every other service is reachable only on
the internal Docker networks.

| Service | Role |
|---|---|
| `caddy` | HTTPS for `DOMAIN`; routes `/` → Open WebUI, `/kb/*` → dashboard, `/traces/*` → Phoenix |
| `open-webui` | Chat frontend (image in `webui-image/`, branded with `APP_NAME`) |
| `dashboard` | Admin dashboard at `/kb/` |
| `haystack-api` | FastAPI backend: extraction, chunking, embeddings, hybrid retrieval, citations, video/audio/OCR/vision pipelines, admin API, Open WebUI content loader |
| `postgres` | Open WebUI database; also haystack-api state and structured-data tables (`data_*`) |
| `qdrant` | Vector store (documents, code, transcripts, visual frames, OCR, object detection) |
| `tika` | Document text extraction |
| `litellm` | OpenAI-compatible gateway to Gemini, Claude, Groq, Voyage |
| `code-sandbox` | Sandboxed Python (Open Terminal) with `data_tools`; read-only filesystem, no capabilities, memory/CPU capped |
| `phoenix` | Tracing UI and OTLP collector |
| `bgutil-provider` | Token provider for video downloads |
| `watchtower` | Keeps `bgutil-provider` updated |

### How a chat turn works

1. The user asks a question in Open WebUI.
2. The `rag_filter` function decides the route: knowledge-base search, data
   analysis in the sandbox, web, or plain chat.
3. For knowledge questions it calls `haystack-api /search`, which plans the
   query, runs hybrid search in Qdrant, reranks and returns cited chunks.
4. The filter builds the context and the model (via LiteLLM) answers with
   numbered citations.
5. Files uploaded in chat are indexed in the background (`/ingest/*`) so they
   become part of the knowledge base.

### Repository layout

| Path | Contents |
|---|---|
| `compose.yml`, `Caddyfile`, `.env.example` | Application stack and configuration template |
| `setup_env.py` | Creates `.env`: asks a few questions, generates every secret |
| `haystack-api/` | Backend: `routers/`, `pipelines/`, `services/`, `extractors/`, `evaluation/`, `scripts/`, `tests/` |
| `filters/` | Open WebUI functions: `rag_filter.py`, `data_tools_filter.py`, tool `dataset_tool.py` |
| `dashboard/` | Next.js + TypeScript admin dashboard |
| `code-execution/` | Sandbox image and the `data_tools` library |
| `litellm-config.yaml`, `litellm-hooks/` | Model definitions and Gemini retry hook |
| `processing/telemetry/` | Shared tracing module |
| `webui-image/` | Branded Open WebUI image |
| `tika-config/` | Tika configuration |
| `ops/` | Backup, restore and storage-monitor scripts with systemd units (Linux host) |

---

## 3. Prerequisites

| | Needed |
|---|---|
| Machine | Linux server (Ubuntu 22.04/24.04 recommended), or a Mac/Windows computer for a local trial |
| Resources | 4 CPU cores and 16 GB RAM minimum; 8 cores / 32 GB recommended for video. 50 GB+ free disk |
| Software | Docker with the Compose plugin, Git, Python 3 |
| API keys | A **Voyage AI** key (required, for embeddings). A **Google Gemini** key is strongly recommended; Anthropic, Groq and reranker keys are optional (section 7) |
| For a public site | A domain name you can add a DNS record to |

---

## 4. Installation — step by step

Two paths: **A** puts the platform online on a server; **B** runs it on your
own computer to try it out. Both take about 15–30 minutes, most of it waiting
for the first build.

### A. On a server (public website with HTTPS)

**Step 1 — Get a server.** Any cloud VPS with Ubuntu 22.04 or 24.04 and the
resources above, with SSH access as root (or a user with `sudo`).

**Step 2 — Point your domain at it.** At your DNS provider, create an **A
record** for the hostname you want (e.g. `chat.client.com`) with the server's
public IP address. It can take a few minutes to take effect; check with
`ping chat.client.com`.

**Step 3 — Install Docker and open the firewall.** On the server:

```bash
sudo apt update && sudo apt -y install git python3 curl ufw
curl -fsSL https://get.docker.com | sudo sh
sudo ufw allow OpenSSH && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw --force enable
```

**Step 4 — Download the code.**

```bash
sudo git clone https://github.com/growlyze/multimodal-knowledge-platform.git /opt/gmac
cd /opt/gmac
```

**Step 5 — Create the configuration.** The setup script asks a few questions
(domain, product name, dashboard password, API keys), generates every secret
and writes `.env`:

```bash
sudo python3 setup_env.py
```

Enter your domain from step 2 when asked. You can edit `.env` later at any
time; section 8 lists every setting.

**Step 6 — Start the platform.**

```bash
sudo docker compose up -d --build
sudo docker compose ps
```

The first build takes 5–15 minutes. When every service shows `running` (or
`healthy`), continue. Caddy obtains the HTTPS certificate automatically.

**Step 7 — Set up the chat.** Open `https://your-domain/`, create the first
account (it becomes the administrator) and follow **section 5** (about 5
minutes: connect the models, document extraction, sandbox, install 3 functions).

**Step 8 — Open the admin dashboard.** Go to `https://your-domain/kb/` and log
in with the username and password you chose in step 5. Upload a document under
**Uploads & Ingestion**, then ask about it in the chat — the answer should cite
it.

**Optional next steps:** Google Drive (section 10.1), automatic backups
(section 11), Phoenix tracing (section 10.3).

### B. On your own computer (local trial)

1. Install **Docker Desktop** (macOS/Windows) or Docker Engine (Linux), Git and
   Python 3. In Docker Desktop give Docker at least 8 GB of memory
   (Settings → Resources), and after step 3 set `SANDBOX_MEM_LIMIT=4g` in `.env`.
2. Download the code:
   ```bash
   git clone https://github.com/growlyze/multimodal-knowledge-platform.git
   cd multimodal-knowledge-platform
   ```
3. Create the configuration — press Enter to accept `localhost` as the domain:
   ```bash
   python3 setup_env.py        # on Windows: python setup_env.py
   ```
   On macOS/Windows the script stores data in `./data` automatically.
4. Start: `docker compose up -d --build`
5. Open `https://localhost/`. The browser warns about the certificate (Caddy
   uses its own local certificate for `localhost`) — choose to continue. Then
   follow section 5. The dashboard is at `https://localhost/kb/`.

### Configuring by hand (instead of the script)

```bash
cp .env.example .env
chmod 600 .env
```

Fill in every value in the **CORE** section. Generate each secret separately
(never reuse one value for two settings):

```bash
openssl rand -hex 32
```

| Setting | Value |
|---|---|
| `POSTGRES_PASSWORD`, `WEBUI_SECRET_KEY`, `EXTERNAL_LOADER_API_KEY`, `ADMIN_API_KEY`, `DATA_API_KEY`, `OPEN_TERMINAL_API_KEY`, `PHOENIX_SECRET`, `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD`, `SESSION_SECRET` | Random secrets |
| `EMBEDDING_API_KEY` | Your Voyage AI key |
| `DOMAIN` / `PUBLIC_BASE_URL` | `localhost` / `https://localhost`, or `chat.client.com` / `https://chat.client.com` |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH` | Dashboard login; create the hash with the command below |
| `DATA_DIR` | `/data` on Linux; `./data` on macOS/Windows (then also `RCLONE_CONFIG_DIR=./data/rclone`) |

```bash
python3 -c "import hashlib,os,sys;s=os.urandom(16).hex();i=310000;print('pbkdf2_sha256:%d:%s:%s'%(i,s,hashlib.pbkdf2_hmac('sha256',sys.argv[1].encode(),s.encode(),i).hex()))" 'your-dashboard-password'
```

Paste the printed `pbkdf2_sha256:310000:...` line as `ADMIN_PASSWORD_HASH`. A
placeholder or malformed value keeps the dashboard login disabled. Keep
comments in `.env` on their own lines: some scripts read the file directly and
a trailing `# comment` would become part of the value.

`docker compose config -q` validates the file (no output means OK). If a
required value is missing, Compose stops with a message such as
`set ADMIN_API_KEY in .env`.

### Addresses

| URL | What |
|---|---|
| `https://DOMAIN/` | Chat (Open WebUI) |
| `https://DOMAIN/kb/` | Admin dashboard |
| `https://DOMAIN/traces/` | Phoenix tracing |

### Everyday commands

Run from the repository folder (prefix with `sudo` on a server if needed):

```bash
docker compose ps                        # status of every service
docker compose logs -f haystack-api      # follow a service's logs
docker compose restart dashboard         # restart one service
docker compose down                      # stop (data in DATA_DIR is kept)
git pull && docker compose up -d --build # update to the latest code
```

---

## 5. Open WebUI setup

Open `https://DOMAIN/`. **The first account created becomes the Open WebUI
administrator** — create it right after the first start.

### 5.1 Connect the models (LiteLLM)

Admin Panel → Settings → Connections → OpenAI API:

- URL: `http://litellm:4000/v1`
- Key: any non-empty value, e.g. `sk-litellm` (LiteLLM has no master key and is
  reachable only inside the Docker network)

The model aliases from section 7 then appear in the model picker. Choose the
default model in Admin Panel → Settings → Models.

### 5.2 Document extraction

Admin Panel → Settings → Documents:

- Content Extraction Engine: **External**
- URL: `http://haystack-api:8000/extract`
- API Key: the value of `EXTERNAL_LOADER_API_KEY`

### 5.3 Code execution sandbox

Admin Panel → Settings → Code Execution (Open Terminal):

- URL: `http://code-sandbox:8000`
- API key: the value of `OPEN_TERMINAL_API_KEY`

### 5.4 Install the functions

Admin Panel → Functions → **+**, paste the file's content, save, enable it and
make it **Global** (or attach it to specific models). Install the tool under
Workspace → Tools.

| File | Type | Purpose |
|---|---|---|
| `filters/rag_filter.py` | Filter | Retrieval, citations, routing (knowledge base / sandbox / web), indexing of chat uploads, durable download copies |
| `filters/data_tools_filter.py` | Filter | Adds `data_tools` instructions and a file summary when a CSV/TSV/XLSX is attached |
| `filters/dataset_tool.py` | Tool | "Search Datasets": finds stored tables for `load_dataset()` |

Functions are stored in the Open WebUI database: after changing a file in
`filters/`, paste the new version into the function again.

**`rag_filter` valves** (defaults match this stack):

| Valve | Default | Meaning |
|---|---|---|
| `haystack_api_url` | `http://haystack-api:8000` | Backend URL |
| `haystack_api_key` | value of `EXTERNAL_LOADER_API_KEY` | Bearer key for ingestion and answer logging (read from Open WebUI's environment automatically) |
| `search_top_k` | 5 | Chunks returned per search |
| `min_relevance_score` | 0.3 | Minimum score for a chunk to be used |
| `search_timeout` | 20.0 | Seconds before a search is abandoned |
| `context_token_limit` / `context_budget_ratio` | 100000 / 0.8 | Context budget |
| `max_attachment_tokens` | 20000 | Inline attachment limit |
| `enable_async_ingestion` | true | Index files uploaded in chat into the knowledge base |
| `enable_code_routing` | true | Send computation requests to the sandbox |
| `skip_search_for_file_compute` | true | No knowledge-base search for pure file calculations |
| `enable_answer_scoring` | true | Send answers to `/eval/log-answer` for sampled quality scoring |
| `sandbox_upload_dir` | `/mnt/uploads` | Where the sandbox sees uploads |
| `data_file_extensions`, `document_extensions`, `media_extensions`, `code_extensions`, `archive_extensions` | see file | File-type routing |
| `app_name` | `Growlyze Multimodal AI Chat` | Name used in user-facing notices |
| `enable_durable_downloads` | true | Keep a durable copy of files the sandbox produces |

**`data_tools_filter` valves:** `priority` 10 (keep it above `rag_filter`),
`enabled`, `skip_if_rag_filter_injected`, `skip_for_pure_knowledge_questions`,
`inject_file_summary_hint` (all true); user valve `enabled`.

### 5.5 Users

Admin Panel → Users: create accounts, or enable sign-up with the default role
"pending" so an administrator approves each new user.

---

## 6. Admin dashboard

`https://DOMAIN/kb/` — log in with `ADMIN_USERNAME` and the password behind
`ADMIN_PASSWORD_HASH`. Each section has its own colour so it is always clear
which part of the console is open.

> **Changed in this release:** the admin panel is a new Next.js + TypeScript
> dashboard. It replaces the earlier Streamlit admin panel, which has been
> removed. Its features were carried over and wired to the real backend
> endpoints and backup system. Two small changes: Drive files open in Google
> Drive instead of downloading through the panel, and old files in the Drive
> intake folder are removed with the intake "cleanup" action rather than
> one by one.

| Section | What it does |
|---|---|
| **Overview** | Platform health, document/chunk/video/table counts, storage, active jobs, recent activity |
| **Knowledge Base** | Browse, search and delete documents; inspect data tables; test retrieval; reindex |
| **Uploads & Ingestion** | Upload documents, code, transcripts, media, structured data and archives into a collection |
| **Processing Jobs** | Ingestion and video job status; retry failed video jobs |
| **Video Library** | Add videos by URL, discover a channel, queue and start processing, cookies, settings, Drive intake |
| **Backups & Restore** | Recovery points, back up now, verify, restore documents/chats/videos or the full system, schedule switches |
| **Costs & Usage** | Retrieval and video processing costs |
| **Integrations** | AI provider status, Google Drive connection, video sources, tracing |
| **Administration** | Configuration overview (keys shown only as set / not set), provider health |

### Security

- Every page and `/kb/api/*` route requires a session, except login and
  `/kb/api/health`.
- Session cookie: HttpOnly, Secure, SameSite=Lax, path `/kb`, 8 hours, signed
  with `SESSION_SECRET` (HMAC-SHA256).
- State-changing requests need a same-origin request and a matching CSRF token.
- Login: constant-time password check; 5 failed attempts per 15 minutes per IP,
  plus a global ceiling.
- The dashboard calls `haystack-api` **server-side** with `ADMIN_API_KEY`
  through an allowlisted proxy. No key ever reaches the browser.

---

## 7. API keys and models

### Providers

| Variable | Provider | Used for | Needed? |
|---|---|---|---|
| `EMBEDDING_API_KEY` | Voyage AI (`voyage-4-large`, 1024 dimensions) | Embeddings for ingestion and search | **Required** |
| `GOOGLE_API_KEY` | Google AI Studio (Gemini) | Gemini chat models, image/frame descriptions, query planner | Strongly recommended |
| `ANTHROPIC_API_KEY` | Anthropic | Claude chat models, query-planner fallback, evaluation judge | Optional |
| `PLANNER_ANTHROPIC_API_KEY` | Anthropic | Separate key for the query planner (defaults to `ANTHROPIC_API_KEY`) | Optional |
| `GROQ_API_KEY` | Groq | Whisper transcription of audio/video, Groq chat model | Required for audio/video |
| `RERANKER_API_KEY` | Voyage AI | Result reranking (`rerank-2.5`) | Recommended |
| `VOYAGE_API_KEY` | Voyage AI | LiteLLM `voyage-embed` model, provider health check | Optional |

Where to get them: Voyage AI dashboard (embedding + reranker), Google AI Studio
(Gemini), Anthropic Console (Claude), Groq Console (Groq).

All keys stay in `.env` on the server. The dashboard never receives them; it
only shows whether each one is set.

> Do not change `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS` once data has been
> ingested: the vector collections use 1024 dimensions and existing vectors
> would become incompatible.

### Chat models (`litellm-config.yaml`)

| Alias | Provider |
|---|---|
| `gemini-flash` | Gemini Flash (default chat model) |
| `gemini-flash-backup` | Gemini Flash, second deployment |
| `gemini-flash-lite` | Gemini Flash Lite (fast, low cost) |
| `claude-opus` | Claude Opus |
| `claude-sonnet` | Claude Sonnet |
| `claude-haiku` | Claude Haiku |
| `groq-llama` | Llama on Groq |
| `voyage-embed` | Voyage embedding model |

Aliases are stable names: to change the model behind one, edit its
`litellm_params.model` and run `docker compose restart litellm`. An alias whose
provider key is empty only fails when someone selects it. To add a provider,
add a `model_list` entry with `api_key: os.environ/<VAR>` and put that variable
in `.env`.

The `gemini_malformed_retry` hook re-sends a Gemini reply that came back empty
(up to 2 retries). `GEMINI_FC_MODE_VALIDATED=1` (in `compose.yml`) enables
Google's validated function-calling mode; set `0` to turn it off.

---

## 8. Configuration reference

Everything is configured in `.env`. **R** = required (Compose refuses to start
without it).

### Core

| Variable | | Description |
|---|---|---|
| `POSTGRES_PASSWORD` | R | Database password |
| `WEBUI_SECRET_KEY` | R | Signs Open WebUI sessions |
| `EXTERNAL_LOADER_API_KEY` | R | Open WebUI content loader and chat filter → haystack-api |
| `ADMIN_API_KEY` | R | Dashboard → haystack-api admin endpoints |
| `DATA_API_KEY` | R | Sandbox → read-only stored-dataset endpoints |
| `EMBEDDING_API_KEY` | R | Embedding provider key |
| `OPEN_TERMINAL_API_KEY` | R | Code sandbox key |
| `PHOENIX_SECRET`, `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` | R | Phoenix authentication |
| `DOMAIN` | R | Hostname Caddy serves |
| `PUBLIC_BASE_URL` | R | `https://DOMAIN` |
| `SESSION_SECRET` | R | Signs dashboard sessions (32+ characters) |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH` | R | Dashboard login |

### Feature keys

`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `PLANNER_ANTHROPIC_API_KEY`,
`GROQ_API_KEY`, `RERANKER_API_KEY`, `VOYAGE_API_KEY`, `PHOENIX_API_KEY`
(sections 7 and 10); `RESTIC_PASSWORD`, `RESTIC_REPOSITORY` (section 11).

### Deployment, paths and branding

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Growlyze Multimodal AI Chat` | Product name in Open WebUI, the dashboard and Drive folder names |
| `STACK_NAME` | `gmac` | Prefix for container, network and project names |
| `DATA_DIR` | `/data` | Host folder for all persistent data |
| `RCLONE_CONFIG_DIR` | `/root/.config/rclone` | Host folder with `rclone.conf` for the optional Drive features |
| `INSTALL_DIR` | `/opt/gmac` | Repository location on a Linux host (backup scripts only) |
| `CORS_ORIGINS` | empty | Browser origins allowed to call haystack-api directly (comma-separated). Empty = none; the stack does not need it |
| `DOMAIN_JARGON` | empty | Comma-separated client terms added to extraction prompts (improves OCR and transcripts) |
| `SANDBOX_MEM_LIMIT` / `SANDBOX_CPUS` | `12g` / `4.0` | Sandbox resource cap — size to the machine |

### Google Drive and video

| Variable | Default | Description |
|---|---|---|
| `DRIVE_OAUTH_CLIENT_ID`, `DRIVE_OAUTH_CLIENT_SECRET` | empty | Google OAuth client; empty = Drive features off |
| `DRIVE_OAUTH_REDIRECT_URI` | `PUBLIC_BASE_URL/kb/api/drive/callback` | Override the redirect URI |
| `DRIVE_INTAKE_FOLDER_ID` | empty | Default Drive folder for document intake |
| `DRIVE_INTAKE_DRIVE_OVERRIDE` | `/data/backup-drives/primary.json` | Drive connection written by the dashboard |
| `VIDEO_DISCOVERY_CHANNEL_URL` | empty | Default channel for "Discover videos" |
| `VIDEO_DESC_SERVICE_TIER` | `flex` | Gemini tier for frame descriptions (flex = lower price, falls back to standard) |
| `VIDEO_VISION_WORKERS` | 16 | Parallel frame-description workers |

### Advanced tuning (haystack-api, optional)

Add any of these to `.env` to override the defaults.

| Area | Variables (default) |
|---|---|
| Collections | `COLLECTION_DOCUMENTS` (documents), `COLLECTION_CODE` (code_docs), `VIDEO_TRANSCRIPT_COLLECTION`, `VIDEO_VISUAL_COLLECTION`, `VIDEO_OCR_COLLECTION`, `VIDEO_OD_COLLECTION` — do not rename once data exists |
| Embeddings | `EMBEDDING_PROVIDER` (voyage), `EMBEDDING_MODEL` (voyage-4-large), `EMBEDDING_DIMENSIONS` (1024), `VOYAGE_BATCH_SIZE` (96), `VOYAGE_RPM_LIMIT` (200), `VOYAGE_MULTIMODAL_MODEL` |
| Retrieval | `RERANKER_PROVIDER` (voyage), `RERANKER_MODEL` (rerank-2.5), `RERANK_TOP_N` (5), `RETRIEVAL_TOP_K` (50), `ENABLE_HYBRID_SEARCH` (true), `ENABLE_TOPIC_EXPANSION` (true) |
| Planner / vision | `VISION_MODEL`, `PLANNER_FALLBACK_ENABLED` (true), `PLANNER_FALLBACK_MODEL`, `PLANNER_CACHE_ENABLED` (true), `GROQ_WHISPER_MODEL` (whisper-large-v3) |
| Chunking | `CHUNK_SIZE_TEXT/CODE/TRANSCRIPT` (500/300/500), `CHUNK_OVERLAP_TEXT/CODE/TRANSCRIPT` (100/50/75), `SEMANTIC_SIMILARITY_THRESHOLD` (0.5), `INGEST_DEDUP` (true), `EXTRACT_MAX_TEXT_CHARS` |
| Archives | `ARCHIVE_MAX_MEMBERS` (512), `ARCHIVE_MAX_MEMBER_BYTES` (128 MiB), `ARCHIVE_MAX_TOTAL_BYTES` (512 MiB), `ARCHIVE_MAX_RATIO` (200) |
| Video | `MAX_CONCURRENT_VIDEOS` (1), `MAX_IN_FLIGHT_VIDEOS` (15), `MAX_QUEUE_BATCH_SIZE` (15), `MAX_INGESTION_SPEND_USD` (50), `MEMORY_WATERMARK_GB` (4), `SSIM_THRESHOLD`, `SAFETY_FRAME_INTERVAL_S` (45); cost estimates `GROQ_COST_PER_HOUR_USD`, `VISION_COST_PER_FRAME_USD`, `VOYAGE_COST_PER_FRAME_USD` |
| Video downloads | `YTDLP_COOKIES_MAX_AGE_DAYS` (14), `YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY` (20), `YTDLP_SLEEP_MIN/MAX` (15/45), `YTDLP_MAX_RETRIES` (3) |
| Evaluation | `EVAL_SAMPLE_RATE` (0.0 = off), `EVAL_JUDGE_PROVIDER`, `EVAL_JUDGE_MODEL`, `EVAL_JUDGE_API_KEY`, `ENABLE_COHERENCE_JUDGE` (false) |

---

## 9. Customizing for a client

Each client gets its own copy of the repository and its own `.env`; the code
stays identical.

| What | How |
|---|---|
| Product name | `APP_NAME` in `.env` (Open WebUI title, dashboard, Drive folders, filter notices via the `app_name` valve) |
| Logo, favicon, chat CSS | Put `favicon.svg`, `favicon.png` and/or `custom.css` in `webui-image/` and uncomment the matching `COPY` lines in `webui-image/Dockerfile`, then `docker compose build open-webui` |
| Domain | `DOMAIN` and `PUBLIC_BASE_URL` |
| Isolation | Separate repository copy, `.env`, `DATA_DIR`, keys and backup repository per client. Use a distinct `STACK_NAME` per client |
| Models | `litellm-config.yaml` (section 7) and the default model in Open WebUI |
| Industry vocabulary | `DOMAIN_JARGON` (comma-separated terms for extraction prompts) |
| Retrieval behaviour | `rag_filter` valves (section 5.4) and the advanced variables in section 8 |
| Which features are on | Leave a feature's keys empty to switch it off: Drive (`DRIVE_*`), audio/video (`GROQ_API_KEY`), tracing (`PHOENIX_API_KEY`), reranking (`RERANKER_API_KEY`) |
| Dashboard colours and sections | `dashboard/components/sections.ts` (one entry per section: label, icon, accent colour, description) |
| Sandbox size | `SANDBOX_MEM_LIMIT`, `SANDBOX_CPUS` |

### Setting up a new client

1. Clone the repository on the client's server (section 4, steps 1–4).
2. Run `python3 setup_env.py` and enter the client's domain, product name, a
   unique stack name, dashboard password and API keys. Every secret is
   generated fresh, so nothing is shared with another client.
3. Optional: logo/CSS, model list, `DOMAIN_JARGON`.
4. `docker compose up -d --build`.
5. Create the Open WebUI admin and complete section 5.
6. Log in to `/kb/`; Administration shows the expected keys as set.
7. Optional: Google Drive, video channel, Phoenix key, backups.
8. Smoke test: upload a document in the dashboard and ask about it in chat
    (check the citation); attach a CSV in chat and ask for a chart.
9. Validate visual and summary-based retrieval with the client's data before
    relying on them (section 15).

---

## 10. Integrations

### 10.1 Google Drive (optional)

With `DRIVE_OAUTH_CLIENT_ID` / `DRIVE_OAUTH_CLIENT_SECRET` empty, the
Integrations page shows "not configured" and no Drive feature is offered.

To enable it:

1. Google Cloud Console → create a project → enable the **Google Drive API**.
2. Configure the OAuth consent screen (add the client's Google account as a
   test user while the app is unpublished).
3. Credentials → Create OAuth client ID → **Web application**; authorised
   redirect URI `https://DOMAIN/kb/api/drive/callback`.
4. Put the client ID and secret in `.env`; `docker compose up -d dashboard`.
5. Dashboard → Integrations → **Connect Google Drive**.

The connection is saved as `DATA_DIR/backup-drives/primary.json` and used for:

- **Backups** — the backup scripts upload to this account.
- **Drive intake** — haystack-api scans a Drive folder and ingests new files,
  with optional auto-approve and cleanup (preview with a dry run first). Intake
  also needs an `rclone.conf` with a `gdrive` remote in `RCLONE_CONFIG_DIR`.
- **Drive browser** — browse and ingest documents from the dashboard.

### 10.2 Video

Add videos by URL in the Video Library, or "Discover videos" from a channel
(entered in the dashboard or `VIDEO_DISCOVERY_CHANNEL_URL`). Upload a cookies
file in the Video Library if the source requires sign-in.
`MAX_INGESTION_SPEND_USD` caps the estimated processing spend. Video processing
needs `GROQ_API_KEY` (transcription) and `GOOGLE_API_KEY` (frame descriptions).

### 10.3 Phoenix tracing

Open `https://DOMAIN/traces/`, sign in as `admin@localhost` with
`PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` and change the password. Create an API
key (Settings → API Keys), set `PHOENIX_API_KEY` and run `docker compose up -d`.
Traces appear under the `multimodal-ai-chat` project.

### 10.4 Stored datasets in the sandbox

CSV/XLSX uploads become PostgreSQL tables prefixed `data_`. Code in the sandbox
loads them with `data_tools.load_dataset(name)`, which authenticates with
`DATA_API_KEY`. That key can only read the two dataset endpoints — the sandbox
runs model-written code and never holds the admin or loader key.

---

## 11. Backups and restore

Backups are encrypted, deduplicated **restic** snapshots stored on **Google
Drive** through **rclone**. They are run on a **Linux host** by the scripts and
systemd units in `ops/`, and controlled from the dashboard's Backups & Restore
section. The `ops/` scripts assume the default `DATA_DIR=/data`.

Each run captures the PostgreSQL database, every Qdrant collection, uploaded
files, haystack-api state and the repository configuration, verifies a
checksummed manifest, uploads one snapshot, runs `restic check` and applies
retention. A failed run is never recorded as a recovery point.

| Schedule (UTC) | Folder | Kept |
|---|---|---|
| 03:00, 11:00, 19:00 (`backup.timer`) | `daily` | newest 3 |
| Friday 20:00 (`backup-weekly.timer`) | `weekly` | newest 2 |
| 1st of month 21:00 (`backup-monthly.timer`) | `monthly` | newest 1 |

### Setup (Linux host, as root, repository at `INSTALL_DIR`)

1. Install `restic` and `rclone`; create a Drive remote named `gdrive` with
   `rclone config` (or connect Drive in the dashboard, section 10.1).
2. In `.env`: `RESTIC_PASSWORD` (`openssl rand -base64 48`, keep an offline
   copy — without it backups cannot be decrypted) and
   `RESTIC_REPOSITORY=rclone:gdrive:backups`.
3. Initialise the three repositories once:
   ```bash
   bash ops/backup.sh --self-test
   ALLOW_REPO_INIT=1 bash ops/backup.sh --target daily
   ALLOW_REPO_INIT=1 bash ops/backup.sh --target weekly
   ALLOW_REPO_INIT=1 bash ops/backup.sh --target monthly
   ```
4. Install the units (edit `Environment=INSTALL_DIR=` / `STACK_NAME=` in them
   first if yours differ):
   ```bash
   install -m 0644 ops/*.service ops/*.timer ops/repoctl.path /etc/systemd/system/
   systemctl daemon-reload
   systemctl enable --now backup.timer backup-weekly.timer backup-monthly.timer
   systemctl enable --now repoctl.path storage-monitor.timer
   ```

Manual runs: `bash ops/backup.sh [--target weekly] [--tag NAME] [--dry-run]`.
Logs: `journalctl -t gmac-backup`, `journalctl -t gmac-repoctl`.

### Restore

- **Dashboard → Backups & Restore** (needs `repoctl.path` enabled): check a
  recovery point, restore deleted documents, chats or videos, or run a full
  restore (type `RESTORE` to confirm; a safety backup is offered first).
- **Restore drill:** `bash ops/restore-drill.sh [--snapshot ID]` restores a
  snapshot into throwaway containers and verifies the database and vectors
  against the manifest without touching live data.
- **New machine:** start the stack with the same `.env` (above all
  `RESTIC_PASSWORD`, `RESTIC_REPOSITORY`, `POSTGRES_PASSWORD`), then run a full
  restore from the dashboard.

---

## 12. Backend API

`haystack-api` listens on port 8000 inside the Docker network (not published).
All protected routes take `Authorization: Bearer <key>`.

| Prefix | Purpose | Accepted key |
|---|---|---|
| `/health` | Health check | none |
| `/extract/process` | Open WebUI external content loader | `EXTERNAL_LOADER_API_KEY` |
| `/search` | Hybrid retrieval with citations | none (internal network only) |
| `/ingest/{document,code,transcript,media,archive,structured}` | Ingest a file | `ADMIN_API_KEY` or `EXTERNAL_LOADER_API_KEY` |
| `/ingest/status`, `/ingest/status/{job_id}` | Ingestion job status | none (internal network only) |
| `/admin/data/search`, `/admin/data/table/{name}` | Read stored datasets | `ADMIN_API_KEY` or `DATA_API_KEY` |
| `/admin/*` | Documents, tables, stats, storage, reindex, backup status | `ADMIN_API_KEY` |
| `/video/*` | Discover, fetch, queue, process, status, settings, cookies, costs, Drive intake | `ADMIN_API_KEY` |
| `/eval/*` | Answer logging (`log-answer`: admin or loader key), evaluation dashboards | `ADMIN_API_KEY` |
| `/costs/retrieval` | Retrieval cost report | `ADMIN_API_KEY` |
| `/dashboard/config`, `/dashboard/providers/health` | Non-secret configuration and provider status | `ADMIN_API_KEY` |

Interactive API docs (FastAPI `/docs`): temporarily add `ports: ["127.0.0.1:8000:8000"]`
to the `haystack-api` service, run `docker compose up -d haystack-api` and open
`http://localhost:8000/docs`. Remove the mapping afterwards.

If no key is configured for a protected route, the API refuses requests
(`503 admin api key not configured`) instead of serving them unauthenticated.

---

## 13. Development and tests

```bash
# Chat filters
cd filters && pip install pytest hypothesis httpx pydantic && python -m pytest -q

# Backend
cd haystack-api && pip install -r requirements.txt pytest && python -m pytest -q

# Sandbox library
cd code-execution && pip install -r requirements.txt -r requirements-dev.txt && python -m pytest tests data_tools/tests -q

# LiteLLM hook / telemetry
cd litellm-hooks && python -m pytest -q
cd processing && python tests/test_telemetry.py

# Dashboard
cd dashboard && npm ci
npx tsc --noEmit && npm run lint && npm test && npm run build

# Backup scripts (offline self-tests)
bash ops/backup.sh --self-test && python3 ops/repoctl.py --self-test
```

Tests in `tests/live/` run against a deployed instance and skip unless
`LIVE_DEPLOYMENT` and `OPENWEBUI_API_KEY` are set.

**Run the dashboard alone** (UI work without the full stack; backend data will
show as unavailable):

```bash
cd dashboard
export ADMIN_USERNAME=admin ADMIN_PASSWORD_HASH='<hash from section 4>' \
       SESSION_SECRET=$(openssl rand -hex 32) ADMIN_API_KEY=dev \
       HAYSTACK_API_URL=http://127.0.0.1:8000 PUBLIC_BASE_URL=http://localhost:3000 \
       APP_NAME="Growlyze Multimodal AI Chat" BACKUP_CTL_DIR=/tmp/backup-ctl
npm run dev        # http://localhost:3000/kb
```

**Verify a running stack:** `bash haystack-api/scripts/verify_deploy.sh` checks
that every service is healthy, the HTTP health endpoints answer, and the Open
WebUI functions match the repository.

---

## 14. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker compose` stops with `set X in .env` | A required value is empty — fill it in |
| Mount errors on macOS/Windows (`/data`, `/root/...`) | Set `DATA_DIR=./data` and `RCLONE_CONFIG_DIR=./data/rclone` |
| Browser warns about the certificate on `https://localhost` | Expected: Caddy's local certificate. Accept it, or use a real domain |
| Caddy logs certificate (ACME) errors on a real domain | DNS does not point at the machine yet, or ports 80/443 are blocked; Caddy retries automatically |
| Irrelevant search results; logs mention `StubProvider` | `EMBEDDING_API_KEY` missing/invalid. Fix it, restart haystack-api, then reindex content ingested meanwhile |
| Chat uploads not added to the knowledge base; `401` on `/ingest/*` in haystack logs | `rag_filter` valve `haystack_api_key` empty or different from `EXTERNAL_LOADER_API_KEY` |
| File upload in chat fails with an extraction error | Documents settings URL/key wrong (5.2) or haystack-api unhealthy (`docker compose logs haystack-api`) |
| `503 admin api key not configured` | `ADMIN_API_KEY` not set for haystack-api |
| Dashboard login always fails | `ADMIN_PASSWORD_HASH` is a placeholder or malformed, or 5 failed attempts — wait 15 minutes |
| Dashboard pages show "backend unreachable" | haystack-api is down, or `ADMIN_API_KEY` differs between containers |
| Integrations shows Drive "not configured" | Expected while `DRIVE_OAUTH_CLIENT_*` are empty |
| Google sign-in `redirect_uri_mismatch` | Register exactly `https://DOMAIN/kb/api/drive/callback` |
| Drive intake "not configured" although Drive is connected | No `rclone.conf` with a `gdrive` remote in `RCLONE_CONFIG_DIR` |
| Model says the sandbox is unavailable | `code-sandbox` not running, or the Open Terminal URL/key is wrong (5.3) |
| `load_dataset()` returns an empty table | `DATA_API_KEY` missing, or the table does not exist (Knowledge Base → tables) |
| A model alias fails | Its provider key is empty or invalid — `docker compose logs litellm` |
| Video downloads fail | Cookies expired (upload new ones) or the source rate-limits; check `bgutil-provider` is running |
| Backup: `RESTIC_REPOSITORY must be an rclone backend` | Use `rclone:gdrive:<folder>` |
| Backup: repository not found on first run | Initialise with `ALLOW_REPO_INIT=1` (section 11) |
| Dashboard backup request stays pending | `repoctl.path` is not enabled on the host |

---

## 15. Known limitations and features requiring further validation

These two items are documented, not fixed, in this release.

### A. Visual retrieval

Visual retrieval (answering from video frames, images, OCR and visual
descriptions) has **not been tested sufficiently** to establish its
reliability. It requires additional testing before being offered as a verified
client capability. Its reliability has not been fully established, and it
should be validated using representative client data before production use. It
must not be presented as tested or production-ready.

### B. Summary-based retrieval

It has been observed that the retrieval system sometimes generates answers
using **summaries rather than retrieving the actual relevant source content**.
Retrieval may sometimes rely on summaries. Answers that require specific
original-source evidence should be independently validated, and this should be
considered when evaluating retrieval reliability.
