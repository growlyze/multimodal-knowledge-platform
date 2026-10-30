#!/usr/bin/env bash
# verify_deploy.sh — standing check for the "deploy looked successful but served stale code"
# failure mode. Run it after every deploy/update instead of trusting `git log`,
# `docker compose ps` alone, or a filter's own `ok`-printing self-check (a filter once printed
# `ok` for days while it was disabled in Postgres).
#
# Failure modes and checks:
#   1. Every service defined in compose.yml must be running (and healthy where the service
#      defines a healthcheck). Service names are read from `docker compose config --services`,
#      so the list always matches the compose file in this checkout.
#   2. haystack-api and code-sandbox bake code into the image at BUILD time. `git pull &&
#      docker compose up -d <svc>` without `--build` restarts the OLD image and looks like a
#      successful deploy. Checks 2-3 catch this (data_tools version parity in the sandbox; a
#      current-tip symbol inside the haystack-api image) and check 4 probes the HTTP health
#      endpoints.
#   3. Open WebUI Functions (filters/rag_filter.py, filters/data_tools_filter.py) and Tools
#      (filters/dataset_tool.py) are deployed by pasting into Admin -> Functions / Admin -> Tools.
#      Open WebUI stores them in PostgreSQL, so a commit changes nothing live until someone
#      pastes it. Checks 5-6 compare is_active and the stored content with the repo files.
#
# Where it runs:
#   - on the VPS, from the install directory (default), or
#   - from a workstation: set VERIFY_SSH_HOST=<ssh alias> and VERIFY_REMOTE_DIR=<INSTALL_DIR>;
#     every docker command then runs over ssh (non-interactive ssh required).
#
# Configuration (environment):
#   VERIFY_SSH_HOST     ssh alias of the VPS (empty = run docker locally)
#   VERIFY_REMOTE_DIR   install directory on the VPS when using ssh (default /opt/gmac)
#   STACK_NAME          compose STACK_NAME, the container-name prefix (default gmac; read from
#                       .env when present)
#   VERIFY_FUNCTIONS    space-separated <open-webui id>=<repo path> pairs for Functions
#                       (default: "rag_filter=filters/rag_filter.py data_tools_filter=filters/data_tools_filter.py")
#   VERIFY_TOOLS        same for Tools (default: "dataset_tool=filters/dataset_tool.py")
#   The ids are the ones chosen when the function/tool was created in Open WebUI.
#
# Usage:
#   cd <INSTALL_DIR> && bash haystack-api/scripts/verify_deploy.sh
#   bash haystack-api/scripts/verify_deploy.sh --self-test     # offline helper checks
#
# Exit code 0 only if every check PASSes. A FAIL line always states the command that fixes it.

set -uo pipefail

# The `version:` line of an Open WebUI Function/Tool front-matter docstring
# (e.g. "version: 1.38"), from a file. Anchored at line start so
# `required_open_webui_version:` never matches. Empty when absent.
extract_version() {   # <file>
    tr -d '\r' < "$1" | sed -n 's/^version:[[:space:]]*\([^[:space:]]*\).*/\1/p' | head -n 1
}

# One line naming both versions, printed under every content check (PASS or
# FAIL). Informational only: PASS/FAIL stays the byte comparison, because a
# same-version edit is still drift.
version_line() {   # <deployed_file> <repo_file>
    local dep repo
    dep=$(extract_version "$1"); repo=$(extract_version "$2")
    if [ -n "$dep" ] && [ "$dep" = "$repo" ]; then
        printf '     version: deployed=%s repo=%s\n' "$dep" "${repo:-<none>}"
    else
        printf '     version: deployed=%s repo=%s  <- DIFFERENT: the deployed copy is not this repo'"'"'s version\n' \
            "${dep:-<none>}" "${repo:-<none>}"
    fi
}

# --self-test: offline checks of the pure helpers above (no docker, no ssh).
if [ "${1:-}" = "--self-test" ]; then
    st_dir="$(mktemp -d)"; n=0
    st_check() {   # <name> <expected> <actual>
        if [ "$2" != "$3" ]; then printf 'FAIL %s\n  expected: %s\n  got:      %s\n' "$1" "$2" "$3" >&2; rm -rf "$st_dir"; exit 1; fi
        n=$((n + 1)); printf 'ok %s\n' "$1"
    }
    printf '"""\ntitle: RAG Filter\nauthor: x\nversion: 1.38\n"""\n' > "$st_dir/f.py"
    st_check extract_version_filter "1.38" "$(extract_version "$st_dir/f.py")"
    # A Tool's header may also have required_open_webui_version: - must not win.
    printf '"""\r\ntitle: Search Datasets\r\nrequired_open_webui_version: 0.11.0\r\nversion: 1.3.0\r\n"""\r\n' > "$st_dir/t.py"
    st_check extract_version_tool_crlf "1.3.0" "$(extract_version "$st_dir/t.py")"
    printf 'no header here\n' > "$st_dir/n.py"
    st_check extract_version_missing "" "$(extract_version "$st_dir/n.py")"
    st_check version_line_same "     version: deployed=1.38 repo=1.38" \
        "$(version_line "$st_dir/f.py" "$st_dir/f.py")"
    printf 'version: 1.37\n' > "$st_dir/old.py"
    st_check version_line_differs "     version: deployed=1.37 repo=1.38  <- DIFFERENT: the deployed copy is not this repo's version" \
        "$(version_line "$st_dir/old.py" "$st_dir/f.py")"
    st_check version_line_missing "     version: deployed=<none> repo=1.38  <- DIFFERENT: the deployed copy is not this repo's version" \
        "$(version_line "$st_dir/n.py" "$st_dir/f.py")"
    rm -rf "$st_dir"
    printf '\n%d verify_deploy self-checks passed\n' "$n"
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_HOST="${VERIFY_SSH_HOST:-}"
REMOTE_DIR="${VERIFY_REMOTE_DIR:-/opt/gmac}"
if [ -z "${STACK_NAME:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
    STACK_NAME=$(sed -n 's/^STACK_NAME=\([^[:space:]#]*\).*/\1/p' "$REPO_ROOT/.env" | head -n 1)
fi
STACK_NAME="${STACK_NAME:-gmac}"
VERIFY_FUNCTIONS="${VERIFY_FUNCTIONS:-rag_filter=filters/rag_filter.py data_tools_filter=filters/data_tools_filter.py}"
VERIFY_TOOLS="${VERIFY_TOOLS:-dataset_tool=filters/dataset_tool.py}"
FAIL=0
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Run a shell command where the stack lives: locally from the repo root, or over ssh from
# the install directory.
on_host() {   # <command string>
    if [ -n "$SSH_HOST" ]; then
        ssh "$SSH_HOST" "cd '$REMOTE_DIR' && $1"
    else
        (cd "$REPO_ROOT" && bash -c "$1")
    fi
}
if [ -n "$SSH_HOST" ]; then
    FIX_PREFIX="ssh $SSH_HOST 'cd $REMOTE_DIR && "; FIX_SUFFIX="'"
else
    FIX_PREFIX="cd $REPO_ROOT && "; FIX_SUFFIX=""
fi
PSQL="docker compose exec -T postgres psql -U openwebui -d openwebui -tA"

echo "=== verify_deploy.sh — $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "repo: $REPO_ROOT   stack: $STACK_NAME   host: ${SSH_HOST:-local}"
echo

# ---------------------------------------------------------------------------------------------
# 1. Every compose service is running; services with a healthcheck report healthy.
# ---------------------------------------------------------------------------------------------
SERVICES=$(on_host "docker compose config --services" 2>/dev/null | tr -d '\r')
RUNNING=$(on_host "docker compose ps --status running --services" 2>/dev/null | tr -d '\r')
if [ -z "$SERVICES" ]; then
    echo "FAIL [1] could not read the compose services (docker compose config failed — is .env complete?)"
    echo "     fix: ${FIX_PREFIX}docker compose config -q${FIX_SUFFIX}"
    FAIL=1
else
    for svc in $SERVICES; do
        if ! printf '%s\n' "$RUNNING" | grep -qx "$svc"; then
            echo "FAIL [1] service '$svc' is not running"
            echo "     fix: ${FIX_PREFIX}docker compose up -d $svc && docker compose logs --tail 50 $svc${FIX_SUFFIX}"
            FAIL=1
            continue
        fi
        cid=$(on_host "docker compose ps -q $svc" 2>/dev/null | tr -d '\r' | head -n 1)
        health=$(on_host "docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' $cid" 2>/dev/null | tr -d '\r')
        case "$health" in
            healthy|none) echo "PASS [1] service '$svc' running (health: $health)" ;;
            starting)     echo "PASS [1] service '$svc' running (health: starting — re-run in a minute)" ;;
            *)            echo "FAIL [1] service '$svc' running but health=$health"
                          echo "     fix: ${FIX_PREFIX}docker compose logs --tail 100 $svc${FIX_SUFFIX}"
                          FAIL=1 ;;
        esac
    done
fi
echo

# ---------------------------------------------------------------------------------------------
# 2. data_tools version parity (code-sandbox) — the sandbox image bakes the library in.
# ---------------------------------------------------------------------------------------------
INIT_FILE="$REPO_ROOT/code-execution/data_tools/__init__.py"
REPO_DT_VERSION=$(grep '__version__ = "' "$INIT_FILE" 2>/dev/null | head -1 | sed -E 's/.*__version__ = "([^"]+)".*/\1/')
DEPLOYED_DT_VERSION=$(on_host "docker compose exec -T code-sandbox python -c 'import data_tools; print(data_tools.__version__)'" 2>/dev/null | tr -d '\r\n')

if [ -n "$DEPLOYED_DT_VERSION" ] && [ "$REPO_DT_VERSION" = "$DEPLOYED_DT_VERSION" ]; then
    echo "PASS [2] data_tools version parity: repo=$REPO_DT_VERSION deployed=$DEPLOYED_DT_VERSION"
else
    echo "FAIL [2] data_tools version mismatch: repo=${REPO_DT_VERSION:-<none>} deployed=${DEPLOYED_DT_VERSION:-<unreachable>}"
    echo "     fix: ${FIX_PREFIX}git pull && docker compose build code-sandbox && docker compose up -d code-sandbox${FIX_SUFFIX}"
    FAIL=1
fi
echo

# ---------------------------------------------------------------------------------------------
# 3. haystack-api silent-rebuild-skip check — a distinctive symbol from the current tip must
#    exist in the running container's baked-in copy. If the symbol is ever renamed, swap
#    SYMBOL/SYMBOL_FILE for one the latest haystack-api change added.
# ---------------------------------------------------------------------------------------------
SYMBOL="require_admin_or_loader"
SYMBOL_FILE="services/auth.py"
REPO_SYMBOL_COUNT=$(grep -c "$SYMBOL" "$REPO_ROOT/haystack-api/$SYMBOL_FILE" 2>/dev/null)
DEPLOYED_SYMBOL_COUNT=$(on_host "docker compose exec -T haystack-api grep -c $SYMBOL /app/$SYMBOL_FILE" 2>/dev/null | tr -d '\r\n')
DEPLOYED_SYMBOL_COUNT=${DEPLOYED_SYMBOL_COUNT:-0}

if [ "$DEPLOYED_SYMBOL_COUNT" -ge 1 ] 2>/dev/null; then
    echo "PASS [3] haystack-api image is current: '$SYMBOL' found $DEPLOYED_SYMBOL_COUNT time(s) in /app/$SYMBOL_FILE (repo: ${REPO_SYMBOL_COUNT:-0})"
else
    echo "FAIL [3] haystack-api image is STALE: '$SYMBOL' not found in /app/$SYMBOL_FILE (repo: ${REPO_SYMBOL_COUNT:-0})"
    echo "     fix: ${FIX_PREFIX}git pull && docker compose build haystack-api && docker compose up -d haystack-api${FIX_SUFFIX}"
    FAIL=1
fi
echo

# ---------------------------------------------------------------------------------------------
# 4. HTTP health endpoints, probed from inside the containers (nothing is published but Caddy).
# ---------------------------------------------------------------------------------------------
probe() {   # <label> <service> <python-or-wget command>
    if on_host "docker compose exec -T $2 $3" > /dev/null 2>&1; then
        echo "PASS [4] $1 answers"
    else
        echo "FAIL [4] $1 does not answer"
        echo "     fix: ${FIX_PREFIX}docker compose logs --tail 100 $2${FIX_SUFFIX}"
        FAIL=1
    fi
}
PYGET="python -c \"import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=10)\""
probe "haystack-api GET /health" haystack-api "$PYGET http://localhost:8000/health"
probe "qdrant GET /readyz (from haystack-api)" haystack-api "$PYGET http://qdrant:6333/readyz"
probe "litellm GET /health/liveliness (from haystack-api)" haystack-api "$PYGET http://litellm:4000/health/liveliness"
probe "dashboard GET /kb/api/health" dashboard "wget -q -O /dev/null http://localhost:3000/kb/api/health"
echo

# ---------------------------------------------------------------------------------------------
# 5. Function is_active flags — read the fact Open WebUI acts on, directly.
# ---------------------------------------------------------------------------------------------
FUNC_IDS=""
for pair in $VERIFY_FUNCTIONS; do FUNC_IDS="$FUNC_IDS${FUNC_IDS:+,}'${pair%%=*}'"; done
FUNCTION_ROWS=$(on_host "$PSQL -c \"SELECT id, is_active FROM function WHERE id IN ($FUNC_IDS);\"" 2>/dev/null | tr -d '\r')

if [ -z "$FUNCTION_ROWS" ]; then
    echo "FAIL [5] no rows in function for ids ($FUNC_IDS) — not pasted yet, different ids (set VERIFY_FUNCTIONS), or Postgres unreachable"
    FAIL=1
else
    while IFS='|' read -r fid active; do
        [ -z "$fid" ] && continue
        if [ "$active" = "t" ]; then
            echo "PASS [5] Function '$fid' is_active=true"
        else
            echo "FAIL [5] Function '$fid' is_active=$active — chat answers without it and nothing errors"
            echo "     fix: Open WebUI -> Admin -> Functions -> enable '$fid'"
            FAIL=1
        fi
    done <<< "$FUNCTION_ROWS"
fi
echo

# ---------------------------------------------------------------------------------------------
# 6. Function / Tool content parity — compares the live Postgres `content` column with the repo
#    file, ignoring CRLF and trailing-newline noise (a browser paste normalises both).
# ---------------------------------------------------------------------------------------------
check_content() {   # <open-webui id> <repo path> <table: function|tool>
    local func_id="$1" repo_rel="$2" table="$3"
    local kind="Function" screen="Admin -> Functions"
    if [ "$table" = "tool" ]; then kind="Tool"; screen="Admin -> Tools"; fi
    local repo_file="$REPO_ROOT/$repo_rel"
    local tmp_file="$TMP_DIR/${func_id}.deployed"

    if [ ! -f "$repo_file" ]; then
        echo "FAIL [6] repo file $repo_rel does not exist (fix VERIFY_FUNCTIONS / VERIFY_TOOLS)"
        FAIL=1
        return
    fi
    on_host "$PSQL -c \"SELECT content FROM ${table} WHERE id='${func_id}';\"" > "$tmp_file" 2>/dev/null

    if [ ! -s "$tmp_file" ]; then
        echo "FAIL [6] could not read ${table}.content for '$func_id' from Postgres (not pasted yet?)"
        echo "     fix: paste $repo_rel into $screen with id '$func_id', Save, Enable"
        FAIL=1
        return
    fi

    local tmp_file_norm="$TMP_DIR/${func_id}.deployed.norm"
    local repo_file_norm="$TMP_DIR/${func_id}.repo.norm"
    printf '%s\n' "$(tr -d '\r' < "$tmp_file")" > "$tmp_file_norm"
    printf '%s\n' "$(tr -d '\r' < "$repo_file")" > "$repo_file_norm"

    local deployed_bytes local_bytes
    deployed_bytes=$(wc -c < "$tmp_file_norm" | tr -d ' ')
    local_bytes=$(wc -c < "$repo_file_norm" | tr -d ' ')

    if diff -q "$tmp_file_norm" "$repo_file_norm" > /dev/null 2>&1; then
        echo "PASS [6] $kind '$func_id' content matches repo $repo_rel (${local_bytes}B, line endings ignored)"
    else
        echo "FAIL [6] $kind '$func_id' content DIVERGES from repo $repo_rel (deployed=${deployed_bytes}B, repo=${local_bytes}B, line endings normalized)"
        echo "     fix: paste the current $repo_rel into $screen -> '$func_id', Save, Enable"
        echo "     to see the actual diff: diff '$repo_file_norm' '$tmp_file_norm' (kept until this script exits)"
        FAIL=1
    fi
    version_line "$tmp_file_norm" "$repo_file_norm"
}

for pair in $VERIFY_FUNCTIONS; do check_content "${pair%%=*}" "${pair#*=}" function; done
for pair in $VERIFY_TOOLS; do check_content "${pair%%=*}" "${pair#*=}" tool; done
echo

# ---------------------------------------------------------------------------------------------
# 7. Qdrant collections exist — the configured document/code pair must be present.
#    Names read from .env (COLLECTION_DOCUMENTS / COLLECTION_CODE) with the shipped defaults.
# ---------------------------------------------------------------------------------------------
COLLECTION_DOCUMENTS="${COLLECTION_DOCUMENTS:-$(sed -n 's/^COLLECTION_DOCUMENTS=\([^[:space:]#]*\).*/\1/p' "$REPO_ROOT/.env" 2>/dev/null | head -n 1)}"
COLLECTION_CODE="${COLLECTION_CODE:-$(sed -n 's/^COLLECTION_CODE=\([^[:space:]#]*\).*/\1/p' "$REPO_ROOT/.env" 2>/dev/null | head -n 1)}"
COLLECTION_DOCUMENTS="${COLLECTION_DOCUMENTS:-documents}"
COLLECTION_CODE="${COLLECTION_CODE:-code_docs}"
QCOLLS=$(on_host "docker compose exec -T haystack-api $PYGET http://qdrant:6333/collections" 2>/dev/null | tr -d '\r')
for coll in "$COLLECTION_DOCUMENTS" "$COLLECTION_CODE"; do
    if printf '%s' "$QCOLLS" | grep -q "\"$coll\""; then
        echo "PASS [7] Qdrant collection '$coll' exists"
    else
        echo "FAIL [7] Qdrant collection '$coll' is missing"
        echo "     fix: ingest a document into '$coll' (POST /ingest/document) or check QDRANT_URL"
        FAIL=1
    fi
done
echo

# ---------------------------------------------------------------------------------------------
# 8. Admin endpoints respond — with the admin key from .env, and NOT without it.
# ---------------------------------------------------------------------------------------------
ADMIN_API_KEY="${ADMIN_API_KEY:-$(sed -n 's/^ADMIN_API_KEY=\([^[:space:]#]*\).*/\1/p' "$REPO_ROOT/.env" 2>/dev/null | head -n 1)}"
if [ -z "$ADMIN_API_KEY" ]; then
    echo "FAIL [8] ADMIN_API_KEY is not set in .env — the admin endpoints fail closed (503)"
    echo "     fix: set ADMIN_API_KEY in $REPO_ROOT/.env and docker compose up -d haystack-api"
    FAIL=1
else
    AUTHGET="python -c \"import sys,urllib.request; r=urllib.request.Request(sys.argv[1], headers={'Authorization':'Bearer '+sys.argv[2]}); urllib.request.urlopen(r, timeout=10)\""
    NOAUTH="python -c \"import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=10)\""
    if on_host "docker compose exec -T haystack-api $AUTHGET http://localhost:8000/admin/stats '$ADMIN_API_KEY'" > /dev/null 2>&1; then
        echo "PASS [8] GET /admin/stats answers with the admin key"
    else
        echo "FAIL [8] GET /admin/stats does not answer with the admin key"
        echo "     fix: ${FIX_PREFIX}docker compose logs --tail 100 haystack-api${FIX_SUFFIX}"
        FAIL=1
    fi
    if on_host "docker compose exec -T haystack-api $NOAUTH http://localhost:8000/admin/stats" > /dev/null 2>&1; then
        echo "FAIL [8] GET /admin/stats answers WITHOUT a key — the auth guard is not active"
        echo "     fix: check services/auth.py wiring in the deployed image (see check 3)"
        FAIL=1
    else
        echo "PASS [8] GET /admin/stats refuses unauthenticated calls"
    fi
fi
echo

echo "=== $([ "$FAIL" -eq 0 ] && echo "ALL CHECKS PASSED" || echo "FAILURES ABOVE — see 'fix:' lines") ==="
exit "$FAIL"
