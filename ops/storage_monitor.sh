#!/usr/bin/env bash
# Storage monitor + sandbox file cleanup (P160). Runs hourly on the VPS host.
#
# Why a host script and not haystack-api: no container can see the host disk,
# the sandbox's tmpfs, the docker log directory and `docker system df` at once.
# The host can, through `docker exec`. haystack-api only READS what this writes
# (GET /admin/storage), and the dashboard's dashboard shows it.
#
# Each run:
#   1. cleanup  — deletes regular files older than 7 days under the sandbox's
#                 /home/user and /tmp (tmpfs, 2 GB + 1 GB, shared by every user;
#                 nothing else ever deletes them, and a full tmpfs fails writes
#                 with no warning). Dotfiles/dot-dirs (/home/user/.local) are kept.
#   2. probes   — measures every store in PROBES below, plus (2026-10-25) the
#                 sandbox's memory against its container limit and one near-free
#                 health call per model provider (Gemini, Voyage, Anthropic).
#                 They ride on this report because it already has the hourly
#                 schedule, the failed-unit signal and the panel's red banner.
#   3. report   — writes ONE row into ops_storage_reports (Open WebUI's Postgres,
#                 the same database haystack-api reads) and drops rows > 30 days.
#
# Never fails silently: a probe that cannot be measured is still written, as a
# probe with an "error" field (the panel shows it red), logged to journald at
# err priority, and the run exits 1 so `systemctl status` shows it failed. The
# probes that did work are written regardless.
#
# Usage:
#   storage_monitor.sh              # normal run (what the timer does)
#   storage_monitor.sh --dry-run    # measure + print JSON; delete nothing, write nothing
#   storage_monitor.sh --self-test  # offline parser checks, no docker needed
#
# Install (on the VPS, as root). /opt/gmac IS the git checkout, so `git pull`
# puts this file at /opt/gmac/ops/storage_monitor.sh, where the unit expects
# it; it is committed with mode 100755, so no chmod (which would also dirty the
# checkout). The unit files must be copied — systemd does not read the repo:
#   cd /opt/gmac && git pull
#   test -x ops/storage_monitor.sh && bash ops/storage_monitor.sh --self-test
#   install -m 0644 ops/storage-monitor.service ops/storage-monitor.timer \
#       /etc/systemd/system/
#   systemctl daemon-reload
#   systemctl enable --now storage-monitor.timer
#   systemctl start storage-monitor.service        # first run now, not in an hour
#   journalctl -t gmac-storage-monitor --since "10 min ago"
#   systemctl list-timers storage-monitor.timer
#
# Adding a store: append one line to PROBES (name|label|function|args...).
# Thresholds travel with the probe (warn_pct/alert_pct, warn_bytes/alert_bytes)
# into the report row, so the endpoint grades by what was configured at the time.

# Not `set -e`: one failed probe must not stop the others from being measured.
set -uo pipefail

STACK_NAME=${STACK_NAME:-gmac}
SANDBOX=${SANDBOX:-${STACK_NAME}-code-sandbox-1}
POSTGRES=${POSTGRES:-${STACK_NAME}-postgres}
HAYSTACK=${HAYSTACK:-${STACK_NAME}-haystack}
DOCKER_ROOT=${DOCKER_ROOT:-/var/lib/docker}
TAG=gmac-storage-monitor
MAX_AGE_DAYS=7
RETAIN_DAYS=30
SANDBOX_CLEAN_PATHS="/home/user /tmp"
# An array so --self-test can run the real find expression locally (empty prefix).
SANDBOX_EXEC=(docker exec "$SANDBOX")
# Arrays for the same reason: --self-test swaps in a fixture function.
# MEM_EXEC is followed by the container name; it prints memory.current,
# memory.max, then memory.stat (cgroup v2, read from inside the container, so
# no host cgroup path or container id is needed). Read-only.
MEM_EXEC=(mem_exec_docker)
# PROVIDER_EXEC is followed by one provider name and prints provider_health.py's
# JSON line. The script ships in the haystack-api image (scripts/), which also
# holds the keys; the host never sees a key. `timeout` is a backstop: each
# probe has its own 15 s HTTP timeout, but a hung `docker exec` must not hold
# the hourly run (TimeoutStartSec=15min) hostage.
PROVIDER_EXEC=(timeout 90 docker exec -e PYTHONPATH=/app -w /app "$HAYSTACK"
               python scripts/provider_health.py probe)

GB=1000000000
MB=1000000

# name|label|function|args... — args are passed to the function after name and
# label. key=value args become extra probe fields (thresholds, unit).
PROBES=(
    # One disk (/dev/sda1) holds all three today; the probe merges paths that
    # share a mount, and splits them by itself if /data ever gets its own disk.
    # 75/90 rather than 70/80: 80% of 415 GB still leaves 80 GB.
    "host|Host disk|probe_host_fs|/ /data $DOCKER_ROOT|warn_pct=75|alert_pct=90"
    "sandbox_home|Sandbox /home/user (tmpfs, all users)|probe_sandbox_df|/home/user|warn_pct=70|alert_pct=80"
    "sandbox_tmp|Sandbox /tmp (tmpfs)|probe_sandbox_df|/tmp|warn_pct=70|alert_pct=80"
    # Inside ${STACK_NAME}-haystack's writable layer, not a mount: it fills the host
    # disk through overlay2, invisibly to every `du /data` below.
    "video_work|Video ingest work dir (${STACK_NAME}-haystack)|probe_video_work|/data/video-ingest-work|alert_bytes=$((20 * GB))"
    "docker|Docker|probe_docker_df|warn_bytes=$((20 * GB))"
    "docker_logs|Container logs|probe_docker_logs|alert_bytes=$((500 * MB))"
    "postgres|Postgres database|probe_postgres|alert_bytes=$((5 * GB))"
    "qdrant|Qdrant storage|probe_du|/data/qdrant|alert_bytes=$((10 * GB))"
    "openwebui_data|Open WebUI data (all)|probe_du|/data/openwebui"
    "openwebui_vector_db|Open WebUI vector DB (Chroma)|probe_du|/data/openwebui/vector_db"
    "openwebui_uploads|Open WebUI uploads|probe_du|/data/openwebui/uploads|alert_bytes=$((20 * GB))"
    "phoenix|Phoenix traces|probe_du|/data/phoenix"
    # O4 (2026-10-25): the sandbox runs every user's code in one container with
    # a 12 GB limit (compose mem_limit); past it the kernel OOM-kills whatever
    # is running, mid-answer, with nothing on screen but a dead tool call. Its
    # tmpfs files count against the same limit. 80% leaves time to act.
    "sandbox_mem|Sandbox memory (of its container limit)|probe_container_mem|$SANDBOX|warn_pct=70|alert_pct=80"
    # Model providers: one bounded, near-free call each (Gemini a few output
    # tokens, Voyage a 1-word embedding, Anthropic a 1-token reply: well under
    # a cent a day at hourly). A model listing would be free but cannot see an
    # empty credit balance, which is exactly what took Gemini down unnoticed on
    # 2026-10-22 (README.md, Troubleshooting). Only credit/key/model problems alert;
    # a rate limit or timeout is written on the row and logged, not alerted.
    "provider_gemini|Model provider: Google Gemini|probe_provider|gemini"
    "provider_voyage|Model provider: Voyage (embeddings)|probe_provider|voyage"
    "provider_anthropic|Model provider: Anthropic|probe_provider|anthropic"
)

# ── logging ───────────────────────────────────────────────

FAILED=0
QUIET_LOG=0     # --self-test: nothing goes to journald

log() {
    [ "$QUIET_LOG" = 1 ] || logger -t "$TAG" -- "$*" 2>/dev/null || true
    # Under systemd stderr is the journal too; only echo for a human at a tty,
    # or every line lands in the journal twice.
    [ -t 2 ] && printf '%s\n' "$*" >&2
    return 0
}

fail() {
    FAILED=1
    [ "$QUIET_LOG" = 1 ] || logger -t "$TAG" -p user.err -- "ERROR: $*" 2>/dev/null || true
    printf 'ERROR: %s\n' "$*" >&2
}

# ── pure helpers (covered by --self-test) ────────────────

# `df -B1 -P` -> "used avail total mount". -P keeps one line per filesystem
# even when the device name is long; without it df wraps and $3 is wrong.
parse_df() {
    awk 'NR == 2 { print $3, $4, $2, $6 }'
}

# `df -i -P` -> "iused itotal mount"
parse_dfi() {
    awk 'NR == 2 { print $3, $2, $6 }'
}

# Percentage with one decimal. df's own Capacity is used/(used+avail), not
# used/total: root-reserved blocks are not writable, so that is what fills.
pct_of() {   # pct_of <used> <denominator>
    awk -v u="$1" -v d="$2" 'BEGIN { if (d > 0) printf "%.1f", u * 100 / d; else print "" }'
}

# Docker's human sizes ("11.68GB", "4.096kB", "0B") are decimal (go-units
# HumanSize, base 1000). `docker system df` has no bytes flag.
docker_size_to_bytes() {
    awk -v s="$1" 'BEGIN {
        if (match(s, /^[0-9.]+/) == 0) { print ""; exit }
        n = substr(s, 1, RLENGTH); u = toupper(substr(s, RLENGTH + 1))
        m = 1
        if (u == "KB") m = 1e3; else if (u == "MB") m = 1e6
        else if (u == "GB") m = 1e9; else if (u == "TB") m = 1e12
        else if (u != "B") { print ""; exit }
        printf "%.0f\n", n * m
    }'
}

# `docker system df --format '{{.Type}}\t{{.Size}}\t{{.Reclaimable}}'` ->
# "type_slug size_bytes reclaimable_bytes". Tab-separated on purpose: the type
# "Build Cache" has a space, and Reclaimable carries " (48%)".
parse_system_df() {
    local type size recl
    while IFS=$'\t' read -r type size recl; do
        [ -n "$type" ] || continue
        type=$(printf '%s' "$type" | tr 'A-Z ' 'a-z_')
        printf '%s %s %s\n' "$type" "$(docker_size_to_bytes "$size")" \
            "$(docker_size_to_bytes "${recl%% *}")"
    done
}

# memory.current, memory.max, memory.stat lines -> "used limit". used is
# memory.current minus inactive_file (reclaimable page cache), which is what
# `docker stats` shows and what the OOM killer acts on. limit stays "max" when
# the cgroup has none. Empty output when the first two lines are not numbers.
parse_cgroup_mem() {
    awk 'NR == 1 { cur = $1 } NR == 2 { lim = $1 }
         NR > 2 && $1 == "inactive_file" { inact = $2 }
         END {
             if (cur !~ /^[0-9]+$/ || (lim !~ /^[0-9]+$/ && lim != "max")) exit
             used = cur - inact; if (used < 0) used = 0
             printf "%.0f %s\n", used, lim
         }'
}

# One provider_health.py JSON line -> "category|http_status|detail". The line
# is json.dumps of a dataclass, so the key order is fixed and detail is last;
# category comes from a fixed vocabulary. detail keeps its JSON escapes (it is
# only ever shown, never parsed). Unparseable -> "||".
parse_provider_line() {
    local line=$1 cat status detail
    cat=$(printf '%s' "$line" | sed -n 's/.*"category": "\([a-z_]*\)".*/\1/p')
    status=$(printf '%s' "$line" | sed -n 's/.*"http_status": \([0-9]*\).*/\1/p')
    detail=$(printf '%s' "$line" | sed -n 's/.*"detail": "\(.*\)"}[[:space:]]*$/\1/p')
    printf '%s|%s|%s' "$cat" "$status" "$detail"
}

# "<size> <path>" lines (find -printf '%s %p\n') -> "total largest_size largest_path"
sum_and_largest() {
    awk '{ t += $1; if ($1 + 0 >= big + 0) { big = $1; $1 = ""; path = substr($0, 2) } }
         END { printf "%.0f %.0f %s\n", t, big, path }'
}

mount_slug() {   # "/" -> root, "/var/lib/docker" -> var_lib_docker
    if [ "$1" = "/" ]; then echo root; return; fi
    printf '%s' "${1#/}" | tr -c 'a-zA-Z0-9\n' '_'
    echo
}

json_str() {
    local s=$1
    s=${s//\\/\\\\}; s=${s//\"/\\\"}
    s=${s//$'\n'/ }; s=${s//$'\r'/ }; s=${s//$'\t'/ }
    printf '"%s"' "$s"
}

json_val() {   # numbers stay numbers, empty is null, anything else a string
    if [ -z "$1" ]; then printf 'null'
    elif [[ $1 =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then printf '%s' "$1"
    else json_str "$1"
    fi
}

# probe_json <name> <label> <used> <total> <pct> <path> [key=value ...]
probe_json() {
    local name=$1 label=$2 used=$3 total=$4 pct=$5 path=$6 kv
    shift 6
    printf '{"name":%s,"label":%s,"used_bytes":%s,"total_bytes":%s,"pct":%s,"path":%s' \
        "$(json_str "$name")" "$(json_str "$label")" "$(json_val "$used")" \
        "$(json_val "$total")" "$(json_val "$pct")" "$(json_str "$path")"
    for kv in "$@"; do
        [[ $kv == *=* ]] || continue
        printf ',%s:%s' "$(json_str "${kv%%=*}")" "$(json_val "${kv#*=}")"
    done
    printf '}\n'
}

# ── probes ────────────────────────────────────────────────
# Each prints one probe_json line per measurement and returns non-zero (with
# the reason on stderr) when it could not measure. Lines printed before a
# failure are kept.

probe_host_fs() {   # <name> <label> "<paths>" [kv...]
    local label=$2 paths=$3; shift 3
    local p used avail total mount i rc=0
    local -a mounts=() mpaths=() mline=()
    for p in $paths; do
        mount=""    # a failed df must not inherit the previous path's mount
        read -r used avail total mount < <(df -B1 -P "$p" | parse_df) || true
        if [ -z "${mount:-}" ]; then echo "df failed for $p" >&2; rc=1; continue; fi
        for i in "${!mounts[@]}"; do
            if [ "${mounts[$i]}" = "$mount" ]; then mpaths[$i]+=", $p"; continue 2; fi
        done
        mounts+=("$mount"); mpaths+=("$p"); mline+=("$used $avail $total")
    done
    for i in "${!mounts[@]}"; do
        read -r used avail total <<< "${mline[$i]}"
        local slug; slug=$(mount_slug "${mounts[$i]}")
        probe_json "host_$slug" "$label (${mounts[$i]})" "$used" "$total" \
            "$(pct_of "$used" $((used + avail)))" "${mpaths[$i]}" "$@"
        local iused itotal
        read -r iused itotal _ < <(df -i -P "${mounts[$i]}" | parse_dfi) || true
        if [ -n "${itotal:-}" ] && [ "$itotal" -gt 0 ] 2>/dev/null; then
            # Inodes: many small files can exhaust a disk with bytes to spare.
            probe_json "host_${slug}_inodes" "$label inodes (${mounts[$i]})" "$iused" "$itotal" \
                "$(pct_of "$iused" "$itotal")" "${mounts[$i]}" unit=inodes warn_pct=70 alert_pct=80
        else
            echo "df -i gave no inode count for ${mounts[$i]}" >&2; rc=1
        fi
    done
    return $rc
}

probe_sandbox_df() {   # <name> <label> <path> [kv...]
    local name=$1 label=$2 path=$3 used avail total mount out; shift 3
    out=$(docker exec "$SANDBOX" df -B1 -P "$path") || { echo "docker exec $SANDBOX df failed" >&2; return 1; }
    read -r used avail total mount <<< "$(parse_df <<< "$out")"
    [ -n "${total:-}" ] || { echo "unparseable df output for $path" >&2; return 1; }
    probe_json "$name" "$label" "$used" "$total" "$(pct_of "$used" $((used + avail)))" "$path" "$@"
}

probe_video_work() {   # <name> <label> <path> [kv...]
    local name=$1 label=$2 path=$3 out; shift 3
    # A missing dir is a normal "0" (it exists only while a video is being
    # processed); a stopped container is an error, not a 0.
    out=$(docker exec "$HAYSTACK" sh -c 'if [ -d "$1" ]; then du -sb "$1"; else echo "0 $1"; fi' \
            sh "$path") || { echo "docker exec $HAYSTACK du failed" >&2; return 1; }
    probe_json "$name" "$label" "${out%%[[:space:]]*}" "" "" "$HAYSTACK:$path" "$@"
}

probe_docker_df() {   # <name> <label> [kv...] — kv applies to the build cache
    local name=$1 label=$2 out type size recl got=0; shift 2
    out=$(docker system df --format '{{.Type}}\t{{.Size}}\t{{.Reclaimable}}') \
        || { echo "docker system df failed" >&2; return 1; }
    while read -r type size recl; do
        case $type in
            images)
                probe_json "${name}_images" "$label images" "$size" "" "" "$DOCKER_ROOT" \
                    "reclaimable_bytes=$recl"; got=$((got + 1)) ;;
            build_cache)
                probe_json "${name}_build_cache" "$label build cache" "$size" "" "" "$DOCKER_ROOT" \
                    "reclaimable_bytes=$recl" "$@"; got=$((got + 1)) ;;
        esac
    done < <(parse_system_df <<< "$out")
    [ "$got" = 2 ] || { echo "docker system df: expected Images and Build Cache rows" >&2; return 1; }
}

probe_docker_logs() {   # <name> <label> [kv...] — kv applies to the largest file
    local name=$1 label=$2 total big path cid cname; shift 2
    # Rotated files (-json.log.1 ...) count too, if rotation is ever configured.
    read -r total big path < <(find "$DOCKER_ROOT/containers" -type f -name '*-json.log*' \
        -printf '%s %p\n' | sum_and_largest) || true
    [ -n "${total:-}" ] || { echo "cannot list $DOCKER_ROOT/containers" >&2; return 1; }
    probe_json "${name}_total" "$label (all containers)" "$total" "" "" "$DOCKER_ROOT/containers"
    cid=$(basename "$(dirname "${path:-x}")")
    cname=$(docker inspect --format '{{.Name}}' "$cid" 2>/dev/null) || cname=$cid
    probe_json "${name}_largest" "$label (largest: ${cname#/})" "$big" "" "" "${path:-}" "$@"
}

PG_USER=""; PG_DB=""
pg_env() {
    [ -n "$PG_USER" ] && return 0
    PG_USER=$(docker exec "$POSTGRES" printenv POSTGRES_USER) || return 1
    PG_DB=$(docker exec "$POSTGRES" printenv POSTGRES_DB) || return 1
}

probe_postgres() {   # <name> <label> [kv...]
    local name=$1 label=$2 size; shift 2
    pg_env || { echo "cannot read POSTGRES_USER/POSTGRES_DB from $POSTGRES" >&2; return 1; }
    size=$(docker exec "$POSTGRES" psql -X -U "$PG_USER" -d "$PG_DB" -tAc \
        "SELECT pg_database_size(current_database())") || { echo "psql failed" >&2; return 1; }
    probe_json "$name" "$label" "$size" "" "" "$PG_DB" "$@"
}

probe_du() {   # <name> <label> <path> [kv...]
    local name=$1 label=$2 path=$3 out; shift 3
    out=$(du -sb "$path") || { echo "du failed for $path" >&2; return 1; }
    probe_json "$name" "$label" "${out%%[[:space:]]*}" "" "" "$path" "$@"
}

mem_exec_docker() {   # <container>
    docker exec "$1" cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max         /sys/fs/cgroup/memory.stat
}

probe_container_mem() {   # <name> <label> <container> [kv...]
    local name=$1 label=$2 ctr=$3 out used limit; shift 3
    out=$("${MEM_EXEC[@]}" "$ctr" 2>&1)         || { echo "cannot read cgroup memory of $ctr: $(printf '%s' "$out" | tail -n 1)" >&2; return 1; }
    read -r used limit <<< "$(parse_cgroup_mem <<< "$out")"
    [ -n "${used:-}" ] || { echo "unparseable cgroup memory output for $ctr" >&2; return 1; }
    if [ "$limit" = max ]; then
        echo "$ctr has no memory limit (cgroup memory.max = max): nothing to grade against, and the limit is part of the sandbox boundary (D07)" >&2
        return 1
    fi
    probe_json "$name" "$label" "$used" "$limit" "$(pct_of "$used" "$limit")" "$ctr" "$@"
}

probe_provider() {   # <name> <label> <provider> [kv...]
    local name=$1 label=$2 provider=$3 out line cat status detail latency; shift 3
    # provider_health.py exits 1 whenever the provider is not "ok"; the JSON
    # line is what says why, so its exit status is deliberately not used here.
    out=$("${PROVIDER_EXEC[@]}" "$provider" 2>&1)
    line=$(printf '%s\n' "$out" | grep -m 1 '"category": ' || true)
    IFS='|' read -r cat status detail <<< "$(parse_provider_line "$line")"
    if [ -z "$cat" ]; then
        echo "provider_health.py gave no result for $provider: $(printf '%s' "$out" | tail -n 1 | cut -c1-200)" >&2
        return 1
    fi
    latency=$(printf '%s' "$line" | sed -n 's/.*"latency_ms": \([0-9]*\).*/\1/p')
    case $cat in
        billing)
            echo "$provider OUT OF CREDIT (HTTP $status): $detail -- top up the account; chats and ingestion using it fail until then" >&2
            return 1 ;;
        auth)
            echo "$provider KEY REJECTED (HTTP $status): $detail -- the API key is invalid, revoked or lacks permission" >&2
            return 1 ;;
        model_unavailable)
            echo "$provider MODEL UNAVAILABLE (HTTP $status): $detail -- the configured model was retired or renamed" >&2
            return 1 ;;
        not_configured)
            echo "$provider NOT CONFIGURED: $detail" >&2
            return 1 ;;
        ok) ;;
        *)
            # Transient (rate_limited, network, server_error) or unknown: on the
            # row and in the journal, but no alert -- the next hour re-checks.
            log "WARNING: provider $provider: $cat (HTTP $status): $detail" ;;
    esac
    probe_json "$name" "$label" "" "" "" "$provider" "category=$cat" "http_status=$status"         "latency_ms=$latency" "$@"
}

collect() {   # prints a JSON array of probes; failures become error probes
    local entry name label func errfile out err sep="" line
    local -a parts
    errfile=$(mktemp)
    printf '['
    for entry in "${PROBES[@]}"; do
        IFS='|' read -r -a parts <<< "$entry"
        name=${parts[0]}; label=${parts[1]}; func=${parts[2]}
        out=$("$func" "$name" "$label" "${parts[@]:3}" 2>"$errfile"); local rc=$?
        while IFS= read -r line; do
            [ -n "$line" ] || continue
            printf '%s%s' "$sep" "$line"; sep=","
        done <<< "$out"
        if [ $rc -ne 0 ]; then
            err=$(tr '\n' ' ' < "$errfile"); err=${err% }
            fail "probe $name: ${err:-exit $rc}"
            printf '%s%s' "$sep" "$(probe_json "$name" "$label" "" "" "" "" "error=${err:-exit $rc}")"
            sep=","
        fi
    done
    printf ']'
    rm -f "$errfile"
}

# ── cleanup ───────────────────────────────────────────────

cleanup() {   # <dry_run 0|1> -> cleanup JSON object
    local dry=$1 path out files bytes total_files=0 total_bytes=0 sep="" paths="" errors=""
    # -mmin, not -mtime +7: find rounds -mtime down to whole days, so +7 means
    # "at least 8 days". -not -path '*/.*' keeps dotfiles and everything under a
    # dot-dir (/home/user/.local holds pip installs). -prune would be the usual
    # way, but -delete implies -depth, under which -prune does nothing.
    local -a act=(-delete)
    [ "$dry" = 1 ] && act=()
    for path in $SANDBOX_CLEAN_PATHS; do
        if out=$("${SANDBOX_EXEC[@]}" find "$path" -xdev -mindepth 1 -type f \
                -mmin +$((MAX_AGE_DAYS * 24 * 60)) -not -path '*/.*' -printf '%s\n' "${act[@]}" 2>&1); then
            files=$(printf '%s' "$out" | grep -c '^[0-9]' || true)
            bytes=$(printf '%s\n' "$out" | awk '/^[0-9]+$/ { s += $1 } END { printf "%.0f", s }')
        else
            files=0; bytes=0
            errors+="${errors:+; }$path: $(printf '%s' "$out" | tail -n 1)"
            fail "cleanup $path: $(printf '%s' "$out" | tail -n 1)"
        fi
        total_files=$((total_files + files)); total_bytes=$((total_bytes + bytes))
        paths+="$sep{\"path\":$(json_str "$path"),\"files\":$files,\"bytes\":$bytes}"; sep=","
    done
    printf '{"max_age_days":%d,"dry_run":%s,"files_deleted":%d,"bytes_deleted":%d,"paths":[%s]' \
        "$MAX_AGE_DAYS" "$([ "$dry" = 1 ] && echo true || echo false)" \
        "$total_files" "$total_bytes" "$paths"
    [ -n "$errors" ] && printf ',"error":%s' "$(json_str "$errors")"
    printf '}'
}

# ── report ────────────────────────────────────────────────

write_report() {   # <probes_json> <cleanup_json>
    pg_env || { fail "cannot read POSTGRES_USER/POSTGRES_DB from $POSTGRES"; return 1; }
    # :'probes' is psql's own literal quoting, so the JSON needs no escaping
    # here; it arrives as argv (no shell inside the container) and is bound
    # as a quoted literal.
    docker exec -i "$POSTGRES" psql -X -q -v ON_ERROR_STOP=1 -U "$PG_USER" -d "$PG_DB" \
        -v probes="$1" -v cleanup="$2" -v retain="$RETAIN_DAYS days" <<'SQL'
CREATE TABLE IF NOT EXISTS ops_storage_reports (
    id          serial PRIMARY KEY,
    reported_at timestamptz NOT NULL DEFAULT now(),
    probes      jsonb NOT NULL,
    cleanup     jsonb
);
INSERT INTO ops_storage_reports (probes, cleanup) VALUES (:'probes'::jsonb, :'cleanup'::jsonb);
DELETE FROM ops_storage_reports WHERE reported_at < now() - :'retain'::interval;
SQL
}

# ── self-test ─────────────────────────────────────────────

self_test() {
    QUIET_LOG=1
    local n=0 got
    # Saved before the collect() checks below swap PROBES for fixtures.
    local -a real_probes=("${PROBES[@]}")
    check() {   # check <name> <expected> <actual>
        if [ "$2" != "$3" ]; then
            printf 'FAIL %s\n  expected: %s\n  got:      %s\n' "$1" "$2" "$3" >&2
            exit 1
        fi
        n=$((n + 1)); printf 'ok %s\n' "$1"
    }

    # Fixtures are the VPS's real output, 2026-10-21.
    got=$(parse_df <<'EOF'
Filesystem         1-blocks        Used    Available Capacity Mounted on
/dev/sda1      414921494528 45647187968 369257529344      12% /
EOF
)
    check parse_df "45647187968 369257529344 414921494528 /" "$got"

    got=$(parse_df <<'EOF'
Filesystem       1-blocks    Used  Available Capacity Mounted on
tmpfs          2147483648 1171456 2146312192       1% /home/user
EOF
)
    check parse_df_tmpfs "1171456 2146312192 2147483648 /home/user" "$got"

    got=$(parse_dfi <<'EOF'
Filesystem       Inodes  IUsed    IFree IUse% Mounted on
/dev/sda1      52297728 443895 51853833    1% /
EOF
)
    check parse_dfi "443895 52297728 /" "$got"

    check pct_of_df_capacity "11.0" "$(pct_of 45647187968 $((45647187968 + 369257529344)))"
    check pct_of_full "100.0" "$(pct_of 5 5)"
    check pct_of_zero_total "" "$(pct_of 0 0)"

    check docker_size_gb "11680000000" "$(docker_size_to_bytes 11.68GB)"
    check docker_size_kb "4096" "$(docker_size_to_bytes 4.096kB)"
    check docker_size_zero "0" "$(docker_size_to_bytes 0B)"
    check docker_size_garbage "" "$(docker_size_to_bytes N/A)"

    got=$(printf 'Images\t17.01GB\t0B (0%%)\nContainers\t222.8MB\t4.096kB (0%%)\nLocal Volumes\t0B\t0B\nBuild Cache\t11.68GB\t1.453GB\n' \
        | parse_system_df | tr '\n' '|')
    check parse_system_df "images 17010000000 0|containers 222800000 4096|local_volumes 0 0|build_cache 11680000000 1453000000|" "$got"

    got=$(printf '6051523 /var/lib/docker/containers/93c9/93c9-json.log\n44312562 /var/lib/docker/containers/b26f/b26f-json.log\n10 /var/lib/docker/containers/aa/aa-json.log.1\n' \
        | sum_and_largest)
    check sum_and_largest "50364095 44312562 /var/lib/docker/containers/b26f/b26f-json.log" "$got"

    check mount_slug_root "root" "$(mount_slug /)"
    check mount_slug_nested "var_lib_docker" "$(mount_slug /var/lib/docker)"

    check json_str_escapes '"a \"q\" b\\c d"' "$(json_str "a \"q\" b\\c
d")"
    check json_val_number "75" "$(json_val 75)"
    check json_val_decimal "11.0" "$(json_val 11.0)"
    check json_val_null "null" "$(json_val "")"
    check json_val_string '"inodes"' "$(json_val inodes)"

    got=$(probe_json postgres "Postgres database" 2000 "" "" openwebui alert_bytes=5000000000)
    check probe_json_capacityless '{"name":"postgres","label":"Postgres database","used_bytes":2000,"total_bytes":null,"pct":null,"path":"openwebui","alert_bytes":5000000000}' "$got"

    got=$(probe_json sandbox_tmp "Sandbox /tmp" "" "" "" "" "error=docker exec failed: \"x\"")
    check probe_json_error '{"name":"sandbox_tmp","label":"Sandbox /tmp","used_bytes":null,"total_bytes":null,"pct":null,"path":"","error":"docker exec failed: \"x\""}' "$got"

    # collect(): a failing probe becomes an error probe, the others survive.
    ok_probe() { probe_json "$1" "$2" 1 2 50.0 /x; }
    bad_probe() { echo "container not running" >&2; return 1; }
    PROBES=("a|A|ok_probe" "b|B|bad_probe")
    got=$(collect 2>/dev/null)
    check collect_keeps_the_good_and_records_the_bad \
        '[{"name":"a","label":"A","used_bytes":1,"total_bytes":2,"pct":50.0,"path":"/x"},{"name":"b","label":"B","used_bytes":null,"total_bytes":null,"pct":null,"path":"","error":"container not running"}]' \
        "$got"
    collect >/dev/null 2>&1
    check collect_marks_the_run_failed "1" "$FAILED"

    # cleanup(): the real find expression, run locally instead of in the sandbox.
    local sand; sand=$(mktemp -d)
    mkdir -p "$sand/home/outputs" "$sand/home/.local/lib" "$sand/tmp"
    printf '12345' > "$sand/home/outputs/old.csv"
    printf 'abc' > "$sand/home/new.csv"
    printf 'x' > "$sand/home/.local/lib/old_pip.py"
    printf 'y' > "$sand/home/.bashrc"
    printf '1234567' > "$sand/tmp/old.png"
    printf 'z' > "$sand/tmp/six_days.txt"
    touch -d '10 days ago' "$sand/home/outputs/old.csv" "$sand/home/.local/lib/old_pip.py" \
        "$sand/home/.bashrc" "$sand/tmp/old.png"
    # 7 days and 1 hour: `-mtime +7` would keep it (rounds to 7 whole days);
    # -mmin deletes it. 6 days stays.
    touch -d '7 days ago 1 hour ago' "$sand/home/new.csv"
    touch -d '6 days ago' "$sand/tmp/six_days.txt"
    local saved_paths=$SANDBOX_CLEAN_PATHS
    SANDBOX_EXEC=() SANDBOX_CLEAN_PATHS="$sand/home $sand/tmp"
    got=$(cleanup 1)
    check cleanup_dry_run_counts "{\"max_age_days\":7,\"dry_run\":true,\"files_deleted\":3,\"bytes_deleted\":15,\"paths\":[{\"path\":\"$sand/home\",\"files\":2,\"bytes\":8},{\"path\":\"$sand/tmp\",\"files\":1,\"bytes\":7}]}" "$got"
    check cleanup_dry_run_deletes_nothing "6" "$(find "$sand" -type f | wc -l | tr -d ' ')"
    cleanup 0 > /dev/null
    check cleanup_keeps_dotfiles_and_recent \
        "$sand/home/.bashrc $sand/home/.local/lib/old_pip.py $sand/tmp/six_days.txt" \
        "$(find "$sand" -type f | sort | tr '\n' ' ' | sed 's/ $//')"
    SANDBOX_EXEC=(docker exec "$SANDBOX") SANDBOX_CLEAN_PATHS=$saved_paths
    rm -rf "$sand"

    # Sandbox memory (O4). Fixture: the VPS's real cgroup v2 files, 2026-10-25
    # (memory.current, memory.max, then memory.stat).
    local cg_real='50737152
12884901888
anon 45985792
file 28672
inactive_file 0
shmem 28672'
    check parse_cgroup_mem "50737152 12884901888" "$(parse_cgroup_mem <<< "$cg_real")"
    # Reclaimable page cache is not memory pressure: docker stats and the OOM
    # killer both discount inactive_file, so the probe does too.
    check parse_cgroup_mem_discounts_inactive_file "600 1000" \
        "$(parse_cgroup_mem <<< $'700\n1000\ninactive_file 100\nanon 600')"
    check parse_cgroup_mem_unlimited "700 max" "$(parse_cgroup_mem <<< $'700\nmax\ninactive_file 0')"
    check parse_cgroup_mem_garbage "" "$(parse_cgroup_mem <<< 'cat: /sys/fs/cgroup/memory.current: No such file')"

    fake_cgroup() { printf '%s\n' "$FAKE_CG"; }
    local saved_mem_exec=("${MEM_EXEC[@]}")
    MEM_EXEC=(fake_cgroup)
    FAKE_CG=$'10307921510\n12884901888\ninactive_file 0'          # 80.0% of 12 GiB
    got=$(probe_container_mem sandbox_mem "Sandbox memory" ${STACK_NAME}-code-sandbox-1 warn_pct=70 alert_pct=80)
    check probe_container_mem_json "{\"name\":\"sandbox_mem\",\"label\":\"Sandbox memory\",\"used_bytes\":10307921510,\"total_bytes\":12884901888,\"pct\":80.0,\"path\":\"${STACK_NAME}-code-sandbox-1\",\"warn_pct\":70,\"alert_pct\":80}" "$got"
    FAKE_CG=$'700\nmax\ninactive_file 0'
    got=$(probe_container_mem sandbox_mem "Sandbox memory" ${STACK_NAME}-code-sandbox-1 2>&1 >/dev/null); local rc=$?
    # No limit = nothing to grade against, and the limit is part of the sandbox's
    # safety boundary (D07): losing it is a finding, not a quiet "ok".
    check probe_container_mem_no_limit_fails "1|no memory limit" "$rc|$(printf '%s' "$got" | grep -o 'no memory limit')"
    MEM_EXEC=(false)
    probe_container_mem sandbox_mem "Sandbox memory" ${STACK_NAME}-code-sandbox-1 >/dev/null 2>&1
    check probe_container_mem_exec_fails "1" "$?"
    MEM_EXEC=("${saved_mem_exec[@]}")

    # Model providers. Fixtures: provider_health.py's real output lines.
    check parse_provider_ok "ok|200|HTTP 200" \
        "$(parse_provider_line '{"provider": "voyage", "category": "ok", "http_status": 200, "latency_ms": 324, "detail": "HTTP 200"}')"
    check parse_provider_billing "billing|400|{\\\"type\\\":\\\"error\\\"} credit balance is too low" \
        "$(parse_provider_line '{"provider": "anthropic", "category": "billing", "http_status": 400, "latency_ms": 90, "detail": "{\"type\":\"error\"} credit balance is too low"}')"
    check parse_provider_garbage "||" "$(parse_provider_line 'Traceback (most recent call last):')"

    fake_provider() { printf '%s\n' "$FAKE_PROVIDER"; return "${FAKE_PROVIDER_RC:-0}"; }
    local saved_prov_exec=("${PROVIDER_EXEC[@]}")
    PROVIDER_EXEC=(fake_provider)
    FAKE_PROVIDER='{"provider": "anthropic", "category": "ok", "http_status": 200, "latency_ms": 702, "detail": "HTTP 200"}'
    got=$(probe_provider provider_anthropic "Model provider: Anthropic" anthropic)
    check probe_provider_ok '{"name":"provider_anthropic","label":"Model provider: Anthropic","used_bytes":null,"total_bytes":null,"pct":null,"path":"anthropic","category":"ok","http_status":200,"latency_ms":702}' "$got"
    # Credit/key problems do not fix themselves: an error probe -> red row,
    # banner, failed unit — the 2026-10-22 Gemini credit outage was found by
    # accident because nothing did this.
    FAKE_PROVIDER='{"provider": "gemini:gemini-3.6-flash", "category": "billing", "http_status": 402, "latency_ms": 80, "detail": "Your prepayment credits are depleted"}'
    FAKE_PROVIDER_RC=1
    got=$(probe_provider provider_gemini "Model provider: Google Gemini" gemini 2>&1 >/dev/null); rc=$?
    check probe_provider_billing_fails "1|OUT OF CREDIT" "$rc|$(printf '%s' "$got" | grep -o 'OUT OF CREDIT')"
    FAKE_PROVIDER='{"provider": "voyage", "category": "auth", "http_status": 401, "latency_ms": 80, "detail": "invalid api key"}'
    got=$(probe_provider provider_voyage "Model provider: Voyage" voyage 2>&1 >/dev/null); rc=$?
    check probe_provider_auth_fails "1|KEY REJECTED" "$rc|$(printf '%s' "$got" | grep -o 'KEY REJECTED')"
    # A blip (429, timeout, 5xx) is recorded on the row but is not an alert: an
    # hourly red banner for a rate limit would teach everyone to ignore it.
    FAKE_PROVIDER='{"provider": "anthropic", "category": "rate_limited", "http_status": 429, "latency_ms": 80, "detail": "rate limit"}'
    got=$(probe_provider provider_anthropic "Model provider: Anthropic" anthropic 2>/dev/null); rc=$?
    check probe_provider_transient_is_not_an_alert "0|\"category\":\"rate_limited\"" "$rc|$(printf '%s' "$got" | grep -o '"category":"rate_limited"')"
    FAKE_PROVIDER='Traceback (most recent call last):'
    probe_provider provider_anthropic "Model provider: Anthropic" anthropic >/dev/null 2>&1
    check probe_provider_no_result_fails "1" "$?"
    FAKE_PROVIDER_RC=0
    PROVIDER_EXEC=("${saved_prov_exec[@]}")

    # The probes are registered (a function nobody calls protects nothing).
    check probes_registered "sandbox_mem provider_gemini provider_voyage provider_anthropic" \
        "$(printf '%s\n' "${real_probes[@]}" | cut -d'|' -f1 | grep -E '^(sandbox_mem|provider_)' | tr '\n' ' ' | sed 's/ $//')"
    check sandbox_mem_alerts_at_80 "1" \
        "$(printf '%s\n' "${real_probes[@]}" | grep -c '^sandbox_mem|.*|alert_pct=80$')"

    printf '\n%d storage monitor self-checks passed\n' "$n"
}

# ── main ──────────────────────────────────────────────────

main() {
    local dry=0 probes cleanup_json
    # A dry run leaves no trace on the host: no deletes, no row, no journal lines.
    [ "${1:-}" = "--dry-run" ] && { dry=1; QUIET_LOG=1; }

    # Through a file, not $(...): a command substitution is a subshell, and
    # the FAILED=1 that fail() sets inside it would vanish — the run would
    # exit 0 with an error probe in the report and nothing in systemctl.
    # Global, not local: the EXIT trap runs after main returns, under set -u.
    TMPFILE=$(mktemp) || { fail "mktemp failed"; exit 1; }
    trap 'rm -f "$TMPFILE"' EXIT
    # Cleanup first, so the probes measure the tmpfs after it.
    cleanup "$dry" > "$TMPFILE"; cleanup_json=$(cat "$TMPFILE")
    collect > "$TMPFILE"; probes=$(cat "$TMPFILE")

    if [ "$dry" = 1 ]; then
        printf '{"probes":%s,"cleanup":%s}\n' "$probes" "$cleanup_json"
    elif write_report "$probes" "$cleanup_json"; then
        log "report written: $(printf '%s' "$probes" | grep -o '"name":' | wc -l) probes;" \
            "cleanup $cleanup_json"
    else
        fail "could not write ops_storage_reports to $POSTGRES"
    fi

    if [ "$FAILED" != 0 ]; then
        fail "run finished with errors (probes that worked were still written)"
        exit 1
    fi
}

case "${1:-}" in
    --self-test) self_test ;;
    --dry-run|"") main "$@" ;;
    *) echo "usage: $0 [--dry-run|--self-test]" >&2; exit 2 ;;
esac
