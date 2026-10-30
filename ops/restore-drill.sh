#!/usr/bin/env bash
# restore-drill.sh — the monthly restore drill (README.md,
# Backups), automated. Proves a recovery point in the restic repository can
# actually be restored: "an untested backup is an assumption".
#
# Everything happens in THROWAWAY places, and is removed again on success,
# failure or Ctrl-C:
#   - a private scratch dir  /var/tmp/gmac-restore-drill.XXXXXX (chmod 700)
#   - a scratch Postgres     container ${STACK_NAME}-drill-pg-<id>, --network none
#                            (unreachable from anything), trust auth, its own
#                            anonymous volume
#   - a scratch Qdrant       container ${STACK_NAME}-drill-qd-<id>, published on
#                            127.0.0.1 only (random port), its own volume
# It never touches gmac-network, /data, /opt/gmac, the ${STACK_NAME}-postgres /
# qdrant containers, or production collections, and writes nothing to the
# production database. It READS the restic repository (restic restore is
# read-only on the repository) — it never forgets, prunes or writes to it.
#
# Steps (any failure -> exit 1, cleanup still runs):
#   1. preflight  root, tools, secrets (same sources as backup.sh),
#                 repository reachable, the backup lock (never overlaps a backup)
#   2. snapshot   --snapshot <id>, or the newest gmac-backup snapshot
#   3. space      restic stats --mode restore-size vs free space in /var/tmp
#   4. restore    restic restore <id> --target <scratch>/restore
#   5. manifest   sha256sum -c MANIFEST.sha256 on the restored artifacts
#   6. postgres   pg_restore the dump into the scratch Postgres; required
#                 tables exist and are non-empty
#   7. qdrant     upload every required collection's snapshot into the scratch
#                 Qdrant (Qdrant verifies the manifest sha256 itself); point
#                 counts must equal MANIFEST.json
#   8. files      uploads + Haystack state restored with the inventoried files
#   9. config     every config file (.env too, hash only) matches MANIFEST.json
#
# Usage (on the VPS, as root; needs python3, docker, restic, rclone):
#   ops/restore-drill.sh                    # newest snapshot
#   ops/restore-drill.sh --snapshot 1a2b3c4d
#   ops/restore-drill.sh --self-test        # offline checks, no docker
#
# Needs free space in /var/tmp for the full restore (~4 GB on 2026-10-23, plus
# the scratch databases). It is a heavy read from Google Drive: run it at a
# quiet hour, not next to the 03:00 backup (the shared lock enforces that).
#
# Secrets: RESTIC_PASSWORD is read the same way backup.sh reads it and is
# never echoed. The restored .env lives only in the chmod-700 scratch dir and is
# compared by hash, never printed, then deleted with the rest.

set -uo pipefail

TAG=gmac-restore-drill
INSTALL_DIR=${INSTALL_DIR:-/opt/gmac}
STACK_NAME=${STACK_NAME:-gmac}
ENV_FILE=${ENV_FILE:-$INSTALL_DIR/.env}
QDRANT_HELPER=${QDRANT_HELPER:-$INSTALL_DIR/ops/qdrant_backup.py}
VERIFY=${VERIFY:-$INSTALL_DIR/ops/restore_drill_verify.py}
LOCK_FILE=${LOCK_FILE:-/run/lock/gmac-backup.lock}     # the backup's own lock
WORK_PARENT=${WORK_PARENT:-/var/tmp}
# Same images production runs (compose.yml) — a restore that only works on a
# different Postgres/Qdrant version proves nothing about recovery.
PG_IMAGE=${PG_IMAGE:-postgres:16-alpine}
QDRANT_IMAGE=${QDRANT_IMAGE:-qdrant/qdrant:v1.19.0}
PG_DB_DUMP=${PG_DB_DUMP:-openwebui}
# Tables that must exist and hold rows after the restore: accounts, chats and
# the Admin-pasted filters/tools are the irreplaceable part of the database.
REQUIRED_TABLES=${REQUIRED_TABLES:-"user chat function"}
COLLECTIONS=${COLLECTIONS:-"documents code_docs visual_frames ocr_frames od_frames transcript_segments"}
SPACE_MARGIN_GB=${SPACE_MARGIN_GB:-3}       # scratch DBs + overhead beyond the restore itself
READY_TIMEOUT_S=${READY_TIMEOUT_S:-120}

FAILED=0
FIRST_ERROR=""
WORK=""
CONTAINERS=()          # ONLY containers this run created — cleanup removes nothing else
RUN_ID="$$-$(date +%s)"

log() { echo "$(date -u +%FT%TZ) $*"; logger -t "$TAG" -- "$*" 2>/dev/null || true; }

fail() {
    FAILED=1
    [ -n "$FIRST_ERROR" ] || FIRST_ERROR="$*"
    echo "$(date -u +%FT%TZ) ERROR: $*" >&2
    logger -p err -t "$TAG" -- "ERROR: $*" 2>/dev/null || true
}

# The dashboard's Google Drive sign-in, if one was made (same as the backup).
# shellcheck source=backup_drive.sh
. "$(dirname "${BASH_SOURCE[0]}")/backup_drive.sh"

env_get() {   # <KEY> -> value or empty; reads only that key (same as backup.sh)
    [ -f "$ENV_FILE" ] || return 0
    sed -n "s/^${1}=//p" "$ENV_FILE" | tail -n 1 | tr -d '\r'
}

load_secrets() {
    # RESTIC_REPOSITORY in .env is the backup FOLDER; it holds one repository
    # per subfolder (daily/weekly/monthly). DRILL_REPO picks which one
    # (daily unless set), appended once.
    local base="${RESTIC_BASE:-${RESTIC_REPOSITORY:-$(env_get RESTIC_REPOSITORY)}}"
    RESTIC_BASE="$base"
    case "${DRILL_REPO:-daily}" in
        daily|weekly|monthly) RESTIC_REPOSITORY="${base%/}/${DRILL_REPO:-daily}" ;;
        *) echo "DRILL_REPO must be daily, weekly or monthly" >&2; return 1 ;;
    esac
    [ -n "${RESTIC_PASSWORD:-}" ] || RESTIC_PASSWORD=$(env_get RESTIC_PASSWORD)
    export RESTIC_REPOSITORY RESTIC_PASSWORD
}

# ── safety guards (self-tested) ─────────────────────────────

is_scratch_dir() {   # <path> -> 0 only for this script's own scratch dirs
    case "$1" in
        "$WORK_PARENT"/gmac-restore-drill.?*) [[ "$1" != *..* ]] ;;
        *) return 1 ;;
    esac
}

is_drill_container() {   # <name> -> 0 only for names this script generates
    case "$1" in
        ${STACK_NAME}-drill-pg-*|${STACK_NAME}-drill-qd-*) return 0 ;;
        *) return 1 ;;
    esac
}

cleanup() {
    local c
    drive_conf_close    # the generated rclone config holds OAuth tokens
    for c in "${CONTAINERS[@]}"; do
        # Belt and braces: even a tracked name is removed only if it carries the
        # drill prefix — this function can never remove a production container.
        is_drill_container "$c" || { log "WARNING: refusing to remove non-drill container $c"; continue; }
        docker rm -f -v "$c" >/dev/null 2>&1 && log "cleanup: removed container $c"
    done
    CONTAINERS=()
    if [ -n "$WORK" ]; then
        if is_scratch_dir "$WORK"; then
            rm -rf -- "$WORK" && log "cleanup: removed $WORK"
        else
            log "WARNING: refusing to delete $WORK — not a drill scratch dir"
        fi
        WORK=""
    fi
}

manifest_sha() {   # <MANIFEST.sha256> <relative-path> -> sha256 or empty
    awk -v f="$2" '{ p = $2; sub(/^\*/, "", p); if (p == f) { print $1; exit } }' "$1"
}

parse_snapshot_id() {   # restic snapshots --json on stdin -> newest short_id
    tr -d '\n' | sed 's/},{/}\n{/g' | sed -n 's/.*"short_id": *"\([0-9a-f]*\)".*/\1/p' | tail -n 1
}

wait_for() {   # <description> <command...> — poll until the command succeeds
    local what="$1" t=0; shift
    until "$@" >/dev/null 2>&1; do
        t=$((t + 2))
        [ "$t" -le "$READY_TIMEOUT_S" ] || { fail "$what not ready after ${READY_TIMEOUT_S}s"; return 1; }
        sleep 2
    done
}

# ── steps ───────────────────────────────────────────────────

preflight() {
    [ "$(id -u)" = 0 ] || { fail "run as root (restic secrets, docker, /run/lock)"; return 1; }
    local cmd
    for cmd in docker restic rclone python3 sha256sum flock df awk; do
        command -v "$cmd" >/dev/null || { fail "required tool missing: $cmd"; return 1; }
    done
    for f in "$QDRANT_HELPER" "$VERIFY"; do
        [ -f "$f" ] || { fail "missing $f"; return 1; }
    done
    load_secrets
    [ -n "$RESTIC_REPOSITORY" ] || { fail "RESTIC_REPOSITORY is not set ($ENV_FILE or environment)"; return 1; }
    [ -n "$RESTIC_PASSWORD" ] || { fail "RESTIC_PASSWORD is not set ($ENV_FILE or environment)"; return 1; }
    drive_conf_setup    # the same drive sign-in the backup uses (none in the panel -> unchanged)
    restic cat config >/dev/null 2>&1 \
        || { fail "restic repository not reachable or wrong password (restic cat config)"; return 1; }
}

pick_snapshot() {   # sets SNAPSHOT
    if [ -n "${SNAPSHOT:-}" ]; then
        restic snapshots "$SNAPSHOT" --json 2>/dev/null | grep -q "\"short_id\"" \
            || { fail "snapshot $SNAPSHOT not found in the repository"; return 1; }
    else
        SNAPSHOT=$(restic snapshots --tag gmac-backup --json 2>/dev/null | parse_snapshot_id)
        [ -n "$SNAPSHOT" ] || { fail "no gmac-backup snapshot in the repository yet"; return 1; }
    fi
    log "snapshot: $SNAPSHOT"
}

check_space() {
    local need avail_kb gb=$((1024 * 1024 * 1024))
    need=$(restic stats "$SNAPSHOT" --mode restore-size --json 2>/dev/null \
           | sed -n 's/.*"total_size": *\([0-9]*\).*/\1/p' | head -n 1)
    [ -n "$need" ] || { fail "could not read the snapshot's restore size (restic stats)"; return 1; }
    avail_kb=$(df -Pk "$WORK_PARENT" | awk 'NR==2 {print $4}')
    if [ $((avail_kb * 1024)) -lt $((need + SPACE_MARGIN_GB * gb)) ]; then
        fail "not enough space in $WORK_PARENT: $((avail_kb / 1024)) MB free, restore needs $((need / 1024 / 1024)) MB + ${SPACE_MARGIN_GB} GB margin"
        return 1
    fi
    log "space: restore is $((need / 1024 / 1024)) MB, $((avail_kb / 1024)) MB free"
}

do_restore() {
    WORK=$(mktemp -d "$WORK_PARENT/gmac-restore-drill.XXXXXX") || { fail "cannot create scratch dir"; return 1; }
    chmod 700 "$WORK"
    log "restore: $SNAPSHOT -> $WORK/restore (read-only on the repository)"
    restic restore "$SNAPSHOT" --target "$WORK/restore" >/dev/null \
        || { fail "restic restore failed"; return 1; }
    STAGE="$WORK/restore/var/tmp/gmac-backup"
    MANIFEST="$STAGE/MANIFEST.json"
    [ -f "$MANIFEST" ] && [ -f "$STAGE/MANIFEST.sha256" ] \
        || { fail "restored snapshot has no MANIFEST.json / MANIFEST.sha256"; return 1; }
}

verify_manifest() {
    ( cd "$STAGE" && sha256sum -c --quiet MANIFEST.sha256 ) \
        || { fail "restored artifacts do not match MANIFEST.sha256"; return 1; }
    log "manifest: every restored artifact matches its checksum"
}

drill_postgres() {
    local name="${STACK_NAME}-drill-pg-$RUN_ID" dump="$STAGE/postgres/$PG_DB_DUMP.dump" t rows
    [ -s "$dump" ] || { fail "postgres: no dump in the snapshot"; return 1; }
    CONTAINERS+=("$name")
    # --network none: nothing can reach it, so trust auth needs no password.
    docker run -d --name "$name" --network none \
        -e POSTGRES_HOST_AUTH_METHOD=trust -e POSTGRES_DB=drill \
        "$PG_IMAGE" >/dev/null || { fail "postgres: could not start scratch container"; return 1; }
    wait_for "scratch postgres" docker exec "$name" pg_isready -U postgres -d drill || return 1
    log "postgres: restoring the dump into scratch container $name"
    # --no-owner/--no-acl: the production role does not exist in the scratch
    # server; ownership is not what this drill is proving.
    docker exec -i "$name" pg_restore -U postgres -d drill --no-owner --no-acl --exit-on-error \
        < "$dump" || { fail "postgres: pg_restore into the scratch database failed"; return 1; }
    for t in $REQUIRED_TABLES; do
        rows=$(docker exec "$name" psql -X -tA -U postgres -d drill \
               -c "SELECT count(*) FROM public.\"$t\"" 2>/dev/null | tail -n 1)
        if [[ "$rows" =~ ^[0-9]+$ ]] && [ "$rows" -gt 0 ]; then
            log "postgres: table $t restored ($rows rows)"
        else
            fail "postgres: table $t missing or empty after restore (${rows:-no result})"
        fi
    done
}

drill_qdrant() {
    local name="${STACK_NAME}-drill-qd-$RUN_ID" port url c snap sha list
    CONTAINERS+=("$name")
    # 127.0.0.1 only: the scratch Qdrant is reachable from this host, nowhere else.
    docker run -d --name "$name" -p 127.0.0.1::6333 "$QDRANT_IMAGE" >/dev/null \
        || { fail "qdrant: could not start scratch container"; return 1; }
    port=$(docker port "$name" 6333/tcp | sed -n 's/^127\.0\.0\.1:\([0-9]*\)$/\1/p' | head -n 1)
    [ -n "$port" ] || { fail "qdrant: scratch container has no loopback port"; return 1; }
    url="http://127.0.0.1:$port"
    wait_for "scratch qdrant" python3 -c "import urllib.request; urllib.request.urlopen('$url/readyz', timeout=5)" || return 1
    for c in $COLLECTIONS; do
        snap="$STAGE/qdrant/$c.snapshot"
        [ -s "$snap" ] || { fail "qdrant: no snapshot for $c in the backup"; continue; }
        sha=$(manifest_sha "$STAGE/MANIFEST.sha256" "qdrant/$c.snapshot")
        log "qdrant: restoring $c into the scratch Qdrant"
        QDRANT_URL="$url" python3 "$QDRANT_HELPER" restore "$c" "$snap" "$sha" >/dev/null \
            || fail "qdrant: restore of $c failed (Qdrant rejected the snapshot or its checksum)"
    done
    list="$WORK/qdrant-list.json"
    QDRANT_URL="$url" python3 "$QDRANT_HELPER" list > "$list" \
        || { fail "qdrant: could not list the scratch collections"; return 1; }
    # shellcheck disable=SC2086
    python3 "$VERIFY" points "$MANIFEST" "$list" $COLLECTIONS || fail "qdrant: point counts differ from the backup"
}

drill_files_and_config() {
    python3 "$VERIFY" files "$MANIFEST" "$WORK/restore" || fail "files: restored uploads/state differ from the backup"
    python3 "$VERIFY" config "$MANIFEST" "$WORK/restore" || fail "config: restored config differs from the backup"
}

# ── self-test (offline: no docker, no restic) ───────────────

self_test() {
    local n=0
    check() {
        if [ "$2" != "$3" ]; then printf 'FAIL %s\n  expected: %s\n  got:      %s\n' "$1" "$2" "$3" >&2; exit 1; fi
        n=$((n + 1)); printf 'ok %s\n' "$1"
    }
    yes_no() { "$@" && echo yes || echo no; }

    WORK_PARENT=/var/tmp
    check scratch_dir_accepted yes "$(yes_no is_scratch_dir /var/tmp/gmac-restore-drill.Ab12Cd)"
    for p in /data /data/openwebui /opt/gmac / /var/tmp /var/tmp/gmac-backup \
             /var/tmp/gmac-restore-drill. /var/tmp/gmac-restore-drill.x/../../data; do
        check "scratch_dir_refused:$p" no "$(yes_no is_scratch_dir "$p")"
    done
    check drill_container_pg yes "$(yes_no is_drill_container ${STACK_NAME}-drill-pg-1-2)"
    for c in ${STACK_NAME}-postgres ${STACK_NAME}-qdrant-1 ${STACK_NAME}-haystack qdrant postgres ${STACK_NAME}-drill; do
        check "production_container_refused:$c" no "$(yes_no is_drill_container "$c")"
    done

    # cleanup removes exactly the tracked drill containers and the scratch dir —
    # with docker replaced by a recorder, so nothing real is touched.
    local calls="" tmp
    docker() { calls+="$*;"; return 0; }
    tmp=$(mktemp -d); WORK_PARENT="$tmp"; WORK=$(mktemp -d "$tmp/gmac-restore-drill.XXXXXX")
    touch "$WORK/restored.env"
    CONTAINERS=("${STACK_NAME}-drill-pg-x" "${STACK_NAME}-postgres" "${STACK_NAME}-drill-qd-x")
    cleanup >/dev/null 2>&1
    check cleanup_removes_drill_containers_only "rm -f -v ${STACK_NAME}-drill-pg-x;rm -f -v ${STACK_NAME}-drill-qd-x;" "$calls"
    check cleanup_removes_scratch_dir no "$( [ -e "$tmp"/gmac-restore-drill.* ] && echo yes || echo no)"
    check cleanup_resets_state "0|" "${#CONTAINERS[@]}|$WORK"
    local outside; outside=$(mktemp -d); WORK="$outside"; cleanup >/dev/null 2>&1
    check cleanup_refuses_foreign_dir yes "$( [ -d "$outside" ] && echo yes || echo no)"
    rm -rf "$outside" "$tmp"; unset -f docker; WORK_PARENT=/var/tmp

    # Trap path: a failing step still leads to cleanup (the EXIT trap below is
    # what main installs) — simulated in a subshell.
    tmp=$(mktemp -d)
    ( WORK_PARENT="$tmp"; WORK=$(mktemp -d "$tmp/gmac-restore-drill.XXXXXX"); CONTAINERS=()
      trap cleanup EXIT; fail "simulated step failure" 2>/dev/null; exit 1 ) >/dev/null 2>&1
    check failure_still_cleans_up no "$( [ -e "$tmp"/gmac-restore-drill.* ] && echo yes || echo no)"
    rm -rf "$tmp"

    # MANIFEST.sha256 lookup — both sha256sum output forms.
    tmp=$(mktemp)
    printf '%s  postgres/openwebui.dump\n%s *qdrant/od_frames.snapshot\n' "$(printf 'a%.0s' {1..64})" "$(printf 'b%.0s' {1..64})" > "$tmp"
    check manifest_sha_text "$(printf 'a%.0s' {1..64})" "$(manifest_sha "$tmp" postgres/openwebui.dump)"
    check manifest_sha_binary "$(printf 'b%.0s' {1..64})" "$(manifest_sha "$tmp" qdrant/od_frames.snapshot)"
    check manifest_sha_missing "" "$(manifest_sha "$tmp" qdrant/nope.snapshot)"
    rm -f "$tmp"

    # Newest snapshot id from `restic snapshots --json` (oldest first).
    check parse_snapshot_newest "cc33dd44" \
        "$(printf '[{"time":"2026-10-21","short_id":"aa11bb22"},{"time":"2026-10-22","short_id":"cc33dd44"}]' | parse_snapshot_id)"
    check parse_snapshot_none "" "$(printf '[]' | parse_snapshot_id)"

    # The required collections are the backup's own six.
    check collections_match_backup "6" "$(printf '%s\n' $COLLECTIONS | sort -u | wc -l | tr -d ' ')"

    # A failure message never carries the passphrase.
    RESTIC_PASSWORD="super-secret-value-42"
    local msg; msg=$(fail "restic restore failed" 2>&1)
    check fail_leaks_no_secret no "$(case "$msg" in *super-secret*) echo yes;; *) echo no;; esac)"
    FAILED=0; FIRST_ERROR=""; unset RESTIC_PASSWORD

    # The Python comparisons have their own fixtures.
    local py; py=$(command -v python3 || command -v python || true)
    if [ -n "$py" ]; then
        local here; here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        if "$py" "$here/restore_drill_verify.py" selftest >/dev/null 2>&1; then
            n=$((n + 1)); printf 'ok restore_drill_verify_selftest\n'
        else
            printf 'FAIL restore_drill_verify_selftest\n' >&2; exit 1
        fi
    fi
    printf '\n%d restore-drill self-checks passed\n' "$n"
}

# ── main ────────────────────────────────────────────────────

main() {
    SNAPSHOT=""
    case "${1:-}" in
        --self-test) self_test; return 0 ;;
        --snapshot) [ -n "${2:-}" ] || { echo "usage: $0 [--snapshot <id> | --self-test]" >&2; return 2; }
                    SNAPSHOT="$2" ;;
        "") ;;
        *) echo "usage: $0 [--snapshot <id> | --self-test]" >&2; return 2 ;;
    esac

    trap cleanup EXIT
    trap 'fail "interrupted"; exit 1' INT TERM
    exec 9>"$LOCK_FILE"
    flock -n 9 || { fail "a backup (or another drill) holds $LOCK_FILE — try later"; return 1; }

    log "restore drill starting (scratch only — production is never touched)"
    preflight && pick_snapshot && check_space && do_restore && verify_manifest || return 1
    drill_postgres
    drill_qdrant
    drill_files_and_config

    if [ "$FAILED" = 0 ]; then
        log "RESTORE DRILL PASSED: snapshot $SNAPSHOT — database, ${COLLECTIONS// /, }, uploads and config all restored and verified"
        return 0
    fi
    log "RESTORE DRILL FAILED: snapshot ${SNAPSHOT:-?} — first problem: $FIRST_ERROR"
    return 1
}

main "$@"
