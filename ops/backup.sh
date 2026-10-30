#!/usr/bin/env bash
# backup.sh — encrypted backup of everything that cannot be
# rebuilt: the openwebui PostgreSQL database, the six Qdrant collections, the
# uploaded files, the Haystack state and the /opt/gmac config. Destination is
# Google Drive through rclone + restic (README.md), one repository per
# subfolder of the backup folder: daily / weekly / monthly (--target). Replaces
# the retired M3 script haystack-api/scripts/backup_to_gdrive.sh.
#
# One run (any failed step -> the run is "failed" and exits 1; a failed or
# partial run is never recorded as a recovery point):
#   0. lock      flock /run/lock/gmac-backup.lock — one run at a time
#   1. preflight tools, containers, rclone remote, passphrase, repository
#                (restic init on the first run)
#   2. postgres  pg_dump -Fc openwebui, parsed back with pg_restore -l
#   3. qdrant    one snapshot per required collection, created and downloaded
#                over gmac-network via ops/qdrant_backup.py (Qdrant's port is
#                NOT published on the host — the retired script's fatal flaw),
#                each download verified against the size Qdrant reported
#   4. manifest  MANIFEST.json + MANIFEST.sha256: timestamp, git revision,
#                sizes, checksums, Postgres/Qdrant/restic/rclone versions
#   5. verify    sha256sum -c MANIFEST.sha256 before anything is uploaded
#   5b. drive    `rclone about` free space vs what this run may upload:
#                insufficient -> the run fails BEFORE uploading; below
#                DRIVE_WARN_FREE_GB -> "Drive nearly full" warning; unreadable
#                -> warning (or failure with DRIVE_QUOTA_REQUIRED=1). Never
#                deletes a backup to make room.
#   6. restic    ONE encrypted, deduplicated snapshot -> Google Drive
#   7. check     restic check (metadata; + --read-data-subset on Sundays)
#   8. retain    restic forget --tag <target> --keep-last N --prune, N = the
#                folder's kept count (daily 3, weekly 2, monthly 1) - only
#                after steps 6-7 succeeded, and it verifies its own snapshot
#                survived the policy. Deleted packs bypass the Drive trash
#                (--drive-use-trash=false), so the space is freed at once.
#                Milestone snapshots carry no target tag: never removed.
#   9. status    one ops_backup_reports row (running -> success/failed) for
#                GET /admin/backup and the dashboard
#
# Usage (on the VPS, as root):
#   ops/backup.sh                     # the daily folder (the 03:00 /
#                                           # 11:00 / 19:00 UTC timers)
#   ops/backup.sh --target weekly     # the weekly folder (Friday 20:00)
#   ops/backup.sh --target monthly    # the monthly folder (the 1st)
#   ops/backup.sh --dry-run           # build + verify locally, upload
#                                           # nothing, write no status row
#   ops/backup.sh --tag pre-m3        # milestone snapshot, exempt from
#                                           # retention
#   ops/backup.sh --self-test         # offline checks, no docker needed
#
# Install (on the VPS, as root) — rclone auth + restic passphrase setup is
# README.md (Backups):
#   cd /opt/gmac && git pull
#   bash ops/backup.sh --self-test
#   install -m 0644 ops/backup.service ops/backup.timer \
#       ops/backup-weekly.service ops/backup-weekly.timer \
#       ops/backup-monthly.service ops/backup-monthly.timer \
#       /etc/systemd/system/
#   systemctl daemon-reload
#   systemctl enable --now backup.timer backup-weekly.timer \
#       backup-monthly.timer
#   journalctl -t gmac-backup --since "10 min ago"
#   systemctl list-timers 'backup*'
#
# Secrets: RESTIC_PASSWORD (from the environment or /opt/gmac/.env) and the
# rclone token (/root/.config/rclone/rclone.conf) are NEVER echoed, logged, or
# put in the manifest or the status row. This file and its units carry none.
#
# Not `set -e`, same reason as ops/storage_monitor.sh: a failed step must not
# abort the run before the failure is recorded. Every step reports through
# fail() and the run's exit code follows the FAILED flag.

set -uo pipefail

TAG=gmac-backup
INSTALL_DIR=${INSTALL_DIR:-/opt/gmac}
STACK_NAME=${STACK_NAME:-gmac}
ENV_FILE=${ENV_FILE:-$INSTALL_DIR/.env}
POSTGRES=${POSTGRES:-${STACK_NAME}-postgres}
HAYSTACK=${HAYSTACK:-${STACK_NAME}-haystack}
QDRANT_HELPER=${QDRANT_HELPER:-$INSTALL_DIR/ops/qdrant_backup.py}
QDRANT_URL=${QDRANT_URL:-http://qdrant:6333}
PG_USER=${PG_USER:-openwebui}
PG_DB=${PG_DB:-openwebui}
# Fixed path on purpose: the restic snapshot then always stores
# /var/tmp/gmac-backup/... so the restore paths in README.md are
# exact. Cleared at start and on exit; it only ever holds this script's own
# artifacts.
STAGING=${STAGING:-/var/tmp/gmac-backup}
LOCK_FILE=${LOCK_FILE:-/run/lock/gmac-backup.lock}
# Staging holds the pg dump (~1.5 GB) + 6 Qdrant snapshots (~250 MB). Refuse to
# start below this much free space on the staging filesystem — running out
# mid-dump is the classic way to get a truncated artifact that then fails late.
STAGE_FREE_GB=${STAGE_FREE_GB:-5}
# Google Drive headroom (step 5b). The account had 4.5 GiB free against a
# 3.9 GiB first snapshot when this was installed (2026-10-23), so the
# destination filling up is the likeliest way the nightly run starts failing.
#   DRIVE_WARN_FREE_GB   free space below this -> "Drive nearly full" warning
#                        (status row + panel); the run still uploads
#   DRIVE_MIN_FREE_GB    margin that must remain AFTER this run's worst-case
#                        upload; less -> the run fails before uploading
#   DRIVE_QUOTA_REQUIRED 1 -> an unreadable quota fails the run; 0 (default)
#                        -> it is a warning and the upload is attempted (a
#                        quota API hiccup must not cost a night's backup)
# Warn at 4 GB, not the original 10: the production Drive is a dedicated
# 10-15 GB one (repository ~1.4 GB in 2026-10-26; it grows with the retained
# history, see KEEP_* below) — a 10 GB threshold would have shown "nearly
# full" on such a drive every night forever. 4 GB is ~2 worst-case nightly
# uploads (~0.9 GB each) above the point where a run refuses to upload
# (need + 1 GB).
DRIVE_WARN_FREE_GB=${DRIVE_WARN_FREE_GB:-4}
DRIVE_MIN_FREE_GB=${DRIVE_MIN_FREE_GB:-1}
DRIVE_QUOTA_REQUIRED=${DRIVE_QUOTA_REQUIRED:-0}
RETAIN_DAYS=${RETAIN_DAYS:-90}          # ops_backup_reports row retention
# Set BACKUP_SKIP_RETENTION=1 for the first verified run(s): the backup is
# taken and uploaded, but no forget/prune happens until the operator has
# confirmed retention behaves as intended (README.md (Backups)).
SKIP_RETENTION=${BACKUP_SKIP_RETENTION:-0}
# Backup targets (2026-10-27): the Google Drive backup folder holds THREE
# restic repositories as subfolders - daily / weekly / monthly - and the
# restore page asks which one to restore from. Each folder keeps a FIXED
# number of recovery points (the agreed retention counts):
#   daily    three runs a day (03:00, 11:00, 19:00 UTC), newest 3 kept
#            (the last ~24 hours are always covered)
#   weekly   Friday 20:00 UTC, newest 2 kept
#   monthly  the 1st of the month 21:00 UTC, newest 1 kept
# Each repository is restic-encrypted with the passphrase in $ENV_FILE and
# de-duplicates inside itself, so a folder only stores its own changes on top
# of the points it keeps. A KEEP_LAST of 0 (or a typo) is refused
# (retention_args). Milestone snapshots carry no target tag: never removed.
TARGET=${BACKUP_TARGET:-daily}         # which subfolder this run writes to
# Name of the Google Drive folder holding the three repositories (the folder
# RESTIC_REPOSITORY=rclone:gdrive:<folder> points into, under root_folder_id).
BACKUP_DRIVE_FOLDER=${BACKUP_DRIVE_FOLDER:-backups}
KEEP_LAST_DAILY=${KEEP_LAST_DAILY:-3}
KEEP_LAST_WEEKLY=${KEEP_LAST_WEEKLY:-2}
KEEP_LAST_MONTHLY=${KEEP_LAST_MONTHLY:-1}
# The weekly/monthly on/off switches the dashboard writes (backup_ctl.
# write_schedule). Read by target_enabled; daily ignores it entirely.
BACKUP_SCHEDULE_FILE=${BACKUP_SCHEDULE_FILE:-/data/backup-drives/ctl/schedule.json}
# The backup catalog: one record per recovery point, read by the dashboard's
# "Restore backup" list. The LOCAL copy is what the panel reads (it never
# talks to Drive); catalog_update uploads a spare copy next to the
# repositories (ops/catalog.py is the writer).
BACKUP_CATALOG_FILE=${BACKUP_CATALOG_FILE:-/data/backup-drives/ctl/catalog.json}
# How restic starts rclone: restic's default rclone.args plus
# --drive-use-trash=false. rclone's Google Drive backend otherwise moves
# every pack that prune deletes into the Drive trash, where it keeps counting
# against the account quota for 30 days — the footprint would not drop when
# retention runs. Scoped to the restic repository's own files: this is a
# flag on `rclone serve restic` for the repository folder only. Never
# replace it with `rclone cleanup gdrive:` — on Drive that empties the
# WHOLE account's trash, including files that are not ours.
RESTIC_RCLONE_ARGS=${RESTIC_RCLONE_ARGS:-"serve restic --stdio --b2-hard-delete --drive-use-trash=false"}
# The six required collections. *_selftest is deliberately absent: rebuildable
# system-check fixtures, never worth a backup slot. ops/qdrant_backup.py
# refuses *selftest* names too, and this list is re-checked below.
COLLECTIONS=${COLLECTIONS:-"documents code_docs visual_frames ocr_frames od_frames transcript_segments"}

FAILED=0
STATUS_ID=""
SNAPSHOT_ID=""
BACKUP_ADDED=""
CAT_CHECK=""            # this run's restic check: ok | failed | '' (not reached)
LAST_ERROR=""
STAGE_STARTED=0
REPO_CREATED=0          # 1 when this run ran `restic init` (everything is new)
DRIVE_JSON="null"       # step 5b's result, carried into the status row's detail

# ── logging ─────────────────────────────────────────────────

log() {   # <message...>
    echo "$(date -u +%FT%TZ) $*"
    logger -t "$TAG" -- "$*" 2>/dev/null || true
}

fail() {   # <message...> — records the failure; the run continues where safe
    FAILED=1
    # First error wins: it is what the status row shows the panel. Messages are
    # this script's own short step names (self-tested to carry no secrets).
    [ -n "$LAST_ERROR" ] || LAST_ERROR="$*"
    echo "$(date -u +%FT%TZ) ERROR: $*" >&2
    logger -p err -t "$TAG" -- "ERROR: $*" 2>/dev/null || true
}

# The backup drive can be connected again from the dashboard ("Backup drive"
# page); that sign-in is applied here. No panel sign-in -> this changes nothing.
# shellcheck source=backup_drive.sh
. "$(dirname "${BASH_SOURCE[0]}")/backup_drive.sh"

# ── tiny JSON helpers (pure bash — no python needed on the host) ──

json_str() {   # <string> -> "quoted, escaped"
    local s=${1:-}
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    printf '"%s"' "$s"
}

json_val() {   # <value> -> null | bare number | "quoted string"
    local v=${1:-}
    if [ -z "$v" ]; then
        printf 'null'
    elif [[ "$v" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf '%s' "$v"
    else
        json_str "$v"
    fi
}

# ── .env access (read-only, silent) ─────────────────────────

env_get() {   # <KEY> -> value or empty. Reads only the one key it names.
    [ -f "$ENV_FILE" ] || return 0
    sed -n "s/^${1}=//p" "$ENV_FILE" | tail -n 1 | tr -d '\r'
}

load_secrets() {
    # RESTIC_REPOSITORY / RESTIC_PASSWORD from the environment win (manual
    # runs); otherwise their lines in /opt/gmac/.env. Exported for restic
    # only — never echoed anywhere.
    # RESTIC_BASE is the backup FOLDER, read once; the target subfolder is
    # appended to it, never to an already-targeted RESTIC_REPOSITORY. This
    # function runs more than once per backup (main and restic_preflight) --
    # appending to RESTIC_REPOSITORY itself wrote the first backups to
    # backups/weekly/weekly (found 2026-10-27).
    [ -n "${RESTIC_BASE:-}" ] || RESTIC_BASE="${RESTIC_REPOSITORY:-$(env_get RESTIC_REPOSITORY)}"
    [ -n "${RESTIC_PASSWORD:-}" ] || RESTIC_PASSWORD=$(env_get RESTIC_PASSWORD)
    # The backup folder holds one repository per subfolder (daily/weekly/
    # monthly). A typo can never write into the wrong place.
    case "$TARGET" in
        daily|weekly|monthly) ;;
        *) fail "unknown backup target '$TARGET' (daily, weekly or monthly)"; return 1 ;;
    esac
    RESTIC_REPOSITORY="${RESTIC_BASE%/}/$TARGET"
    export RESTIC_REPOSITORY RESTIC_PASSWORD
}

# ── status rows (ops_backup_reports -> GET /admin/backup) ───

pg() {   # psql in the postgres container; stdin is the SQL
    docker exec -i "$POSTGRES" psql -X -q -v ON_ERROR_STOP=1 -U "$PG_USER" -d "$PG_DB" "$@"
}

status_start() {
    # A "running" row first, so a long upload is visible while it runs.
    pg >/dev/null <<'SQL' \
        || { fail "could not create ops_backup_reports in $POSTGRES"; return 1; }
CREATE TABLE IF NOT EXISTS ops_backup_reports (
    id           serial PRIMARY KEY,
    started_at   timestamptz NOT NULL,
    finished_at  timestamptz,
    ok           boolean NOT NULL DEFAULT false,
    status       text NOT NULL,
    snapshot_id  text,
    bytes_added  bigint,
    error        text,
    detail       jsonb
);
SQL
    STATUS_ID=$(pg -tA <<'SQL' | tail -n 1
INSERT INTO ops_backup_reports (started_at, status) VALUES (now(), 'running') RETURNING id;
SQL
) || STATUS_ID=""
    [ -n "$STATUS_ID" ] || fail "could not write the 'running' row to ops_backup_reports"
}

status_finish() {   # <ok true|false> <snapshot_id> <bytes_added> <error> <detail_json>
    [ -n "$STATUS_ID" ] || return 0
    pg -v id="$STATUS_ID" -v ok="$1" -v snapshot="$2" -v bytes="$3" \
       -v error="$4" -v detail="$5" -v retain="$RETAIN_DAYS days" >/dev/null <<'SQL' \
       || fail "could not update ops_backup_reports row $STATUS_ID (backup result is only in the journal)"
UPDATE ops_backup_reports SET
    finished_at = now(),
    ok          = :'ok'::boolean,
    status      = CASE WHEN :'ok'::boolean THEN 'success' ELSE 'failed' END,
    snapshot_id = NULLIF(:'snapshot',''),
    bytes_added = NULLIF(:'bytes','')::bigint,
    error       = NULLIF(:'error',''),
    detail      = NULLIF(:'detail','')::jsonb
WHERE id = :'id'::int;
DELETE FROM ops_backup_reports WHERE started_at < now() - :'retain'::interval;
SQL
}

# ── staging ─────────────────────────────────────────────────

stage_open() {
    case "$STAGING" in
        /var/tmp/*) ;;
        *) fail "refusing to use a staging dir outside /var/tmp: $STAGING"; return 1 ;;
    esac
    # Disk headroom BEFORE creating a single byte of dump. df -Pk: portable
    # 1024-byte blocks (POSIX -P and -k), $4 = Available on the header line.
    # dirname, not "$STAGING/..": on the first run $STAGING does not exist yet
    # and an unresolvable path makes df fail silently into the check below.
    local avail_kb need_kb=$((STAGE_FREE_GB * 1024 * 1024))
    avail_kb=$(df -Pk "$(dirname "$STAGING")" 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -n "$avail_kb" ] && [ "$avail_kb" -lt "$need_kb" ]; then
        fail "only $((avail_kb / 1024)) MB free on the staging filesystem — need ${STAGE_FREE_GB} GB"
        return 1
    fi
    mkdir -p "$STAGING/postgres" "$STAGING/qdrant" \
        || { fail "cannot create staging dir $STAGING"; return 1; }
    # Root-only: the staging tree holds the pg dump (users' chats) and the
    # Qdrant snapshots before they are encrypted into the restic repository.
    chmod 700 "$STAGING" "$STAGING/postgres" "$STAGING/qdrant"
    # Clear leftovers of a previous run (the flock is already held, so no other
    # run can be reading this directory; it only ever holds this script's own
    # artifacts).
    find "$STAGING" -mindepth 1 -maxdepth 1 ! -name postgres ! -name qdrant \
        -exec rm -rf -- {} +
    find "$STAGING/postgres" "$STAGING/qdrant" -mindepth 1 -exec rm -rf -- {} +
    STAGE_STARTED=1
}

stage_close() {
    [ "$STAGE_STARTED" = 1 ] || return 0
    find "$STAGING" -mindepth 1 -exec rm -rf -- {} + 2>/dev/null || true
    STAGE_STARTED=0
}

# ── step 2: PostgreSQL dump ─────────────────────────────────

backup_postgres() {
    log "postgres: dumping $PG_DB"
    # docker exec (no -t) is binary-safe and propagates pg_dump's exit code.
    docker exec "$POSTGRES" pg_dump -U "$PG_USER" -d "$PG_DB" -Fc \
        > "$STAGING/postgres/$PG_DB.dump" \
        || { fail "pg_dump failed"; return 1; }
    local size
    size=$(stat -c%s "$STAGING/postgres/$PG_DB.dump" 2>/dev/null || echo 0)
    [ "$size" -gt 0 ] || { fail "pg_dump produced an empty dump"; return 1; }
    # Parse the archive back — catches a truncated or corrupt dump BEFORE upload.
    docker exec -i "$POSTGRES" pg_restore -l \
        < "$STAGING/postgres/$PG_DB.dump" >/dev/null \
        || { fail "pg_dump output does not parse as a pg_restore archive"; return 1; }
    log "postgres: dump ok ($size bytes, archive parses)"
}

# ── step 3: Qdrant snapshots ────────────────────────────────

qdrant_helper() {   # <args...> — runs ops/qdrant_backup.py inside ${STACK_NAME}-haystack
    docker exec -i -e QDRANT_URL="$QDRANT_URL" "$HAYSTACK" \
        python3 - "$@" < "$QDRANT_HELPER"
}

backup_qdrant() {
    local line name size checksum out out_size
    for collection in $COLLECTIONS; do
        case "$collection" in
            *selftest*) fail "refusing to back up selftest collection $collection"; return 1 ;;
        esac
        log "qdrant: snapshotting $collection"
        line=$(qdrant_helper snap "$collection") \
            || { fail "qdrant: could not create snapshot for $collection"; return 1; }
        # snap prints name<TAB>size<TAB>checksum
        IFS=$'\t' read -r name size checksum <<< "$line"
        [ -n "$name" ] && [ -n "$size" ] \
            || { fail "qdrant: unusable snapshot description for $collection: $line"; return 1; }
        out="$STAGING/qdrant/$collection.snapshot"
        qdrant_helper get "$collection" "$name" > "$out" \
            || { fail "qdrant: could not download snapshot for $collection"; return 1; }
        out_size=$(stat -c%s "$out" 2>/dev/null || echo 0)
        if [ "$out_size" != "$size" ]; then
            fail "qdrant: $collection snapshot is $out_size bytes, Qdrant reported $size"
            return 1
        fi
        # Download verified — remove the server-side copy Qdrant keeps under its
        # storage dir (delete_snapshot, v1.19.0 spec). Without this every nightly
        # run leaves ~250 MB behind and /data/qdrant grows until the disk fills.
        # Scoped strictly to the snapshot THIS run created by name; the
        # collection itself is never touched. A failure here is a loud warning,
        # not a failed backup: the recovery point is already safe on disk.
        qdrant_helper delete "$collection" "$name" >/dev/null 2>&1 \
            || log "WARNING: qdrant: could not delete server-side snapshot $collection/$name — remove it by hand or /data/qdrant will keep growing"
        log "qdrant: $collection ok ($out_size bytes)"
    done
}

# ── steps 4-5: manifest + verification ───────────────────────

file_entry() {   # <relative-name> <absolute-path> -> JSON object
    local rel="$1" abs="$2" sum size
    sum=$(sha256sum "$abs" | cut -d' ' -f1)
    size=$(stat -c%s "$abs")
    printf '{"file":%s,"size":%s,"sha256":%s}' "$(json_str "$rel")" "$size" "$(json_str "$sum")"
}

path_stats() {   # <dir> -> {"path","bytes","files"} — cheap inventory, no hashing
    local dir="$1" bytes files=""
    if [ -d "$dir" ]; then
        bytes=$(du -sb "$dir" 2>/dev/null | cut -f1)
        files=$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
    else
        bytes=""
    fi
    printf '{"path":%s,"bytes":%s,"files":%s}' \
        "$(json_str "$dir")" "$(json_val "$bytes")" "$(json_val "$files")"
}

write_manifest() {
    log "manifest: writing"
    local created hostname git_rev git_dirty restic_ver rclone_ver pg_ver qd_info qd_ver
    local db_size entries sep f rel sum size cfg_entries uploads_json list_json
    created=$(date -u +%FT%TZ)
    hostname=$(hostname)
    git_rev=$(git -C "$INSTALL_DIR" rev-parse HEAD 2>/dev/null || echo "")
    git_dirty=$( [ -n "$(git -C "$INSTALL_DIR" status --porcelain 2>/dev/null)" ] && echo true || echo false )
    restic_ver=$(restic version 2>/dev/null | head -n 1 || echo "")
    rclone_ver=$(rclone version 2>/dev/null | head -n 1 || echo "")
    pg_ver=$(docker exec "$POSTGRES" psql -X -tA -U "$PG_USER" -d "$PG_DB" \
                 -c "SELECT version()" 2>/dev/null | head -n 1 || echo "")
    db_size=$(docker exec "$POSTGRES" psql -X -tA -U "$PG_USER" -d "$PG_DB" \
                  -c "SELECT pg_database_size('$PG_DB')" 2>/dev/null | tail -n 1 || echo "")
    qd_info=$(qdrant_helper info 2>/dev/null || echo '{"version":"","title":""}')
    qd_ver=$(printf '%s' "$qd_info" | sed -n 's/.*"version": *"\([^"]*\)".*/\1/p')

    # Artifact inventory: every staged file. These are the checksums
    # MANIFEST.sha256 verifies before upload.
    entries=""; sep=""
    for f in "$STAGING/postgres/$PG_DB.dump" "$STAGING"/qdrant/*.snapshot; do
        [ -f "$f" ] || continue
        rel=${f#"$STAGING"/}
        entries+="$sep$(file_entry "$rel" "$f")"; sep=","
    done

    # Config originals (checksummed for provenance only — the config is not
    # staged, restic backs it up from where it runs).
    cfg_entries=""; sep=""
    for f in "$ENV_FILE" "$INSTALL_DIR/compose.yml" "$INSTALL_DIR/Caddyfile" \
             "$INSTALL_DIR/litellm-config.yaml"; do
        [ -f "$f" ] || continue
        sum=$(sha256sum "$f" | cut -d' ' -f1)
        size=$(stat -c%s "$f")
        cfg_entries+="$sep{\"path\":$(json_str "$f"),\"size\":$size,\"sha256\":$(json_str "$sum")}"
        sep=","
    done
    if [ -d "$INSTALL_DIR/tika-config" ]; then
        sum=$(find "$INSTALL_DIR/tika-config" -type f | sort | xargs -r sha256sum | sha256sum | cut -d' ' -f1)
        size=$(du -sb "$INSTALL_DIR/tika-config" | cut -f1)
        cfg_entries+="$sep{\"path\":$(json_str "$INSTALL_DIR/tika-config"),\"size\":$size,\"sha256\":$(json_str "$sum")}"
    fi

    # Inventory exactly what restic_backup uploads (backup_paths), so the
    # restore drill never expects a directory the snapshot does not hold.
    uploads_json=""
    for src in $(backup_paths); do
        uploads_json+="${uploads_json:+,}$(path_stats "$src")"
    done

    # Per-collection point counts ride along: if a collection is empty after a
    # restore, the manifest proves whether it was empty at backup time.
    list_json=$(qdrant_helper list 2>/dev/null || echo '[]')

    # The item inventory (documents, spreadsheets, videos, uploads): which
    # items this recovery point holds. The weekly review and the approved
    # cleanup read it back from every snapshot. Best effort - an extraction
    # failure is recorded as null and NEVER fails a backup.
    inventory_json=$(python3 "$INSTALL_DIR/ops/inventory.py" live 2>/dev/null || true)
    case "$inventory_json" in
        '{'*) ;;
        *) inventory_json=null ;;
    esac

    {
        printf '{\n'
        printf '  "created_at": %s,\n' "$(json_str "$created")"
        printf '  "hostname": %s,\n' "$(json_str "$hostname")"
        printf '  "git_revision": %s,\n' "$(json_str "$git_rev")"
        printf '  "git_dirty": %s,\n' "$git_dirty"
        printf '  "tool_versions": {"restic": %s, "rclone": %s, "postgres": %s, "qdrant": %s},\n' \
            "$(json_str "$restic_ver")" "$(json_str "$rclone_ver")" \
            "$(json_str "$pg_ver")" "$(json_str "$qd_ver")"
        printf '  "database": {"name": %s, "size_bytes": %s, "artifacts": [%s]},\n' \
            "$(json_str "$PG_DB")" "$(json_val "$db_size")" "$entries"
        printf '  "qdrant": {"url": %s, "collections": %s},\n' \
            "$(json_str "$QDRANT_URL")" "$list_json"
        printf '  "source_files": [%s],\n' "$uploads_json"
        printf '  "config": [%s],\n' "$cfg_entries"
        printf '  "target": %s,\n' "$(json_str "$TARGET")"
        printf '  "retention": {"keep_last": %d},\n' "$(keep_last_for "$TARGET")"
        printf '  "inventory": %s,\n' "$inventory_json"
        printf '  "restic_repository": %s\n' "$(json_str "${RESTIC_REPOSITORY:-}")"
        printf '}\n'
    } > "$STAGING/MANIFEST.json"

    # Checksums of the artifacts, paths relative to the staging dir, so the
    # pre-upload check here and the post-restore check in the runbook are the
    # same command:
    #   cd /restore/var/tmp/gmac-backup && sha256sum -c MANIFEST.sha256
    # The sed normalizes sha256sum's two output forms ("hash  file" text vs
    # "hash *file" binary, which Git Bash uses) to the text form, so the file
    # looks the same wherever it was written. sha256sum -c reads both.
    ( cd "$STAGING" && find postgres qdrant -type f | sort | xargs -r sha256sum \
        | sed 's/^\([0-9a-f]\{64\}\) [* ]/\1  /' ) \
        > "$STAGING/MANIFEST.sha256" \
        || { fail "could not write MANIFEST.sha256"; return 1; }
    log "manifest: written ($(stat -c%s "$STAGING/MANIFEST.json") bytes)"
}

verify_stage() {
    log "verify: checking staged artifacts against MANIFEST.sha256"
    ( cd "$STAGING" && sha256sum -c --quiet MANIFEST.sha256 ) \
        || { fail "staged artifacts do not match MANIFEST.sha256"; return 1; }
    log "verify: all staged checksums match"
}

# ── steps 6-8: restic ───────────────────────────────────────

restic_preflight() {
    load_secrets
    [ -n "$RESTIC_REPOSITORY" ] \
        || { fail "RESTIC_REPOSITORY is not set ($ENV_FILE or environment)"; return 1; }
    case "$RESTIC_REPOSITORY" in
        rclone:*) ;;
        *) fail "RESTIC_REPOSITORY must be an rclone backend (rclone:...) — see README.md"
           return 1 ;;
    esac
    [ -n "$RESTIC_PASSWORD" ] \
        || { fail "RESTIC_PASSWORD is not set ($ENV_FILE or environment) — see README.md (Backups)"; return 1; }
    [ -f "$QDRANT_HELPER" ] || { fail "missing $QDRANT_HELPER"; return 1; }

    # A sign-in made in the dashboard replaces the remote's token/folder (in a
    # temporary copy of the rclone config) before anything below talks to Drive.
    drive_conf_setup
    if [ "$DRIVE_SOURCE" = "disconnected" ]; then
        fail "Google Drive is not connected (signed out on the Drive Backup & Upload Handler page) — nothing was backed up; sign in with Google to resume backups"
        return 1
    fi

    # The rclone remote must be configured and reachable. root_folder_id on the
    # remote points it at the designated shared backup Drive folder ($BACKUP_DRIVE_FOLDER)
    # (README.md, Backups). stderr passes through: "didn't find section" and
    # "token expired" need different operator fixes.
    local remote="${RESTIC_REPOSITORY#rclone:}"
    remote="${remote%%:*}"
    rclone lsd "$remote:" >/dev/null \
        || { fail "rclone remote '$remote:' is not configured or not reachable (README.md (Backups))"; return 1; }

    # The repository must already exist. A scheduled or manual backup NEVER
    # creates one by itself: `restic cat config` also fails on a wrong
    # passphrase, a network error or a different Google account, and an
    # automatic `restic init` then silently started an empty backup history
    # next to (or instead of) the real one. Creating a repository is an
    # explicit, one-time operator step on the host, once per folder:
    #   ALLOW_REPO_INIT=1 bash ops/backup.sh --target daily   (and weekly, monthly)
    local probe
    if ! probe=$(restic cat config 2>&1 >/dev/null); then
        if [ "${ALLOW_REPO_INIT:-0}" = 1 ] && printf '%s' "$probe" | grep -q "does not exist"; then
            log "creating the backup repository in ${RESTIC_REPOSITORY} (ALLOW_REPO_INIT=1)"
            restic init \
                || { fail "restic init failed"; return 1; }
            REPO_CREATED=1
        else
            fail "the $TARGET backup repository (${RESTIC_REPOSITORY}) cannot be opened — nothing was created or changed: $(printf '%s' "$probe" | grep -m1 -iE 'fatal|error|does not exist|wrong password' | sed 's/^Fatal: *//' | cut -c1-200)"
            return 1
        fi
    fi
}

# ── step 5b: Google Drive free space ────────────────────────

drive_about() {   # <remote> -> `rclone about --json` (tests replace this function)
    rclone about "$1:" --json 2>/dev/null
}

drive_field() {   # <json> <field> -> the field's integer value, or empty
    printf '%s' "$1" | tr -d ' \n\r\t' | sed -n "s/.*\"$2\":\([0-9][0-9]*\).*/\1/p" | head -n 1
}

drive_grade() {   # <free_bytes|''> <need_bytes> <min_free_bytes> <warn_free_bytes>
    # -> unknown | insufficient | low | ok. Pure: the self-test covers every branch.
    local free="$1" need="$2" min="$3" warn="$4"
    if [ -z "$free" ]; then echo unknown
    elif [ "$free" -lt $((need + min)) ]; then echo insufficient
    elif [ "$free" -lt "$warn" ]; then echo low
    else echo ok
    fi
}

drive_check() {
    # Worst case for this run's upload: the staged artifacts (a fresh dump and
    # fresh snapshots dedupe poorly) — plus, on the first run, the uploaded
    # files and Haystack state, which are entirely new to the repository.
    # restic's own deduplication usually needs far less; this errs high on
    # purpose, because running out mid-upload fails late and leaves a partial
    # pack behind.
    local remote json free total used need status gb=$((1024 * 1024 * 1024)) src
    local min=$((DRIVE_MIN_FREE_GB * gb)) warn=$((DRIVE_WARN_FREE_GB * gb))
    remote="${RESTIC_REPOSITORY#rclone:}"; remote="${remote%%:*}"
    need=$(du -sb "$STAGING" 2>/dev/null | cut -f1); need=${need:-0}
    if [ "$REPO_CREATED" = 1 ]; then
        for src in $(backup_paths); do
            [ -d "$src" ] && need=$((need + $(du -sb "$src" 2>/dev/null | cut -f1)))
        done
    fi

    json=""; free=""; total=""; used=""
    if json=$(drive_about "$remote") && [ -n "$json" ]; then
        free=$(drive_field "$json" free)
        total=$(drive_field "$json" total)
        used=$(drive_field "$json" used)
    fi
    status=$(drive_grade "$free" "$need" "$min" "$warn")
    DRIVE_JSON=$(printf '{"status":%s,"free_bytes":%s,"total_bytes":%s,"used_bytes":%s,"need_bytes":%s,"warn_free_bytes":%s,"min_free_bytes":%s}' \
        "$(json_str "$status")" "$(json_val "$free")" "$(json_val "$total")" "$(json_val "$used")" \
        "$(json_val "$need")" "$(json_val "$warn")" "$(json_val "$min")")

    case "$status" in
        ok)
            log "drive: $((free / 1024 / 1024)) MB free on '$remote:' — this run may add up to $((need / 1024 / 1024)) MB" ;;
        low)
            log "WARNING: drive: Google Drive nearly full — $((free / 1024 / 1024)) MB free (warning below ${DRIVE_WARN_FREE_GB} GB); this run may add up to $((need / 1024 / 1024)) MB. Free space or upgrade the account's storage" ;;
        insufficient)
            fail "Google Drive has only $((free / 1024 / 1024)) MB free; this run may need up to $((need / 1024 / 1024)) MB plus a ${DRIVE_MIN_FREE_GB} GB margin — nothing uploaded, no backup deleted; free space or upgrade the account's storage"
            return 1 ;;
        unknown)
            if [ "$DRIVE_QUOTA_REQUIRED" = 1 ]; then
                fail "could not read Google Drive free space for '$remote:' (rclone about) and DRIVE_QUOTA_REQUIRED=1 — nothing uploaded"
                return 1
            fi
            log "WARNING: drive: could not read Google Drive free space for '$remote:' (rclone about) — uploading anyway (DRIVE_QUOTA_REQUIRED=0)" ;;
    esac
}

backup_paths() {   # -> the file trees each snapshot holds besides staging/config, one per line
    # /data/openwebui/uploads (chat attachments) is deliberately NOT here since
    # 2026-10-27: it was the bulk of every snapshot, is not knowledge-base
    # content, and a restore of it resurrected files whose chats were gone.
    # One list, used by restic_backup, drive_check and the manifest, so the
    # three can never disagree about what a snapshot contains.
    printf '%s\n' /data/haystack-uploads
}

target_enabled() {   # <target> -> exit 0 = run, 1 = switched off in the dashboard
    # Daily is not switchable by anything: it is the only folder that covers
    # the last day, so no file, typo or UI state may ever turn it off.
    [ "$1" = daily ] && return 0
    # The panel (a container) owns $BACKUP_SCHEDULE_FILE's directory; this
    # runs as root, so the file is opened O_NOFOLLOW — a symlink planted there
    # is refused instead of read. Every failure (missing, unreadable, broken,
    # symlink, not an object, any value but a literal JSON false) prints
    # "on": a bad file may cost an extra backup, never a missing one.
    command -v python3 >/dev/null 2>&1 || return 0
    local verdict
    verdict=$(python3 - "$BACKUP_SCHEDULE_FILE" "$1" <<'PY' 2>/dev/null
import json, os, sys
try:
    fd = os.open(sys.argv[1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, encoding="utf-8") as fh:
        rec = json.load(fh)
    print("off" if isinstance(rec, dict) and rec.get(sys.argv[2]) is False else "on")
except Exception:
    print("on")
PY
)
    [ "$verdict" != off ]
}

restic_backup() {   # <tag> — one snapshot of staging + uploads + state + config
    local tag="$1" rc json_out
    log "restic: backing up to ${RESTIC_REPOSITORY:-?} (tag $tag)"
    # --json goes to stdout (parsed below for the summary); restic's human
    # messages go to stderr and stay visible in the journal. No secret on the
    # command line: RESTIC_PASSWORD comes from the environment.
    json_out=$(mktemp /var/tmp/gmac-restic.XXXXXX)
    restic backup --json --tag gmac-backup --tag "$tag" \
        "$STAGING" \
        $(backup_paths) \
        "$ENV_FILE" \
        "$INSTALL_DIR/compose.yml" \
        "$INSTALL_DIR/Caddyfile" \
        "$INSTALL_DIR/litellm-config.yaml" \
        "$INSTALL_DIR/tika-config" > "$json_out"
    rc=$?
    if [ "$rc" != 0 ]; then
        rm -f "$json_out"
        fail "restic backup failed (exit $rc)"
        return 1
    fi
    # The summary message carries data_added and snapshot_id — the real numbers,
    # not guesses.
    BACKUP_ADDED=$(sed -n 's/.*"data_added":\([0-9]*\).*/\1/p' "$json_out" | tail -n 1)
    SNAPSHOT_ID=$(sed -n 's/.*"snapshot_id":"\([0-9a-f]*\)".*/\1/p' "$json_out" | tail -n 1)
    rm -f "$json_out"
    if [ -z "$SNAPSHOT_ID" ]; then
        # Older restic without snapshot_id in the summary: read it back.
        SNAPSHOT_ID=$(restic snapshots --latest 1 --host "$(hostname)" --json 2>/dev/null \
                   | sed -n 's/.*"short_id": *"\([0-9a-f]*\)".*/\1/p' | head -n 1)
    fi
    [ -n "$SNAPSHOT_ID" ] || { fail "restic backup succeeded but no snapshot is listed"; return 1; }
    log "restic: snapshot $SNAPSHOT_ID created (${BACKUP_ADDED:-?} new bytes in repo)"
}

restic_check() {
    # Metadata check every night (cheap). A rotating 5% data sample on Sundays;
    # a full `restic check --read-data` is a documented monthly manual check.
    # CAT_CHECK carries the result into the catalog record: only a run whose
    # check passed is "verified".
    if [ "$(date -u +%u)" = "7" ]; then
        log "restic: check + 5% data sample (Sunday)"
        if ! restic check --read-data-subset=5%; then
            CAT_CHECK=failed
            fail "restic check --read-data-subset failed"
            return 1
        fi
    else
        log "restic: check (metadata)"
        if ! restic check; then
            CAT_CHECK=failed
            fail "restic check failed"
            return 1
        fi
    fi
    CAT_CHECK=ok
    log "restic: repository verified"
}

keep_last_for() {   # <target> -> the folder's kept count (empty on a typo)
    case "$1" in
        daily)   printf '%s' "${KEEP_LAST_DAILY:-3}" ;;
        weekly)  printf '%s' "${KEEP_LAST_WEEKLY:-2}" ;;
        monthly) printf '%s' "${KEEP_LAST_MONTHLY:-1}" ;;
    esac
}

retention_args() {   # -> the `restic forget` arguments; exit 1 on an unsafe policy
    # --tag <target> selects THIS folder's scheduled snapshots only: milestone
    # snapshots are outside the policy and never removed. A kept count of 0
    # (or a typo) would empty the folder — refuse it.
    local keep
    keep=$(keep_last_for "$TARGET")
    case "$keep" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$keep" -ge 1 ] || return 1
    printf -- '--tag %s --keep-last %s --prune' "$TARGET" "$keep"
}

restic_retention() {
    # ONLY after a verified backup (restic_backup + restic_check both passed).
    # The folder keeps its newest N recovery points; everything older in THIS
    # repository is forgotten and pruned. The other folders are untouched.
    local args
    args=$(retention_args) \
        || { fail "unsafe retention policy for '$TARGET' (kept count must be a number >= 1) — retention not applied"; return 1; }
    log "retention ($TARGET): $args"
    # shellcheck disable=SC2086  # $args is this script's own fixed word list
    restic -o rclone.args="$RESTIC_RCLONE_ARGS" forget $args \
        || { fail "restic forget/prune failed (the new snapshot is safe; retention did not apply)"; return 1; }
    # The snapshot this run just made must survive its own retention pass.
    restic snapshots "$SNAPSHOT_ID" --json 2>/dev/null | grep -q "$SNAPSHOT_ID" \
        || { fail "retention removed this run's own snapshot $SNAPSHOT_ID — investigate"; return 1; }
    log "retention: applied"
}

# ── step 9: the backup catalog (catalog.json) ────────────────

catalog_ids_from_json() {   # stdin: `restic snapshots --json` -> one short id per line
    # grep -o (not sed): restic pretty-prints one "short_id" per line, but a
    # compact array on one line must yield every id, not just the last.
    grep -o '"short_id": *"[0-9a-f]*"' | sed 's/.*"\([0-9a-f]*\)"$/\1/'
}

catalog_ids() {   # -> every snapshot id restic still holds in THIS repository
    restic snapshots --json 2>/dev/null | catalog_ids_from_json
}

catalog_update() {   # <combined-tag> — record this run in the backup catalog
    # The catalog (ops/catalog.py) is what the dashboard's "Restore backup"
    # list reads: when this recovery point was taken, what it holds and what
    # it does not, and whether its check passed. Written at the end of every
    # run — and so again after retention, which marks removed snapshots
    # pruned. Best effort throughout: the snapshot is already safe and
    # verified here; a catalog problem is logged and NEVER fails a run.
    local combined="$1" facts ids py
    [ -n "$SNAPSHOT_ID" ] || return 0
    py=$(command -v python3 || command -v python || true)
    local helper; helper="$(dirname "${BASH_SOURCE[0]}")/catalog.py"
    if [ -z "$py" ] || [ ! -f "$helper" ]; then
        log "catalog: python3 or $helper missing — catalog.json not updated"
        return 0
    fi
    local status="unverified" verified_at=""
    case "$CAT_CHECK" in
        ok)     status="verified"; verified_at=$(date -u +%FT%TZ) ;;
        failed) status="failed-check" ;;
    esac
    facts=$(mktemp); ids=$(mktemp)
    catalog_ids > "$ids"
    printf '{"id":%s,"short_id":%s,"folder":%s,"tags":["gmac-backup",%s],"status":%s,"verified_at":%s,"size_added_bytes":%s}' \
        "$(json_str "$SNAPSHOT_ID")" "$(json_str "${SNAPSHOT_ID:0:8}")" "$(json_str "$TARGET")" \
        "$(json_str "$combined")" "$(json_str "$status")" "$(json_str "$verified_at")" \
        "$(json_val "${BACKUP_ADDED:-}")" > "$facts"
    if "$py" "$helper" add "$BACKUP_CATALOG_FILE" "$STAGING/MANIFEST.json" "$facts" "$ids"; then
        log "catalog: written ($BACKUP_CATALOG_FILE)"
    else
        log "WARNING: catalog: catalog.json could not be updated (the snapshot is safe; see the message above)"
        rm -f "$facts" "$ids"
        return 0
    fi
    rm -f "$facts" "$ids"
    # The Drive copy next to the repositories is a spare for disaster
    # recovery; best effort, the local copy is the one the panel reads.
    case "${RESTIC_BASE:-}" in
        rclone:*)
            if rclone copyto "$BACKUP_CATALOG_FILE" "${RESTIC_BASE#rclone:}/catalog.json"; then
                log "catalog: uploaded to ${RESTIC_BASE#rclone:}/catalog.json"
            else
                log "WARNING: catalog: the Drive copy of catalog.json could not be uploaded (the local copy is current)"
            fi ;;
    esac
}

# ── self-test (offline: no docker, no restic, no network) ───

self_test() {
    local n=0 py
    check() {   # <name> <expected> <actual>
        if [ "$2" != "$3" ]; then
            printf 'FAIL %s\n  expected: %s\n  got:      %s\n' "$1" "$2" "$3" >&2
            exit 1
        fi
        n=$((n + 1)); printf 'ok %s\n' "$1"
    }

    check json_str_escapes '"a \"q\" b\\c d"' "$(json_str "a \"q\" b\\c d")"
    check json_val_number "75" "$(json_val 75)"
    check json_val_decimal "11.5" "$(json_val 11.5)"
    check json_val_null "null" "$(json_val "")"
    check json_val_string '"text"' "$(json_val text)"

    # Open WebUI chat uploads left the backup on 2026-10-27: the knowledge
    # base's own files stay, the chat attachments must not creep back in.
    check backup_paths_has_haystack_uploads "1" "$(backup_paths | grep -cx /data/haystack-uploads)"
    check backup_paths_exclude_openwebui_uploads "0" "$(backup_paths | grep -c openwebui/uploads || true)"

    # Weekly/monthly switches (dashboard -> schedule.json). Daily can never be
    # switched off; only a literal JSON false disables; anything unreadable,
    # broken or symlinked reads as ON, so a bad file can only cause an extra
    # backup, never a missing one.
    if command -v python3 >/dev/null 2>&1; then
        local sd sf; sd=$(mktemp -d); sf="$sd/schedule.json"
        en() { BACKUP_SCHEDULE_FILE="$sf" target_enabled "$1" && echo on || echo off; }
        printf '{"daily": false, "weekly": false, "monthly": false}' > "$sf"
        check schedule_daily_always_on on "$(en daily)"
        check schedule_weekly_off off "$(en weekly)"
        check schedule_monthly_off off "$(en monthly)"
        printf '{"monthly": true}' > "$sf"
        check schedule_monthly_on on "$(en monthly)"
        check schedule_weekly_key_missing_on on "$(en weekly)"
        printf '{"weekly": "false", "monthly": 0}' > "$sf"
        check schedule_only_literal_false_disables "on|on" "$(en weekly)|$(en monthly)"
        printf '{broken' > "$sf"
        check schedule_broken_file_on on "$(en weekly)"
        rm -f "$sf"
        check schedule_missing_file_on on "$(en weekly)"
        printf '{"weekly": false}' > "$sd/elsewhere.json"
        # Git Bash on NTFS copies instead of linking unless symlinks are
        # enabled; the check only means something where ln -s makes a link.
        if ln -s "$sd/elsewhere.json" "$sf" 2>/dev/null && [ -L "$sf" ]; then
            check schedule_symlink_not_followed on "$(en weekly)"
        else
            printf 'skip schedule_symlink_not_followed (no symlinks on this filesystem)\n'
        fi
        rm -rf "$sd"; unset -f en
    else
        printf 'skip schedule switch checks (no python3)\n'
    fi

    # The collection list: exactly the six required, no selftest, no duplicates.
    check collections_count "6" "$(printf '%s\n' $COLLECTIONS | wc -l | tr -d ' ')"
    check collections_no_selftest "0" "$(printf '%s\n' $COLLECTIONS | grep -c selftest || true)"
    check collections_no_duplicates "6" \
        "$(printf '%s\n' $COLLECTIONS | sort -u | wc -l | tr -d ' ')"

    # Retention args exactly as they are passed to restic forget, per folder:
    # daily keeps the newest 3, weekly 2, monthly 1 (the agreed retention counts).
    check retention_args_daily "--tag daily --keep-last 3 --prune" "$(TARGET=daily retention_args)"
    check retention_args_weekly "--tag weekly --keep-last 2 --prune" "$(TARGET=weekly retention_args)"
    check retention_args_monthly "--tag monthly --keep-last 1 --prune" "$(TARGET=monthly retention_args)"
    # An emptied folder (kept 0), a typo or an unknown target must be refused
    # rather than run.
    check retention_refuses_keep_last_0 "1" "$(TARGET=monthly KEEP_LAST_MONTHLY=0 retention_args >/dev/null 2>&1; echo $?)"
    check retention_refuses_garbage "1" "$(TARGET=daily KEEP_LAST_DAILY=two retention_args >/dev/null 2>&1; echo $?)"
    check retention_refuses_bad_target "1" "$(TARGET=yearly retention_args >/dev/null 2>&1; echo $?)"
    # Safety backups (pre-restore milestones, tag milestone:pre-restore-*) are
    # NEVER removed by retention: the forget selector is exactly this folder's
    # scheduled tag, so no snapshot carrying milestone:* can ever match it.
    check retention_never_selects_safety "0" "$(TARGET=daily retention_args | grep -c milestone)"

    # The repository path: the target is appended ONCE however often
    # load_secrets runs (main + restic_preflight) -- never backups/weekly/weekly.
    check repo_path_once "rclone:gdrive:backups/weekly" \
        "$(RESTIC_BASE='' RESTIC_REPOSITORY=rclone:gdrive:backups RESTIC_PASSWORD=x TARGET=weekly \
            bash -c "$(declare -f env_get load_secrets fail log); load_secrets; load_secrets; load_secrets; echo \$RESTIC_REPOSITORY")"
    check repo_path_daily "rclone:gdrive:backups/daily" \
        "$(RESTIC_BASE='' RESTIC_REPOSITORY=rclone:gdrive:backups/ RESTIC_PASSWORD=x TARGET=daily \
            bash -c "$(declare -f env_get load_secrets fail log); load_secrets; load_secrets; echo \$RESTIC_REPOSITORY")"
    # No automatic `restic init`: creating a repository needs ALLOW_REPO_INIT=1
    # (an explicit operator step). The only init call is inside that branch.
    check no_unconfirmed_init "1" "$(grep -c '^ *restic init' "${BASH_SOURCE[0]}")"
    # Signed out on the handler page -> not connected: the backup refuses, it
    # never falls back to the server's own remote (the previous account).
    local so; so=$(mktemp -d)
    : > "$so/signed-out.json"
    check drive_signed_out_is_disconnected "disconnected" "$(BACKUP_DRIVE_OVERRIDE="$so/primary.json"         RESTIC_REPOSITORY=rclone:gdrive:backups/daily bash -c "$(declare -f drive_conf_setup drive_conf_close log);         DRIVE_RUN_DIR=$so; drive_conf_setup; echo \$DRIVE_SOURCE")"
    rm -f "$so/signed-out.json"
    check drive_no_signout_is_server "server" "$(BACKUP_DRIVE_OVERRIDE="$so/primary.json"         RESTIC_REPOSITORY=rclone:gdrive:backups/daily bash -c "$(declare -f drive_conf_setup drive_conf_close log);         DRIVE_RUN_DIR=$so; drive_conf_setup; echo \$DRIVE_SOURCE")"
    rm -rf "$so"
    check init_is_gated "1" "$(grep -c '^ *if \[ "${ALLOW_REPO_INIT:-0}" = 1 \] && ' "${BASH_SOURCE[0]}")"
    # Deleted packs must leave Drive at once: rclone's Drive default moves
    # them to the trash, which still counts against the quota for 30 days.
    check rclone_args_hard_delete "serve restic --stdio --b2-hard-delete --drive-use-trash=false" \
        "$RESTIC_RCLONE_ARGS"
    # The "nearly full" warning must stay quiet on a 10 GB drive reserved for
    # backups (flat ~1.4-3 GB footprint -> 7+ GB free).
    check drive_warn_default_fits_10gb_drive "1" "$([ "$DRIVE_WARN_FREE_GB" -le 4 ] && echo 1 || echo 0)"
    check drive_grade_10gb_drive_3gb_footprint_ok "ok" \
        "$(drive_grade $((7 * 1024 * 1024 * 1024)) $((900 * 1024 * 1024)) $((DRIVE_MIN_FREE_GB * 1024 * 1024 * 1024)) $((DRIVE_WARN_FREE_GB * 1024 * 1024 * 1024)))"

    # Manifest JSON + its checksums, built from fixture artifacts.
    STAGING=$(mktemp -d); STAGE_STARTED=1
    mkdir -p "$STAGING/postgres" "$STAGING/qdrant"
    printf 'DUMPDATA' > "$STAGING/postgres/openwebui.dump"
    printf 'SNAPDATA' > "$STAGING/qdrant/documents.snapshot"
    ENV_FILE="$STAGING/no.env" write_manifest >/dev/null 2>&1
    py=$(command -v python3 || command -v python || true)
    if [ -n "$py" ]; then
        if "$py" -c 'import json,sys; json.load(open(sys.argv[1]))' "$STAGING/MANIFEST.json" 2>/dev/null; then
            n=$((n + 1)); printf 'ok manifest_json_valid\n'
        else
            printf 'FAIL manifest_json_valid\n' >&2; exit 1
        fi
        if "$py" -c '
import hashlib, sys
base, lines = sys.argv[1], open(sys.argv[2]).read().splitlines()
for line in lines:
    want, rel = line.split(None, 1)      # tolerates "hash  f" and "hash *f"
    rel = rel.lstrip("*")
    got = hashlib.sha256(open(base + "/" + rel, "rb").read()).hexdigest()
    assert got == want, rel' "$STAGING" "$STAGING/MANIFEST.sha256" 2>/dev/null; then
            n=$((n + 1)); printf 'ok manifest_checksums_match\n'
        else
            printf 'FAIL manifest_checksums_match\n' >&2; exit 1
        fi
        if "$py" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert "inventory" in m and m["target"] == "daily" and m["retention"] == {"keep_last": 3}' "$STAGING/MANIFEST.json" 2>/dev/null; then
            n=$((n + 1)); printf 'ok manifest_records_inventory_and_policy\n'
        else
            printf 'FAIL manifest_records_inventory_and_policy\n' >&2; exit 1
        fi
    else
        printf 'skip manifest_json_valid / manifest_checksums_match (no python)\n'
    fi
    # The checksum file covers the two staged artifacts only — never config,
    # never .env.
    check manifest_covers_artifacts_only "2" "$(wc -l < "$STAGING/MANIFEST.sha256" | tr -d ' ')"
    stage_close; rmdir "$STAGING" 2>/dev/null || true

    # A failure message never contains the passphrase.
    RESTIC_PASSWORD="super-secret-value-42"
    local msg; msg=$(fail "restic backup failed for ${RESTIC_REPOSITORY:-repo}" 2>&1)
    case "$msg" in
        *"$RESTIC_PASSWORD"*) printf 'FAIL fail_leaks_secret\n' >&2; exit 1 ;;
        *) n=$((n + 1)); printf 'ok fail_leaks_no_secret\n' ;;
    esac
    FAILED=0

    # Step 5b — Google Drive free space. Grading on its own first (every branch),
    # then drive_check end to end with rclone replaced by a fixture.
    local G=$((1024 * 1024 * 1024))
    check drive_grade_ok "ok" "$(drive_grade $((20 * G)) $((2 * G)) "$G" $((10 * G)))"
    check drive_grade_low "low" "$(drive_grade $((5 * G)) $((2 * G)) "$G" $((10 * G)))"
    check drive_grade_insufficient "insufficient" "$(drive_grade $((2 * G)) $((2 * G)) "$G" $((10 * G)))"
    check drive_grade_unknown "unknown" "$(drive_grade "" $((2 * G)) "$G" $((10 * G)))"
    # The real `rclone about --json` shape (multi-line, Drive fields).
    local about_json='{
	"total": 16106127360,
	"used": 11274289152,
	"trashed": 1048576,
	"other": 2147483648,
	"free": 4831838208
}'
    check drive_field_free "4831838208" "$(drive_field "$about_json" free)"
    check drive_field_total "16106127360" "$(drive_field "$about_json" total)"
    check drive_field_missing "" "$(drive_field '{"total": 5}' free)"

    local saved_staging="$STAGING" free_fixture about_ok=1
    STAGING=$(mktemp -d); printf 'x%.0s' $(seq 1 2048) > "$STAGING/artifact"
    RESTIC_REPOSITORY="rclone:gdrive:backups"
    drive_about() {   # fixture: honours free_fixture / about_ok
        [ "$about_ok" = 1 ] || return 1
        printf '{"total": %s, "used": 1, "free": %s}' $((100 * G)) "$free_fixture"
    }
    drive_run() {   # <case> -> "<FAILED>|<status in DRIVE_JSON>"; resets state
        FAILED=0; LAST_ERROR=""; DRIVE_JSON="null"
        drive_check >/dev/null 2>&1
        local st; st=$(printf '%s' "$DRIVE_JSON" | sed -n 's/.*"status":"\([a-z]*\)".*/\1/p')
        printf '%s|%s' "$FAILED" "$st"
    }
    free_fixture=$((20 * G)); check drive_check_sufficient "0|ok" "$(drive_run)"
    free_fixture=$((3 * G));  check drive_check_low_warns_and_continues "0|low" "$(drive_run)"
    free_fixture=$((G / 2));  check drive_check_insufficient_fails "1|insufficient" "$(drive_run)"
    FAILED=0; LAST_ERROR=""; drive_check >/dev/null 2>&1
    case "$LAST_ERROR" in
        *"nothing uploaded"*"no backup deleted"*) n=$((n + 1)); printf 'ok drive_insufficient_message_names_the_fix\n' ;;
        *) printf 'FAIL drive_insufficient_message_names_the_fix: %s\n' "$LAST_ERROR" >&2; exit 1 ;;
    esac
    about_ok=0
    DRIVE_QUOTA_REQUIRED=0; check drive_check_unreadable_warns "0|unknown" "$(drive_run)"
    DRIVE_QUOTA_REQUIRED=1; check drive_check_unreadable_required_fails "1|unknown" "$(drive_run)"
    DRIVE_QUOTA_REQUIRED=0; about_ok=1; free_fixture=$((20 * G))
    # First run (repository just created): the uploaded files count as new data.
    REPO_CREATED=1; drive_run >/dev/null
    local need_first; need_first=$(drive_field "$DRIVE_JSON" need_bytes)
    REPO_CREATED=0; drive_run >/dev/null
    local need_later; need_later=$(drive_field "$DRIVE_JSON" need_bytes)
    check drive_need_counts_staged_artifacts "1" "$([ "${need_later:-0}" -ge 2048 ] && echo 1 || echo 0)"
    check drive_need_first_run_not_smaller "1" "$([ "${need_first:-0}" -ge "${need_later:-0}" ] && echo 1 || echo 0)"
    if [ -n "$py" ]; then
        free_fixture=$((3 * G)); drive_run >/dev/null
        if "$py" -c 'import json,sys; d=json.loads(sys.argv[1]); assert d["status"]=="low" and d["free_bytes"]>0' "$DRIVE_JSON" 2>/dev/null; then
            n=$((n + 1)); printf 'ok drive_json_valid_for_status_row\n'
        else
            printf 'FAIL drive_json_valid_for_status_row: %s\n' "$DRIVE_JSON" >&2; exit 1
        fi
    fi
    rm -rf "$STAGING"; STAGING="$saved_staging"; FAILED=0; LAST_ERROR=""; DRIVE_JSON="null"

    # The dashboard's Google Drive sign-in (ops/backup_drive.sh).
    if command -v python3 >/dev/null; then
        local pt; pt=$(mktemp -d)
        mkdir -p "$pt/run"
        printf '[gdrive]\ntype = drive\nclient_id = CID\nclient_secret = CS\nroot_folder_id = OLD\ntoken = {"refresh_token":"server"}\n' > "$pt/rclone.conf"
        BACKUP_DRIVE_OVERRIDE="$pt/primary.json"; RCLONE_BASE_CONFIG="$pt/rclone.conf"; DRIVE_RUN_DIR="$pt/run"
        unset RCLONE_CONFIG
        RESTIC_REPOSITORY="rclone:gdrive:backups"; drive_conf_setup >/dev/null 2>&1
        check drive_no_panel_signin_changes_nothing "server|unset" "$DRIVE_SOURCE|${RCLONE_CONFIG:-unset}"
        printf '{"root_folder_id":"NEW","account":"a@b.c","token":{"refresh_token":"panel"}}' > "$pt/primary.json"
        drive_conf_setup >/dev/null 2>&1
        check drive_panel_signin_used "panel|rclone:gdrive:backups" "$DRIVE_SOURCE|$RESTIC_REPOSITORY"
        check drive_config_has_panel_token "1|1" \
            "$(grep -c '"refresh_token":"panel"' "${RCLONE_CONFIG:-/dev/null}")|$(grep -c '^root_folder_id = NEW$' "${RCLONE_CONFIG:-/dev/null}")"
        # File modes only mean something on Linux (Git Bash on NTFS reports 644).
        if [ "$(uname -s)" = Linux ]; then
            check drive_config_private "600" "$(stat -c %a "$RCLONE_CONFIG")"
        fi
        local conf="$RCLONE_CONFIG"; drive_conf_close
        check drive_close_removes_tokens no "$([ -e "$conf" ] && echo yes || echo no)"
        check drive_server_config_untouched "1" "$(grep -c '"refresh_token":"server"' "$pt/rclone.conf")"
        unset RCLONE_CONFIG
        printf '{broken' > "$pt/primary.json"                        # unusable sign-in -> fall back
        drive_conf_setup >/dev/null 2>&1
        check drive_bad_signin_falls_back "server|unset|1" \
            "$DRIVE_SOURCE|${RCLONE_CONFIG:-unset}|$([ -n "$DRIVE_NOTE" ] && echo 1 || echo 0)"
        check drive_failure_leaves_no_temp "0" "$(find "$pt/run" -type f | wc -l | tr -d ' ')"
        rm -rf "$pt"; DRIVE_NOTE=""; DRIVE_SOURCE=server; unset RCLONE_CONFIG RESTIC_REPOSITORY
    else
        printf 'skip drive sign-in checks (no python3)\n'
    fi

    # Step 9 — the backup catalog (catalog.json): the run's record and the
    # Drive copy next to the repositories. catalog.py is offline; rclone and
    # the snapshot id list are fixtures here.
    if [ -n "$py" ] && [ -f "$(dirname "${BASH_SOURCE[0]}")/catalog.py" ]; then
        local ctd snap_ids st2 saved_cat saved_base saved_target
        ctd=$(mktemp -d); snap_ids="$ctd/ids"
        saved_cat="$BACKUP_CATALOG_FILE"; saved_base="${RESTIC_BASE:-}"; saved_target="$TARGET"
        check catalog_ids_parse "27053f1a 9f8e7d6c" \
            "$(printf '{"short_id":"27053f1a"},{"short_id":"9f8e7d6c"}\n' | catalog_ids_from_json | xargs)"
        st2="$STAGING"; STAGING=$(mktemp -d)
        mkdir -p "$STAGING/postgres" "$STAGING/qdrant"
        printf 'DUMPDATA' > "$STAGING/postgres/openwebui.dump"
        printf 'SNAPDATA' > "$STAGING/qdrant/documents.snapshot"
        ENV_FILE="$STAGING/no.env" write_manifest >/dev/null 2>&1
        catalog_ids() { cat "$snap_ids"; }
        rclone() { printf '%s\n' "$*" > "$ctd/rclone.args"; }
        RESTIC_BASE="rclone:gdrive:backups"; TARGET=daily; CAT_CHECK=ok
        SNAPSHOT_ID="27053f1a5116856d85f9545244f387ade5511e70a532720b777575927dd50c74"
        BACKUP_ADDED=73289021; BACKUP_CATALOG_FILE="$ctd/catalog.json"
        printf '%s\n' "27053f1a" > "$snap_ids"            # restic still lists it
        catalog_update "daily" >/dev/null 2>&1
        check catalog_records_the_run "verified|scheduled" \
            "$("$py" -c 'import json,sys; s=json.load(open(sys.argv[1]))["snapshots"][0]; print(s["status"] + "|" + s["kind"])' "$BACKUP_CATALOG_FILE" 2>/dev/null)"
        check catalog_upload_reaches_folder_root "gdrive:backups/catalog.json" \
            "$(awk '{print $NF}' "$ctd/rclone.args" 2>/dev/null)"
        printf '%s\n' "ffffffff" > "$snap_ids"            # retention removed it
        catalog_update "daily" >/dev/null 2>&1
        check catalog_prunes_removed_snapshots "pruned" \
            "$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshots"][0]["status"])' "$BACKUP_CATALOG_FILE" 2>/dev/null)"
        rm -rf "$ctd" "$STAGING"; STAGING="$st2"; CAT_CHECK=""
        BACKUP_CATALOG_FILE="$saved_cat"; RESTIC_BASE="$saved_base"; TARGET="$saved_target"
    else
        printf 'skip catalog checks (no python3 or catalog.py)\n'
    fi

    printf '\n%d gmac-backup self-checks passed\n' "$n"
}

# ── main ────────────────────────────────────────────────────

main() {
    local dry=0 tag="" extra_tag="" detail bytes
    SNAPSHOT_ID=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return 0 ;;
            --dry-run)   dry=1 ;;
            --tag)
                [ -n "${2:-}" ] || { echo "usage: $0 --tag <name>" >&2; return 2; }
                tag="milestone"; extra_tag="$2"; shift ;;
            --target)
                [ -n "${2:-}" ] || { echo "usage: $0 --target daily|weekly|monthly" >&2; return 2; }
                TARGET="$2"; shift ;;
            *) echo "usage: $0 [--dry-run] [--target daily|weekly|monthly] [--tag <name>] [--self-test]" >&2
               return 2 ;;
        esac
        shift
    done
    case "$TARGET" in
        daily|weekly|monthly) ;;
        *) echo "unknown backup target '$TARGET' (daily, weekly or monthly)" >&2; return 2 ;;
    esac
    # A scheduled weekly/monthly run the admin switched off ends here — before
    # the lock and before a status row, so the panel's "last run" keeps
    # showing the last real backup instead of a row per skipped timer. Exit 0:
    # an intended skip must not turn the systemd unit red. A milestone run
    # (--tag) is always an explicit human request, so it ignores the switch.
    if [ -z "$tag" ] && ! target_enabled "$TARGET"; then
        log "$TARGET backups are turned off in the dashboard — skipped"
        return 0
    fi
    tag="${tag:-$TARGET}"

    # One run at a time: a second start must not corrupt staging or double-push.
    exec 9>"$LOCK_FILE"
    flock -n 9 || { fail "another backup run holds $LOCK_FILE — refusing to run"; return 1; }

    for cmd in docker flock sha256sum stat; do
        command -v "$cmd" >/dev/null || { fail "required tool missing: $cmd"; return 1; }
    done
    if [ "$dry" = 0 ]; then
        for cmd in restic rclone; do
            command -v "$cmd" >/dev/null \
                || { fail "required tool missing: $cmd (README.md (Backups))"; return 1; }
        done
    fi
    for container in "$POSTGRES" "$HAYSTACK"; do
        [ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null)" = "true" ] \
            || { fail "container $container is not running"; return 1; }
    done

    if [ "$dry" = 1 ]; then
        # Dry run: build + verify locally. No restic, no status row — like
        # storage_monitor.sh --dry-run it leaves no trace anywhere.
        log "DRY RUN — nothing will be uploaded"
        stage_open && backup_postgres && backup_qdrant && write_manifest && verify_stage
        log "DRY RUN staged: $(find "$STAGING" -type f | wc -l | tr -d ' ') files"
        stage_close
        [ "$FAILED" = 0 ] || return 1
        log "DRY RUN complete — staged artifacts verified, nothing uploaded"
        return 0
    fi

    load_secrets
    restic_preflight || return 1
    status_start

    stage_open && backup_postgres && backup_qdrant && write_manifest && verify_stage
    if [ "$FAILED" = 0 ]; then
        drive_check
    fi
    if [ "$FAILED" = 0 ]; then
        restic_backup "$tag${extra_tag:+:$extra_tag}"
    fi
    if [ "$FAILED" = 0 ]; then
        restic_check
    fi
    if [ "$FAILED" = 0 ] && [ "$tag" = "$TARGET" ]; then
        if [ "$SKIP_RETENTION" = "1" ]; then
            log "BACKUP_SKIP_RETENTION=1 — retention NOT applied (first-run safety)"
        else
            restic_retention
        fi
    elif [ "$tag" = "milestone" ] && [ "$FAILED" = 0 ]; then
        log "milestone snapshot — retention policy not applied to it"
    fi

    # Step 9 — the catalog: what this recovery point holds. Every run that
    # made a snapshot records it (a failed check as "failed-check"), and the
    # id list above is read AFTER retention, so removed snapshots are marked
    # pruned. Never fatal: the snapshot itself is already safe.
    catalog_update "$tag${extra_tag:+:$extra_tag}"

    bytes=$(stat -c%s "$STAGING/postgres/$PG_DB.dump" 2>/dev/null || echo "")
    detail=$(printf '{"stage_dump_bytes":%s,"collections":%d,"retention":{"keep_last":%d,"target":%s},"milestone":%s,"drive":%s,"drive_signin":{"source":%s,"note":%s}}' \
        "$(json_val "$bytes")" "$(printf '%s\n' $COLLECTIONS | wc -l | tr -d ' ')" \
        "$(keep_last_for "$TARGET")" "$(json_str "$TARGET")" \
        "$([ "$tag" = "milestone" ] && echo true || echo false)" "$DRIVE_JSON" \
        "$(json_str "$DRIVE_SOURCE")" "$(json_val "$DRIVE_NOTE")")
    if [ "$FAILED" = 0 ]; then
        status_finish true "${SNAPSHOT_ID:-}" "${BACKUP_ADDED:-}" "" "$detail"
        log "backup complete: snapshot ${SNAPSHOT_ID:-unknown}"
    else
        # First failing step, as the panel shows it. The journal has everything.
        status_finish false "${SNAPSHOT_ID:-}" "" "${LAST_ERROR:-run failed}" "$detail"
        log "backup FAILED — see the ERROR lines above"
    fi
    stage_close
    [ "$FAILED" = 0 ] || return 1
}

trap 'stage_close; drive_conf_close' EXIT
main "$@"
