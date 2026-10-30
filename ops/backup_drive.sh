# backup_drive.sh — sourced by ops/backup.sh and ops/restore-drill.sh.
#
# The backup goes to ONE Google Drive: the server's rclone remote (`gdrive`).
# When an admin has connected it again from the panel's "Backup drive" page,
# the sign-in is in $BACKUP_DRIVE_OVERRIDE; drive_conf_setup applies it to a
# temporary copy of the rclone config (ops/backup_drives.py) and points rclone
# at that copy. No panel sign-in -> nothing changes. The repository name never
# changes. The copy holds OAuth tokens: 0600 under /run, removed by
# drive_conf_close (both scripts call it on exit).
#
# Needs: log, fail (the caller's); python3 only when a panel sign-in exists.

BACKUP_DRIVE_OVERRIDE=${BACKUP_DRIVE_OVERRIDE:-/data/backup-drives/primary.json}
RCLONE_BASE_CONFIG=${RCLONE_BASE_CONFIG:-/root/.config/rclone/rclone.conf}
DRIVES_HELPER=${DRIVES_HELPER:-$(dirname "${BASH_SOURCE[0]}")/backup_drives.py}
DRIVE_RUN_DIR=${DRIVE_RUN_DIR:-/run}
DRIVE_CONF=""
DRIVE_SOURCE="server"   # "panel" when the panel's sign-in is in use
DRIVE_NOTE=""           # set when the panel's sign-in could not be applied

drive_conf_setup() {
    DRIVE_SOURCE="server"; DRIVE_NOTE=""
    local rest remote out
    rest="${RESTIC_REPOSITORY#rclone:}"; remote="${rest%%:*}"
    [ -n "$remote" ] && [ "$remote" != "$rest" ] || return 0
    if [ ! -f "$BACKUP_DRIVE_OVERRIDE" ]; then
        # Signed out on the handler page: NOT connected. Never fall back to the
        # server's own remote -- that would keep backing up to the previous
        # account while the page says disconnected.
        if [ -f "$(dirname "$BACKUP_DRIVE_OVERRIDE")/signed-out.json" ]; then
            DRIVE_SOURCE="disconnected"
        fi
        return 0
    fi
    if ! command -v python3 >/dev/null || [ ! -f "$DRIVES_HELPER" ]; then
        DRIVE_NOTE="a panel sign-in exists but python3/$DRIVES_HELPER is missing"
        log "WARNING: drive: $DRIVE_NOTE — using the server's rclone config"
        return 0
    fi
    DRIVE_CONF=$(mktemp "$DRIVE_RUN_DIR/gmac-rclone.XXXXXX") \
        || { DRIVE_NOTE="cannot create a temp config in $DRIVE_RUN_DIR"; log "WARNING: drive: $DRIVE_NOTE"; return 0; }
    chmod 600 "$DRIVE_CONF"
    if ! out=$(python3 "$DRIVES_HELPER" build --base "$RCLONE_BASE_CONFIG" \
                   --override "$BACKUP_DRIVE_OVERRIDE" --remote "$remote" --out "$DRIVE_CONF"); then
        drive_conf_close
        DRIVE_NOTE="the panel's Google Drive sign-in could not be applied (see 'backup_drives:' lines)"
        log "WARNING: drive: $DRIVE_NOTE — using the server's rclone config"
        return 0
    fi
    case "$out" in
        "override "*)
            export RCLONE_CONFIG="$DRIVE_CONF"; DRIVE_SOURCE="panel"
            log "drive: using the Google Drive connected in the dashboard (${out#override })" ;;
        *) drive_conf_close ;;
    esac
}

drive_conf_close() {
    if [ -n "$DRIVE_CONF" ]; then
        case "$DRIVE_CONF" in "$DRIVE_RUN_DIR"/gmac-rclone.*) rm -f -- "$DRIVE_CONF" ;; esac
    fi
    DRIVE_CONF=""
}
