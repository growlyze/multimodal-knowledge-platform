/**
 * Backup & restore control channel - the dashboard side of the one-request
 * file protocol served on the host by ops/repoctl.py (started by
 * ops/repoctl.path when request.json appears in BACKUP_CTL_DIR):
 *
 *   request.json   dashboard -> host  {id, cmd, repo, snapshot, paths, docs,
 *                                      chats?, videos?, confirm?, safety_backup?,
 *                                      requested_at, by}
 *   status.json    host -> dashboard  {id, cmd, state, step, at, result, error,
 *                                      progress, by, requested_at}
 *   schedule.json  dashboard -> host  {weekly: bool, monthly: bool, updated_at, by}
 *                                     (read by ops/backup.sh; daily is always on)
 *   catalog.json   host -> dashboard  {snapshots: [...]} - the recovery points
 *
 * The dashboard never holds the repository secrets. Everything here mirrors
 * the host-side validation (ops/repoctl.py load_request): the host checks the
 * same rules again and is the final authority.
 */

export const BACKUP_COMMANDS = [
  'backup-now',
  'snapshots',
  'inspect',
  'restore-files',
  'find-deleted',
  'restore-docs',
  'restore-full',
  'rebuild-catalog',
  'find-deleted-chats',
  'restore-chats',
  'find-deleted-videos',
  'restore-videos',
] as const;
export type BackupCommand = (typeof BACKUP_COMMANDS)[number];

/** The three restic repositories (subfolders of the backup folder). */
export const BACKUP_REPOS = ['daily', 'weekly', 'monthly'] as const;
export type BackupRepo = (typeof BACKUP_REPOS)[number];

/** Weekly/monthly can be switched off; daily never (it is the only recent copy). */
export const SCHEDULE_TARGETS = ['weekly', 'monthly'] as const;
export type ScheduleTarget = (typeof SCHEDULE_TARGETS)[number];

/** Typed confirmation for a full (disaster-recovery) restore, checked here AND host-side. */
export const FULL_RESTORE_PHRASE = 'RESTORE';

/** File roots a single-file restore may write into (ops/repoctl.py RESTORE_ROOTS). */
export const RESTORE_ROOTS = ['/data/haystack-uploads'] as const;

const NEEDS_SNAPSHOT = new Set<BackupCommand>([
  'inspect',
  'restore-files',
  'find-deleted',
  'restore-docs',
  'restore-full',
  'find-deleted-chats',
  'restore-chats',
  'find-deleted-videos',
  'restore-videos',
]);

export const COMMAND_LABELS: Record<BackupCommand, string> = {
  'backup-now': 'Back up now',
  snapshots: 'List backups',
  inspect: 'Check backup',
  'restore-files': 'Restore files',
  'find-deleted': 'Find deleted documents',
  'restore-docs': 'Restore documents',
  'restore-full': 'Full restore',
  'rebuild-catalog': 'Rebuild the backup list',
  'find-deleted-chats': 'Find deleted chats',
  'restore-chats': 'Restore chats',
  'find-deleted-videos': 'Find deleted videos',
  'restore-videos': 'Restore videos',
};

export interface BackupDoc {
  collection: string;
  name: string;
}

export interface BackupRequest {
  id: string;
  cmd: BackupCommand;
  repo: BackupRepo;
  snapshot: string;
  paths: string[];
  docs: BackupDoc[];
  chats?: string[];
  videos?: string[];
  confirm?: string;
  safety_backup?: boolean;
  requested_at: string;
  by: string;
}

export type ValidationResult<T> = { ok: true; value: T } | { ok: false; error: string };

export function isBackupCommand(value: unknown): value is BackupCommand {
  return typeof value === 'string' && (BACKUP_COMMANDS as readonly string[]).includes(value);
}

/** A restic short id (or 'latest') - never a flag, never a path. */
export function validSnapshotId(value: unknown): value is string {
  return typeof value === 'string' && (value === 'latest' || /^[A-Za-z0-9]{6,64}$/.test(value));
}

/** A chat / video id: a bounded opaque string, never a flag or a line. */
export function validItemId(value: unknown): value is string {
  return (
    typeof value === 'string' && value.length > 0 && value.length <= 300 && !/[\r\n]/.test(value) && !value.startsWith('-')
  );
}

/** An absolute file path inside one of the restore roots, no "..". */
export function safeRestorePath(value: unknown): value is string {
  if (typeof value !== 'string' || !value || /[\r\n\0]/.test(value) || !value.startsWith('/')) return false;
  if (value.split('/').includes('..')) return false;
  return RESTORE_ROOTS.some((root) => value === root || value.startsWith(`${root}/`));
}

export function newRequestId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(6));
  return `r-${Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')}`;
}

function nowIso(now: Date): string {
  return now.toISOString().replace(/\.\d{3}Z$/, '+00:00');
}

/**
 * Builds the request record from an untrusted body. Refuses unknown commands,
 * folders and ids; restore-full needs the exact typed phrase and a concrete
 * recovery point (never 'latest': the safety backup taken first would itself
 * become 'latest').
 */
export function buildBackupRequest(input: unknown, by: string, now = new Date()): ValidationResult<BackupRequest> {
  if (typeof input !== 'object' || input === null || Array.isArray(input)) {
    return { ok: false, error: 'Request body must be a JSON object' };
  }
  const body = input as Record<string, unknown>;
  const cmd = body.cmd;
  if (!isBackupCommand(cmd)) {
    return { ok: false, error: `Unknown backup command ${JSON.stringify(cmd ?? null)}` };
  }
  const repo = body.repo === undefined || body.repo === '' ? 'daily' : body.repo;
  if (!(BACKUP_REPOS as readonly unknown[]).includes(repo)) {
    return { ok: false, error: 'Unknown backup folder (daily, weekly or monthly)' };
  }
  const snapshot = typeof body.snapshot === 'string' ? body.snapshot.trim() : '';
  if (NEEDS_SNAPSHOT.has(cmd) && !validSnapshotId(snapshot)) {
    return { ok: false, error: 'Choose a backup (recovery point) first' };
  }

  const request: BackupRequest = {
    id: newRequestId(),
    cmd,
    repo: repo as BackupRepo,
    snapshot,
    paths: [],
    docs: [],
    requested_at: nowIso(now),
    by: by.slice(0, 100),
  };

  if (cmd === 'restore-files') {
    const paths = Array.isArray(body.paths) ? body.paths : [];
    if (paths.length === 0) return { ok: false, error: 'Select at least one file to restore' };
    if (paths.length > 2000) return { ok: false, error: 'Too many files in one request' };
    for (const p of paths) {
      if (!safeRestorePath(p)) {
        return { ok: false, error: `Refusing ${JSON.stringify(String(p).slice(0, 120))}: only knowledge-base files can be restored` };
      }
    }
    request.paths = paths as string[];
  }

  if (cmd === 'restore-docs') {
    const docs = Array.isArray(body.docs) ? body.docs : [];
    const clean: BackupDoc[] = [];
    for (const d of docs) {
      const rec = (typeof d === 'object' && d !== null ? d : {}) as Record<string, unknown>;
      if (typeof rec.collection !== 'string' || !rec.collection || typeof rec.name !== 'string' || !rec.name) {
        return { ok: false, error: 'Each document needs a collection and a name' };
      }
      clean.push({ collection: rec.collection, name: rec.name });
    }
    if (clean.length === 0) return { ok: false, error: 'Select at least one deleted document to restore' };
    request.docs = clean;
  }

  if (cmd === 'restore-chats' || cmd === 'restore-videos') {
    const key = cmd === 'restore-chats' ? 'chats' : 'videos';
    const items = Array.isArray(body[key]) ? (body[key] as unknown[]) : [];
    if (items.length === 0 || items.length > 500 || !items.every(validItemId)) {
      return { ok: false, error: `Select at least one deleted ${key.slice(0, -1)} to restore` };
    }
    request[key] = items as string[];
  }

  if (cmd === 'restore-full') {
    // Exact match, no trim/upper-casing: typing it is the decision.
    if (body.confirm !== FULL_RESTORE_PHRASE) {
      return { ok: false, error: `Type ${FULL_RESTORE_PHRASE} exactly to start a full restore` };
    }
    if (snapshot === 'latest') return { ok: false, error: "A full restore needs a specific backup, not 'latest'" };
    request.confirm = FULL_RESTORE_PHRASE;
    request.safety_backup = body.safety_backup !== false;
  }

  return { ok: true, value: request };
}

export interface ScheduleState {
  weekly: boolean;
  monthly: boolean;
}

/** Anything but a literal JSON false reads as ON (ops/backup.sh applies the same rule). */
export function readSchedule(raw: unknown): ScheduleState {
  const rec = (typeof raw === 'object' && raw !== null && !Array.isArray(raw) ? raw : {}) as Record<string, unknown>;
  return { weekly: rec.weekly !== false, monthly: rec.monthly !== false };
}

export function validateScheduleChange(input: unknown): ValidationResult<{ target: ScheduleTarget; enabled: boolean }> {
  if (typeof input !== 'object' || input === null || Array.isArray(input)) {
    return { ok: false, error: 'Body must be a JSON object' };
  }
  const { target, enabled } = input as Record<string, unknown>;
  if (target === 'daily') return { ok: false, error: 'Daily backups are always on' };
  if (!(SCHEDULE_TARGETS as readonly unknown[]).includes(target)) {
    return { ok: false, error: 'Unknown backup folder (weekly or monthly)' };
  }
  if (typeof enabled !== 'boolean') return { ok: false, error: 'enabled must be true or false' };
  return { ok: true, value: { target: target as ScheduleTarget, enabled } };
}
