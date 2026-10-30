import { describe, expect, it } from 'vitest';

import {
  BACKUP_COMMANDS,
  FULL_RESTORE_PHRASE,
  buildBackupRequest,
  isBackupCommand,
  readSchedule,
  safeRestorePath,
  validSnapshotId,
  validateScheduleChange,
} from '@/lib/backup';

const SNAP = '1a2b3c4d';

function valid(cmd: string): Record<string, unknown> {
  switch (cmd) {
    case 'restore-full':
      return { cmd, snapshot: SNAP, confirm: FULL_RESTORE_PHRASE };
    case 'restore-files':
      return { cmd, snapshot: SNAP, paths: ['/data/haystack-uploads/a.pdf'] };
    case 'restore-docs':
      return { cmd, snapshot: SNAP, docs: [{ collection: 'documents', name: 'a.pdf' }] };
    case 'restore-chats':
      return { cmd, snapshot: SNAP, chats: ['chat-1'] };
    case 'restore-videos':
      return { cmd, snapshot: SNAP, videos: ['vid-1'] };
    case 'backup-now':
    case 'snapshots':
    case 'rebuild-catalog':
      return { cmd };
    default:
      return { cmd, snapshot: SNAP };
  }
}

describe('backup request builder (mirrors ops/repoctl.py load_request)', () => {
  it('accepts every host command with a valid body', () => {
    for (const cmd of BACKUP_COMMANDS) {
      const result = buildBackupRequest(valid(cmd), 'admin');
      expect(result.ok, cmd).toBe(true);
      if (result.ok) {
        expect(result.value.cmd).toBe(cmd);
        expect(result.value.id).toMatch(/^r-[0-9a-f]{12}$/);
        expect(result.value.repo).toBe('daily');
        expect(result.value.by).toBe('admin');
        expect(result.value.requested_at).toMatch(/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$/);
      }
    }
  });

  it('refuses unknown commands and non-object bodies', () => {
    for (const cmd of ['backup', 'check', 'trigger', 'rm -rf', 'SNAPSHOTS', '', null, 42]) {
      expect(buildBackupRequest({ cmd }, 'admin').ok, String(cmd)).toBe(false);
      expect(isBackupCommand(cmd)).toBe(false);
    }
    expect(buildBackupRequest(null, 'admin').ok).toBe(false);
    expect(buildBackupRequest(['snapshots'], 'admin').ok).toBe(false);
  });

  it('only allows the three backup folders', () => {
    expect(buildBackupRequest({ cmd: 'snapshots', repo: 'weekly' }, 'a').ok).toBe(true);
    expect(buildBackupRequest({ cmd: 'snapshots', repo: 'yearly' }, 'a').ok).toBe(false);
  });

  it('requires a recovery point id for reads and restores', () => {
    expect(buildBackupRequest({ cmd: 'inspect' }, 'a').ok).toBe(false);
    expect(buildBackupRequest({ cmd: 'inspect', snapshot: '--help' }, 'a').ok).toBe(false);
    expect(buildBackupRequest({ cmd: 'inspect', snapshot: '../x' }, 'a').ok).toBe(false);
    expect(validSnapshotId('latest')).toBe(true);
    expect(validSnapshotId('abc')).toBe(false);
  });

  it('restore-full needs the exact typed phrase and a concrete snapshot', () => {
    expect(buildBackupRequest({ cmd: 'restore-full', snapshot: SNAP }, 'a').ok).toBe(false);
    expect(buildBackupRequest({ cmd: 'restore-full', snapshot: SNAP, confirm: 'restore' }, 'a').ok).toBe(false);
    expect(buildBackupRequest({ cmd: 'restore-full', snapshot: SNAP, confirm: 'RESTORE ' }, 'a').ok).toBe(false);
    expect(buildBackupRequest({ cmd: 'restore-full', snapshot: 'latest', confirm: 'RESTORE' }, 'a').ok).toBe(false);
    const ok = buildBackupRequest({ cmd: 'restore-full', snapshot: SNAP, confirm: 'RESTORE', safety_backup: false }, 'a');
    expect(ok.ok && ok.value.safety_backup).toBe(false);
    const dflt = buildBackupRequest({ cmd: 'restore-full', snapshot: SNAP, confirm: 'RESTORE' }, 'a');
    expect(dflt.ok && dflt.value.safety_backup).toBe(true);
  });

  it('restore-files only accepts knowledge-base paths', () => {
    expect(safeRestorePath('/data/haystack-uploads/x/y.pdf')).toBe(true);
    expect(safeRestorePath('/data/haystack-uploads/../../etc/shadow')).toBe(false);
    expect(safeRestorePath('/etc/passwd')).toBe(false);
    expect(safeRestorePath('relative/path')).toBe(false);
    expect(buildBackupRequest({ cmd: 'restore-files', snapshot: SNAP, paths: ['/etc/passwd'] }, 'a').ok).toBe(false);
    expect(buildBackupRequest({ cmd: 'restore-files', snapshot: SNAP, paths: [] }, 'a').ok).toBe(false);
  });

  it('restore-docs / chats / videos need a selection', () => {
    expect(buildBackupRequest({ cmd: 'restore-docs', snapshot: SNAP, docs: [] }, 'a').ok).toBe(false);
    expect(buildBackupRequest({ cmd: 'restore-docs', snapshot: SNAP, docs: [{ name: 'x' }] }, 'a').ok).toBe(false);
    expect(buildBackupRequest({ cmd: 'restore-chats', snapshot: SNAP, chats: [] }, 'a').ok).toBe(false);
    expect(buildBackupRequest({ cmd: 'restore-videos', snapshot: SNAP, videos: ['-flag'] }, 'a').ok).toBe(false);
    expect(buildBackupRequest({ cmd: 'restore-videos', snapshot: SNAP, videos: ['a\nb'] }, 'a').ok).toBe(false);
  });
});

describe('schedule switches (schedule.json, read by ops/backup.sh)', () => {
  it('reads anything but a literal false as ON', () => {
    expect(readSchedule(null)).toEqual({ weekly: true, monthly: true });
    expect(readSchedule({ weekly: false })).toEqual({ weekly: false, monthly: true });
    expect(readSchedule({ weekly: 'false', monthly: 0 })).toEqual({ weekly: true, monthly: true });
  });

  it('only weekly/monthly can be switched; daily is always on', () => {
    expect(validateScheduleChange({ target: 'weekly', enabled: false }).ok).toBe(true);
    expect(validateScheduleChange({ target: 'monthly', enabled: true }).ok).toBe(true);
    expect(validateScheduleChange({ target: 'daily', enabled: false }).ok).toBe(false);
    expect(validateScheduleChange({ target: 'weekly', enabled: 'no' }).ok).toBe(false);
  });
});
