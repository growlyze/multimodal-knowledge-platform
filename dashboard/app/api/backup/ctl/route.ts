import { randomBytes } from 'node:crypto';
import { promises as fs } from 'node:fs';
import path from 'node:path';

import { NextRequest } from 'next/server';

import { buildBackupRequest, readSchedule, validateScheduleChange } from '@/lib/backup';
import { guard, noStore } from '@/lib/session';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

/**
 * /kb/api/backup/ctl - the dashboard side of the backup control channel
 * (lib/backup.ts documents the files; ops/repoctl.py is the host side).
 *
 *   GET   -> {available, pending, status, schedule, catalog}
 *   POST  {action: "request", cmd, repo, snapshot, ...}   write request.json
 *   POST  {action: "schedule", target, enabled}           write schedule.json
 *
 * POST returns at once with the request id; the page polls GET until
 * status.json carries that id in a terminal state. Requests are refused while
 * a previous one is still waiting (two clicks never queue two restores).
 */

const DEFAULT_CTL_DIR = '/data/backup-drives/ctl';

function ctlDir(): string {
  return path.resolve(process.env.BACKUP_CTL_DIR?.trim() || DEFAULT_CTL_DIR);
}

const file = (name: string) => path.join(ctlDir(), name);

async function readJson<T>(name: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(file(name), 'utf8')) as T;
  } catch {
    return null;
  }
}

async function exists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

/** Atomic, private write: the host never reads half a file. */
async function writeJson(name: string, value: unknown): Promise<void> {
  await fs.mkdir(ctlDir(), { recursive: true, mode: 0o700 });
  const tmp = `${file(name)}.${randomBytes(4).toString('hex')}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(value), { mode: 0o600, flag: 'wx' });
  await fs.rename(tmp, file(name));
}

async function ctlAvailable(): Promise<boolean> {
  try {
    return (await fs.stat(ctlDir())).isDirectory();
  } catch {
    return false;
  }
}

export async function GET(request: NextRequest) {
  const { denied } = await guard(request);
  if (denied) return denied;

  const available = await ctlAvailable();
  if (!available) {
    return noStore({
      available: false,
      reason: `The backup control directory is not mounted (${ctlDir()}).`,
      pending: false,
      status: null,
      schedule: readSchedule(null),
      catalog: null,
    });
  }
  const pendingRequest = await readJson<Record<string, unknown>>('request.json');
  const catalog = await readJson<Record<string, unknown>>('catalog.json');
  return noStore({
    available: true,
    pending: (await exists(file('request.json'))) ? { id: pendingRequest?.id ?? null, cmd: pendingRequest?.cmd ?? null } : false,
    status: await readJson('status.json'),
    schedule: readSchedule(await readJson('schedule.json')),
    catalog: catalog && Array.isArray(catalog.snapshots) ? catalog : null,
  });
}

export async function POST(request: NextRequest) {
  const { session, denied } = await guard(request);
  if (denied) return denied;

  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return noStore({ error: 'Request body must be JSON', code: 'bad_request' }, 400);
  }
  const body = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>;

  if (!(await ctlAvailable())) {
    return noStore(
      {
        error: `The backup control directory is not mounted (${ctlDir()}). Backups are not set up on this deployment.`,
        code: 'not_configured',
      },
      503,
    );
  }

  if (body.action === 'schedule') {
    const change = validateScheduleChange(body);
    if (!change.ok) return noStore({ error: change.error, code: 'bad_request' }, 400);
    const current = readSchedule(await readJson('schedule.json'));
    const next = {
      ...current,
      [change.value.target]: change.value.enabled,
      updated_at: new Date().toISOString(),
      by: session.sub.slice(0, 100),
    };
    await writeJson('schedule.json', next);
    return noStore({ ok: true, schedule: readSchedule(next) });
  }

  if (body.action !== undefined && body.action !== 'request') {
    return noStore({ error: 'Unknown action', code: 'bad_request' }, 400);
  }

  const built = buildBackupRequest(body, session.sub);
  if (!built.ok) return noStore({ error: built.error, code: 'bad_request' }, 400);

  if (await exists(file('request.json'))) {
    return noStore({ error: 'A backup request is already waiting to run - wait for it to finish.', code: 'busy' }, 409);
  }
  await writeJson('request.json', built.value);
  return noStore({ ok: true, id: built.value.id, cmd: built.value.cmd }, 202);
}
