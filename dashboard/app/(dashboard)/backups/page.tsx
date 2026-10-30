'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

import { ConfirmDialog } from '@/components/ConfirmDialog';
import type { DataTableColumn } from '@/components/DataTable';
import { DataTable } from '@/components/DataTable';
import { EmptyState, ErrorState, SkeletonRows, UnavailableNotice } from '@/components/Feedback';
import { StatusBanner } from '@/components/StatusBanner';
import { useToast } from '@/components/Toast';
import { IconBackups, IconRefresh, IconSearch, IconWarning } from '@/components/icons';
import { Badge, Button, Field, Input, Panel, Select, statusTone } from '@/components/ui';
import { ApiClientError, apiFetch, apiSend, attempt } from '@/lib/client';
import { BACKUP_REPOS, COMMAND_LABELS, FULL_RESTORE_PHRASE, type BackupCommand, type BackupRepo } from '@/lib/backup';
import { formatBytes, formatDateTime, relativeTime } from '@/lib/format';
import { useAsyncData, usePolling } from '@/lib/hooks';
import { isRecord, pickArray, pickBoolean, pickNumber, pickString, type AnyRecord } from '@/lib/normalize';
import { SectionTitle } from '@/components/SectionTitle';

type RestoreKind = 'files' | 'docs' | 'chats' | 'videos' | 'full';

const FIND_COMMAND: Record<Exclude<RestoreKind, 'full'>, BackupCommand> = {
  files: 'inspect',
  docs: 'find-deleted',
  chats: 'find-deleted-chats',
  videos: 'find-deleted-videos',
};

const RESTORE_LABELS: Record<RestoreKind, string> = {
  files: 'Knowledge-base files',
  docs: 'Deleted documents',
  chats: 'Deleted chats',
  videos: 'Deleted videos',
  full: 'Everything (disaster recovery)',
};

interface SentRequest {
  id: string;
  cmd: BackupCommand;
}

export default function BackupsPage() {
  const toast = useToast();
  const health = useAsyncData(() => attempt(apiFetch('proxy/admin/backup')), []);
  const ctl = useAsyncData(() => attempt(apiFetch<AnyRecord>('backup/ctl')), []);

  const [sent, setSent] = useState<SentRequest | null>(null);
  const [results, setResults] = useState<Record<string, unknown>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [backupRepo, setBackupRepo] = useState<BackupRepo>('daily');

  const value = ctl.data?.ok ? ctl.data.value : null;
  const available = value ? pickBoolean(value, ['available'], false) : false;
  const status = value && isRecord(value.status) ? value.status : null;
  const pending = Boolean(value && value.pending);
  const state = pickString(status, ['state']);
  const running = pending || state === 'running' || state === 'queued';

  // Poll fast while our request (or any job) is in flight.
  usePolling(ctl.reload, running || sent ? 3_000 : 30_000);

  // Collect the result of the request this page sent, once the host finishes it.
  useEffect(() => {
    if (!sent || !status || pickString(status, ['id']) !== sent.id) return;
    if (state === 'done' || state === 'failed' || state === 'busy') {
      if (state === 'done') {
        setResults((current) => ({ ...current, [sent.cmd]: status.result }));
        toast.push('success', `${COMMAND_LABELS[sent.cmd]}: finished`);
        if (sent.cmd === 'backup-now' || sent.cmd === 'rebuild-catalog') health.reload();
      } else {
        toast.push('error', `${COMMAND_LABELS[sent.cmd]}: ${state}`, pickString(status, ['error']) || undefined);
      }
      setSent(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, sent, state]);

  const send = useCallback(
    async (body: Record<string, unknown>) => {
      setBusy(String(body.cmd));
      try {
        const res = await apiSend<{ id: string; cmd: BackupCommand }>('POST', 'backup/ctl', { action: 'request', ...body });
        setSent({ id: res.id, cmd: res.cmd });
        toast.push('info', `${COMMAND_LABELS[res.cmd]}: sent`, 'The host runs it within seconds; progress shows below.');
        ctl.reload();
        return true;
      } catch (err) {
        toast.push('error', 'Request refused', err instanceof ApiClientError ? err.message : undefined);
        return false;
      } finally {
        setBusy(null);
      }
    },
    [ctl, toast],
  );

  const toggleSchedule = async (target: 'weekly' | 'monthly', enabled: boolean) => {
    try {
      await apiSend('POST', 'backup/ctl', { action: 'schedule', target, enabled });
      toast.push('success', `${target} backups ${enabled ? 'on' : 'off'}`);
      ctl.reload();
    } catch (err) {
      toast.push('error', 'Could not update the schedule', err instanceof ApiClientError ? err.message : undefined);
    }
  };

  const schedule = value && isRecord(value.schedule) ? value.schedule : { weekly: true, monthly: true };

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <SectionTitle>Backups &amp; Restore</SectionTitle>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Encrypted restic backups to Google Drive (daily, weekly and monthly folders). Requests run on the host through
            ops/repoctl.py; the dashboard never holds the backup secrets.
          </p>
        </div>
        <Button variant="secondary" onClick={() => { health.reload(); ctl.reload(); }}>
          <IconRefresh className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {health.loading && !health.data ? (
        <SkeletonRows rows={1} columns={2} />
      ) : health.data?.ok === false && !health.data.error.notAvailable ? (
        <ErrorState error={health.data.error.message} onRetry={health.reload} title="Backup health unavailable" />
      ) : (
        <BackupHealthBanner backup={health.data?.ok ? health.data.value : null} />
      )}

      {ctl.loading && !ctl.data ? (
        <SkeletonRows rows={2} columns={2} />
      ) : !available ? (
        <UnavailableNotice
          feature="Backup control channel"
          detail={
            pickString(value, ['reason']) ||
            'The backup control directory is not mounted. Mount /data/backup-drives into the dashboard container and install ops/repoctl.path on the host.'
          }
        />
      ) : null}

      <JobPanel status={status} pending={pending} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Back up now" description="Runs ops/backup.sh on the host (its own lock: never overlaps a scheduled run)">
          <div className="flex flex-wrap items-end gap-2">
            <Field label="Into folder" htmlFor="backup-repo" className="min-w-[160px]">
              <Select id="backup-repo" value={backupRepo} onChange={(e) => setBackupRepo(e.target.value as BackupRepo)}>
                {BACKUP_REPOS.map((repo) => (
                  <option key={repo} value={repo}>
                    {repo}
                  </option>
                ))}
              </Select>
            </Field>
            <Button onClick={() => send({ cmd: 'backup-now', repo: backupRepo })} loading={busy === 'backup-now'} disabled={!available || running}>
              <IconBackups className="h-4 w-4" />
              Back up now
            </Button>
          </div>
        </Panel>

        <Panel title="Schedule" description="Daily runs at 03:00, 11:00 and 19:00 UTC and cannot be switched off">
          <div className="space-y-2 text-sm text-slate-700 dark:text-slate-200">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked disabled className="h-4 w-4 rounded border-slate-300" />
              Daily (newest 3 kept)
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={schedule.weekly !== false}
                disabled={!available}
                onChange={(e) => toggleSchedule('weekly', e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              Weekly - Fridays 20:00 UTC (newest 2 kept)
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={schedule.monthly !== false}
                disabled={!available}
                onChange={(e) => toggleSchedule('monthly', e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              Monthly - the 1st (newest 1 kept)
            </label>
            <p className="text-xs text-slate-500 dark:text-slate-400">The timers themselves are systemd units on the host (backup*.timer).</p>
          </div>
        </Panel>
      </div>

      <RecoveryPoints
        catalog={value && isRecord(value.catalog) ? value.catalog : null}
        listed={results.snapshots}
        available={available}
        running={running}
        busy={busy}
        send={send}
        results={results}
      />
    </div>
  );
}

/* --------------------------------------------------------------- job panel */

function JobPanel({ status, pending }: { status: AnyRecord | null; pending: boolean }) {
  if (!status && !pending) return null;
  const state = pending ? 'queued' : pickString(status, ['state'], 'unknown');
  const cmd = pickString(status, ['cmd']) as BackupCommand;
  const progress = pickNumber(status, ['progress'], Number.NaN);
  const label =
    state === 'queued'
      ? 'Waiting for the host to start it'
      : state === 'running'
        ? `Running${pickString(status, ['step']) ? ` - ${pickString(status, ['step'])}` : ''}`
        : state === 'busy'
          ? 'Another backup or restore is running - try again when it finishes'
          : state === 'failed'
            ? `Failed - ${pickString(status, ['error']) || 'see journalctl -t gmac-repoctl'}`
            : `Finished${pickString(status, ['step']) ? ` - ${pickString(status, ['step'])}` : ''}`;
  const at = pickString(status, ['at']);
  return (
    <Panel title={`Last request: ${COMMAND_LABELS[cmd] ?? (cmd || 'pending')}`} description={at ? `updated ${relativeTime(at)}` : undefined}>
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Badge tone={statusTone(state)}>{state}</Badge>
          <span className="text-sm text-slate-700 dark:text-slate-200">{label}</span>
        </div>
        {Number.isFinite(progress) && (state === 'running' || state === 'done') ? (
          <div className="h-2 w-full overflow-hidden rounded bg-slate-100 dark:bg-slate-800">
            <div className="h-full bg-brand-500" style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} />
          </div>
        ) : null}
        {pickString(status, ['by']) ? (
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Started by {pickString(status, ['by'])}
            {pickString(status, ['requested_at']) ? ` at ${formatDateTime(pickString(status, ['requested_at']))}` : ''}
          </p>
        ) : null}
      </div>
    </Panel>
  );
}

function BackupHealthBanner({ backup }: { backup: unknown }) {
  if (!backup) {
    return (
      <StatusBanner tone="info" title="No backup status reported">
        GET /admin/backup returned nothing yet - no backup has run on this deployment.
      </StatusBanner>
    );
  }
  const status = pickString(backup, ['status', 'state', 'health'], 'unknown');
  const latestRun = isRecord(backup) && isRecord(backup.latest_run) ? backup.latest_run : null;
  const lastRun =
    pickString(backup, ['last_success_at', 'last_run', 'last_run_at', 'updated_at']) ||
    pickString(latestRun, ['started_at', 'finished_at']);
  const size = pickNumber(backup, ['last_backup_size', 'size_bytes'], 0) || pickNumber(latestRun, ['bytes_added', 'size_bytes'], 0);
  const tone = statusTone(status);
  return (
    <StatusBanner tone={tone === 'neutral' ? 'info' : tone} title={`Backup status: ${status}`}>
      <div className="flex flex-wrap gap-x-6 gap-y-1">
        <p>Last run: {lastRun ? `${formatDateTime(lastRun)} (${relativeTime(lastRun)})` : 'never'}</p>
        {size > 0 ? <p>Added: {formatBytes(size)}</p> : null}
        {pickString(latestRun, ['error']) ? <p>Error: {pickString(latestRun, ['error'])}</p> : null}
      </div>
    </StatusBanner>
  );
}

/* --------------------------------------------------------- recovery points */

interface PointRow extends AnyRecord {
  id: string;
  repo: BackupRepo;
}

function RecoveryPoints({
  catalog,
  listed,
  available,
  running,
  busy,
  send,
  results,
}: {
  catalog: AnyRecord | null;
  listed: unknown;
  available: boolean;
  running: boolean;
  busy: string | null;
  send: (body: Record<string, unknown>) => Promise<boolean>;
  results: Record<string, unknown>;
}) {
  const [listRepo, setListRepo] = useState<BackupRepo>('daily');
  const [selected, setSelected] = useState<PointRow | null>(null);

  const rows: PointRow[] = useMemo(() => {
    const fromCatalog = pickArray(catalog, ['snapshots'])
      .filter(isRecord)
      .filter((r) => pickString(r, ['status']) !== 'pruned')
      .map((r) => ({
        ...r,
        id: pickString(r, ['short_id', 'id']),
        repo: ((BACKUP_REPOS as readonly string[]).includes(pickString(r, ['folder'])) ? pickString(r, ['folder']) : 'daily') as BackupRepo,
      }));
    if (fromCatalog.length > 0) return fromCatalog;
    return pickArray(listed, ['snapshots'])
      .filter(isRecord)
      .map((r) => ({ ...r, id: pickString(r, ['id']), repo: listRepo }));
  }, [catalog, listed, listRepo]);

  const columns: Array<DataTableColumn<PointRow>> = [
    { key: 'id', header: 'Backup', render: (row) => <span className="font-mono text-xs">{row.id}</span> },
    { key: 'repo', header: 'Folder', render: (row) => row.repo },
    {
      key: 'when',
      header: 'Taken',
      render: (row) => {
        const at = pickString(row, ['taken_at', 'time']);
        return at ? <span title={formatDateTime(at)}>{relativeTime(at)}</span> : '—';
      },
    },
    { key: 'kind', header: 'Kind', render: (row) => pickString(row, ['kind'], '—') },
    {
      key: 'status',
      header: 'Status',
      render: (row) => {
        const s = pickString(row, ['status']);
        return s ? <Badge tone={s === 'verified' ? 'ok' : s === 'failed-check' ? 'error' : 'neutral'}>{s}</Badge> : '—';
      },
    },
    {
      key: 'size',
      header: 'Size',
      className: 'text-right tabular',
      render: (row) => {
        const bytes = pickNumber(row, ['size_added_bytes', 'size_bytes'], 0);
        return bytes ? formatBytes(bytes) : '—';
      },
    },
    {
      key: 'pick',
      header: <span className="sr-only">Select</span>,
      render: (row) => (
        <Button size="sm" variant={selected?.id === row.id ? 'primary' : 'secondary'} onClick={() => setSelected(row)}>
          {selected?.id === row.id ? 'Selected' : 'Select'}
        </Button>
      ),
    },
  ];

  return (
    <>
      <Panel
        title="Recovery points"
        description="From the host's backup catalog (catalog.json). Rebuild it from the repositories if it looks wrong."
        bodyClassName="px-0 py-0 sm:px-0"
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <Select aria-label="Folder to list" value={listRepo} onChange={(e) => setListRepo(e.target.value as BackupRepo)}>
              {BACKUP_REPOS.map((repo) => (
                <option key={repo} value={repo}>
                  {repo}
                </option>
              ))}
            </Select>
            <Button size="sm" variant="secondary" disabled={!available || running} loading={busy === 'snapshots'} onClick={() => send({ cmd: 'snapshots', repo: listRepo })}>
              List from repository
            </Button>
            <Button size="sm" variant="secondary" disabled={!available || running} loading={busy === 'rebuild-catalog'} onClick={() => send({ cmd: 'rebuild-catalog' })}>
              Rebuild list
            </Button>
          </div>
        }
      >
        {rows.length === 0 ? (
          <EmptyState icon={<IconBackups className="h-8 w-8" />} title="No recovery points known yet" description="Run a backup, or rebuild the list from the repositories." />
        ) : (
          <DataTable columns={columns} rows={rows} rowKey={(row, index) => `${row.repo}-${row.id}-${index}`} caption="Recovery points" />
        )}
      </Panel>

      {selected ? (
        <RestorePanel
          key={`${selected.repo}-${selected.id}`}
          point={selected}
          available={available}
          running={running}
          busy={busy}
          send={send}
          results={results}
        />
      ) : null}
    </>
  );
}

/* ------------------------------------------------------------------ restore */

function RestorePanel({
  point,
  available,
  running,
  busy,
  send,
  results,
}: {
  point: PointRow;
  available: boolean;
  running: boolean;
  busy: string | null;
  send: (body: Record<string, unknown>) => Promise<boolean>;
  results: Record<string, unknown>;
}) {
  const [kind, setKind] = useState<RestoreKind>('docs');
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [safetyBackup, setSafetyBackup] = useState(true);
  const [filter, setFilter] = useState('');
  const base = { repo: point.repo, snapshot: point.id };

  const found = kind === 'full' ? null : results[FIND_COMMAND[kind]];
  const foundFor = found && isRecord(found) && isRecord(found.snapshot) ? pickString(found.snapshot, ['id']) : '';
  const matches = Boolean(found) && (!foundFor || point.id.startsWith(foundFor) || foundFor.startsWith(point.id));

  const items: Array<{ key: string; label: string; detail: string; raw: AnyRecord }> = useMemo(() => {
    if (!matches || !isRecord(found)) return [];
    if (kind === 'files') {
      return pickArray(found, ['files']).filter(isRecord).map((f) => ({
        key: pickString(f, ['path']),
        label: pickString(f, ['path']),
        detail: formatBytes(pickNumber(f, ['size'], 0)),
        raw: f,
      }));
    }
    if (kind === 'docs') {
      return pickArray(found, ['deleted']).filter(isRecord).map((d) => ({
        key: `${pickString(d, ['collection'])}\u0000${pickString(d, ['name'])}`,
        label: pickString(d, ['name']),
        detail: pickString(d, ['collection']),
        raw: d,
      }));
    }
    if (kind === 'chats') {
      return pickArray(found, ['deleted']).filter(isRecord).map((c) => ({
        key: pickString(c, ['id']),
        label: pickString(c, ['title'], '(untitled chat)'),
        detail: `${pickString(c, ['account'])} · ${pickNumber(c, ['messages'], 0)} messages`,
        raw: c,
      }));
    }
    return pickArray(found, ['deleted']).filter(isRecord).map((v) => ({
      key: pickString(v, ['id']),
      label: pickString(v, ['title'], pickString(v, ['id'])),
      detail: pickString(v, ['id']),
      raw: v,
    }));
  }, [found, kind, matches]);

  const visible = filter ? items.filter((i) => i.label.toLowerCase().includes(filter.toLowerCase())) : items;

  const restore = async (typed?: string) => {
    let body: Record<string, unknown>;
    if (kind === 'full') body = { ...base, cmd: 'restore-full', confirm: typed, safety_backup: safetyBackup };
    else if (kind === 'files') body = { ...base, cmd: 'restore-files', paths: [...picked] };
    else if (kind === 'docs') {
      body = {
        ...base,
        cmd: 'restore-docs',
        docs: [...picked].map((key) => {
          const [collection, name] = key.split('\u0000');
          return { collection, name };
        }),
      };
    } else body = { ...base, cmd: kind === 'chats' ? 'restore-chats' : 'restore-videos', [kind]: [...picked] };
    if (await send(body)) {
      setConfirmOpen(false);
      setPicked(new Set());
    }
  };

  const disabled = !available || running;

  return (
    <Panel
      title={`Restore from ${point.id} (${point.repo})`}
      description="Restores are additive: nothing that exists now is overwritten and no backup is ever modified - except a full restore."
    >
      <div className="space-y-4">
        <div className="flex flex-wrap items-end gap-2">
          <Field label="What to restore" htmlFor="restore-kind" className="min-w-[240px]">
            <Select id="restore-kind" value={kind} onChange={(e) => { setKind(e.target.value as RestoreKind); setPicked(new Set()); }}>
              {(Object.keys(RESTORE_LABELS) as RestoreKind[]).map((k) => (
                <option key={k} value={k}>
                  {RESTORE_LABELS[k]}
                </option>
              ))}
            </Select>
          </Field>
          {kind !== 'full' ? (
            <Button variant="secondary" disabled={disabled} loading={busy === FIND_COMMAND[kind]} onClick={() => send({ ...base, cmd: FIND_COMMAND[kind] })}>
              <IconSearch className="h-4 w-4" />
              {kind === 'files' ? 'Check backup & list files' : `Find ${RESTORE_LABELS[kind].toLowerCase()}`}
            </Button>
          ) : null}
        </div>

        {kind === 'full' ? (
          <div className="space-y-3">
            <p className="flex items-start gap-2 rounded-md bg-amber-50 px-3 py-2.5 text-xs text-amber-800 dark:bg-amber-950/50 dark:text-amber-200">
              <IconWarning className="mt-0.5 h-4 w-4 shrink-0" />
              Replaces the live database (accounts, chats, settings), every vector collection and the knowledge-base files
              with this backup. Open WebUI and the backend are stopped during the restore. Everything after this backup is lost
              unless the safety backup below is taken first.
            </p>
            <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
              <input
                type="checkbox"
                checked={safetyBackup}
                onChange={(e) => setSafetyBackup(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              Take a safety backup first (recommended; a milestone in the daily folder)
            </label>
            <Button variant="danger" disabled={disabled} onClick={() => setConfirmOpen(true)}>
              Full restore…
            </Button>
          </div>
        ) : !matches ? (
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Run the search above first; the host reads the backup (this can take a few minutes for large backups).
          </p>
        ) : (
          <div className="space-y-2">
            {isRecord(found) && pickString(found, ['note']) ? (
              <p className="text-xs text-slate-500 dark:text-slate-400">{pickString(found, ['note'])}</p>
            ) : null}
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Input aria-label="Filter" placeholder="Filter…" value={filter} onChange={(e) => setFilter(e.target.value)} className="max-w-xs" />
              <div className="flex gap-2">
                <Button size="sm" variant="secondary" onClick={() => setPicked(new Set(visible.map((i) => i.key)))} disabled={visible.length === 0}>
                  Select all
                </Button>
                <Button size="sm" onClick={() => setConfirmOpen(true)} disabled={disabled || picked.size === 0}>
                  Restore {picked.size || ''} selected
                </Button>
              </div>
            </div>
            {visible.length === 0 ? (
              <EmptyState title="Nothing to restore" description={kind === 'files' ? 'This backup holds no restorable files.' : 'Nothing in this backup is missing from the live system.'} />
            ) : (
              <ul className="max-h-96 divide-y divide-slate-100 overflow-auto rounded-md border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                {visible.slice(0, 1000).map((item) => (
                  <li key={item.key} className="flex items-center gap-2 px-3 py-2">
                    <input
                      type="checkbox"
                      checked={picked.has(item.key)}
                      onChange={(e) => {
                        const next = new Set(picked);
                        if (e.target.checked) next.add(item.key);
                        else next.delete(item.key);
                        setPicked(next);
                      }}
                      className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
                    />
                    <span className="min-w-0 flex-1 truncate text-sm text-slate-800 dark:text-slate-200">{item.label}</span>
                    <span className="shrink-0 text-xs text-slate-400">{item.detail}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      <ConfirmDialog
        open={confirmOpen}
        title={kind === 'full' ? 'Full restore' : `Restore ${picked.size} item(s)`}
        description={
          kind === 'full' ? (
            <>
              Replace the live system with backup <strong>{point.id}</strong> ({point.repo}). Type {FULL_RESTORE_PHRASE} to confirm.
            </>
          ) : (
            <>
              Restore {picked.size} item(s) from backup <strong>{point.id}</strong>. Items that exist now are skipped, never
              overwritten.
            </>
          )
        }
        confirmLabel={kind === 'full' ? 'Start full restore' : 'Restore'}
        confirmWord={kind === 'full' ? FULL_RESTORE_PHRASE : undefined}
        loading={busy !== null}
        error={null}
        onConfirm={(typed) => restore(typed || undefined)}
        onCancel={() => setConfirmOpen(false)}
      />
    </Panel>
  );
}
