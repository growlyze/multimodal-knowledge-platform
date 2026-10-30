'use client';

import Link from 'next/link';
import { useCallback, useState, type ReactNode } from 'react';

import type { DataTableColumn } from '@/components/DataTable';
import { DataTable } from '@/components/DataTable';
import { ErrorState, SkeletonRows, UnavailableNotice } from '@/components/Feedback';
import { Tabs, TabPanel } from '@/components/Tabs';
import { IconExternal, IconInfo, IconRefresh, IconUser, IconWarning } from '@/components/icons';
import { Badge, Button, KeyValue, Panel, cx } from '@/components/ui';
import { apiFetch, attempt } from '@/lib/client';
import { formatBytes, formatNumber, formatPercent, relativeTime } from '@/lib/format';
import { useAsyncData } from '@/lib/hooks';
import { isRecord, pickArray, pickNumber, pickString, rowKey, type AnyRecord } from '@/lib/normalize';
import { SectionTitle } from '@/components/SectionTitle';

const TABS = [
  { id: 'users', label: 'Users' },
  { id: 'logs', label: 'Logs & monitoring' },
  { id: 'system', label: 'System' },
];

const PASSWORD_HASH_COMMAND =
  "python3 -c \"import hashlib,os,sys;s=os.urandom(16).hex();i=310000;print('pbkdf2_sha256:%d:%s:%s'%(i,s,hashlib.pbkdf2_hmac('sha256',sys.argv[1].encode(),s.encode(),i).hex()))\" 'new-password'";

export default function AdminContent({ appName }: { appName: string }) {
  const [tab, setTab] = useState('users');
  return (
    <div className="space-y-4">
      <div>
        <SectionTitle>Administration</SectionTitle>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Account status, monitoring endpoints and system configuration.
        </p>
      </div>
      <Tabs tabs={TABS} active={tab} onChange={setTab} ariaLabel="Administration sections" />
      <TabPanel id="users" active={tab}>
        <UsersTab />
      </TabPanel>
      <TabPanel id="logs" active={tab}>
        <LogsTab />
      </TabPanel>
      <TabPanel id="system" active={tab}>
        <SystemTab appName={appName} />
      </TabPanel>
    </div>
  );
}

/* =================================================================== Users */

function UsersTab() {
  const me = useAsyncData(() => attempt(apiFetch('auth/me')), []);

  return (
    <div className="space-y-4">
      <Panel title="Admin account" description="This deployment runs a single administrator account">
        {me.loading ? (
          <SkeletonRows rows={2} columns={2} />
        ) : (
          <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
            <KeyValue
              label="Account"
              value={
                <span className="inline-flex items-center gap-2">
                  <IconUser className="h-4 w-4 text-slate-400" />
                  {me.data?.ok ? pickString(me.data.value, ['username'], 'admin') : 'admin'}
                  <Badge tone="ok">active</Badge>
                </span>
              }
            />
            <KeyValue
              label="Role"
              value={<Badge tone="info">administrator (only role)</Badge>}
            />
            <KeyValue
              label="Session expiry"
              value="8 hours (signed HttpOnly cookie, Path=/kb)"
            />
          </dl>
        )}
      </Panel>

      <Panel title="Credential rotation" description="There is no user management UI — credentials live in the environment">
        <div className="space-y-3 text-sm text-slate-600 dark:text-slate-300">
          <p className="flex items-start gap-2">
            <IconInfo className="mt-0.5 h-4 w-4 shrink-0 text-brand-500" />
            <span>
              The admin username and password are read from the dashboard environment at startup. To rotate them,
              update the container environment and restart the dashboard — no data migration is involved.
            </span>
          </p>
          <ol className="ml-6 list-decimal space-y-1.5 text-xs">
            <li>
              Generate a new hash on the server:{' '}
              <code className="break-all rounded bg-slate-100 px-1 font-mono dark:bg-slate-800">
                {PASSWORD_HASH_COMMAND}
              </code>
            </li>
            <li>
              Set <code className="rounded bg-slate-100 px-1 font-mono dark:bg-slate-800">ADMIN_USERNAME</code> and{' '}
              <code className="rounded bg-slate-100 px-1 font-mono dark:bg-slate-800">ADMIN_PASSWORD_HASH</code> (the printed pbkdf2_sha256:… line) in{' '}
              <code className="rounded bg-slate-100 px-1 font-mono dark:bg-slate-800">.env</code>
            </li>
            <li>
              Optionally rotate <code className="rounded bg-slate-100 px-1 font-mono dark:bg-slate-800">SESSION_SECRET</code>{' '}
              to invalidate all existing sessions immediately
            </li>
            <li>Restart the dashboard container — login refuses to work until all three values are present</li>
          </ol>
          <p className="flex items-start gap-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:bg-amber-950/50 dark:text-amber-200">
            <IconWarning className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>
              If any of SESSION_SECRET, ADMIN_USERNAME or ADMIN_PASSWORD_HASH is missing, the dashboard fails closed:
              every login attempt is refused with a &quot;not configured&quot; error and no session can be created.
            </span>
          </p>
        </div>
      </Panel>
    </div>
  );
}

/* ==================================================================== Logs */

function LogsTab() {
  return (
    <div className="space-y-4">
      <EvalPanel
        title="Recent queries"
        endpoint="proxy/eval/queries"
        description="GET /eval/queries — latest retrieval queries handled by the platform"
        columns={[
          { key: 'timestamp', header: 'When', render: (row: AnyRecord) => relOrDash(pickString(row, ['timestamp', 'created_at', 'time'])) },
          { key: 'query', header: 'Query', render: (row: AnyRecord) => truncate(pickString(row, ['query', 'q', 'text'], '—'), 80) },
          { key: 'top_k', header: 'Top K', render: (row: AnyRecord) => String(pickNumber(row, ['top_k', 'k'], 0) || '—') },
          { key: 'log_id', header: 'Log ID', render: (row: AnyRecord) => truncate(pickString(row, ['log_id', 'id'], '—'), 16) },
        ]}
      />

      <EvalPanel
        title="Answer scores"
        endpoint="proxy/eval/answer-scores"
        description="GET /eval/answer-scores — quality scores for generated answers"
        columns={[
          { key: 'timestamp', header: 'When', render: (row: AnyRecord) => relOrDash(pickString(row, ['timestamp', 'created_at', 'time'])) },
          { key: 'question', header: 'Question', render: (row: AnyRecord) => truncate(pickString(row, ['question', 'query', 'q'], '—'), 70) },
          {
            key: 'score',
            header: 'Score',
            render: (row: AnyRecord) => {
              const score = pickNumber(row, ['score', 'rating', 'value'], Number.NaN);
              return Number.isFinite(score) ? (
                <Badge tone={score >= 0.7 ? 'ok' : score >= 0.4 ? 'warning' : 'error'}>{score.toFixed(2)}</Badge>
              ) : (
                '—'
              );
            },
          },
          { key: 'model', header: 'Model', render: (row: AnyRecord) => pickString(row, ['model', 'llm'], '—') },
        ]}
      />

      <EvalPanel
        title="Retrieval coverage"
        endpoint="proxy/eval/coverage"
        description="GET /eval/coverage — coverage of the indexed knowledge per collection"
        columns={[
          { key: 'collection', header: 'Collection', render: (row: AnyRecord) => pickString(row, ['collection', 'name', 'index'], '—') },
          {
            key: 'coverage',
            header: 'Coverage',
            render: (row: AnyRecord) => {
              const value = pickNumber(row, ['coverage', 'ratio', 'percent'], Number.NaN);
              return Number.isFinite(value) ? formatPercent(value > 1 ? value / 100 : value) : '—';
            },
          },
          { key: 'documents', header: 'Documents', render: (row: AnyRecord) => formatNumber(pickNumber(row, ['documents', 'docs', 'sources'])) },
          { key: 'chunks', header: 'Chunks', render: (row: AnyRecord) => formatNumber(pickNumber(row, ['chunks', 'vectors'])) },
        ]}
      />

      <EvalPanel
        title="System metrics"
        endpoint="proxy/eval/system"
        description="GET /eval/system — platform-level evaluation metrics"
        columns={[
          { key: 'metric', header: 'Metric', render: (row: AnyRecord) => pickString(row, ['metric', 'name', 'key'], '—') },
          { key: 'value', header: 'Value', render: (row: AnyRecord) => pickString(row, ['value', 'result', 'count'], '—') },
          { key: 'updated', header: 'Updated', render: (row: AnyRecord) => relOrDash(pickString(row, ['updated_at', 'timestamp'])) },
        ]}
      />

      <Panel
        title="Distributed tracing (Phoenix)"
        description="OpenTelemetry traces are served by the Phoenix collector at /traces"
        actions={
          <a
            href="/traces"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 rounded-md bg-white px-3 py-1.5 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 dark:bg-slate-800 dark:text-slate-200 dark:ring-slate-600 dark:hover:bg-slate-700"
          >
            Open /traces
            <IconExternal className="h-3.5 w-3.5" />
          </a>
        }
      >
        <p className="text-xs text-slate-500 dark:text-slate-400">
          The Phoenix UI is served by Caddy at <code className="font-mono">https://&lt;DOMAIN&gt;/traces</code> (routed
          to the tracing collector; the dashboard does not proxy it).
        </p>
      </Panel>

      <StoragePanel />
    </div>
  );
}

function EvalPanel({
  title,
  description,
  endpoint,
  columns,
}: {
  title: string;
  description: string;
  endpoint: string;
  columns: Array<{ key: string; header: string; render: (row: AnyRecord) => ReactNode }>;
}) {
  const load = useCallback(() => attempt(apiFetch(endpoint)), [endpoint]);
  const { data, loading, reload } = useAsyncData(load, [endpoint]);
  const rows = data?.ok ? pickArray(data.value, ['items', 'rows', 'results', 'queries', 'scores', 'entries', 'metrics']) : [];

  return (
    <Panel
      title={title}
      description={description}
      actions={
        <Button variant="secondary" size="sm" onClick={reload}>
          <IconRefresh className="h-3.5 w-3.5" />
          Refresh
        </Button>
      }
      bodyClassName="px-0 py-0 sm:px-0"
    >
      {data?.ok === false && data.error.notAvailable ? (
        <div className="p-4">
          <UnavailableNotice feature={endpoint.replace('proxy/', '')} />
        </div>
      ) : data?.ok === false ? (
        <div className="p-4">
          <ErrorState error={data.error.message} onRetry={reload} title={`Could not load ${title.toLowerCase()}`} />
        </div>
      ) : (
        <DataTable
          columns={columns as Array<DataTableColumn<AnyRecord>>}
          rows={rows as AnyRecord[]}
          rowKey={(row, index) => rowKey(row, index)}
          loading={loading}
          emptyTitle="No data yet"
          emptyDescription="Values appear here once the evaluation endpoints have recorded activity."
          caption={title}
        />
      )}
    </Panel>
  );
}

function StoragePanel() {
  const load = useCallback(() => attempt(apiFetch('proxy/admin/storage')), []);
  const { data, loading, reload } = useAsyncData(load, []);
  const rows = data?.ok
    ? pickArray(data.value, ['probes', 'volumes', 'mounts', 'disks', 'storage', 'rows']).map((row) => {
        if (!isRecord(row)) return row;
        const total = pickNumber(row, ['total', 'total_bytes', 'size'], 0);
        const used = pickNumber(row, ['used', 'used_bytes'], 0);
        const free = pickNumber(row, ['free', 'free_bytes', 'available'], Number.NaN);
        return { ...row, free: Number.isFinite(free) ? free : Math.max(0, total - used) };
      })
    : [];

  const columns: Array<DataTableColumn<AnyRecord>> = [
    { key: 'path', header: 'Path / volume', render: (row) => <span className="font-mono text-xs">{pickString(row, ['path', 'label', 'mount', 'name', 'device'], '—')}</span> },
    {
      key: 'total',
      header: 'Total',
      className: 'tabular text-right',
      render: (row) => formatBytes(pickNumber(row, ['total', 'total_bytes', 'size'], 0)),
    },
    {
      key: 'used',
      header: 'Used',
      className: 'tabular text-right',
      render: (row) => formatBytes(pickNumber(row, ['used', 'used_bytes'], 0)),
    },
    {
      key: 'free',
      header: 'Free',
      className: 'tabular text-right',
      render: (row) => formatBytes(pickNumber(row, ['free', 'free_bytes', 'available'], 0)),
    },
    {
      key: 'usage',
      header: 'Usage',
      render: (row) => {
        const total = pickNumber(row, ['total', 'total_bytes', 'size'], 0);
        const used = pickNumber(row, ['used', 'used_bytes'], 0);
        const ratio = total > 0 ? used / total : 0;
        return (
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className={cx(
                  'h-full rounded-full',
                  ratio >= 0.95 ? 'bg-rose-500' : ratio >= 0.85 ? 'bg-amber-500' : 'bg-brand-600',
                )}
                style={{ width: `${Math.min(100, ratio * 100)}%` }}
              />
            </div>
            <span className="text-xs tabular text-slate-500">{total > 0 ? formatPercent(ratio) : '—'}</span>
          </div>
        );
      },
    },
  ];

  return (
    <Panel
      title="Storage"
      description="GET /admin/storage — volumes used by the platform"
      actions={
        <Button variant="secondary" size="sm" onClick={reload}>
          <IconRefresh className="h-3.5 w-3.5" />
          Refresh
        </Button>
      }
      bodyClassName="px-0 py-0 sm:px-0"
    >
      {data?.ok === false && data.error.notAvailable ? (
        <div className="p-4">
          <UnavailableNotice feature="Storage metrics (GET /admin/storage)" />
        </div>
      ) : (
        <DataTable
          columns={columns}
          rows={rows as AnyRecord[]}
          rowKey={(row, index) => rowKey(row, index, ['path'])}
          loading={loading}
          emptyTitle="No storage metrics"
          emptyDescription="Volume information appears when the backend reports it."
          caption="Storage volumes"
        />
      )}
    </Panel>
  );
}

/* ================================================================== System */

function SystemTab({ appName }: { appName: string }) {
  // GET /health is the only configuration-bearing endpoint the backend exposes
  // (status + the configured embedding model); everything else is environment.
  const health = useAsyncData(() => attempt(apiFetch('proxy/health')), []);
  const healthValue = health.data?.ok ? health.data.value : null;
  const backendUp = isRecord(healthValue) && pickString(healthValue, ['status'], '') === 'ok';

  return (
    <div className="space-y-4">
      <Panel title="Display configuration" description="Read-only. Configure via the deployment environment.">
        <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
          <KeyValue label="APP_NAME" value={<span className="font-medium">{appName}</span>} />
          <KeyValue
            label="Backend"
            value={
              <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs dark:bg-slate-800">
                {'$HAYSTACK_API_URL'}
              </code>
            }
          />
          <KeyValue
            label="Backend status"
            value={
              health.loading ? (
                'checking…'
              ) : (
                <Badge tone={backendUp ? 'ok' : 'error'}>{backendUp ? 'connected (GET /health)' : 'unreachable'}</Badge>
              )
            }
          />
          <KeyValue
            label="Embedding model"
            value={
              <span className="font-mono text-xs">
                {pickString(healthValue, ['embedding'], '—')}
              </span>
            }
          />
        </dl>
      </Panel>

      <Panel title="Storage cleanup" description="How disk space is reclaimed on this platform">
        <ul className="space-y-2 text-xs text-slate-600 dark:text-slate-300">
          <li className="flex items-start gap-2">
            <IconInfo className="mt-0.5 h-3.5 w-3.5 shrink-0 text-brand-500" />
            <span>
              Delete ingested sources from <Link href="/knowledge" className="font-medium text-brand-600 hover:underline">Knowledge Base</Link>{' '}
              to remove their chunks from Qdrant and free vector storage.
            </span>
          </li>
          <li className="flex items-start gap-2">
            <IconInfo className="mt-0.5 h-3.5 w-3.5 shrink-0 text-brand-500" />
            <span>
              Remove processed videos from the <Link href="/videos" className="font-medium text-brand-600 hover:underline">Video Library</Link>{' '}
              (delete keeps the source file but drops transcriptions and chunks).
            </span>
          </li>
          <li className="flex items-start gap-2">
            <IconInfo className="mt-0.5 h-3.5 w-3.5 shrink-0 text-brand-500" />
            <span>
              Run <strong>find-deleted</strong> and <strong>rebuild-catalog</strong> from{' '}
              <Link href="/backups" className="font-medium text-brand-600 hover:underline">Backups &amp; Restore</Link> to
              compact restic history after large deletions.
            </span>
          </li>
        </ul>
      </Panel>

      <Panel title="Retention notes" description="Defaults applied by this deployment">
        <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
          <KeyValue label="Admin sessions" value="8 hours, then re-authentication is required" />
          <KeyValue label="Processing job history" value="Retained by the backend; prune via backend maintenance endpoints" />
          <KeyValue label="Backups" value="restic snapshots to Google Drive — retention is managed by the backup schedule" />
          <KeyValue label="Eval logs" value="Query and score logs are kept for monitoring (see Logs & monitoring)" />
        </dl>
        <p className="mt-3 flex items-start gap-2 rounded-md bg-slate-50 px-3 py-2.5 text-xs text-slate-600 dark:bg-slate-900 dark:text-slate-300">
          <IconInfo className="mt-0.5 h-4 w-4 shrink-0 text-brand-500" />
          Retention of documents, chats and videos themselves is content-driven: nothing is deleted automatically except
          through the cleanup actions above and the backup schedule.
        </p>
      </Panel>
    </div>
  );
}

/* ---------------------------------------------------------------- helpers */

function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

function relOrDash(value: string): string {
  return value ? relativeTime(value) : '—';
}
