'use client';

import Link from 'next/link';
import { useCallback, useMemo, useState } from 'react';

import type { DataTableColumn } from '@/components/DataTable';
import { DataTable } from '@/components/DataTable';
import { ConfirmDialog } from '@/components/ConfirmDialog';
import { UnavailableNotice } from '@/components/Feedback';
import { useToast } from '@/components/Toast';
import { IconPlay, IconRefresh, IconStop, IconVideos } from '@/components/icons';
import { Badge, Button, Panel, Select, statusTone } from '@/components/ui';
import { ApiClientError, apiFetch, apiSend, attempt, type Attempt } from '@/lib/client';
import { formatDateTime, relativeTime } from '@/lib/format';
import { useAsyncData, usePolling } from '@/lib/hooks';
import { jobIsActive, jobIsFailed, normalizeJobRow, type JobRow } from '@/lib/jobs';
import { pickArray, pickString, rowKey, type AnyRecord } from '@/lib/normalize';
import { SectionTitle } from '@/components/SectionTitle';

type StatusFilter = 'all' | 'active' | 'queued' | 'completed' | 'failed';

interface JobsData {
  ingest: Attempt<unknown>;
  video: Attempt<unknown>;
}

async function loadJobs(): Promise<JobsData> {
  const [ingest, video] = await Promise.all([
    attempt(apiFetch('proxy/ingest/status')),
    attempt(apiFetch('proxy/video/status')),
  ]);
  return { ingest, video };
}

export default function JobsPage() {
  const toast = useToast();
  const [filter, setFilter] = useState<StatusFilter>('all');
  const [retryTarget, setRetryTarget] = useState<JobRow | null>(null);
  const [cancelTarget, setCancelTarget] = useState<JobRow | null>(null);
  const [busy, setBusy] = useState(false);
  const [dialogError, setDialogError] = useState<string | null>(null);

  const load = useCallback(() => loadJobs(), []);
  const { data, error, loading, reload } = useAsyncData(load, []);
  usePolling(reload, 10_000);

  const rows = useMemo(() => {
    const ingestRows = data?.ingest.ok
      ? normalizeMany(data.ingest.value, 'ingest')
      : [];
    const videoRows = data?.video.ok ? normalizeMany(data.video.value, 'video') : [];
    const all = [...ingestRows, ...videoRows];
    switch (filter) {
      case 'active':
        return all.filter((job) => jobIsActive(job.status));
      case 'queued':
        return all.filter((job) => job.status.toLowerCase() === 'queued' || job.status.toLowerCase() === 'pending');
      case 'completed':
        return all.filter((job) => ['completed', 'complete', 'done', 'success'].includes(job.status.toLowerCase()));
      case 'failed':
        return all.filter((job) => jobIsFailed(job.status));
      default:
        return all;
    }
  }, [data, filter]);

  const videoUnavailable = data?.video.ok === false && data.video.error.notAvailable;
  const ingestUnavailable = data?.ingest.ok === false && data.ingest.error.notAvailable;

  const runRetry = async () => {
    if (!retryTarget) return;
    setBusy(true);
    setDialogError(null);
    try {
      // Only video jobs can be re-run server-side (POST /video/queue/{id}?force=true):
      // an ingestion job holds no copy of its file, so it is re-uploaded instead.
      await apiSend('POST', `proxy/video/queue/${encodeURIComponent(retryTarget.id)}?force=true`);
      toast.push('success', 'Job re-submitted', retryTarget.label);
      setRetryTarget(null);
      reload();
    } catch (err) {
      setDialogError(err instanceof ApiClientError ? err.message : 'Retry failed');
    } finally {
      setBusy(false);
    }
  };

  const runCancel = async () => {
    if (!cancelTarget) return;
    setBusy(true);
    setDialogError(null);
    try {
      await apiSend('POST', `proxy/video/cancel/${encodeURIComponent(cancelTarget.id)}`);
      toast.push('info', 'Cancellation requested', cancelTarget.label);
      setCancelTarget(null);
      reload();
    } catch (err) {
      setDialogError(err instanceof ApiClientError ? err.message : 'Cancel failed');
    } finally {
      setBusy(false);
    }
  };

  const columns: Array<DataTableColumn<AnyRecord>> = [
    {
      key: 'job',
      header: 'Job',
      render: (_row, index) => {
        const job = rows[index];
        return (
          <div className="min-w-0">
            <p className="truncate font-medium text-slate-900 dark:text-white">{job.label}</p>
            <p className="mt-0.5 truncate text-xs text-slate-400">
              {job.kind} · {job.id}
            </p>
          </div>
        );
      },
    },
    {
      key: 'status',
      header: 'Status',
      render: (_row, index) => <Badge tone={statusTone(rows[index].status)}>{rows[index].status}</Badge>,
    },
    {
      key: 'progress',
      header: 'Progress',
      className: 'w-32',
      render: (_row, index) => {
        const job = rows[index];
        const pct = Number.isFinite(job.progress) ? Math.max(0, Math.min(100, job.progress)) : null;
        return (
          <div
            className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct ?? undefined}
            aria-label={`${job.label} progress`}
          >
            <div
              className={`h-full rounded-full bg-brand-600 ${pct === null ? 'w-1/3 animate-pulse' : ''}`}
              style={pct !== null ? { width: `${pct}%` } : undefined}
            />
          </div>
        );
      },
    },
    {
      key: 'error',
      header: 'Error reason',
      render: (_row, index) =>
        rows[index].error ? (
          <span className="text-xs text-rose-600 dark:text-rose-400">{rows[index].error}</span>
        ) : (
          <span className="text-xs text-slate-400">—</span>
        ),
    },
    {
      key: 'updated',
      header: 'Updated',
      className: 'tabular',
      render: (_row, index) => {
        const at = rows[index].updatedAt;
        return at ? <span title={formatDateTime(at)}>{relativeTime(at)}</span> : '—';
      },
    },
    {
      key: 'actions',
      header: <span className="sr-only">Actions</span>,
      render: (_row, index) => {
        const job = rows[index];
        return (
          <div className="flex justify-end gap-1">
            {job.kind === 'video' && !jobIsActive(job.status) ? (
              <Button
                size="sm"
                variant="secondary"
                onClick={(event) => {
                  event.stopPropagation();
                  setDialogError(null);
                  setRetryTarget(job);
                }}
              >
                <IconRefresh className="h-3.5 w-3.5" />
                Retry
              </Button>
            ) : null}
            {job.kind !== 'video' && jobIsFailed(job.status) ? (
              <Link
                href="/uploads"
                onClick={(event) => event.stopPropagation()}
                className="rounded-md px-2 py-1 text-xs font-medium text-brand-600 hover:underline dark:text-brand-400"
              >
                Re-upload
              </Link>
            ) : null}
            {job.kind === 'video' && jobIsActive(job.status) ? (
              <Button
                size="sm"
                variant="ghost"
                onClick={(event) => {
                  event.stopPropagation();
                  setDialogError(null);
                  setCancelTarget(job);
                }}
              >
                <IconStop className="h-3.5 w-3.5 text-rose-600 dark:text-rose-400" />
                Cancel
              </Button>
            ) : null}
          </div>
        );
      },
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <SectionTitle>Processing Jobs</SectionTitle>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Unified view of ingestion jobs (GET /ingest/status) and video jobs (GET /video/status). Refreshes every 10
            seconds.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select
            aria-label="Filter by status"
            value={filter}
            onChange={(event) => setFilter(event.target.value as StatusFilter)}
            className="h-9 w-40"
          >
            <option value="all">All statuses</option>
            <option value="active">Active</option>
            <option value="queued">Queued</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </Select>
          <Button variant="secondary" onClick={reload}>
            <IconRefresh className="h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {videoUnavailable ? (
        <UnavailableNotice
          feature="Video processing (GET /video/status)"
          detail="Video ingestion is disabled or not routed in this environment. Ingestion jobs below are unaffected."
        />
      ) : null}
      {ingestUnavailable ? (
        <UnavailableNotice
          feature="Ingestion status (GET /ingest/status)"
          detail="The ingestion status endpoint is disabled or not routed in this environment."
        />
      ) : null}

      <Panel bodyClassName="px-0 py-0 sm:px-0">
        <DataTable
          columns={columns}
          rows={rows.map((job) => job.raw)}
          rowKey={(row, index) => `${rows[index].kind}-${rowKey(row, index)}`}
          loading={loading}
          error={error?.message ?? (data?.ingest.ok === false && !data.ingest.error.notAvailable ? data.ingest.error.message : null)}
          onRetry={reload}
          emptyTitle="No jobs"
          emptyDescription="Jobs appear here as documents and videos are ingested."
          emptyAction={
            <Link href="/uploads">
              <Button variant="secondary" size="sm">
                <IconPlay className="h-3.5 w-3.5" />
                Start an ingestion
              </Button>
            </Link>
          }
          caption="Processing jobs"
        />
      </Panel>

      {videoUnavailable ? (
        <Panel title="Video jobs">
          <div className="flex items-start gap-2 text-xs text-slate-500 dark:text-slate-400">
            <IconVideos className="mt-0.5 h-4 w-4 shrink-0" />
            <p>
              This deployment does not expose the video endpoints. When they are enabled, queued and running videos show
              up here together with retry and cancel actions.
            </p>
          </div>
        </Panel>
      ) : null}

      <ConfirmDialog
        open={retryTarget !== null}
        title="Re-submit job"
        description={
          <>
            Re-submit <strong className="break-all">{retryTarget?.label}</strong> to the{' '}
            video queue (force=true)? The whole paid pipeline runs again.
          </>
        }
        confirmLabel="Re-submit"
        danger={false}
        loading={busy}
        error={dialogError}
        onConfirm={runRetry}
        onCancel={() => setRetryTarget(null)}
      />

      <ConfirmDialog
        open={cancelTarget !== null}
        title="Cancel video job"
        description={
          <>
            Request cancellation of <strong className="break-all">{cancelTarget?.label}</strong>? Work already committed
            to the knowledge base is kept.
          </>
        }
        confirmLabel="Cancel job"
        loading={busy}
        error={dialogError}
        onConfirm={runCancel}
        onCancel={() => setCancelTarget(null)}
      />
    </div>
  );
}

function normalizeMany(payload: unknown, kind: 'ingest' | 'video'): JobRow[] {
  return pickArray(payload, kind === 'video' ? ['videos', 'jobs', 'items'] : ['jobs', 'items']).map((row, index) =>
    normalizeJobRow(row, kind, index),
  );
}
