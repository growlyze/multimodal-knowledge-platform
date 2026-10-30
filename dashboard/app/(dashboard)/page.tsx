'use client';

import Link from 'next/link';

import { ErrorState, EmptyState, Skeleton } from '@/components/Feedback';
import { JobProgress } from '@/components/JobProgress';
import { MetricCard } from '@/components/MetricCard';
import { StatusBanner } from '@/components/StatusBanner';
import { IconDatabase, IconFile, IconJobs, IconTable, IconVideos } from '@/components/icons';
import { Badge, Button, Panel, statusTone } from '@/components/ui';
import { attempt, apiFetch, type ApiClientError } from '@/lib/client';
import { formatBytes, formatNumber, relativeTime } from '@/lib/format';
import { useAsyncData, usePolling } from '@/lib/hooks';
import { jobIsActive, normalizeJobRow, type JobRow } from '@/lib/jobs';
import { isRecord, pickArray, pickBoolean, pickNumber, pickString } from '@/lib/normalize';
import { SectionTitle } from '@/components/SectionTitle';

interface OverviewData {
  stats: unknown;
  storage: unknown;
  backup: unknown;
  ingestJobs: unknown;
  videoJobs: unknown;
  errors: Record<string, ApiClientError>;
}

async function loadOverview(): Promise<OverviewData> {
  const [stats, storage, backup, ingestJobs, videoJobs] = await Promise.all([
    attempt(apiFetch('proxy/admin/stats')),
    attempt(apiFetch('proxy/admin/storage')),
    attempt(apiFetch('proxy/admin/backup')),
    attempt(apiFetch('proxy/ingest/status')),
    attempt(apiFetch('proxy/video/status')),
  ]);

  const errors: Record<string, ApiClientError> = {};
  if (!stats.ok) errors.stats = stats.error;
  if (!storage.ok) errors.storage = storage.error;
  if (!backup.ok) errors.backup = backup.error;
  if (!ingestJobs.ok) errors.ingest = ingestJobs.error;
  if (!videoJobs.ok) errors.video = videoJobs.error;

  return {
    stats: stats.ok ? stats.value : null,
    storage: storage.ok ? storage.value : null,
    backup: backup.ok ? backup.value : null,
    ingestJobs: ingestJobs.ok ? ingestJobs.value : null,
    videoJobs: videoJobs.ok ? videoJobs.value : null,
    errors,
  };
}

export default function OverviewPage() {
  const { data, error, loading, reload } = useAsyncData(loadOverview, []);
  usePolling(reload, 30_000);

  if (error) {
    return <ErrorState error={error.message} onRetry={reload} title="Could not load the overview" />;
  }

  const stats = data?.stats ?? null;
  const jobs: JobRow[] = [
    ...(data ? pickArray(data.ingestJobs, ['jobs', 'items']).map((row, i) => normalizeJobRow(row, 'ingest', i)) : []),
    ...(data ? pickArray(data.videoJobs, ['videos', 'jobs', 'items']).map((row, i) => normalizeJobRow(row, 'video', i)) : []),
  ];
  const activeJobs = jobs.filter((job) => jobIsActive(job.status));

  return (
    <div className="space-y-5">
      <SectionTitle>Overview</SectionTitle>
      {/* ---------------- metric cards ---------------- */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Documents"
          icon={<IconFile className="h-5 w-5" />}
          tone="sky"
          loading={loading}
          value={formatNumber(
            pickNumber(stats, [
              'total_documents',
              'documents',
              'document_count',
              'files',
              'sources',
              'num_documents',
            ]),
          )}
          sub={statSummary(stats)}
        />
        <MetricCard
          label="Chunks"
          icon={<IconDatabase className="h-5 w-5" />}
          tone="brand"
          loading={loading}
          value={formatNumber(pickNumber(stats, ['total_chunks', 'chunks', 'chunk_count', 'vectors', 'points', 'num_chunks']))}
          sub={<CollectionsBreakdown stats={stats} />}
        />
        <MetricCard
          label="Videos"
          icon={<IconVideos className="h-5 w-5" />}
          tone="rose"
          loading={loading}
          value={formatNumber(pickNumber(stats, ['total_videos', 'videos', 'video_count', 'video_chunks']))}
          sub="indexed video segments"
        />
        <MetricCard
          label="Tables"
          icon={<IconTable className="h-5 w-5" />}
          tone="teal"
          loading={loading}
          value={formatNumber(pickTableCount(stats))}
          sub="spreadsheet datasets"
        />
      </div>

      {/* ---------------- status banners ---------------- */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <StorageBanner storage={data?.storage ?? null} loading={loading} />
        <BackupBanner backup={data?.backup ?? null} loading={loading} />
      </div>

      {/* ---------------- active jobs strip ---------------- */}
      <Panel
        title="Active jobs"
        description="Running and queued ingestion / video jobs"
        actions={
          <Link href="/jobs" className="text-xs font-medium text-brand-600 hover:underline dark:text-brand-400">
            View all jobs →
          </Link>
        }
      >
        {loading ? (
          <div className="space-y-2">
            <Skeleton className="h-14 w-full" />
            <Skeleton className="h-14 w-full" />
          </div>
        ) : activeJobs.length === 0 ? (
          <EmptyState
            icon={<IconJobs className="h-8 w-8" />}
            title="No active jobs"
            description="Ingestion and video jobs will appear here while they run."
            action={
              <Link href="/uploads">
                <Button variant="secondary" size="sm">
                  Upload content
                </Button>
              </Link>
            }
          />
        ) : (
          <div className="space-y-2">
            {activeJobs.slice(0, 6).map((job, index) => (
              <JobProgress
                key={`${job.kind}-${job.id}-${index}`}
                label={job.label}
                status={job.status}
                progress={Number.isFinite(job.progress) ? job.progress : null}
                detail={job.detail || `${job.kind} job`}
                error={job.error || null}
                right={
                  <Link
                    href="/jobs"
                    className="text-xs font-medium text-brand-600 hover:underline dark:text-brand-400"
                  >
                    Details
                  </Link>
                }
              />
            ))}
          </div>
        )}
      </Panel>

      {/* ---------------- recent activity ---------------- */}
      <Panel title="Recent activity" description="Newest job and backup events across the platform">
        {loading ? (
          <Skeleton className="h-40 w-full" />
        ) : jobs.length === 0 && !data?.backup ? (
          <EmptyState title="No recent activity" description="Events show up here as content is ingested." />
        ) : (
          <ul className="divide-y divide-slate-100 dark:divide-slate-800">
            {buildActivity(jobs, data?.backup ?? null).map((item, index) => (
              <li key={index} className="flex items-start justify-between gap-3 py-2.5">
                <div className="min-w-0">
                  <p className="truncate text-sm text-slate-800 dark:text-slate-200">{item.label}</p>
                  <p className="mt-0.5 truncate text-xs text-slate-500 dark:text-slate-400">{item.detail}</p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <Badge tone={statusTone(item.status)}>{item.status}</Badge>
                  <span className="w-24 text-right text-xs text-slate-400">{relativeTime(item.at)}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      {data?.errors.stats && !loading ? (
        <ErrorState error={data.errors.stats.message} onRetry={reload} title="Some data could not be loaded" />
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ parts */

function statSummary(stats: unknown): string {
  const collections = Array.isArray((stats as Record<string, unknown> | null)?.collections)
    ? ((stats as { collections: unknown[] }).collections.length)
    : pickNumber(stats, ['collection_count', 'collections']);
  return collections > 0 ? `${formatNumber(collections)} collections` : 'indexed sources';
}

/** GET /admin/stats reports the table count as postgresql.structured_tables. */
function pickTableCount(stats: unknown): number {
  const direct = pickNumber(stats, ['tables', 'table_count', 'spreadsheets'], Number.NaN);
  if (Number.isFinite(direct)) return direct;
  const postgres = isRecord(stats) && isRecord(stats.postgresql) ? stats.postgresql : null;
  return pickNumber(postgres, ['structured_tables', 'tables'], 0);
}

function CollectionsBreakdown({ stats }: { stats: unknown }) {
  const perCollection = chunksPerCollection(stats);
  if (perCollection.length === 0) return <span>across all collections</span>;
  return (
    <span className="inline-flex flex-wrap gap-x-2 gap-y-0.5">
      {perCollection.slice(0, 4).map(([name, count]) => (
        <span key={name} className="whitespace-nowrap">
          <span className="font-medium text-slate-600 dark:text-slate-300">{name}</span> {formatNumber(count)}
        </span>
      ))}
    </span>
  );
}

function chunksPerCollection(stats: unknown): Array<[string, number]> {
  if (!isRecord(stats)) return [];
  for (const key of ['chunks_per_collection', 'chunks_by_collection', 'collections']) {
    const value = stats[key];
    if (Array.isArray(value)) {
      return value
        .map((entry): [string, number] => [
          pickString(entry, ['name', 'collection', 'key'], 'collection'),
          pickNumber(entry, ['points_count', 'chunks', 'count', 'value', 'points']),
        ])
        .filter(([, count]) => count > 0);
    }
    if (isRecord(value)) {
      return Object.entries(value)
        .filter(([, count]) => typeof count === 'number')
        .map(([name, count]) => [name, count as number]);
    }
  }
  return [];
}

interface StorageRow {
  path: string;
  total: number;
  used: number;
  free: number;
}

function storageRows(storage: unknown): StorageRow[] {
  // GET /admin/storage answers with { reported_at, probes: [...] }; each probe
  // carries label / path / used_bytes / total_bytes. Older shapes kept too.
  const rows = pickArray(storage, ['probes', 'volumes', 'mounts', 'disks', 'storage']);
  const mapped = rows
    .map((row) => ({
      path: pickString(row, ['path', 'mount', 'name', 'label', 'volume', 'device'], '/'),
      total: pickNumber(row, ['total', 'total_bytes', 'size', 'capacity'], 0),
      used: pickNumber(row, ['used', 'used_bytes', 'used_gb'], 0),
      free: pickNumber(row, ['free', 'free_bytes', 'available'], -1),
    }))
    .map((row) => ({ ...row, free: row.free >= 0 ? row.free : Math.max(0, row.total - row.used) }))
    .filter((row) => row.total > 0 || row.used > 0);
  if (mapped.length > 0) return mapped;

  const total = pickNumber(storage, ['total', 'total_bytes', 'size'], 0);
  const used = pickNumber(storage, ['used', 'used_bytes'], 0);
  return total > 0 || used > 0 ? [{ path: pickString(storage, ['path', 'mount', 'name'], 'storage'), total, used, free: Math.max(0, total - used) }] : [];
}

function StorageBanner({ storage, loading }: { storage: unknown; loading: boolean }) {
  if (loading) return <Skeleton className="h-24 w-full" />;
  if (!storage) {
    return (
      <StatusBanner tone="info" title="Storage status unavailable">
        The backend did not return storage information (GET /admin/storage). Check the Administration → Logs tab for
        capacity details.
      </StatusBanner>
    );
  }
  const rows = storageRows(storage);
  if (rows.length === 0) {
    const reportedAt = pickString(storage, ['reported_at', 'last_run', 'updated_at']);
    const stale = pickBoolean(storage, ['stale'], false);
    return (
      <StatusBanner
        tone={stale ? 'warning' : 'info'}
        title={reportedAt ? 'Storage report is out of date' : 'No storage report yet'}
      >
        {reportedAt
          ? `The last storage report was written ${relativeTime(reportedAt)}; figures may be out of date.`
          : 'The storage monitor has not written a report yet (GET /admin/storage).'}
      </StatusBanner>
    );
  }
  const worst = rows.reduce((a, b) => (a.total > 0 && a.used / a.total > b.used / (b.total || 1) ? a : b));
  const ratio = worst.total > 0 ? worst.used / worst.total : 0;
  const tone = ratio >= 0.95 ? 'error' : ratio >= 0.85 ? 'warning' : 'ok';
  return (
    <StatusBanner
      tone={tone}
      title={tone === 'ok' ? 'Storage healthy' : tone === 'warning' ? 'Storage filling up' : 'Storage critically full'}
    >
      <div className="space-y-1">
        {rows.slice(0, 3).map((row) => (
          <p key={row.path} className="tabular">
            <span className="font-medium">{row.path}</span>: {formatBytes(row.used)} of {formatBytes(row.total)} used
            {row.total > 0 ? ` (${Math.round((row.used / row.total) * 100)}%)` : ''}
          </p>
        ))}
      </div>
    </StatusBanner>
  );
}

function BackupBanner({ backup, loading }: { backup: unknown; loading: boolean }) {
  if (loading) return <Skeleton className="h-24 w-full" />;
  if (!backup) {
    return (
      <StatusBanner tone="info" title="Backup status unavailable">
        GET /admin/backup did not respond. Open Backups &amp; Restore for the control-channel status.
      </StatusBanner>
    );
  }
  const status = pickString(backup, ['status', 'state', 'health', 'result'], 'unknown');
  const latestRun = isRecord(backup) && isRecord(backup.latest_run) ? backup.latest_run : null;
  const lastRun =
    pickString(backup, ['last_success_at', 'last_run', 'last_backup', 'last_run_at', 'last_backup_at', 'updated_at']) ||
    pickString(latestRun, ['started_at', 'finished_at']);
  const size = pickNumber(backup, ['last_backup_size', 'size', 'size_bytes', 'last_size'], 0) ||
    pickNumber(latestRun, ['bytes_added', 'size', 'size_bytes'], 0);
  const tone = statusTone(status);
  return (
    <StatusBanner
      tone={tone === 'neutral' ? 'info' : tone}
      title={`Backups: ${status}`}
      actions={
        <Link href="/backups">
          <Button variant="secondary" size="sm">
            Manage
          </Button>
        </Link>
      }
    >
      <p className="tabular">
        Last run: {lastRun ? `${relativeTime(lastRun)}` : 'never'}
        {size > 0 ? ` · last snapshot ${formatBytes(size)}` : ''}
      </p>
    </StatusBanner>
  );
}

interface ActivityItem {
  label: string;
  detail: string;
  status: string;
  at: string;
}

function buildActivity(jobs: JobRow[], backup: unknown): ActivityItem[] {
  const items: ActivityItem[] = jobs
    .filter((job) => job.updatedAt)
    .map((job) => ({
      label: job.label,
      detail: `${job.kind === 'video' ? 'Video' : 'Ingestion'} job ${job.id}${job.error ? ` · ${job.error}` : ''}`,
      status: job.status,
      at: job.updatedAt,
    }));

  const backupAt = pickString(backup, ['last_success_at', 'last_run', 'last_backup', 'last_run_at', 'last_backup_at', 'updated_at']);
  if (backupAt) {
    items.push({
      label: 'Backup run completed',
      detail: pickString(backup, ['detail', 'message', 'snapshot', 'result'], 'restic snapshot'),
      status: pickString(backup, ['status', 'state'], 'ok'),
      at: backupAt,
    });
  }

  return items
    .sort((a, b) => new Date(b.at).getTime() - new Date(a.at).getTime())
    .slice(0, 8);
}
