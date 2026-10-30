'use client';

import { useCallback, useMemo, useState } from 'react';

import type { DataTableColumn } from '@/components/DataTable';
import { DataTable } from '@/components/DataTable';
import { ConfirmDialog } from '@/components/ConfirmDialog';
import { Drawer } from '@/components/Drawer';
import { EmptyState, ErrorState, SkeletonRows, UnavailableNotice } from '@/components/Feedback';
import { JobProgress } from '@/components/JobProgress';
import { Tabs, TabPanel } from '@/components/Tabs';
import { useToast } from '@/components/Toast';
import {
  IconDownload,
  IconPlus,
  IconRefresh,
  IconSearch,
  IconStop,
  IconTrash,
  IconVideos,
} from '@/components/icons';
import {
  Badge,
  Button,
  Field,
  Input,
  KeyValue,
  Panel,
  Select,
  Textarea,
  statusTone,
} from '@/components/ui';
import { ApiClientError, apiFetch, apiSend, apiUpload, attempt } from '@/lib/client';
import { formatBytes, formatDuration, formatNumber, formatUSD, relativeTime } from '@/lib/format';
import { useAsyncData, usePolling } from '@/lib/hooks';
import { jobIsActive, normalizeJobRow, type JobRow } from '@/lib/jobs';
import {
  isRecord,
  pickArray,
  pickBoolean,
  pickNumber,
  pickString,
  rowKey,
  scalarEntries,
  type AnyRecord,
} from '@/lib/normalize';
import { channelDiscoverSchema, parseUrlList, videoUrlSchema } from '@/lib/schemas';
import { SectionTitle } from '@/components/SectionTitle';

const TABS = [
  { id: 'library', label: 'Library' },
  { id: 'queue', label: 'Queue' },
  { id: 'drive', label: 'Drive intake' },
  { id: 'settings', label: 'Settings' },
];

export default function VideosPage() {
  const [tab, setTab] = useState('library');
  return (
    <div className="space-y-4">
      <div>
        <SectionTitle>Video Library</SectionTitle>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Ingested videos, the processing queue, Google Drive intake and video settings.
        </p>
      </div>
      <Tabs tabs={TABS} active={tab} onChange={setTab} ariaLabel="Video library sections" />
      <TabPanel id="library" active={tab}>
        <LibraryTab />
      </TabPanel>
      <TabPanel id="queue" active={tab}>
        <QueueTab />
      </TabPanel>
      <TabPanel id="drive" active={tab}>
        <DriveIntakeTab />
      </TabPanel>
      <TabPanel id="settings" active={tab}>
        <SettingsTab />
      </TabPanel>
    </div>
  );
}

/* ================================================================= Library */

function LibraryTab() {
  const toast = useToast();
  const [selected, setSelected] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [reingestTarget, setReingestTarget] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dialogError, setDialogError] = useState<string | null>(null);

  const load = useCallback(() => attempt(apiFetch('proxy/video/status')), []);
  const { data, loading, error, reload } = useAsyncData(load, []);
  usePolling(reload, 15_000);

  const rows = useMemo(() => {
    if (!data?.ok) return [];
    return pickArray(data.value, ['videos', 'jobs', 'items']).map((row, index) => normalizeJobRow(row, 'video', index));
  }, [data]);

  const unavailable = data?.ok === false && data.error.notAvailable;

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    setBusy(true);
    setDialogError(null);
    try {
      await apiSend('DELETE', `proxy/video/ingested/${encodeURIComponent(deleteTarget)}`);
      toast.push('success', 'Video removed', deleteTarget);
      setDeleteTarget(null);
      setSelected(null);
      reload();
    } catch (err) {
      setDialogError(err instanceof ApiClientError ? err.message : 'Delete failed');
    } finally {
      setBusy(false);
    }
  };

  const confirmReingest = async () => {
    if (!reingestTarget) return;
    setBusy(true);
    setDialogError(null);
    try {
      await apiSend('POST', `proxy/video/queue/${encodeURIComponent(reingestTarget)}?force=true`);
      toast.push('success', 'Re-ingestion queued', reingestTarget);
      setReingestTarget(null);
      reload();
    } catch (err) {
      setDialogError(err instanceof ApiClientError ? err.message : 'Re-ingest failed');
    } finally {
      setBusy(false);
    }
  };

  const columns: Array<DataTableColumn<AnyRecord>> = [
    {
      key: 'title',
      header: 'Video',
      render: (_row, index) => {
        const job = rows[index];
        return (
          <div className="min-w-0">
            <p className="truncate font-medium text-slate-900 dark:text-white">{job.label}</p>
            <p className="mt-0.5 truncate text-xs text-slate-400">{pickString(job.raw, ['url', 'video_url', 'source'], job.id)}</p>
          </div>
        );
      },
    },
    { key: 'status', header: 'Status', render: (_row, index) => <Badge tone={statusTone(rows[index].status)}>{rows[index].status}</Badge> },
    {
      key: 'duration',
      header: 'Duration',
      className: 'tabular',
      render: (_row, index) => {
        const seconds = pickNumber(rows[index].raw, ['duration', 'duration_seconds', 'length'], Number.NaN);
        return Number.isFinite(seconds) ? formatDuration(seconds) : '—';
      },
    },
    {
      key: 'updated',
      header: 'Updated',
      render: (_row, index) => (rows[index].updatedAt ? relativeTime(rows[index].updatedAt) : '—'),
    },
    {
      key: 'actions',
      header: <span className="sr-only">Actions</span>,
      render: (_row, index) => {
        const id = rows[index].id;
        return (
          <div className="flex justify-end gap-1">
            <Button
              size="sm"
              variant="secondary"
              onClick={(event) => {
                event.stopPropagation();
                setSelected(id);
              }}
            >
              Details
            </Button>
            <Button
              size="sm"
              variant="ghost"
              aria-label={`Re-ingest ${id}`}
              onClick={(event) => {
                event.stopPropagation();
                setDialogError(null);
                setReingestTarget(id);
              }}
            >
              <IconRefresh className="h-3.5 w-3.5 text-brand-600 dark:text-brand-400" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              aria-label={`Delete ${id}`}
              onClick={(event) => {
                event.stopPropagation();
                setDialogError(null);
                setDeleteTarget(id);
              }}
            >
              <IconTrash className="h-3.5 w-3.5 text-rose-600 dark:text-rose-400" />
            </Button>
          </div>
        );
      },
    },
  ];

  return (
    <Panel
      title="Library"
      description="Videos known to the platform (GET /video/status)"
      actions={
        <Button variant="secondary" size="sm" onClick={reload}>
          Refresh
        </Button>
      }
      bodyClassName="px-0 py-0 sm:px-0"
    >
      {unavailable ? (
        <div className="p-4">
          <UnavailableNotice
            feature="Video processing (GET /video/status)"
            detail="Video endpoints are disabled or not routed on this deployment. Enable the video services to use the library."
          />
        </div>
      ) : (
        <DataTable
          columns={columns}
          rows={rows.map((job) => job.raw)}
          rowKey={(row, index) => rowKey(row, index, ['video_id'])}
          loading={loading}
          error={error?.message ?? null}
          onRetry={reload}
          emptyTitle="No videos yet"
          emptyDescription="Queue a URL from the Queue tab or enable Drive intake to ingest videos."
          caption="Video library"
        />
      )}

      <VideoDrawer
        videoId={selected}
        onClose={() => setSelected(null)}
        onDelete={(id) => {
          setDialogError(null);
          setDeleteTarget(id);
        }}
        onReingest={(id) => {
          setDialogError(null);
          setReingestTarget(id);
        }}
      />

      <ConfirmDialog
        open={deleteTarget !== null}
        title="Delete ingested video"
        description={
          <>
            Delete <strong className="break-all">{deleteTarget}</strong> and its chunks from the knowledge base? The
            source file is kept.
          </>
        }
        confirmLabel="Delete"
        loading={busy}
        error={dialogError}
        onConfirm={confirmDelete}
        onCancel={() => setDeleteTarget(null)}
      />

      <ConfirmDialog
        open={reingestTarget !== null}
        title="Re-ingest video"
        description={
          <>
            Re-run the full pipeline (download → transcribe → chunk → embed) for{' '}
            <strong className="break-all">{reingestTarget}</strong>? Existing chunks are replaced.
          </>
        }
        confirmLabel="Re-ingest"
        danger={false}
        loading={busy}
        error={dialogError}
        onConfirm={confirmReingest}
        onCancel={() => setReingestTarget(null)}
      />
    </Panel>
  );
}

function VideoDrawer({
  videoId,
  onClose,
  onDelete,
  onReingest,
}: {
  videoId: string | null;
  onClose: () => void;
  onDelete: (id: string) => void;
  onReingest: (id: string) => void;
}) {
  const load = useCallback(
    () =>
      Promise.all([
        attempt(apiFetch(`proxy/video/status/${encodeURIComponent(videoId ?? '')}`)),
        attempt(apiFetch('proxy/video/costs')),
      ]),
    [videoId],
  );
  const { data, loading, reload } = useAsyncData(load, [videoId]);

  const detailAttempt = data ? data[0] : null;
  const costsAttempt = data ? data[1] : null;
  const detail = detailAttempt && detailAttempt.ok ? detailAttempt.value : null;
  const costs = costsAttempt && costsAttempt.ok ? pickArray(costsAttempt.value, ['costs', 'rows', 'items', 'videos']) : [];
  const detailError = detailAttempt && !detailAttempt.ok ? detailAttempt.error : null;

  const phases = pickArray(detail, ['phases', 'checkpoints', 'steps', 'stages']);
  const warnings = pickArray(detail, ['warnings', 'issues', 'notes']);
  const progress = pickNumber(detail, ['progress', 'percent', 'progress_pct'], Number.NaN);
  const eta = pickString(detail, ['eta', 'eta_human', 'estimated_completion', 'eta_seconds']);

  const costRows = costs.filter((row) => {
    const id = pickString(row, ['video_id', 'id', 'name']);
    return !videoId || id === videoId;
  });

  return (
    <Drawer
      open={videoId !== null}
      title={videoId ?? ''}
      subtitle={pickString(detail, ['title', 'name'], '') || 'Video processing detail'}
      onClose={onClose}
      width="max-w-2xl"
      footer={
        videoId ? (
          <>
            <Button variant="secondary" onClick={() => onReingest(videoId)}>
              <IconRefresh className="h-4 w-4" />
              Re-ingest
            </Button>
            <Button variant="danger" onClick={() => onDelete(videoId)}>
              <IconTrash className="h-4 w-4" />
              Delete
            </Button>
          </>
        ) : null
      }
    >
      {loading ? (
        <SkeletonRows rows={6} columns={2} />
      ) : detailError && !detailError.notAvailable ? (
        <ErrorState error={detailError.message} onRetry={reload} title="Could not load video detail" />
      ) : detailError?.notAvailable ? (
        <UnavailableNotice feature="Video status detail (GET /video/status/{id})" />
      ) : (
        <div className="space-y-5">
          <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
            <KeyValue label="Status" value={<Badge tone={statusTone(pickString(detail, ['status', 'state'], 'unknown'))}>{pickString(detail, ['status', 'state'], 'unknown')}</Badge>} />
            <KeyValue
              label="Progress"
              value={Number.isFinite(progress) ? `${progress.toFixed(0)}%` : '—'}
            />
            <KeyValue label="ETA" value={eta || '—'} />
            <KeyValue
              label="URL"
              value={<span className="break-all">{pickString(detail, ['url', 'video_url', 'source'], '—')}</span>}
            />
            <KeyValue
              label="Duration"
              value={(() => {
                const seconds = pickNumber(detail, ['duration', 'duration_seconds'], Number.NaN);
                return Number.isFinite(seconds) ? formatDuration(seconds) : '—';
              })()}
            />
          </dl>

          {Number.isFinite(progress) ? (
            <JobProgress label="Pipeline progress" status={pickString(detail, ['status', 'state'], 'running')} progress={progress} />
          ) : null}

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Phase checkpoints
            </h3>
            {phases.length === 0 ? (
              <p className="mt-2 text-xs text-slate-400">No phase checkpoints reported.</p>
            ) : (
              <ol className="mt-2 space-y-1.5">
                {phases.map((phase, index) => {
                  const name = pickString(phase, ['name', 'phase', 'step', 'stage'], `phase ${index + 1}`);
                  const state = pickString(phase, ['status', 'state', 'result'], 'unknown');
                  return (
                    <li
                      key={index}
                      className="flex items-center justify-between gap-3 rounded-md border border-slate-200 px-3 py-2 dark:border-slate-800"
                    >
                      <div className="min-w-0">
                        <p className="truncate text-sm text-slate-800 dark:text-slate-200">{name}</p>
                        {pickString(phase, ['detail', 'message']) ? (
                          <p className="mt-0.5 truncate text-xs text-slate-500 dark:text-slate-400">
                            {pickString(phase, ['detail', 'message'])}
                          </p>
                        ) : null}
                      </div>
                      <div className="flex shrink-0 items-center gap-2">
                        {pickString(phase, ['duration', 'elapsed']) ? (
                          <span className="text-xs tabular text-slate-400">{pickString(phase, ['duration', 'elapsed'])}</span>
                        ) : null}
                        <Badge tone={statusTone(state)}>{state}</Badge>
                      </div>
                    </li>
                  );
                })}
              </ol>
            )}
          </div>

          {warnings.length > 0 ? (
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">Warnings</h3>
              <ul className="mt-2 space-y-1">
                {warnings.map((warning, index) => (
                  <li key={index} className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:bg-amber-950/50 dark:text-amber-200">
                    {pickString(warning, ['message', 'warning', 'detail', 'text'], String(warning))}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Cost rows (GET /video/costs)
            </h3>
            {costRows.length === 0 ? (
              <p className="mt-2 text-xs text-slate-400">No cost rows recorded for this video.</p>
            ) : (
              <table className="mt-2 w-full text-xs">
                <thead>
                  <tr className="text-left text-slate-400">
                    <th className="py-1 font-medium">Stage</th>
                    <th className="py-1 text-right font-medium">Units</th>
                    <th className="py-1 text-right font-medium">Cost</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  {costRows.map((row, index) => (
                    <tr key={index}>
                      <td className="py-1.5 text-slate-700 dark:text-slate-200">
                        {pickString(row, ['stage', 'name', 'kind', 'model'], '—')}
                      </td>
                      <td className="py-1.5 text-right tabular text-slate-500">
                        {formatNumber(pickNumber(row, ['units', 'tokens', 'seconds', 'count']))}
                      </td>
                      <td className="py-1.5 text-right tabular text-slate-700 dark:text-slate-200">
                        {formatUSD(pickNumber(row, ['cost', 'cost_usd', 'amount']))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}
    </Drawer>
  );
}

/* =================================================================== Queue */

function QueueTab() {
  const toast = useToast();
  const [singleUrl, setSingleUrl] = useState('');
  const [batchUrls, setBatchUrls] = useState('');
  const [channelUrl, setChannelUrl] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  // Backend contract (haystack-api/routers/video.py): a video is first
  // catalogued (POST /video/fetch-url {url} -> {video_id}) and then queued by
  // id (POST /video/queue/{video_id} or POST /video/queue/batch {video_ids}).
  // Queued videos run when the worker is started (POST /video/process).
  const catalogue = async (url: string): Promise<string> => {
    const fetched = await apiSend<Record<string, unknown>>('POST', 'proxy/video/fetch-url', { url });
    const videoId = pickString(fetched, ['video_id']);
    if (!videoId) throw new ApiClientError(`The backend returned no video id for ${url}`);
    return videoId;
  };

  const queueSingle = async () => {
    setBusy('single');
    setFormError(null);
    try {
      const parsed = videoUrlSchema.parse({ url: singleUrl.trim() });
      const videoId = await catalogue(parsed.url);
      await apiSend('POST', `proxy/video/queue/${encodeURIComponent(videoId)}`);
      toast.push('success', 'Video queued', `${videoId} - start the worker to process the queue`);
      setSingleUrl('');
    } catch (err) {
      setFormError(err instanceof ApiClientError ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const queueBatch = async () => {
    setBusy('batch');
    setFormError(null);
    try {
      const batch = parseUrlList(batchUrls);
      const videoIds: string[] = [];
      const failed: string[] = [];
      for (const url of batch.urls) {
        try {
          // eslint-disable-next-line no-await-in-loop
          videoIds.push(await catalogue(url));
        } catch {
          failed.push(url);
        }
      }
      if (videoIds.length === 0) throw new ApiClientError('None of the URLs could be catalogued');
      const result = await apiSend<Record<string, unknown>>('POST', 'proxy/video/queue/batch', { video_ids: videoIds });
      const queued = pickNumber(result, ['queued'], 0);
      const already = pickArray(result, ['already_ingested']).length;
      toast.push(
        failed.length ? 'warning' : 'success',
        `Queued ${queued} of ${batch.urls.length}`,
        [already ? `${already} already ingested` : '', failed.length ? `${failed.length} URL(s) failed` : '']
          .filter(Boolean)
          .join(' · ') || undefined,
      );
      setBatchUrls('');
    } catch (err) {
      setFormError(err instanceof ApiClientError ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const discoverChannel = async () => {
    setBusy('discover');
    setFormError(null);
    try {
      const url = channelUrl.trim();
      if (url) channelDiscoverSchema.parse({ url });
      const query = url ? `?channel_url=${encodeURIComponent(url)}` : '';
      const result = await apiSend<Record<string, unknown>>('POST', `proxy/video/discover${query}`);
      const found = pickNumber(result, ['discovered'], 0);
      const total = pickNumber(result, ['total'], 0);
      toast.push('success', `Discovered ${found} new video${found === 1 ? '' : 's'} (${total} on the channel)`, url || 'configured channel');
    } catch (err) {
      setFormError(err instanceof ApiClientError ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const startWorker = async () => {
    setBusy('process');
    setFormError(null);
    try {
      const result = await apiSend<Record<string, unknown>>('POST', 'proxy/video/process');
      toast.push('success', 'Queue worker started', pickString(result, ['message', 'status'], undefined));
    } catch (err) {
      setFormError(err instanceof ApiClientError ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-4">
      <Panel
        title="Add a video by URL"
        description="Catalogue the video (POST /video/fetch-url), then queue it (POST /video/queue/{video_id})"
      >
        <div className="flex flex-wrap items-end gap-2">
          <Field label="Video URL" htmlFor="queue-url" className="min-w-[260px] flex-1">
            <Input
              id="queue-url"
              value={singleUrl}
              onChange={(event) => setSingleUrl(event.target.value)}
              placeholder="https://www.youtube.com/watch?v=…"
            />
          </Field>
          <Button onClick={queueSingle} loading={busy === 'single'} disabled={!singleUrl.trim()}>
            <IconPlus className="h-4 w-4" />
            Queue video
          </Button>
        </div>
      </Panel>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Batch queue" description="One URL per line (POST /video/queue/batch)">
          <Textarea
            aria-label="Batch URLs"
            value={batchUrls}
            onChange={(event) => setBatchUrls(event.target.value)}
            rows={5}
            placeholder={'https://…/video-1\nhttps://…/video-2'}
          />
          <div className="mt-3 flex items-center justify-between gap-2">
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {batchUrls.split(/[\n\r,;\s]+/).filter(Boolean).length} URL(s) parsed
            </p>
            <Button onClick={queueBatch} loading={busy === 'batch'} disabled={!batchUrls.trim()}>
              Queue batch
            </Button>
          </div>
        </Panel>

        <Panel
          title="Channel discovery"
          description="Catalogue a channel's videos (POST /video/discover). Leave the URL empty to use VIDEO_DISCOVERY_CHANNEL_URL."
        >
          <div className="space-y-3">
            <Field label="Channel URL (optional)" htmlFor="channel-url">
              <Input
                id="channel-url"
                value={channelUrl}
                onChange={(event) => setChannelUrl(event.target.value)}
                placeholder="https://www.youtube.com/@channel"
              />
            </Field>
            <Button onClick={discoverChannel} loading={busy === 'discover'} className="w-full">
              <IconSearch className="h-4 w-4" />
              Discover videos
            </Button>
          </div>
        </Panel>
      </div>

      <Panel
        title="Queue worker"
        description="Queued videos are processed one at a time once the worker runs (POST /video/process). Spend limits are enforced by the backend."
      >
        <Button onClick={startWorker} loading={busy === 'process'}>
          <IconPlus className="h-4 w-4" />
          Start processing the queue
        </Button>
      </Panel>

      {formError ? (
        <p role="alert" className="rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950/60 dark:text-rose-300">
          {formError}
        </p>
      ) : null}

      <QueueStatusPanel />
    </div>
  );
}

function QueueStatusPanel() {
  const toast = useToast();
  const load = useCallback(() => attempt(apiFetch('proxy/video/status')), []);
  const { data, loading, reload } = useAsyncData(load, []);
  usePolling(reload, 15_000);

  const rows = useMemo(() => {
    if (!data?.ok) return [] as JobRow[];
    return pickArray(data.value, ['videos', 'jobs', 'items'])
      .map((row, index) => normalizeJobRow(row, 'video', index))
      .filter((job) => jobIsActive(job.status));
  }, [data]);

  const cancel = async (id: string) => {
    try {
      await apiSend('POST', `proxy/video/cancel/${encodeURIComponent(id)}`);
      toast.push('info', 'Cancellation requested', id);
      reload();
    } catch (err) {
      toast.push('error', 'Cancel failed', err instanceof ApiClientError ? err.message : undefined);
    }
  };

  if (data?.ok === false && data.error.notAvailable) {
    return (
      <Panel title="Queue status">
        <UnavailableNotice feature="Video queue (GET /video/status)" />
      </Panel>
    );
  }

  return (
    <Panel
      title={`Queued & running (${rows.length})`}
      description="Live queue state (GET /video/status)"
      actions={
        <Button variant="secondary" size="sm" onClick={reload}>
          <IconRefresh className="h-3.5 w-3.5" />
          Refresh
        </Button>
      }
    >
      {loading ? (
        <SkeletonRows rows={3} columns={3} />
      ) : rows.length === 0 ? (
        <EmptyState
          icon={<IconVideos className="h-8 w-8" />}
          title="Queue is empty"
          description="Queued videos appear here with progress and a cancel action."
        />
      ) : (
        <div className="space-y-2">
          {rows.map((job, index) => (
            <JobProgress
              key={`${job.id}-${index}`}
              label={job.label}
              status={job.status}
              progress={Number.isFinite(job.progress) ? job.progress : null}
              detail={job.detail || job.id}
              error={job.error || null}
              right={
                <Button size="sm" variant="ghost" onClick={() => cancel(job.id)}>
                  <IconStop className="h-3.5 w-3.5 text-rose-600 dark:text-rose-400" />
                  Cancel
                </Button>
              }
            />
          ))}
        </div>
      )}
    </Panel>
  );
}

/* ============================================================= Drive intake */

interface DriveFolderView {
  intake_folder_id: string;
  intake_folder_name: string;
  intake_folder_source: string;
  autodelete_enabled: boolean;
  autodelete_hours: number;
  autoapprove_enabled: boolean;
  autopilot: AnyRecord | null;
}

function readDriveSettings(value: unknown): DriveFolderView {
  return {
    intake_folder_id: pickString(value, ['intake_folder_id']),
    intake_folder_name: pickString(value, ['intake_folder_name']),
    intake_folder_source: pickString(value, ['intake_folder_source']),
    autodelete_enabled: pickBoolean(value, ['autodelete_enabled'], false),
    autodelete_hours: pickNumber(value, ['autodelete_hours'], 24),
    autoapprove_enabled: pickBoolean(value, ['autoapprove_enabled'], false),
    autopilot: isRecord(value) && isRecord(value.autopilot) ? value.autopilot : null,
  };
}

const DRIVE_STATE_TONE: Record<string, 'ok' | 'warning' | 'error' | 'neutral' | 'info'> = {
  new: 'info',
  queued: 'neutral',
  processing: 'warning',
  ingested: 'ok',
  failed: 'error',
};

/**
 * Google Drive upload folder (haystack-api routers/video.py drive-intake/*):
 * choose the folder, scan it (dry run), queue selected videos, add selected
 * documents to the knowledge base (dashboard /api/drive/ingest), and the
 * auto-approve / auto-delete switches. Optional: everything reports "not
 * configured" when Drive is not set up.
 */
function DriveIntakeTab() {
  const toast = useToast();
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [scan, setScan] = useState<AnyRecord | null>(null);
  const [pickedVideos, setPickedVideos] = useState<Set<string>>(new Set());
  const [pickedDocs, setPickedDocs] = useState<Set<string>>(new Set());
  const [folderInput, setFolderInput] = useState('');
  const [hoursInput, setHoursInput] = useState('');
  const [cleanupReport, setCleanupReport] = useState<unknown>(null);

  const drive = useAsyncData(() => attempt(apiFetch('drive/status')), []);
  const settingsLoad = useAsyncData(() => attempt(apiFetch('proxy/video/drive-intake/settings')), []);

  const driveConfigured = drive.data?.ok ? pickBoolean(drive.data.value, ['configured'], false) : false;
  const settings = settingsLoad.data?.ok ? readDriveSettings(settingsLoad.data.value) : null;
  const settingsUnavailable = settingsLoad.data?.ok === false && settingsLoad.data.error.notAvailable;
  const folderChosen = Boolean(settings?.intake_folder_id);

  const videos = useMemo(() => pickArray(scan, ['videos']).filter(isRecord), [scan]);
  const docs = useMemo(() => pickArray(scan, ['ignored']).filter(isRecord), [scan]);

  const run = async (key: string, fn: () => Promise<void>) => {
    setBusy(key);
    setNotice(null);
    try {
      await fn();
    } catch (err) {
      setNotice(err instanceof ApiClientError ? err.message : `${key} failed`);
    } finally {
      setBusy(null);
    }
  };

  const chooseFolder = () =>
    run('folder', async () => {
      const id = extractDriveFolderId(folderInput);
      if (!id) throw new ApiClientError('Paste a Google Drive folder link or folder id');
      await apiSend('POST', 'proxy/video/drive-intake/folder', { folder_id: id, name: '', confirm: true });
      toast.push('success', 'Upload folder saved');
      setFolderInput('');
      setScan(null);
      settingsLoad.reload();
    });

  const runScan = () =>
    run('scan', async () => {
      const result = await apiSend<AnyRecord>('POST', 'proxy/video/drive-intake/scan', {});
      setScan(result);
      setPickedVideos(new Set());
      setPickedDocs(new Set());
      toast.push('success', `Scan complete: ${pickArray(result, ['videos']).length} video(s), ${pickArray(result, ['ignored']).length} other file(s)`);
    });

  const queueVideos = () =>
    run('queue', async () => {
      const result = await apiSend<AnyRecord>('POST', 'proxy/video/drive-intake/queue', {
        file_ids: [...pickedVideos],
        confirm: true,
      });
      toast.push('success', `Queued ${pickNumber(result, ['queued'], 0)} video(s)`, 'Start the worker on the Queue tab to process them');
      setPickedVideos(new Set());
      await runScanSilently();
    });

  const ingestDocs = () =>
    run('docs', async () => {
      const result = await apiSend<AnyRecord>('POST', 'drive/ingest', { file_ids: [...pickedDocs] });
      const failed = pickArray(result, ['results']).filter((r) => isRecord(r) && r.ok === false);
      toast.push(
        failed.length ? 'warning' : 'success',
        `Sent ${pickNumber(result, ['queued'], 0)} of ${pickNumber(result, ['requested'], 0)} document(s) to the knowledge base`,
        failed.map((r) => `${pickString(r, ['name', 'file_id'])}: ${pickString(r, ['error'])}`).join(' · ') || undefined,
      );
      setPickedDocs(new Set());
    });

  const runScanSilently = async () => {
    try {
      setScan(await apiSend<AnyRecord>('POST', 'proxy/video/drive-intake/scan', {}));
    } catch {
      /* the next manual scan reports it */
    }
  };

  const saveSwitch = (patch: Record<string, boolean | number>) =>
    run('settings', async () => {
      await apiSend('POST', 'proxy/video/drive-intake/settings', patch);
      toast.push('success', 'Drive intake settings saved');
      settingsLoad.reload();
    });

  const cleanup = (dryRun: boolean) =>
    run(dryRun ? 'cleanup-dry' : 'cleanup', async () => {
      const report = await apiSend('POST', 'proxy/video/drive-intake/cleanup', { dry_run: dryRun });
      setCleanupReport(report);
      toast.push('success', dryRun ? 'Cleanup preview ready' : 'Cleanup finished');
    });

  if (drive.loading || settingsLoad.loading) return <SkeletonRows rows={4} columns={3} />;

  if (!driveConfigured) {
    return (
      <UnavailableNotice
        feature="Google Drive intake"
        detail="Google Drive is not configured on this deployment (optional). Set DRIVE_OAUTH_CLIENT_ID and DRIVE_OAUTH_CLIENT_SECRET, then connect Drive on the Integrations page."
      />
    );
  }
  if (settingsUnavailable) {
    return <UnavailableNotice feature="Drive intake (GET /video/drive-intake/settings)" />;
  }

  return (
    <div className="space-y-4">
      {notice ? (
        <p role="alert" className="rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950/60 dark:text-rose-300">
          {notice}
        </p>
      ) : null}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Upload folder" description="The Drive folder the intake scans (POST /video/drive-intake/folder)">
          <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
            <KeyValue
              label="Folder"
              value={
                folderChosen ? (
                  <a
                    className="font-medium text-brand-600 hover:underline dark:text-brand-400"
                    href={`https://drive.google.com/drive/folders/${encodeURIComponent(settings?.intake_folder_id ?? '')}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {settings?.intake_folder_name || settings?.intake_folder_id}
                  </a>
                ) : (
                  'not chosen'
                )
              }
            />
            <KeyValue label="Source" value={settings?.intake_folder_source || '—'} />
          </dl>
          <div className="mt-3 flex flex-wrap items-end gap-2">
            <Field label="Folder link or id" htmlFor="drive-folder" className="min-w-[220px] flex-1">
              <Input
                id="drive-folder"
                value={folderInput}
                onChange={(event) => setFolderInput(event.target.value)}
                placeholder="https://drive.google.com/drive/folders/…"
              />
            </Field>
            <Button onClick={chooseFolder} loading={busy === 'folder'} disabled={!folderInput.trim()}>
              Use this folder
            </Button>
          </div>
        </Panel>

        <Panel title="Automation" description="Both default OFF (POST /video/drive-intake/settings)">
          <div className="space-y-3 text-sm text-slate-700 dark:text-slate-200">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={settings?.autoapprove_enabled ?? false}
                disabled={busy !== null}
                onChange={(event) => saveSwitch({ autoapprove_enabled: event.target.checked })}
                className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              Auto-approve: queue new videos from the folder automatically (paid processing)
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={settings?.autodelete_enabled ?? false}
                disabled={busy !== null}
                onChange={(event) => saveSwitch({ autodelete_enabled: event.target.checked })}
                className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              Auto-delete: remove videos from Drive once ingested
            </label>
            <div className="flex flex-wrap items-end gap-2">
              <Field label={`Delete after (hours, now ${settings?.autodelete_hours ?? '—'})`} htmlFor="drive-hours" className="max-w-[220px]">
                <Input
                  id="drive-hours"
                  type="number"
                  min={1}
                  value={hoursInput}
                  onChange={(event) => setHoursInput(event.target.value)}
                />
              </Field>
              <Button
                size="sm"
                variant="secondary"
                disabled={!hoursInput || busy !== null}
                onClick={() => saveSwitch({ autodelete_hours: Number(hoursInput) })}
              >
                Save
              </Button>
            </div>
            {settings?.autopilot ? (
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Last automatic round:{' '}
                {pickString(settings.autopilot, ['last_run_at']) ? relativeTime(pickString(settings.autopilot, ['last_run_at'])) : 'never'}
                {pickString(settings.autopilot, ['last_error']) ? ` - error: ${pickString(settings.autopilot, ['last_error'])}` : ''}
              </p>
            ) : null}
            <div className="flex flex-wrap gap-2 pt-1">
              <Button size="sm" variant="secondary" onClick={() => cleanup(true)} loading={busy === 'cleanup-dry'} disabled={!folderChosen}>
                Preview cleanup
              </Button>
              <Button size="sm" variant="danger" onClick={() => cleanup(false)} loading={busy === 'cleanup'} disabled={!folderChosen || !settings?.autodelete_enabled}>
                <IconTrash className="h-3.5 w-3.5" />
                Delete ingested files now
              </Button>
            </div>
            {cleanupReport !== null ? (
              <pre className="max-h-40 overflow-auto rounded-md bg-slate-50 p-2 text-xs dark:bg-slate-950">
                {JSON.stringify(cleanupReport, null, 2)}
              </pre>
            ) : null}
          </div>
        </Panel>
      </div>

      <Panel
        title="Folder contents"
        description="Scan is a dry run (POST /video/drive-intake/scan): nothing is queued until you choose"
        actions={
          <Button size="sm" onClick={runScan} loading={busy === 'scan'} disabled={!folderChosen}>
            <IconSearch className="h-3.5 w-3.5" />
            Scan folder
          </Button>
        }
      >
        {!folderChosen ? (
          <EmptyState title="No upload folder chosen" description="Choose the Drive folder to scan above." />
        ) : scan === null ? (
          <EmptyState title="Not scanned yet" description="Scan the folder to review new videos and documents." />
        ) : (
          <div className="space-y-5">
            <DrivePickList
              title={`Videos (${videos.length})`}
              rows={videos}
              picked={pickedVideos}
              setPicked={setPickedVideos}
              selectable={(row) => ['new', 'failed'].includes(pickString(row, ['state']))}
              badge={(row) => {
                const state = pickString(row, ['state'], 'unknown');
                return <Badge tone={DRIVE_STATE_TONE[state] ?? 'neutral'}>{state}</Badge>;
              }}
              action={
                <Button size="sm" onClick={queueVideos} loading={busy === 'queue'} disabled={pickedVideos.size === 0}>
                  Queue {pickedVideos.size || ''} video(s)
                </Button>
              }
            />
            <DrivePickList
              title={`Documents and other files (${docs.length})`}
              rows={docs}
              picked={pickedDocs}
              setPicked={setPickedDocs}
              selectable={(row) => !pickString(row, ['mime_type']).startsWith('application/vnd.google-apps.')}
              badge={(row) => <Badge tone="neutral">{pickString(row, ['mime_type'], 'file')}</Badge>}
              action={
                <Button size="sm" onClick={ingestDocs} loading={busy === 'docs'} disabled={pickedDocs.size === 0}>
                  Add {pickedDocs.size || ''} to knowledge base
                </Button>
              }
            />
          </div>
        )}
      </Panel>
    </div>
  );
}

function DrivePickList({
  title,
  rows,
  picked,
  setPicked,
  selectable,
  badge,
  action,
}: {
  title: string;
  rows: AnyRecord[];
  picked: Set<string>;
  setPicked: (next: Set<string>) => void;
  selectable: (row: AnyRecord) => boolean;
  badge: (row: AnyRecord) => React.ReactNode;
  action: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">{title}</h3>
        {action}
      </div>
      {rows.length === 0 ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">Nothing here.</p>
      ) : (
        <ul className="divide-y divide-slate-100 dark:divide-slate-800">
          {rows.map((row, index) => {
            const id = pickString(row, ['file_id'], `row-${index}`);
            const canPick = selectable(row);
            return (
              <li key={id} className="flex items-center justify-between gap-3 py-2">
                <label className="flex min-w-0 flex-1 items-center gap-2.5">
                  <input
                    type="checkbox"
                    disabled={!canPick}
                    checked={picked.has(id)}
                    onChange={(event) => {
                      const next = new Set(picked);
                      if (event.target.checked) next.add(id);
                      else next.delete(id);
                      setPicked(next);
                    }}
                    className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500 disabled:opacity-40"
                  />
                  <span className="min-w-0">
                    <span className="block truncate text-sm text-slate-800 dark:text-slate-200">{pickString(row, ['name'], id)}</span>
                    <span className="block truncate text-xs text-slate-400">
                      {formatBytes(pickNumber(row, ['size'], 0))}
                      {pickString(row, ['mod_time']) ? ` · ${relativeTime(pickString(row, ['mod_time']))}` : ''}
                    </span>
                  </span>
                </label>
                {badge(row)}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

/** Accepts a Drive folder URL or a bare folder id. */
function extractDriveFolderId(input: string): string {
  const value = input.trim();
  const match = value.match(/\/folders\/([A-Za-z0-9_-]{10,})/) ?? value.match(/[?&]id=([A-Za-z0-9_-]{10,})/);
  if (match) return match[1];
  return /^[A-Za-z0-9_-]{10,}$/.test(value) ? value : '';
}

/* ================================================================ Settings */

interface VideoAutoSettings {
  auto_check_enabled: boolean;
  auto_check_hour: number;
  auto_queue: boolean;
}

function SettingsTab() {
  const toast = useToast();
  const [form, setForm] = useState<VideoAutoSettings>({ auto_check_enabled: false, auto_check_hour: 6, auto_queue: false });
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [cookieText, setCookieText] = useState('');
  const [cookieFile, setCookieFile] = useState<File | null>(null);

  // GET/POST /video/settings: {auto_check_enabled, auto_check_hour, auto_queue}
  const settingsLoad = useAsyncData(() => attempt(apiFetch('proxy/video/settings')), []);
  const cookiesLoad = useAsyncData(() => attempt(apiFetch('proxy/video/cookies/status')), []);

  const [seedKey, setSeedKey] = useState<string | null>(null);
  const seed = settingsLoad.data?.ok && isRecord(settingsLoad.data.value) ? settingsLoad.data.value : null;
  const seedSignature = seed ? JSON.stringify(seed) : null;
  if (seed && seedSignature !== seedKey) {
    setSeedKey(seedSignature);
    setForm({
      auto_check_enabled: pickBoolean(seed, ['auto_check_enabled'], false),
      auto_check_hour: pickNumber(seed, ['auto_check_hour'], 6),
      auto_queue: pickBoolean(seed, ['auto_queue'], false),
    });
  }

  const settingsUnavailable = settingsLoad.data?.ok === false && settingsLoad.data.error.notAvailable;
  const cookiesUnavailable = cookiesLoad.data?.ok === false && cookiesLoad.data.error.notAvailable;

  const saveSettings = async () => {
    setBusy('save');
    setNotice(null);
    try {
      const hour = Math.min(23, Math.max(0, Math.round(Number(form.auto_check_hour) || 0)));
      await apiSend('POST', 'proxy/video/settings', { ...form, auto_check_hour: hour });
      toast.push('success', 'Video settings saved');
      settingsLoad.reload();
    } catch (err) {
      setNotice(err instanceof ApiClientError ? err.message : 'Save failed');
    } finally {
      setBusy(null);
    }
  };

  const uploadCookies = async () => {
    if (!cookieFile) return;
    setBusy('cookies-upload');
    setNotice(null);
    try {
      const form = new FormData();
      form.append('file', cookieFile);
      await apiUpload('proxy/video/cookies', form);
      toast.push('success', 'YouTube cookies uploaded', cookieFile.name);
      setCookieFile(null);
      cookiesLoad.reload();
    } catch (err) {
      setNotice(err instanceof ApiClientError ? err.message : 'Upload failed');
    } finally {
      setBusy(null);
    }
  };

  const saveCookiesText = async () => {
    setBusy('cookies-text');
    setNotice(null);
    try {
      await apiSend('POST', 'proxy/video/cookies/text', { content: cookieText });
      toast.push('success', 'YouTube cookies saved from text');
      setCookieText('');
      cookiesLoad.reload();
    } catch (err) {
      setNotice(err instanceof ApiClientError ? err.message : 'Save failed');
    } finally {
      setBusy(null);
    }
  };

  const validateCookies = async () => {
    setBusy('cookies-validate');
    setNotice(null);
    try {
      const result = await apiSend<Record<string, unknown>>('POST', 'proxy/video/cookies/validate', {});
      const validation = isRecord(result.validation) ? result.validation : {};
      const valid = pickBoolean(validation, ['valid'], false);
      cookiesLoad.reload();
      toast.push(valid ? 'success' : 'warning', valid ? 'Cookies look valid' : 'Cookies failed validation');
    } catch (err) {
      setNotice(err instanceof ApiClientError ? err.message : 'Validation failed');
    } finally {
      setBusy(null);
    }
  };

  const deleteCookies = async () => {
    setBusy('cookies-delete');
    setNotice(null);
    try {
      await apiSend('DELETE', 'proxy/video/cookies');
      toast.push('info', 'YouTube cookies deleted');
      cookiesLoad.reload();
    } catch (err) {
      setNotice(err instanceof ApiClientError ? err.message : 'Delete failed');
    } finally {
      setBusy(null);
    }
  };

  const cookieStatus: AnyRecord | null = cookiesLoad.data?.ok && isRecord(cookiesLoad.data.value) ? cookiesLoad.data.value : null;

  return (
    <div className="space-y-4">
      {notice ? (
        <p role="alert" className="rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950/60 dark:text-rose-300">
          {notice}
        </p>
      ) : null}

      <Panel
        title="Automatic channel check"
        description="Daily discovery of new channel videos (GET/POST /video/settings)"
        actions={
          <Button size="sm" onClick={saveSettings} loading={busy === 'save'} disabled={settingsUnavailable}>
            Save settings
          </Button>
        }
      >
        {settingsUnavailable ? (
          <UnavailableNotice feature="Video settings (GET /video/settings)" />
        ) : settingsLoad.loading ? (
          <SkeletonRows rows={3} columns={2} />
        ) : (
          <div className="space-y-3">
            <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
              <input
                type="checkbox"
                checked={form.auto_check_enabled}
                onChange={(event) => setForm({ ...form, auto_check_enabled: event.target.checked })}
                className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              Check the discovery channel for new videos every day
            </label>
            <Field label="Check at hour (UTC, 0-23)" htmlFor="auto-check-hour" className="max-w-[220px]">
              <Input
                id="auto-check-hour"
                type="number"
                min={0}
                max={23}
                value={String(form.auto_check_hour)}
                onChange={(event) => setForm({ ...form, auto_check_hour: Number(event.target.value) })}
              />
            </Field>
            <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
              <input
                type="checkbox"
                checked={form.auto_queue}
                onChange={(event) => setForm({ ...form, auto_queue: event.target.checked })}
                className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              Queue newly discovered videos automatically (paid processing)
            </label>
          </div>
        )}
      </Panel>

      <Panel
        title="YouTube cookies"
        description="Cookies let yt-dlp access members-only / age-restricted videos (GET/POST/DELETE /video/cookies)"
        actions={
          cookieStatus ? (
            <Badge tone={pickBoolean(cookieStatus, ['configured'], false) ? 'ok' : 'warning'}>
              {pickBoolean(cookieStatus, ['configured'], false) ? 'cookies present' : 'no cookies'}
            </Badge>
          ) : null
        }
      >
        {cookiesUnavailable ? (
          <UnavailableNotice feature="YouTube cookies (GET /video/cookies/status)" />
        ) : (
          <div className="space-y-4">
            <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
              <KeyValue
                label="Age"
                value={
                  cookieStatus && pickNumber(cookieStatus, ['age_days'], -1) >= 0
                    ? `${pickNumber(cookieStatus, ['age_days'], 0)} days (max ${pickNumber(cookieStatus, ['max_age_days'], 0)})`
                    : '—'
                }
              />
              <KeyValue
                label="Downloads today"
                value={cookieStatus ? `${pickNumber(cookieStatus, ['usage_today'], 0)} / ${pickNumber(cookieStatus, ['daily_cap'], 0)}` : '—'}
              />
              <KeyValue
                label="Last used"
                value={pickString(cookieStatus, ['last_used_at']) ? relativeTime(pickString(cookieStatus, ['last_used_at'])) : '—'}
              />
              <KeyValue
                label="Last validation"
                value={
                  cookieStatus && isRecord(cookieStatus.last_validation)
                    ? pickBoolean(cookieStatus.last_validation, ['valid'], false)
                      ? 'valid'
                      : `failed${pickString(cookieStatus.last_validation, ['error']) ? ` - ${pickString(cookieStatus.last_validation, ['error'])}` : ''}`
                    : '—'
                }
              />
            </dl>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div>
                <Field label="Upload cookies.txt file" htmlFor="cookies-file" hint="Netscape format, as exported by a browser extension">
                  <input
                    id="cookies-file"
                    type="file"
                    accept=".txt,text/plain"
                    onChange={(event) => setCookieFile(event.target.files?.[0] ?? null)}
                    className="block w-full text-xs text-slate-500 file:mr-3 file:rounded-md file:border-0 file:bg-brand-50 file:px-3 file:py-2 file:text-xs file:font-medium file:text-brand-700 hover:file:bg-brand-100 dark:file:bg-brand-950 dark:file:text-brand-300"
                  />
                </Field>
                <Button className="mt-2" size="sm" onClick={uploadCookies} loading={busy === 'cookies-upload'} disabled={!cookieFile}>
                  <IconUploadInline />
                  Upload cookies
                </Button>
              </div>

              <div>
                <Field label="Or paste cookie data" htmlFor="cookies-text">
                  <Textarea
                    id="cookies-text"
                    rows={4}
                    value={cookieText}
                    onChange={(event) => setCookieText(event.target.value)}
                    placeholder="# Netscape HTTP Cookie File"
                  />
                </Field>
                <div className="mt-2 flex flex-wrap gap-2">
                  <Button size="sm" onClick={saveCookiesText} loading={busy === 'cookies-text'} disabled={!cookieText.trim()}>
                    Save text
                  </Button>
                  <Button size="sm" variant="secondary" onClick={validateCookies} loading={busy === 'cookies-validate'}>
                    Validate
                  </Button>
                  <Button size="sm" variant="danger" onClick={deleteCookies} loading={busy === 'cookies-delete'}>
                    <IconTrash className="h-3.5 w-3.5" />
                    Delete cookies
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </Panel>
    </div>
  );
}

function IconUploadInline() {
  return <IconDownload className="h-3.5 w-3.5 rotate-180" />;
}
