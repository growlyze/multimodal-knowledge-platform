'use client';

import { Suspense, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';

import { ErrorState, LoadingBlock, SkeletonRows, UnavailableNotice } from '@/components/Feedback';
import { useToast } from '@/components/Toast';
import { IconExternal, IconInfo, IconLink, IconRefresh } from '@/components/icons';
import { Badge, Button, KeyValue, Panel } from '@/components/ui';
import { ApiClientError, apiFetch, apiSend, attempt } from '@/lib/client';
import { formatBytes, relativeTime } from '@/lib/format';
import { useAsyncData } from '@/lib/hooks';
import { isRecord, pickArray, pickBoolean, pickNumber, pickString, type AnyRecord } from '@/lib/normalize';
import { SectionTitle } from '@/components/SectionTitle';

export default function IntegrationsContent() {
  return (
    <Suspense fallback={<LoadingBlock label="Loading integrations…" />}>
      <IntegrationsBody />
    </Suspense>
  );
}

const CREDENTIAL_LABELS: Array<[string, string, string]> = [
  ['google_api_key', 'Google AI (Gemini)', 'GOOGLE_API_KEY'],
  ['anthropic_api_key', 'Anthropic (Claude)', 'ANTHROPIC_API_KEY'],
  ['embedding_api_key', 'Embeddings', 'EMBEDDING_API_KEY'],
  ['reranker_api_key', 'Reranker', 'RERANKER_API_KEY'],
  ['groq_api_key', 'Groq (transcription)', 'GROQ_API_KEY'],
  ['voyage_api_key', 'Voyage', 'VOYAGE_API_KEY'],
  ['external_loader_api_key', 'Open WebUI content loader', 'EXTERNAL_LOADER_API_KEY'],
];

const CATEGORY_TONE: Record<string, 'ok' | 'warning' | 'error' | 'neutral'> = {
  ok: 'ok',
  not_configured: 'neutral',
  rate_limited: 'warning',
  network: 'warning',
  server_error: 'warning',
  billing: 'error',
  auth: 'error',
  model_unavailable: 'error',
  unknown: 'warning',
};

function IntegrationsBody() {
  const toast = useToast();
  const searchParams = useSearchParams();

  const backend = useAsyncData(() => attempt(apiFetch('proxy/health')), []);
  const config = useAsyncData(() => attempt(apiFetch<AnyRecord>('proxy/dashboard/config')), []);
  const drive = useAsyncData(() => attempt(apiFetch<AnyRecord>('drive/status?live=1')), []);
  const cookies = useAsyncData(() => attempt(apiFetch('proxy/video/cookies/status')), []);
  const [probe, setProbe] = useState<AnyRecord | null>(null);
  const [probing, setProbing] = useState(false);
  const [busy, setBusy] = useState(false);

  // Return from GET /kb/api/drive/callback: ?drive=connected|error&message=…
  const driveParam = searchParams.get('drive');
  const driveMessage = searchParams.get('message');
  useEffect(() => {
    if (driveParam === 'connected') {
      toast.push('success', 'Google Drive connected');
      drive.reload();
    } else if (driveParam === 'error') {
      toast.push('error', 'Drive connection failed', driveMessage ?? undefined);
    }
    if (driveParam) window.history.replaceState({}, '', window.location.pathname);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [driveParam]);

  const connectDrive = async () => {
    setBusy(true);
    try {
      const result = await apiSend<{ url?: string }>('POST', 'drive/connect');
      if (result?.url) {
        window.location.assign(result.url);
        return;
      }
      toast.push('error', 'Drive connect did not return a sign-in URL');
    } catch (err) {
      toast.push('error', 'Could not start the Google sign-in', err instanceof ApiClientError ? err.message : undefined);
    } finally {
      setBusy(false);
    }
  };

  const disconnectDrive = async () => {
    setBusy(true);
    try {
      await apiSend('DELETE', 'drive/status');
      toast.push('info', 'Google Drive disconnected', 'The credential was set aside; backups and Drive intake stop using it.');
      drive.reload();
    } catch (err) {
      toast.push('error', 'Could not disconnect Drive', err instanceof ApiClientError ? err.message : undefined);
    } finally {
      setBusy(false);
    }
  };

  const runProbe = async () => {
    setProbing(true);
    try {
      setProbe(await apiFetch<AnyRecord>('proxy/dashboard/providers/health'));
    } catch (err) {
      toast.push('error', 'Provider check failed', err instanceof ApiClientError ? err.message : undefined);
    } finally {
      setProbing(false);
    }
  };

  const cfg = config.data?.ok ? config.data.value : null;
  const credentials = cfg && isRecord(cfg.credentials) ? cfg.credentials : {};
  const models = cfg && isRecord(cfg.models) ? cfg.models : {};
  const integrations = cfg && isRecord(cfg.integrations) ? cfg.integrations : {};

  return (
    <div className="space-y-5">
      <div>
        <SectionTitle>Integrations</SectionTitle>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          AI providers, Google Drive and video credentials. Secrets are never displayed - only whether they are configured.
        </p>
      </div>

      <Panel
        title="Backend & AI providers"
        description="GET /health and GET /dashboard/config; credentials are set in the deployment .env"
        actions={
          <Button variant="secondary" size="sm" onClick={runProbe} loading={probing}>
            <IconRefresh className="h-3.5 w-3.5" />
            Check providers now
          </Button>
        }
      >
        {backend.loading || config.loading ? (
          <SkeletonRows rows={4} columns={3} />
        ) : backend.data?.ok === false ? (
          <ErrorState error={backend.data.error.message} onRetry={backend.reload} title="Could not reach the backend" />
        ) : (
          <div className="space-y-4">
            <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
              <KeyValue
                label="haystack-api"
                value={
                  <Badge tone={pickString(backend.data?.ok ? backend.data.value : null, ['status']) === 'ok' ? 'ok' : 'error'}>
                    {pickString(backend.data?.ok ? backend.data.value : null, ['status']) === 'ok' ? 'connected' : 'not reachable'}
                  </Badge>
                }
              />
              <KeyValue
                label="Embedding model"
                value={<span className="font-mono text-xs">{pickString(models, ['embedding_model'], '—')}</span>}
              />
              <KeyValue label="Reranker" value={<span className="font-mono text-xs">{pickString(models, ['reranker_model'], '—')}</span>} />
              <KeyValue label="Vision model" value={<span className="font-mono text-xs">{pickString(models, ['vision_model'], '—')}</span>} />
              <KeyValue label="Chat models" value={<Badge tone="neutral">via LiteLLM (litellm-config.yaml)</Badge>} />
            </dl>

            {config.data?.ok === false ? (
              <UnavailableNotice feature="Configuration status (GET /dashboard/config)" detail={config.data.error.message} />
            ) : (
              <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
                {CREDENTIAL_LABELS.map(([key, label, envName]) => (
                  <KeyValue
                    key={key}
                    label={label}
                    value={
                      <span className="flex items-center gap-2">
                        <Badge tone={credentials[key] === true ? 'ok' : 'neutral'}>
                          {credentials[key] === true ? 'configured' : 'not set'}
                        </Badge>
                        <code className="font-mono text-[11px] text-slate-400">{envName}</code>
                      </span>
                    }
                  />
                ))}
              </dl>
            )}

            {probe ? (
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Provider check (GET /dashboard/providers/health)
                </h3>
                <dl className="mt-2 rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
                  {pickArray(probe, ['providers']).filter(isRecord).map((row) => {
                    const category = pickString(row, ['category'], 'unknown');
                    return (
                      <KeyValue
                        key={pickString(row, ['provider'])}
                        label={pickString(row, ['provider'])}
                        value={
                          <span className="flex flex-wrap items-center gap-2">
                            <Badge tone={CATEGORY_TONE[category] ?? 'warning'}>{category.replace(/_/g, ' ')}</Badge>
                            <span className="text-xs text-slate-500 dark:text-slate-400">
                              {pickNumber(row, ['latency_ms'], 0)} ms {pickString(row, ['detail']) ? `· ${pickString(row, ['detail'])}` : ''}
                            </span>
                          </span>
                        }
                      />
                    );
                  })}
                </dl>
              </div>
            ) : null}
          </div>
        )}
      </Panel>

      <Panel
        title="Google Drive (optional)"
        description="One connected account for backups (restic repository folder) and the video/document upload folder"
        actions={<DriveStatusBadge data={drive.data?.ok ? drive.data.value : null} loading={drive.loading} />}
      >
        {drive.loading ? (
          <SkeletonRows rows={2} columns={2} />
        ) : drive.data?.ok === false ? (
          <ErrorState error={drive.data.error.message} onRetry={drive.reload} title="Could not load Drive status" />
        ) : (
          <DriveCard
            status={drive.data?.ok ? drive.data.value : null}
            busy={busy}
            onConnect={connectDrive}
            onDisconnect={disconnectDrive}
          />
        )}
      </Panel>

      {drive.data?.ok && pickBoolean(drive.data.value, ['connected'], false) ? <DriveBrowser /> : null}

      <Panel title="Video sources" description="Channel discovery and yt-dlp cookies">
        <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
          <KeyValue
            label="Discovery channel"
            value={
              pickBoolean(integrations, ['video_discovery_channel_configured'], false) ? (
                <span className="break-all font-mono text-xs">{pickString(integrations, ['video_discovery_channel_url'])}</span>
              ) : (
                <span className="text-xs text-slate-500">not configured (VIDEO_DISCOVERY_CHANNEL_URL) - a channel can still be entered per scan</span>
              )
            }
          />
          <KeyValue
            label="YouTube cookies"
            value={
              cookies.data?.ok === false ? (
                <span className="text-xs text-slate-500">unavailable</span>
              ) : (
                <Badge tone={pickBoolean(cookies.data?.ok ? cookies.data.value : null, ['configured'], false) ? 'ok' : 'neutral'}>
                  {pickBoolean(cookies.data?.ok ? cookies.data.value : null, ['configured'], false) ? 'present' : 'not set'}
                </Badge>
              )
            }
          />
          <KeyValue
            label="Manage"
            value={
              <a className="font-medium text-brand-600 hover:underline dark:text-brand-400" href="/kb/videos">
                Video Library → Settings
              </a>
            }
          />
        </dl>
      </Panel>

      <Panel title="Observability" description="Phoenix tracing UI at /traces/ (its own login)">
        <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
          <KeyValue
            label="Tracing export"
            value={
              <Badge tone={pickBoolean(integrations, ['tracing_configured'], false) ? 'ok' : 'neutral'}>
                {pickBoolean(integrations, ['tracing_configured'], false) ? 'configured' : 'not configured'}
              </Badge>
            }
          />
          <KeyValue
            label="Open"
            value={
              <a className="font-medium text-brand-600 hover:underline dark:text-brand-400" href="/traces/" target="_blank" rel="noreferrer">
                /traces/
              </a>
            }
          />
        </dl>
      </Panel>
    </div>
  );
}

function DriveStatusBadge({ data, loading }: { data: unknown; loading: boolean }) {
  if (loading) return <Badge tone="neutral">checking…</Badge>;
  if (!pickBoolean(data, ['configured'], false)) return <Badge tone="neutral">not configured</Badge>;
  return pickBoolean(data, ['connected'], false) ? <Badge tone="ok">connected</Badge> : <Badge tone="neutral">not connected</Badge>;
}

function DriveCard({
  status,
  busy,
  onConnect,
  onDisconnect,
}: {
  status: AnyRecord | null;
  busy: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
}) {
  const configured = pickBoolean(status, ['configured'], false);
  const connected = pickBoolean(status, ['connected'], false);
  const quota = status && isRecord(status.quota) ? status.quota : null;
  const redirectUri = pickString(status, ['redirectUri']);

  if (!configured) {
    return (
      <div className="space-y-3">
        <p className="flex items-start gap-2 rounded-md bg-slate-50 px-3 py-2.5 text-xs text-slate-600 dark:bg-slate-900 dark:text-slate-300">
          <IconInfo className="mt-0.5 h-4 w-4 shrink-0 text-brand-500" />
          <span>
            Google Drive is not configured. It is optional: backups can use the server&apos;s own rclone remote and
            everything else works without it. To enable it, create an OAuth &quot;Web application&quot; client in Google
            Cloud, register <code className="font-mono">{redirectUri || 'https://<DOMAIN>/kb/api/drive/callback'}</code> as
            its redirect URI and set <code className="font-mono">DRIVE_OAUTH_CLIENT_ID</code> and{' '}
            <code className="font-mono">DRIVE_OAUTH_CLIENT_SECRET</code> in .env.
          </span>
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
        <KeyValue label="Account" value={pickString(status, ['account']) || (connected ? 'unknown' : '—')} />
        <KeyValue
          label="Backup folder"
          value={
            pickString(status, ['backupFolderUrl']) ? (
              <a
                className="font-medium text-brand-600 hover:underline dark:text-brand-400"
                href={pickString(status, ['backupFolderUrl'])}
                target="_blank"
                rel="noreferrer"
              >
                open in Drive
              </a>
            ) : (
              '—'
            )
          }
        />
        <KeyValue
          label="Connected"
          value={pickString(status, ['connectedAt']) ? relativeTime(pickString(status, ['connectedAt'])) : '—'}
        />
        {quota ? (
          <KeyValue
            label="Storage"
            value={`${formatBytes(pickNumber(quota, ['usage_bytes'], 0))} used${
              quota.limit_bytes !== null ? ` of ${formatBytes(pickNumber(quota, ['limit_bytes'], 0))}` : ' (unlimited)'
            }`}
          />
        ) : null}
        {pickString(status, ['liveError']) ? <KeyValue label="Live check" value={pickString(status, ['liveError'])} /> : null}
        <KeyValue label="Redirect URI" value={<code className="break-all font-mono text-xs">{redirectUri || '—'}</code>} />
      </dl>
      {!connected ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {pickBoolean(status, ['signedOut'], false)
            ? 'Disconnected: backups and Drive intake do not use any Drive account until one is connected again.'
            : "Not connected from the dashboard: backups use the server's own rclone remote (if configured)."}
        </p>
      ) : null}
      <div className="flex flex-wrap gap-2">
        <Button onClick={onConnect} loading={busy}>
          <IconExternal className="h-4 w-4" />
          {connected ? 'Reconnect / switch account' : 'Connect Google Drive'}
        </Button>
        {connected ? (
          <Button variant="secondary" onClick={onDisconnect} loading={busy}>
            <IconLink className="h-4 w-4" />
            Disconnect
          </Button>
        ) : null}
      </div>
    </div>
  );
}

/** Read-only browser of the connected Drive (GET /kb/api/drive/browse). */
function DriveBrowser() {
  const [trail, setTrail] = useState<Array<{ id: string; name: string }>>([{ id: 'root', name: 'My Drive' }]);
  const current = trail[trail.length - 1];
  const listing = useAsyncData(
    () => attempt(apiFetch<AnyRecord>(`drive/browse?folder=${encodeURIComponent(current.id)}`)),
    [current.id],
  );
  const items = listing.data?.ok ? pickArray(listing.data.value, ['items']).filter(isRecord) : [];

  return (
    <Panel title="Browse Drive" description="Read-only view of what the connected account can see">
      <nav className="mb-3 flex flex-wrap items-center gap-1 text-xs text-slate-500 dark:text-slate-400" aria-label="Folder path">
        {trail.map((crumb, index) => (
          <span key={`${crumb.id}-${index}`} className="flex items-center gap-1">
            {index > 0 ? <span>/</span> : null}
            <button
              type="button"
              className="font-medium text-brand-600 hover:underline disabled:text-slate-500 disabled:no-underline dark:text-brand-400"
              disabled={index === trail.length - 1}
              onClick={() => setTrail(trail.slice(0, index + 1))}
            >
              {crumb.name}
            </button>
          </span>
        ))}
      </nav>
      {listing.loading ? (
        <SkeletonRows rows={4} columns={3} />
      ) : listing.data?.ok === false ? (
        <ErrorState error={listing.data.error.message} onRetry={listing.reload} />
      ) : items.length === 0 ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">This folder is empty.</p>
      ) : (
        <ul className="max-h-96 divide-y divide-slate-100 overflow-auto dark:divide-slate-800">
          {items.map((item) => (
            <li key={pickString(item, ['id'])} className="flex items-center justify-between gap-3 py-2 text-sm">
              {item.isFolder === true ? (
                <button
                  type="button"
                  className="min-w-0 truncate text-left font-medium text-slate-800 hover:underline dark:text-slate-100"
                  onClick={() => setTrail([...trail, { id: pickString(item, ['id']), name: pickString(item, ['name']) }])}
                >
                  {pickString(item, ['name'])}/
                </button>
              ) : (
                <span className="min-w-0 truncate text-slate-700 dark:text-slate-200">{pickString(item, ['name'])}</span>
              )}
              <span className="flex shrink-0 items-center gap-3 text-xs text-slate-400">
                {item.size !== null && item.size !== undefined ? formatBytes(pickNumber(item, ['size'], 0)) : ''}
                {pickString(item, ['modifiedTime']) ? relativeTime(pickString(item, ['modifiedTime'])) : ''}
                {pickString(item, ['webViewLink']) ? (
                  <a className="text-brand-600 hover:underline dark:text-brand-400" href={pickString(item, ['webViewLink'])} target="_blank" rel="noreferrer">
                    open
                  </a>
                ) : null}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
