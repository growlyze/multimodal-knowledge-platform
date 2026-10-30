import { NextRequest } from 'next/server';

import { UpstreamError, haystackFetch, haystackJson } from '@/lib/api';
import { DriveError, accessToken, downloadFromFolder, isDriveConfigured } from '@/lib/drive';
import { guard, noStore } from '@/lib/session';
import { isVideoFileName, suggestSourceType } from '@/lib/uploads';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

const MAX_FILES = 15;

/**
 * POST /kb/api/drive/ingest {file_ids: string[]} - adds DOCUMENTS from the
 * Drive upload folder to the knowledge base: each file (which must sit in the
 * upload folder reported by GET /video/drive-intake/settings) is downloaded
 * server-side and sent to POST /ingest/{type}, exactly like an upload on the
 * Uploads page. Videos are refused here - they go through the video queue
 * (POST /video/drive-intake/queue).
 */
export async function POST(request: NextRequest) {
  const { denied } = await guard(request);
  if (denied) return denied;
  if (!isDriveConfigured()) {
    return noStore({ error: 'Google Drive is not configured on this deployment.', code: 'not_configured' }, 404);
  }

  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return noStore({ error: 'Request body must be JSON', code: 'bad_request' }, 400);
  }
  const ids = Array.isArray((raw as { file_ids?: unknown })?.file_ids)
    ? ((raw as { file_ids: unknown[] }).file_ids.filter((v) => typeof v === 'string' && v.trim()) as string[])
    : [];
  if (ids.length === 0) return noStore({ error: 'Select at least one document', code: 'bad_request' }, 400);
  if (ids.length > MAX_FILES) return noStore({ error: `At most ${MAX_FILES} documents per request`, code: 'bad_request' }, 400);

  let folderId = '';
  try {
    const settings = await haystackJson<Record<string, unknown>>('video/drive-intake/settings');
    folderId = typeof settings?.intake_folder_id === 'string' ? settings.intake_folder_id : '';
  } catch (error) {
    const status = error instanceof UpstreamError ? error.status : 502;
    return noStore({ error: error instanceof Error ? error.message : 'Backend unreachable', code: 'upstream' }, status);
  }
  if (!folderId) {
    return noStore({ error: 'No Drive upload folder is chosen yet (Video Library -> Drive intake).', code: 'not_configured' }, 409);
  }

  let token: string;
  try {
    token = await accessToken();
  } catch (error) {
    const status = error instanceof DriveError ? error.status : 502;
    return noStore({ error: error instanceof Error ? error.message : 'Drive unavailable', code: 'drive_error' }, status);
  }

  const results: Array<{ file_id: string; name?: string; ok: boolean; job_id?: string; source_type?: string; error?: string }> = [];
  for (const fileId of ids) {
    try {
      // eslint-disable-next-line no-await-in-loop
      const file = await downloadFromFolder(token, fileId, folderId);
      if (isVideoFileName(file.name) || file.mimeType.startsWith('video/')) {
        results.push({ file_id: fileId, name: file.name, ok: false, error: 'videos are queued from the video list, not here' });
        continue;
      }
      const sourceType = suggestSourceType(file.name);
      const form = new FormData();
      form.append('file', new Blob([file.bytes], { type: file.mimeType || 'application/octet-stream' }), file.name);
      form.append('metadata', JSON.stringify({ origin: 'google_drive', drive_file_id: fileId }));
      // eslint-disable-next-line no-await-in-loop
      const res = await haystackFetch(`ingest/${sourceType}`, { method: 'POST', body: form, timeoutMs: 300_000 });
      // eslint-disable-next-line no-await-in-loop
      const body = (await res.json().catch(() => ({}))) as Record<string, unknown>;
      if (!res.ok) {
        const detail = typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`;
        results.push({ file_id: fileId, name: file.name, ok: false, error: detail });
      } else {
        results.push({
          file_id: fileId,
          name: file.name,
          ok: true,
          source_type: sourceType,
          job_id: typeof body.job_id === 'string' ? body.job_id : undefined,
        });
      }
    } catch (error) {
      results.push({ file_id: fileId, ok: false, error: error instanceof Error ? error.message : 'failed' });
    }
  }
  return noStore({ results, queued: results.filter((r) => r.ok).length, requested: ids.length });
}
