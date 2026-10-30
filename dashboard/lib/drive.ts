/**
 * Google Drive integration (server-side only). OPTIONAL: with
 * DRIVE_OAUTH_CLIENT_ID / DRIVE_OAUTH_CLIENT_SECRET unset every Drive feature
 * reports "not configured" and nothing here touches the network.
 *
 * There is ONE Drive connection per deployment. Signing in with Google here
 * writes the connection record the rest of the stack already reads:
 *
 *   <drives dir>/primary.json      {token (rclone format), root_folder_id,
 *                                   account, connected_at, client_id,
 *                                   client_secret}   - mode 0600
 *   <drives dir>/signed-out.json   the record set aside by "Disconnect"
 *
 * ops/backup_drives.py applies primary.json to the backup's rclone remote
 * (the restic repository lives in its root_folder_id), and haystack-api's
 * drive_intake applies it for scans / downloads of the upload folder
 * (compose mounts /data/backup-drives into both). The <drives dir> is the
 * parent of BACKUP_CTL_DIR (default /data/backup-drives).
 *
 * Sign-in: OAuth "Web application" client, full `drive` scope (backups create
 * folders and upload; intake cleanup deletes processed files), offline access,
 * PKCE (S256) + single-use `state` kept server-side for 10 minutes.
 * Redirect URI: DRIVE_OAUTH_REDIRECT_URI, else <PUBLIC_BASE_URL>/kb/api/drive/callback.
 * No token ever reaches the browser.
 */
import { createHash, randomBytes } from 'node:crypto';
import { promises as fs } from 'node:fs';
import path from 'node:path';

export const AUTH_URL = 'https://accounts.google.com/o/oauth2/v2/auth';
export const TOKEN_URL = 'https://oauth2.googleapis.com/token';
export const DRIVE_API = 'https://www.googleapis.com/drive/v3';
export const DRIVE_SCOPE = 'https://www.googleapis.com/auth/drive';
export const FOLDER_MIME = 'application/vnd.google-apps.folder';
const PENDING_TTL_MS = 10 * 60 * 1000;
const HTTP_TIMEOUT_MS = 20_000;

export class DriveError extends Error {
  readonly status: number;
  constructor(message: string, status = 400) {
    super(message);
    this.name = 'DriveError';
    this.status = status;
  }
}

/* ------------------------------------------------------------ configuration */

export function driveClient(): { id: string; secret: string } | null {
  const id = process.env.DRIVE_OAUTH_CLIENT_ID?.trim();
  const secret = process.env.DRIVE_OAUTH_CLIENT_SECRET?.trim();
  return id && secret ? { id, secret } : null;
}

export function isDriveConfigured(): boolean {
  return driveClient() !== null;
}

/** https://host or host -> https://host (no trailing slash). */
export function publicBaseUrl(): string {
  const raw = (process.env.PUBLIC_BASE_URL || '').trim().replace(/\/+$/, '');
  if (!raw) return '';
  return /^https?:\/\//i.test(raw) ? raw : `https://${raw}`;
}

/** The OAuth redirect URI that must be registered on the Google client. */
export function driveRedirectUri(): string {
  const explicit = process.env.DRIVE_OAUTH_REDIRECT_URI?.trim();
  if (explicit) return explicit;
  const base = publicBaseUrl();
  return base ? `${base}/kb/api/drive/callback` : '';
}

export function drivesDir(): string {
  const ctl = process.env.BACKUP_CTL_DIR?.trim() || '/data/backup-drives/ctl';
  return path.dirname(path.resolve(ctl));
}

const primaryPath = () => path.join(drivesDir(), 'primary.json');
const signedOutPath = () => path.join(drivesDir(), 'signed-out.json');
const pendingDir = () => path.join(drivesDir(), 'dashboard-oauth');

export function folderUrl(folderId: string): string {
  return folderId ? `https://drive.google.com/drive/folders/${encodeURIComponent(folderId)}` : '';
}

export function appFolderName(): string {
  return `${process.env.APP_NAME?.trim() || 'Growlyze Multimodal AI Chat'} backups`;
}

/* ------------------------------------------------------------------- files */

async function readJson<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(file, 'utf8')) as T;
  } catch {
    return null;
  }
}

/** Atomic, private write (tmp + rename): a backup never reads half a file. */
async function writePrivateJson(file: string, value: unknown): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
  const tmp = `${file}.${randomBytes(4).toString('hex')}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(value), { mode: 0o600 });
  await fs.rename(tmp, file);
}

export interface RcloneToken {
  access_token: string;
  token_type: string;
  refresh_token: string;
  expiry: string;
}

export interface DriveRecord {
  token: RcloneToken;
  root_folder_id: string;
  account?: string;
  connected_at?: string;
  client_id?: string;
  client_secret?: string;
}

export async function readPrimary(): Promise<DriveRecord | null> {
  const rec = await readJson<DriveRecord>(primaryPath());
  if (rec && rec.token?.refresh_token && rec.root_folder_id) return rec;
  return null;
}

/* ------------------------------------------------------------------ sign-in */

function base64Url(buf: Buffer): string {
  return buf.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function stateKey(state: string): string {
  return createHash('sha256').update(state).digest('hex');
}

/** Starts a sign-in: returns Google's consent URL; PKCE verifier stays server-side. */
export async function startConnect(by: string): Promise<string> {
  const client = driveClient();
  const redirectUri = driveRedirectUri();
  if (!client || !redirectUri) {
    throw new DriveError(
      'Google Drive is not configured: set DRIVE_OAUTH_CLIENT_ID, DRIVE_OAUTH_CLIENT_SECRET and PUBLIC_BASE_URL.',
      400,
    );
  }
  const verifier = base64Url(randomBytes(64));
  const challenge = base64Url(createHash('sha256').update(verifier).digest());
  const state = base64Url(randomBytes(24));

  // Purge expired attempts, then remember this one (single use).
  await fs.mkdir(pendingDir(), { recursive: true, mode: 0o700 });
  const now = Date.now();
  for (const name of await fs.readdir(pendingDir()).catch(() => [] as string[])) {
    const rec = await readJson<{ created: number }>(path.join(pendingDir(), name));
    if (!rec || now - rec.created > PENDING_TTL_MS) await fs.rm(path.join(pendingDir(), name), { force: true });
  }
  await writePrivateJson(path.join(pendingDir(), `${stateKey(state)}.json`), {
    verifier,
    redirect_uri: redirectUri,
    created: now,
    by,
  });

  const params = new URLSearchParams({
    client_id: client.id,
    redirect_uri: redirectUri,
    response_type: 'code',
    scope: DRIVE_SCOPE,
    state,
    access_type: 'offline',
    prompt: 'consent select_account',
    code_challenge: challenge,
    code_challenge_method: 'S256',
  });
  return `${AUTH_URL}?${params.toString()}`;
}

async function takePending(state: string): Promise<{ verifier: string; redirect_uri: string }> {
  if (!state || !/^[A-Za-z0-9_-]{16,128}$/.test(state)) {
    throw new DriveError('This sign-in link is invalid - start the connection again.');
  }
  const file = path.join(pendingDir(), `${stateKey(state)}.json`);
  const rec = await readJson<{ verifier: string; redirect_uri: string; created: number }>(file);
  await fs.rm(file, { force: true });
  if (!rec || Date.now() - rec.created > PENDING_TTL_MS) {
    throw new DriveError('This sign-in attempt expired or was already used - start the connection again.');
  }
  return rec;
}

async function http(url: string, init: RequestInit = {}): Promise<Response> {
  try {
    return await fetch(url, { ...init, cache: 'no-store', signal: AbortSignal.timeout(HTTP_TIMEOUT_MS) });
  } catch {
    throw new DriveError('Google could not be reached from the dashboard.', 502);
  }
}

const auth = (accessToken: string) => ({ authorization: `Bearer ${accessToken}` });

async function accountEmail(accessToken: string): Promise<string> {
  const res = await http(`${DRIVE_API}/about?fields=user(emailAddress)`, { headers: auth(accessToken) });
  if (!res.ok) throw new DriveError(`Could not read the Google account (HTTP ${res.status}).`, 502);
  const body = (await res.json()) as { user?: { emailAddress?: string } };
  return body.user?.emailAddress ?? '';
}

export async function folderAccessible(accessToken: string, folderId: string): Promise<boolean> {
  if (!folderId) return false;
  const res = await http(
    `${DRIVE_API}/files/${encodeURIComponent(folderId)}?fields=id,trashed&supportsAllDrives=true`,
    { headers: auth(accessToken) },
  );
  if (!res.ok) return false;
  const body = (await res.json()) as { trashed?: boolean };
  return !body.trashed;
}

/** The backup folder in the connected account (found again when it exists). */
async function ensureFolder(accessToken: string, name: string): Promise<string> {
  const q = `name = '${name.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}' and mimeType = '${FOLDER_MIME}' and trashed = false and 'root' in parents`;
  const res = await http(`${DRIVE_API}/files?${new URLSearchParams({ q, fields: 'files(id,name)', spaces: 'drive' })}`, {
    headers: auth(accessToken),
  });
  if (!res.ok) throw new DriveError(`Could not look for the backup folder (HTTP ${res.status}).`, 502);
  const found = ((await res.json()) as { files?: Array<{ id: string }> }).files ?? [];
  if (found[0]?.id) return found[0].id;
  const created = await http(`${DRIVE_API}/files?fields=id`, {
    method: 'POST',
    headers: { ...auth(accessToken), 'content-type': 'application/json' },
    body: JSON.stringify({ name, mimeType: FOLDER_MIME }),
  });
  if (!created.ok) throw new DriveError(`Could not create the backup folder (HTTP ${created.status}).`, 502);
  return ((await created.json()) as { id: string }).id;
}

/** Completes a sign-in from Google's redirect (?code=&state=). */
export async function finishConnect(code: string, state: string): Promise<{ account: string; folderId: string; plan: string }> {
  const client = driveClient();
  if (!client) throw new DriveError('Google Drive is not configured on this deployment.');
  if (!code) throw new DriveError('Google did not return an authorization code.');
  const pending = await takePending(state);

  const res = await http(TOKEN_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      code,
      client_id: client.id,
      client_secret: client.secret,
      redirect_uri: pending.redirect_uri,
      grant_type: 'authorization_code',
      code_verifier: pending.verifier,
    }),
  });
  if (!res.ok) {
    throw new DriveError(
      `Google refused the sign-in code (HTTP ${res.status}) - it expires after a few minutes and works once; start again.`,
      502,
    );
  }
  const body = (await res.json()) as Record<string, unknown>;
  if (typeof body.access_token !== 'string' || typeof body.refresh_token !== 'string') {
    throw new DriveError(
      'Google returned no refresh token, so scheduled backups could not use this sign-in. Remove this app in the Google account permissions and connect again.',
      502,
    );
  }
  const expiresIn = typeof body.expires_in === 'number' ? body.expires_in : 3600;
  const token: RcloneToken = {
    access_token: body.access_token,
    token_type: typeof body.token_type === 'string' ? body.token_type : 'Bearer',
    refresh_token: body.refresh_token,
    expiry: new Date(Date.now() + expiresIn * 1000).toISOString().replace(/\.\d{3}Z$/, 'Z'),
  };

  const account = await accountEmail(token.access_token);
  // Keep the previous backup folder whenever this account can still open it
  // (same account, or one it was shared with): the backup history continues.
  // Otherwise backups restart in this account's own "<APP_NAME> backups" folder.
  const previous = (await readPrimary()) ?? (await readJson<DriveRecord>(signedOutPath()));
  let folderId = '';
  let plan = 'new';
  if (previous?.root_folder_id && (await folderAccessible(token.access_token, previous.root_folder_id))) {
    folderId = previous.root_folder_id;
    plan = 'renew';
  } else {
    folderId = await ensureFolder(token.access_token, appFolderName());
    plan = previous?.root_folder_id ? 'switch' : 'new';
  }

  const record: DriveRecord = {
    token,
    root_folder_id: folderId,
    account,
    connected_at: new Date().toISOString(),
    client_id: client.id,
    client_secret: client.secret,
  };
  await writePrivateJson(primaryPath(), record);
  // The set-aside credential is kept as history, never deleted.
  try {
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
    await fs.rename(signedOutPath(), path.join(drivesDir(), `signed-out-${stamp}.json`));
  } catch {
    /* nothing was signed out */
  }
  return { account, folderId, plan };
}

/** Disconnect: set the credential aside (moved, never deleted). */
export async function signOut(by: string): Promise<boolean> {
  const rec = await readJson<Record<string, unknown>>(primaryPath());
  if (!rec) return false;
  await writePrivateJson(signedOutPath(), { ...rec, signed_out_at: new Date().toISOString(), signed_out_by: by });
  await fs.rm(primaryPath(), { force: true });
  return true;
}

/* ----------------------------------------------------------- using a token */

/** A fresh access token for the connected account (refreshed every call). */
export async function accessToken(): Promise<string> {
  const rec = await readPrimary();
  if (!rec) throw new DriveError('Google Drive is not connected.', 409);
  const client = rec.client_id && rec.client_secret ? { id: rec.client_id, secret: rec.client_secret } : driveClient();
  if (!client) throw new DriveError('No OAuth client is available to refresh the Drive sign-in.', 409);
  const res = await http(TOKEN_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: client.id,
      client_secret: client.secret,
      refresh_token: rec.token.refresh_token,
      grant_type: 'refresh_token',
    }),
  });
  if (!res.ok) {
    throw new DriveError(`The Drive sign-in has expired or was revoked (HTTP ${res.status}) - connect it again.`, 502);
  }
  const body = (await res.json()) as { access_token?: string };
  if (!body.access_token) throw new DriveError('Google returned no access token.', 502);
  return body.access_token;
}

export interface DriveQuota {
  email: string;
  limit_bytes: number | null;
  usage_bytes: number;
  free_bytes: number | null;
}

export async function quota(token: string): Promise<DriveQuota> {
  const res = await http(`${DRIVE_API}/about?fields=user(emailAddress),storageQuota`, { headers: auth(token) });
  if (!res.ok) throw new DriveError(`Could not read the Google account's storage (HTTP ${res.status}).`, 502);
  const body = (await res.json()) as {
    user?: { emailAddress?: string };
    storageQuota?: { limit?: string; usage?: string };
  };
  const limit = body.storageQuota?.limit ? Number(body.storageQuota.limit) : null;
  const usage = Number(body.storageQuota?.usage ?? 0);
  return { email: body.user?.emailAddress ?? '', limit_bytes: limit, usage_bytes: usage, free_bytes: limit === null ? null : limit - usage };
}

export function validFolderId(value: string): boolean {
  return value === 'root' || /^[A-Za-z0-9_-]{10,200}$/.test(value);
}

export interface DriveItem {
  id: string;
  name: string;
  mimeType: string;
  size: number | null;
  modifiedTime: string;
  webViewLink: string;
  isFolder: boolean;
}

/** Read-only listing of one folder (names, sizes, dates, Google's own links). */
export async function listFolder(token: string, folderId: string, pageToken?: string): Promise<{ items: DriveItem[]; nextPageToken: string | null }> {
  if (!validFolderId(folderId)) throw new DriveError('Not a Google Drive folder id.');
  const params = new URLSearchParams({
    q: `'${folderId}' in parents and trashed = false`,
    fields: 'nextPageToken,files(id,name,mimeType,size,modifiedTime,webViewLink)',
    pageSize: '100',
    orderBy: 'folder,name',
    supportsAllDrives: 'true',
    includeItemsFromAllDrives: 'true',
  });
  if (pageToken) params.set('pageToken', pageToken);
  const res = await http(`${DRIVE_API}/files?${params}`, { headers: auth(token) });
  if (!res.ok) throw new DriveError(`Could not list the folder (HTTP ${res.status}).`, 502);
  const body = (await res.json()) as { nextPageToken?: string; files?: Array<Record<string, string>> };
  return {
    items: (body.files ?? []).map((f) => ({
      id: f.id,
      name: f.name,
      mimeType: f.mimeType,
      size: f.size ? Number(f.size) : null,
      modifiedTime: f.modifiedTime ?? '',
      webViewLink: f.webViewLink ?? '',
      isFolder: f.mimeType === FOLDER_MIME,
    })),
    nextPageToken: body.nextPageToken ?? null,
  };
}

export const MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024;

/** Downloads one file that must sit directly in `parentFolderId`. */
export async function downloadFromFolder(
  token: string,
  fileId: string,
  parentFolderId: string,
): Promise<{ name: string; mimeType: string; bytes: ArrayBuffer }> {
  if (!/^[A-Za-z0-9_-]{10,200}$/.test(fileId)) throw new DriveError('Not a Google Drive file id.');
  const meta = await http(
    `${DRIVE_API}/files/${fileId}?fields=id,name,mimeType,size,parents,trashed&supportsAllDrives=true`,
    { headers: auth(token) },
  );
  if (!meta.ok) throw new DriveError(`File not found in Drive (HTTP ${meta.status}).`, 404);
  const info = (await meta.json()) as { name: string; mimeType: string; size?: string; parents?: string[]; trashed?: boolean };
  if (info.trashed || !(info.parents ?? []).includes(parentFolderId)) {
    throw new DriveError(`${info.name} is not in the upload folder.`, 409);
  }
  if (info.mimeType.startsWith('application/vnd.google-apps.')) {
    throw new DriveError(`${info.name} is a Google Docs/Sheets file - export it (PDF, DOCX, XLSX) into the folder first.`);
  }
  if (Number(info.size ?? 0) > MAX_DOWNLOAD_BYTES) {
    throw new DriveError(`${info.name} is larger than ${MAX_DOWNLOAD_BYTES / 1024 / 1024} MB - upload it on the Uploads page.`);
  }
  const res = await http(`${DRIVE_API}/files/${fileId}?alt=media&supportsAllDrives=true`, { headers: auth(token) });
  if (!res.ok) throw new DriveError(`Could not download ${info.name} (HTTP ${res.status}).`, 502);
  return { name: info.name, mimeType: info.mimeType, bytes: await res.arrayBuffer() };
}

/* ------------------------------------------------------------------ status */

export interface DriveStatus {
  configured: boolean;
  connected: boolean;
  signedOut: boolean;
  account: string | null;
  connectedAt: string | null;
  backupFolderId: string | null;
  backupFolderUrl: string | null;
  redirectUri: string;
}

export async function driveStatus(): Promise<DriveStatus> {
  const rec = await readPrimary();
  let signedOut = false;
  try {
    await fs.access(signedOutPath());
    signedOut = !rec;
  } catch {
    signedOut = false;
  }
  return {
    configured: isDriveConfigured(),
    connected: Boolean(rec),
    signedOut,
    account: rec?.account || null,
    connectedAt: rec?.connected_at || null,
    backupFolderId: rec?.root_folder_id || null,
    backupFolderUrl: rec ? folderUrl(rec.root_folder_id) : null,
    redirectUri: driveRedirectUri(),
  };
}
