/**
 * Client-side helpers for talking to THIS app's server routes (never directly
 * to the backend). All backend credentials stay server-side; the browser only
 * ever calls /kb/api/* with the session cookie and the CSRF header.
 */

export const BASE_PATH = '/kb';
export const CSRF_HEADER = 'x-csrf-token';
export const CSRF_COOKIE = 'gmac_csrf';

export class ApiClientError extends Error {
  readonly status: number;
  readonly code: string | null;

  constructor(message: string, status = 0, code: string | null = null) {
    super(message);
    this.name = 'ApiClientError';
    this.status = status;
    this.code = code;
  }

  /** Feature endpoints missing on this deployment degrade gracefully. */
  get notAvailable(): boolean {
    return this.status === 404 || this.status === 501 || this.code === 'not_available' || this.code === 'not_configured';
  }
}

export function toApiError(error: unknown): ApiClientError {
  if (error instanceof ApiClientError) return error;
  if (error instanceof Error) return new ApiClientError(error.message || 'Request failed');
  return new ApiClientError('Request failed');
}

/** URL of one of this app's route handlers, e.g. apiUrl('auth/login'). */
export function apiUrl(path: string): string {
  return `${BASE_PATH}/api/${path.replace(/^\/+/, '')}`;
}

/** URL of the generic backend proxy, e.g. proxyUrl('admin/stats', { page: 1 }). */
export function proxyUrl(
  path: string,
  query: Record<string, string | number | boolean | undefined | null> = {},
): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === '') continue;
    params.set(key, String(value));
  }
  const qs = params.toString();
  return `${BASE_PATH}/api/proxy/${path.replace(/^\/+/, '')}${qs ? `?${qs}` : ''}`;
}

export function getCookieValue(name: string): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`));
  return match ? decodeURIComponent(match[1]) : null;
}

export function csrfToken(): string {
  return getCookieValue(CSRF_COOKIE) ?? '';
}

async function parseFailure(res: Response): Promise<ApiClientError> {
  let message = `Request failed (HTTP ${res.status})`;
  let code: string | null = null;
  try {
    const body: unknown = await res.json();
    if (typeof body === 'object' && body !== null) {
      const record = body as Record<string, unknown>;
      for (const key of ['error', 'detail', 'message']) {
        if (typeof record[key] === 'string' && record[key]) {
          message = record[key] as string;
          break;
        }
      }
      if (typeof record.code === 'string') code = record.code;
    }
  } catch {
    /* keep default message */
  }
  return new ApiClientError(message, res.status, code);
}

export async function apiFetch<T = unknown>(path: string, init: RequestInit = {}): Promise<T> {
  const extraHeaders = (init.headers ?? {}) as Record<string, string>;
  const res = await fetch(apiUrl(path), {
    ...init,
    credentials: 'same-origin',
    cache: 'no-store',
    headers: { accept: 'application/json', ...extraHeaders },
  });
  if (!res.ok) throw await parseFailure(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export async function apiSend<T = unknown>(
  method: 'POST' | 'PUT' | 'PATCH' | 'DELETE',
  path: string,
  body?: unknown,
): Promise<T> {
  return apiFetch<T>(path, {
    method,
    headers: {
      'content-type': 'application/json',
      [CSRF_HEADER]: csrfToken(),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

/** Multipart upload with progress (fetch cannot report upload progress). */
export function apiUpload<T = unknown>(
  path: string,
  form: FormData,
  onProgress?: (percent: number) => void,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', apiUrl(path));
    xhr.setRequestHeader(CSRF_HEADER, csrfToken());
    xhr.upload.onprogress = (event) => {
      if (onProgress && event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onerror = () => reject(new ApiClientError('Network error during upload'));
    xhr.ontimeout = () => reject(new ApiClientError('Upload timed out'));
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as T);
        return;
      }
      const record = (typeof body === 'object' && body !== null ? body : {}) as Record<string, unknown>;
      const message =
        typeof record.error === 'string'
          ? record.error
          : typeof record.detail === 'string'
            ? record.detail
            : `Upload failed (HTTP ${xhr.status})`;
      reject(new ApiClientError(message, xhr.status, typeof record.code === 'string' ? record.code : null));
    };
    xhr.send(form);
  });
}

export type Attempt<T> = { ok: true; value: T } | { ok: false; error: ApiClientError };

/** Never-throwing fetch for parallel loads where one endpoint may 404. */
export async function attempt<T>(promise: Promise<T>): Promise<Attempt<T>> {
  try {
    return { ok: true, value: await promise };
  } catch (error) {
    return { ok: false, error: toApiError(error) };
  }
}

export function notAvailableNotice(error: ApiClientError | null | undefined, feature: string): string | null {
  if (error && error.notAvailable) {
    return `This data source is not available on this deployment (${feature}).`;
  }
  return null;
}
