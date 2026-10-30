/**
 * Server-side typed fetch wrapper for the haystack-api backend.
 *
 * - Every request is made from the server with the ADMIN_API_KEY credential.
 * - The key is read from env here and never leaves the server.
 * - Requests are bounded by a timeout and failures are normalized into
 *   UpstreamError so pages can render friendly "not available" states.
 */

export class UpstreamError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(message: string, status: number, body: unknown = null) {
    super(message);
    this.name = 'UpstreamError';
    this.status = status;
    this.body = body;
  }

  /** Endpoints that do not exist on this deployment must degrade gracefully. */
  get notAvailable(): boolean {
    return this.status === 404 || this.status === 501;
  }
}

export function haystackBaseUrl(): string {
  return (process.env.HAYSTACK_API_URL || 'http://haystack-api:8000').replace(/\/+$/, '');
}

export function adminApiKey(): string | null {
  return process.env.ADMIN_API_KEY?.trim() || null;
}

/**
 * The backend credential, exactly as haystack-api/services/auth.py expects it
 * (HTTPBearer: "Authorization: Bearer <ADMIN_API_KEY>"). Server-side only.
 */
export function adminAuthHeaders(): Record<string, string> {
  const key = adminApiKey();
  return key ? { authorization: `Bearer ${key}` } : {};
}

export type QueryValue = string | number | boolean | undefined | null | string[];

export interface HaystackOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  query?: Record<string, QueryValue>;
  /** JSON body shortcut - serialized when `body` is not provided. */
  json?: unknown;
  body?: BodyInit | null;
  headers?: Record<string, string>;
  timeoutMs?: number;
}

export function buildHaystackUrl(path: string, query?: Record<string, QueryValue>): string {
  const clean = path.replace(/^\/+/, '');
  const url = new URL(`${haystackBaseUrl()}/${clean}`);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined || value === null || value === '') continue;
      if (Array.isArray(value)) {
        for (const item of value) if (item !== '') url.searchParams.append(key, item);
      } else {
        url.searchParams.set(key, String(value));
      }
    }
  }
  return url.toString();
}

/**
 * Low-level request against the backend. Throws UpstreamError on network
 * failure (502) and on non-2xx responses (upstream status preserved).
 */
export async function haystackFetch(path: string, options: HaystackOptions = {}): Promise<Response> {
  const apiKey = adminApiKey();
  if (!apiKey) {
    throw new UpstreamError('ADMIN_API_KEY is not configured on the dashboard', 503, {
      code: 'not_configured',
    });
  }

  const method = options.method ?? 'GET';
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...adminAuthHeaders(),
    ...options.headers,
  };

  let body: BodyInit | null | undefined = options.body ?? null;
  if (options.json !== undefined && body === null) {
    headers['content-type'] = 'application/json';
    body = JSON.stringify(options.json);
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? 30_000);

  try {
    return await fetch(buildHaystackUrl(path, options.query), {
      method,
      headers,
      body: method === 'GET' || method === 'DELETE' ? undefined : body ?? undefined,
      cache: 'no-store',
      signal: controller.signal,
    });
  } catch (error) {
    const aborted = error instanceof Error && error.name === 'AbortError';
    throw new UpstreamError(
      aborted ? `Backend request timed out: ${path}` : `Backend unreachable: ${path}`,
      502,
      { cause: aborted ? 'timeout' : 'network' },
    );
  } finally {
    clearTimeout(timeout);
  }
}

/** Request + JSON parse. 204/empty responses resolve to `undefined`. */
export async function haystackJson<T = unknown>(path: string, options: HaystackOptions = {}): Promise<T> {
  const res = await haystackFetch(path, { ...options, headers: { accept: 'application/json', ...options.headers } });
  const text = await res.text();
  const parsed: unknown = text ? safeJson(text) : null;

  if (!res.ok) {
    const detail =
      typeof parsed === 'object' && parsed !== null
        ? pickErrorMessage(parsed) ?? text.slice(0, 300)
        : text.slice(0, 300);
    throw new UpstreamError(detail || `Backend request failed (${res.status})`, res.status, parsed);
  }
  return (parsed ?? null) as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function pickErrorMessage(body: unknown): string | null {
  if (typeof body === 'string') return body || null;
  if (typeof body !== 'object' || body === null) return null;
  const record = body as Record<string, unknown>;
  for (const key of ['error', 'detail', 'message']) {
    const value = record[key];
    if (typeof value === 'string' && value) return value;
  }
  return null;
}

export function isNotAvailable(error: unknown): boolean {
  return error instanceof UpstreamError && error.notAvailable;
}

export function errorMessage(error: unknown, fallback = 'Request failed'): string {
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}
