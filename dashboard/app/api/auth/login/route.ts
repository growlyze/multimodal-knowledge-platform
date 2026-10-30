import { NextRequest, NextResponse } from 'next/server';

import {
  COOKIE_CSRF,
  COOKIE_SESSION,
  CSRF_COOKIE_OPTIONS,
  SESSION_COOKIE_OPTIONS,
  getAuthConfig,
  newCsrfToken,
  signSession,
  timingSafeEqualString,
  verifyCsrf,
  verifyPassword,
  SESSION_TTL_SECONDS,
} from '@/lib/auth';
import { loginSchema } from '@/lib/schemas';

export const dynamic = 'force-dynamic';

/**
 * POST /kb/api/auth/login
 *
 * Fail-closed: when SESSION_SECRET / ADMIN_USERNAME / ADMIN_PASSWORD_HASH are
 * not all configured the endpoint refuses with 503 "not configured" and no
 * session can ever be created.
 *
 * Brute-force protection: a small in-memory rate limiter keyed by client IP
 * (5 failed attempts per 15 minutes, then 429; plus a 50-failure ceiling
 * across all clients). Enough to stop online password
 * guessing against a single-admin deployment without extra infrastructure.
 */

const LOGIN_WINDOW_MS = 15 * 60 * 1000;
const LOGIN_MAX_FAILURES = 5;
// Ceiling across ALL clients in one window: rotating source addresses cannot
// turn the per-IP limit into unlimited guessing against the single admin.
const LOGIN_MAX_GLOBAL_FAILURES = 50;
const MAX_TRACKED_CLIENTS = 10_000;
const failures = new Map<string, { count: number; resetAt: number }>();
const globalFailures = { count: 0, resetAt: 0 };

function clientKey(request: NextRequest): string {
  // Caddy sets X-Forwarded-For itself (client-supplied values from untrusted
  // peers are replaced), so the first entry is the real client address.
  const forwarded = request.headers.get('x-forwarded-for');
  if (forwarded) return forwarded.split(',')[0].trim();
  return request.headers.get('x-real-ip') ?? 'unknown';
}

/** Returns milliseconds until a new attempt is allowed, or 0 when under the limit. */
function loginRetryAfterMs(request: NextRequest, now = Date.now()): number {
  if (globalFailures.resetAt > now && globalFailures.count >= LOGIN_MAX_GLOBAL_FAILURES) {
    return globalFailures.resetAt - now;
  }
  const key = clientKey(request);
  const entry = failures.get(key);
  if (!entry) return 0;
  if (entry.resetAt <= now) {
    failures.delete(key);
    return 0;
  }
  return entry.count >= LOGIN_MAX_FAILURES ? entry.resetAt - now : 0;
}

function recordFailure(request: NextRequest, now = Date.now()): void {
  if (globalFailures.resetAt <= now) {
    globalFailures.count = 0;
    globalFailures.resetAt = now + LOGIN_WINDOW_MS;
  }
  globalFailures.count += 1;

  if (failures.size >= MAX_TRACKED_CLIENTS) {
    for (const [key, entry] of failures) if (entry.resetAt <= now) failures.delete(key);
    if (failures.size >= MAX_TRACKED_CLIENTS) failures.clear();
  }
  const key = clientKey(request);
  const entry = failures.get(key);
  if (!entry || entry.resetAt <= now) {
    failures.set(key, { count: 1, resetAt: now + LOGIN_WINDOW_MS });
    return;
  }
  entry.count += 1;
}

function clearFailures(request: NextRequest): void {
  failures.delete(clientKey(request));
}

export async function POST(request: NextRequest) {
  if (!verifyCsrf(request)) {
    return NextResponse.json({ error: 'CSRF token missing or invalid', code: 'csrf' }, { status: 403 });
  }

  const retryAfter = loginRetryAfterMs(request);
  if (retryAfter > 0) {
    return NextResponse.json(
      { error: 'Too many failed sign-in attempts. Try again later.', code: 'rate_limited' },
      { status: 429, headers: { 'retry-after': String(Math.ceil(retryAfter / 1000)), 'cache-control': 'no-store' } },
    );
  }

  const config = getAuthConfig();
  if (!config) {
    return NextResponse.json(
      {
        error:
          'Authentication is not configured: set SESSION_SECRET, ADMIN_USERNAME and ADMIN_PASSWORD_HASH (pbkdf2_sha256:... or sha256 hex) in the dashboard environment.',
        code: 'not_configured',
      },
      { status: 503, headers: { 'cache-control': 'no-store' } },
    );
  }

  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return NextResponse.json({ error: 'Request body must be JSON', code: 'bad_request' }, { status: 400 });
  }

  const parsed = loginSchema.safeParse(raw);
  if (!parsed.success) {
    return NextResponse.json(
      { error: parsed.error.issues[0]?.message ?? 'Invalid credentials payload', code: 'bad_request' },
      { status: 400 },
    );
  }

  const { username, password } = parsed.data;
  // Constant-time comparisons on both halves: no short-circuit on the username.
  const usernameOk = timingSafeEqualString(username, config.username);
  const passwordOk = await verifyPassword(password, config.passwordHash);
  if (!usernameOk || !passwordOk) {
    recordFailure(request);
    return NextResponse.json({ error: 'Invalid username or password', code: 'unauthenticated' }, { status: 401 });
  }
  clearFailures(request);

  const now = Math.floor(Date.now() / 1000);
  const token = await signSession({ sub: config.username, iat: now, exp: now + SESSION_TTL_SECONDS });

  const response = NextResponse.json(
    { ok: true, username: config.username },
    { headers: { 'cache-control': 'no-store' } },
  );
  response.cookies.set(COOKIE_SESSION, token, SESSION_COOKIE_OPTIONS);
  if (!request.cookies.get(COOKIE_CSRF)?.value) {
    response.cookies.set(COOKIE_CSRF, newCsrfToken(), CSRF_COOKIE_OPTIONS);
  }
  return response;
}
