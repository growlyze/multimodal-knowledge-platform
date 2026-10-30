import { NextRequest, NextResponse } from 'next/server';

import { COOKIE_SESSION, verifyCsrf, verifySession, type SessionPayload } from './auth';

/**
 * Route-handler guard (defense in depth on top of middleware.ts):
 * a valid signed session, plus the double-submit CSRF token on mutating
 * methods. Returns the session, or the error response to send.
 */
export async function guard(
  request: NextRequest,
): Promise<{ session: SessionPayload; denied: null } | { session: null; denied: NextResponse }> {
  const session = await verifySession(request.cookies.get(COOKIE_SESSION)?.value ?? null);
  if (!session) {
    return {
      session: null,
      denied: NextResponse.json({ error: 'Authentication required', code: 'unauthenticated' }, { status: 401 }),
    };
  }
  if (!['GET', 'HEAD', 'OPTIONS'].includes(request.method) && !verifyCsrf(request)) {
    return {
      session: null,
      denied: NextResponse.json({ error: 'CSRF token missing or invalid', code: 'csrf' }, { status: 403 }),
    };
  }
  return { session, denied: null };
}

export function noStore(body: unknown, status = 200): NextResponse {
  return NextResponse.json(body, { status, headers: { 'cache-control': 'no-store' } });
}
