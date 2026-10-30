import { NextRequest, NextResponse } from 'next/server';

import { COOKIE_CSRF, COOKIE_SESSION, CSRF_COOKIE_OPTIONS, newCsrfToken, verifyCsrf } from '@/lib/auth';

export const dynamic = 'force-dynamic';

/** POST /kb/api/auth/logout - clears the signed session cookie. */
export async function POST(request: NextRequest) {
  if (!verifyCsrf(request)) {
    return NextResponse.json({ error: 'CSRF token missing or invalid', code: 'csrf' }, { status: 403 });
  }

  const response = NextResponse.json({ ok: true }, { headers: { 'cache-control': 'no-store' } });
  response.cookies.set(COOKIE_SESSION, '', { path: '/kb', maxAge: 0, httpOnly: true, secure: true, sameSite: 'lax' });
  // Keep a fresh CSRF token so the login form can immediately re-submit.
  response.cookies.set(COOKIE_CSRF, newCsrfToken(), CSRF_COOKIE_OPTIONS);
  return response;
}
