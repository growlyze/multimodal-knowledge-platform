import { NextRequest, NextResponse } from 'next/server';

import { COOKIE_SESSION, isAuthConfigured, verifySession } from '@/lib/auth';

export const dynamic = 'force-dynamic';

/**
 * GET /kb/api/auth/me - session introspection for the UI (never exposes
 * secrets; also reports whether auth is configured at all).
 */
export async function GET(request: NextRequest) {
  const configured = isAuthConfigured();
  const token = request.cookies.get(COOKIE_SESSION)?.value ?? null;
  const session = await verifySession(token);

  return NextResponse.json(
    {
      authenticated: Boolean(session),
      username: session?.sub ?? null,
      configured,
      appName: process.env.APP_NAME || 'Growlyze Multimodal AI Chat',
      expiresAt: session ? session.exp * 1000 : null,
    },
    { headers: { 'cache-control': 'no-store' } },
  );
}
