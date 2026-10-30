import { NextRequest, NextResponse } from 'next/server';

import { COOKIE_SESSION, verifySession } from '@/lib/auth';
import { finishConnect, publicBaseUrl } from '@/lib/drive';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

/**
 * GET /kb/api/drive/callback?code=&state= - Google's OAuth redirect target
 * (register <PUBLIC_BASE_URL>/kb/api/drive/callback on the OAuth client).
 * Exchanges the code (PKCE verifier + single-use state checked server-side),
 * saves the connection and returns to the Integrations page.
 */
export async function GET(request: NextRequest) {
  const base = publicBaseUrl() || request.nextUrl.origin;
  const integrations = new URL('/kb/integrations', base);
  const fail = (message: string) => {
    integrations.searchParams.set('drive', 'error');
    integrations.searchParams.set('message', message.slice(0, 300));
    return NextResponse.redirect(integrations);
  };

  const session = await verifySession(request.cookies.get(COOKIE_SESSION)?.value ?? null);
  if (!session) return fail('Sign in to the dashboard first, then connect Google Drive again.');

  const error = request.nextUrl.searchParams.get('error');
  if (error) return fail(`Google sign-in was not completed (${error}).`);

  try {
    const result = await finishConnect(
      request.nextUrl.searchParams.get('code')?.trim() ?? '',
      request.nextUrl.searchParams.get('state')?.trim() ?? '',
    );
    integrations.searchParams.set('drive', 'connected');
    integrations.searchParams.set('plan', result.plan);
    return NextResponse.redirect(integrations);
  } catch (err) {
    return fail(err instanceof Error ? err.message : 'The Google sign-in failed.');
  }
}
