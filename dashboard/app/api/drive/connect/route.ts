import { NextRequest } from 'next/server';

import { DriveError, startConnect } from '@/lib/drive';
import { guard, noStore } from '@/lib/session';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

/**
 * POST /kb/api/drive/connect - starts the Google sign-in (PKCE + state kept
 * server-side). Returns Google's consent URL; the browser navigates there and
 * Google returns to GET /kb/api/drive/callback.
 */
export async function POST(request: NextRequest) {
  const { session, denied } = await guard(request);
  if (denied) return denied;
  try {
    return noStore({ url: await startConnect(session.sub) });
  } catch (error) {
    const status = error instanceof DriveError ? error.status : 500;
    const code = status === 400 ? 'not_configured' : 'drive_error';
    return noStore({ error: error instanceof Error ? error.message : 'Could not start the sign-in', code }, status);
  }
}
