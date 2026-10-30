import { NextRequest } from 'next/server';

import { DriveError, accessToken, driveStatus, quota, signOut } from '@/lib/drive';
import { guard, noStore } from '@/lib/session';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

/**
 * GET /kb/api/drive/status[?live=1] - is Drive configured / connected, which
 * account, which backup folder; with live=1 also the account's storage, read
 * from Google. Never returns a token.
 */
export async function GET(request: NextRequest) {
  const { denied } = await guard(request);
  if (denied) return denied;
  const status = await driveStatus();
  if (request.nextUrl.searchParams.get('live') !== '1' || !status.connected) return noStore(status);
  try {
    return noStore({ ...status, quota: await quota(await accessToken()), liveError: null });
  } catch (error) {
    return noStore({ ...status, quota: null, liveError: error instanceof Error ? error.message : 'Drive unreachable' });
  }
}

/** DELETE /kb/api/drive/status - disconnect (the credential is set aside, not deleted). */
export async function DELETE(request: NextRequest) {
  const { session, denied } = await guard(request);
  if (denied) return denied;
  try {
    const was = await signOut(session.sub);
    return noStore({ ok: true, connected: false, changed: was });
  } catch (error) {
    const status = error instanceof DriveError ? error.status : 500;
    return noStore({ error: error instanceof Error ? error.message : 'Disconnect failed', code: 'drive_error' }, status);
  }
}
