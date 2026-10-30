import { NextRequest } from 'next/server';

import { DriveError, accessToken, isDriveConfigured, listFolder } from '@/lib/drive';
import { guard, noStore } from '@/lib/session';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

/**
 * GET /kb/api/drive/browse?folder=<id|root>&page=<token> - read-only listing
 * of one folder of the connected Drive (names, sizes, dates, Google links).
 */
export async function GET(request: NextRequest) {
  const { denied } = await guard(request);
  if (denied) return denied;
  if (!isDriveConfigured()) {
    return noStore({ error: 'Google Drive is not configured on this deployment.', code: 'not_configured' }, 404);
  }
  const folder = request.nextUrl.searchParams.get('folder')?.trim() || 'root';
  const page = request.nextUrl.searchParams.get('page')?.trim() || undefined;
  try {
    return noStore({ folder, ...(await listFolder(await accessToken(), folder, page)) });
  } catch (error) {
    const status = error instanceof DriveError ? error.status : 500;
    return noStore({ error: error instanceof Error ? error.message : 'Listing failed', code: 'drive_error' }, status);
  }
}
