import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/** Container health probe (Docker HEALTHCHECK hits /kb/api/health). */
export function GET() {
  return NextResponse.json({ status: 'ok' }, { headers: { 'cache-control': 'no-store' } });
}
