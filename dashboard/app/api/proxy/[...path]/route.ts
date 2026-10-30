import { NextRequest, NextResponse } from 'next/server';

import { adminAuthHeaders, adminApiKey, haystackBaseUrl } from '@/lib/api';
import { COOKIE_SESSION, verifyCsrf, verifySession } from '@/lib/auth';
import { isAllowedProxyPath } from '@/lib/proxy';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

/**
 * Authenticated proxy to the haystack-api backend.
 *
 * GET /kb/api/proxy/<path>  ->  GET ${HAYSTACK_API_URL}/<path>
 * (also POST / PUT / PATCH / DELETE)
 *
 * Every request is made server-side with the ADMIN_API_KEY credential
 * (haystack-api/services/auth.py: "Authorization: Bearer <ADMIN_API_KEY>");
 * the key is never exposed to the browser. Only the route prefixes in
 * lib/proxy.ts are reachable - the proxy is not a general tunnel.
 */

type RouteContext = { params: { path: string[] } };

const MUTATING = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);
const PROXY_TIMEOUT_MS = 300_000;

async function handle(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  // Defense in depth: middleware already gates /kb/api/*, re-check here too.
  const token = request.cookies.get(COOKIE_SESSION)?.value ?? null;
  const session = await verifySession(token);
  if (!session) {
    return NextResponse.json({ error: 'Authentication required', code: 'unauthenticated' }, { status: 401 });
  }
  if (MUTATING.has(request.method) && !verifyCsrf(request)) {
    return NextResponse.json({ error: 'CSRF token missing or invalid', code: 'csrf' }, { status: 403 });
  }

  const segments = context.params.path ?? [];
  const path = segments.join('/');
  if (!isAllowedProxyPath(segments)) {
    return NextResponse.json(
      { error: `Proxying to "/${path}" is not allowed`, code: 'forbidden' },
      { status: 403 },
    );
  }

  if (!adminApiKey()) {
    return NextResponse.json(
      { error: 'ADMIN_API_KEY is not configured on the dashboard', code: 'not_configured' },
      { status: 503 },
    );
  }

  const target = `${haystackBaseUrl()}/${segments.map(encodeURIComponent).join('/')}${request.nextUrl.search}`;
  const headers: Record<string, string> = {
    accept: request.headers.get('accept') ?? 'application/json',
    ...adminAuthHeaders(),
  };

  let body: BodyInit | undefined;
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    const contentType = request.headers.get('content-type') ?? '';
    if (contentType.includes('multipart/form-data')) {
      // Rebuild the form so fetch sets its own multipart boundary.
      const form = await request.formData();
      const forwarded = new FormData();
      form.forEach((value, key) => {
        forwarded.append(key, value);
      });
      body = forwarded;
    } else if (contentType) {
      headers['content-type'] = contentType;
      body = await request.text();
    } else {
      const raw = await request.arrayBuffer();
      body = raw.byteLength > 0 ? raw : undefined;
    }
  }

  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body,
      cache: 'no-store',
      redirect: 'manual',
      signal: AbortSignal.timeout(PROXY_TIMEOUT_MS),
    });

    const text = await upstream.text();
    return new NextResponse(text, {
      status: upstream.status,
      headers: {
        'content-type': upstream.headers.get('content-type') ?? 'application/json',
        'cache-control': 'no-store',
      },
    });
  } catch (error) {
    const aborted = error instanceof Error && (error.name === 'AbortError' || error.name === 'TimeoutError');
    return NextResponse.json(
      {
        error: aborted ? `Backend request timed out: /${path}` : `Backend unreachable: /${path}`,
        code: aborted ? 'timeout' : 'upstream_unreachable',
      },
      { status: 502 },
    );
  }
}

export function GET(request: NextRequest, context: RouteContext) {
  return handle(request, context);
}
export function POST(request: NextRequest, context: RouteContext) {
  return handle(request, context);
}
export function PUT(request: NextRequest, context: RouteContext) {
  return handle(request, context);
}
export function PATCH(request: NextRequest, context: RouteContext) {
  return handle(request, context);
}
export function DELETE(request: NextRequest, context: RouteContext) {
  return handle(request, context);
}
