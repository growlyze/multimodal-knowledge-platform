import { NextRequest, NextResponse } from 'next/server';

import {
  COOKIE_CSRF,
  COOKIE_SESSION,
  CSRF_COOKIE_OPTIONS,
  newCsrfToken,
  verifyCsrf,
  verifySession,
} from '@/lib/auth';

/**
 * Route protection for the whole dashboard (deployed under the /kb base path):
 * - everything requires a valid signed session except the login page, the
 *   auth endpoints and the health probe;
 * - unauthenticated page requests redirect to /kb/login (with ?next=),
 *   unauthenticated API requests get 401 JSON;
 * - every mutating /kb/api/* request must pass the double-submit CSRF check;
 * - a CSRF cookie is issued when missing so the login form can echo it.
 *
 * The matcher intentionally excludes Next internals; static asset requests
 * (anything with a file extension) pass through untouched.
 */
export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};

const BASE_PATH = '/kb';
const STATIC_EXT = /\.(?:css|js|mjs|map|png|jpe?g|gif|svg|ico|webp|avif|woff2?|ttf|eot|txt|xml|json|mp4|webm)$/i;
const MUTATING = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

/** Strips the base path so route logic is written in app-relative terms. */
function appPath(pathname: string): string {
  const stripped = pathname.startsWith(BASE_PATH) ? pathname.slice(BASE_PATH.length) : pathname;
  return stripped === '' ? '/' : stripped;
}

function withCsrfCookie(request: NextRequest, response: NextResponse): NextResponse {
  if (!request.cookies.get(COOKIE_CSRF)?.value) {
    response.cookies.set(COOKIE_CSRF, newCsrfToken(), CSRF_COOKIE_OPTIONS);
  }
  return response;
}

function jsonError(request: NextRequest, status: number, error: string, code: string): NextResponse {
  return withCsrfCookie(
    request,
    NextResponse.json({ error, code }, { status, headers: { 'cache-control': 'no-store' } }),
  );
}

/**
 * Same-origin guard for state-changing requests: the browser must come from
 * this deployment itself (Origin or Referer). Requests without either header
 * (non-browser clients) already need the session cookie + CSRF token, so only
 * a cross-origin browser request is refused here.
 */
function isSameOrigin(request: NextRequest): boolean {
  const origin = request.headers.get('origin');
  if (origin && origin !== 'null') {
    try {
      return new URL(origin).host === request.nextUrl.host;
    } catch {
      return false;
    }
  }
  const referer = request.headers.get('referer');
  if (referer) {
    try {
      return new URL(referer).host === request.nextUrl.host;
    } catch {
      return false;
    }
  }
  return true;
}

export async function middleware(request: NextRequest): Promise<NextResponse> {
  const rawPath = request.nextUrl.pathname;
  // Next internals and static assets always pass through untouched
  // (with basePath they arrive as /kb/_next/...).
  if (rawPath.includes('/_next/') || rawPath.endsWith('/favicon.ico') || STATIC_EXT.test(rawPath)) {
    return NextResponse.next();
  }

  const path = appPath(rawPath);
  const isApi = path.startsWith('/api/');
  const isHealth = path === '/api/health';
  const isAuthApi = path.startsWith('/api/auth/');
  const isLoginPage = path === '/login';
  const isMutating = MUTATING.has(request.method);

  // CSRF: double-submit token required for every mutating API request, plus a
  // same-origin check so a foreign page cannot drive state changes.
  if (isApi && isMutating && !isSameOrigin(request)) {
    return jsonError(request, 403, 'Cross-origin request refused', 'origin');
  }
  if (isApi && isMutating && !verifyCsrf(request)) {
    return jsonError(request, 403, 'CSRF token missing or invalid', 'csrf');
  }

  if (isHealth) {
    return withCsrfCookie(request, NextResponse.next());
  }

  const sessionToken = request.cookies.get(COOKIE_SESSION)?.value;
  const session = sessionToken ? await verifySession(sessionToken) : null;

  // Public auth endpoints and the login page.
  if (isAuthApi) {
    return withCsrfCookie(request, NextResponse.next());
  }
  if (isLoginPage) {
    if (session) {
      // Already signed in: bounce away from the login screen.
      return withCsrfCookie(request, NextResponse.redirect(new URL(BASE_PATH + '/', request.url)));
    }
    return withCsrfCookie(request, NextResponse.next());
  }

  if (!session) {
    if (isApi) {
      return jsonError(request, 401, 'Authentication required', 'unauthenticated');
    }
    const loginUrl = new URL(BASE_PATH + '/login', request.url);
    const next = rawPath + (request.nextUrl.search || '');
    loginUrl.searchParams.set('next', next);
    return withCsrfCookie(request, NextResponse.redirect(loginUrl));
  }

  return withCsrfCookie(request, NextResponse.next());
}
