/**
 * Same-origin check for state-changing requests.
 *
 * Behind a reverse proxy (Caddy, Traefik, nginx) the standalone Next.js server
 * reports its own bind address in request.nextUrl (e.g. 0.0.0.0:3000), not the
 * public hostname, so comparing against nextUrl alone refuses every browser
 * request. The public host is taken from the Host header (set by the browser,
 * not forgeable cross-site), X-Forwarded-Host (set by the proxy) and
 * PUBLIC_BASE_URL.
 */

export interface HostSources {
  nextUrlHost: string;
  host: string | null;
  forwardedHost: string | null;
  publicBaseUrl?: string;
}

export function allowedHosts({ nextUrlHost, host, forwardedHost, publicBaseUrl }: HostSources): Set<string> {
  const hosts = new Set<string>();
  const add = (value: string | null | undefined) => {
    const trimmed = value?.split(',')[0]?.trim().toLowerCase();
    if (trimmed) hosts.add(trimmed);
  };
  add(nextUrlHost);
  add(host);
  add(forwardedHost);
  if (publicBaseUrl) {
    try {
      add(new URL(publicBaseUrl).host);
    } catch {
      /* malformed PUBLIC_BASE_URL: ignore */
    }
  }
  return hosts;
}

function hostOf(url: string): string | null {
  try {
    return new URL(url).host.toLowerCase();
  } catch {
    return null;
  }
}

/**
 * True when the browser request comes from one of `allowed` (Origin first,
 * then Referer). Requests with neither header (non-browser clients) pass here;
 * they still need the session cookie and CSRF token.
 */
export function isSameOriginRequest(origin: string | null, referer: string | null, allowed: Set<string>): boolean {
  if (origin && origin !== 'null') {
    const host = hostOf(origin);
    return host !== null && allowed.has(host);
  }
  if (referer) {
    const host = hostOf(referer);
    return host !== null && allowed.has(host);
  }
  return true;
}
