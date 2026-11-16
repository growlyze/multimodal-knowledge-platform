import { describe, expect, it } from 'vitest';

import { allowedHosts, isSameOriginRequest } from '@/lib/origin';

// Behind Traefik/Caddy the standalone server sees its bind address in nextUrl.
const behindProxy = allowedHosts({
  nextUrlHost: '0.0.0.0:3000',
  host: 'chat.example.com',
  forwardedHost: 'chat.example.com',
  publicBaseUrl: 'https://chat.example.com',
});

describe('same-origin check', () => {
  it('accepts the public host behind a reverse proxy', () => {
    expect(isSameOriginRequest(null, 'https://chat.example.com/kb/login', behindProxy)).toBe(true);
    expect(isSameOriginRequest('https://chat.example.com', null, behindProxy)).toBe(true);
  });

  it('refuses a foreign site', () => {
    expect(isSameOriginRequest('https://evil.example.org', null, behindProxy)).toBe(false);
    expect(isSameOriginRequest(null, 'https://evil.example.org/page', behindProxy)).toBe(false);
  });

  it('refuses a malformed origin', () => {
    expect(isSameOriginRequest('not a url', null, behindProxy)).toBe(false);
  });

  it('lets non-browser requests through (they still need session + CSRF)', () => {
    expect(isSameOriginRequest(null, null, behindProxy)).toBe(true);
  });

  it('still works without a proxy (local development)', () => {
    const local = allowedHosts({ nextUrlHost: 'localhost:3000', host: 'localhost:3000', forwardedHost: null });
    expect(isSameOriginRequest('http://localhost:3000', null, local)).toBe(true);
  });

  it('uses PUBLIC_BASE_URL even when Host is missing', () => {
    const fromConfig = allowedHosts({
      nextUrlHost: '0.0.0.0:3000',
      host: null,
      forwardedHost: null,
      publicBaseUrl: 'https://chat.example.com',
    });
    expect(isSameOriginRequest('https://chat.example.com', null, fromConfig)).toBe(true);
  });

  it('takes the first X-Forwarded-Host value and ignores case', () => {
    const chained = allowedHosts({ nextUrlHost: '0.0.0.0:3000', host: null, forwardedHost: 'Chat.Example.com, proxy.internal' });
    expect(isSameOriginRequest('https://chat.example.com', null, chained)).toBe(true);
  });
});
