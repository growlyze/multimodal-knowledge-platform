import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  getAuthConfig,
  isAuthConfigured,
  isSupportedPasswordHash,
  newCsrfToken,
  pbkdf2Hex,
  parseCookies,
  sha256Hex,
  signSession,
  timingSafeEqualString,
  verifyCsrf,
  verifyPassword,
  verifySession,
  type SessionPayload,
} from '@/lib/auth';

const ENV_KEYS = ['SESSION_SECRET', 'ADMIN_USERNAME', 'ADMIN_PASSWORD_HASH'] as const;

function clearEnv(): void {
  for (const key of ENV_KEYS) delete process.env[key];
}

function setEnv(): void {
  process.env.SESSION_SECRET = 'test-secret-value';
  process.env.ADMIN_USERNAME = 'admin';
  process.env.ADMIN_PASSWORD_HASH = '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8'; // sha256('password')
}

beforeEach(() => {
  clearEnv();
  setEnv();
});

afterEach(() => {
  clearEnv();
});

function payload(overrides: Partial<SessionPayload> = {}): SessionPayload {
  const now = Math.floor(Date.now() / 1000);
  return { sub: 'admin', iat: now, exp: now + 3600, ...overrides };
}

describe('auth configuration (fail-closed)', () => {
  it('returns config when all variables are present', () => {
    expect(isAuthConfigured()).toBe(true);
    expect(getAuthConfig()).toMatchObject({ username: 'admin' });
  });

  it.each(ENV_KEYS)('refuses access when %s is missing', (key) => {
    delete process.env[key];
    expect(getAuthConfig()).toBeNull();
    expect(isAuthConfigured()).toBe(false);
  });

  it('treats blank values as missing', () => {
    process.env.SESSION_SECRET = '   ';
    expect(isAuthConfigured()).toBe(false);
  });

  it('signSession throws when not configured', async () => {
    clearEnv();
    await expect(signSession(payload())).rejects.toThrow(/SESSION_SECRET/);
  });

  it('verifySession returns null when not configured', async () => {
    const token = await signSession(payload());
    clearEnv();
    expect(await verifySession(token)).toBeNull();
  });
});

describe('HMAC session sign/verify', () => {
  it('round-trips a valid payload', async () => {
    const input = payload();
    const token = await signSession(input);
    const output = await verifySession(token);
    expect(output).toEqual(input);
  });

  it('rejects a tampered payload', async () => {
    const token = await signSession(payload());
    const [body, mac] = token.split('.');
    const forgedBody = Buffer.from(JSON.stringify(payload({ sub: 'attacker' })))
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
    expect(await verifySession(`${forgedBody}.${mac}`)).toBeNull();
    expect(await verifySession(`${body}.AAAA${mac.slice(4)}`)).toBeNull();
  });

  it('rejects malformed tokens', async () => {
    expect(await verifySession('')).toBeNull();
    expect(await verifySession('not-a-token')).toBeNull();
    expect(await verifySession('a.b.c')).toBeNull();
    expect(await verifySession(undefined)).toBeNull();
    expect(await verifySession(null)).toBeNull();
  });

  it('rejects an expired session', async () => {
    const now = Math.floor(Date.now() / 1000);
    const token = await signSession(payload({ iat: now - 7200, exp: now - 3600 }));
    expect(await verifySession(token)).toBeNull();
  });

  it('rejects tokens signed with a different secret', async () => {
    const token = await signSession(payload(), 'other-secret');
    expect(await verifySession(token)).toBeNull();
    expect(await verifySession(token, 'other-secret')).not.toBeNull();
  });
});

describe('password hashing', () => {
  it('computes sha256 hex digests', async () => {
    expect(await sha256Hex('password')).toBe(
      '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8',
    );
  });

  it('verifies correct passwords and rejects wrong ones', async () => {
    const hash = await sha256Hex('s3cret');
    expect(await verifyPassword('s3cret', hash)).toBe(true);
    expect(await verifyPassword('s3cret ', hash)).toBe(false);
    expect(await verifyPassword('wrong', hash)).toBe(false);
    // hash comparison is case-insensitive on the hex digest
    expect(await verifyPassword('s3cret', hash.toUpperCase())).toBe(true);
  });

  it('verifies pbkdf2_sha256:<iterations>:<salt>:<hex> hashes', async () => {
    const salt = '0123456789abcdef';
    const digest = await pbkdf2Hex('s3cret', salt, 100_000);
    const stored = `pbkdf2_sha256:100000:${salt}:${digest}`;
    expect(isSupportedPasswordHash(stored)).toBe(true);
    expect(await verifyPassword('s3cret', stored)).toBe(true);
    expect(await verifyPassword('wrong', stored)).toBe(false);
  });

  it('matches Python hashlib.pbkdf2_hmac output (the documented generator)', async () => {
    // python3 -c "import hashlib;print(hashlib.pbkdf2_hmac('sha256',b'password',b'salt',100000).hex())"
    expect(await pbkdf2Hex('password', 'salt', 100_000)).toBe(
      '0394a2ede332c9a13eb82e9b24631604c31df978b4e2f0fbd2c549944f9d79a5',
    );
  });

  it('refuses placeholders and malformed hashes (fail-closed config)', async () => {
    for (const bad of ['change-me', '', 'pbkdf2_sha256:10:salt:' + 'a'.repeat(64), 'pbkdf2_sha256:100000:salt:xyz']) {
      expect(isSupportedPasswordHash(bad)).toBe(false);
      expect(await verifyPassword('anything', bad)).toBe(false);
    }
    process.env.ADMIN_PASSWORD_HASH = 'change-me';
    expect(getAuthConfig()).toBeNull();
  });
});

describe('CSRF double-submit', () => {
  it('accepts a header matching the cookie', () => {
    const token = newCsrfToken();
    const req = new Request('https://example.test/kb/api/proxy/ingest', {
      method: 'POST',
      headers: { cookie: `gmac_csrf=${token}`, 'x-csrf-token': token },
    });
    expect(verifyCsrf(req)).toBe(true);
  });

  it('rejects a missing or mismatched token', () => {
    const token = newCsrfToken();
    const mismatch = new Request('https://example.test/kb/api/proxy/ingest', {
      method: 'POST',
      headers: { cookie: `gmac_csrf=${token}`, 'x-csrf-token': 'other' },
    });
    expect(verifyCsrf(mismatch)).toBe(false);

    const missingHeader = new Request('https://example.test/kb/api/proxy/ingest', {
      method: 'POST',
      headers: { cookie: `gmac_csrf=${token}` },
    });
    expect(verifyCsrf(missingHeader)).toBe(false);

    const missingCookie = new Request('https://example.test/kb/api/proxy/ingest', {
      method: 'POST',
      headers: { 'x-csrf-token': token },
    });
    expect(verifyCsrf(missingCookie)).toBe(false);
  });

  it('generates unique tokens', () => {
    expect(newCsrfToken()).not.toBe(newCsrfToken());
    expect(newCsrfToken()).toMatch(/^[A-Za-z0-9_-]{20,}$/);
  });
});

describe('cookie parsing + constant-time compare', () => {
  it('parses cookie headers', () => {
    expect(parseCookies('a=1; b=hello%20world; broken; c=3')).toEqual({
      a: '1',
      b: 'hello world',
      c: '3',
    });
    expect(parseCookies(null)).toEqual({});
  });

  it('compares strings without leaking length mismatches', () => {
    expect(timingSafeEqualString('abc', 'abc')).toBe(true);
    expect(timingSafeEqualString('abc', 'abd')).toBe(false);
    expect(timingSafeEqualString('abc', 'abcd')).toBe(false);
    expect(timingSafeEqualString('', '')).toBe(true);
  });
});
