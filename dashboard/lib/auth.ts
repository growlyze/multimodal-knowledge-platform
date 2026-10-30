/**
 * Session + CSRF helpers.
 *
 * Uses the Web Crypto API only, so the identical implementation runs both in
 * Node route handlers and in the Edge middleware.
 *
 * Security contract:
 * - Secrets come exclusively from server-side env vars (never NEXT_PUBLIC_*).
 * - The session cookie is HttpOnly, Secure, SameSite=Lax, Path=/kb and its
 *   payload is signed with HMAC-SHA256 keyed by SESSION_SECRET.
 * - Configuration is fail-closed: when SESSION_SECRET, ADMIN_USERNAME or
 *   ADMIN_PASSWORD_HASH are missing, no session can be issued or verified.
 * - Mutating requests must carry an `x-csrf-token` header matching the
 *   `gmac_csrf` cookie (double-submit).
 */

export const COOKIE_SESSION = 'gmac_session';
export const COOKIE_CSRF = 'gmac_csrf';
export const SESSION_TTL_SECONDS = 60 * 60 * 8; // 8 hours

export interface SessionPayload {
  sub: string;
  iat: number;
  exp: number;
}

export interface AuthConfig {
  secret: string;
  username: string;
  passwordHash: string;
}

/**
 * Returns the auth configuration, or null when it is incomplete.
 * Incomplete configuration must NEVER grant access (fail-closed).
 */
export function getAuthConfig(): AuthConfig | null {
  const secret = process.env.SESSION_SECRET?.trim();
  const username = process.env.ADMIN_USERNAME?.trim();
  const passwordHash = process.env.ADMIN_PASSWORD_HASH?.trim();
  if (!secret || !username || !passwordHash) return null;
  // A placeholder or malformed hash is "not configured", never "any password".
  if (!isSupportedPasswordHash(passwordHash)) return null;
  return { secret, username, passwordHash };
}

export function isAuthConfigured(): boolean {
  return getAuthConfig() !== null;
}

export const SESSION_COOKIE_OPTIONS = {
  httpOnly: true,
  secure: true,
  sameSite: 'lax' as const,
  path: '/kb',
  maxAge: SESSION_TTL_SECONDS,
};

export const CSRF_COOKIE_OPTIONS = {
  httpOnly: false, // the browser must read it to echo it in x-csrf-token
  secure: true,
  sameSite: 'lax' as const,
  path: '/kb',
  maxAge: SESSION_TTL_SECONDS,
};

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

function toBase64Url(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(value: string): Uint8Array | null {
  try {
    const normalized = value.replace(/-/g, '+').replace(/_/g, '/');
    const padded = normalized + '='.repeat((4 - (normalized.length % 4)) % 4);
    const binary = atob(padded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return bytes;
  } catch {
    return null;
  }
}

/** Constant-time string comparison (no early return on first difference). */
export function timingSafeEqualString(a: string, b: string): boolean {
  const max = Math.max(a.length, b.length);
  let diff = a.length === b.length ? 0 : 1;
  for (let i = 0; i < max; i += 1) {
    diff |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  }
  return diff === 0;
}

async function hmacSha256(secret: string, data: string): Promise<Uint8Array> {
  const key = await crypto.subtle.importKey(
    'raw',
    textEncoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign('HMAC', key, textEncoder.encode(data));
  return new Uint8Array(signature);
}

/**
 * Signs a session payload. Throws when the deployment is not configured;
 * callers must treat that as "access refused".
 */
export async function signSession(payload: SessionPayload, secret?: string): Promise<string> {
  const key = secret ?? getAuthConfig()?.secret;
  if (!key) throw new Error('SESSION_SECRET is not configured');
  const body = toBase64Url(textEncoder.encode(JSON.stringify(payload)));
  const mac = toBase64Url(await hmacSha256(key, body));
  return `${body}.${mac}`;
}

/**
 * Verifies a session token. Returns the payload or null (fail-closed on any
 * tampering, expiry, malformed input or missing configuration).
 */
export async function verifySession(token: string | null | undefined, secret?: string): Promise<SessionPayload | null> {
  const key = secret ?? getAuthConfig()?.secret;
  if (!key || !token) return null;
  const parts = token.split('.');
  if (parts.length !== 2) return null;
  const [body, mac] = parts;
  if (!body || !mac) return null;
  const expected = toBase64Url(await hmacSha256(key, body));
  if (!timingSafeEqualString(expected, mac)) return null;
  const raw = fromBase64Url(body);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(textDecoder.decode(raw)) as Partial<SessionPayload>;
    if (typeof parsed.sub !== 'string' || !parsed.sub) return null;
    if (typeof parsed.iat !== 'number' || typeof parsed.exp !== 'number') return null;
    if (parsed.exp * 1000 <= Date.now()) return null;
    return { sub: parsed.sub, iat: parsed.iat, exp: parsed.exp };
  } catch {
    return null;
  }
}

/** sha256 hex digest (matches the ADMIN_PASSWORD_HASH format). */
export async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', textEncoder.encode(value));
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * ADMIN_PASSWORD_HASH formats:
 *   pbkdf2_sha256:<iterations>:<salt>:<hex digest>   (recommended)
 *   <64 hex chars>                                   (legacy: plain sha256)
 *
 * Generate the recommended form with:
 *   python3 -c "import hashlib,os,sys;s=os.urandom(16).hex();i=310000;print('pbkdf2_sha256:%d:%s:%s'%(i,s,hashlib.pbkdf2_hmac('sha256',sys.argv[1].encode(),s.encode(),i).hex()))" 'your-password'
 * (colons, not "$", so the value needs no quoting in .env / compose.)
 */
export const PBKDF2_PREFIX = 'pbkdf2_sha256';
const PBKDF2_MIN_ITERATIONS = 100_000;
const PBKDF2_MAX_ITERATIONS = 5_000_000;

export async function pbkdf2Hex(plain: string, salt: string, iterations: number): Promise<string> {
  const key = await crypto.subtle.importKey('raw', textEncoder.encode(plain), 'PBKDF2', false, ['deriveBits']);
  const bits = await crypto.subtle.deriveBits(
    { name: 'PBKDF2', hash: 'SHA-256', salt: textEncoder.encode(salt), iterations },
    key,
    256,
  );
  return Array.from(new Uint8Array(bits))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

/** True when the configured hash is in a format verifyPassword understands. */
export function isSupportedPasswordHash(stored: string): boolean {
  const value = stored.trim();
  if (/^[0-9a-fA-F]{64}$/.test(value)) return true;
  const parts = value.split(':');
  if (parts.length !== 4 || parts[0] !== PBKDF2_PREFIX) return false;
  const iterations = Number(parts[1]);
  return (
    Number.isInteger(iterations) &&
    iterations >= PBKDF2_MIN_ITERATIONS &&
    iterations <= PBKDF2_MAX_ITERATIONS &&
    parts[2].length >= 8 &&
    /^[0-9a-fA-F]{64}$/.test(parts[3])
  );
}

/** Constant-time password check against ADMIN_PASSWORD_HASH (either format). */
export async function verifyPassword(plain: string, stored: string): Promise<boolean> {
  const value = stored.trim();
  if (!isSupportedPasswordHash(value)) {
    // Still burn comparable time so a misconfiguration is not a timing oracle.
    await sha256Hex(plain);
    return false;
  }
  if (value.startsWith(`${PBKDF2_PREFIX}:`)) {
    const [, iterations, salt, digest] = value.split(':');
    const computed = await pbkdf2Hex(plain, salt, Number(iterations));
    return timingSafeEqualString(computed, digest.toLowerCase());
  }
  const computed = await sha256Hex(plain);
  return timingSafeEqualString(computed, value.toLowerCase());
}

export function newCsrfToken(): string {
  return toBase64Url(crypto.getRandomValues(new Uint8Array(32)));
}

export function parseCookies(header: string | null | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  if (!header) return out;
  for (const part of header.split(';')) {
    const idx = part.indexOf('=');
    if (idx <= 0) continue;
    const name = part.slice(0, idx).trim();
    const value = part.slice(idx + 1).trim();
    if (!name) continue;
    try {
      out[name] = decodeURIComponent(value);
    } catch {
      out[name] = value;
    }
  }
  return out;
}

export function getCookieValue(header: string | null | undefined, name: string): string | null {
  return parseCookies(header)[name] ?? null;
}

/** Double-submit CSRF check: `x-csrf-token` header must match the `gmac_csrf` cookie. */
export function verifyCsrf(req: Request): boolean {
  const cookie = getCookieValue(req.headers.get('cookie'), COOKIE_CSRF);
  const header = req.headers.get('x-csrf-token');
  return Boolean(cookie && header && timingSafeEqualString(cookie, header));
}
