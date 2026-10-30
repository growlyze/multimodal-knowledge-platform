/**
 * Defensive normalizers for backend payloads.
 *
 * The haystack-api response shapes vary slightly per deployment, so pages read
 * values through these pickers instead of assuming one rigid schema. Nothing
 * here throws: unknown shapes degrade to safe defaults.
 */

export type AnyRecord = Record<string, unknown>;

export function isRecord(value: unknown): value is AnyRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function pickArray(source: unknown, keys: string[] = []): unknown[] {
  if (Array.isArray(source)) return source;
  if (isRecord(source)) {
    for (const key of ['items', 'results', 'data', 'rows', 'list', ...keys]) {
      const value = source[key];
      if (Array.isArray(value)) return value;
    }
  }
  return [];
}

export function pickNumber(source: unknown, keys: string[], fallback = 0): number {
  if (isRecord(source)) {
    for (const key of keys) {
      const value = source[key];
      if (typeof value === 'number' && Number.isFinite(value)) return value;
      if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) {
        return Number(value);
      }
    }
  }
  return fallback;
}

export function pickString(source: unknown, keys: string[], fallback = ''): string {
  if (isRecord(source)) {
    for (const key of keys) {
      const value = source[key];
      if (typeof value === 'string' && value) return value;
      if (typeof value === 'number') return String(value);
    }
  }
  return fallback;
}

export function pickBoolean(source: unknown, keys: string[], fallback = false): boolean {
  if (isRecord(source)) {
    for (const key of keys) {
      const value = source[key];
      if (typeof value === 'boolean') return value;
      if (value === 'true' || value === 1) return true;
      if (value === 'false' || value === 0) return false;
    }
  }
  return fallback;
}

export function pickRecord(source: unknown, keys: string[] = []): AnyRecord | null {
  if (isRecord(source)) {
    for (const key of ['config', 'data', 'result', 'settings', ...keys]) {
      const value = source[key];
      if (isRecord(value)) return value;
    }
  }
  return null;
}

/** Row helper: stable identity for tables built from unknown payloads. */
export function rowKey(row: unknown, index: number, keys: string[] = []): string {
  return pickString(row, ['id', 'source', 'name', 'key', 'video_id', 'path', ...keys], `row-${index}`);
}

/** Merge unknown record entries into a plain object of scalars (settings forms). */
export function scalarEntries(source: unknown): Record<string, string | number | boolean> {
  const out: Record<string, string | number | boolean> = {};
  if (isRecord(source)) {
    const record = pickRecord(source, ['settings', 'config']) ?? source;
    for (const [key, value] of Object.entries(record)) {
      if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
        out[key] = value;
      }
    }
  }
  return out;
}
