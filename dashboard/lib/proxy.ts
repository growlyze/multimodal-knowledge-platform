/**
 * Allowlist for the backend proxy (app/api/proxy/[...path]/route.ts).
 *
 * Only these haystack-api route prefixes (see haystack-api/main.py and
 * routers/*.py) are reachable from the browser through the dashboard:
 *
 *   /health                     liveness + embedding model name
 *   /admin/*                    stats, documents, tables, storage, backup status
 *   /ingest/*                   uploads (POST /ingest/{type}) + job status
 *   /search                     retrieval test (GET /search)
 *   /eval/*                     query logs, coverage, scores, system metrics
 *   /video/*                    video library, queue, Drive intake, cookies, costs
 *   /costs/*                    retrieval costs
 *   /dashboard/*                non-secret config + provider health (routers/dashboard.py)
 *
 * Deliberately NOT proxied: /extract (Open WebUI's content loader, keyed by
 * EXTERNAL_LOADER_API_KEY) and anything else.
 */

export const PROXY_PREFIXES = ['health', 'admin', 'ingest', 'search', 'eval', 'video', 'costs', 'dashboard'] as const;

/** Prefixes that are a single endpoint (no sub-paths). */
const EXACT_ONLY = new Set<string>(['health', 'search']);

export function isAllowedProxyPath(segments: readonly string[]): boolean {
  if (segments.length === 0) return false;
  for (const segment of segments) {
    if (!segment || segment.includes('\\') || segment.includes('\0')) return false;
    // A segment may carry an encoded "/" (e.g. a document name with folders,
    // DELETE /admin/documents/{source_filename:path}) but never a dot segment.
    if (segment.split('/').some((part) => part === '.' || part === '..')) return false;
  }
  const [head] = segments;
  if (!(PROXY_PREFIXES as readonly string[]).includes(head)) return false;
  if (EXACT_ONLY.has(head)) return segments.length === 1;
  return segments.length >= 2;
}
