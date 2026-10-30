import { describe, expect, it } from 'vitest';

import { isAllowedProxyPath } from '@/lib/proxy';

const allowed = (path: string) => isAllowedProxyPath(path.split('/'));

describe('backend proxy allowlist', () => {
  it('allows the routes the dashboard uses', () => {
    for (const path of [
      'health',
      'search',
      'admin/stats',
      'admin/documents/report.pdf',
      'ingest/document',
      'ingest/status',
      'eval/queries',
      'video/queue/batch',
      'video/drive-intake/scan',
      'costs/retrieval',
      'dashboard/config',
      'dashboard/providers/health',
    ]) {
      expect(allowed(path), path).toBe(true);
    }
  });

  it('allows an encoded slash inside a document name', () => {
    expect(isAllowedProxyPath(['admin', 'documents', 'folder/report.pdf'])).toBe(true);
  });

  it('refuses everything else', () => {
    for (const path of ['extract/process', 'docs', 'openapi.json', 'admin', 'video', 'health/x', 'search/x', '']) {
      expect(allowed(path), path).toBe(false);
    }
    expect(isAllowedProxyPath([])).toBe(false);
    expect(isAllowedProxyPath(['admin', '..', 'extract'])).toBe(false);
    expect(isAllowedProxyPath(['admin', 'documents', 'a/../../extract'])).toBe(false);
    expect(isAllowedProxyPath(['admin', 'x\\y'])).toBe(false);
  });
});
