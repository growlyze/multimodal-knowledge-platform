import { describe, expect, it } from 'vitest';

import {
  channelDiscoverSchema,
  loginSchema,
  parseUrlList,
  restoreSchema,
  searchSchema,
  tableDropSchema,
  uploadMetaSchema,
  videoBatchSchema,
  videoUrlSchema,
  cookiesTextSchema,
} from '@/lib/schemas';

describe('loginSchema', () => {
  it('accepts credentials and rejects blanks', () => {
    expect(loginSchema.safeParse({ username: 'admin', password: 'pw' }).success).toBe(true);
    expect(loginSchema.safeParse({ username: '', password: 'pw' }).success).toBe(false);
    expect(loginSchema.safeParse({ username: 'admin', password: '' }).success).toBe(false);
    expect(loginSchema.safeParse({}).success).toBe(false);
  });
});

describe('uploadMetaSchema', () => {
  it('accepts the six source types (one per POST /ingest/{type})', () => {
    for (const source_type of ['document', 'code', 'transcript', 'structured', 'media', 'archive'] as const) {
      expect(uploadMetaSchema.safeParse({ source_type, collection: 'documents' }).success).toBe(true);
    }
  });

  it('rejects unknown source types; an empty collection means the backend default', () => {
    expect(uploadMetaSchema.safeParse({ source_type: 'video', collection: 'documents' }).success).toBe(false);
    expect(uploadMetaSchema.parse({ source_type: 'document', collection: '  ' }).collection).toBe('');
    expect(uploadMetaSchema.parse({ source_type: 'structured' }).collection).toBe('');
  });
});

describe('searchSchema', () => {
  it('coerces top_k and applies defaults', () => {
    const parsed = searchSchema.parse({ q: 'what is rag?' });
    expect(parsed.top_k).toBe(10);
    expect(parsed.collections).toBe('');

    expect(searchSchema.parse({ q: 'q', top_k: '25', source_type: 'code' })).toMatchObject({
      top_k: 25,
      source_type: 'code',
    });
  });

  it('enforces bounds', () => {
    expect(searchSchema.safeParse({ q: '' }).success).toBe(false);
    expect(searchSchema.safeParse({ q: 'x', top_k: 0 }).success).toBe(false);
    expect(searchSchema.safeParse({ q: 'x', top_k: 51 }).success).toBe(false);
    expect(searchSchema.safeParse({ q: 'x', source_type: 'nope' }).success).toBe(false);
  });
});

describe('tableDropSchema typed confirmation', () => {
  it('requires the exact word DROP', () => {
    expect(tableDropSchema.safeParse({ name: 'sales', confirm: 'DROP' }).success).toBe(true);
    expect(tableDropSchema.safeParse({ name: 'sales', confirm: 'drop' }).success).toBe(false);
    expect(tableDropSchema.safeParse({ name: 'sales', confirm: 'DROP PLEASE' }).success).toBe(false);
    expect(tableDropSchema.safeParse({ name: '', confirm: 'DROP' }).success).toBe(false);
  });
});

describe('video schemas', () => {
  it('validates single URLs', () => {
    expect(videoUrlSchema.safeParse({ url: 'https://youtu.be/abc' }).success).toBe(true);
    expect(videoUrlSchema.safeParse({ url: 'not-a-url' }).success).toBe(false);
    expect(videoUrlSchema.safeParse({ url: '' }).success).toBe(false);
  });

  it('validates batches and parses URL lists from text', () => {
    expect(videoBatchSchema.safeParse({ urls: ['https://a.example/x', 'https://b.example/y'] }).success).toBe(true);
    expect(videoBatchSchema.safeParse({ urls: [] }).success).toBe(false);
    expect(videoBatchSchema.safeParse({ urls: ['nope'] }).success).toBe(false);

    const batch = parseUrlList('https://a.example/1\nhttps://b.example/2, https://c.example/3');
    expect(batch.urls).toEqual(['https://a.example/1', 'https://b.example/2', 'https://c.example/3']);
    expect(() => parseUrlList('   ')).toThrow();
    expect(() => parseUrlList('definitely-not-a-url')).toThrow();
  });

  it('validates channel discovery', () => {
    expect(channelDiscoverSchema.safeParse({ url: 'https://youtube.com/@channel' }).success).toBe(true);
    expect(channelDiscoverSchema.safeParse({ url: 'nope' }).success).toBe(false);
  });
});

describe('cookies schema', () => {
  it('validates cookie text payloads (POST /video/cookies/text {content})', () => {
    expect(cookiesTextSchema.safeParse({ content: '# Netscape HTTP Cookie File' }).success).toBe(true);
    expect(cookiesTextSchema.safeParse({ content: '  ' }).success).toBe(false);
    expect(cookiesTextSchema.safeParse({}).success).toBe(false);
  });
});

describe('restoreSchema', () => {
  it('accepts the five restore kinds only', () => {
    for (const kind of ['restore-docs', 'restore-files', 'restore-chats', 'restore-videos', 'restore-full'] as const) {
      expect(restoreSchema.safeParse({ kind }).success).toBe(true);
    }
    expect(restoreSchema.safeParse({ kind: 'restore-db' }).success).toBe(false);
    expect(restoreSchema.safeParse({ kind: 'restore-full', confirm: 'RESTORE' }).success).toBe(true);
  });
});
