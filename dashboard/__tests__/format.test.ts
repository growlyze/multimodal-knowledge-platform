import { describe, expect, it } from 'vitest';

import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatNumber,
  formatPercent,
  formatUSD,
  relativeTime,
} from '@/lib/format';

describe('formatBytes', () => {
  it('formats byte counts across units', () => {
    expect(formatBytes(0)).toBe('0.0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1024)).toBe('1 KB');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(1024 * 1024 * 3.25)).toBe('3.3 MB');
    expect(formatBytes(1024 ** 3)).toBe('1 GB');
    expect(formatBytes(1024 ** 4 * 2)).toBe('2 TB');
  });

  it('handles negatives and non-finite input', () => {
    expect(formatBytes(-2048)).toBe('-2 KB');
    expect(formatBytes(Number.NaN)).toBe('—');
  });

  it('honours the decimals argument', () => {
    expect(formatBytes(1536, 0)).toBe('2 KB');
    expect(formatBytes(1536, 2)).toBe('1.5 KB');
  });
});

describe('formatUSD', () => {
  it('always renders 4 decimals', () => {
    expect(formatUSD(0)).toBe('$0.0000');
    expect(formatUSD(1)).toBe('$1.0000');
    expect(formatUSD(0.00005)).toBe('$0.0001');
    expect(formatUSD(12.34567)).toBe('$12.3457');
    expect(formatUSD(-0.5)).toBe('$-0.5000');
    expect(formatUSD(Number.NaN)).toBe('$—');
  });
});

describe('relativeTime', () => {
  const now = new Date('2026-10-30T12:00:00Z');

  it('renders past and future values', () => {
    expect(relativeTime(new Date('2026-10-30T11:59:40Z'), now)).toBe('just now');
    expect(relativeTime(new Date('2026-10-30T11:55:00Z'), now)).toBe('5 minutes ago');
    expect(relativeTime(new Date('2026-10-30T09:00:00Z'), now)).toBe('3 hours ago');
    expect(relativeTime(new Date('2026-10-28T12:00:00Z'), now)).toBe('2 days ago');
    expect(relativeTime(new Date('2026-10-30T14:00:00Z'), now)).toBe('in 2 hours');
    expect(relativeTime(new Date('2026-11-01T12:00:00Z'), now)).toBe('in 2 days');
  });

  it('accepts strings and timestamps and rejects invalid input', () => {
    expect(relativeTime('2026-10-30T11:59:50Z', now)).toBe('just now');
    expect(relativeTime(now.getTime() - 60 * 60 * 1000, now)).toBe('1 hour ago');
    expect(relativeTime('not-a-date', now)).toBe('—');
  });
});

describe('formatDuration', () => {
  it('renders second/minute/hour/day granularity', () => {
    expect(formatDuration(37)).toBe('37s');
    expect(formatDuration(60)).toBe('1m');
    expect(formatDuration(252)).toBe('4m 12s');
    expect(formatDuration(3900)).toBe('1h 5m');
    expect(formatDuration(3600)).toBe('1h');
    expect(formatDuration(86400 * 2 + 3600 * 3)).toBe('2d 3h');
    expect(formatDuration(0)).toBe('0s');
    expect(formatDuration(-5)).toBe('—');
    expect(formatDuration(Number.NaN)).toBe('—');
  });
});

describe('misc formatters', () => {
  it('formats numbers with grouping', () => {
    expect(formatNumber(1234567)).toBe('1,234,567');
    expect(formatNumber(0)).toBe('0');
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe('—');
  });

  it('formats percents', () => {
    expect(formatPercent(0.4172)).toBe('41.7%');
    expect(formatPercent(1)).toBe('100.0%');
    expect(formatPercent(Number.NaN)).toBe('—');
  });

  it('formats date-times as local YYYY-MM-DD HH:MM', () => {
    const date = new Date(2026, 9, 30, 14, 5, 42);
    expect(formatDateTime(date)).toBe('2026-10-30 14:05');
    expect(formatDateTime('invalid')).toBe('—');
  });
});
