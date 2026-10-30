/**
 * Formatting helpers shared by the dashboard UI and unit tests.
 * Pure functions only (no I/O, no globals) so they stay trivially testable.
 */

const SIZE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'] as const;

/** Human readable byte size using a 1024 base (e.g. 1536 -> "1.5 KB"). */
export function formatBytes(bytes: number, decimals = 1): string {
  if (!Number.isFinite(bytes)) return '—';
  const negative = bytes < 0;
  let value = Math.abs(bytes);
  if (value < 1) return `${negative ? '-' : ''}${value.toFixed(Math.max(0, decimals))} B`;
  let unit = 0;
  while (value >= 1024 && unit < SIZE_UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const rounded = unit === 0 ? Math.round(value) : Number(value.toFixed(Math.max(0, decimals)));
  return `${negative ? '-' : ''}${rounded} ${SIZE_UNITS[unit]}`;
}

/** USD amount with exactly 4 decimals (costs are tracked per 1/10000 of a cent). */
export function formatUSD(value: number): string {
  if (!Number.isFinite(value)) return '$—';
  return `$${value.toFixed(4)}`;
}

function toDate(input: string | number | Date): Date | null {
  const date = input instanceof Date ? input : new Date(input);
  return Number.isNaN(date.getTime()) ? null : date;
}

/** Compact relative time ("just now", "3 minutes ago", "in 2 days"). */
export function relativeTime(input: string | number | Date, now: Date = new Date()): string {
  const date = toDate(input);
  if (!date) return '—';
  const diffMs = date.getTime() - now.getTime();
  const absSec = Math.abs(diffMs) / 1000;

  const units: Array<[number, Intl.RelativeTimeFormatUnit]> = [
    [60, 'second'],
    [60, 'minute'],
    [24, 'hour'],
    [7, 'day'],
    [4.34524, 'week'],
    [12, 'month'],
    [Number.POSITIVE_INFINITY, 'year'],
  ];

  if (absSec < 45) return 'just now';

  let amount = absSec;
  let unitIndex = 0;
  while (unitIndex < units.length - 1 && amount >= units[unitIndex][0]) {
    amount /= units[unitIndex][0];
    unitIndex += 1;
  }
  const rounded = Math.round(amount);
  const unit = units[Math.min(unitIndex, units.length - 1)][1];
  const formatter = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
  return formatter.format(diffMs >= 0 ? rounded : -rounded, unit);
}

/** Duration in seconds -> "2d 3h", "1h 5m", "4m 12s", "37s". */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const total = Math.floor(seconds);
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;

  if (days > 0) return hours > 0 ? `${days}d ${hours}h` : `${days}d`;
  if (hours > 0) return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
  if (minutes > 0) return secs > 0 ? `${minutes}m ${secs}s` : `${minutes}m`;
  return `${secs}s`;
}

/** Thousands-separated integer/decimal (e.g. 1234567 -> "1,234,567"). */
export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return new Intl.NumberFormat('en-US').format(value);
}

/** Local date-time string, stable output for tables ("2026-10-30 14:05"). */
export function formatDateTime(input: string | number | Date): string {
  const date = toDate(input);
  if (!date) return '—';
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** Percent with one decimal (0.4172 -> "41.7%"). */
export function formatPercent(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(1)}%`;
}
