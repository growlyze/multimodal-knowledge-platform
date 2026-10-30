import type { ReactNode } from 'react';

import { Skeleton } from './Feedback';
import { cx } from './ui';

export type MetricTone = 'neutral' | 'brand' | 'ok' | 'warning' | 'sky' | 'teal' | 'rose' | 'violet';

/** [icon tile, left edge] per tone. */
const METRIC_TONES: Record<MetricTone, [string, string]> = {
  // Neutral cards take the current section's accent (--accent, set by Shell).
  neutral: ['bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300', 'border-l-[color:var(--accent)]'],
  brand: ['bg-brand-50 text-brand-700 dark:bg-brand-950 dark:text-brand-300', 'border-l-brand-500'],
  ok: ['bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300', 'border-l-emerald-500'],
  warning: ['bg-amber-50 text-amber-700 dark:bg-amber-950 dark:text-amber-300', 'border-l-amber-500'],
  sky: ['bg-sky-50 text-sky-700 dark:bg-sky-950 dark:text-sky-300', 'border-l-sky-500'],
  teal: ['bg-teal-50 text-teal-700 dark:bg-teal-950 dark:text-teal-300', 'border-l-teal-500'],
  rose: ['bg-rose-50 text-rose-700 dark:bg-rose-950 dark:text-rose-300', 'border-l-rose-500'],
  violet: ['bg-violet-50 text-violet-700 dark:bg-violet-950 dark:text-violet-300', 'border-l-violet-500'],
};

export function MetricCard({
  label,
  value,
  sub,
  icon,
  tone = 'neutral',
  loading = false,
  className,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  icon?: ReactNode;
  tone?: MetricTone;
  loading?: boolean;
  className?: string;
}) {
  const [toneClass, edgeClass] = METRIC_TONES[tone];

  return (
    <div
      className={cx(
        'flex items-start gap-3 rounded-lg border-l-4 bg-white p-4 shadow-card ring-1 ring-slate-200 dark:bg-slate-900 dark:ring-slate-800',
        edgeClass,
        className,
      )}
    >
      {icon ? (
        <span className={cx('flex h-10 w-10 shrink-0 items-center justify-center rounded-lg', toneClass)}>{icon}</span>
      ) : null}
      <div className="min-w-0 flex-1">
        <p className="truncate text-xs font-medium text-slate-500 dark:text-slate-400">{label}</p>
        {loading ? (
          <Skeleton className="mt-1.5 h-7 w-20" />
        ) : (
          <p className="mt-1 truncate text-2xl font-semibold text-slate-900 dark:text-white">{value}</p>
        )}
        {sub ? <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{sub}</div> : null}
      </div>
    </div>
  );
}
