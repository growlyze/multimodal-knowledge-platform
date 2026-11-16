import type { ReactNode } from 'react';

import { Skeleton } from './Feedback';
import { cx } from './ui';

export type MetricTone = 'neutral' | 'brand' | 'ok' | 'warning' | 'sky' | 'teal' | 'rose' | 'violet';

/** Icon circle colour per tone. 'brand' is the highlighted (solid orange) card. */
const METRIC_TONES: Record<MetricTone, string> = {
  neutral: 'text-slate-700 ring-slate-200 dark:text-slate-200 dark:ring-slate-700',
  brand: 'bg-white/20 text-white ring-white/30',
  ok: 'text-emerald-600 ring-emerald-200 dark:text-emerald-400 dark:ring-emerald-900',
  warning: 'text-amber-600 ring-amber-200 dark:text-amber-400 dark:ring-amber-900',
  sky: 'text-sky-600 ring-sky-200 dark:text-sky-400 dark:ring-sky-900',
  teal: 'text-teal-600 ring-teal-200 dark:text-teal-400 dark:ring-teal-900',
  rose: 'text-rose-600 ring-rose-200 dark:text-rose-400 dark:ring-rose-900',
  violet: 'text-violet-600 ring-violet-200 dark:text-violet-400 dark:ring-violet-900',
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
  const highlight = tone === 'brand';

  return (
    <div
      className={cx(
        'flex flex-col rounded-3xl p-5 shadow-card',
        highlight ? 'bg-brand-500 text-white' : 'bg-white dark:bg-slate-900',
        className,
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <p className={cx('truncate text-sm font-medium', highlight ? 'text-white/90' : 'text-slate-600 dark:text-slate-300')}>
          {label}
        </p>
        {icon ? (
          <span
            className={cx(
              'flex h-9 w-9 shrink-0 items-center justify-center rounded-full ring-1 [&>svg]:h-4 [&>svg]:w-4',
              METRIC_TONES[tone],
            )}
          >
            {icon}
          </span>
        ) : null}
      </div>
      {loading ? (
        <Skeleton className="mt-3 h-8 w-24" />
      ) : (
        <p className={cx('mt-3 truncate text-3xl font-semibold tracking-tight', highlight ? 'text-white' : 'text-slate-900 dark:text-white')}>
          {value}
        </p>
      )}
      {sub ? (
        <div className={cx('mt-1.5 text-xs', highlight ? 'text-white/85' : 'text-slate-500 dark:text-slate-400')}>{sub}</div>
      ) : null}
    </div>
  );
}
