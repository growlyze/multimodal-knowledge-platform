import type { ReactNode } from 'react';

import { Badge, cx, statusTone } from './ui';

/**
 * Compact job progress row used in the jobs strip and job tables.
 */
export function JobProgress({
  label,
  status,
  progress,
  detail,
  error,
  right,
  className,
}: {
  label: string;
  status: string;
  /** 0-100. Omit for an indeterminate bar. */
  progress?: number | null;
  detail?: string;
  error?: string | null;
  right?: ReactNode;
  className?: string;
}) {
  const determinate = typeof progress === 'number' && Number.isFinite(progress);
  const pct = determinate ? Math.max(0, Math.min(100, progress)) : null;
  const running = ['running', 'processing', 'in_progress', 'queued', 'pending', 'scanning'].includes(status.toLowerCase());

  return (
    <div className={cx('rounded-md border border-slate-200 px-3 py-2.5 dark:border-slate-800', className)}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-slate-900 dark:text-white">{label}</p>
          {detail ? <p className="mt-0.5 truncate text-xs text-slate-500 dark:text-slate-400">{detail}</p> : null}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Badge tone={statusTone(status)}>{status}</Badge>
          {right}
        </div>
      </div>

      {running || determinate ? (
        <div
          className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct ?? undefined}
          aria-label={`${label} progress`}
        >
          <div
            className={cx(
              'h-full rounded-full bg-brand-600 transition-all dark:bg-brand-500',
              determinate ? '' : 'animate-pulse w-1/3',
            )}
            style={determinate ? { width: `${pct}%` } : undefined}
          />
        </div>
      ) : null}

      {error ? (
        <p role="alert" className="mt-2 text-xs text-rose-600 dark:text-rose-400">
          {error}
        </p>
      ) : null}
    </div>
  );
}
