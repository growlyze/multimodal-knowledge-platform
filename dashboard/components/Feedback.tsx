import type { ReactNode } from 'react';

import { IconError, IconInfo, IconWarning } from './icons';
import { Button, Spinner, cx } from './ui';

/* --------------------------------------------------------------- Skeleton */

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={cx('animate-pulse rounded-md bg-slate-200/80 dark:bg-slate-700/60', className)}
    />
  );
}

export function SkeletonRows({ rows = 5, columns = 4 }: { rows?: number; columns?: number }) {
  return (
    <div className="space-y-2" aria-hidden="true">
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div key={rowIndex} className="flex gap-3">
          {Array.from({ length: columns }).map((__, colIndex) => (
            <Skeleton key={colIndex} className={cx('h-8', colIndex === 0 ? 'w-1/4' : 'flex-1')} />
          ))}
        </div>
      ))}
    </div>
  );
}

export function LoadingBlock({ label = 'Loading…', className }: { label?: string; className?: string }) {
  return (
    <div className={cx('flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400', className)}>
      <Spinner className="h-4 w-4" />
      <span>{label}</span>
    </div>
  );
}

/* ------------------------------------------------------------- EmptyState */

export function EmptyState({
  title,
  description,
  action,
  icon,
  className,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  icon?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx('flex flex-col items-center justify-center gap-2 px-6 py-10 text-center', className)}>
      <div className="mb-1 text-slate-300 dark:text-slate-600">{icon ?? <IconInfo className="h-8 w-8" />}</div>
      <p className="text-sm font-medium text-slate-900 dark:text-white">{title}</p>
      {description ? <p className="max-w-md text-xs text-slate-500 dark:text-slate-400">{description}</p> : null}
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}

/* -------------------------------------------------------------- ErrorState */

export function ErrorState({
  error,
  onRetry,
  className,
  title = 'Something went wrong',
}: {
  error: string;
  onRetry?: () => void;
  className?: string;
  title?: string;
}) {
  return (
    <div
      role="alert"
      className={cx(
        'flex flex-col items-start gap-2 rounded-md border border-rose-200 bg-rose-50 px-4 py-3 dark:border-rose-900 dark:bg-rose-950/50',
        className,
      )}
    >
      <div className="flex items-start gap-2">
        <IconError className="mt-0.5 h-4 w-4 shrink-0 text-rose-600 dark:text-rose-400" />
        <div>
          <p className="text-sm font-medium text-rose-800 dark:text-rose-200">{title}</p>
          <p className="mt-0.5 text-xs text-rose-700 dark:text-rose-300">{error}</p>
        </div>
      </div>
      {onRetry ? (
        <Button variant="secondary" size="sm" onClick={onRetry}>
          Retry
        </Button>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------ Unavailable */

/**
 * A feature endpoint that 404s must never break the page: render a calm,
 * explicit "not available on this deployment" notice instead.
 */
export function UnavailableNotice({ feature, detail }: { feature: string; detail?: string }) {
  return (
    <div
      role="status"
      className="flex items-start gap-2 rounded-md border border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-800 dark:bg-slate-900"
    >
      <IconWarning className="mt-0.5 h-4 w-4 shrink-0 text-slate-500 dark:text-slate-400" />
      <div>
        <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
          This data source is not available on this deployment
        </p>
        <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
          {detail ?? `${feature} is disabled or not routed in this environment. Nothing is broken — the rest of the page keeps working.`}
        </p>
      </div>
    </div>
  );
}
