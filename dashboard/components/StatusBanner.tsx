import type { ReactNode } from 'react';

import { IconCheck, IconError, IconInfo, IconWarning } from './icons';
import { cx } from './ui';

export type BannerTone = 'ok' | 'warning' | 'error' | 'info';

const TONES: Record<BannerTone, { className: string; icon: ReactNode }> = {
  ok: {
    className: 'border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-100',
    icon: <IconCheck className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />,
  },
  warning: {
    className: 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-100',
    icon: <IconWarning className="h-5 w-5 text-amber-600 dark:text-amber-400" />,
  },
  error: {
    className: 'border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-100',
    icon: <IconError className="h-5 w-5 text-rose-600 dark:text-rose-400" />,
  },
  info: {
    className: 'border-brand-200 bg-brand-50 text-brand-900 dark:border-brand-900 dark:bg-brand-950/50 dark:text-brand-100',
    icon: <IconInfo className="h-5 w-5 text-brand-600 dark:text-brand-400" />,
  },
};

export function StatusBanner({
  tone = 'info',
  title,
  children,
  actions,
  className,
}: {
  tone?: BannerTone;
  title: string;
  children?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  const style = TONES[tone];
  return (
    <div
      role="status"
      className={cx('flex items-start gap-3 rounded-lg border px-4 py-3', style.className, className)}
    >
      <span className="mt-0.5 shrink-0">{style.icon}</span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold">{title}</p>
        {children ? <div className="mt-1 text-xs opacity-90">{children}</div> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}
