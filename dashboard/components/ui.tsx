import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes, TextareaHTMLAttributes } from 'react';

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ');
}

/* ---------------------------------------------------------------- Spinner */

export function Spinner({ className = 'h-5 w-5', label }: { className?: string; label?: string }) {
  return (
    <span role="status" aria-label={label ?? 'Loading'} className="inline-flex items-center">
      <svg
        className={cx('animate-spin text-brand-600 dark:text-brand-400', className)}
        viewBox="0 0 24 24"
        fill="none"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
        <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </svg>
      {label ? <span className="sr-only">{label}</span> : null}
    </span>
  );
}

/* ------------------------------------------------------------------ Button */

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md';

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  // Follows the current section's accent (--accent, set by Shell).
  primary:
    'bg-[color:var(--accent,#4f46e5)] text-white hover:brightness-110 focus-visible:ring-[color:var(--accent,#4f46e5)]',
  secondary:
    'bg-white text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 focus-visible:ring-brand-500 dark:bg-slate-800 dark:text-slate-200 dark:ring-slate-600 dark:hover:bg-slate-700',
  ghost:
    'bg-transparent text-slate-600 hover:bg-slate-100 hover:text-slate-900 focus-visible:ring-brand-500 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white',
  danger:
    'bg-rose-600 text-white hover:bg-rose-700 focus-visible:ring-rose-500 disabled:bg-rose-300 dark:disabled:bg-rose-900',
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: 'h-8 px-2.5 text-xs gap-1.5',
  md: 'h-9 px-3.5 text-sm gap-2',
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled,
  className,
  children,
  type = 'button',
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cx(
        'inline-flex items-center justify-center rounded-md font-medium transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-slate-950',
        'disabled:cursor-not-allowed disabled:opacity-60',
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className,
      )}
      {...rest}
    >
      {loading ? <Spinner className="h-4 w-4" /> : null}
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ Fields */

const FIELD_CLASSES =
  'block w-full rounded-md border-0 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm ring-1 ring-inset ring-slate-300 placeholder:text-slate-400 focus:ring-2 focus:ring-inset focus:ring-brand-500 disabled:bg-slate-50 disabled:text-slate-500 dark:bg-slate-800 dark:text-slate-100 dark:ring-slate-600 dark:placeholder:text-slate-500 dark:disabled:bg-slate-900';

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cx(FIELD_CLASSES, className)} {...rest} />;
}

export function Textarea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={cx(FIELD_CLASSES, 'min-h-[80px]', className)} {...rest} />;
}

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cx(FIELD_CLASSES, 'pr-8', className)} {...rest}>
      {children}
    </select>
  );
}

export function Field({
  label,
  hint,
  htmlFor,
  required,
  children,
  className,
}: {
  label: ReactNode;
  hint?: ReactNode;
  htmlFor?: string;
  required?: boolean;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx('space-y-1.5', className)}>
      <label htmlFor={htmlFor} className="block text-xs font-medium text-slate-600 dark:text-slate-300">
        {label}
        {required ? (
          <span className="ml-0.5 text-rose-600 dark:text-rose-400" aria-hidden="true">
            *
          </span>
        ) : null}
      </label>
      {children}
      {hint ? <p className="text-xs text-slate-500 dark:text-slate-400">{hint}</p> : null}
    </div>
  );
}

/* ------------------------------------------------------------------- Panel */

export function Panel({
  title,
  description,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={cx(
        'overflow-hidden rounded-lg bg-white shadow-card ring-1 ring-slate-200 dark:bg-slate-900 dark:ring-slate-800',
        title || actions ? 'border-t-[3px] border-t-[color:var(--accent)]' : null,
        className,
      )}
    >
      {title || actions ? (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-800 dark:bg-slate-900/60 sm:px-5">
          <div className="min-w-0">
            <h2 className="text-[15px] font-semibold text-slate-900 dark:text-white">{title}</h2>
            {description ? (
              <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{description}</p>
            ) : null}
          </div>
          {actions ? <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div> : null}
        </header>
      ) : null}
      <div className={cx('px-4 py-4 sm:px-5', bodyClassName)}>{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------------- Badge */

export type BadgeTone = 'neutral' | 'ok' | 'warning' | 'error' | 'info';

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: 'bg-slate-100 text-slate-700 ring-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:ring-slate-700',
  ok: 'bg-emerald-50 text-emerald-700 ring-emerald-200 dark:bg-emerald-950 dark:text-emerald-300 dark:ring-emerald-900',
  warning: 'bg-amber-50 text-amber-700 ring-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-900',
  error: 'bg-rose-50 text-rose-700 ring-rose-200 dark:bg-rose-950 dark:text-rose-300 dark:ring-rose-900',
  info: 'bg-brand-50 text-brand-700 ring-brand-200 dark:bg-brand-950 dark:text-brand-300 dark:ring-brand-900',
};

export function Badge({ tone = 'neutral', children, className }: { tone?: BadgeTone; children: ReactNode; className?: string }) {
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset',
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/** Maps common job/status words to badge tones. */
export function statusTone(status: string | undefined | null): BadgeTone {
  const value = (status ?? '').toLowerCase();
  if (['ok', 'healthy', 'running', 'completed', 'complete', 'done', 'success', 'connected', 'ready', 'active'].includes(value)) {
    return 'ok';
  }
  if (['queued', 'pending', 'in_progress', 'processing', 'scanning', 'waiting', 'scheduled'].includes(value)) return 'info';
  if (['warning', 'degraded', 'partial', 'skipped', 'cancelled', 'canceled', 'paused'].includes(value)) return 'warning';
  if (['error', 'failed', 'failure', 'unhealthy', 'down', 'missing'].includes(value)) return 'error';
  return 'neutral';
}

/* --------------------------------------------------------------- PageHead */

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 className="text-lg font-semibold text-slate-900 dark:text-white">{title}</h1>
        {description ? <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

/* ------------------------------------------------------------ Key/Value */

export function KeyValue({ label, value }: { label: ReactNode; value: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <dt className="text-xs font-medium text-slate-500 dark:text-slate-400">{label}</dt>
      <dd className="break-all text-right text-xs text-slate-900 dark:text-slate-100">{value}</dd>
    </div>
  );
}
