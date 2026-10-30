'use client';

import { useEffect, useId, useState, type ReactNode } from 'react';

import { Button } from './ui';

/**
 * Accessible modal dialog with an optional typed-confirmation variant:
 * destructive actions can require typing an exact word (e.g. "DROP",
 * "RESTORE") before the confirm button is enabled.
 */
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = 'Confirm',
  confirmWord,
  danger = true,
  loading = false,
  error,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  description?: ReactNode;
  confirmLabel?: string;
  /** When set, the user must type this exact word to enable confirmation. */
  confirmWord?: string;
  danger?: boolean;
  loading?: boolean;
  error?: string | null;
  onConfirm: (typed: string) => void;
  onCancel: () => void;
}) {
  const titleId = useId();
  const inputId = useId();
  const [typed, setTyped] = useState('');

  // Reset the typed confirmation every time the dialog opens.
  useEffect(() => {
    if (open) setTyped('');
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onCancel();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onCancel]);

  if (!open) return null;

  const gateSatisfied = !confirmWord || typed === confirmWord;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-slate-950/50" aria-hidden="true" onClick={onCancel} />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="relative w-full max-w-md rounded-lg bg-white p-5 shadow-xl dark:bg-slate-900"
      >
        <h2 id={titleId} className="text-base font-semibold text-slate-900 dark:text-white">
          {title}
        </h2>
        {description ? <div className="mt-2 text-sm text-slate-600 dark:text-slate-300">{description}</div> : null}

        {confirmWord ? (
          <div className="mt-4">
            <label htmlFor={inputId} className="block text-xs font-medium text-slate-600 dark:text-slate-300">
              Type <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-xs dark:bg-slate-800">{confirmWord}</code>{' '}
              to confirm
            </label>
            <input
              id={inputId}
              type="text"
              value={typed}
              autoComplete="off"
              spellCheck={false}
              autoFocus
              onChange={(event) => setTyped(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && gateSatisfied && !loading) onConfirm(typed);
              }}
              className="mt-1.5 block w-full rounded-md border-0 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm ring-1 ring-inset ring-slate-300 focus:ring-2 focus:ring-inset focus:ring-brand-500 dark:bg-slate-800 dark:text-white dark:ring-slate-600"
            />
          </div>
        ) : null}

        {error ? (
          <p
            role="alert"
            className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950/60 dark:text-rose-300"
          >
            {error}
          </p>
        ) : null}

        <div className="mt-5 flex justify-end gap-2">
          <Button variant="secondary" onClick={onCancel} disabled={loading}>
            Cancel
          </Button>
          <Button
            variant={danger ? 'danger' : 'primary'}
            loading={loading}
            disabled={!gateSatisfied}
            onClick={() => onConfirm(typed)}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}
