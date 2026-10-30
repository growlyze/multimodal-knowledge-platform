'use client';

import { useEffect, useId, type ReactNode } from 'react';

import { IconClose } from './icons';
import { Button } from './ui';

/**
 * Right-hand detail drawer. Keyboard accessible: Escape closes, focus moves
 * into the drawer when it opens and returns to the page on close.
 */
export function Drawer({
  open,
  title,
  subtitle,
  onClose,
  children,
  footer,
  width = 'max-w-xl',
}: {
  open: boolean;
  title: ReactNode;
  subtitle?: ReactNode;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  width?: string;
}) {
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-40">
      <div className="absolute inset-0 bg-slate-950/40" aria-hidden="true" onClick={onClose} />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={`absolute inset-y-0 right-0 flex w-full ${width} flex-col bg-white shadow-2xl dark:bg-slate-900`}
      >
        <header className="flex items-start justify-between gap-3 border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <div className="min-w-0">
            <h2 id={titleId} className="truncate text-sm font-semibold text-slate-900 dark:text-white">
              {title}
            </h2>
            {subtitle ? <p className="mt-0.5 truncate text-xs text-slate-500 dark:text-slate-400">{subtitle}</p> : null}
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close panel" autoFocus>
            <IconClose className="h-4 w-4" />
          </Button>
        </header>
        <div className="flex-1 overflow-y-auto px-5 py-4">{children}</div>
        {footer ? (
          <footer className="flex items-center justify-end gap-2 border-t border-slate-200 px-5 py-3 dark:border-slate-800">
            {footer}
          </footer>
        ) : null}
      </aside>
    </div>
  );
}
