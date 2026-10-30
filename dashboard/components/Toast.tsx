'use client';

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';

import { IconCheck, IconClose, IconError, IconInfo, IconWarning } from './icons';
import { cx } from './ui';

export type ToastTone = 'success' | 'error' | 'info' | 'warning';

export interface ToastMessage {
  id: number;
  tone: ToastTone;
  title: string;
  description?: string;
}

interface ToastContextValue {
  toasts: ToastMessage[];
  push: (tone: ToastTone, title: string, description?: string) => void;
  dismiss: (id: number) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

let nextId = 1;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (tone: ToastTone, title: string, description?: string) => {
      const id = nextId++;
      setToasts((current) => [...current.slice(-4), { id, tone, title, description }]);
      window.setTimeout(() => dismiss(id), tone === 'error' ? 12_000 : 6_000);
    },
    [dismiss],
  );

  const value = useMemo(() => ({ toasts, push, dismiss }), [toasts, push, dismiss]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastViewport toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  if (!context) throw new Error('useToast must be used inside <ToastProvider>');
  return context;
}

const TONE_STYLES: Record<ToastTone, { className: string; icon: ReactNode }> = {
  success: {
    className: 'border-emerald-200 bg-white text-slate-800 dark:border-emerald-900 dark:bg-slate-900 dark:text-slate-100',
    icon: <IconCheck className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />,
  },
  error: {
    className: 'border-rose-200 bg-white text-slate-800 dark:border-rose-900 dark:bg-slate-900 dark:text-slate-100',
    icon: <IconError className="h-4 w-4 text-rose-600 dark:text-rose-400" />,
  },
  warning: {
    className: 'border-amber-200 bg-white text-slate-800 dark:border-amber-900 dark:bg-slate-900 dark:text-slate-100',
    icon: <IconWarning className="h-4 w-4 text-amber-600 dark:text-amber-400" />,
  },
  info: {
    className: 'border-brand-200 bg-white text-slate-800 dark:border-brand-900 dark:bg-slate-900 dark:text-slate-100',
    icon: <IconInfo className="h-4 w-4 text-brand-600 dark:text-brand-400" />,
  },
};

function ToastViewport({ toasts, onDismiss }: { toasts: ToastMessage[]; onDismiss: (id: number) => void }) {
  return (
    <div
      aria-live="polite"
      aria-atomic="false"
      className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-full max-w-sm flex-col gap-2"
    >
      {toasts.map((toast) => {
        const tone = TONE_STYLES[toast.tone];
        return (
          <div
            key={toast.id}
            role={toast.tone === 'error' ? 'alert' : 'status'}
            className={cx(
              'pointer-events-auto flex items-start gap-2.5 rounded-lg border px-3.5 py-3 shadow-lg',
              tone.className,
            )}
          >
            <span className="mt-0.5 shrink-0">{tone.icon}</span>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">{toast.title}</p>
              {toast.description ? <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{toast.description}</p> : null}
            </div>
            <button
              type="button"
              onClick={() => onDismiss(toast.id)}
              aria-label="Dismiss notification"
              className="shrink-0 rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 dark:hover:bg-slate-800"
            >
              <IconClose className="h-3.5 w-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
