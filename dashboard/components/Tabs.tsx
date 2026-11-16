'use client';

import { useRef, type ReactNode } from 'react';

import { cx } from './ui';

export interface TabItem {
  id: string;
  label: string;
  /** Optional count badge, e.g. number of jobs. */
  count?: number;
}

/**
 * WAI-ARIA tablist with arrow-key navigation and roving tabindex.
 */
export function Tabs({
  tabs,
  active,
  onChange,
  className,
  ariaLabel = 'Sections',
}: {
  tabs: TabItem[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
  ariaLabel?: string;
}) {
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});

  const move = (delta: number) => {
    const index = tabs.findIndex((tab) => tab.id === active);
    const next = tabs[(index + delta + tabs.length) % tabs.length];
    onChange(next.id);
    refs.current[next.id]?.focus();
  };

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cx('flex w-fit max-w-full gap-1 overflow-x-auto rounded-full bg-white p-1 shadow-card dark:bg-slate-900', className)}
      onKeyDown={(event) => {
        if (event.key === 'ArrowRight') {
          event.preventDefault();
          move(1);
        } else if (event.key === 'ArrowLeft') {
          event.preventDefault();
          move(-1);
        } else if (event.key === 'Home') {
          event.preventDefault();
          onChange(tabs[0].id);
          refs.current[tabs[0].id]?.focus();
        } else if (event.key === 'End') {
          event.preventDefault();
          const last = tabs[tabs.length - 1];
          onChange(last.id);
          refs.current[last.id]?.focus();
        }
      }}
    >
      {tabs.map((tab) => {
        const selected = tab.id === active;
        return (
          <button
            key={tab.id}
            ref={(node) => {
              refs.current[tab.id] = node;
            }}
            type="button"
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={selected}
            aria-controls={`tabpanel-${tab.id}`}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(tab.id)}
            className={cx(
              'whitespace-nowrap rounded-full px-4 py-1.5 text-sm font-medium transition-colors',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500',
              selected
                ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900'
                : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200',
            )}
          >
            {tab.label}
            {typeof tab.count === 'number' ? (
              <span
                className={cx(
                  'ml-1.5 rounded-full px-1.5 py-0.5 text-[10px] font-semibold',
                  selected
                    ? 'bg-brand-500 text-white'
                    : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400',
                )}
              >
                {tab.count}
              </span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}

export function TabPanel({ id, active, children }: { id: string; active: string; children: ReactNode }) {
  if (id !== active) return null;
  return (
    <div role="tabpanel" id={`tabpanel-${id}`} aria-labelledby={`tab-${id}`} tabIndex={0} className="focus-visible:outline-none">
      {children}
    </div>
  );
}
