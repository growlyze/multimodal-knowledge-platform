'use client';

import type { ReactNode } from 'react';
import { usePathname } from 'next/navigation';

import { currentSection } from './sections';
import { cx } from './ui';

/** Page title with the section's solid colour tile and its one-line purpose. */
export function SectionTitle({ children }: { children: ReactNode }) {
  const section = currentSection(usePathname());
  const Icon = section?.icon;
  return (
    <div className="mb-2 flex items-center gap-3">
      {Icon ? (
        <span
          className={cx(
            'flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-white shadow-sm',
            section.theme.tile,
          )}
        >
          <Icon className="h-6 w-6" />
        </span>
      ) : null}
      <div className="min-w-0">
        <h1 className="truncate text-xl font-semibold text-slate-900 dark:text-white">{children}</h1>
        {section ? (
          <p className="text-xs font-medium uppercase tracking-wide text-[color:var(--accent)]">{section.tagline}</p>
        ) : null}
      </div>
    </div>
  );
}
