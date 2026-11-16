'use client';

import type { ReactNode } from 'react';
import { usePathname } from 'next/navigation';

import { currentSection } from './sections';

/** Large page title with the section's one-line purpose underneath. */
export function SectionTitle({ children }: { children: ReactNode }) {
  const section = currentSection(usePathname());
  const Icon = section?.icon;
  return (
    <div className="mb-3 flex items-center gap-4">
      {Icon ? (
        <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-white text-slate-900 shadow-card dark:bg-slate-900 dark:text-white">
          <Icon className="h-5 w-5" />
        </span>
      ) : null}
      <div className="min-w-0">
        <h1 className="truncate text-2xl font-semibold tracking-tight text-slate-900 dark:text-white sm:text-3xl">{children}</h1>
        {section ? <p className="mt-0.5 text-sm text-slate-500 dark:text-slate-400">{section.tagline}</p> : null}
      </div>
    </div>
  );
}
