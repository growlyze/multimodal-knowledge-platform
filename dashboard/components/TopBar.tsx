'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

import { IconLogout, IconMenu } from './icons';
import { ThemeToggle } from './ThemeToggle';
import { apiSend } from '@/lib/client';

export function TopBar({
  title,
  username,
  onOpenMobile,
}: {
  title: string;
  username: string;
  onOpenMobile: () => void;
}) {
  const router = useRouter();
  const [loggingOut, setLoggingOut] = useState(false);

  const logout = async () => {
    setLoggingOut(true);
    try {
      await apiSend('POST', 'auth/logout');
    } catch {
      /* session may already be gone */
    }
    router.replace('/login');
    router.refresh();
  };

  const initial = (username.trim()[0] ?? 'A').toUpperCase();
  const circle =
    'inline-flex h-10 w-10 items-center justify-center rounded-full bg-white text-slate-600 shadow-card hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 disabled:opacity-50 dark:bg-slate-900 dark:text-slate-300 dark:hover:text-white';

  return (
    <header className="sticky top-0 z-20 flex h-[4.5rem] items-center justify-between gap-3 bg-slate-100/85 px-4 backdrop-blur dark:bg-slate-950/85 sm:px-6 lg:pl-2 lg:pr-6">
      <div className="flex min-w-0 items-center gap-2">
        <button type="button" onClick={onOpenMobile} aria-label="Open navigation" className={`${circle} lg:hidden`}>
          <IconMenu className="h-5 w-5" />
        </button>
        <nav
          aria-label="Breadcrumb"
          className="flex min-w-0 items-center gap-1 rounded-full bg-white p-1 text-sm shadow-card dark:bg-slate-900"
        >
          <span className="hidden px-3 text-slate-500 dark:text-slate-400 sm:inline">Admin console</span>
          <span className="truncate rounded-full bg-slate-900 px-4 py-1.5 font-medium text-white dark:bg-white dark:text-slate-900">
            {title}
          </span>
        </nav>
      </div>

      <div className="flex items-center gap-2">
        <ThemeToggle />
        <button type="button" onClick={logout} disabled={loggingOut} aria-label="Sign out" title="Sign out" className={circle}>
          <IconLogout className="h-5 w-5" />
        </button>
        <span className="hidden items-center gap-2.5 rounded-full bg-white py-1 pl-1 pr-4 shadow-card dark:bg-slate-900 sm:inline-flex">
          <span
            className="flex h-8 w-8 items-center justify-center rounded-full bg-brand-500 text-sm font-semibold text-white"
            aria-hidden="true"
          >
            {initial}
          </span>
          <span className="flex min-w-0 flex-col leading-tight">
            <span className="max-w-[12rem] truncate text-sm font-medium text-slate-900 dark:text-white">{username}</span>
            <span className="text-[11px] text-slate-500 dark:text-slate-400">Administrator</span>
          </span>
        </span>
      </div>
    </header>
  );
}
