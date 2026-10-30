'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

import { IconLogout, IconMenu, IconUser } from './icons';
import { ThemeToggle } from './ThemeToggle';
import { Badge } from './ui';
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

  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-between gap-3 border-b border-slate-200 border-t-[3px] border-t-[color:var(--accent)] bg-white/95 px-4 backdrop-blur dark:border-slate-800 dark:border-t-[color:var(--accent)] dark:bg-slate-950/95 sm:px-6">
      <div className="flex min-w-0 items-center gap-2">
        <button
          type="button"
          onClick={onOpenMobile}
          aria-label="Open navigation"
          className="rounded-md p-2 text-slate-500 hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 dark:hover:bg-slate-800 lg:hidden"
        >
          <IconMenu className="h-5 w-5" />
        </button>
        <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-2 text-sm">
          <span className="hidden text-slate-400 sm:inline">Admin console</span>
          <span className="hidden text-slate-300 dark:text-slate-600 sm:inline" aria-hidden="true">/</span>
          <span className="h-2.5 w-2.5 shrink-0 rounded-full bg-[color:var(--accent)]" aria-hidden="true" />
          <span className="truncate font-semibold text-slate-900 dark:text-white">{title}</span>
        </nav>
      </div>

      <div className="flex items-center gap-1.5">
        <ThemeToggle />
        <span className="hidden items-center gap-1.5 rounded-md px-2 py-1 text-xs text-slate-500 dark:text-slate-400 sm:inline-flex">
          <IconUser className="h-4 w-4" />
          {username}
        </span>
        <Badge tone="info" className="hidden md:inline-flex">
          admin
        </Badge>
        <button
          type="button"
          onClick={logout}
          disabled={loggingOut}
          aria-label="Sign out"
          className="inline-flex h-9 w-9 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 disabled:opacity-50 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
        >
          <IconLogout className="h-5 w-5" />
        </button>
      </div>
    </header>
  );
}
