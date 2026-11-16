'use client';

import { useEffect, useState } from 'react';

import { IconMoon, IconSun } from './icons';

/**
 * Dark/light toggle. The choice is persisted in the `gmac_theme` cookie and
 * read server-side in the root layout (plus a tiny inline script to avoid a
 * flash of the wrong theme before hydration).
 */
export function ThemeToggle() {
  const [dark, setDark] = useState<boolean | null>(null);

  useEffect(() => {
    setDark(document.documentElement.classList.contains('dark'));
  }, []);

  const toggle = () => {
    const next = !(document.documentElement.classList.contains('dark'));
    document.documentElement.classList.toggle('dark', next);
    document.cookie = `gmac_theme=${next ? 'dark' : 'light'}; Path=/; Max-Age=31536000; SameSite=Lax`;
    setDark(next);
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-white text-slate-600 shadow-card hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 dark:bg-slate-900 dark:text-slate-300 dark:hover:text-white"
    >
      {dark ? <IconSun className="h-5 w-5" /> : <IconMoon className="h-5 w-5" />}
    </button>
  );
}
