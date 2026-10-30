'use client';

import { useState, type CSSProperties, type ReactNode } from 'react';
import { usePathname } from 'next/navigation';

import { Sidebar } from './Sidebar';
import { currentSection } from './sections';
import { TopBar } from './TopBar';
import { usePersistentState } from '@/lib/hooks';

/**
 * Enterprise app shell: collapsible sidebar + compact top bar. Below `lg` the
 * sidebar becomes an overlay drawer; above `lg` it collapses to icon width.
 */
export function Shell({
  children,
  username,
  appName,
}: {
  children: ReactNode;
  username: string;
  appName: string;
}) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = usePersistentState<boolean>('gmac_sidebar_collapsed', false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const section = currentSection(pathname);
  // Section accent for the top bar, page title and panels (see sections.ts).
  const accentStyle = { '--accent': section?.theme.accent ?? '#4f46e5' } as CSSProperties;

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100" style={accentStyle}>
      <Sidebar
        collapsed={collapsed}
        mobileOpen={mobileOpen}
        onToggleCollapse={() => setCollapsed(!collapsed)}
        onCloseMobile={() => setMobileOpen(false)}
        appName={appName}
      />
      <div className={`transition-[padding] duration-200 ${collapsed ? 'lg:pl-16' : 'lg:pl-64'}`}>
        <TopBar title={section?.label ?? 'Dashboard'} username={username} onOpenMobile={() => setMobileOpen(true)} />
        <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8">{children}</main>
      </div>
    </div>
  );
}
