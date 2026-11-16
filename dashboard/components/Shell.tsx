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
  // One brand accent everywhere (orange on neutral greys).
  const accentStyle = { '--accent': '#f26430' } as CSSProperties;

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100" style={accentStyle}>
      <Sidebar
        collapsed={collapsed}
        mobileOpen={mobileOpen}
        onToggleCollapse={() => setCollapsed(!collapsed)}
        onCloseMobile={() => setMobileOpen(false)}
        appName={appName}
      />
      <div className={`transition-[padding] duration-200 ${collapsed ? 'lg:pl-[5.5rem]' : 'lg:pl-[17.5rem]'}`}>
        <TopBar title={section?.label ?? 'Dashboard'} username={username} onOpenMobile={() => setMobileOpen(true)} />
        <main className="mx-auto w-full max-w-7xl px-4 pb-10 pt-2 sm:px-6 lg:pl-2 lg:pr-6">{children}</main>
      </div>
    </div>
  );
}
