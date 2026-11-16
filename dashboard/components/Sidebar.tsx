'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

import { IconChevronLeft, IconClose } from './icons';
import { NAV_ITEMS, isActive, normalizePath, type NavGroup } from './sections';
import { cx } from './ui';

export { NAV_ITEMS, navTitle, normalizePath } from './sections';

const GROUPS: Array<NavGroup | undefined> = [undefined, 'Content', 'Operations', 'Settings'];

export function Sidebar({
  collapsed,
  mobileOpen,
  onToggleCollapse,
  onCloseMobile,
  appName,
}: {
  collapsed: boolean;
  mobileOpen: boolean;
  onToggleCollapse: () => void;
  onCloseMobile: () => void;
  appName: string;
}) {
  const pathname = usePathname();
  const path = normalizePath(pathname);

  const nav = (
    <nav aria-label="Main navigation" className="flex-1 overflow-y-auto px-3 py-4">
      {GROUPS.map((group) => {
        const items = NAV_ITEMS.filter((item) => item.group === group);
        return (
          <div key={group ?? 'top'} className={cx(group && 'mt-5')}>
            {group && !collapsed ? (
              <p className="mb-1.5 px-3 text-[11px] font-medium text-slate-400 dark:text-slate-500">{group}</p>
            ) : null}
            {group && collapsed ? <div className="mx-3 mb-2 border-t border-slate-200 dark:border-slate-800" /> : null}
            <div className="space-y-0.5">
              {items.map((item) => {
                const active = isActive(item, path);
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={onCloseMobile}
                    aria-current={active ? 'page' : undefined}
                    title={collapsed ? item.label : undefined}
                    className={cx(
                      'flex items-center gap-3 rounded-full py-1.5 pl-1.5 pr-3 text-sm font-medium transition-colors',
                      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500',
                      active
                        ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900'
                        : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white',
                      collapsed && 'justify-center !px-1.5',
                    )}
                  >
                    <span
                      className={cx(
                        'flex h-8 w-8 shrink-0 items-center justify-center rounded-full',
                        active ? 'bg-white/15 dark:bg-slate-900/10' : 'bg-slate-100 dark:bg-slate-800',
                      )}
                    >
                      <Icon className="h-4 w-4" />
                    </span>
                    {collapsed ? <span className="sr-only">{item.label}</span> : <span className="truncate">{item.label}</span>}
                  </Link>
                );
              })}
            </div>
          </div>
        );
      })}
    </nav>
  );

  return (
    <>
      {/* Mobile overlay */}
      {mobileOpen ? (
        <div className="fixed inset-0 z-30 bg-slate-950/40 lg:hidden" aria-hidden="true" onClick={onCloseMobile} />
      ) : null}

      <aside
        aria-label="Sidebar"
        className={cx(
          'fixed inset-y-0 left-0 z-40 flex flex-col bg-white shadow-card transition-all duration-200 dark:bg-slate-900',
          'lg:inset-y-3 lg:left-3 lg:rounded-3xl',
          collapsed ? 'w-16' : 'w-64',
          mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
        )}
      >
        <div className="flex min-h-16 items-center justify-between gap-2 px-4 py-3">
          {collapsed ? (
            <span className="mx-auto flex h-8 w-8 items-center justify-center rounded-full bg-brand-500 text-sm font-bold text-white">
              G
            </span>
          ) : (
            <div className="flex min-w-0 items-center gap-2.5">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-brand-500 text-sm font-bold text-white">
                G
              </span>
              <div className="min-w-0">
                <p className="line-clamp-2 text-sm font-semibold leading-tight text-slate-900 dark:text-white">{appName}</p>
                <p className="truncate text-[11px] text-slate-500 dark:text-slate-400">Knowledge base admin</p>
              </div>
            </div>
          )}
          <button
            type="button"
            onClick={onCloseMobile}
            aria-label="Close navigation"
            className="rounded-full p-1.5 text-slate-500 hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 dark:hover:bg-slate-800 lg:hidden"
          >
            <IconClose className="h-4 w-4" />
          </button>
        </div>

        {nav}

        <div className="p-3">
          <button
            type="button"
            onClick={onToggleCollapse}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="hidden w-full items-center justify-center gap-2 rounded-full bg-slate-100 px-3 py-2 text-xs font-medium text-slate-600 hover:bg-slate-200 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700 lg:flex"
          >
            <IconChevronLeft className={cx('h-4 w-4 transition-transform', collapsed && 'rotate-180')} />
            {collapsed ? null : <span>Collapse sidebar</span>}
          </button>
        </div>
      </aside>
    </>
  );
}
