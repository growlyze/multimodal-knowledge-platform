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
              <p className="mb-1.5 px-3 text-[10px] font-semibold uppercase tracking-widest text-slate-500">{group}</p>
            ) : null}
            {group && collapsed ? <div className="mx-3 mb-2 border-t border-slate-800" /> : null}
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
                    style={active ? { boxShadow: `inset 3px 0 0 ${item.theme.accent}` } : undefined}
                    className={cx(
                      'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/60',
                      active ? 'bg-white/10 text-white' : 'text-slate-300 hover:bg-white/5 hover:text-white',
                      collapsed && 'justify-center px-2',
                    )}
                  >
                    <span
                      className={cx(
                        'flex h-7 w-7 shrink-0 items-center justify-center rounded-md',
                        active ? cx(item.theme.tile, 'text-white') : cx('bg-white/5', item.theme.navIcon),
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
        <div className="fixed inset-0 z-30 bg-slate-950/50 lg:hidden" aria-hidden="true" onClick={onCloseMobile} />
      ) : null}

      <aside
        aria-label="Sidebar"
        className={cx(
          'fixed inset-y-0 left-0 z-40 flex flex-col border-r border-slate-800 bg-slate-900 transition-all duration-200 dark:bg-slate-950',
          collapsed ? 'w-16' : 'w-64',
          mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
        )}
      >
        <div className="flex min-h-14 items-center justify-between gap-2 border-b border-slate-800 px-4 py-2.5">
          {collapsed ? (
            <span className="mx-auto flex h-8 w-8 items-center justify-center rounded-md bg-brand-600 text-sm font-bold text-white">
              G
            </span>
          ) : (
            <div className="flex min-w-0 items-center gap-2.5">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-brand-600 text-sm font-bold text-white">
                G
              </span>
              <div className="min-w-0">
                <p className="line-clamp-2 text-sm font-semibold leading-tight text-white">{appName}</p>
                <p className="truncate text-[10px] uppercase tracking-wide text-slate-400">Admin console</p>
              </div>
            </div>
          )}
          <button
            type="button"
            onClick={onCloseMobile}
            aria-label="Close navigation"
            className="rounded-md p-1.5 text-slate-400 hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/60 lg:hidden"
          >
            <IconClose className="h-4 w-4" />
          </button>
        </div>

        {nav}

        <div className="border-t border-slate-800 p-3">
          <button
            type="button"
            onClick={onToggleCollapse}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="hidden w-full items-center justify-center gap-2 rounded-md px-3 py-2 text-xs font-medium text-slate-400 hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/60 lg:flex"
          >
            <IconChevronLeft className={cx('h-4 w-4 transition-transform', collapsed && 'rotate-180')} />
            {collapsed ? null : <span>Collapse sidebar</span>}
          </button>
        </div>
      </aside>
    </>
  );
}
