import {
  IconAdmin,
  IconBackups,
  IconCosts,
  IconIntegrations,
  IconJobs,
  IconKnowledge,
  IconOverview,
  IconUpload,
  IconVideos,
  type IconProps,
} from './icons';

/**
 * Visual identity per dashboard section: every section has its own accent
 * colour, so the sidebar, top bar, page title and panels all say where you are.
 * Class strings are written out in full so Tailwind keeps them in the build.
 */
export interface SectionTheme {
  /** Hex accent; exposed as the CSS variable --accent on <main>. */
  accent: string;
  /** Solid icon tile (page title). */
  tile: string;
  /** Icon colour in the dark sidebar. */
  navIcon: string;
}

export type NavGroup = 'Content' | 'Operations' | 'Settings';

export interface NavItem {
  href: string;
  label: string;
  icon: (props: IconProps) => JSX.Element;
  /** One-line purpose shown under the page title. */
  tagline: string;
  group?: NavGroup;
  theme: SectionTheme;
  /** Exact match (used for the overview root). */
  exact?: boolean;
}

export const NAV_ITEMS: NavItem[] = [
  {
    href: '/',
    label: 'Overview',
    icon: IconOverview,
    exact: true,
    tagline: 'Health of the whole platform at a glance',
    theme: { accent: '#4f46e5', tile: 'bg-indigo-600', navIcon: 'text-indigo-300' },
  },
  {
    href: '/knowledge',
    label: 'Knowledge Base',
    icon: IconKnowledge,
    group: 'Content',
    tagline: 'Everything the assistant can answer from',
    theme: { accent: '#0284c7', tile: 'bg-sky-600', navIcon: 'text-sky-300' },
  },
  {
    href: '/uploads',
    label: 'Uploads & Ingestion',
    icon: IconUpload,
    group: 'Content',
    tagline: 'Add documents, media and data to the knowledge base',
    theme: { accent: '#0d9488', tile: 'bg-teal-600', navIcon: 'text-teal-300' },
  },
  {
    href: '/jobs',
    label: 'Processing Jobs',
    icon: IconJobs,
    group: 'Content',
    tagline: 'Ingestion and video work in progress',
    theme: { accent: '#d97706', tile: 'bg-amber-600', navIcon: 'text-amber-300' },
  },
  {
    href: '/videos',
    label: 'Video Library',
    icon: IconVideos,
    group: 'Content',
    tagline: 'Videos, queue and transcription settings',
    theme: { accent: '#e11d48', tile: 'bg-rose-600', navIcon: 'text-rose-300' },
  },
  {
    href: '/backups',
    label: 'Backups & Restore',
    icon: IconBackups,
    group: 'Operations',
    tagline: 'Recovery points, schedules and restores',
    theme: { accent: '#059669', tile: 'bg-emerald-600', navIcon: 'text-emerald-300' },
  },
  {
    href: '/costs',
    label: 'Costs & Usage',
    icon: IconCosts,
    group: 'Operations',
    tagline: 'What ingestion, chat and retrieval cost',
    theme: { accent: '#7c3aed', tile: 'bg-violet-600', navIcon: 'text-violet-300' },
  },
  {
    href: '/integrations',
    label: 'Integrations',
    icon: IconIntegrations,
    group: 'Settings',
    tagline: 'AI providers, Google Drive and video sources',
    theme: { accent: '#ea580c', tile: 'bg-orange-600', navIcon: 'text-orange-300' },
  },
  {
    href: '/admin',
    label: 'Administration',
    icon: IconAdmin,
    group: 'Settings',
    tagline: 'Account, monitoring and system configuration',
    theme: { accent: '#475569', tile: 'bg-slate-600', navIcon: 'text-slate-300' },
  },
];

/** App-relative path (usePathname may include the /kb base path). */
export function normalizePath(pathname: string | null): string {
  const stripped = (pathname ?? '/').replace(/^\/kb(?=\/|$)/, '');
  return stripped === '' ? '/' : stripped;
}

export function isActive(item: NavItem, path: string): boolean {
  return item.exact ? path === item.href : path.startsWith(item.href);
}

export function currentSection(pathname: string | null): NavItem | undefined {
  const path = normalizePath(pathname);
  return NAV_ITEMS.find((item) => isActive(item, path));
}

export function navTitle(pathname: string | null): string {
  return currentSection(pathname)?.label ?? 'Dashboard';
}
