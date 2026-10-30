import type { Metadata } from 'next';
import { cookies } from 'next/headers';
import type { ReactNode } from 'react';

import { ToastProvider } from '@/components/Toast';
import './globals.css';

const APP_NAME = process.env.APP_NAME || 'Growlyze Multimodal AI Chat';

export const metadata: Metadata = {
  title: {
    default: `${APP_NAME} · Admin`,
    template: '%s · Admin',
  },
  description: 'Administration console for the Growlyze Multimodal AI knowledge platform.',
  robots: { index: false, follow: false },
};

/** Theme is stored in the `gmac_theme` cookie (set by <ThemeToggle />). */
function themeClass(): string {
  return cookies().get('gmac_theme')?.value === 'dark' ? 'dark' : '';
}

const themeScript = `(function(){try{var m=document.cookie.match(/(?:^|; )gmac_theme=([^;]*)/);var t=m?decodeURIComponent(m[1]):null;if(!t){t=window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}if(t==='dark'){document.documentElement.classList.add('dark');}}catch(e){}})();`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={themeClass()} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <ToastProvider>{children}</ToastProvider>
      </body>
    </html>
  );
}
