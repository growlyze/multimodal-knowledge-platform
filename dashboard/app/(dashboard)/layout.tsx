import type { ReactNode } from 'react';

import { Shell } from '@/components/Shell';
import { cookies } from 'next/headers';
import { verifySession, COOKIE_SESSION } from '@/lib/auth';

/**
 * Authenticated app chrome. Access control itself happens in middleware.ts;
 * here we only read the verified session for the top bar identity.
 */
export default async function DashboardLayout({ children }: { children: ReactNode }) {
  const token = cookies().get(COOKIE_SESSION)?.value ?? null;
  const session = await verifySession(token);

  return (
    <Shell username={session?.sub ?? 'admin'} appName={process.env.APP_NAME || 'Growlyze Multimodal AI Chat'}>
      {children}
    </Shell>
  );
}
