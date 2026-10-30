import { LoginSuspense } from './LoginForm';

export const metadata = {
  title: 'Sign in',
};

/**
 * Login screen at /kb/login. The APP_NAME display value is passed in from the
 * server so no NEXT_PUBLIC_* variables are needed in the browser bundle.
 */
export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-100 px-4 py-10 dark:bg-slate-950">
      <LoginSuspense appName={process.env.APP_NAME || 'Growlyze Multimodal AI Chat'} />
    </main>
  );
}
