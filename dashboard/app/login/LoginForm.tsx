'use client';

import { Suspense, useState, type FormEvent } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

import { LoadingBlock } from '@/components/Feedback';
import { IconAdmin, IconError, IconInfo, IconWarning } from '@/components/icons';
import { Button, Field, Input, Panel } from '@/components/ui';
import { ApiClientError, apiFetch, csrfToken } from '@/lib/client';

/** Router paths are base-path-less: /kb/knowledge -> /knowledge. */
function stripBasePath(path: string | null): string {
  if (!path) return '/';
  const stripped = path.startsWith('/kb') ? path.slice(3) : path;
  return stripped.startsWith('/') ? stripped : `/${stripped}`;
}

export function LoginForm({ appName }: { appName: string }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notConfigured, setNotConfigured] = useState(false);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await apiFetch('auth/login', {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-csrf-token': csrfToken() },
        body: JSON.stringify({ username, password }),
      });
      router.replace(stripBasePath(searchParams.get('next')));
      router.refresh();
    } catch (err) {
      const failure = err instanceof ApiClientError ? err : null;
      if (failure?.code === 'not_configured' || failure?.status === 503) {
        setNotConfigured(true);
        setError('Authentication is not configured on this deployment.');
      } else {
        setError(failure?.message ?? 'Sign-in failed. Check your credentials and try again.');
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Panel className="w-full max-w-md">
      <div className="mb-5 flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-500 text-white">
          <IconAdmin className="h-5 w-5" />
        </span>
        <div>
          <h1 className="text-base font-semibold text-slate-900 dark:text-white">{appName}</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400">Administration console</p>
        </div>
      </div>

      {notConfigured ? (
        <div
          role="alert"
          className="mb-4 flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2.5 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-200"
        >
          <IconWarning className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p className="font-semibold">Authentication is not configured</p>
            <p className="mt-1">
              Set <code className="rounded bg-amber-100 px-1 font-mono dark:bg-amber-900">SESSION_SECRET</code>,{' '}
              <code className="rounded bg-amber-100 px-1 font-mono dark:bg-amber-900">ADMIN_USERNAME</code> and{' '}
              <code className="rounded bg-amber-100 px-1 font-mono dark:bg-amber-900">ADMIN_PASSWORD_HASH</code> in the
              dashboard environment. Access stays refused until all three are present.
            </p>
          </div>
        </div>
      ) : null}

      <form onSubmit={onSubmit} className="space-y-4" noValidate>
        <Field label="Username" htmlFor="username" required>
          <Input
            id="username"
            name="username"
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            required
            autoFocus
          />
        </Field>
        <Field label="Password" htmlFor="password" required>
          <Input
            id="password"
            name="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
        </Field>

        {error ? (
          <p
            role="alert"
            className="flex items-start gap-2 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950/60 dark:text-rose-300"
          >
            <IconError className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        ) : null}

        <Button type="submit" loading={submitting} className="w-full" disabled={notConfigured}>
          Sign in
        </Button>
      </form>

      <p className="mt-5 flex items-start gap-2 text-[11px] text-slate-500 dark:text-slate-400">
        <IconInfo className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        Single admin account. Session cookies are HttpOnly and scoped to /kb; every request is CSRF-protected. Backend
        credentials never reach the browser.
      </p>
    </Panel>
  );
}

export function LoginFallback() {
  return <LoadingBlock label="Loading sign-in…" />;
}

export function LoginSuspense({ appName }: { appName: string }) {
  return (
    <Suspense fallback={<LoginFallback />}>
      <LoginForm appName={appName} />
    </Suspense>
  );
}
