'use client';

/**
 * Small client-side data hooks: async loading with reload + error state, and
 * polling for job tables. Keep dependencies explicit so pages stay simple.
 */
import { useCallback, useEffect, useRef, useState, type DependencyList } from 'react';

import { toApiError, type ApiClientError } from './client';

export interface AsyncState<T> {
  data: T | null;
  error: ApiClientError | null;
  loading: boolean;
  reload: () => void;
}

export function useAsyncData<T>(load: () => Promise<T>, deps: DependencyList = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiClientError | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const loader = useRef(load);
  loader.current = load;

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    loader
      .current()
      .then((value) => {
        if (!alive) return;
        setData(value);
        setError(null);
      })
      .catch((err: unknown) => {
        if (!alive) return;
        setError(toApiError(err));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  const reload = useCallback(() => setTick((value) => value + 1), []);
  return { data, error, loading, reload };
}

/** Runs `callback` every `intervalMs` while `enabled` (tab polling). */
export function usePolling(callback: () => void, intervalMs: number, enabled = true): void {
  const saved = useRef(callback);
  saved.current = callback;

  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => saved.current(), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, enabled]);
}

/** Local storage backed state (sidebar collapse, view preferences). */
export function usePersistentState<T extends string | boolean>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(initial);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(key);
      if (stored !== null) {
        setValue((stored === 'true' ? true : stored === 'false' ? false : stored) as T);
      }
    } catch {
      /* storage unavailable - keep default */
    }
  }, [key]);

  const update = useCallback(
    (next: T) => {
      setValue(next);
      try {
        window.localStorage.setItem(key, String(next));
      } catch {
        /* ignore */
      }
    },
    [key],
  );

  return [value, update];
}
