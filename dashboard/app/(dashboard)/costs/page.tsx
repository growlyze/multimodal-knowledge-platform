'use client';

import { useCallback, useMemo, useState, type ReactNode } from 'react';

import { BarChart, type BarChartDatum } from '@/components/BarChart';
import { ErrorState, SkeletonRows, UnavailableNotice } from '@/components/Feedback';
import { MetricCard } from '@/components/MetricCard';
import { IconDatabase, IconSearch, IconVideos } from '@/components/icons';
import { Badge, Button, Panel, Select } from '@/components/ui';
import { apiFetch, attempt } from '@/lib/client';
import { formatDateTime, formatNumber, formatUSD, relativeTime } from '@/lib/format';
import { useAsyncData } from '@/lib/hooks';
import { isRecord, pickArray, pickNumber, pickString } from '@/lib/normalize';
import { SectionTitle } from '@/components/SectionTitle';

const PERIODS = [7, 30, 90];

export default function CostsPage() {
  const [days, setDays] = useState(30);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <SectionTitle>Costs &amp; Usage</SectionTitle>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Ingestion, chat and retrieval costs for the selected period. Every section degrades gracefully when its
            data source is not deployed.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select
            aria-label="Reporting period"
            value={String(days)}
            onChange={(event) => setDays(Number(event.target.value))}
            className="h-9 w-36"
          >
            {PERIODS.map((period) => (
              <option key={period} value={period}>
                Last {period} days
              </option>
            ))}
          </Select>
        </div>
      </div>

      <CostSection
        key={`ingestion-${days}`}
        title="Ingestion"
        description={`Video and document ingestion costs (GET /video/costs · last ${days} days)`}
        endpoint="proxy/video/costs"
        days={days}
        icon={<IconVideos className="h-5 w-5" />}
      />
      <ChatCostNotice />
      <CostSection
        key={`retrieval-${days}`}
        title="Retrieval"
        description={`Embedding and retrieval usage (GET /costs/retrieval · last ${days} days)`}
        endpoint="proxy/costs/retrieval"
        days={days}
        icon={<IconDatabase className="h-5 w-5" />}
      />
    </div>
  );
}

/**
 * Chat (LLM) spend has no backend endpoint yet — GET /costs/chat does not exist
 * in haystack-api, so this section intentionally performs no request rather
 * than calling a route that would always 404.
 */
function ChatCostNotice() {
  return (
    <Panel title="Chat" description="Chat model usage">
      <UnavailableNotice
        feature="Chat costs (GET /costs/chat)"
        detail="The backend does not expose GET /costs/chat on this build, so per-chat model spend cannot be reported yet. Ingestion and retrieval costs below are unaffected."
      />
    </Panel>
  );
}

function CostSection({
  title,
  description,
  endpoint,
  days,
  icon,
  knownMissing = false,
}: {
  title: string;
  description: string;
  endpoint: string;
  days: number;
  icon: ReactNode;
  knownMissing?: boolean;
}) {
  const load = useCallback(() => attempt(apiFetch(endpoint)), [endpoint]);
  const { data, loading, reload } = useAsyncData(load, [endpoint, days]);

  const unavailable = data?.ok === false && data.error.notAvailable;
  const payload = data?.ok ? data.value : null;

  const daily: BarChartDatum[] = useMemo(() => {
    const rows = pickArray(payload, ['daily', 'by_day', 'series', 'days', 'history']);
    return rows
      .map((row): BarChartDatum => ({
        label: shortLabel(pickString(row, ['date', 'day', 'label', 'bucket'])),
        value: pickNumber(row, ['cost', 'cost_usd', 'amount', 'value', 'total'], 0),
      }))
      .filter((datum) => datum.label !== '')
      .slice(-days);
  }, [payload, days]);

  const total = pickNumber(payload, ['total', 'total_cost', 'cost', 'amount', 'sum'], daily.reduce((sum, d) => sum + d.value, 0));
  const units = pickNumber(payload, ['units', 'tokens', 'requests', 'count', 'items']);
  const lastUpdated = pickString(payload, ['updated_at', 'generated_at', 'last_run']);

  return (
    <Panel
      title={title}
      description={description}
      actions={
        <>
          {knownMissing && unavailable ? <Badge tone="warning">known missing</Badge> : null}
          <Button variant="secondary" size="sm" onClick={reload}>
            Refresh
          </Button>
        </>
      }
    >
      {unavailable ? (
        <UnavailableNotice
          feature={`${title} costs (${endpoint.replace('proxy/', '')})`}
          detail={
            knownMissing
              ? 'This deployment does not expose GET /costs/chat — it is a known gap on current builds. The other cost sections are unaffected.'
              : undefined
          }
        />
      ) : loading ? (
        <SkeletonRows rows={4} columns={3} />
      ) : data?.ok === false ? (
        <ErrorState error={data.error.message} onRetry={reload} title={`Could not load ${title.toLowerCase()} costs`} />
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <MetricCard
              label={`Total ${title.toLowerCase()} cost`}
              value={formatUSD(total)}
              tone="brand"
              icon={icon}
            />
            <MetricCard label="Units" value={units > 0 ? formatNumber(units) : '—'} sub="requests / tokens / items" />
            <MetricCard
              label="Daily average"
              value={formatUSD(days > 0 ? total / days : 0)}
              sub={lastUpdated ? `updated ${relativeTime(lastUpdated)}` : `last ${days} days`}
            />
          </div>

          {daily.length > 0 ? (
            <BarChart data={daily} formatValue={formatUSD} height={140} />
          ) : (
            <p className="rounded-md border border-dashed border-slate-300 px-3 py-6 text-center text-xs text-slate-500 dark:border-slate-700 dark:text-slate-400">
              No daily series reported for this period.
            </p>
          )}

          <details className="rounded-md border border-slate-200 px-3 py-2 text-xs dark:border-slate-800">
            <summary className="cursor-pointer font-medium text-slate-600 dark:text-slate-300">Raw response</summary>
            <pre className="mt-2 max-h-64 overflow-auto text-slate-600 dark:text-slate-300">
              {JSON.stringify(payload, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </Panel>
  );
}

function shortLabel(value: string): string {
  if (!value) return '';
  // 2026-10-30T00:00:00Z / 2026-10-30 -> 10-30
  const match = value.match(/\d{4}-(\d{2}-\d{2})/);
  if (match) return match[1];
  if (value.length <= 8) return value;
  return formatDateTime(value).slice(5, 11);
}
