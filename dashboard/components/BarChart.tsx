import { cx } from './ui';

export interface BarChartDatum {
  label: string;
  value: number;
  /** Optional tooltip/title text. */
  hint?: string;
}

/**
 * Dependency-free CSS bar chart (no SVG/canvas) for daily cost series.
 * Bars are labeled and carry a text alternative for screen readers.
 */
export function BarChart({
  data,
  height = 160,
  formatValue,
  emptyLabel = 'No data for this period',
  className,
}: {
  data: BarChartDatum[];
  height?: number;
  formatValue?: (value: number) => string;
  emptyLabel?: string;
  className?: string;
}) {
  if (data.length === 0) {
    return (
      <div
        className={cx(
          'flex items-center justify-center rounded-md border border-dashed border-slate-300 text-xs text-slate-500 dark:border-slate-700 dark:text-slate-400',
          className,
        )}
        style={{ height }}
      >
        {emptyLabel}
      </div>
    );
  }

  const max = Math.max(...data.map((datum) => datum.value), 0.0001);
  const fmt = formatValue ?? ((value: number) => value.toFixed(2));

  return (
    <div className={cx('rounded-md border border-slate-200 p-3 dark:border-slate-800', className)}>
      <div className="flex items-end gap-1" style={{ height }} role="img" aria-label={buildSummary(data, fmt)}>
        {data.map((datum) => {
          const ratio = Math.max(datum.value > 0 ? 0.02 : 0, datum.value / max);
          return (
            <div key={datum.label} className="group flex h-full min-w-0 flex-1 flex-col justify-end gap-1">
              <div
                className="w-full rounded-t bg-brand-500/80 transition-colors group-hover:bg-brand-600 dark:bg-brand-600/80 dark:group-hover:bg-brand-500"
                style={{ height: `${Math.round(ratio * 100)}%` }}
                title={`${datum.label}: ${fmt(datum.value)}${datum.hint ? ` · ${datum.hint}` : ''}`}
              />
            </div>
          );
        })}
      </div>
      <div className="mt-1.5 flex gap-1">
        {data.map((datum) => (
          <div key={datum.label} className="min-w-0 flex-1 truncate text-center text-[9px] text-slate-400 dark:text-slate-500">
            {datum.label}
          </div>
        ))}
      </div>
    </div>
  );
}

function buildSummary(data: BarChartDatum[], fmt: (value: number) => string): string {
  const total = data.reduce((sum, datum) => sum + datum.value, 0);
  return `Bar chart of ${data.length} values, total ${fmt(total)}, peak ${fmt(Math.max(...data.map((d) => d.value)))}`;
}
