'use client';

import type { ReactNode } from 'react';

import { EmptyState, ErrorState, SkeletonRows } from './Feedback';
import { IconChevronLeft, IconChevronRight } from './icons';
import { Button, cx } from './ui';

export interface DataTableColumn<T> {
  key: string;
  header: ReactNode;
  /** Enables the sort button in the header (server-side sorting). */
  sortable?: boolean;
  className?: string;
  render?: (row: T, index: number) => ReactNode;
}

export interface DataTableSort {
  key: string;
  dir: 'asc' | 'desc';
}

export interface DataTableProps<T> {
  columns: Array<DataTableColumn<T>>;
  rows: T[];
  rowKey: (row: T, index: number) => string;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  emptyTitle?: string;
  emptyDescription?: string;
  emptyAction?: ReactNode;
  sort?: DataTableSort | null;
  onSortChange?: (sort: DataTableSort) => void;
  /** Server-side pagination. Omit for client-only lists. */
  page?: number;
  pageSize?: number;
  total?: number;
  onPageChange?: (page: number) => void;
  onRowClick?: (row: T, index: number) => void;
  caption?: string;
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  loading = false,
  error = null,
  onRetry,
  emptyTitle = 'Nothing here yet',
  emptyDescription,
  emptyAction,
  sort = null,
  onSortChange,
  page,
  pageSize,
  total,
  onPageChange,
  onRowClick,
  caption,
}: DataTableProps<T>) {
  const paged =
    page !== undefined && pageSize !== undefined && total !== undefined && onPageChange !== undefined;
  const currentPage = page ?? 1;
  const size = pageSize ?? 20;
  const totalCount = total ?? 0;
  const pageCount = paged ? Math.max(1, Math.ceil(totalCount / size)) : 1;

  const handleSort = (column: DataTableColumn<T>) => {
    if (!column.sortable || !onSortChange) return;
    const dir: 'asc' | 'desc' = sort && sort.key === column.key && sort.dir === 'asc' ? 'desc' : 'asc';
    onSortChange({ key: column.key, dir });
  };

  return (
    <div className="overflow-hidden rounded-md border border-slate-200 dark:border-slate-800">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200 text-sm dark:divide-slate-800">
          {caption ? <caption className="sr-only">{caption}</caption> : null}
          <thead className="bg-slate-50 dark:bg-slate-900/60">
            <tr>
              {columns.map((column) => {
                const activeSort = sort && sort.key === column.key ? sort : null;
                return (
                  <th
                    key={column.key}
                    scope="col"
                    aria-sort={
                      activeSort ? (activeSort.dir === 'asc' ? 'ascending' : 'descending') : undefined
                    }
                    className={cx(
                      'px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400',
                      column.className,
                    )}
                  >
                    {column.sortable && onSortChange ? (
                      <button
                        type="button"
                        onClick={() => handleSort(column)}
                        className="inline-flex items-center gap-1 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
                      >
                        {column.header}
                        <span
                          aria-hidden="true"
                          className={cx('text-[10px]', activeSort ? 'text-brand-600 dark:text-brand-400' : 'text-slate-400')}
                        >
                          {activeSort ? (activeSort.dir === 'asc' ? '▲' : '▼') : '↕'}
                        </span>
                      </button>
                    ) : (
                      column.header
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white dark:divide-slate-800/80 dark:bg-slate-900">
            {loading ? (
              <tr>
                <td colSpan={columns.length} className="p-3">
                  <SkeletonRows rows={Math.min(pageSize ?? 5, 6)} columns={columns.length} />
                </td>
              </tr>
            ) : error ? (
              <tr>
                <td colSpan={columns.length} className="p-4">
                  <ErrorState error={error} onRetry={onRetry} />
                </td>
              </tr>
            ) : rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="p-0">
                  <EmptyState title={emptyTitle} description={emptyDescription} action={emptyAction} />
                </td>
              </tr>
            ) : (
              rows.map((row, index) => (
                <tr
                  key={rowKey(row, index)}
                  className={cx(
                    'transition-colors',
                    onRowClick
                      ? 'cursor-pointer hover:bg-slate-50 focus-visible:bg-slate-50 dark:hover:bg-slate-800/60 dark:focus-visible:bg-slate-800/60'
                      : 'hover:bg-slate-50/60 dark:hover:bg-slate-800/40',
                  )}
                  tabIndex={onRowClick ? 0 : undefined}
                  role={onRowClick ? 'button' : undefined}
                  onClick={onRowClick ? () => onRowClick(row, index) : undefined}
                  onKeyDown={
                    onRowClick
                      ? (event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            onRowClick(row, index);
                          }
                        }
                      : undefined
                  }
                >
                  {columns.map((column) => (
                    <td key={column.key} className={cx('px-3 py-2 text-slate-700 dark:text-slate-200', column.className)}>
                      {column.render ? column.render(row, index) : (row as Record<string, unknown>)[column.key] as ReactNode}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {paged ? (
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 bg-slate-50 px-3 py-2 dark:border-slate-800 dark:bg-slate-900/60">
          <p className="text-xs text-slate-500 dark:text-slate-400">
            {totalCount > 0
              ? `Showing ${(currentPage - 1) * size + 1}–${Math.min(currentPage * size, totalCount)} of ${totalCount}`
              : 'No results'}
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => onPageChange?.(Math.max(1, currentPage - 1))}
              disabled={currentPage <= 1}
              aria-label="Previous page"
            >
              <IconChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-xs text-slate-600 dark:text-slate-300">
              Page {currentPage} of {pageCount}
            </span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => onPageChange?.(Math.min(pageCount, currentPage + 1))}
              disabled={currentPage >= pageCount}
              aria-label="Next page"
            >
              <IconChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
