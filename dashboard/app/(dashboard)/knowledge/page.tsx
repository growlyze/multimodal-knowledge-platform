'use client';

import { useCallback, useMemo, useState, type FormEvent } from 'react';

import type { DataTableColumn } from '@/components/DataTable';
import { DataTable } from '@/components/DataTable';
import { ConfirmDialog } from '@/components/ConfirmDialog';
import { Drawer } from '@/components/Drawer';
import { EmptyState, ErrorState, LoadingBlock, SkeletonRows } from '@/components/Feedback';
import { Tabs, TabPanel } from '@/components/Tabs';
import { useToast } from '@/components/Toast';
import { IconFile, IconSearch, IconTable, IconTrash } from '@/components/icons';
import {
  Badge,
  Button,
  Field,
  Input,
  KeyValue,
  Panel,
  Select,
} from '@/components/ui';
import { ApiClientError, apiFetch, apiSend } from '@/lib/client';
import { formatBytes, formatNumber, relativeTime } from '@/lib/format';
import { useAsyncData } from '@/lib/hooks';
import { isRecord, pickArray, pickNumber, pickRecord, pickString, rowKey, type AnyRecord } from '@/lib/normalize';
import { SectionTitle } from '@/components/SectionTitle';

const TABS = [
  { id: 'documents', label: 'Documents' },
  { id: 'tables', label: 'Spreadsheet data' },
  { id: 'search', label: 'Search tester' },
];

export default function KnowledgePage() {
  const [tab, setTab] = useState('documents');
  return (
    <div className="space-y-4">
      <div>
        <SectionTitle>Knowledge Base</SectionTitle>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Browse ingested documents, inspect spreadsheet tables and run retrieval tests.
        </p>
      </div>
      <Tabs tabs={TABS} active={tab} onChange={setTab} ariaLabel="Knowledge base sections" />
      <TabPanel id="documents" active={tab}>
        <DocumentsTab />
      </TabPanel>
      <TabPanel id="tables" active={tab}>
        <TablesTab />
      </TabPanel>
      <TabPanel id="search" active={tab}>
        <SearchTab />
      </TabPanel>
    </div>
  );
}

/* ================================================================ Documents */

function DocumentsTab() {
  const toast = useToast();
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [sort, setSort] = useState<{ key: string; dir: 'asc' | 'desc' } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const load = useCallback(
    () =>
      apiFetch(
        `proxy/admin/documents?${new URLSearchParams({
          page: String(page),
          per_page: String(pageSize),
          ...(sort ? { sort: sort.key, order: sort.dir } : {}),
        })}`,
      ),
    [page, pageSize, sort],
  );
  const { data, error, loading, reload } = useAsyncData(load, [page, pageSize, sort]);

  const items = useMemo(() => pickArray(data, ['documents', 'sources', 'items']), [data]);
  const total = pickNumber(data, ['total', 'total_count', 'count', 'total_items'], items.length);

  const columns: Array<DataTableColumn<AnyRecord>> = [
    {
      key: 'source',
      header: 'Source',
      sortable: true,
      render: (row) => (
        <div className="flex items-center gap-2">
          <IconFile className="h-4 w-4 shrink-0 text-slate-400" />
          <span className="truncate font-medium text-slate-900 dark:text-white">
            {pickString(row, ['source', 'name', 'id'], '—')}
          </span>
        </div>
      ),
    },
    { key: 'collection', header: 'Collection', render: (row) => pickString(row, ['collection', 'index', 'target'], '—') },
    {
      key: 'chunks',
      header: 'Chunks',
      className: 'tabular text-right',
      sortable: true,
      render: (row) => formatNumber(pickNumber(row, ['chunks', 'chunk_count', 'num_chunks'])),
    },
    {
      key: 'size',
      header: 'Size',
      className: 'tabular',
      render: (row) => {
        const bytes = pickNumber(row, ['size', 'size_bytes', 'bytes'], Number.NaN);
        return Number.isFinite(bytes) ? formatBytes(bytes) : '—';
      },
    },
    {
      key: 'updated',
      header: 'Updated',
      className: 'tabular',
      sortable: true,
      render: (row) => {
        const at = pickString(row, ['updated_at', 'created_at', 'modified', 'ingested_at']);
        return at ? relativeTime(at) : '—';
      },
    },
    {
      key: 'actions',
      header: <span className="sr-only">Actions</span>,
      render: (row) => {
        const source = pickString(row, ['source', 'name', 'id']);
        return (
          <div className="flex justify-end gap-1">
            <Button
              size="sm"
              variant="secondary"
              onClick={(event) => {
                event.stopPropagation();
                setSelected(source);
              }}
            >
              Details
            </Button>
            <Button
              size="sm"
              variant="ghost"
              aria-label={`Delete ${source}`}
              onClick={(event) => {
                event.stopPropagation();
                setDeleteError(null);
                setDeleteTarget(source);
              }}
            >
              <IconTrash className="h-4 w-4 text-rose-600 dark:text-rose-400" />
            </Button>
          </div>
        );
      },
    },
  ];

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await apiSend('DELETE', `proxy/admin/documents/${encodeURIComponent(deleteTarget)}`);
      toast.push('success', 'Document deleted', deleteTarget);
      setDeleteTarget(null);
      reload();
    } catch (err) {
      setDeleteError(err instanceof ApiClientError ? err.message : 'Delete failed');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Panel
      title="Documents"
      description="Server-paginated view of ingested sources (GET /admin/documents)"
      actions={
        <>
          <Select
            aria-label="Rows per page"
            value={String(pageSize)}
            onChange={(event) => {
              setPageSize(Number(event.target.value));
              setPage(1);
            }}
            className="h-8 w-24 py-1 text-xs"
          >
            {[10, 20, 50, 100].map((size) => (
              <option key={size} value={size}>
                {size} / page
              </option>
            ))}
          </Select>
          <Button variant="secondary" size="sm" onClick={reload}>
            Refresh
          </Button>
        </>
      }
      bodyClassName="px-0 py-0 sm:px-0"
    >
      <DataTable
        columns={columns}
        rows={items as AnyRecord[]}
        rowKey={(row, index) => rowKey(row, index, ['source'])}
        loading={loading}
        error={error?.message ?? null}
        onRetry={reload}
        emptyTitle="No documents ingested"
        emptyDescription="Upload files from Uploads & Ingestion to populate the knowledge base."
        sort={sort}
        onSortChange={setSort}
        page={page}
        pageSize={pageSize}
        total={total}
        onPageChange={setPage}
        onRowClick={(row) => setSelected(pickString(row, ['source', 'name', 'id']))}
        caption="Ingested documents"
      />

      <DocumentDrawer source={selected} onClose={() => setSelected(null)} />

      <ConfirmDialog
        open={deleteTarget !== null}
        title="Delete document"
        description={
          <>
            Delete <strong className="break-all">{deleteTarget}</strong> and all of its chunks from the knowledge base?
            This cannot be undone.
          </>
        }
        confirmLabel="Delete"
        loading={deleting}
        error={deleteError}
        onConfirm={confirmDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </Panel>
  );
}

function DocumentDrawer({ source, onClose }: { source: string | null; onClose: () => void }) {
  const load = useCallback(
    () => apiFetch(`proxy/admin/documents/${encodeURIComponent(source ?? '')}`),
    [source],
  );
  const { data, error, loading, reload } = useAsyncData(load, [source]);

  const chunks = useMemo(() => pickArray(data, ['chunks', 'items', 'documents', 'segments']), [data]);

  return (
    <Drawer
      open={source !== null}
      title={source ?? ''}
      subtitle={pickString(data, ['collection', 'index'], '') || undefined}
      onClose={onClose}
    >
      {loading ? (
        <SkeletonRows rows={6} columns={2} />
      ) : error ? (
        <ErrorState error={error.message} onRetry={reload} title="Could not load document" />
      ) : (
        <div className="space-y-4">
          <dl className="rounded-md border border-slate-200 px-3 py-1 dark:border-slate-800">
            <KeyValue label="Chunks" value={formatNumber(pickNumber(data, ['chunks', 'chunk_count', 'total_chunks'], chunks.length))} />
            <KeyValue label="Collection" value={pickString(data, ['collection', 'index'], '—')} />
            <KeyValue label="Size" value={formatBytes(pickNumber(data, ['size', 'size_bytes'], 0))} />
            <KeyValue
              label="Updated"
              value={pickString(data, ['updated_at', 'created_at']) ? relativeTime(pickString(data, ['updated_at', 'created_at'])) : '—'}
            />
          </dl>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Chunks ({chunks.length})
            </h3>
            {chunks.length === 0 ? (
              <EmptyState title="No chunks returned" description="The backend returned no chunk detail for this source." />
            ) : (
              <ol className="mt-2 space-y-2">
                {chunks.slice(0, 100).map((chunk, index) => (
                  <li key={index} className="rounded-md border border-slate-200 px-3 py-2 dark:border-slate-800">
                    <div className="flex items-center justify-between gap-2">
                      <Badge tone="neutral">#{pickNumber(chunk, ['chunk_id', 'id', 'index'], index)}</Badge>
                      {pickString(chunk, ['collection', 'source_type']) ? (
                        <Badge tone="info">{pickString(chunk, ['collection', 'source_type'])}</Badge>
                      ) : null}
                    </div>
                    <p className="mt-1.5 whitespace-pre-wrap break-words text-xs text-slate-700 dark:text-slate-300">
                      {pickString(chunk, ['text', 'content', 'preview', 'snippet', 'body'], '(no text)')}
                    </p>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>
      )}
    </Drawer>
  );
}

/* =================================================================== Tables */

function TablesTab() {
  const toast = useToast();
  const [preview, setPreview] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const [dropping, setDropping] = useState(false);
  const [dropError, setDropError] = useState<string | null>(null);

  const { data, error, loading, reload } = useAsyncData(() => apiFetch('proxy/admin/tables'), []);
  const items = useMemo(() => pickArray(data, ['tables', 'items', 'datasets']), [data]);

  const columns: Array<DataTableColumn<AnyRecord>> = [
    {
      key: 'name',
      header: 'Table',
      render: (row) => (
        <div className="flex items-center gap-2">
          <IconTable className="h-4 w-4 shrink-0 text-slate-400" />
          <span className="font-medium text-slate-900 dark:text-white">{pickString(row, ['name', 'table', 'id'], '—')}</span>
        </div>
      ),
    },
    {
      key: 'rows',
      header: 'Rows',
      className: 'tabular text-right',
      render: (row) => formatNumber(pickNumber(row, ['rows', 'row_count', 'num_rows'])),
    },
    {
      key: 'columns',
      header: 'Columns',
      className: 'tabular text-right',
      render: (row) => formatNumber(pickNumber(row, ['columns', 'column_count', 'num_columns'])),
    },
    {
      key: 'updated',
      header: 'Updated',
      render: (row) => {
        const at = pickString(row, ['updated_at', 'created_at', 'ingested_at']);
        return at ? relativeTime(at) : '—';
      },
    },
    {
      key: 'actions',
      header: <span className="sr-only">Actions</span>,
      render: (row) => {
        const name = pickString(row, ['name', 'table', 'id']);
        return (
          <div className="flex justify-end gap-1">
            <Button size="sm" variant="secondary" onClick={(event) => { event.stopPropagation(); setPreview(name); }}>
              Preview
            </Button>
            <Button
              size="sm"
              variant="ghost"
              aria-label={`Drop table ${name}`}
              onClick={(event) => {
                event.stopPropagation();
                setDropError(null);
                setDropTarget(name);
              }}
            >
              <IconTrash className="h-4 w-4 text-rose-600 dark:text-rose-400" />
            </Button>
          </div>
        );
      },
    },
  ];

  const confirmDrop = async () => {
    if (!dropTarget) return;
    setDropping(true);
    setDropError(null);
    try {
      await apiSend('DELETE', `proxy/admin/tables/${encodeURIComponent(dropTarget)}`);
      toast.push('success', 'Table dropped', dropTarget);
      setDropTarget(null);
      reload();
    } catch (err) {
      setDropError(err instanceof ApiClientError ? err.message : 'Drop failed');
    } finally {
      setDropping(false);
    }
  };

  return (
    <Panel
      title="Spreadsheet data"
      description="Structured datasets loaded from spreadsheets and CSVs (GET /admin/tables)"
      actions={
        <Button variant="secondary" size="sm" onClick={reload}>
          Refresh
        </Button>
      }
      bodyClassName="px-0 py-0 sm:px-0"
    >
      <DataTable
        columns={columns}
        rows={items as AnyRecord[]}
        rowKey={(row, index) => rowKey(row, index, ['name'])}
        loading={loading}
        error={error?.message ?? null}
        onRetry={reload}
        emptyTitle="No spreadsheet tables"
        emptyDescription="Upload .csv / .xlsx files with the “structured” source type to create tables."
        caption="Spreadsheet tables"
      />

      <TablePreviewDrawer name={preview} onClose={() => setPreview(null)} />

      <ConfirmDialog
        open={dropTarget !== null}
        title="Drop table"
        description={
          <>
            Drop table <strong className="break-all">{dropTarget}</strong>? All rows and the derived knowledge will be
            removed. This cannot be undone.
          </>
        }
        confirmLabel="Drop table"
        confirmWord="DROP"
        loading={dropping}
        error={dropError}
        onConfirm={confirmDrop}
        onCancel={() => setDropTarget(null)}
      />
    </Panel>
  );
}

function TablePreviewDrawer({ name, onClose }: { name: string | null; onClose: () => void }) {
  const load = useCallback(() => apiFetch(`proxy/admin/tables/${encodeURIComponent(name ?? '')}`), [name]);
  const { data, error, loading, reload } = useAsyncData(load, [name]);

  const columns = useMemo(() => {
    const cols = pickArray(data, ['columns', 'fields', 'headers']);
    return cols.map((col) => pickString(col, ['name', 'field', 'key', 'column'], String(col)));
  }, [data]);
  const rows = useMemo(() => pickArray(data, ['rows', 'records', 'items', 'data']), [data]);

  return (
    <Drawer open={name !== null} title={name ?? ''} subtitle="Table preview" onClose={onClose} width="max-w-3xl">
      {loading ? (
        <SkeletonRows rows={8} columns={4} />
      ) : error ? (
        <ErrorState error={error.message} onRetry={reload} title="Could not load table" />
      ) : rows.length === 0 ? (
        <EmptyState title="No rows" description="This table has no previewable rows." />
      ) : (
        <div className="overflow-x-auto rounded-md border border-slate-200 dark:border-slate-800">
          <table className="min-w-full divide-y divide-slate-200 text-xs dark:divide-slate-800">
            <thead className="bg-slate-50 dark:bg-slate-900/60">
              <tr>
                {(columns.length > 0 ? columns : Object.keys(rows[0] as Record<string, unknown>)).map((col) => (
                  <th key={col} scope="col" className="whitespace-nowrap px-2.5 py-2 text-left font-semibold text-slate-500 dark:text-slate-400">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
              {rows.slice(0, 100).map((row, index) => (
                <tr key={index}>
                  {(columns.length > 0 ? columns : Object.keys((isRecord(row) ? row : {}) as Record<string, unknown>)).map((col) => (
                    <td key={col} className="max-w-[280px] truncate whitespace-nowrap px-2.5 py-1.5 text-slate-700 dark:text-slate-200">
                      {formatCell(isRecord(row) ? row[col] : undefined)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Drawer>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') return formatNumber(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
}

/* ============================================================ Search tester */

interface SearchHit {
  score: number;
  content: string;
  source: string;
  meta: AnyRecord;
}

function SearchTab() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState('10');
  const [collections, setCollections] = useState('');
  const [sourceType, setSourceType] = useState('');
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<SearchHit[] | null>(null);
  const [logId, setLogId] = useState<string | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  const runSearch = async (event: FormEvent) => {
    event.preventDefault();
    if (!query.trim()) {
      setSearchError('Enter a query first.');
      return;
    }
    setRunning(true);
    setSearchError(null);
    try {
      const params = new URLSearchParams({
        q: query.trim(),
        top_k: topK || '10',
        ...(collections.trim() ? { collections: collections.trim() } : {}),
        ...(sourceType ? { source_type: sourceType } : {}),
      });
      const data = await apiFetch(`proxy/search?${params}`);
      const hits = pickArray(data, ['results', 'hits', 'documents', 'answers']).map((hit): SearchHit => {
        const meta = pickRecord(hit, ['meta', 'metadata', 'document']) ?? {};
        return {
          score: pickNumber(hit, ['score', 'similarity', 'relevance'], Number.NaN),
          content: pickString(hit, ['content', 'text', 'answer', 'snippet', 'body'], '(empty)'),
          source: pickString(hit, ['source', 'file', 'document', 'name', 'title'], 'unknown source'),
          meta,
        };
      });
      setResults(hits);
      setLogId(pickString(data, ['log_id', 'eval_log_id', 'query_id', 'trace_id'], '') || null);
    } catch (err) {
      setResults(null);
      setSearchError(err instanceof ApiClientError ? err.message : 'Search failed');
    } finally {
      setRunning(false);
    }
  };

  const sources = useMemo(() => {
    const counts = new Map<string, { count: number; best: number }>();
    for (const hit of results ?? []) {
      const entry = counts.get(hit.source) ?? { count: 0, best: 0 };
      entry.count += 1;
      entry.best = Math.max(entry.best, Number.isFinite(hit.score) ? hit.score : 0);
      counts.set(hit.source, entry);
    }
    return [...counts.entries()].sort((a, b) => b[1].best - a[1].best);
  }, [results]);

  return (
    <div className="space-y-4">
      <Panel title="Search tester" description="Run a retrieval query (GET /search) and inspect ranked chunks + sources">
        <form onSubmit={runSearch} className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-6">
          <Field label="Query" htmlFor="search-q" className="lg:col-span-3" required>
            <Input
              id="search-q"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="e.g. What is the refund policy?"
            />
          </Field>
          <Field label="Top K" htmlFor="search-topk" hint="1–50">
            <Input id="search-topk" type="number" min={1} max={50} value={topK} onChange={(event) => setTopK(event.target.value)} />
          </Field>
          <Field label="Collections" htmlFor="search-collections" hint="comma separated">
            <Input
              id="search-collections"
              value={collections}
              onChange={(event) => setCollections(event.target.value)}
              placeholder="documents, code_docs"
            />
          </Field>
          <Field label="Source type" htmlFor="search-type">
            <Select id="search-type" value={sourceType} onChange={(event) => setSourceType(event.target.value)}>
              <option value="">Any</option>
              <option value="document">document</option>
              <option value="code">code</option>
              <option value="transcript">transcript</option>
              <option value="structured">structured</option>
              <option value="media">media</option>
            </Select>
          </Field>
          <div className="flex items-end lg:col-span-1">
            <Button type="submit" loading={running} className="w-full">
              <IconSearch className="h-4 w-4" />
              Search
            </Button>
          </div>
        </form>

        {searchError ? (
          <p role="alert" className="mt-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-950/60 dark:text-rose-300">
            {searchError}
          </p>
        ) : null}
      </Panel>

      {running ? (
        <Panel title="Results">
          <LoadingBlock label="Searching…" />
        </Panel>
      ) : results ? (
        <>
          <Panel
            title={`Results (${results.length})`}
            description={logId ? `eval log_id: ${logId}` : 'Ranked by retrieval score'}
            actions={logId ? <Badge tone="info">log_id {logId}</Badge> : undefined}
          >
            {results.length === 0 ? (
              <EmptyState title="No matches" description="Try a different query or widen the collections filter." />
            ) : (
              <ol className="space-y-2">
                {results.map((hit, index) => (
                  <li key={index} className="rounded-md border border-slate-200 px-3 py-2.5 dark:border-slate-800">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="truncate text-sm font-medium text-slate-900 dark:text-white">{hit.source}</span>
                      <div className="flex items-center gap-2">
                        {Number.isFinite(hit.score) ? (
                          <Badge tone={hit.score >= 0.7 ? 'ok' : hit.score >= 0.4 ? 'info' : 'neutral'}>
                            score {hit.score.toFixed(3)}
                          </Badge>
                        ) : null}
                        <span className="text-xs text-slate-400">#{index + 1}</span>
                      </div>
                    </div>
                    <p className="mt-1.5 whitespace-pre-wrap break-words text-xs text-slate-700 dark:text-slate-300">
                      {hit.content}
                    </p>
                  </li>
                ))}
              </ol>
            )}
          </Panel>

          <Panel title="Sources" description="Distinct sources contributing to the answer">
            {sources.length === 0 ? (
              <EmptyState title="No sources" />
            ) : (
              <ul className="divide-y divide-slate-100 dark:divide-slate-800">
                {sources.map(([source, info]) => (
                  <li key={source} className="flex items-center justify-between gap-3 py-2">
                    <span className="truncate text-sm text-slate-800 dark:text-slate-200">{source}</span>
                    <div className="flex shrink-0 items-center gap-2">
                      <Badge tone="neutral">{info.count} chunk{info.count === 1 ? '' : 's'}</Badge>
                      <Badge tone="info">best {info.best.toFixed(3)}</Badge>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </>
      ) : null}
    </div>
  );
}
