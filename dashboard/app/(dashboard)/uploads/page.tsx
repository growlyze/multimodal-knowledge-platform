'use client';

import Link from 'next/link';
import { useCallback, useMemo, useRef, useState, type DragEvent } from 'react';

import { EmptyState } from '@/components/Feedback';
import { JobProgress } from '@/components/JobProgress';
import { useToast } from '@/components/Toast';
import { IconFile, IconJobs, IconUpload } from '@/components/icons';
import { Badge, Button, Field, Input, Panel, Select, statusTone } from '@/components/ui';
import { ApiClientError, apiFetch, apiUpload, attempt } from '@/lib/client';
import { formatBytes } from '@/lib/format';
import { useAsyncData } from '@/lib/hooks';
import { pickArray, pickString } from '@/lib/normalize';
import { uploadMetaSchema, type SourceType } from '@/lib/schemas';
import { defaultCollectionFor, suggestSourceType } from '@/lib/uploads';
import { SectionTitle } from '@/components/SectionTitle';

/* ------------------------------------------------------- source type logic */

const SOURCE_TYPES: SourceType[] = ['document', 'code', 'transcript', 'structured', 'media', 'archive'];
const DEFAULT_COLLECTIONS = ['documents', 'code_docs'];

interface FileEntry {
  id: string;
  file: File;
  sourceType: SourceType;
  collection: string;
  status: 'queued' | 'uploading' | 'done' | 'error';
  progress: number;
  message: string;
  jobId: string | null;
}

let entrySeq = 0;

export default function UploadsPage() {
  const toast = useToast();
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [defaultCollection, setDefaultCollection] = useState('documents');
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Collection options: env-configured defaults + whatever the backend reports
  // (GET /admin/stats returns every Qdrant collection with its name).
  const { data: configData } = useAsyncData(() => attempt(apiFetch('proxy/admin/stats')), []);
  const collectionOptions = useMemo(() => {
    const options = new Set(DEFAULT_COLLECTIONS);
    const reported = configData?.ok ? pickArray(configData.value, ['collections', 'indexes']) : [];
    for (const entry of reported) {
      const name = pickString(entry, ['name', 'collection', 'index']);
      if (name) options.add(name);
    }
    return [...options];
  }, [configData]);

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const next: FileEntry[] = Array.from(files).map((file) => {
        const sourceType = suggestSourceType(file.name);
        return {
        id: `file-${entrySeq++}`,
        file,
        sourceType,
        // Code goes to the code collection, spreadsheets/archives to the
        // backend's own choice; everything else to the default collection.
        collection: sourceType === 'document' || sourceType === 'transcript' || sourceType === 'media'
          ? defaultCollection
          : defaultCollectionFor(sourceType),
        status: 'queued',
        progress: 0,
        message: '',
        jobId: null,
        } satisfies FileEntry;
      });
      setEntries((current) => [...current, ...next]);
    },
    [defaultCollection],
  );

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files.length > 0) addFiles(event.dataTransfer.files);
  };

  const updateEntry = (id: string, patch: Partial<FileEntry>) => {
    setEntries((current) => current.map((entry) => (entry.id === id ? { ...entry, ...patch } : entry)));
  };

  const uploadOne = async (entry: FileEntry) => {
    const meta = uploadMetaSchema.safeParse({ source_type: entry.sourceType, collection: entry.collection });
    if (!meta.success) {
      updateEntry(entry.id, { status: 'error', message: meta.error.issues[0]?.message ?? 'Invalid metadata' });
      return;
    }

    updateEntry(entry.id, { status: 'uploading', progress: 0, message: '' });
    const form = new FormData();
    form.append('file', entry.file);
    // POST /ingest/{type}: multipart file (+ optional collection). The
    // structured and archive endpoints choose their collection themselves.
    if (meta.data.collection && meta.data.source_type !== 'structured' && meta.data.source_type !== 'archive') {
      form.append('collection', meta.data.collection);
    }

    try {
      const response = await apiUpload<Record<string, unknown>>(
        `proxy/ingest/${meta.data.source_type}`,
        form,
        (percent) => updateEntry(entry.id, { progress: percent }),
      );
      const jobId =
        (typeof response?.job_id === 'string' && response.job_id) ||
        (typeof response?.id === 'string' && response.id) ||
        null;
      updateEntry(entry.id, {
        status: 'done',
        progress: 100,
        message: pickString(response, ['message', 'status'], 'Queued for ingestion'),
        jobId,
      });
      toast.push('success', `Ingesting ${entry.file.name}`, `Sent to /ingest/${meta.data.source_type}`);
    } catch (err) {
      const message = err instanceof ApiClientError ? err.message : 'Upload failed';
      updateEntry(entry.id, { status: 'error', message });
      toast.push('error', `Upload failed: ${entry.file.name}`, message);
    }
  };

  const uploadAll = async () => {
    setUploading(true);
    for (const entry of entries.filter((item) => item.status === 'queued')) {
      // Sequential uploads keep per-file progress readable and avoid
      // saturating the ingestion pipeline.
      // eslint-disable-next-line no-await-in-loop
      await uploadOne(entry);
    }
    setUploading(false);
  };

  const queuedCount = entries.filter((entry) => entry.status === 'queued').length;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <SectionTitle>Uploads &amp; Ingestion</SectionTitle>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Drop files, pick a source type and collection, then ingest (POST /ingest/&#123;type&#125;).
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/jobs">
            <Button variant="secondary">
              <IconJobs className="h-4 w-4" />
              Processing jobs
            </Button>
          </Link>
          <Button onClick={uploadAll} loading={uploading} disabled={queuedCount === 0}>
            <IconUpload className="h-4 w-4" />
            Upload {queuedCount > 0 ? `${queuedCount} file${queuedCount === 1 ? '' : 's'}` : ''}
          </Button>
        </div>
      </div>

      {/* ---------------------------------------------- defaults */}
      <Panel title="Ingestion settings" description="Applied to every file added below">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:max-w-xl">
          <Field
            label="Default collection"
            htmlFor="default-collection"
            hint="Configured in the deployment environment (defaults: documents, code_docs)"
          >
            <Input
              id="default-collection"
              list="collection-options"
              value={defaultCollection}
              onChange={(event) => setDefaultCollection(event.target.value)}
            />
            <datalist id="collection-options">
              {collectionOptions.map((option) => (
                <option key={option} value={option} />
              ))}
            </datalist>
          </Field>
          <Field label="Supported source types" hint="Auto-suggested from the file extension; adjustable per file">
            <div className="flex flex-wrap gap-1.5 pt-1">
              {SOURCE_TYPES.map((type) => (
                <Badge key={type} tone="neutral">
                  {type}
                </Badge>
              ))}
            </div>
          </Field>
        </div>
      </Panel>

      {/* ---------------------------------------------- drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Add files to ingest"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors ${
          dragging
            ? 'border-brand-500 bg-brand-50 dark:border-brand-400 dark:bg-brand-950/40'
            : 'border-slate-300 bg-white hover:border-brand-400 dark:border-slate-700 dark:bg-slate-900 dark:hover:border-brand-600'
        }`}
      >
        <IconUpload className="h-8 w-8 text-slate-400" />
        <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
          Drag &amp; drop files here, or <span className="text-brand-600 dark:text-brand-400">browse</span>
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Documents, code, transcripts, spreadsheets and media · multiple files supported
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          className="sr-only"
          onChange={(event) => {
            if (event.target.files) addFiles(event.target.files);
            event.target.value = '';
          }}
        />
      </div>

      {/* ---------------------------------------------- file queue */}
      {entries.length === 0 ? (
        <Panel>
          <EmptyState
            icon={<IconFile className="h-8 w-8" />}
            title="No files queued"
            description="Files added above appear here with per-file source type, collection and ingestion progress."
          />
        </Panel>
      ) : (
        <Panel
          title={`Queued files (${entries.length})`}
          actions={
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setEntries((current) => current.filter((entry) => entry.status !== 'done'))}
            >
              Clear finished
            </Button>
          }
        >
          <ul className="space-y-3">
            {entries.map((entry) => (
              <li key={entry.id} className="rounded-md border border-slate-200 p-3 dark:border-slate-800">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-slate-900 dark:text-white">{entry.file.name}</p>
                    <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                      {formatBytes(entry.file.size)} ·{' '}
                      <code className="rounded bg-slate-100 px-1 font-mono dark:bg-slate-800">
                        POST /ingest/{entry.sourceType}
                      </code>
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Select
                      aria-label={`Source type for ${entry.file.name}`}
                      value={entry.sourceType}
                      disabled={entry.status === 'uploading' || entry.status === 'done'}
                      onChange={(event) =>
                        updateEntry(entry.id, { sourceType: event.target.value as SourceType })
                      }
                      className="h-8 w-32 py-1 text-xs"
                    >
                      {SOURCE_TYPES.map((type) => (
                        <option key={type} value={type}>
                          {type}
                        </option>
                      ))}
                    </Select>
                    <Input
                      aria-label={`Collection for ${entry.file.name}`}
                      list="collection-options"
                      value={entry.collection}
                      disabled={entry.status === 'uploading' || entry.status === 'done'}
                      onChange={(event) => updateEntry(entry.id, { collection: event.target.value })}
                      className="h-8 w-36 py-1 text-xs"
                    />
                    {entry.status === 'queued' ? (
                      <Button size="sm" variant="secondary" onClick={() => uploadOne(entry)}>
                        Upload
                      </Button>
                    ) : null}
                    {entry.status === 'queued' || entry.status === 'error' ? (
                      <Button
                        size="sm"
                        variant="ghost"
                        aria-label={`Remove ${entry.file.name}`}
                        onClick={() => setEntries((current) => current.filter((item) => item.id !== entry.id))}
                      >
                        Remove
                      </Button>
                    ) : null}
                  </div>
                </div>

                {entry.status === 'uploading' ? (
                  <JobProgress label={entry.file.name} status="uploading" progress={entry.progress} />
                ) : entry.status === 'done' ? (
                  <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Badge tone={statusTone('completed')}>sent</Badge>
                      <span className="text-xs text-slate-500 dark:text-slate-400">{entry.message}</span>
                    </div>
                    <Link
                      href="/jobs"
                      className="inline-flex items-center gap-1 text-xs font-medium text-brand-600 hover:underline dark:text-brand-400"
                    >
                      <IconJobs className="h-3.5 w-3.5" />
                      Track on the jobs page
                      {entry.jobId ? ` (${entry.jobId})` : ''}
                    </Link>
                  </div>
                ) : entry.status === 'error' ? (
                  <p role="alert" className="mt-2 text-xs text-rose-600 dark:text-rose-400">
                    {entry.message}
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}
