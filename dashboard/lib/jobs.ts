/**
 * Shared normalizers for ingestion / video job rows (payload shapes vary
 * slightly per deployment; everything is read defensively).
 */
import { pickNumber, pickString, isRecord, type AnyRecord } from './normalize';

export type JobKind = 'ingest' | 'video';

export interface JobRow {
  id: string;
  kind: JobKind;
  label: string;
  status: string;
  /** 0-100, or NaN when the endpoint reports no progress. */
  progress: number;
  detail: string;
  error: string;
  updatedAt: string;
  raw: AnyRecord;
}

const RUNNING = ['running', 'processing', 'in_progress', 'started', 'downloading', 'transcribing', 'embedding', 'scanning'];
const QUEUED = ['queued', 'pending', 'waiting', 'scheduled', 'created', 'new'];
const FAILED = ['failed', 'error', 'cancelled', 'canceled'];

export function jobIsRunning(status: string): boolean {
  return RUNNING.includes(status.toLowerCase());
}

export function jobIsQueued(status: string): boolean {
  return QUEUED.includes(status.toLowerCase());
}

export function jobIsFailed(status: string): boolean {
  return FAILED.includes(status.toLowerCase());
}

export function jobIsActive(status: string): boolean {
  return jobIsRunning(status) || jobIsQueued(status);
}

export function normalizeJobRow(row: unknown, kind: JobKind, index: number): JobRow {
  const raw: AnyRecord = isRecord(row) ? row : {};
  const id = pickString(row, ['id', 'job_id', 'video_id', 'uuid', 'source', 'name', 'url'], `${kind}-${index}`);
  const label = pickString(row, ['title', 'name', 'source', 'filename', 'file_name', 'url', 'video_url', 'id'], id);
  const status = pickString(row, ['status', 'state', 'phase', 'stage'], 'unknown');
  return {
    id,
    kind,
    label,
    status,
    progress: pickNumber(row, ['progress', 'percent', 'progress_pct', 'progress_percent', 'completion'], Number.NaN),
    detail: pickString(row, ['detail', 'message', 'phase', 'stage', 'collection', 'source_type', 'status_text']),
    error: pickString(row, ['error', 'error_message', 'failure_reason', 'reason', 'warning']),
    updatedAt: pickString(row, ['updated_at', 'finished_at', 'completed_at', 'started_at', 'created_at', 'timestamp']),
    raw,
  };
}
