/**
 * Source-type suggestion for uploaded files: pick the ingestion endpoint
 * (POST /ingest/{type}) from the file extension.
 *
 * Kept out of the uploads page itself because Next.js only allows specific
 * named exports from a route file.
 */
import { sourceTypeSchema, type SourceType } from './schemas';

const EXTENSION_MAP: Record<string, SourceType> = {
  pdf: 'document', doc: 'document', docx: 'document', txt: 'document', md: 'document', rtf: 'document',
  odt: 'document', html: 'document', htm: 'document', epub: 'document',
  py: 'code', js: 'code', ts: 'code', jsx: 'code', tsx: 'code', java: 'code', go: 'code', rs: 'code',
  rb: 'code', php: 'code', c: 'code', cpp: 'code', h: 'code', cs: 'code', sql: 'code', sh: 'code',
  yaml: 'code', yml: 'code', toml: 'code', ini: 'code',
  srt: 'transcript', vtt: 'transcript',
  csv: 'structured', tsv: 'structured', xlsx: 'structured', xls: 'structured', json: 'structured', parquet: 'structured',
  mp4: 'media', mov: 'media', mkv: 'media', webm: 'media', avi: 'media', mp3: 'media', wav: 'media',
  m4a: 'media', flac: 'media', ogg: 'media', png: 'media', jpg: 'media', jpeg: 'media', webp: 'media',
  zip: 'archive', gz: 'archive', tgz: 'archive', tar: 'archive',
};

const VIDEO_EXTENSIONS = new Set(['mp4', 'mov', 'mkv', 'webm', 'avi', 'm4v']);

function extensionOf(fileName: string): string {
  return fileName.includes('.') ? fileName.split('.').pop()?.toLowerCase() ?? '' : '';
}

export function isVideoFileName(fileName: string): boolean {
  return VIDEO_EXTENSIONS.has(extensionOf(fileName));
}

/** Default collection for a type: '' lets the backend choose (documents / code_docs). */
export function defaultCollectionFor(type: SourceType): string {
  return type === 'code' ? 'code_docs' : type === 'structured' || type === 'archive' ? '' : 'documents';
}

export function suggestSourceType(fileName: string): SourceType {
  const ext = extensionOf(fileName);
  const suggested = EXTENSION_MAP[ext];
  const parsed = sourceTypeSchema.safeParse(suggested);
  return parsed.success ? parsed.data : 'document';
}
