/**
 * Zod schemas for every dashboard form. Server route handlers and client
 * forms share these so validation rules can never drift apart.
 */
import { z } from 'zod';

export const loginSchema = z.object({
  username: z.string().trim().min(1, 'Username is required'),
  password: z.string().min(1, 'Password is required'),
});
export type LoginInput = z.infer<typeof loginSchema>;

export const sourceTypeSchema = z.enum(['document', 'code', 'transcript', 'structured', 'media', 'archive']);
export type SourceType = z.infer<typeof sourceTypeSchema>;

export const uploadMetaSchema = z.object({
  source_type: sourceTypeSchema,
  // Empty = the backend's default collection for that type (documents / code_docs).
  collection: z.string().trim().max(120).optional().default(''),
});
export type UploadMeta = z.infer<typeof uploadMetaSchema>;

export const searchSchema = z.object({
  q: z.string().trim().min(1, 'Enter a query'),
  top_k: z.coerce.number().int().min(1).max(50).default(10),
  collections: z.string().trim().max(500).optional().default(''),
  source_type: sourceTypeSchema.optional(),
});
export type SearchInput = z.infer<typeof searchSchema>;

export const documentDeleteSchema = z.object({
  source: z.string().trim().min(1, 'Source is required'),
});
export type DocumentDeleteInput = z.infer<typeof documentDeleteSchema>;

export const tableDropSchema = z.object({
  name: z.string().trim().min(1, 'Table name is required'),
  confirm: z.literal('DROP', {
    errorMap: () => ({ message: 'Type DROP to confirm' }),
  }),
});
export type TableDropInput = z.infer<typeof tableDropSchema>;

export const videoUrlSchema = z.object({
  url: z.string().trim().url('Enter a valid URL').max(2048),
});
export type VideoUrlInput = z.infer<typeof videoUrlSchema>;

export const videoBatchSchema = z.object({
  urls: z
    .array(z.string().trim().url().max(2048))
    .min(1, 'Add at least one URL')
    .max(100, 'At most 100 URLs per batch'),
});
export type VideoBatchInput = z.infer<typeof videoBatchSchema>;

export const channelDiscoverSchema = z.object({
  url: z.string().trim().url('Enter a valid channel URL').max(2048),
});
export type ChannelDiscoverInput = z.infer<typeof channelDiscoverSchema>;

/** Parses a textarea of URLs (newline / comma separated) into a validated batch. */
export function parseUrlList(text: string): VideoBatchInput {
  const urls = text
    .split(/[\n\r,;\s]+/)
    .map((line) => line.trim())
    .filter(Boolean);
  return videoBatchSchema.parse({ urls });
}

export const cookiesTextSchema = z.object({
  content: z.string().trim().min(1, 'Paste Netscape cookie data first').max(2_000_000),
});
export type CookiesTextInput = z.infer<typeof cookiesTextSchema>;

export const restoreSchema = z.object({
  kind: z.enum(['restore-docs', 'restore-files', 'restore-chats', 'restore-videos', 'restore-full']),
  confirm: z.string().optional(),
});
export type RestoreInput = z.infer<typeof restoreSchema>;
