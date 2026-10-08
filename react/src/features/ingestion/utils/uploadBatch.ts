import type { UploadBatch, UploadRejection } from '../types';

interface UploadItemResponse {
  id?: number;
  source_index?: number;
  filename?: string;
  error?: string;
}

export function rejectionMessage(rejected: UploadRejection[]): string {
  return rejected.map(item => `${item.filename}: ${item.error}${item.file ? '' : ' Select this file again to retry.'}`).join(' ');
}

export function readUploadBatch(
  payload: { batch_id?: number; items?: UploadItemResponse[]; rejected?: UploadItemResponse[] },
  files: File[],
): UploadBatch {
  const id = Number(payload.batch_id);
  if (!Number.isInteger(id) || id <= 0) throw new Error('Upload batch was queued but no batch_id was returned.');
  const sourceFile = (item: UploadItemResponse): File | undefined => {
    if (Number.isInteger(item.source_index) && item.source_index! >= 0 && item.source_index! < files.length) {
      return files[item.source_index!];
    }
    // Older servers provide only names. Never guess between same-name files:
    // that could re-upload an accepted file when retrying a rejection.
    const matching = files.filter(file => file.name === item.filename);
    return matching.length === 1 ? matching[0] : undefined;
  };
  return {
    id,
    accepted: (payload.items || []).map(item => ({ id: Number(item.id), file: sourceFile(item) })),
    rejected: (payload.rejected || []).map(item => ({
      filename: item.filename || 'File', error: item.error || 'Upload rejected.', file: sourceFile(item),
    })),
  };
}
