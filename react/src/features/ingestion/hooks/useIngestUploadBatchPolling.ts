import { useEffect, useRef, type Dispatch, type SetStateAction } from 'react';
import type { UploadSummary } from '../types';

export interface UseIngestUploadBatchPollingParams {
  uploadBatchIds: { [key: number]: number };
  setUploadBatchIds: Dispatch<SetStateAction<{ [key: number]: number }>>;
  setUploadSummaries: Dispatch<SetStateAction<{ [key: number]: UploadSummary }>>;
  setErrorMessages: Dispatch<SetStateAction<{ [key: number]: string }>>;
  setSubmissionStatus: Dispatch<SetStateAction<{ [key: number]: import('../types').SubmissionStatus }>>;
  ingestUploadsUrl: string;
  onUploadSuccess?: () => void;
  updateRow: (id: number, updates: Partial<import('../types').IngestRowData>) => void;
}

export function useIngestUploadBatchPolling({
  uploadBatchIds,
  setUploadBatchIds,
  setUploadSummaries,
  setErrorMessages,
  setSubmissionStatus,
  ingestUploadsUrl,
  onUploadSuccess,
  updateRow,
}: UseIngestUploadBatchPollingParams): void {
  const pollingHandlesRef = useRef<{ [key: number]: number }>({});

  const clearPollingForRow = (rowId: number) => {
    const handle = pollingHandlesRef.current[rowId];
    if (handle !== undefined) {
      window.clearInterval(handle);
      delete pollingHandlesRef.current[rowId];
    }
  };

  useEffect(() => {
    return () => {
      Object.values(pollingHandlesRef.current).forEach((handle) =>
        window.clearInterval(handle)
      );
      pollingHandlesRef.current = {};
    };
  }, []);

  useEffect(() => {
    Object.entries(uploadBatchIds).forEach(([rowIdRaw, batchId]) => {
      const rowId = Number(rowIdRaw);
      if (!Number.isFinite(rowId) || pollingHandlesRef.current[rowId] !== undefined) {
        return;
      }

      const pollBatch = async () => {
        try {
          const response = await fetch(`${ingestUploadsUrl}${batchId}/`, {
            method: 'GET',
            credentials: 'same-origin',
          });
          if (!response.ok) {
            throw new Error(`Upload status check failed (${response.status}).`);
          }
          const payload = await response.json();
          const items = Array.isArray(payload?.items) ? payload.items : [];
          const modalities = Array.from(
            new Set(
              items.flatMap((batchItem: { modalities?: unknown[] }) =>
                Array.isArray(batchItem?.modalities) ? batchItem.modalities : []
              )
            )
          ).map((value) => String(value));
          const providers = Array.from(
            new Set(
              items.flatMap((batchItem: { providers?: unknown[] }) =>
                Array.isArray(batchItem?.providers) ? batchItem.providers : []
              )
            )
          ).map((value) => String(value));
          const rawMediaSaved = items.some((batchItem: { raw_media_saved?: boolean }) =>
            Boolean(batchItem?.raw_media_saved)
          );
          const textExtracted = items.some((batchItem: { text_extracted?: boolean }) =>
            Boolean(batchItem?.text_extracted)
          );
          setUploadSummaries((prev) => ({
            ...prev,
            [rowId]: { modalities, providers, rawMediaSaved, textExtracted },
          }));

          const counts = payload?.counts || {};
          const queued = Number(counts.queued || 0);
          const processing = Number(counts.processing || 0);
          const success = Number(counts.success || 0);
          const failed = Number(counts.error || 0);
          if (queued + processing > 0) {
            return;
          }

          clearPollingForRow(rowId);
          setUploadBatchIds((prev) => {
            const next = { ...prev };
            delete next[rowId];
            return next;
          });

          if (success > 0) {
            onUploadSuccess?.();
            updateRow(rowId, { uploadFiles: [] });
          }

          if (failed > 0) {
            const failedItems = Array.isArray(payload?.items)
              ? payload.items.filter((item: { status?: string }) => item?.status === 'error')
              : [];
            const firstError = failedItems[0]?.error_message || 'Unknown extraction error.';
            const prefix =
              success > 0
                ? `Uploaded ${success} file(s), but ${failed} failed. `
                : `Upload failed for ${failed} file(s). `;
            setErrorMessages((prev) => ({
              ...prev,
              [rowId]: `${prefix}${firstError}`,
            }));
            setSubmissionStatus((prev) => ({ ...prev, [rowId]: 'error' }));
            return;
          }

          if (success > 0) {
            setSubmissionStatus((prev) => ({ ...prev, [rowId]: 'success' }));
          } else {
            setErrorMessages((prev) => ({
              ...prev,
              [rowId]: 'Upload finished with no ingested files.',
            }));
            setSubmissionStatus((prev) => ({ ...prev, [rowId]: 'error' }));
          }
        } catch (error: unknown) {
          clearPollingForRow(rowId);
          setUploadBatchIds((prev) => {
            const next = { ...prev };
            delete next[rowId];
            return next;
          });
          const msg = error instanceof Error ? error.message : 'Failed to poll upload status.';
          setErrorMessages((prev) => ({
            ...prev,
            [rowId]: msg,
          }));
          setSubmissionStatus((prev) => ({ ...prev, [rowId]: 'error' }));
        }
      };

      pollBatch();
      pollingHandlesRef.current[rowId] = window.setInterval(pollBatch, 2000);
    });
    /* Legacy effect deps: batch ids + URL + success callback drive polling. */
  }, [uploadBatchIds, ingestUploadsUrl, onUploadSuccess]);
}
