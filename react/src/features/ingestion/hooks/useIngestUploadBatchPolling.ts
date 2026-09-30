import { useCallback, useEffect, useRef, type Dispatch, type SetStateAction } from 'react';
import type { IngestRowData, SubmissionStatus, UploadBatch, UploadSummary } from '../types';
import { rejectionMessage } from '../utils/uploadBatch';

export interface UseIngestUploadBatchPollingParams {
  uploadBatches: Record<number, UploadBatch>;
  setUploadBatches: Dispatch<SetStateAction<Record<number, UploadBatch>>>;
  setUploadSummaries: Dispatch<SetStateAction<Record<number, UploadSummary>>>;
  setErrorMessages: Dispatch<SetStateAction<Record<number, string>>>;
  setSubmissionStatus: Dispatch<SetStateAction<Record<number, SubmissionStatus>>>;
  ingestUploadsUrl: string;
  onUploadSuccess?: () => void;
  updateRow: (id: number, updates: Partial<IngestRowData>) => void;
}

interface BatchItem {
  id: number;
  original_filename: string;
  status: string;
  error_message?: string;
  modalities?: string[];
  providers?: string[];
  raw_media_saved?: boolean;
  text_extracted?: boolean;
}

export function useIngestUploadBatchPolling(params: UseIngestUploadBatchPollingParams) {
  const current = useRef(params);
  current.current = params;
  const jobs = useRef(new Map<number, { batchId: number; cancel: () => void; retry: () => void }>());

  useEffect(() => () => {
    jobs.current.forEach(job => job.cancel());
    jobs.current.clear();
  }, []);

  useEffect(() => {
    for (const [rowId, job] of jobs.current) {
      if (params.uploadBatches[rowId]?.id !== job.batchId) {
        job.cancel();
        jobs.current.delete(rowId);
      }
    }
    for (const [rawRowId, batch] of Object.entries(params.uploadBatches)) {
      const rowId = Number(rawRowId);
      if (jobs.current.has(rowId)) continue;
      let active = true;
      let inFlight = false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      let controller: AbortController | undefined;

      const poll = async () => {
        if (!active || inFlight) return;
        inFlight = true;
        controller = new AbortController();
        try {
          const response = await fetch(`${params.ingestUploadsUrl}${batch.id}/`, {
            method: 'GET', credentials: 'same-origin', signal: controller.signal,
          });
          if (!response.ok) throw new Error(`Upload status check failed (${response.status}).`);
          const payload = await response.json();
          if (!active) return;
          if (!payload.counts || !Array.isArray(payload.items)) throw new Error('Invalid upload status response.');
          const items: BatchItem[] = payload.items;
          const state = current.current;
          state.setUploadSummaries(prev => ({ ...prev, [rowId]: {
            modalities: [...new Set(items.flatMap(item => item.modalities || []))],
            providers: [...new Set(items.flatMap(item => item.providers || []))],
            rawMediaSaved: items.some(item => item.raw_media_saved),
            textExtracted: items.some(item => item.text_extracted),
          } }));
          state.setErrorMessages(prev => ({ ...prev, [rowId]: rejectionMessage(batch.rejected) }));
          if (Number(payload.counts.queued || 0) + Number(payload.counts.processing || 0) > 0) {
            timer = setTimeout(poll, 2000);
            return;
          }

          const failedItems = items.filter(item => item.status === 'error');
          const rejected = [...batch.rejected, ...failedItems.map(item => ({
            filename: item.original_filename || 'File',
            error: item.error_message || 'Unknown extraction error.',
            file: batch.accepted.find(accepted => accepted.id === item.id)?.file,
          }))];
          const success = Number(payload.counts.success || 0);
          state.updateRow(rowId, { uploadFiles: rejected.flatMap(item => item.file ? [item.file] : []) });
          state.setErrorMessages(prev => ({ ...prev, [rowId]: rejected.length
            ? rejectionMessage(rejected) : success ? '' : 'Upload finished with no ingested files.' }));
          state.setSubmissionStatus(prev => ({ ...prev, [rowId]: success && !rejected.length ? 'success' : 'error' }));
          state.setUploadBatches(prev => {
            if (prev[rowId]?.id !== batch.id) return prev;
            const next = { ...prev }; delete next[rowId]; return next;
          });
          active = false;
          if (success) state.onUploadSuccess?.();
        } catch (error) {
          if (!active) return;
          const message = error instanceof Error ? error.message : 'Failed to poll upload status.';
          // Keep the accepted batch identity: retrying its status must never upload it again.
          current.current.setErrorMessages(prev => ({ ...prev, [rowId]:
            `${rejectionMessage(batch.rejected)} ${message} Retry the status check to continue.`.trim() }));
          current.current.setSubmissionStatus(prev => ({ ...prev, [rowId]: 'status-error' }));
        } finally {
          inFlight = false;
        }
      };
      jobs.current.set(rowId, {
        batchId: batch.id,
        cancel: () => { active = false; clearTimeout(timer); controller?.abort(); },
        retry: () => {
          clearTimeout(timer);
          current.current.setSubmissionStatus(prev => ({ ...prev, [rowId]: 'initiated' }));
          void poll();
        },
      });
      void poll();
    }
  }, [params.uploadBatches, params.ingestUploadsUrl]);

  return useCallback((rowId: number) => jobs.current.get(rowId)?.retry(), []);
}
