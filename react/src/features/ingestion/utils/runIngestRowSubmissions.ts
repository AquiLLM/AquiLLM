import type { Dispatch, SetStateAction } from 'react';
import { getCsrfCookie } from '../../../main';
import {
  DocType,
  type IngestRowData,
  type IngestRowsContainerProps,
  type SubmissionStatus,
} from '../types';

type Urls = Pick<
  IngestRowsContainerProps,
  | 'ingestUploadsUrl'
  | 'ingestArxivUrl'
  | 'ingestPdfUrl'
  | 'ingestVttUrl'
  | 'ingestWebpageUrl'
  | 'ingestHandwrittenUrl'
>;

export interface RunIngestRowSubmissionsActions {
  setErrorMessages: Dispatch<SetStateAction<{ [key: number]: string }>>;
  setSubmissionStatus: Dispatch<SetStateAction<{ [key: number]: SubmissionStatus }>>;
  setUploadBatchIds: Dispatch<SetStateAction<{ [key: number]: number }>>;
  setUploadSummaries: Dispatch<SetStateAction<{ [key: number]: import('../types').UploadSummary }>>;
  updateRow: (id: number, updates: Partial<IngestRowData>) => void;
  onUploadSuccess?: () => void;
}

export async function runIngestRowSubmissions(
  rows: IngestRowData[],
  collectionId: string,
  urls: Urls,
  actions: RunIngestRowSubmissionsActions
): Promise<void> {
  const {
    setErrorMessages,
    setSubmissionStatus,
    setUploadBatchIds,
    setUploadSummaries,
    updateRow,
    onUploadSuccess,
  } = actions;
  const {
    ingestUploadsUrl,
    ingestArxivUrl,
    ingestPdfUrl,
    ingestVttUrl,
    ingestWebpageUrl,
    ingestHandwrittenUrl,
  } = urls;

  const csrfToken = getCsrfCookie();
  setErrorMessages({});

  for (const row of rows) {
    setSubmissionStatus((prev) => ({ ...prev, [row.id]: 'submitting' }));
    let url: string;
    let body: FormData | FormData[] | string;
    let headers: HeadersInit = {
      'X-CSRFToken': csrfToken,
      'X-Requested-With': 'XMLHttpRequest',
    };

    try {
      switch (row.docType) {
        case DocType.UPLOADS:
          if (!row.uploadFiles?.length) {
            throw new Error('At least one file is required.');
          }
          setUploadSummaries((prev) => {
            const next = { ...prev };
            delete next[row.id];
            return next;
          });
          url = ingestUploadsUrl;
          const uploadsBody = new FormData();
          uploadsBody.append('collection', collectionId);
          row.uploadFiles.forEach((file) => uploadsBody.append('files', file));
          body = uploadsBody;
          break;
        case DocType.PDF:
          if (!row.pdfFiles?.length) {
            throw new Error('At least one PDF file is required.');
          }
          if (row.pdfFiles.length === 1 && !row.pdfTitle.trim()) {
            throw new Error('PDF title is required when uploading a single file.');
          }
          url = ingestPdfUrl;
          const pdfBodies: FormData[] = row.pdfFiles.map((file) => {
            const fd = new FormData();
            fd.append('pdf_file', file);
            fd.append(
              'title',
              row.pdfFiles.length === 1 ? row.pdfTitle.trim() : file.name.replace(/\.pdf$/i, '')
            );
            fd.append('collection', collectionId);
            return fd;
          });
          body = pdfBodies;
          break;
        case DocType.ARXIV:
          if (!row.arxivId) {
            throw new Error('arXiv ID is required.');
          }
          url = ingestArxivUrl;
          body = new FormData();
          body.append('arxiv_id', row.arxivId);
          body.append('collection', collectionId);
          break;
        case DocType.VTT:
          if (!row.vttFile || !row.vttTitle) {
            throw new Error('VTT file and title are required.');
          }
          url = ingestVttUrl;
          body = new FormData();
          body.append('vtt_file', row.vttFile);
          body.append('title', row.vttTitle);
          body.append('collection', collectionId);
          break;
        case DocType.HANDWRITTEN:
          if (!row.handwrittenFile || !row.handwrittenTitle) {
            throw new Error('Image file and title are required.');
          }
          url = ingestHandwrittenUrl;
          body = new FormData();
          body.append('image_file', row.handwrittenFile);
          body.append('title', row.handwrittenTitle);
          body.append('collection', collectionId);
          body.append('convert_to_latex', row.convertToLatex ? 'on' : '');
          break;
        case DocType.WEBPAGE:
          if (!row.webpageUrl) {
            throw new Error('Webpage URL is required.');
          }
          try {
            new URL(row.webpageUrl);
          } catch {
            throw new Error('Invalid URL format.');
          }
          url = ingestWebpageUrl;
          body = JSON.stringify({
            url: row.webpageUrl,
            collection_id: collectionId,
            depth: row.webpageCrawlDepth,
          });
          headers['Content-Type'] = 'application/json';
          break;
        default:
          throw new Error('Invalid document type selected.');
      }

      let response: Response;
      if (Array.isArray(body)) {
        for (const singleBody of body) {
          response = await fetch(url, {
            method: 'POST',
            headers,
            body: singleBody,
          });
          if (!response.ok) {
            let errorData: { error?: string };
            try {
              errorData = await response.json();
            } catch {
              errorData = { error: `HTTP error! status: ${response.status}` };
            }
            throw new Error(errorData.error || `Request failed with status ${response.status}`);
          }
        }
        response = { ok: true, status: 200 } as Response;
      } else {
        response = await fetch(url, {
          method: 'POST',
          headers,
          body,
        });
      }

      if (response.ok) {
        if (row.docType === DocType.UPLOADS && response.status === 202) {
          const payload = await response.json();
          const batchId = Number(payload?.batch_id);
          if (!Number.isFinite(batchId)) {
            throw new Error('Upload batch was queued but no batch_id was returned.');
          }
          setUploadBatchIds((prev) => ({ ...prev, [row.id]: batchId }));
          setSubmissionStatus((prev) => ({ ...prev, [row.id]: 'initiated' }));
        } else if (row.docType === DocType.WEBPAGE && response.status === 202) {
          setSubmissionStatus((prev) => ({ ...prev, [row.id]: 'initiated' }));
        } else {
          setSubmissionStatus((prev) => ({ ...prev, [row.id]: 'success' }));
          onUploadSuccess?.();
          updateRow(row.id, {
            uploadFiles: [],
            pdfTitle: '',
            pdfFiles: [],
            arxivId: '',
            vttTitle: '',
            vttFile: null,
            handwrittenTitle: '',
            handwrittenFile: null,
            convertToLatex: false,
            webpageUrl: row.docType !== DocType.WEBPAGE ? '' : row.webpageUrl,
          });
        }
      } else {
        let errorData: { error?: string };
        try {
          errorData = await response.json();
        } catch {
          errorData = { error: `HTTP error! status: ${response.status}` };
        }
        throw new Error(errorData.error || `Request failed with status ${response.status}`);
      }
    } catch (error: unknown) {
      console.error('Submission error for row', row.id, ':', error);
      const message = error instanceof Error ? error.message : String(error);
      setErrorMessages((prev) => ({ ...prev, [row.id]: message }));
      setSubmissionStatus((prev) => ({ ...prev, [row.id]: 'error' }));
    }
  }
}
