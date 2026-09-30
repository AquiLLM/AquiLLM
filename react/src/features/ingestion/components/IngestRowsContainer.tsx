import React, { useState, useCallback, useRef } from 'react';

import {
  DocType,
  IngestRowsContainerProps,
  IngestRowData,
  UploadSummary,
  SubmissionStatus,
  UploadBatch,
} from '../types';
import { IngestRow } from './IngestRow';
import { useIngestUploadBatchPolling } from '../hooks/useIngestUploadBatchPolling';
import { runIngestRowSubmissions } from '../utils/runIngestRowSubmissions';
import IngestRowStatusBlocks from './IngestRowStatusBlocks';

export { DocType } from '../types';

const IngestRowsContainer: React.FC<IngestRowsContainerProps> = ({
  ingestUploadsUrl,
  ingestArxivUrl,
  ingestPdfUrl,
  ingestVttUrl,
  ingestWebpageUrl,
  ingestHandwrittenUrl,
  collectionId,
  onUploadSuccess,
  layout = 'default',
}) => {
  const [rows, setRows] = useState<IngestRowData[]>([
    {
      id: 0,
      docType: DocType.UPLOADS,
      uploadFiles: [],
      pdfTitle: '',
      pdfFiles: [],
      arxivId: '',
      vttTitle: '',
      vttFile: null,
      webpageUrl: '',
      webpageCrawlDepth: 1,
      handwrittenTitle: '',
      handwrittenFile: null,
      convertToLatex: false,
    },
  ]);
  const [submissionStatus, setSubmissionStatus] = useState<{
    [key: number]: SubmissionStatus;
  }>({});
  const [errorMessages, setErrorMessages] = useState<{ [key: number]: string }>({});
  const [uploadBatches, setUploadBatches] = useState<Record<number, UploadBatch>>({});
  const submitting = useRef(false);
  const [uploadSummaries, setUploadSummaries] = useState<{ [key: number]: UploadSummary }>({});

  const updateRow = useCallback((id: number, updates: Partial<IngestRowData>) => {
    setRows((prevRows) =>
      prevRows.map((row) => (row.id === id ? { ...row, ...updates } : row))
    );
  }, []);

  const retryStatus = useIngestUploadBatchPolling({
    uploadBatches,
    setUploadBatches,
    setUploadSummaries,
    setErrorMessages,
    setSubmissionStatus,
    ingestUploadsUrl,
    onUploadSuccess,
    updateRow,
  });

  const addRow = () => {
    setRows((prevRows) => [
      ...prevRows,
      {
        id: prevRows.length > 0 ? prevRows[prevRows.length - 1].id + 1 : 0,
        docType: DocType.UPLOADS,
        uploadFiles: [],
        pdfTitle: '',
        pdfFiles: [],
        arxivId: '',
        vttTitle: '',
        vttFile: null,
        webpageUrl: '',
        webpageCrawlDepth: 1,
        handwrittenTitle: '',
        handwrittenFile: null,
        convertToLatex: false,
      },
    ]);
  };

  const updateRowDocType = (id: number, newDocType: DocType) => {
    setRows((prevRows) =>
      prevRows.map((row) => {
        if (row.id === id) {
          const newRow = { ...row, docType: newDocType };
          if (newDocType !== DocType.UPLOADS) {
            newRow.uploadFiles = [];
          }
          if (newDocType !== DocType.PDF) {
            newRow.pdfFiles = [];
            newRow.pdfTitle = '';
          }
          if (newDocType !== DocType.ARXIV) {
            newRow.arxivId = '';
          }
          if (newDocType !== DocType.VTT) {
            newRow.vttFile = null;
            newRow.vttTitle = '';
          }
          if (newDocType !== DocType.WEBPAGE) {
            newRow.webpageUrl = '';
            newRow.webpageCrawlDepth = 1;
          }
          if (newDocType !== DocType.HANDWRITTEN) {
            newRow.handwrittenFile = null;
            newRow.handwrittenTitle = '';
            newRow.convertToLatex = false;
          }
          return newRow;
        }
        return row;
      })
    );
  };

  const handleSubmit = async () => {
    if (submitting.current) return;
    const eligibleRows = rows.filter(row => !uploadBatches[row.id] &&
      submissionStatus[row.id] !== 'submitting' && submissionStatus[row.id] !== 'initiated' &&
      submissionStatus[row.id] !== 'success');
    if (!eligibleRows.length) return;
    submitting.current = true;
    try {
      await runIngestRowSubmissions(
        eligibleRows,
        collectionId,
        {
          ingestUploadsUrl,
          ingestArxivUrl,
          ingestPdfUrl,
          ingestVttUrl,
          ingestWebpageUrl,
          ingestHandwrittenUrl,
        },
        {
          setErrorMessages,
          setSubmissionStatus,
          setUploadBatches,
          setUploadSummaries,
          updateRow,
          onUploadSuccess,
        }
      );
    } finally {
      submitting.current = false;
    }
  };

  const actionButtons = (
    <>
      <button
        onClick={addRow}
        className="h-[40px] px-4 rounded-[20px] bg-scheme-shade_4 text-text-normal border border-border-high_contrast hover:bg-scheme-shade_5 transition-colors"
        type="button"
      >
        Add Another
      </button>
      <button
        onClick={handleSubmit}
        disabled={Object.values(submissionStatus).some((s) => s === 'submitting') ||
          rows.every(row => uploadBatches[row.id] || submissionStatus[row.id] === 'initiated' || submissionStatus[row.id] === 'success')}
        className="h-[40px] px-4 rounded-[20px] bg-accent text-text-normal border border-border-high_contrast hover:bg-accent-dark transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
        type="button"
      >
        Submit All
      </button>
    </>
  );

  return (
    <div className={layout === 'compact' ? 'space-y-3' : 'space-y-4'}>
      {rows.map((row, index) => (
        <div
          key={row.id}
          className={`${
            layout === 'compact'
              ? 'bg-transparent p-0 rounded-[16px]'
              : 'bg-scheme-shade_1 p-4 border border-border-mid_contrast rounded-lg shadow'
          }`}
        >
          <IngestRow
            row={row}
            onDocTypeChange={(id, type) => {
              if (uploadBatches[id] || submissionStatus[id] === 'submitting') return;
              updateRowDocType(id, type);
              setSubmissionStatus(prev => ({ ...prev, [id]: 'idle' }));
            }}
            onRowChange={(id, changes) => {
              if (uploadBatches[id] || submissionStatus[id] === 'submitting') return;
              updateRow(id, changes);
              setSubmissionStatus(prev => ({ ...prev, [id]: 'idle' }));
            }}
            layout={layout}
            actions={layout === 'compact' && index === 0 ? actionButtons : undefined}
          />
          <IngestRowStatusBlocks
            row={row}
            submissionStatus={submissionStatus[row.id]}
            errorMessage={errorMessages[row.id]}
            uploadSummary={uploadSummaries[row.id]}
          />
          {submissionStatus[row.id] === 'status-error' && (
            <button type="button" onClick={() => retryStatus(row.id)} className="mt-2 underline text-accent">Retry status</button>
          )}
        </div>
      ))}
      {layout !== 'compact' && <div className="flex items-center gap-3 pt-1">{actionButtons}</div>}
    </div>
  );
};

export default IngestRowsContainer;
