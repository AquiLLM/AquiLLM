import React, { useState, useCallback } from 'react';

import {
  DocType,
  IngestRowsContainerProps,
  IngestRowData,
  UploadSummary,
  SubmissionStatus,
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
  const [uploadBatchIds, setUploadBatchIds] = useState<{ [key: number]: number }>({});
  const [uploadSummaries, setUploadSummaries] = useState<{ [key: number]: UploadSummary }>({});

  const updateRow = useCallback((id: number, updates: Partial<IngestRowData>) => {
    setRows((prevRows) =>
      prevRows.map((row) => (row.id === id ? { ...row, ...updates } : row))
    );
  }, []);

  useIngestUploadBatchPolling({
    uploadBatchIds,
    setUploadBatchIds,
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
    await runIngestRowSubmissions(
      rows,
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
        setUploadBatchIds,
        setUploadSummaries,
        updateRow,
        onUploadSuccess,
      }
    );
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
        disabled={Object.values(submissionStatus).some((s) => s === 'submitting')}
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
            onDocTypeChange={updateRowDocType}
            onRowChange={updateRow}
            layout={layout}
            actions={layout === 'compact' && index === 0 ? actionButtons : undefined}
          />
          <IngestRowStatusBlocks
            row={row}
            submissionStatus={submissionStatus[row.id]}
            errorMessage={errorMessages[row.id]}
            uploadSummary={uploadSummaries[row.id]}
          />
        </div>
      ))}
      {layout !== 'compact' && <div className="flex items-center gap-3 pt-1">{actionButtons}</div>}
    </div>
  );
};

export default IngestRowsContainer;
