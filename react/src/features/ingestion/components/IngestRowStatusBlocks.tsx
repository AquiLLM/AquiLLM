import React from 'react';
import { DocType, type IngestRowData, type SubmissionStatus, type UploadSummary } from '../types';

export interface IngestRowStatusBlocksProps {
  row: IngestRowData;
  submissionStatus?: SubmissionStatus;
  errorMessage?: string;
  uploadSummary?: UploadSummary;
}

const IngestRowStatusBlocks: React.FC<IngestRowStatusBlocksProps> = ({
  row,
  submissionStatus,
  errorMessage,
  uploadSummary,
}) => (
  <>
    {submissionStatus === 'submitting' && (
      <p className="text-text-low_contrast mt-2">Submitting...</p>
    )}
    {submissionStatus === 'success' && (
      <p className="text-accent-light mt-2">Submission successful!</p>
    )}
    {submissionStatus === 'initiated' && (
      <p className="text-accent-light mt-2">
        {row.docType === DocType.UPLOADS ? 'Batch ingestion queued...' : 'Webpage crawl initiated...'}
      </p>
    )}
    {submissionStatus === 'error' && errorMessage && (
      <p className="text-red-dark mt-2">Error: {errorMessage}</p>
    )}
    {uploadSummary && (
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
        {uploadSummary.modalities.map((modality) => (
          <span
            key={`${row.id}-${modality}`}
            className="px-2 py-1 rounded-full bg-scheme-shade_5 border border-border-mid_contrast text-text-normal"
          >
            {modality}
          </span>
        ))}
        <span className="px-2 py-1 rounded-full bg-scheme-shade_5 border border-border-mid_contrast text-text-normal">
          raw media: {uploadSummary.rawMediaSaved ? 'saved' : 'none'}
        </span>
        <span className="px-2 py-1 rounded-full bg-scheme-shade_5 border border-border-mid_contrast text-text-normal">
          text: {uploadSummary.textExtracted ? 'extracted' : 'none'}
        </span>
        {uploadSummary.providers.map((provider) => (
          <span
            key={`${row.id}-provider-${provider}`}
            className="px-2 py-1 rounded-full bg-scheme-shade_5 border border-border-mid_contrast text-text-normal"
          >
            provider: {provider}
          </span>
        ))}
      </div>
    )}
  </>
);

export default IngestRowStatusBlocks;
