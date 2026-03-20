import React from 'react';
import { DocType, IngestRowData } from '../types';
import { DocTypeToggle } from './DocTypeToggle';
import {
  UploadsForm,
  PDFForm,
  ArxivForm,
  VTTForm,
  HandwrittenForm,
  WebpageForm,
} from './forms';

interface IngestRowProps {
  row: IngestRowData;
  onDocTypeChange: (id: number, newDocType: DocType) => void;
  onRowChange: (id: number, updates: Partial<IngestRowData>) => void;
  layout?: "default" | "compact";
  actions?: React.ReactNode;
}

export const IngestRow: React.FC<IngestRowProps> = ({
  row,
  onDocTypeChange,
  onRowChange,
  layout = "default",
  actions,
}) => {
  const rowForm = row.docType === DocType.UPLOADS ? (
    <UploadsForm
      uploadFiles={row.uploadFiles}
      onFileChange={(files) => onRowChange(row.id, { uploadFiles: files })}
    />
  ) : row.docType === DocType.PDF ? (
    <PDFForm
      pdfTitle={row.pdfTitle}
      pdfFiles={row.pdfFiles}
      onTitleChange={(value) => onRowChange(row.id, { pdfTitle: value })}
      onFileChange={(files) => onRowChange(row.id, { pdfFiles: files })}
    />
  ) : row.docType === DocType.ARXIV ? (
    <ArxivForm
      value={row.arxivId}
      onValueChange={(value) => onRowChange(row.id, { arxivId: value })}
    />
  ) : row.docType === DocType.VTT ? (
    <VTTForm
      vttTitle={row.vttTitle}
      vttFile={row.vttFile}
      onTitleChange={(value) => onRowChange(row.id, { vttTitle: value })}
      onFileChange={(file) => onRowChange(row.id, { vttFile: file })}
    />
  ) : row.docType === DocType.HANDWRITTEN ? (
    <HandwrittenForm
      handwrittenTitle={row.handwrittenTitle}
      handwrittenFile={row.handwrittenFile}
      convertToLatex={row.convertToLatex}
      onTitleChange={(value) => onRowChange(row.id, { handwrittenTitle: value })}
      onFileChange={(file) => onRowChange(row.id, { handwrittenFile: file })}
      onConvertChange={(value) => onRowChange(row.id, { convertToLatex: value })}
    />
  ) : (
    <WebpageForm
      urlValue={row.webpageUrl}
      depthValue={row.webpageCrawlDepth}
      onUrlChange={(value) => onRowChange(row.id, { webpageUrl: value })}
      onDepthChange={(value) => onRowChange(row.id, { webpageCrawlDepth: value })}
    />
  );

  if (layout === "compact") {
    return (
      <div className="flex flex-col lg:flex-row items-start gap-3">
        <div className="flex-1 w-full">{rowForm}</div>
        <div className="shrink-0 flex flex-col items-end gap-3">
          <DocTypeToggle
            docType={row.docType}
            setDocType={(newType) => onDocTypeChange(row.id, newType)}
          />
          {actions && (
            <div className="flex items-center gap-3">
              {actions}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="pt-[1rem] flex items-start space-x-4">
      <DocTypeToggle
        docType={row.docType}
        setDocType={(newType) => onDocTypeChange(row.id, newType)}
      />
      <div className="flex-1">
        {rowForm}
      </div>
    </div>
  );
};
