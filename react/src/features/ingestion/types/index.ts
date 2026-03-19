export enum DocType {
  UPLOADS = "uploads",
  PDF = "pdf",
  ARXIV = "arxiv",
  VTT = "vtt",
  WEBPAGE = "webpage",
  HANDWRITTEN = "handwritten",
}

export interface IngestRowsContainerProps {
  ingestUploadsUrl: string;
  ingestArxivUrl: string;
  ingestPdfUrl: string;
  ingestVttUrl: string;
  ingestWebpageUrl: string;
  ingestHandwrittenUrl: string;
  collectionId: string;
  onUploadSuccess?: () => void;
  layout?: "default" | "compact";
}

export interface IngestRowData {
  id: number;
  docType: DocType;
  uploadFiles: File[];
  pdfTitle: string;
  pdfFiles: File[];
  arxivId: string;
  vttTitle: string;
  vttFile: File | null;
  webpageUrl: string;
  webpageCrawlDepth: number;
  handwrittenTitle: string;
  handwrittenFile: File | null;
  convertToLatex: boolean;
}

export interface UploadSummary {
  modalities: string[];
  providers: string[];
  rawMediaSaved: boolean;
  textExtracted: boolean;
}

export type SubmissionStatus = "idle" | "submitting" | "success" | "error" | "initiated";
