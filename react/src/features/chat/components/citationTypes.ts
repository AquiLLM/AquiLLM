/** Document fields returned by both compact and full chunk-detail responses. */
export interface CitationDocumentSummary {
  id: string;
  title: string;
  /** Django classname: PDFDocument, RawTextDocument, VTTDocument, … */
  type: string;
  /** True when the doc has a PDF (native, compiled, or crawl-rendered). */
  has_pdf: boolean;
  /** Origin URL for crawled webpages; null otherwise. */
  source_url: string | null;
}

/** Compact response shape from `api_chunk_detail?include_full_text=0`. */
export interface CitationChunkSummary {
  content: string;
  chunk_number: number;
  start_position: number;
  end_position: number;
  /** VTT transcript chunks carry the timestamp; null for everything else. */
  start_time: number | null;
  /** Chunk modality: "text" or "image". */
  modality: string;
  /** Served image URL for image chunks (figures); null for text chunks. */
  image_url: string | null;
  document: CitationDocumentSummary;
}

/** Full response shape from `api_chunk_detail`. */
export interface CitationChunkDetail extends CitationChunkSummary {
  document: CitationDocumentSummary & {
    /** Full document text, possibly windowed around the chunk for very long docs. */
    full_text: string;
    /** When full_text is windowed, this is the character offset into the original full_text. */
    text_offset: number;
  };
}
