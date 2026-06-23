/** Response shape from `api_chunk_detail`. Shared between the PDF and
 *  text citation modals (and the provider that prefetches it). */
export interface CitationChunkDetail {
  content: string;
  chunk_number: number;
  start_position: number;
  end_position: number;
  /** VTT transcript chunks carry the timestamp; null for everything else. */
  start_time: number | null;
  document: {
    id: string;
    title: string;
    /** Django classname: PDFDocument, RawTextDocument, VTTDocument, … */
    type: string;
    /** True when the doc has a PDF (native, compiled, or crawl-rendered). */
    has_pdf: boolean;
    /** Origin URL for crawled webpages; null otherwise. */
    source_url: string | null;
    /** Full document text, possibly windowed around the chunk for very long docs. */
    full_text: string;
    /** When full_text is windowed, this is the character offset into the original full_text. */
    text_offset: number;
  };
}
