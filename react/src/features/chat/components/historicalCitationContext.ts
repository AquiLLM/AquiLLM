import type { CitationChunkDetail } from './citationTypes';
import formatUrl from '../../../utils/formatUrl';
import { findNormalizedOffset, normalizeForSearch } from '../../../utils/textOffsetMatch';

/** Verified offsets are Python/Unicode codepoints; rendering uses UTF-16. */
export function citationHighlight(chunk: CitationChunkDetail, narrowQuote: string | null) {
  const text = chunk.document.full_text;
  const start = chunk.start_position - chunk.document.text_offset;
  const end = chunk.end_position - chunk.document.text_offset;
  if (chunk.source_context_status === 'unavailable') return null;
  let chunkStart: number;
  let chunkEnd: number;
  if (chunk.source_context_status === 'verified') {
    const points = Array.from(text);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > points.length) return null;
    if (points.slice(start, end).join('') !== chunk.content) return null;
    chunkStart = points.slice(0, start).join('').length;
    chunkEnd = chunkStart + chunk.content.length;
  } else {
    // Older API responses did not certify offsets; preserve their search fallback.
    chunkStart = Math.max(0, Math.min(start, text.length));
    chunkEnd = Math.max(chunkStart, Math.min(end, text.length));
    const content = normalizeForSearch(chunk.content).trim();
    if (content && normalizeForSearch(text.slice(chunkStart, chunkEnd)).trim() !== content) {
      const located = findNormalizedOffset(text, chunk.content);
      if (located) { chunkStart = located.start; chunkEnd = located.end; }
    }
  }
  const hit = narrowQuote ? findNormalizedOffset(text.slice(chunkStart, chunkEnd), narrowQuote) : null;
  return { chunkStart, chunkEnd,
    tightStart: hit ? chunkStart + hit.start : null,
    tightEnd: hit ? chunkStart + hit.end : null };
}

/** Keep the caller's resource query/hash while selecting the retained chunk. */
export function citationResourceUrl(url: string, chunkId: string): string {
  const parsed = new URL(url, window.location.href);
  parsed.searchParams.set('chunk_id', chunkId);
  return /^[a-z][a-z\d+.-]*:/i.test(url) || url.startsWith('//')
    ? parsed.toString() : `${parsed.pathname}${parsed.search}${parsed.hash}`;
}

export function citationPdfUrl(pattern: string | undefined, docId: string, chunkId: string) {
  return pattern ? citationResourceUrl(formatUrl(pattern, { doc_id: docId }), chunkId) : null;
}
