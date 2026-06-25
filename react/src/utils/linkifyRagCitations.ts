import formatUrl from './formatUrl';

/** Matches RAG citation tokens like `[doc:<uuid> chunk:<id>]`. */
export const DOC_CHUNK_CITATION_RE =
  /\[doc:([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\s+chunk:(\d+)\]/g;

/**
 * Wrap citation tokens in anchor tags so they open the document page with ?chunk=…
 * (handled by the Django document view). Requires `window.pageUrls.document`.
 */
export function linkifyRagCitations(content: string): string {
  if (typeof window === 'undefined' || !window.pageUrls?.document) {
    return content;
  }
  const docPattern = window.pageUrls.document;
  return content.replace(DOC_CHUNK_CITATION_RE, (match, docId: string, chunkId: string) => {
    try {
      // Keep the document-page URL as the anchor's href so modifier-click
      // (ctrl/cmd/shift/middle) falls through to the native "open in new
      // tab" behaviour. Plain clicks are intercepted by the chat-level
      // delegate (see MessageBubble) and routed to the PDF citation modal.
      const href = `${formatUrl(docPattern, { doc_id: docId })}?chunk=${chunkId}`;
      return (
        `<a href="${href}" target="_blank" rel="noopener noreferrer" ` +
        `class="rag-citation-link inline-block rounded-md px-1.5 py-0.5 align-baseline ` +
        `border border-border-mid_contrast bg-scheme-shade_4 text-accent text-[0.9em] ` +
        `no-underline hover:underline font-medium cursor-pointer" ` +
        `data-doc-id="${docId}" data-chunk-id="${chunkId}">` +
        `[doc:${docId} chunk:${chunkId}]</a>`
      );
    } catch {
      return match;
    }
  });
}
