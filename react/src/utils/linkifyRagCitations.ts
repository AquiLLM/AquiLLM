import formatUrl from './formatUrl';

/** Matches RAG citation tokens like `[doc:<uuid> chunk:<id>]`. */
export const DOC_CHUNK_CITATION_RE =
  /\[doc:([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\s+chunk:(\d+)\]/g;

/**
 * Wrap citation tokens in anchor tags so they open the document page with ?chunk=…
 * (handled by the Django document view). Requires `window.pageUrls.document`.
 */
export function linkifyRagCitations(content: string): string {
  const docPattern = typeof window === 'undefined' ? undefined : window.pageUrls?.document;
  const sourceNumbers = new Map<string, number>();
  return content.replace(DOC_CHUNK_CITATION_RE, (_match, docId: string, chunkId: string) => {
    docId = docId.toLowerCase();
    if (!sourceNumbers.has(docId)) sourceNumbers.set(docId, sourceNumbers.size + 1);
    const label = `Source ${sourceNumbers.get(docId)}`;
    if (!docPattern) return `[${label}]`;
    try {
      // Keep the document-page URL as the anchor's href so modifier-click
      // (ctrl/cmd/shift/middle) falls through to the native "open in new
      // tab" behaviour. Plain clicks are intercepted by the chat-level
      // delegate (see MessageBubble) and routed to the PDF citation modal.
      const href = `${formatUrl(docPattern, { doc_id: docId })}?chunk=${chunkId}`;
      return (
        `<a href="${href}" target="_blank" rel="noopener noreferrer" ` +
        `class="rag-citation-link" ` +
        `data-doc-id="${docId}" data-chunk-id="${chunkId}">` +
        `${label}</a>`
      );
    } catch {
      return `[${label}]`;
    }
  });
}
