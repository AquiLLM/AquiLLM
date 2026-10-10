import { useEffect, useMemo, useState } from 'react';
import { getCsrfCookie } from '../../../main';
import { DOC_CHUNK_CITATION_RE } from '../../../utils/linkifyRagCitations';

interface SourceRow {
  chunk_id: number;
  doc_id: string;
  title: string;
  modality: string;
}

export interface MessageCitation {
  docId: string;
  chunkId: string;
  sourceNumber: number;
  passageNumber: number;
  title?: string;
  modality?: string;
}

/** One batched, permission-checked lookup shared by inline citations and the footer. */
export function useMessageCitations(content: string): MessageCitation[] {
  const references = useMemo(() => {
    const documents = new Map<string, Map<string, MessageCitation>>();
    const citations: MessageCitation[] = [];
    for (const match of content.matchAll(new RegExp(DOC_CHUNK_CITATION_RE))) {
      const docId = match[1].toLowerCase();
      const chunkId = match[2];
      let passages = documents.get(docId);
      if (!passages) {
        passages = new Map();
        documents.set(docId, passages);
      }
      if (passages.has(chunkId)) continue;
      const citation = {
        docId, chunkId,
        sourceNumber: passages.values().next().value?.sourceNumber ?? documents.size,
        passageNumber: passages.size + 1,
      };
      passages.set(chunkId, citation);
      citations.push(citation);
    }
    return citations;
  }, [content]);
  const [rows, setRows] = useState<SourceRow[]>([]);
  const chunkKey = JSON.stringify([...new Set(references.map((citation) => Number(citation.chunkId)))]);

  useEffect(() => {
    const chunkIds: number[] = JSON.parse(chunkKey);
    const apiUrl = window.apiUrls?.api_citation_sources;
    if (!apiUrl || chunkIds.length === 0) return;
    const controller = new AbortController();
    let cancelled = false;
    async function loadSources() {
      try {
        const result = await fetch(apiUrl, {
          method: 'POST',
          credentials: 'include',
          signal: controller.signal,
          headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrfCookie() },
          body: JSON.stringify({ chunk_ids: chunkIds }),
        });
        const data = result.ok ? await result.json() : null;
        if (!cancelled) setRows(Array.isArray(data?.sources) ? data.sources : []);
      } catch {
        // Numbered links still open the original passage if metadata is unavailable.
        if (!cancelled) setRows([]);
      }
    }
    void loadSources();
    return () => { cancelled = true; controller.abort(); };
  }, [chunkKey]);

  return useMemo(() => {
    const byReference = new Map(rows.filter((row) => row && typeof row.doc_id === 'string')
      .map((row) => [`${row.doc_id.toLowerCase()}:${row.chunk_id}`, row]));
    return references.map((citation) => {
      // Never substitute a title when the chunk belongs to a different document.
      const row = byReference.get(`${citation.docId}:${citation.chunkId}`);
      return { ...citation, title: typeof row?.title === 'string' ? row.title.trim() : undefined,
        modality: row?.modality };
    });
  }, [references, rows]);
}
