import React, { useEffect, useMemo, useState } from 'react';
import { ChevronRight, FileText, Image as ImageIcon } from 'lucide-react';
import { DOC_CHUNK_CITATION_RE } from '../../../utils/linkifyRagCitations';
import { useCitationModal } from './CitationModalProvider';

interface MessageSourcesProps {
  content: string;
  messageUuid?: string;
}

interface SourceRow {
  chunk_id: number;
  doc_id: string;
  title: string;
  modality: string;
}

interface DocGroup {
  docId: string;
  title: string;
  chunkIds: number[];
  hasImage: boolean;
}

/** Parse the cited (docId, chunkId) pairs out of an assistant message, in
 *  first-seen order and de-duplicated. */
function extractCitedChunkIds(content: string): number[] {
  DOC_CHUNK_CITATION_RE.lastIndex = 0;
  const seen = new Set<number>();
  const ids: number[] = [];
  let match: RegExpExecArray | null;
  while ((match = DOC_CHUNK_CITATION_RE.exec(content))) {
    const chunkId = Number(match[2]);
    if (Number.isNaN(chunkId) || seen.has(chunkId)) continue;
    seen.add(chunkId);
    ids.push(chunkId);
  }
  return ids;
}

/**
 * Per-message "Sources" footer: groups every citation in an assistant message
 * by document so a paper cited across several chunks shows once, expandable to
 * its individual passages. Titles come from the batched citation_sources
 * endpoint (one request per message, not one per citation).
 */
const MessageSources: React.FC<MessageSourcesProps> = ({ content, messageUuid }) => {
  const { openCitation } = useCitationModal();
  const [rows, setRows] = useState<SourceRow[] | null>(null);
  const [expanded, setExpanded] = useState(false);

  const chunkIds = useMemo(() => extractCitedChunkIds(content || ''), [content]);
  // Stable dependency key so the effect only refetches when the set changes.
  const chunkKey = chunkIds.join(',');

  useEffect(() => {
    if (chunkIds.length === 0) {
      setRows(null);
      return;
    }
    const apiUrl = window.apiUrls?.api_citation_sources;
    if (!apiUrl) return;
    let cancelled = false;
    fetch(apiUrl, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunk_ids: chunkIds }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled || !data) return;
        setRows(Array.isArray(data.sources) ? data.sources : null);
      })
      .catch(() => {
        /* best-effort — the inline citation links still work. */
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chunkKey]);

  const groups = useMemo<DocGroup[]>(() => {
    if (!rows) return [];
    const byDoc = new Map<string, DocGroup>();
    // Preserve citation order from chunkIds.
    const rowByChunk = new Map<number, SourceRow>();
    for (const row of rows) rowByChunk.set(row.chunk_id, row);
    for (const chunkId of chunkIds) {
      const row = rowByChunk.get(chunkId);
      if (!row) continue;
      let group = byDoc.get(row.doc_id);
      if (!group) {
        group = { docId: row.doc_id, title: row.title, chunkIds: [], hasImage: false };
        byDoc.set(row.doc_id, group);
      }
      group.chunkIds.push(chunkId);
      if (row.modality === 'image') group.hasImage = true;
    }
    return Array.from(byDoc.values());
  }, [rows, chunkIds]);

  if (chunkIds.length === 0 || groups.length === 0) return null;

  const totalPassages = groups.reduce((n, g) => n + g.chunkIds.length, 0);

  return (
    <div className="mt-2 w-full border-t border-border-mid_contrast pt-2">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1 text-[11px] font-medium text-text-low_contrast hover:text-text-normal"
        aria-expanded={expanded}
      >
        <ChevronRight
          className={`w-3.5 h-3.5 transition-transform ${expanded ? 'rotate-90' : ''}`}
        />
        Sources · {groups.length} {groups.length === 1 ? 'document' : 'documents'}
        {' · '}
        {totalPassages} {totalPassages === 1 ? 'passage' : 'passages'}
      </button>
      {expanded && (
        <ul className="mt-1.5 space-y-1.5">
          {groups.map((group) => (
            <li key={group.docId} className="text-[12px]">
              <div className="flex items-center gap-1 text-text-normal">
                {group.hasImage ? (
                  <ImageIcon className="w-3.5 h-3.5 flex-shrink-0 text-text-low_contrast" />
                ) : (
                  <FileText className="w-3.5 h-3.5 flex-shrink-0 text-text-low_contrast" />
                )}
                <span className="truncate font-medium" title={group.title}>
                  {group.title}
                </span>
              </div>
              <div className="flex flex-wrap gap-1 mt-0.5 ml-5">
                {group.chunkIds.map((chunkId) => (
                  <button
                    key={chunkId}
                    type="button"
                    onClick={() =>
                      openCitation({ docId: group.docId, chunkId: String(chunkId), messageUuid })
                    }
                    className="px-1.5 py-0.5 rounded bg-scheme-shade_4 text-text-low_contrast hover:text-text-normal hover:bg-scheme-shade_5 text-[11px]"
                  >
                    chunk {chunkId}
                  </button>
                ))}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default MessageSources;
