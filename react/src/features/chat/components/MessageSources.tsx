import React, { useMemo, useState } from 'react';
import { ChevronRight, FileText, Image as ImageIcon } from 'lucide-react';
import type { MessageCitation } from '../hooks/useMessageCitations';
import { useCitationModal } from './CitationModalProvider';

interface MessageSourcesProps {
  citations: MessageCitation[];
  messageUuid?: string;
}

interface DocGroup {
  docId: string;
  title: string;
  citations: MessageCitation[];
  hasImage: boolean;
}

/** The same resolved citations used by the inline links, grouped by document. */
const MessageSources: React.FC<MessageSourcesProps> = ({ citations, messageUuid }) => {
  const { openCitation } = useCitationModal();
  const [expanded, setExpanded] = useState(false);
  const groups = useMemo(() => {
    const byDoc = new Map<string, DocGroup>();
    for (const citation of citations) {
      let group = byDoc.get(citation.docId);
      if (!group) {
        group = { docId: citation.docId, title: citation.title || `Source ${citation.sourceNumber}`,
          citations: [], hasImage: false };
        byDoc.set(citation.docId, group);
      }
      if (citation.title) group.title = citation.title;
      group.citations.push(citation);
      if (citation.modality === 'image') group.hasImage = true;
    }
    return Array.from(byDoc.values());
  }, [citations]);

  if (groups.length === 0) return null;
  const totalPassages = citations.length;

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
                {group.citations.map((citation) => (
                  <button
                    key={citation.chunkId}
                    type="button"
                    aria-label={`${group.title} · Passage ${citation.passageNumber}`}
                    onClick={() =>
                      openCitation({ docId: group.docId, chunkId: citation.chunkId, messageUuid })
                    }
                    className="px-1.5 py-0.5 rounded bg-scheme-shade_4 text-text-low_contrast hover:text-text-normal hover:bg-scheme-shade_5 text-[11px]"
                  >
                    Passage {citation.passageNumber}
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
