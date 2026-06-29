import React, { useEffect, useState } from 'react';
import { X, ExternalLink } from 'lucide-react';
import type { CitationChunkDetail } from './citationTypes';
import CitationPinButton from './CitationPinButton';

interface ImageCitationModalProps {
  docId: string;
  chunkId: string;
  /** Unused for image chunks (no prose to narrow) — accepted for a uniform
   *  modal signature with the PDF/text modals. */
  messageUuid?: string;
  /** Chunk prefetched by the provider; image chunks are always preloaded. */
  preloadedChunk?: CitationChunkDetail | null;
  onClose: () => void;
  /** Pin state for the slide-out panel. */
  pinned?: boolean;
  onTogglePin?: () => void;
}

/**
 * Citation modal for image/figure chunks. The chunk's `content` holds the
 * extracted caption / OCR text and `image_url` points at the served figure
 * binary (see DocumentFigure / the document_image view). There is nothing to
 * highlight or LLM-narrow, so this simply renders the figure with its caption.
 */
const ImageCitationModal: React.FC<ImageCitationModalProps> = ({
  preloadedChunk,
  onClose,
  pinned = false,
  onTogglePin,
}) => {
  const chunk = preloadedChunk ?? null;
  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    setImageError(false);
  }, [chunk?.image_url]);

  const renderBody = () => {
    if (!chunk) {
      return <div className="p-6 text-text-low_contrast text-sm">Loading citation…</div>;
    }
    return (
      <div className="flex-1 min-h-0 overflow-auto bg-scheme-shade_5 p-4 space-y-4">
        {chunk.image_url && !imageError ? (
          <a href={chunk.image_url} target="_blank" rel="noopener noreferrer" className="block">
            <img
              src={chunk.image_url}
              alt={chunk.content || 'Cited figure'}
              onError={() => setImageError(true)}
              className="max-w-full h-auto rounded border border-border-mid_contrast bg-white"
            />
          </a>
        ) : (
          <div className="text-sm text-text-low_contrast">
            Couldn't load the figure image.
          </div>
        )}
        {chunk.content && (
          <div>
            <div className="text-xs uppercase tracking-wide text-text-low_contrast mb-1">
              Caption
            </div>
            <p className="text-sm text-text-normal whitespace-pre-wrap leading-relaxed">
              {chunk.content}
            </p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-scheme-shade_3 h-full flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-mid_contrast">
        <div className="min-w-0">
          <div className="text-text-normal font-semibold truncate">
            {chunk?.document.title || 'Figure citation'}
          </div>
          {chunk && (
            <div className="text-xs text-text-low_contrast truncate">
              Figure · chunk {chunk.chunk_number}
            </div>
          )}
          {chunk?.document.source_url && (
            <a
              href={chunk.document.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline truncate inline-block max-w-full"
              title={chunk.document.source_url}
            >
              View source
            </a>
          )}
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          {chunk?.image_url && (
            <a
              href={chunk.image_url}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Open figure image in new tab"
              className="p-1 rounded hover:bg-scheme-shade_4 text-text-normal"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
          )}
          {onTogglePin && <CitationPinButton pinned={pinned} onToggle={onTogglePin} />}
          <button
            type="button"
            onClick={onClose}
            aria-label="Close citation panel"
            className="p-1 rounded hover:bg-scheme-shade_4 text-text-normal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
      {renderBody()}
    </div>
  );
};

export default ImageCitationModal;
