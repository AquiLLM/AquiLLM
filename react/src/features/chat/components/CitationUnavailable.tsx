import React, { useEffect } from 'react';
import { X, FileX } from 'lucide-react';
import CitationPinButton from './CitationPinButton';

interface CitationUnavailableProps {
  onClose: () => void;
  pinned?: boolean;
  onTogglePin?: () => void;
}

/**
 * Panel shown when a citation points at a document/chunk that no longer
 * exists (chunk_detail returned 404). The source was deleted after the
 * message was written, so there is nothing to open — we say so plainly
 * instead of offering an "Open document page" link that also 404s.
 */
const CitationUnavailable: React.FC<CitationUnavailableProps> = ({
  onClose,
  pinned = false,
  onTogglePin,
}) => {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !pinned) onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, pinned]);

  return (
    <div className="bg-scheme-shade_3 h-full flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-mid_contrast">
        <div className="min-w-0">
          <div className="text-text-normal font-semibold truncate">Source unavailable</div>
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
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
      <div className="flex-1 min-h-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
        <FileX className="w-8 h-8 text-text-low_contrast" />
        <p className="text-sm font-medium text-text-normal">
          This document is no longer available.
        </p>
        <p className="text-xs text-text-low_contrast max-w-xs">
          The cited source was removed after this message was written, so the
          referenced passage can't be shown.
        </p>
      </div>
    </div>
  );
};

export default CitationUnavailable;
