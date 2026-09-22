import React from 'react';
import { Pin, PinOff } from 'lucide-react';

interface CitationPinButtonProps {
  pinned: boolean;
  onToggle: () => void;
}

/** Shared header control for the citation panel: pins the slide-out open so
 *  it survives navigation and ignores Escape. */
const CitationPinButton: React.FC<CitationPinButtonProps> = ({ pinned, onToggle }) => (
  <button
    type="button"
    onClick={onToggle}
    aria-label={pinned ? 'Unpin citation panel' : 'Pin citation panel open'}
    aria-pressed={pinned}
    title={pinned ? 'Unpin panel' : 'Pin panel open'}
    className={`p-1 rounded hover:bg-scheme-shade_4 ${
      pinned ? 'text-accent' : 'text-text-normal'
    }`}
  >
    {pinned ? <PinOff className="w-4 h-4" /> : <Pin className="w-4 h-4" />}
  </button>
);

export default CitationPinButton;
