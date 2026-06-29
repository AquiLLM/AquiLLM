import React, { createContext, lazy, Suspense, useCallback, useContext, useEffect, useState } from 'react';
import formatUrl from '../../../utils/formatUrl';
import type { CitationChunkDetail } from './citationTypes';

// Lazy-loaded to keep react-pdf / pdfjs-dist (which use `import.meta`) out
// of the main bundle. main.js is loaded as a classic <script>, so any
// top-level `import.meta` would be a parse error and stop React from
// mounting on every page. With Vite's IIFE + inlineDynamicImports the
// dynamic import becomes synchronous in the bundle, but the lazy wrapper
// is harmless and preserves code-splitting if we ever switch back to ESM.
const PDFCitationModal = lazy(() => import('./PDFCitationModal'));
const TextCitationModal = lazy(() => import('./TextCitationModal'));
const ImageCitationModal = lazy(() => import('./ImageCitationModal'));
const CitationUnavailable = lazy(() => import('./CitationUnavailable'));

/** Width of the slide-out citation panel. The outer container animates
 * between 0 and this width; the inner content stays fixed-width so it
 * doesn't reflow as the panel opens. */
export const CITATION_PANEL_WIDTH = 640;

interface CitationTarget {
  docId: string;
  chunkId: string;
  /** Optional: assistant message UUID, used to enable LLM narrowing of the
   * highlight. When omitted (e.g. citations clicked outside chat), the
   * panel falls back to whole-chunk highlighting. */
  messageUuid?: string;
}

interface CitationModalContextValue {
  openCitation: (target: CitationTarget) => void;
  closeCitation: () => void;
  target: CitationTarget | null;
  isOpen: boolean;
  /** When pinned, the panel stays open across navigation and ignores Escape. */
  pinned: boolean;
  togglePin: () => void;
}

const CitationModalContext = createContext<CitationModalContextValue | null>(null);

const PIN_STORAGE_KEY = 'aquillm.citationPanel.pinned';

export function useCitationModal(): CitationModalContextValue {
  const ctx = useContext(CitationModalContext);
  if (!ctx) {
    return {
      openCitation: () => {},
      closeCitation: () => {},
      target: null,
      isOpen: false,
      pinned: false,
      togglePin: () => {},
    };
  }
  return ctx;
}

export const CitationModalProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [target, setTarget] = useState<CitationTarget | null>(null);
  const [pinned, setPinned] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem(PIN_STORAGE_KEY) === '1';
    } catch {
      return false;
    }
  });

  const openCitation = useCallback((next: CitationTarget) => setTarget(next), []);
  const closeCitation = useCallback(() => setTarget(null), []);
  const togglePin = useCallback(() => {
    setPinned((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem(PIN_STORAGE_KEY, next ? '1' : '0');
      } catch {
        /* ignore storage failures */
      }
      return next;
    });
  }, []);

  return (
    <CitationModalContext.Provider
      value={{ openCitation, closeCitation, target, isOpen: target !== null, pinned, togglePin }}
    >
      {children}
    </CitationModalContext.Provider>
  );
};

/** Fetches chunk metadata and dispatches to the PDF or text modal. The
 *  fetch happens here (not inside the modals) so the dispatch decision
 *  is made before either heavy component mounts. */
const CitationDispatcher: React.FC<{ target: CitationTarget; onClose: () => void }> = ({
  target,
  onClose,
}) => {
  const { pinned, togglePin } = useCitationModal();
  const [chunk, setChunk] = useState<CitationChunkDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Distinct from a transient error: the chunk/document no longer exists
  // (deleted after the message was written) → a dead-end, so we show a
  // dedicated "unavailable" panel rather than an error with a link that 404s.
  const [gone, setGone] = useState(false);

  useEffect(() => {
    setChunk(null);
    setError(null);
    setGone(false);
    let cancelled = false;
    const apiPattern = window.apiUrls?.api_chunk_detail;
    if (!apiPattern) {
      setError('Chunk detail API not configured.');
      return;
    }
    const url = formatUrl(apiPattern, { chunk_id: target.chunkId });
    fetch(url, { credentials: 'include' })
      .then((r) => {
        if (r.status === 404) {
          if (!cancelled) setGone(true);
          return null;
        }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: CitationChunkDetail | null) => {
        if (!cancelled && data) setChunk(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message || 'Failed to load chunk.');
      });
    return () => {
      cancelled = true;
    };
  }, [target.chunkId]);

  if (gone) {
    return (
      <Suspense fallback={null}>
        <CitationUnavailable onClose={onClose} pinned={pinned} onTogglePin={togglePin} />
      </Suspense>
    );
  }

  if (error) {
    // Hand off to the text modal which renders a friendly error state.
    return (
      <Suspense fallback={null}>
        <TextCitationModal
          docId={target.docId}
          chunkId={target.chunkId}
          messageUuid={target.messageUuid}
          preloadedChunk={null}
          onClose={onClose}
          pinned={pinned}
          onTogglePin={togglePin}
        />
      </Suspense>
    );
  }

  if (!chunk) {
    return (
      <div className="bg-scheme-shade_3 h-full flex items-center justify-center text-text-low_contrast text-sm">
        Loading citation…
      </div>
    );
  }

  const Modal =
    chunk.modality === 'image'
      ? ImageCitationModal
      : chunk.document.has_pdf
        ? PDFCitationModal
        : TextCitationModal;
  return (
    <Suspense fallback={null}>
      <Modal
        docId={target.docId}
        chunkId={target.chunkId}
        messageUuid={target.messageUuid}
        preloadedChunk={chunk}
        onClose={onClose}
        pinned={pinned}
        onTogglePin={togglePin}
      />
    </Suspense>
  );
};

/**
 * Slide-out panel slot. Rendered as a sibling of the chat in ChatShell.
 * The outer container animates `width: 0 → CITATION_PANEL_WIDTH` with a
 * CSS transition; the inner content keeps its fixed width so the PDF
 * viewer doesn't reflow during the slide.
 */
export const CitationPanelSlot: React.FC = () => {
  const { target, closeCitation } = useCitationModal();
  const isOpen = target !== null;
  return (
    <div
      aria-hidden={!isOpen}
      className="h-full flex-shrink-0 overflow-hidden transition-[width] duration-300 ease-out border-l border-border-mid_contrast"
      style={{ width: isOpen ? CITATION_PANEL_WIDTH : 0 }}
    >
      <div className="h-full" style={{ width: CITATION_PANEL_WIDTH }}>
        {target && <CitationDispatcher target={target} onClose={closeCitation} />}
      </div>
    </div>
  );
};

export default CitationModalProvider;
