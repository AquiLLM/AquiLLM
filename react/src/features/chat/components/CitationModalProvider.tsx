import React, { createContext, lazy, Suspense, useCallback, useContext, useState } from 'react';

// Lazy-loaded to keep react-pdf / pdfjs-dist (which use `import.meta`) out
// of the main bundle. main.js is loaded as a classic <script>, so any
// top-level `import.meta` would be a parse error and stop React from
// mounting on every page. With Vite's IIFE + inlineDynamicImports the
// dynamic import becomes synchronous in the bundle, but the lazy wrapper
// is harmless and preserves code-splitting if we ever switch back to ESM.
const PDFCitationModal = lazy(() => import('./PDFCitationModal'));

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
}

const CitationModalContext = createContext<CitationModalContextValue | null>(null);

export function useCitationModal(): CitationModalContextValue {
  const ctx = useContext(CitationModalContext);
  if (!ctx) {
    return { openCitation: () => {}, closeCitation: () => {}, target: null, isOpen: false };
  }
  return ctx;
}

export const CitationModalProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [target, setTarget] = useState<CitationTarget | null>(null);

  const openCitation = useCallback((next: CitationTarget) => setTarget(next), []);
  const closeCitation = useCallback(() => setTarget(null), []);

  return (
    <CitationModalContext.Provider
      value={{ openCitation, closeCitation, target, isOpen: target !== null }}
    >
      {children}
    </CitationModalContext.Provider>
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
        {target && (
          <Suspense fallback={null}>
            <PDFCitationModal
              docId={target.docId}
              chunkId={target.chunkId}
              messageUuid={target.messageUuid}
              onClose={closeCitation}
            />
          </Suspense>
        )}
      </div>
    </div>
  );
};

export default CitationModalProvider;
