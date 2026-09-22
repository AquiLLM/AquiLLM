import React, { createContext, lazy, Suspense, useCallback, useContext, useEffect, useState } from 'react';
import formatUrl from '../../../utils/formatUrl';
import type { CitationChunkDetail, CitationChunkSummary } from './citationTypes';

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
const CITATION_OPEN_START_MARK = 'aquillm:citation:open-start';
const CITATION_COMPACT_READY_MARK = 'aquillm:citation:compact-ready';
const CITATION_COMPACT_MEASURE = 'aquillm:citation:compact-metadata';

function markCitationOpenStart(): void {
  if (!import.meta.env.DEV || typeof performance === 'undefined') return;
  try {
    if (typeof performance.clearMarks === 'function') {
      performance.clearMarks(CITATION_OPEN_START_MARK);
      performance.clearMarks(CITATION_COMPACT_READY_MARK);
    }
    if (typeof performance.clearMeasures === 'function') {
      performance.clearMeasures(CITATION_COMPACT_MEASURE);
    }
    if (typeof performance.mark === 'function') {
      performance.mark(CITATION_OPEN_START_MARK);
    }
  } catch {
    // User Timing is diagnostic-only and must never block opening a citation.
  }
}

function markCitationCompactReady(): void {
  if (!import.meta.env.DEV || typeof performance === 'undefined') return;
  try {
    if (typeof performance.mark !== 'function') return;
    performance.mark(CITATION_COMPACT_READY_MARK);
    if (typeof performance.measure === 'function') {
      performance.measure(
        CITATION_COMPACT_MEASURE,
        CITATION_OPEN_START_MARK,
        CITATION_COMPACT_READY_MARK,
      );
    }
  } catch {
    // Some browsers throw when marks are missing or User Timing is restricted.
  }
}

function compactChunkDetailUrl(url: string): string {
  const parsed = new URL(url, window.location.href);
  parsed.searchParams.set('include_full_text', '0');
  if (/^[a-z][a-z\d+.-]*:/i.test(url) || url.startsWith('//')) {
    return parsed.toString();
  }
  return `${parsed.pathname}${parsed.search}${parsed.hash}`;
}

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

  const openCitation = useCallback((next: CitationTarget) => {
    markCitationOpenStart();
    setTarget(next);
  }, []);
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
  const [summary, setSummary] = useState<CitationChunkSummary | null>(null);
  const [detail, setDetail] = useState<CitationChunkDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Distinct from a transient error: the chunk/document no longer exists
  // (deleted after the message was written) → a dead-end, so we show a
  // dedicated "unavailable" panel rather than an error with a link that 404s.
  const [gone, setGone] = useState(false);

  useEffect(() => {
    setSummary(null);
    setDetail(null);
    setError(null);
    setGone(false);
    const apiPattern = window.apiUrls?.api_chunk_detail;
    if (!apiPattern) {
      setError('Chunk detail API not configured.');
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    const detailUrl = formatUrl(apiPattern, { chunk_id: target.chunkId });

    const loadCitation = async () => {
      try {
        const compactResponse = await fetch(compactChunkDetailUrl(detailUrl), {
          credentials: 'include',
          signal: controller.signal,
        });
        if (compactResponse.status === 404) {
          if (!cancelled) setGone(true);
          return;
        }
        if (!compactResponse.ok) throw new Error(`HTTP ${compactResponse.status}`);
        const compactChunk = (await compactResponse.json()) as CitationChunkSummary;
        if (cancelled) return;
        markCitationCompactReady();
        setSummary(compactChunk);

        if (compactChunk.modality === 'image' || compactChunk.document.has_pdf) return;

        const detailResponse = await fetch(detailUrl, {
          credentials: 'include',
          signal: controller.signal,
        });
        if (detailResponse.status === 404) {
          if (!cancelled) setGone(true);
          return;
        }
        if (!detailResponse.ok) throw new Error(`HTTP ${detailResponse.status}`);
        const fullChunk = (await detailResponse.json()) as CitationChunkDetail;
        if (!cancelled) setDetail(fullChunk);
      } catch (err) {
        if (cancelled || (err instanceof DOMException && err.name === 'AbortError')) return;
        setError(err instanceof Error && err.message ? err.message : 'Failed to load chunk.');
      }
    };

    void loadCitation();
    return () => {
      cancelled = true;
      controller.abort();
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

  if (summary?.modality === 'image') {
    return (
      <Suspense fallback={null}>
        <ImageCitationModal
          docId={target.docId}
          chunkId={target.chunkId}
          messageUuid={target.messageUuid}
          preloadedChunk={summary}
          onClose={onClose}
          pinned={pinned}
          onTogglePin={togglePin}
        />
      </Suspense>
    );
  }

  if (summary?.document.has_pdf) {
    return (
      <Suspense fallback={null}>
        <PDFCitationModal
          docId={target.docId}
          chunkId={target.chunkId}
          messageUuid={target.messageUuid}
          preloadedChunk={summary}
          onClose={onClose}
          pinned={pinned}
          onTogglePin={togglePin}
        />
      </Suspense>
    );
  }

  if (detail) {
    return (
      <Suspense fallback={null}>
        <TextCitationModal
          docId={target.docId}
          chunkId={target.chunkId}
          messageUuid={target.messageUuid}
          preloadedChunk={detail}
          onClose={onClose}
          pinned={pinned}
          onTogglePin={togglePin}
        />
      </Suspense>
    );
  }

  return (
    <div className="bg-scheme-shade_3 h-full flex items-center justify-center text-text-low_contrast text-sm">
      Loading citation…
    </div>
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
          <CitationDispatcher
            key={JSON.stringify([target.docId, target.chunkId, target.messageUuid ?? null])}
            target={target}
            onClose={closeCitation}
          />
        )}
      </div>
    </div>
  );
};

export default CitationModalProvider;
