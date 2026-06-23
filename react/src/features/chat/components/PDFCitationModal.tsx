import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Document, Page } from 'react-pdf';
import type { PDFDocumentProxy, PageViewport } from 'pdfjs-dist';
import { X, ExternalLink } from 'lucide-react';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import formatUrl from '../../../utils/formatUrl';
import { configurePdfWorker } from '../../../utils/pdfWorker';
import CitationPinButton from './CitationPinButton';
import {
  locateAcrossDocument,
  type PageHighlightRange,
} from '../../../utils/pdfTextMatch';
import { getCsrfCookie } from '../../../main';
import type { CitationChunkDetail } from './citationTypes';

configurePdfWorker();

type ChunkDetail = CitationChunkDetail;

interface PDFCitationModalProps {
  docId: string;
  chunkId: string;
  /** Assistant message UUID — enables LLM-narrowed highlight. */
  messageUuid?: string;
  /** Optional chunk prefetched by the provider, skipping the initial fetch. */
  preloadedChunk?: CitationChunkDetail | null;
  onClose: () => void;
  /** Pin state for the slide-out panel (keeps it open, ignores Escape). */
  pinned?: boolean;
  onTogglePin?: () => void;
}

/** Each PDF text-layer item we need to compute a highlight rect. */
interface PdfTextItem {
  str: string;
  /** PDF.js 2D affine matrix: [a, b, c, d, e, f]. (e, f) is the baseline. */
  transform: number[];
  /** Item width in PDF user units (pre-scale). */
  width: number;
}

interface PageScan {
  pageNumber: number;
  items: PdfTextItem[];
  /** Viewport at the rendered scale — used to convert PDF coords to pixels. */
  viewport: PageViewport;
}

interface HighlightRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

const DATA_HIGHLIGHT_ATTR = 'data-citation-hit';

export const PDFCitationModal: React.FC<PDFCitationModalProps> = ({
  docId,
  chunkId,
  messageUuid,
  preloadedChunk,
  onClose,
  pinned = false,
  onTogglePin,
}) => {
  const [chunk, setChunk] = useState<ChunkDetail | null>(preloadedChunk ?? null);
  const [chunkError, setChunkError] = useState<string | null>(null);

  // LLM-narrowed quote. When set, replaces chunk.content as the locator input.
  const [narrowQuote, setNarrowQuote] = useState<string | null>(null);
  const [narrowState, setNarrowState] = useState<'idle' | 'pending' | 'tightened' | 'failed'>(
    'idle',
  );

  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [numPages, setNumPages] = useState(0);
  const [pageHighlights, setPageHighlights] = useState<Map<number, PageHighlightRange>>(
    new Map(),
  );
  const [startPage, setStartPage] = useState<number | null>(null);
  const [searchState, setSearchState] = useState<'idle' | 'searching' | 'found' | 'notfound'>(
    'idle',
  );

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const hasScrolledRef = useRef(false);
  const scannedPagesRef = useRef<PageScan[] | null>(null);

  // Escape key closes the modal, unless the panel is pinned.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !pinned) onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, pinned]);

  // Fetch chunk metadata (content + document title + PDF availability).
  // Skipped when the provider has already fetched it.
  useEffect(() => {
    if (preloadedChunk) return;
    let cancelled = false;
    const apiPattern = window.apiUrls?.api_chunk_detail;
    if (!apiPattern) {
      setChunkError('Chunk detail API not configured.');
      return;
    }
    const url = formatUrl(apiPattern, { chunk_id: chunkId });
    fetch(url, { credentials: 'include' })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: ChunkDetail) => {
        if (!cancelled) setChunk(data);
      })
      .catch((err) => {
        if (!cancelled) setChunkError(err.message || 'Failed to load chunk.');
      });
    return () => {
      cancelled = true;
    };
  }, [chunkId, preloadedChunk]);

  // In parallel with the chunk fetch, ask the backend to LLM-narrow the
  // citation to the most relevant verbatim quote.
  useEffect(() => {
    if (!messageUuid) {
      setNarrowState('idle');
      return;
    }
    const apiUrl = window.apiUrls?.api_citation_narrow;
    if (!apiUrl) {
      setNarrowState('idle');
      return;
    }
    let cancelled = false;
    setNarrowState('pending');
    fetch(apiUrl, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCsrfCookie(),
      },
      body: JSON.stringify({ message_uuid: messageUuid, chunk_id: Number(chunkId) }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled) return;
        const quote =
          data && typeof data.quote === 'string' ? data.quote.trim() : '';
        if (quote) {
          setNarrowQuote(quote);
          hasScrolledRef.current = false;
          setNarrowState('tightened');
        } else {
          setNarrowState('failed');
        }
      })
      .catch(() => {
        if (!cancelled) setNarrowState('failed');
      });
    return () => {
      cancelled = true;
    };
  }, [messageUuid, chunkId]);

  const pdfUrl = useMemo(() => {
    if (!window.pageUrls?.pdf) return null;
    return formatUrl(window.pageUrls.pdf, { doc_id: docId });
  }, [docId]);

  const fallbackDocUrl = useMemo(() => {
    if (!window.pageUrls?.document) return null;
    return `${formatUrl(window.pageUrls.document, { doc_id: docId })}?chunk=${chunkId}`;
  }, [docId, chunkId]);

  const onDocumentLoad = useCallback((pdf: PDFDocumentProxy) => {
    setPdfDoc(pdf);
    setNumPages(pdf.numPages);
  }, []);

  // The panel is 640px wide; subtract scroll padding (p-4 = 16px each
  // side) and a margin for the scrollbar so pages fit without horizontal
  // overflow.
  const pageWidth = 600;

  // Scan all pages once, then run the locator. Cached in scannedPagesRef
  // so the re-locate (after LLM narrow) doesn't re-fetch PDF text.
  useEffect(() => {
    if (!pdfDoc || !chunk || !chunk.content) return;
    const queryText = narrowQuote ?? chunk.content;
    let cancelled = false;
    setSearchState('searching');

    (async () => {
      if (!scannedPagesRef.current) {
        const pages: PageScan[] = [];
        for (let p = 1; p <= pdfDoc.numPages; p++) {
          const page = await pdfDoc.getPage(p);
          // Compute the viewport at the same scale as the rendered canvas
          // (react-pdf derives scale from the `width` prop in the same way).
          const unscaled = page.getViewport({ scale: 1 });
          const scale = pageWidth / unscaled.width;
          const viewport = page.getViewport({ scale });
          const tc = await page.getTextContent();
          const items: PdfTextItem[] = [];
          for (const it of tc.items) {
            if (
              'str' in it &&
              typeof it.str === 'string' &&
              'transform' in it &&
              Array.isArray(it.transform)
            ) {
              items.push({
                str: it.str,
                transform: it.transform,
                width: typeof it.width === 'number' ? it.width : 0,
              });
            }
          }
          pages.push({ pageNumber: p, items, viewport });
          if (cancelled) return;
        }
        scannedPagesRef.current = pages;
      }
      // The locator only needs the strings.
      const pagesForLocator = scannedPagesRef.current.map((s) => ({
        pageNumber: s.pageNumber,
        items: s.items,
      }));
      const match = locateAcrossDocument(pagesForLocator, queryText);
      if (cancelled) return;
      if (match) {
        setPageHighlights(match.pageHighlights);
        setStartPage(match.startPage);
        setSearchState('found');
      } else if (narrowQuote) {
        const fullMatch = locateAcrossDocument(pagesForLocator, chunk.content);
        if (fullMatch) {
          setPageHighlights(fullMatch.pageHighlights);
          setStartPage(fullMatch.startPage);
          setSearchState('found');
        } else {
          setSearchState('notfound');
        }
      } else {
        setSearchState('notfound');
      }
    })().catch(() => {
      if (!cancelled) setSearchState('notfound');
    });

    return () => {
      cancelled = true;
    };
  }, [pdfDoc, chunk, narrowQuote, pageWidth]);

  // Compute canvas-pixel rectangles for each highlighted item, using the
  // item's PDF transform matrix and the page viewport. This bypasses the
  // text-layer (whose span positions are based on fallback web-font
  // metrics that drift a few px from where canvas glyphs are actually
  // drawn).
  const pageRects = useMemo(() => {
    const out = new Map<number, HighlightRect[]>();
    if (!scannedPagesRef.current) return out;
    for (const scan of scannedPagesRef.current) {
      const range = pageHighlights.get(scan.pageNumber);
      if (!range) continue;
      const rects: HighlightRect[] = [];
      for (let i = range.firstItem; i <= range.lastItem; i++) {
        const item = scan.items[i];
        if (!item || !item.transform || item.transform.length < 6) continue;
        const tx = item.transform;
        // tx = [a, b, c, d, e, f]; (e, f) is the BASELINE of the text in
        // PDF user space (origin bottom-left of the page).
        // fontHeight in user space = sqrt(b^2 + d^2). For non-rotated
        // text this collapses to |d|.
        const fontHeight = Math.hypot(tx[1], tx[3]);
        const widthUser = item.width;
        const [baselineX, baselineY] = scan.viewport.convertToViewportPoint(tx[4], tx[5]);
        const widthPx = widthUser * scan.viewport.scale;
        const heightPx = fontHeight * scan.viewport.scale;
        // Canvas Y axis points down; the baseline in viewport coords is
        // already converted, so the top of the glyph row is baseline - height.
        rects.push({
          left: baselineX,
          top: baselineY - heightPx,
          width: widthPx,
          height: heightPx,
        });
      }
      out.set(scan.pageNumber, rects);
    }
    return out;
  }, [pageHighlights]);

  // After search finds a match, poll for the first highlight rect to
  // mount in the DOM and scroll it into view.
  useEffect(() => {
    if (searchState !== 'found') return;
    if (hasScrolledRef.current) return;
    const container = scrollContainerRef.current;
    if (!container) return;
    let attempts = 0;
    let timer: number | null = null;
    const maxAttempts = 50; // ~5s at 100ms
    const tick = () => {
      attempts += 1;
      const first = container.querySelector(`[${DATA_HIGHLIGHT_ATTR}]`);
      if (first) {
        (first as HTMLElement).scrollIntoView({ block: 'center', behavior: 'smooth' });
        hasScrolledRef.current = true;
        return;
      }
      if (attempts < maxAttempts) {
        timer = window.setTimeout(tick, 100);
      }
    };
    timer = window.setTimeout(tick, 100);
    return () => {
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [searchState, pageRects]);

  const renderBody = () => {
    if (chunkError) {
      return (
        <div className="p-6 text-text-normal">
          <p className="font-semibold mb-2">Couldn't load citation.</p>
          <p className="text-sm text-text-low_contrast">{chunkError}</p>
          {fallbackDocUrl && (
            <a
              href={fallbackDocUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 mt-3 text-accent hover:underline text-sm"
            >
              Open document page <ExternalLink className="w-3.5 h-3.5" />
            </a>
          )}
        </div>
      );
    }

    if (!chunk) {
      return (
        <div className="p-6 text-text-low_contrast text-sm">Loading citation…</div>
      );
    }

    if (!chunk.document.has_pdf || !pdfUrl) {
      return (
        <div className="p-6 text-text-normal">
          <p className="font-semibold mb-2">No PDF available for this document.</p>
          <p className="text-sm text-text-low_contrast mb-3 whitespace-pre-wrap">
            {chunk.content}
          </p>
          {fallbackDocUrl && (
            <a
              href={fallbackDocUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-accent hover:underline text-sm"
            >
              Open document page <ExternalLink className="w-3.5 h-3.5" />
            </a>
          )}
        </div>
      );
    }

    return (
      <div className="flex-1 min-h-0 flex flex-col">
        {searchState === 'notfound' && (
          <div className="px-4 py-2 bg-scheme-shade_4 border-b border-border-mid_contrast text-xs text-text-low_contrast">
            Couldn't locate the cited passage in the PDF text layer.
          </div>
        )}
        <div
          ref={scrollContainerRef}
          className="flex-1 min-h-0 overflow-auto bg-scheme-shade_5 flex flex-col items-center p-4 gap-4"
        >
          <Document
            file={pdfUrl}
            onLoadSuccess={onDocumentLoad}
            loading={<div className="text-text-low_contrast text-sm py-8">Loading PDF…</div>}
            error={
              <div className="text-text-normal p-4">
                <p className="font-semibold">Failed to load PDF.</p>
                {fallbackDocUrl && (
                  <a
                    href={fallbackDocUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 mt-2 text-accent hover:underline text-sm"
                  >
                    Open document page <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                )}
              </div>
            }
          >
            {Array.from({ length: numPages }, (_, i) => i + 1).map((pn) => {
              const rects = pageRects.get(pn);
              return (
                <div key={pn} className="relative shadow-md">
                  <Page
                    pageNumber={pn}
                    renderAnnotationLayer={false}
                    width={pageWidth}
                  />
                  {rects && rects.length > 0 && (
                    <div
                      className="absolute inset-0 pointer-events-none"
                      style={{ mixBlendMode: 'multiply' }}
                    >
                      {rects.map((r, i) => (
                        <div
                          key={i}
                          {...{ [DATA_HIGHLIGHT_ATTR]: '' }}
                          style={{
                            position: 'absolute',
                            left: r.left,
                            top: r.top,
                            width: r.width,
                            height: r.height,
                            backgroundColor: 'rgba(253, 224, 71, 0.55)',
                            borderRadius: '2px',
                          }}
                        />
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </Document>
        </div>
      </div>
    );
  };

  const headerStatus = useMemo(() => {
    if (!chunk) return '';
    if (searchState === 'searching') return ' · searching…';
    if (searchState === 'notfound') return ' · passage not located';
    if (searchState === 'found' && startPage) {
      const pageList = Array.from(pageHighlights.keys()).sort((a, b) => a - b);
      if (pageList.length === 0) return '';
      const pageRange =
        pageList.length === 1
          ? `page ${pageList[0]}`
          : `pages ${pageList[0]}–${pageList[pageList.length - 1]}`;
      const narrowSuffix =
        narrowState === 'pending'
          ? ' · narrowing…'
          : narrowState === 'tightened'
            ? ' · tightened'
            : '';
      return ` · highlighted on ${pageRange}${narrowSuffix}`;
    }
    return '';
  }, [chunk, searchState, startPage, pageHighlights, narrowState]);

  // Slide-out panel: rendered inside CitationPanelSlot, sized by its
  // parent (CITATION_PANEL_WIDTH). No backdrop / fixed positioning.
  return (
    <div className="bg-scheme-shade_3 h-full flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-mid_contrast">
        <div className="min-w-0">
          <div className="text-text-normal font-semibold truncate">
            {chunk?.document.title || 'PDF citation'}
          </div>
          {chunk && (
            <div className="text-xs text-text-low_contrast">
              Chunk {chunk.chunk_number}
              {headerStatus}
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
          {fallbackDocUrl && (
            <a
              href={fallbackDocUrl}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Open document page in new tab"
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

export default PDFCitationModal;
