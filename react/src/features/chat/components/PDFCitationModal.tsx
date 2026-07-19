import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Document } from 'react-pdf';
import type { PDFDocumentProxy } from 'pdfjs-dist';
import { ExternalLink, X } from 'lucide-react';
import 'react-pdf/dist/Page/AnnotationLayer.css';

import formatUrl from '../../../utils/formatUrl';
import { configurePdfWorker } from '../../../utils/pdfWorker';
import type {
  DocumentMatchResult,
  PageHighlightRange,
} from '../../../utils/pdfTextMatch';
import { getCsrfCookie } from '../../../main';
import CitationPinButton from './CitationPinButton';
import type { CitationChunkSummary } from './citationTypes';
import {
  createPdfPageExtractionSession,
  type PageScan,
  type PdfPageExtractionSession,
} from './pdfPageExtraction';
import VirtualizedPdfPages from './VirtualizedPdfPages';

configurePdfWorker();

interface PDFCitationModalProps {
  docId: string;
  chunkId: string;
  /** Assistant message UUID — enables LLM-narrowed highlight. */
  messageUuid?: string;
  /** Optional chunk prefetched by the provider, skipping the initial fetch. */
  preloadedChunk?: CitationChunkSummary | null;
  onClose: () => void;
  /** Pin state for the slide-out panel (keeps it open, ignores Escape). */
  pinned?: boolean;
  onTogglePin?: () => void;
}

interface HighlightRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

type SearchState = 'idle' | 'searching' | 'found' | 'notfound';
type QueryStatus = 'idle' | 'pending' | 'done';

interface MatchQueryState {
  status: QueryStatus;
  match: DocumentMatchResult | null;
}

const IDLE_QUERY: MatchQueryState = { status: 'idle', match: null };
const DATA_HIGHLIGHT_ATTR = 'data-citation-hit';
const PAGE_WIDTH = 600;

const PERF_OPEN_START = 'aquillm:citation:open-start';
const PERF_PDF_READY = 'aquillm:citation:pdf-ready';
const PERF_FIRST_MATCH = 'aquillm:citation:first-match';
const PERF_CANVAS_MOUNTED = 'aquillm:citation:canvas-mounted';
const PERF_HIGHLIGHT_SCROLLED = 'aquillm:citation:highlight-scrolled';

function markAndMeasure(markName: string, measureName: string): void {
  if (!import.meta.env.DEV || typeof performance === 'undefined') return;
  try {
    if (typeof performance.mark !== 'function') return;
    performance.mark(markName);
    if (typeof performance.measure === 'function') {
      performance.measure(measureName, PERF_OPEN_START, markName);
    }
  } catch {
    // User Timing is diagnostic-only and can be unavailable or restricted.
  }
}

function stableMatchKey(match: DocumentMatchResult | null): string {
  if (!match) return '';
  const ranges = Array.from(match.pageHighlights.entries())
    .sort(([left], [right]) => left - right)
    .map(([page, range]) => `${page}:${range.firstItem}-${range.lastItem}`)
    .join('|');
  return `${match.startPage}|${ranges}`;
}

export const PDFCitationModal: React.FC<PDFCitationModalProps> = ({
  docId,
  chunkId,
  messageUuid,
  preloadedChunk,
  onClose,
  pinned = false,
  onTogglePin,
}) => {
  const pdfPattern = window.pageUrls?.pdf;
  const pdfUrl = useMemo(() => {
    if (!pdfPattern) return null;
    return formatUrl(pdfPattern, { doc_id: docId });
  }, [docId, pdfPattern]);
  const identity = `${docId}\u0000${pdfUrl ?? ''}`;

  const identityRef = useRef(identity);
  const generationRef = useRef(0);
  if (identityRef.current !== identity) {
    identityRef.current = identity;
    generationRef.current += 1;
  }
  const generation = generationRef.current;

  const [chunk, setChunk] = useState<CitationChunkSummary | null>(preloadedChunk ?? null);
  const [chunkError, setChunkError] = useState<string | null>(null);
  const [narrowQuote, setNarrowQuote] = useState<string | null>(null);
  const [narrowState, setNarrowState] = useState<'idle' | 'pending' | 'tightened' | 'failed'>(
    'idle',
  );
  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [numPages, setNumPages] = useState(0);
  const [pageScans, setPageScans] = useState<Map<number, PageScan>>(new Map());
  const [fullQuery, setFullQuery] = useState<MatchQueryState>(IDLE_QUERY);
  const [narrowQuery, setNarrowQuery] = useState<MatchQueryState>(IDLE_QUERY);
  const [matchIdentity, setMatchIdentity] = useState(identity);
  const [pageHighlights, setPageHighlights] = useState<Map<number, PageHighlightRange>>(
    new Map(),
  );
  const [startPage, setStartPage] = useState<number | null>(null);
  const [initialJumpComplete, setInitialJumpComplete] = useState(false);
  const [canvasReadyTargetKey, setCanvasReadyTargetKey] = useState('');

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const sessionRef = useRef<PdfPageExtractionSession | null>(null);
  const chunkAbortRef = useRef<AbortController | null>(null);
  const narrowHttpAbortRef = useRef<AbortController | null>(null);
  const fullAbortRef = useRef<AbortController | null>(null);
  const narrowFindAbortRef = useRef<AbortController | null>(null);
  const scrolledMatchRef = useRef('');
  const firstMatchIdentityRef = useRef('');
  const pdfReadyIdentityRef = useRef('');
  const measuredCanvasRef = useRef('');
  const readyCanvasPagesRef = useRef(new Set<string>());

  // A document identity change invalidates every publication synchronously before paint.
  useLayoutEffect(() => {
    chunkAbortRef.current?.abort();
    narrowHttpAbortRef.current?.abort();
    fullAbortRef.current?.abort();
    narrowFindAbortRef.current?.abort();
    chunkAbortRef.current = null;
    narrowHttpAbortRef.current = null;
    fullAbortRef.current = null;
    narrowFindAbortRef.current = null;
    setChunk(preloadedChunk ?? null);
    setChunkError(null);
    setNarrowQuote(null);
    setNarrowState('idle');
    setPdfDoc(null);
    setNumPages(0);
    setPageScans(new Map());
    setFullQuery(IDLE_QUERY);
    setNarrowQuery(IDLE_QUERY);
    setMatchIdentity(identity);
    setPageHighlights(new Map());
    setStartPage(null);
    setInitialJumpComplete(false);
    setCanvasReadyTargetKey('');
    scrolledMatchRef.current = '';
    firstMatchIdentityRef.current = '';
    pdfReadyIdentityRef.current = '';
    measuredCanvasRef.current = '';
    readyCanvasPagesRef.current.clear();
  }, [identity]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !pinned) onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, pinned]);

  // Fetch chunk metadata when the provider did not preload it.
  useEffect(() => {
    const effectIdentity = identity;
    const effectGeneration = generationRef.current;
    if (preloadedChunk) {
      setChunk(preloadedChunk);
      setChunkError(null);
      return;
    }

    setChunk(null);
    setChunkError(null);
    const apiPattern = window.apiUrls?.api_chunk_detail;
    if (!apiPattern) {
      setChunkError('Chunk detail API not configured.');
      return;
    }

    const controller = new AbortController();
    chunkAbortRef.current = controller;
    const url = formatUrl(apiPattern, { chunk_id: chunkId });
    void fetch(url, { credentials: 'include', signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((data: CitationChunkSummary) => {
        if (
          !controller.signal.aborted
          && identityRef.current === effectIdentity
          && generationRef.current === effectGeneration
        ) {
          setChunk(data);
        }
      })
      .catch((error: unknown) => {
        if (
          !controller.signal.aborted
          && identityRef.current === effectIdentity
          && generationRef.current === effectGeneration
        ) {
          setChunkError(error instanceof Error ? error.message : 'Failed to load chunk.');
        }
      });

    return () => controller.abort();
  }, [chunkId, identity, preloadedChunk]);

  // Narrowing starts independently of PDF loading and full-text matching.
  useEffect(() => {
    const effectIdentity = identity;
    const effectGeneration = generationRef.current;
    setNarrowQuote(null);
    setNarrowQuery(IDLE_QUERY);
    if (!messageUuid) {
      setNarrowState('idle');
      return;
    }
    const apiUrl = window.apiUrls?.api_citation_narrow;
    if (!apiUrl) {
      setNarrowState('idle');
      return;
    }

    const controller = new AbortController();
    narrowHttpAbortRef.current = controller;
    setNarrowState('pending');
    void fetch(apiUrl, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCsrfCookie(),
      },
      body: JSON.stringify({ message_uuid: messageUuid, chunk_id: Number(chunkId) }),
      signal: controller.signal,
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((data: unknown) => {
        if (
          controller.signal.aborted
          || identityRef.current !== effectIdentity
          || generationRef.current !== effectGeneration
        ) {
          return;
        }
        const quote = data
          && typeof data === 'object'
          && 'quote' in data
          && typeof data.quote === 'string'
          ? data.quote.trim()
          : '';
        if (quote) {
          setNarrowQuote(quote);
          setNarrowState('tightened');
        } else {
          setNarrowState('failed');
        }
      })
      .catch(() => {
        if (
          !controller.signal.aborted
          && identityRef.current === effectIdentity
          && generationRef.current === effectGeneration
        ) {
          setNarrowState('failed');
        }
      });

    return () => controller.abort();
  }, [chunkId, identity, messageUuid]);

  const fallbackDocUrl = useMemo(() => {
    if (!window.pageUrls?.document) return null;
    return `${formatUrl(window.pageUrls.document, { doc_id: docId })}?chunk=${chunkId}`;
  }, [docId, chunkId]);

  const onDocumentLoad = useCallback(
    (pdf: PDFDocumentProxy) => {
      if (identityRef.current !== identity || generationRef.current !== generation) return;
      setPdfDoc(pdf);
      setNumPages(pdf.numPages);
      if (pdfReadyIdentityRef.current !== identity) {
        pdfReadyIdentityRef.current = identity;
        markAndMeasure(PERF_PDF_READY, 'aquillm:citation:pdf-load');
      }
    },
    [generation, identity],
  );

  const session = useMemo(
    () => (pdfDoc ? createPdfPageExtractionSession(pdfDoc, PAGE_WIDTH) : null),
    [pdfDoc],
  );
  sessionRef.current = session;

  // Observe immutable PageScan snapshots from the shared extraction cache.
  useEffect(() => {
    setPageScans(new Map());
    if (!session) return;
    const effectIdentity = identity;
    const effectGeneration = generationRef.current;
    let active = true;
    const publish = (scan: PageScan) => {
      if (
        !active
        || identityRef.current !== effectIdentity
        || generationRef.current !== effectGeneration
        || sessionRef.current !== session
      ) {
        return;
      }
      setPageScans((current) => {
        if (current.get(scan.pageNumber) === scan) return current;
        const next = new Map(current);
        next.set(scan.pageNumber, scan);
        return next;
      });
    };
    const seed = () => {
      for (let page = 1; page <= numPages; page += 1) {
        const cached = session.getCached(page);
        if (cached) publish(cached);
      }
    };

    seed();
    const unsubscribe = session.subscribe(publish);
    seed();
    return () => {
      active = false;
      unsubscribe();
    };
  }, [identity, numPages, session]);

  useEffect(() => {
    const query = chunk?.content.trim() ?? '';
    if (!session || !query) {
      setFullQuery(IDLE_QUERY);
      return;
    }
    const effectIdentity = identity;
    const effectGeneration = generationRef.current;
    const controller = new AbortController();
    fullAbortRef.current?.abort();
    fullAbortRef.current = controller;
    setFullQuery({ status: 'pending', match: null });
    void session.find(query, controller.signal)
      .then((match) => {
        if (
          !controller.signal.aborted
          && identityRef.current === effectIdentity
          && generationRef.current === effectGeneration
          && sessionRef.current === session
        ) {
          setFullQuery({ status: 'done', match });
        }
      })
      .catch(() => {
        if (
          !controller.signal.aborted
          && identityRef.current === effectIdentity
          && generationRef.current === effectGeneration
          && sessionRef.current === session
        ) {
          setFullQuery({ status: 'done', match: null });
        }
      });
    return () => controller.abort();
  }, [chunk?.content, identity, session]);

  useEffect(() => {
    const query = narrowQuote?.trim() ?? '';
    if (!session || !query) {
      setNarrowQuery(IDLE_QUERY);
      return;
    }
    const effectIdentity = identity;
    const effectGeneration = generationRef.current;
    const controller = new AbortController();
    narrowFindAbortRef.current?.abort();
    narrowFindAbortRef.current = controller;
    setNarrowQuery({ status: 'pending', match: null });
    void session.find(query, controller.signal)
      .then((match) => {
        if (
          !controller.signal.aborted
          && identityRef.current === effectIdentity
          && generationRef.current === effectGeneration
          && sessionRef.current === session
        ) {
          setNarrowQuery({ status: 'done', match });
        }
      })
      .catch(() => {
        if (
          !controller.signal.aborted
          && identityRef.current === effectIdentity
          && generationRef.current === effectGeneration
          && sessionRef.current === session
        ) {
          setNarrowQuery({ status: 'done', match: null });
        }
      });
    return () => controller.abort();
  }, [identity, narrowQuote, session]);

  // A successful narrow match always wins; completion order never mutates this preference.
  const preferredMatch = matchIdentity === identity
    ? narrowQuery.match ?? fullQuery.match
    : null;
  const preferredMatchKey = useMemo(() => stableMatchKey(preferredMatch), [preferredMatch]);

  useLayoutEffect(() => {
    if (!preferredMatch || !preferredMatchKey) {
      setPageHighlights(new Map());
      setStartPage(null);
      setInitialJumpComplete(false);
      setCanvasReadyTargetKey('');
      scrolledMatchRef.current = '';
      return;
    }
    setPageHighlights(new Map(preferredMatch.pageHighlights));
    setStartPage(preferredMatch.startPage);
    setInitialJumpComplete(false);
    const targetKey = `${identity}|${preferredMatchKey}`;
    const targetPlaceholder = scrollContainerRef.current?.querySelector(
      `[data-pdf-page-placeholder][data-page-number="${preferredMatch.startPage}"]`,
    );
    setCanvasReadyTargetKey(
      readyCanvasPagesRef.current.has(`${identity}|${preferredMatch.startPage}`)
        && targetPlaceholder?.querySelector('canvas')
        ? targetKey
        : '',
    );
    scrolledMatchRef.current = '';
    if (firstMatchIdentityRef.current !== identity) {
      firstMatchIdentityRef.current = identity;
      markAndMeasure(PERF_FIRST_MATCH, 'aquillm:citation:match');
    }
  }, [identity, preferredMatchKey]);

  const searchState: SearchState = useMemo(() => {
    if (preferredMatch) return 'found';
    const narrowingCanStillPublish = narrowState === 'pending'
      || narrowQuery.status === 'pending'
      || (narrowState === 'tightened' && narrowQuery.status !== 'done');
    if (fullQuery.status === 'pending' || narrowingCanStillPublish) return 'searching';
    if (
      fullQuery.status === 'done'
      && (narrowState === 'idle' || narrowState === 'failed' || narrowQuery.status === 'done')
    ) {
      return 'notfound';
    }
    return 'idle';
  }, [fullQuery.status, narrowQuery.status, narrowState, preferredMatch]);

  const pageRects = useMemo(() => {
    const rectangles = new Map<number, HighlightRect[]>();
    for (const [pageNumber, range] of pageHighlights) {
      const scan = pageScans.get(pageNumber);
      if (!scan) continue;
      const pageRectangles: HighlightRect[] = [];
      for (let itemIndex = range.firstItem; itemIndex <= range.lastItem; itemIndex += 1) {
        const item = scan.items[itemIndex];
        if (!item || !item.transform || item.transform.length < 6) continue;
        const transform = item.transform;
        const fontHeight = Math.hypot(transform[1], transform[3]);
        const [baselineX, baselineY] = scan.viewport.convertToViewportPoint(
          transform[4],
          transform[5],
        );
        const width = item.width * scan.viewport.scale;
        const height = fontHeight * scan.viewport.scale;
        pageRectangles.push({
          left: baselineX,
          top: baselineY - height,
          width,
          height,
        });
      }
      rectangles.set(pageNumber, pageRectangles);
    }
    return rectangles;
  }, [pageHighlights, pageScans]);

  const renderOverlay = useCallback(
    (pageNumber: number) => {
      const rectangles = pageRects.get(pageNumber);
      if (!rectangles || rectangles.length === 0) return null;
      return (
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ mixBlendMode: 'multiply' }}
        >
          {rectangles.map((rectangle, index) => (
            <div
              key={index}
              {...{ [DATA_HIGHLIGHT_ATTR]: '' }}
              style={{
                position: 'absolute',
                left: rectangle.left,
                top: rectangle.top,
                width: rectangle.width,
                height: rectangle.height,
                backgroundColor: 'rgba(253, 224, 71, 0.55)',
                borderRadius: '2px',
              }}
            />
          ))}
        </div>
      );
    },
    [pageRects],
  );

  const onPageRenderSuccess = useCallback(
    (pageNumber: number) => {
      if (identityRef.current !== identity) return;
      readyCanvasPagesRef.current.add(`${identity}|${pageNumber}`);
      if (
        !preferredMatchKey
        || pageNumber !== startPage
      ) {
        return;
      }
      const measurementKey = `${identity}|${preferredMatchKey}`;
      setCanvasReadyTargetKey(measurementKey);
      if (measuredCanvasRef.current === measurementKey) return;
      measuredCanvasRef.current = measurementKey;
      markAndMeasure(PERF_CANVAS_MOUNTED, 'aquillm:citation:canvas');
    },
    [identity, preferredMatchKey, startPage],
  );

  // Wait for the current target canvas and highlight marker before centering it.
  useEffect(() => {
    if (!preferredMatchKey || startPage === null) return;
    const targetKey = `${identity}|${preferredMatchKey}`;
    if (canvasReadyTargetKey !== targetKey) return;
    if (scrolledMatchRef.current === targetKey) return;
    const container = scrollContainerRef.current;
    if (!container) return;
    let attempts = 0;
    let timer: number | null = null;
    let visibilityFrame: number | null = null;
    let completionTimer: number | null = null;
    let completed = false;

    const stopCompletionWatch = () => {
      container.removeEventListener('scroll', scheduleVisibilityCheck);
      container.removeEventListener('scrollend', completeJump);
      if (visibilityFrame !== null) {
        globalThis.cancelAnimationFrame(visibilityFrame);
        visibilityFrame = null;
      }
      if (completionTimer !== null) {
        window.clearTimeout(completionTimer);
        completionTimer = null;
      }
    };
    const completeJump = () => {
      if (completed || scrolledMatchRef.current !== targetKey) return;
      completed = true;
      stopCompletionWatch();
      setInitialJumpComplete(true);
    };
    const targetReachedViewport = (targetPage: Element) => {
      const containerRect = container.getBoundingClientRect();
      const targetRect = targetPage.getBoundingClientRect();
      return targetRect.bottom > containerRect.top && targetRect.top < containerRect.bottom;
    };
    const scheduleVisibilityCheck = () => {
      if (visibilityFrame !== null || completed) return;
      visibilityFrame = globalThis.requestAnimationFrame(() => {
        visibilityFrame = null;
        const targetPage = container.querySelector(
          `[data-pdf-page-placeholder][data-page-number="${startPage}"]`,
        );
        if (targetPage && targetReachedViewport(targetPage)) completeJump();
      });
    };
    const tick = () => {
      attempts += 1;
      const marker = container.querySelector(
        `[data-pdf-page-placeholder][data-page-number="${startPage}"] [${DATA_HIGHLIGHT_ATTR}]`,
      );
      if (marker) {
        timer = null;
        (marker as HTMLElement).scrollIntoView({ block: 'center', behavior: 'smooth' });
        scrolledMatchRef.current = targetKey;
        markAndMeasure(PERF_HIGHLIGHT_SCROLLED, 'aquillm:citation:highlight');
        container.addEventListener('scroll', scheduleVisibilityCheck, { passive: true });
        container.addEventListener('scrollend', completeJump);
        completionTimer = window.setTimeout(completeJump, 1_200);
        scheduleVisibilityCheck();
        return;
      }
      if (attempts < 50) timer = window.setTimeout(tick, 100);
    };
    timer = window.setTimeout(tick, 100);
    return () => {
      if (timer !== null) window.clearTimeout(timer);
      stopCompletionWatch();
    };
  }, [canvasReadyTargetKey, identity, preferredMatchKey, startPage]);

  const headerStatus = useMemo(() => {
    if (!chunk) return '';
    if (searchState === 'searching') return ' · searching…';
    if (searchState === 'notfound') return ' · passage not located';
    if (searchState === 'found' && startPage) {
      const pages = Array.from(pageHighlights.keys()).sort((left, right) => left - right);
      if (pages.length === 0) return '';
      const range = pages.length === 1
        ? `page ${pages[0]}`
        : `pages ${pages[0]}–${pages[pages.length - 1]}`;
      const narrowSuffix = narrowState === 'pending'
        ? ' · narrowing…'
        : narrowState === 'tightened'
          ? ' · tightened'
          : '';
      return ` · highlighted on ${range}${narrowSuffix}`;
    }
    return '';
  }, [chunk, narrowState, pageHighlights, searchState, startPage]);

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
      return <div className="p-6 text-text-low_contrast text-sm">Loading citation…</div>;
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
            key={pdfUrl}
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
            {session && (
              <VirtualizedPdfPages
                numPages={numPages}
                pageWidth={PAGE_WIDTH}
                session={session}
                scrollContainerRef={scrollContainerRef}
                citationStartPage={startPage}
                initialJumpComplete={initialJumpComplete}
                renderOverlay={renderOverlay}
                onPageRenderSuccess={onPageRenderSuccess}
              />
            )}
          </Document>
        </div>
      </div>
    );
  };

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
