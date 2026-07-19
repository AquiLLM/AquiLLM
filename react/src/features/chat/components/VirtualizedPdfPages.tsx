import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from 'react';
import { Page } from 'react-pdf';

import type { PageScan, PdfPageExtractionSession } from './pdfPageExtraction';
import {
  calculateRenderedPages,
  findViewportPage,
  type PageMeasurement,
} from './pdfPageWindow';

const DEFAULT_PAGE_ASPECT_RATIO = 11 / 8.5;
const PAGE_GAP = 0;

export interface VirtualizedPdfPagesProps {
  numPages: number;
  pageWidth: number;
  session: PdfPageExtractionSession;
  scrollContainerRef: RefObject<HTMLDivElement | null>;
  citationStartPage: number | null;
  initialJumpComplete: boolean;
  renderOverlay?: (pageNumber: number) => ReactNode;
  onPageRenderSuccess?: (pageNumber: number) => void;
}

function normalizePageCount(numPages: number): number {
  return Number.isFinite(numPages) ? Math.max(0, Math.floor(numPages)) : 0;
}

function estimatedHeight(pageWidth: number): number {
  const estimate = pageWidth * DEFAULT_PAGE_ASPECT_RATIO;
  return Number.isFinite(estimate) ? Math.max(1, estimate) : 1;
}

function makeEstimatedHeights(numPages: number, pageWidth: number): number[] {
  return Array.from(
    { length: normalizePageCount(numPages) },
    () => estimatedHeight(pageWidth),
  );
}

function makePageMeasurements(pageHeights: ReadonlyArray<number>): PageMeasurement[] {
  let top = 0;
  return pageHeights.map((height, index) => {
    const measurement = { pageNumber: index + 1, top, height };
    top += height + PAGE_GAP;
    return measurement;
  });
}

interface VisiblePageState {
  session: PdfPageExtractionSession;
  pageNumber: number;
}

export function VirtualizedPdfPages({
  numPages,
  pageWidth,
  session,
  scrollContainerRef,
  citationStartPage,
  initialJumpComplete,
  renderOverlay,
  onPageRenderSuccess,
}: VirtualizedPdfPagesProps) {
  const pageCount = normalizePageCount(numPages);
  const [visiblePageState, setVisiblePageState] = useState<VisiblePageState>(() => ({
    session,
    pageNumber: 1,
  }));
  const visiblePage = visiblePageState.session === session
    ? visiblePageState.pageNumber
    : 1;
  const [pageHeights, setPageHeights] = useState(() =>
    makeEstimatedHeights(numPages, pageWidth),
  );
  const heightsRef = useRef(pageHeights);
  const measurementsRef = useRef(makePageMeasurements(pageHeights));
  const placeholderRefs = useRef(new Map<number, HTMLDivElement>());
  const listRef = useRef<HTMLDivElement>(null);
  const visibilityFrameRef = useRef<number | null>(null);
  const sessionRef = useRef(session);
  sessionRef.current = session;

  const calculateVisiblePage = useCallback(() => {
    visibilityFrameRef.current = null;
    const container = scrollContainerRef.current;
    const list = listRef.current;
    if (!container || !list) return;

    const containerRect = container.getBoundingClientRect();
    const listRect = list.getBoundingClientRect();
    const listTop = listRect.top - containerRect.top + container.scrollTop;
    const pageNumber = findViewportPage(
      measurementsRef.current,
      container.scrollTop - listTop,
      container.clientHeight,
    );
    const currentSession = sessionRef.current;
    setVisiblePageState((current) =>
      current.session === currentSession && current.pageNumber === pageNumber
        ? current
        : { session: currentSession, pageNumber },
    );
  }, [scrollContainerRef]);

  const scheduleVisibility = useCallback(() => {
    if (visibilityFrameRef.current !== null) return;
    visibilityFrameRef.current = globalThis.requestAnimationFrame(calculateVisiblePage);
  }, [calculateVisiblePage]);

  const updateHeight = useCallback(
    (scan: PageScan) => {
      const pageIndex = scan.pageNumber - 1;
      const nextHeight = scan.viewport.height;
      if (
        pageIndex < 0
        || pageIndex >= normalizePageCount(numPages)
        || !Number.isFinite(nextHeight)
        || nextHeight <= 0
      ) {
        return;
      }

      const currentHeights = heightsRef.current;
      const previousHeight = currentHeights[pageIndex] ?? estimatedHeight(pageWidth);
      if (previousHeight === nextHeight) return;

      const container = scrollContainerRef.current;
      const list = listRef.current;
      const measurement = measurementsRef.current[pageIndex];
      if (container && list && measurement) {
        const containerRect = container.getBoundingClientRect();
        const listRect = list.getBoundingClientRect();
        const listTop = listRect.top - containerRect.top + container.scrollTop;
        if (listTop + measurement.top + previousHeight <= container.scrollTop) {
          container.scrollTop += nextHeight - previousHeight;
        }
      }

      const nextHeights = currentHeights.slice();
      nextHeights[pageIndex] = nextHeight;
      heightsRef.current = nextHeights;
      measurementsRef.current = makePageMeasurements(nextHeights);
      setPageHeights(nextHeights);
    },
    [numPages, pageWidth, scrollContainerRef],
  );

  useEffect(() => {
    const estimates = makeEstimatedHeights(numPages, pageWidth);
    for (let pageNumber = 1; pageNumber <= pageCount; pageNumber += 1) {
      const cached = session.getCached(pageNumber);
      if (cached && Number.isFinite(cached.viewport.height) && cached.viewport.height > 0) {
        estimates[pageNumber - 1] = cached.viewport.height;
      }
    }
    heightsRef.current = estimates;
    measurementsRef.current = makePageMeasurements(estimates);
    setPageHeights(estimates);

    let active = true;
    const onScan = (scan: PageScan) => {
      if (active) updateHeight(scan);
    };
    const unsubscribe = session.subscribe(onScan);

    for (let pageNumber = 1; pageNumber <= pageCount; pageNumber += 1) {
      const cached = session.getCached(pageNumber);
      if (cached) onScan(cached);
    }

    return () => {
      active = false;
      unsubscribe();
    };
  }, [numPages, pageCount, pageWidth, session, updateHeight]);

  useLayoutEffect(() => {
    measurementsRef.current = makePageMeasurements(pageHeights);
    scheduleVisibility();
  }, [pageHeights, scheduleVisibility]);

  useEffect(() => {
    if (visiblePage < 1 || visiblePage > pageCount) return;
    if (session.getCached(visiblePage)) return;
    void session.ensurePage(visiblePage).catch(() => undefined);
  }, [pageCount, session, visiblePage]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    let intersectionObserver: IntersectionObserver | null = null;
    if (typeof globalThis.IntersectionObserver === 'function') {
      intersectionObserver = new globalThis.IntersectionObserver(
        scheduleVisibility,
        { root: container },
      );
      for (let pageNumber = 1; pageNumber <= pageCount; pageNumber += 1) {
        const placeholder = placeholderRefs.current.get(pageNumber);
        if (placeholder) intersectionObserver.observe(placeholder);
      }
    }

    container.addEventListener('scroll', scheduleVisibility, { passive: true });

    let resizeObserver: ResizeObserver | null = null;
    if (typeof globalThis.ResizeObserver === 'function') {
      resizeObserver = new globalThis.ResizeObserver(scheduleVisibility);
      resizeObserver.observe(container);
    } else {
      globalThis.addEventListener('resize', scheduleVisibility);
    }

    scheduleVisibility();
    return () => {
      intersectionObserver?.disconnect();
      resizeObserver?.disconnect();
      container.removeEventListener('scroll', scheduleVisibility);
      if (!resizeObserver) globalThis.removeEventListener('resize', scheduleVisibility);
      if (visibilityFrameRef.current !== null) {
        globalThis.cancelAnimationFrame(visibilityFrameRef.current);
        visibilityFrameRef.current = null;
      }
    };
  }, [pageCount, scheduleVisibility, scrollContainerRef, session]);

  const renderedPages = useMemo(
    () =>
      new Set(
        calculateRenderedPages({
          numPages: pageCount,
          visiblePage,
          citationStartPage,
          initialJumpComplete,
        }),
      ),
    [citationStartPage, initialJumpComplete, pageCount, visiblePage],
  );

  return (
    <div
      ref={listRef}
      data-pdf-page-list=""
      style={{ display: 'flex', flexDirection: 'column', gap: PAGE_GAP }}
    >
      {Array.from({ length: pageCount }, (_, index) => {
        const pageNumber = index + 1;
        const isRendered = renderedPages.has(pageNumber);
        return (
          <div
            key={pageNumber}
            ref={(element) => {
              if (element) placeholderRefs.current.set(pageNumber, element);
              else placeholderRefs.current.delete(pageNumber);
            }}
            data-pdf-page-placeholder=""
            data-page-number={pageNumber}
            style={{
              width: pageWidth,
              height: pageHeights[index] ?? estimatedHeight(pageWidth),
              flex: '0 0 auto',
            }}
          >
            {isRendered && (
              <div
                className="relative shadow-md"
                style={{ position: 'relative', width: '100%', height: '100%' }}
              >
                <Page
                  pageNumber={pageNumber}
                  width={pageWidth}
                  renderTextLayer={false}
                  renderAnnotationLayer={false}
                  onRenderSuccess={() => onPageRenderSuccess?.(pageNumber)}
                />
                {renderOverlay && (
                  <div
                    data-pdf-page-overlay=""
                    style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
                  >
                    {renderOverlay(pageNumber)}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default VirtualizedPdfPages;
