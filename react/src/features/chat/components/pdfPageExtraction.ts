import type { PDFDocumentProxy, PageViewport } from 'pdfjs-dist';

import {
  createIncrementalDocumentLocator,
  type DocumentMatchResult,
} from '../../../utils/pdfTextMatch';

export interface PdfTextItem {
  str: string;
  transform: number[];
  width: number;
}

export interface PageScan {
  pageNumber: number;
  items: PdfTextItem[];
  viewport: PageViewport;
}

export type YieldToBrowser = () => Promise<void>;

export interface PdfPageExtractionSession {
  getCached(pageNumber: number): PageScan | undefined;
  ensurePage(pageNumber: number): Promise<PageScan>;
  find(query: string, signal?: AbortSignal): Promise<DocumentMatchResult | null>;
  subscribe(subscriber: (scan: PageScan) => void): () => void;
}

interface PageYieldBarrier {
  promise: Promise<void> | null;
  resolved: boolean;
}

function defaultYieldToBrowser(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof globalThis.requestAnimationFrame === 'function') {
      globalThis.requestAnimationFrame(() => resolve());
      return;
    }
    globalThis.setTimeout(resolve, 0);
  });
}

export function createPdfPageExtractionSession(
  pdfDoc: PDFDocumentProxy,
  pageWidth: number,
  options: { yieldToBrowser?: YieldToBrowser } = {},
): PdfPageExtractionSession {
  const pagePromises = new Map<number, Promise<PageScan>>();
  const pageYieldBarriers = new Map<number, PageYieldBarrier>();
  const completedScans = new Map<number, PageScan>();
  const subscribers = new Set<(scan: PageScan) => void>();
  const yieldToBrowser = options.yieldToBrowser ?? defaultYieldToBrowser;

  const ensurePage = (pageNumber: number): Promise<PageScan> => {
    const cachedPromise = pagePromises.get(pageNumber);
    if (cachedPromise) return cachedPromise;

    let resolveScan!: (scan: PageScan) => void;
    let rejectScan!: (reason?: unknown) => void;
    const scanPromise = new Promise<PageScan>((resolve, reject) => {
      resolveScan = resolve;
      rejectScan = reject;
    });
    pagePromises.set(pageNumber, scanPromise);

    void (async () => {
      try {
        const page = await pdfDoc.getPage(pageNumber);
        const unscaledViewport = page.getViewport({ scale: 1 });
        const viewport = page.getViewport({ scale: pageWidth / unscaledViewport.width });
        const textContent = await page.getTextContent();
        const items: PdfTextItem[] = [];

        for (const item of textContent.items) {
          if (
            'str' in item
            && typeof item.str === 'string'
            && 'transform' in item
            && Array.isArray(item.transform)
          ) {
            items.push({
              str: item.str,
              transform: item.transform,
              width: typeof item.width === 'number' ? item.width : 0,
            });
          }
        }

        const scan = { pageNumber, items, viewport };
        completedScans.set(pageNumber, scan);
        for (const subscriber of Array.from(subscribers)) {
          try {
            subscriber(scan);
          } catch {
            // Subscribers observe completed scans; their failures must not poison extraction.
          }
        }
        resolveScan(scan);
      } catch (error) {
        rejectScan(error);
      }
    })();

    return scanPromise;
  };

  return {
    getCached(pageNumber) {
      return completedScans.get(pageNumber);
    },
    ensurePage,
    async find(query, signal) {
      const locator = createIncrementalDocumentLocator(query);

      for (let pageNumber = 1; pageNumber <= pdfDoc.numPages; pageNumber += 1) {
        if (signal?.aborted) return null;

        const wasAlreadyCompleted = completedScans.has(pageNumber);
        if (!wasAlreadyCompleted && !pageYieldBarriers.has(pageNumber)) {
          pageYieldBarriers.set(pageNumber, { promise: null, resolved: false });
        }
        let scan: PageScan;
        try {
          scan = await ensurePage(pageNumber);
        } catch (error) {
          if (signal?.aborted) return null;
          throw error;
        }
        if (signal?.aborted) return null;

        const result = locator.pushPage(scan);
        if (result) return result;

        const barrier = pageYieldBarriers.get(pageNumber);
        if (barrier && !barrier.resolved && pageNumber < pdfDoc.numPages) {
          if (!barrier.promise && !wasAlreadyCompleted) {
            barrier.promise = yieldToBrowser().then(() => {
              barrier.resolved = true;
            });
          }
          if (barrier.promise) {
            await barrier.promise;
            if (signal?.aborted) return null;
          }
        }
      }

      return locator.finish();
    },
    subscribe(subscriber) {
      subscribers.add(subscriber);
      return () => subscribers.delete(subscriber);
    },
  };
}
