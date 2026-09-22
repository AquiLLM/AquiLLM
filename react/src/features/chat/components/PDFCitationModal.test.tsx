// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import type { PDFDocumentProxy, PageViewport } from 'pdfjs-dist';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { DocumentMatchResult } from '../../../utils/pdfTextMatch';
import type { CitationChunkSummary } from './citationTypes';
import type { PageScan, PdfPageExtractionSession } from './pdfPageExtraction';
import { PDFCitationModal } from './PDFCitationModal';

interface MockDocumentProps {
  file: string;
  onLoadSuccess: (pdf: PDFDocumentProxy) => void;
  error?: React.ReactNode;
  children?: React.ReactNode;
}

interface MockPageProps {
  pageNumber: number;
  onRenderSuccess?: () => void;
}

const pdfProxy = { numPages: 30 } as PDFDocumentProxy;
const originalPerformance = Object.getOwnPropertyDescriptor(globalThis, 'performance');
const sessionFactoryMock = vi.hoisted(() => vi.fn());
const pageRenderMock = vi.hoisted(() => vi.fn());
const pageMock = vi.hoisted(() => ({ autoRender: true }));
const documentMock = vi.hoisted(() => ({
  autoLoad: true,
  fail: false,
  loads: [] as Array<{ file: string; onLoadSuccess: (pdf: PDFDocumentProxy) => void }>,
}));

vi.mock('react-pdf', async () => {
  const { useEffect } = await vi.importActual<typeof import('react')>('react');
  return {
    Document: ({ file, onLoadSuccess, error, children }: MockDocumentProps) => {
      useEffect(() => {
        documentMock.loads.push({ file, onLoadSuccess });
        if (documentMock.autoLoad && !documentMock.fail) onLoadSuccess(pdfProxy);
      }, [file, onLoadSuccess]);
      return <div data-pdf-document="">{documentMock.fail ? error : children}</div>;
    },
    Page: (props: MockPageProps) => {
      pageRenderMock(props);
      useEffect(() => {
        if (pageMock.autoRender) props.onRenderSuccess?.();
      }, [props.onRenderSuccess]);
      return <canvas data-react-pdf-page={props.pageNumber} />;
    },
  };
});

vi.mock('./pdfPageExtraction', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./pdfPageExtraction')>()),
  createPdfPageExtractionSession: sessionFactoryMock,
}));

vi.mock('../../../utils/pdfWorker', () => ({ configurePdfWorker: vi.fn() }));
vi.mock('../../../main', () => ({ getCsrfCookie: () => 'csrf-token' }));

const chunk: CitationChunkSummary = {
  content: 'The complete citation text is long enough to be located in the PDF.',
  chunk_number: 9,
  start_position: 100,
  end_position: 169,
  start_time: null,
  modality: 'text',
  image_url: null,
  document: {
    id: 'doc-1',
    title: 'Thirty page PDF',
    type: 'PDFDocument',
    has_pdf: true,
    source_url: 'https://example.test/source',
  },
};

function makeScan(pageNumber: number): PageScan {
  return {
    pageNumber,
    items: [
      { str: 'first', transform: [1, 6, 0, 8, 10, 50], width: 12 },
      { str: 'second', transform: [1, 0, 0, 5, 30, 70], width: 20 },
    ],
    viewport: {
      width: 600,
      height: 800,
      scale: 2,
      convertToViewportPoint: (x: number, y: number) => [x * 2, 300 - y * 2],
    } as unknown as PageViewport,
  };
}

function makeMatch(pageNumber: number, firstItem = 0, lastItem = 1): DocumentMatchResult {
  return {
    startPage: pageNumber,
    pageHighlights: new Map([[pageNumber, { firstItem, lastItem }]]),
  };
}

function makeSession(match: DocumentMatchResult | null): PdfPageExtractionSession {
  const scans = new Map<number, PageScan>([[19, makeScan(19)]]);
  return {
    getCached: vi.fn((pageNumber: number) => scans.get(pageNumber)),
    ensurePage: vi.fn(async (pageNumber: number) => scans.get(pageNumber) ?? makeScan(pageNumber)),
    find: vi.fn(async () => match),
    subscribe: vi.fn(() => vi.fn()),
  };
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}

interface SessionHarness {
  session: PdfPageExtractionSession;
  find: ReturnType<typeof vi.fn<(query: string, signal?: AbortSignal) => Promise<DocumentMatchResult | null>>>;
  publish: (scan: PageScan) => void;
}

function makeSessionHarness(
  find: (query: string, signal?: AbortSignal) => Promise<DocumentMatchResult | null>,
  cachedPages: number[] = [5, 19],
): SessionHarness {
  const scans = new Map(cachedPages.map((pageNumber) => [pageNumber, makeScan(pageNumber)]));
  let subscriber: ((scan: PageScan) => void) | null = null;
  const findMock = vi.fn(find);
  return {
    session: {
      getCached: vi.fn((pageNumber: number) => scans.get(pageNumber)),
      ensurePage: vi.fn(async (pageNumber: number) => scans.get(pageNumber) ?? makeScan(pageNumber)),
      find: findMock,
      subscribe: vi.fn((nextSubscriber) => {
        subscriber = nextSubscriber;
        return () => {
          subscriber = null;
        };
      }),
    },
    find: findMock,
    publish(scan) {
      scans.set(scan.pageNumber, scan);
      subscriber?.(scan);
    },
  };
}

function response(body: unknown, ok = true): Response {
  return {
    ok,
    status: ok ? 200 : 500,
    json: vi.fn(async () => body),
  } as unknown as Response;
}

function chunkFor(docId: string, title: string, chunkNumber = 9): CitationChunkSummary {
  return {
    ...chunk,
    chunk_number: chunkNumber,
    document: { ...chunk.document, id: docId, title },
  };
}

const frameCallbacks = new Map<number, FrameRequestCallback>();
let nextFrame = 0;

function flushNextFrame(): void {
  const entry = frameCallbacks.entries().next().value as [number, FrameRequestCallback] | undefined;
  if (!entry) return;
  frameCallbacks.delete(entry[0]);
  act(() => entry[1](0));
}

class ControlledIntersectionObserver {
  static instances: ControlledIntersectionObserver[] = [];
  readonly root: Element | Document | null;
  private readonly callback: IntersectionObserverCallback;
  readonly observe = vi.fn();
  readonly unobserve = vi.fn();
  readonly disconnect = vi.fn();

  constructor(callback: IntersectionObserverCallback, options?: IntersectionObserverInit) {
    this.callback = callback;
    this.root = options?.root ?? null;
    ControlledIntersectionObserver.instances.push(this);
  }

  emit(): void {
    this.callback([], this as unknown as IntersectionObserver);
  }
}

async function flushEffects(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe('PDFCitationModal', () => {
  const scrolledPageNumbers: number[] = [];
  const scrollIntoView = vi.fn(function scrollIntoViewMock(this: Element) {
    scrolledPageNumbers.push(Number(
      this.closest('[data-pdf-page-placeholder]')?.getAttribute('data-page-number'),
    ));
  });

  beforeEach(() => {
    vi.useFakeTimers();
    sessionFactoryMock.mockReset();
    pageRenderMock.mockReset();
    pageMock.autoRender = true;
    documentMock.autoLoad = true;
    documentMock.fail = false;
    documentMock.loads.length = 0;
    ControlledIntersectionObserver.instances.length = 0;
    frameCallbacks.clear();
    nextFrame = 0;
    Object.defineProperty(window, 'pageUrls', {
      configurable: true,
      value: {
        pdf: '/documents/%(doc_id)s/pdf/',
        document: '/documents/%(doc_id)s/',
      },
    });
    Object.defineProperty(window, 'apiUrls', {
      configurable: true,
      value: { api_citation_narrow: '/api/citations/narrow/' },
    });
    Object.defineProperty(globalThis, 'requestAnimationFrame', {
      configurable: true,
      value: vi.fn((callback: FrameRequestCallback) => {
        nextFrame += 1;
        frameCallbacks.set(nextFrame, callback);
        return nextFrame;
      }),
    });
    Object.defineProperty(globalThis, 'cancelAnimationFrame', {
      configurable: true,
      value: vi.fn(),
    });
    Object.defineProperty(globalThis, 'IntersectionObserver', {
      configurable: true,
      value: ControlledIntersectionObserver,
    });
    Object.defineProperty(globalThis, 'ResizeObserver', {
      configurable: true,
      value: class {
        observe = vi.fn();
        unobserve = vi.fn();
        disconnect = vi.fn();
      },
    });
    Object.defineProperty(Element.prototype, 'scrollIntoView', {
      configurable: true,
      value: scrollIntoView,
    });
    scrollIntoView.mockReset();
    scrollIntoView.mockImplementation(function scrollIntoViewMock(this: Element) {
      scrolledPageNumbers.push(Number(
        this.closest('[data-pdf-page-placeholder]')?.getAttribute('data-page-number'),
      ));
    });
    scrolledPageNumbers.length = 0;
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    if (originalPerformance) {
      Object.defineProperty(globalThis, 'performance', originalPerformance);
    } else {
      Reflect.deleteProperty(globalThis, 'performance');
    }
  });

  it('virtualizes 30 pages and mounts the exact page 19 highlight before centering it', async () => {
    sessionFactoryMock.mockReturnValue(makeSession(makeMatch(19)));

    render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        preloadedChunk={chunk}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();
    act(() => vi.runOnlyPendingTimers());

    expect(document.querySelectorAll('[data-pdf-page-placeholder]')).toHaveLength(30);
    expect(document.querySelectorAll('canvas[data-react-pdf-page]').length).toBeLessThanOrEqual(7);
    expect(screen.getByText(/highlighted on page 19/)).not.toBeNull();

    const page19 = document.querySelector(
      '[data-pdf-page-placeholder][data-page-number="19"]',
    );
    expect(page19).not.toBeNull();
    const hits = page19!.querySelectorAll<HTMLElement>('[data-citation-hit]');
    expect(hits).toHaveLength(2);
    expect(hits[0].style.left).toBe('20px');
    expect(hits[0].style.top).toBe('180px');
    expect(hits[0].style.width).toBe('24px');
    expect(hits[0].style.height).toBe('20px');
    expect(hits[0].style.backgroundColor).toBe('rgba(253, 224, 71, 0.55)');
    expect(hits[0].style.borderRadius).toBe('2px');
    expect(hits[1].style.left).toBe('60px');
    expect(hits[1].style.top).toBe('150px');
    expect(hits[1].style.width).toBe('40px');
    expect(hits[1].style.height).toBe('10px');
    expect(scrollIntoView).toHaveBeenCalledWith({ block: 'center', behavior: 'smooth' });

    const list = document.querySelector<HTMLElement>('[data-pdf-page-list]');
    const container = list?.parentElement?.parentElement;
    if (!container || !list) throw new Error('Missing virtual page scroll geometry');
    fireEvent(container, new Event('scrollend'));
    Object.defineProperty(container, 'clientHeight', { configurable: true, value: 700 });
    container.scrollTop = 29 * 800;
    vi.spyOn(container, 'getBoundingClientRect').mockReturnValue({
      top: 0,
      bottom: 700,
      height: 700,
      left: 0,
      right: 600,
      width: 600,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
    vi.spyOn(list, 'getBoundingClientRect').mockReturnValue({
      top: -container.scrollTop,
      bottom: 30 * 800 - container.scrollTop,
      height: 30 * 800,
      left: 0,
      right: 600,
      width: 600,
      x: 0,
      y: -container.scrollTop,
      toJSON: () => ({}),
    });
    ControlledIntersectionObserver.instances[
      ControlledIntersectionObserver.instances.length - 1
    ]?.emit();
    while (frameCallbacks.size > 0) flushNextFrame();

    expect(document.querySelector('canvas[data-react-pdf-page="30"]')).not.toBeNull();
    expect(document.querySelector('canvas[data-react-pdf-page="19"]')).toBeNull();
    expect(document.querySelectorAll('canvas[data-react-pdf-page]').length).toBeLessThanOrEqual(7);
  });

  it('waits for the preferred page canvas and keeps its render window locked until scrolling ends', async () => {
    pageMock.autoRender = false;
    sessionFactoryMock.mockReturnValue(makeSession(makeMatch(19)));

    render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        preloadedChunk={chunk}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();
    act(() => vi.advanceTimersByTime(1_000));

    expect(scrolledPageNumbers).toEqual([]);
    const targetRender = [...pageRenderMock.mock.calls]
      .reverse()
      .find(([props]) => props.pageNumber === 19)?.[0] as MockPageProps | undefined;
    if (!targetRender?.onRenderSuccess) throw new Error('Page 19 render callback was not registered');

    act(() => targetRender.onRenderSuccess?.());
    await flushEffects();
    act(() => vi.advanceTimersByTime(100));
    expect(scrolledPageNumbers).toEqual([19]);

    const list = document.querySelector<HTMLElement>('[data-pdf-page-list]');
    const container = list?.parentElement?.parentElement;
    if (!container || !list) throw new Error('Missing virtual page scroll geometry');
    Object.defineProperty(container, 'clientHeight', { configurable: true, value: 700 });
    container.scrollTop = 29 * 800;
    vi.spyOn(container, 'getBoundingClientRect').mockReturnValue({
      top: 0,
      bottom: 700,
      height: 700,
      left: 0,
      right: 600,
      width: 600,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
    vi.spyOn(list, 'getBoundingClientRect').mockReturnValue({
      top: -container.scrollTop,
      bottom: 30 * 800 - container.scrollTop,
      height: 30 * 800,
      left: 0,
      right: 600,
      width: 600,
      x: 0,
      y: -container.scrollTop,
      toJSON: () => ({}),
    });

    ControlledIntersectionObserver.instances[
      ControlledIntersectionObserver.instances.length - 1
    ]?.emit();
    while (frameCallbacks.size > 0) flushNextFrame();
    expect(document.querySelector('canvas[data-react-pdf-page="19"]')).not.toBeNull();

    fireEvent(container, new Event('scrollend'));
    await flushEffects();
    expect(document.querySelector('canvas[data-react-pdf-page="19"]')).toBeNull();
  });

  it('uses a target canvas that finished rendering before the preferred match was known', async () => {
    pageMock.autoRender = false;
    const fullFind = deferred<DocumentMatchResult | null>();
    const harness = makeSessionHarness(() => fullFind.promise, [1]);
    sessionFactoryMock.mockReturnValue(harness.session);

    render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        preloadedChunk={chunk}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();

    const pageOneRender = [...pageRenderMock.mock.calls]
      .reverse()
      .find(([props]) => props.pageNumber === 1)?.[0] as MockPageProps | undefined;
    if (!pageOneRender?.onRenderSuccess) throw new Error('Page 1 render callback was not registered');
    act(() => pageOneRender.onRenderSuccess?.());

    await act(async () => {
      fullFind.resolve(makeMatch(1));
      await Promise.resolve();
    });
    act(() => vi.advanceTimersByTime(100));

    expect(scrolledPageNumbers).toEqual([1]);
  });

  it('centers an early preferred match while unrelated page scans keep publishing', async () => {
    const harness = makeSessionHarness(async () => makeMatch(19), [19]);
    sessionFactoryMock.mockReturnValue(harness.session);

    render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        preloadedChunk={chunk}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();

    for (let pageNumber = 1; pageNumber <= 4; pageNumber += 1) {
      act(() => {
        harness.publish(makeScan(pageNumber));
        vi.advanceTimersByTime(50);
      });
    }

    expect(scrolledPageNumbers).toEqual([19]);
  });

  it('starts PDF load, narrowing, and the full find independently, then reuses one session', async () => {
    const narrowHttp = deferred<Response>();
    const fullFind = deferred<DocumentMatchResult | null>();
    const harness = makeSessionHarness((query) =>
      query === chunk.content ? fullFind.promise : Promise.resolve(makeMatch(19)),
    );
    sessionFactoryMock.mockReturnValue(harness.session);
    vi.mocked(fetch).mockReturnValue(narrowHttp.promise);

    render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        messageUuid="message-1"
        preloadedChunk={chunk}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();

    expect(fetch).toHaveBeenCalledWith(
      '/api/citations/narrow/',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(sessionFactoryMock).toHaveBeenCalledTimes(1);
    expect(harness.find).toHaveBeenCalledWith(chunk.content, expect.any(AbortSignal));

    await act(async () => {
      narrowHttp.resolve(response({ quote: 'a narrower verbatim quote' }));
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(harness.find).toHaveBeenCalledWith(
      'a narrower verbatim quote',
      expect.any(AbortSignal),
    );
    expect(sessionFactoryMock).toHaveBeenCalledTimes(1);
  });

  it('jumps to the full match first and promotes a later narrow match deterministically', async () => {
    const fullFind = deferred<DocumentMatchResult | null>();
    const narrowFind = deferred<DocumentMatchResult | null>();
    const harness = makeSessionHarness((query) =>
      query === chunk.content ? fullFind.promise : narrowFind.promise,
    );
    sessionFactoryMock.mockReturnValue(harness.session);
    vi.mocked(fetch).mockResolvedValue(response({ quote: 'narrow page nineteen quote' }));

    render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        messageUuid="message-1"
        preloadedChunk={chunk}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();

    await act(async () => {
      fullFind.resolve(makeMatch(5));
      await Promise.resolve();
    });
    act(() => vi.runOnlyPendingTimers());
    expect(scrolledPageNumbers).toEqual([5]);
    expect(screen.getByText(/highlighted on page 5/)).not.toBeNull();

    await act(async () => {
      narrowFind.resolve(makeMatch(19));
      await Promise.resolve();
    });
    act(() => vi.runOnlyPendingTimers());
    expect(scrolledPageNumbers).toEqual([5, 19]);
    expect(screen.getByText(/highlighted on page 19/)).not.toBeNull();
  });

  it('keeps the narrow page when its match wins before a late full result', async () => {
    const fullFind = deferred<DocumentMatchResult | null>();
    const narrowFind = deferred<DocumentMatchResult | null>();
    const harness = makeSessionHarness((query) =>
      query === chunk.content ? fullFind.promise : narrowFind.promise,
    );
    sessionFactoryMock.mockReturnValue(harness.session);
    vi.mocked(fetch).mockResolvedValue(response({ quote: 'narrow page nineteen quote' }));

    render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        messageUuid="message-1"
        preloadedChunk={chunk}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();

    await act(async () => {
      narrowFind.resolve(makeMatch(19));
      await Promise.resolve();
    });
    act(() => vi.runOnlyPendingTimers());
    expect(scrolledPageNumbers).toEqual([19]);

    await act(async () => {
      fullFind.resolve(makeMatch(5));
      await Promise.resolve();
    });
    act(() => vi.runOnlyPendingTimers());
    expect(scrolledPageNumbers).toEqual([19]);
    expect(screen.getByText(/highlighted on page 19/)).not.toBeNull();
  });

  it('synchronously resets document state and ignores late proxy and query publications', async () => {
    documentMock.autoLoad = false;
    const oldFind = deferred<DocumentMatchResult | null>();
    const newFind = deferred<DocumentMatchResult | null>();
    const oldHarness = makeSessionHarness(() => oldFind.promise, [5]);
    const newHarness = makeSessionHarness(() => newFind.promise, [19]);
    sessionFactoryMock
      .mockReturnValueOnce(oldHarness.session)
      .mockReturnValueOnce(newHarness.session);
    const onClose = vi.fn();
    const view = render(
      <PDFCitationModal
        docId="old-doc"
        chunkId="5"
        preloadedChunk={chunkFor('old-doc', 'Old PDF', 5)}
        onClose={onClose}
      />,
    );
    await flushEffects();
    const oldLoad = documentMock.loads[0];
    act(() => oldLoad.onLoadSuccess({ numPages: 30 } as PDFDocumentProxy));
    await flushEffects();
    expect(sessionFactoryMock).toHaveBeenCalledTimes(1);

    view.rerender(
      <PDFCitationModal
        docId="new-doc"
        chunkId="19"
        preloadedChunk={chunkFor('new-doc', 'New PDF', 19)}
        onClose={onClose}
      />,
    );

    expect(screen.getByText('New PDF')).not.toBeNull();
    expect(screen.queryByText(/highlighted on page 5/)).toBeNull();
    expect(document.querySelectorAll('[data-pdf-page-placeholder]')).toHaveLength(0);

    act(() => oldLoad.onLoadSuccess({ numPages: 12 } as PDFDocumentProxy));
    await flushEffects();
    expect(sessionFactoryMock).toHaveBeenCalledTimes(1);

    const newLoad = documentMock.loads[documentMock.loads.length - 1];
    if (!newLoad) throw new Error('New PDF load callback was not registered');
    act(() => newLoad.onLoadSuccess({ numPages: 30 } as PDFDocumentProxy));
    await flushEffects();
    expect(sessionFactoryMock).toHaveBeenCalledTimes(2);

    await act(async () => {
      oldFind.resolve(makeMatch(5));
      await Promise.resolve();
    });
    act(() => vi.runOnlyPendingTimers());
    expect(screen.queryByText(/highlighted on page 5/)).toBeNull();
    expect(document.querySelector('[data-page-number="5"] [data-citation-hit]')).toBeNull();

    await act(async () => {
      newFind.resolve(makeMatch(19));
      await Promise.resolve();
    });
    act(() => vi.runOnlyPendingTimers());
    expect(screen.getByText(/highlighted on page 19/)).not.toBeNull();
    expect(scrolledPageNumbers).toEqual([19]);
    expect(sessionFactoryMock).toHaveBeenCalledTimes(2);
  });

  it('preserves source, document, close, Escape, and pin controls', async () => {
    sessionFactoryMock.mockReturnValue(makeSession(null));
    const onClose = vi.fn();
    const onTogglePin = vi.fn();
    const view = render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        preloadedChunk={chunk}
        onClose={onClose}
        onTogglePin={onTogglePin}
      />,
    );
    await flushEffects();

    expect(screen.getByText('View source').getAttribute('href')).toBe(chunk.document.source_url);
    expect(
      screen.getByRole('link', { name: 'Open document page in new tab' }).getAttribute('href'),
    ).toBe('/documents/doc-1/?chunk=9');
    fireEvent.click(screen.getByRole('button', { name: 'Pin citation panel open' }));
    expect(onTogglePin).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Close citation panel' }));
    expect(onClose).toHaveBeenCalledTimes(2);

    view.rerender(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        preloadedChunk={chunk}
        onClose={onClose}
        pinned
        onTogglePin={onTogglePin}
      />,
    );
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(2);
    fireEvent.click(screen.getByRole('button', { name: 'Unpin citation panel' }));
    expect(onTogglePin).toHaveBeenCalledTimes(2);
  });

  it('renders the PDF error fallback and a no-match banner', async () => {
    documentMock.fail = true;
    sessionFactoryMock.mockReturnValue(makeSession(null));
    const view = render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        preloadedChunk={chunk}
        onClose={vi.fn()}
      />,
    );
    expect(screen.getByText('Failed to load PDF.')).not.toBeNull();
    expect(screen.getByText('Open document page').closest('a')).not.toBeNull();

    documentMock.fail = false;
    view.rerender(
      <PDFCitationModal
        docId="doc-2"
        chunkId="10"
        preloadedChunk={chunkFor('doc-2', 'No match PDF', 10)}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();
    expect(screen.getByText("Couldn't locate the cited passage in the PDF text layer.")).not.toBeNull();
  });

  it('records ordered PDF timing marks from provider open-start and first-match only once', async () => {
    const performanceMock = {
      mark: vi.fn(),
      measure: vi.fn(),
    };
    Object.defineProperty(globalThis, 'performance', {
      configurable: true,
      writable: true,
      value: performanceMock,
    });
    const fullFind = deferred<DocumentMatchResult | null>();
    const narrowFind = deferred<DocumentMatchResult | null>();
    const harness = makeSessionHarness((query) =>
      query === chunk.content ? fullFind.promise : narrowFind.promise,
    );
    sessionFactoryMock.mockReturnValue(harness.session);
    vi.mocked(fetch).mockResolvedValue(response({ quote: 'narrow timing quote' }));

    render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        messageUuid="message-1"
        preloadedChunk={chunk}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();
    await act(async () => {
      fullFind.resolve(makeMatch(5));
      await Promise.resolve();
    });
    act(() => vi.runOnlyPendingTimers());
    await act(async () => {
      narrowFind.resolve(makeMatch(19));
      await Promise.resolve();
    });
    act(() => vi.runOnlyPendingTimers());

    expect(performanceMock.mark.mock.calls.map(([name]) => name)).toEqual([
      'aquillm:citation:pdf-ready',
      'aquillm:citation:first-match',
      'aquillm:citation:canvas-mounted',
      'aquillm:citation:highlight-scrolled',
      'aquillm:citation:canvas-mounted',
      'aquillm:citation:highlight-scrolled',
    ]);
    expect(performanceMock.measure.mock.calls).toEqual([
      ['aquillm:citation:pdf-load', 'aquillm:citation:open-start', 'aquillm:citation:pdf-ready'],
      ['aquillm:citation:match', 'aquillm:citation:open-start', 'aquillm:citation:first-match'],
      ['aquillm:citation:canvas', 'aquillm:citation:open-start', 'aquillm:citation:canvas-mounted'],
      ['aquillm:citation:highlight', 'aquillm:citation:open-start', 'aquillm:citation:highlight-scrolled'],
      ['aquillm:citation:canvas', 'aquillm:citation:open-start', 'aquillm:citation:canvas-mounted'],
      ['aquillm:citation:highlight', 'aquillm:citation:open-start', 'aquillm:citation:highlight-scrolled'],
    ]);
  });

  it('keeps PDF viewing nonfatal when User Timing methods are missing or throw', async () => {
    Object.defineProperty(globalThis, 'performance', {
      configurable: true,
      writable: true,
      value: {},
    });
    sessionFactoryMock.mockReturnValue(makeSession(makeMatch(19)));
    const first = render(
      <PDFCitationModal
        docId="doc-1"
        chunkId="9"
        preloadedChunk={chunk}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();
    act(() => vi.runOnlyPendingTimers());
    expect(screen.getByText(/highlighted on page 19/)).not.toBeNull();
    first.unmount();

    const timingFailure = () => {
      throw new Error('User Timing unavailable');
    };
    Object.defineProperty(globalThis, 'performance', {
      configurable: true,
      writable: true,
      value: { mark: timingFailure, measure: timingFailure },
    });
    sessionFactoryMock.mockReturnValue(makeSession(makeMatch(19)));
    render(
      <PDFCitationModal
        docId="doc-2"
        chunkId="10"
        preloadedChunk={chunkFor('doc-2', 'Restricted timing PDF', 10)}
        onClose={vi.fn()}
      />,
    );
    await flushEffects();
    act(() => vi.runOnlyPendingTimers());
    expect(screen.getByText(/highlighted on page 19/)).not.toBeNull();
  });
});
