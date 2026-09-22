// @vitest-environment jsdom

import { useRef } from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import type { PageViewport } from 'pdfjs-dist';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { PageScan, PdfPageExtractionSession } from './pdfPageExtraction';
import { MAX_RENDERED_PAGES, calculateRenderedPages } from './pdfPageWindow';
import {
  VirtualizedPdfPages,
  type VirtualizedPdfPagesProps,
} from './VirtualizedPdfPages';

interface MockPageProps {
  pageNumber: number;
  width: number;
  renderTextLayer: boolean;
  renderAnnotationLayer: boolean;
  onRenderSuccess?: () => void;
}

const pageRenderMock = vi.hoisted(() => vi.fn<(props: MockPageProps) => void>());

vi.mock('react-pdf', () => ({
  Page: (props: MockPageProps) => {
    pageRenderMock(props);
    return (
      <button
        type="button"
        data-react-pdf-page={props.pageNumber}
        onClick={props.onRenderSuccess}
      >
        Page {props.pageNumber}
      </button>
    );
  },
}));

type Subscriber = (scan: PageScan) => void;

interface SessionHarness {
  session: PdfPageExtractionSession;
  ensurePage: ReturnType<typeof vi.fn<(pageNumber: number) => Promise<PageScan>>>;
  unsubscribe: ReturnType<typeof vi.fn>;
  publish: (scan: PageScan) => void;
}

function makeScan(pageNumber: number, height: number, width = 600): PageScan {
  return {
    pageNumber,
    items: [],
    viewport: { width, height } as PageViewport,
  };
}

function makeSession(initialScans: PageScan[] = []): SessionHarness {
  const cached = new Map(initialScans.map((scan) => [scan.pageNumber, scan]));
  let subscriber: Subscriber | null = null;
  const unsubscribe = vi.fn(() => {
    subscriber = null;
  });
  const ensurePage = vi.fn(async (pageNumber: number) => {
    const cachedScan = cached.get(pageNumber);
    if (cachedScan) return cachedScan;
    return makeScan(pageNumber, 700);
  });
  const session: PdfPageExtractionSession = {
    getCached: vi.fn((pageNumber) => cached.get(pageNumber)),
    ensurePage,
    find: vi.fn(async () => null),
    subscribe: vi.fn((nextSubscriber) => {
      subscriber = nextSubscriber;
      return unsubscribe;
    }),
  };

  return {
    session,
    ensurePage,
    unsubscribe,
    publish(scan) {
      cached.set(scan.pageNumber, scan);
      subscriber?.(scan);
    },
  };
}

const defaultProps = {
  numPages: 20,
  pageWidth: 600,
  citationStartPage: 8,
  initialJumpComplete: false,
};

function Harness(
  props: Omit<VirtualizedPdfPagesProps, 'scrollContainerRef'>,
) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  return (
    <div ref={scrollContainerRef} data-testid="scroll-container">
      <VirtualizedPdfPages {...props} scrollContainerRef={scrollContainerRef} />
    </div>
  );
}

function renderPages(
  sessionHarness: SessionHarness,
  props: Partial<Omit<VirtualizedPdfPagesProps, 'session' | 'scrollContainerRef'>> = {},
) {
  return render(
    <Harness {...defaultProps} {...props} session={sessionHarness.session} />,
  );
}

function placeholders(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>('[data-pdf-page-placeholder]'));
}

function placeholder(pageNumber: number): HTMLElement {
  const element = document.querySelector<HTMLElement>(
    `[data-pdf-page-placeholder][data-page-number="${pageNumber}"]`,
  );
  if (!element) throw new Error(`Missing placeholder for page ${pageNumber}`);
  return element;
}

function rect(top: number, height: number): DOMRect {
  return {
    x: 0,
    y: top,
    top,
    bottom: top + height,
    left: 0,
    right: 600,
    width: 600,
    height,
    toJSON: () => ({}),
  };
}

class ControlledIntersectionObserver {
  static instances: ControlledIntersectionObserver[] = [];

  readonly root: Element | Document | null;
  readonly observed: Element[] = [];
  readonly disconnect = vi.fn();
  private readonly callback: IntersectionObserverCallback;

  constructor(callback: IntersectionObserverCallback, options?: IntersectionObserverInit) {
    this.callback = callback;
    this.root = options?.root ?? null;
    ControlledIntersectionObserver.instances.push(this);
  }

  observe = vi.fn((target: Element) => {
    this.observed.push(target);
  });

  unobserve = vi.fn();
  takeRecords = vi.fn((): IntersectionObserverEntry[] => []);
  readonly rootMargin = '0px';
  readonly thresholds = [0];

  emit(pageNumbers: number[], isIntersecting = true) {
    const entries = pageNumbers.map((pageNumber) => {
      const target = placeholder(pageNumber);
      return {
        target,
        isIntersecting,
        intersectionRatio: isIntersecting ? 1 : 0,
        boundingClientRect: target.getBoundingClientRect(),
        intersectionRect: target.getBoundingClientRect(),
        rootBounds: null,
        time: 0,
      } as IntersectionObserverEntry;
    });
    this.callback(entries, this as unknown as IntersectionObserver);
  }
}

class ControlledResizeObserver {
  static instances: ControlledResizeObserver[] = [];

  readonly observed: Element[] = [];
  readonly disconnect = vi.fn();
  private readonly callback: ResizeObserverCallback;

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
    ControlledResizeObserver.instances.push(this);
  }

  observe = vi.fn((target: Element) => {
    this.observed.push(target);
  });

  unobserve = vi.fn();

  emit(target = this.observed[0]) {
    this.callback(
      [{ target, contentRect: target.getBoundingClientRect() } as ResizeObserverEntry],
      this as unknown as ResizeObserver,
    );
  }
}

const originalIntersectionObserver = Object.getOwnPropertyDescriptor(
  globalThis,
  'IntersectionObserver',
);
const originalRequestAnimationFrame = Object.getOwnPropertyDescriptor(
  globalThis,
  'requestAnimationFrame',
);
const originalCancelAnimationFrame = Object.getOwnPropertyDescriptor(
  globalThis,
  'cancelAnimationFrame',
);
const originalResizeObserver = Object.getOwnPropertyDescriptor(
  globalThis,
  'ResizeObserver',
);

let frameCallbacks: Map<number, FrameRequestCallback>;
let requestAnimationFrameMock: ReturnType<
  typeof vi.fn<(callback: FrameRequestCallback) => number>
>;
let cancelAnimationFrameMock: ReturnType<typeof vi.fn<(frame: number) => boolean>>;
let nextFrame: number;

function flushNextFrame() {
  const pending = frameCallbacks.entries().next().value as
    | [number, FrameRequestCallback]
    | undefined;
  if (!pending) throw new Error('Expected a pending animation frame');
  frameCallbacks.delete(pending[0]);
  act(() => pending[1](0));
}

function mockListGeometry(container: HTMLElement, scrollTop: number) {
  const pageHeight = 600 * (11 / 8.5);
  Object.defineProperty(container, 'clientHeight', { configurable: true, value: 500 });
  container.scrollTop = scrollTop;
  vi.spyOn(container, 'getBoundingClientRect').mockReturnValue(rect(100, 500));
  const list = placeholder(1).parentElement;
  if (!list) throw new Error('Missing virtual page list');
  vi.spyOn(list, 'getBoundingClientRect').mockImplementation(() =>
    rect(100 - container.scrollTop, pageHeight * placeholders().length),
  );
}

beforeEach(() => {
  pageRenderMock.mockClear();
  ControlledIntersectionObserver.instances = [];
  ControlledResizeObserver.instances = [];
  frameCallbacks = new Map();
  nextFrame = 0;
  requestAnimationFrameMock = vi.fn((callback: FrameRequestCallback) => {
    nextFrame += 1;
    frameCallbacks.set(nextFrame, callback);
    return nextFrame;
  });
  cancelAnimationFrameMock = vi.fn((frame: number) => frameCallbacks.delete(frame));
  Object.defineProperty(globalThis, 'IntersectionObserver', {
    configurable: true,
    writable: true,
    value: ControlledIntersectionObserver,
  });
  Object.defineProperty(globalThis, 'ResizeObserver', {
    configurable: true,
    writable: true,
    value: ControlledResizeObserver,
  });
  Object.defineProperty(globalThis, 'requestAnimationFrame', {
    configurable: true,
    writable: true,
    value: requestAnimationFrameMock,
  });
  Object.defineProperty(globalThis, 'cancelAnimationFrame', {
    configurable: true,
    writable: true,
    value: cancelAnimationFrameMock,
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  for (const [key, descriptor] of [
    ['IntersectionObserver', originalIntersectionObserver],
    ['requestAnimationFrame', originalRequestAnimationFrame],
    ['cancelAnimationFrame', originalCancelAnimationFrame],
    ['ResizeObserver', originalResizeObserver],
  ] as const) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else Reflect.deleteProperty(globalThis, key);
  }
});

describe('VirtualizedPdfPages', () => {
  it('creates one nonzero estimated-height placeholder for every PDF page', () => {
    const session = makeSession();
    renderPages(session, { numPages: 12, pageWidth: 340 });

    const pagePlaceholders = placeholders();
    expect(pagePlaceholders).toHaveLength(12);
    for (const pagePlaceholder of pagePlaceholders) {
      expect(pagePlaceholder.style.width).toBe('340px');
      expect(pagePlaceholder.style.height).toBe('440px');
    }
  });

  it('uses the calculated citation and viewport window and never mounts over the cap', () => {
    const session = makeSession();
    renderPages(session);

    const expected = calculateRenderedPages({
      numPages: defaultProps.numPages,
      visiblePage: 1,
      citationStartPage: defaultProps.citationStartPage,
      initialJumpComplete: defaultProps.initialJumpComplete,
    });
    const mounted = screen
      .getAllByRole('button')
      .map((page) => Number(page.getAttribute('data-react-pdf-page')));

    expect(mounted).toEqual(expected);
    expect(mounted.length).toBeLessThanOrEqual(MAX_RENDERED_PAGES);
    expect(placeholders()).toHaveLength(defaultProps.numPages);
    expect(placeholder(12).style.height).not.toBe('0px');
    expect(placeholder(12).querySelector('[data-react-pdf-page]')).toBeNull();
  });

  it('keeps the original page shadow on each mounted PDF page', () => {
    const session = makeSession();
    renderPages(session);

    const mountedPage = placeholder(1).firstElementChild;
    expect(mountedPage?.classList.contains('relative')).toBe(true);
    expect(mountedPage?.classList.contains('shadow-md')).toBe(true);
  });

  it('disables React-PDF layers on every mounted page', () => {
    const session = makeSession();
    renderPages(session, { numPages: 4, citationStartPage: null, initialJumpComplete: true });

    expect(pageRenderMock).not.toHaveBeenCalledTimes(0);
    for (const [props] of pageRenderMock.mock.calls) {
      expect(props.renderTextLayer).toBe(false);
      expect(props.renderAnnotationLayer).toBe(false);
      expect(props.width).toBe(600);
    }
  });

  it('roots an observer at the scroll container and updates the render window deterministically', () => {
    const session = makeSession([makeScan(1, 700), makeScan(15, 700)]);
    renderPages(session, { initialJumpComplete: true });
    const container = screen.getByTestId('scroll-container');
    const observer = ControlledIntersectionObserver.instances[0];
    const pageHeight = 600 * (11 / 8.5);
    mockListGeometry(container, 700 + 13 * pageHeight);

    expect(observer.root).toBe(container);
    expect(observer.observed).toHaveLength(defaultProps.numPages);

    act(() => observer.emit([16, 15]));
    flushNextFrame();

    expect(screen.getByRole('button', { name: 'Page 15' })).not.toBeNull();
    expect(screen.queryAllByRole('button').length).toBeLessThanOrEqual(MAX_RENDERED_PAGES);
  });

  it('calculates the initial window from a nonzero scroll position', () => {
    const session = makeSession();
    renderPages(session, {
      numPages: 8,
      citationStartPage: null,
      initialJumpComplete: true,
    });
    const container = screen.getByTestId('scroll-container');
    const pageHeight = 600 * (11 / 8.5);
    mockListGeometry(container, pageHeight * 3);
    fireEvent.scroll(container);
    fireEvent.scroll(container);
    expect(requestAnimationFrameMock).toHaveBeenCalledTimes(1);

    flushNextFrame();

    expect(screen.getByRole('button', { name: 'Page 4' })).not.toBeNull();
    expect(screen.getByRole('button', { name: 'Page 5' })).not.toBeNull();
    expect(screen.queryByRole('button', { name: 'Page 1' })).toBeNull();
  });

  it('recalculates the midpoint after a page height changes', () => {
    const session = makeSession();
    renderPages(session, {
      numPages: 5,
      citationStartPage: null,
      initialJumpComplete: true,
    });
    const container = screen.getByTestId('scroll-container');
    mockListGeometry(container, 600);
    flushNextFrame();
    expect(screen.getByRole('button', { name: 'Page 3' })).not.toBeNull();

    act(() => session.publish(makeScan(1, 1200)));
    flushNextFrame();

    expect(screen.queryByRole('button', { name: 'Page 3' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Page 1' })).not.toBeNull();
  });

  it('recalculates visibility when the scroll container resizes', () => {
    const session = makeSession();
    renderPages(session, {
      numPages: 5,
      citationStartPage: null,
      initialJumpComplete: true,
    });
    const container = screen.getByTestId('scroll-container');
    let viewportHeight = 100;
    Object.defineProperty(container, 'clientHeight', {
      configurable: true,
      get: () => viewportHeight,
    });
    container.scrollTop = 700;
    vi.spyOn(container, 'getBoundingClientRect').mockImplementation(() =>
      rect(100, viewportHeight),
    );
    const list = placeholder(1).parentElement;
    if (!list) throw new Error('Missing virtual page list');
    vi.spyOn(list, 'getBoundingClientRect').mockImplementation(() =>
      rect(100 - container.scrollTop, 4000),
    );
    flushNextFrame();
    expect(screen.queryByRole('button', { name: 'Page 3' })).toBeNull();

    viewportHeight = 400;
    act(() => ControlledResizeObserver.instances[0].emit(container));
    flushNextFrame();

    expect(screen.getByRole('button', { name: 'Page 3' })).not.toBeNull();
  });

  it('cancels a pending visibility frame on unmount', () => {
    const session = makeSession();
    const view = renderPages(session);
    const container = screen.getByTestId('scroll-container');
    const intersectionObserver = ControlledIntersectionObserver.instances[0];
    const resizeObserver = ControlledResizeObserver.instances[0];
    expect(frameCallbacks.size).toBe(1);

    view.unmount();

    expect(cancelAnimationFrameMock).toHaveBeenCalledTimes(1);
    expect(frameCallbacks.size).toBe(0);
    expect(intersectionObserver.disconnect).toHaveBeenCalledTimes(1);
    expect(resizeObserver.disconnect).toHaveBeenCalledTimes(1);
    requestAnimationFrameMock.mockClear();
    fireEvent.scroll(container);
    expect(requestAnimationFrameMock).not.toHaveBeenCalled();
  });

  it('synchronously resets the effective visible page when the session changes', () => {
    const first = makeSession([makeScan(1, 700), makeScan(5, 700)]);
    const view = renderPages(first, {
      numPages: 8,
      citationStartPage: null,
      initialJumpComplete: true,
    });
    const container = screen.getByTestId('scroll-container');
    const pageHeight = 600 * (11 / 8.5);
    mockListGeometry(container, pageHeight * 4);
    act(() => ControlledIntersectionObserver.instances[0].emit([5]));
    while (frameCallbacks.size > 0) flushNextFrame();
    expect(screen.getByRole('button', { name: 'Page 5' })).not.toBeNull();

    const replacement = makeSession();
    view.rerender(
      <Harness
        {...defaultProps}
        numPages={8}
        citationStartPage={null}
        initialJumpComplete
        session={replacement.session}
      />,
    );

    expect(screen.getByRole('button', { name: 'Page 1' })).not.toBeNull();
    expect(screen.queryByRole('button', { name: 'Page 5' })).toBeNull();
    expect(replacement.ensurePage).toHaveBeenCalledWith(1);
    expect(replacement.ensurePage).not.toHaveBeenCalledWith(5);
  });

  it('keeps visibility current across incremental observer callbacks and scroll', () => {
    const session = makeSession([makeScan(1, 700), makeScan(5, 700), makeScan(6, 700)]);
    renderPages(session, { initialJumpComplete: true });
    const container = screen.getByTestId('scroll-container');
    const pageHeight = 600 * (11 / 8.5);
    const page5Top = 700 + 3 * pageHeight;
    mockListGeometry(container, page5Top);
    const observer = ControlledIntersectionObserver.instances[0];

    act(() => observer.emit([5]));
    flushNextFrame();
    expect(screen.getByRole('button', { name: 'Page 5' })).not.toBeNull();

    act(() => observer.emit([6]));
    flushNextFrame();
    expect(screen.getByRole('button', { name: 'Page 4' })).not.toBeNull();
    expect(screen.queryByRole('button', { name: 'Page 7' })).toBeNull();

    container.scrollTop = page5Top + 700;
    fireEvent.scroll(container);
    flushNextFrame();
    expect(screen.getByRole('button', { name: 'Page 7' })).not.toBeNull();

    act(() => observer.emit([5], false));
    flushNextFrame();
    expect(screen.getByRole('button', { name: 'Page 7' })).not.toBeNull();
  });

  it('requests an approaching visible uncached page and handles extraction rejection', async () => {
    const session = makeSession([makeScan(1, 700)]);
    session.ensurePage.mockRejectedValue(new Error('scan failed'));
    renderPages(session, { initialJumpComplete: true });
    const container = screen.getByTestId('scroll-container');
    const pageHeight = 600 * (11 / 8.5);
    mockListGeometry(container, 700 + 3 * pageHeight);

    act(() => ControlledIntersectionObserver.instances[0].emit([5]));
    flushNextFrame();

    await waitFor(() => expect(session.ensurePage).toHaveBeenCalledWith(5));
  });

  it('uses a single requestAnimationFrame per scroll burst when observers are unavailable', () => {
    Reflect.deleteProperty(globalThis, 'IntersectionObserver');
    const callbacks = new Map<number, FrameRequestCallback>();
    let nextFrame = 0;
    const requestAnimationFrame = vi.fn((callback: FrameRequestCallback) => {
      nextFrame += 1;
      callbacks.set(nextFrame, callback);
      return nextFrame;
    });
    const cancelAnimationFrame = vi.fn((frame: number) => callbacks.delete(frame));
    Object.defineProperty(globalThis, 'requestAnimationFrame', {
      configurable: true,
      value: requestAnimationFrame,
    });
    Object.defineProperty(globalThis, 'cancelAnimationFrame', {
      configurable: true,
      value: cancelAnimationFrame,
    });
    const session = makeSession([makeScan(1, 700), makeScan(2, 700)]);
    renderPages(session, {
      numPages: 6,
      citationStartPage: null,
      initialJumpComplete: true,
    });
    const container = screen.getByTestId('scroll-container');
    Object.defineProperty(container, 'clientHeight', { configurable: true, value: 500 });
    container.scrollTop = 900;
    vi.spyOn(container, 'getBoundingClientRect').mockReturnValue(rect(100, 500));
    const list = placeholder(1).parentElement;
    if (!list) throw new Error('Missing virtual page list');
    vi.spyOn(list, 'getBoundingClientRect').mockImplementation(() =>
      rect(100 - container.scrollTop, 4200),
    );
    const placeholderRectSpies = placeholders().map((element) =>
      vi.spyOn(element, 'getBoundingClientRect'),
    );

    fireEvent.scroll(container);
    fireEvent.scroll(container);
    expect(requestAnimationFrame).toHaveBeenCalledTimes(1);

    act(() => callbacks.get(1)?.(0));

    expect(screen.getByRole('button', { name: 'Page 2' })).not.toBeNull();
    expect(screen.getByRole('button', { name: 'Page 3' })).not.toBeNull();
    expect(requestAnimationFrame).toHaveBeenCalledTimes(1);
    placeholderRectSpies.forEach((spy) => expect(spy).not.toHaveBeenCalled());
  });

  it('seeds and rechecks cached scans around subscription, then unsubscribes', () => {
    const initial = makeScan(2, 820);
    const raced = makeScan(3, 930);
    const cached = new Map<number, PageScan>([[2, initial]]);
    let subscriber: Subscriber | null = null;
    const unsubscribe = vi.fn();
    const session: PdfPageExtractionSession = {
      getCached: vi.fn((pageNumber) => cached.get(pageNumber)),
      ensurePage: vi.fn(async (pageNumber) => makeScan(pageNumber, 700)),
      find: vi.fn(async () => null),
      subscribe: vi.fn((nextSubscriber) => {
        subscriber = nextSubscriber;
        cached.set(3, raced);
        return unsubscribe;
      }),
    };
    const view = render(
      <Harness {...defaultProps} numPages={4} session={session} />,
    );

    expect(placeholder(2).style.height).toBe('820px');
    expect(placeholder(3).style.height).toBe('930px');

    act(() => subscriber?.(makeScan(4, 1040)));
    expect(placeholder(4).style.height).toBe('1040px');

    const replacement = makeSession();
    view.rerender(<Harness {...defaultProps} numPages={4} session={replacement.session} />);
    expect(unsubscribe).toHaveBeenCalledTimes(1);
    view.unmount();
    expect(replacement.unsubscribe).toHaveBeenCalledTimes(1);
  });

  it('compensates scrollTop by the exact height delta only for a page wholly above', () => {
    const session = makeSession();
    renderPages(session, { numPages: 3, citationStartPage: null, initialJumpComplete: true });
    const container = screen.getByTestId('scroll-container');
    container.scrollTop = 1000;
    vi.spyOn(container, 'getBoundingClientRect').mockReturnValue(rect(100, 500));
    const list = placeholder(1).parentElement;
    if (!list) throw new Error('Missing virtual page list');
    vi.spyOn(list, 'getBoundingClientRect').mockImplementation(() =>
      rect(100 - container.scrollTop, 2400),
    );
    const estimate = 600 * (11 / 8.5);

    act(() => session.publish(makeScan(1, estimate + 125)));
    expect(container.scrollTop).toBe(1125);

    act(() => session.publish(makeScan(2, estimate + 200)));
    act(() => session.publish(makeScan(3, estimate + 300)));
    expect(container.scrollTop).toBe(1125);
  });

  it('positions overlays against the mounted page wrapper and reports render success by page', () => {
    const session = makeSession();
    const onPageRenderSuccess = vi.fn();
    const renderOverlay = vi.fn((pageNumber: number) => (
      <span data-testid={`overlay-${pageNumber}`} />
    ));
    renderPages(session, {
      numPages: 2,
      citationStartPage: null,
      initialJumpComplete: true,
      renderOverlay,
      onPageRenderSuccess,
    });

    const page = screen.getByRole('button', { name: 'Page 1' });
    const wrapper = page.parentElement;
    const overlay = screen.getByTestId('overlay-1').parentElement;
    expect(wrapper?.style.position).toBe('relative');
    expect(overlay?.style.position).toBe('absolute');
    expect(overlay?.style.inset).toBe('0px');

    fireEvent.click(page);
    expect(onPageRenderSuccess).toHaveBeenCalledWith(1);
  });
});
