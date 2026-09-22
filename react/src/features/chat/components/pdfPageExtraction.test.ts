import type { PDFDocumentProxy, PDFPageProxy, PageViewport } from 'pdfjs-dist';
import { describe, expect, it } from 'vitest';

import {
  createPdfPageExtractionSession,
  type PageScan,
  type YieldToBrowser,
} from './pdfPageExtraction';

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: Deferred<T>['resolve'];
  let reject!: Deferred<T>['reject'];
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

interface FakePageSpec {
  items: unknown[];
  width?: number;
  height?: number;
  getPageGate?: Promise<void>;
  textContentGate?: Promise<void>;
}

function textItem(str: string, width = str.length): unknown {
  return {
    str,
    transform: [1, 0, 0, 1, 12, 24],
    width,
  };
}

function createFakePdf(pageSpecs: FakePageSpec[]) {
  const getPageCalls: number[] = [];
  const getTextContentCalls: number[] = [];

  const pdfDoc = {
    numPages: pageSpecs.length,
    getPage: async (pageNumber: number) => {
      getPageCalls.push(pageNumber);
      const spec = pageSpecs[pageNumber - 1];
      if (!spec) throw new Error(`Unexpected page ${pageNumber}`);
      await spec.getPageGate;

      const width = spec.width ?? 300;
      const height = spec.height ?? 450;
      return {
        getViewport: ({ scale }: { scale: number }) => ({
          width: width * scale,
          height: height * scale,
          scale,
        }) as PageViewport,
        getTextContent: async () => {
          getTextContentCalls.push(pageNumber);
          await spec.textContentGate;
          return { items: spec.items };
        },
      } as PDFPageProxy;
    },
  } as PDFDocumentProxy;

  return { pdfDoc, getPageCalls, getTextContentCalls };
}

const noYield: YieldToBrowser = () => Promise.resolve();

describe('createPdfPageExtractionSession', () => {
  it('shares an in-flight page promise and extracts each PDF page at most once', async () => {
    const textGate = deferred<void>();
    const transform = [1, 0, 0, 1, 12, 24];
    const fake = createFakePdf([{
      width: 300,
      height: 450,
      textContentGate: textGate.promise,
      items: [
        { str: 'kept', transform, width: 42 },
        { str: 'defaults width', transform },
        { str: 'missing transform', width: 10 },
        { transform, width: 10 },
      ],
    }]);
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, { yieldToBrowser: noYield });

    const first = session.ensurePage(1);
    const second = session.ensurePage(1);

    expect(second).toBe(first);
    expect(session.getCached(1)).toBeUndefined();
    expect(fake.getPageCalls).toEqual([1]);

    textGate.resolve();
    const [firstScan, secondScan] = await Promise.all([first, second]);

    expect(secondScan).toBe(firstScan);
    expect(fake.getPageCalls).toEqual([1]);
    expect(fake.getTextContentCalls).toEqual([1]);
    expect(firstScan).toMatchObject({
      pageNumber: 1,
      items: [
        { str: 'kept', transform, width: 42 },
        { str: 'defaults width', transform, width: 0 },
      ],
      viewport: { width: 600, height: 900, scale: 2 },
    });
    expect(session.getCached(1)).toBe(firstScan);
    expect(session.ensurePage(1)).toBe(first);
  });

  it('stops scanning as soon as the incremental locator completes a match', async () => {
    const fake = createFakePdf([
      { items: [textItem('unrelated first page')] },
      { items: [textItem('the precise target phrase lives on page two')] },
      { items: [textItem('a later page must never be requested')] },
    ]);
    let yieldCalls = 0;
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, {
      yieldToBrowser: async () => {
        yieldCalls += 1;
      },
    });

    const result = await session.find('the precise target phrase lives on page two');

    expect(result).toEqual({
      startPage: 2,
      pageHighlights: new Map([[2, { firstItem: 0, lastItem: 0 }]]),
    });
    expect(fake.getPageCalls).toEqual([1, 2]);
    expect(fake.getTextContentCalls).toEqual([1, 2]);
    expect(yieldCalls).toBe(1);
  });

  it('uses the injected browser yield between page requests', async () => {
    const yieldGate = deferred<void>();
    const yieldStarted = deferred<void>();
    const fake = createFakePdf([
      { items: [textItem('unrelated first page')] },
      { items: [textItem('the awaited target phrase is on the second page')] },
    ]);
    let yieldCalls = 0;
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, {
      yieldToBrowser: () => {
        yieldCalls += 1;
        yieldStarted.resolve();
        return yieldGate.promise;
      },
    });

    const resultPromise = session.find('the awaited target phrase is on the second page');
    await yieldStarted.promise;

    expect(yieldCalls).toBe(1);
    expect(fake.getPageCalls).toEqual([1]);

    yieldGate.resolve();
    const result = await resultPromise;

    expect(result?.startPage).toBe(2);
    expect(fake.getPageCalls).toEqual([1, 2]);
  });

  it('shares one post-page yield barrier across concurrent searches', async () => {
    const yieldGate = deferred<void>();
    const yieldStarted = deferred<void>();
    const firstQuery = 'the first concurrent target appears on page two';
    const secondQuery = 'the second concurrent target appears on page two';
    const fake = createFakePdf([
      { items: [textItem('unrelated first page')] },
      { items: [textItem(firstQuery), textItem(secondQuery)] },
    ]);
    let yieldCalls = 0;
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, {
      yieldToBrowser: () => {
        yieldCalls += 1;
        yieldStarted.resolve();
        return yieldGate.promise;
      },
    });

    const firstResult = session.find(firstQuery);
    const secondResult = session.find(secondQuery);
    await yieldStarted.promise;

    expect(yieldCalls).toBe(1);
    expect(fake.getPageCalls).toEqual([1]);

    yieldGate.resolve();
    await expect(Promise.all([firstResult, secondResult])).resolves.toEqual([
      {
        startPage: 2,
        pageHighlights: new Map([[2, { firstItem: 0, lastItem: 0 }]]),
      },
      {
        startPage: 2,
        pageHighlights: new Map([[2, { firstItem: 1, lastItem: 1 }]]),
      },
    ]);
    expect(fake.getPageCalls).toEqual([1, 2]);
    expect(yieldCalls).toBe(1);

    yieldCalls = 0;
    expect((await session.find(secondQuery))?.startPage).toBe(2);
    expect(yieldCalls).toBe(0);
  });

  it('joins an active barrier when starting after its page scan completed', async () => {
    const yieldGate = deferred<void>();
    const yieldStarted = deferred<void>();
    const query = 'the shared target appears only on the second page';
    const fake = createFakePdf([
      { items: [textItem('unrelated first page')] },
      { items: [textItem(query)] },
    ]);
    let yieldCalls = 0;
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, {
      yieldToBrowser: () => {
        yieldCalls += 1;
        yieldStarted.resolve();
        return yieldGate.promise;
      },
    });

    const firstResult = session.find(query);
    await yieldStarted.promise;
    expect(session.getCached(1)?.pageNumber).toBe(1);

    const lateResult = session.find(query);
    await Promise.resolve();

    expect(fake.getPageCalls).toEqual([1]);
    expect(yieldCalls).toBe(1);

    yieldGate.resolve();
    await expect(Promise.all([firstResult, lateResult])).resolves.toEqual([
      {
        startPage: 2,
        pageHighlights: new Map([[2, { firstItem: 0, lastItem: 0 }]]),
      },
      {
        startPage: 2,
        pageHighlights: new Map([[2, { firstItem: 0, lastItem: 0 }]]),
      },
    ]);
    expect(fake.getPageCalls).toEqual([1, 2]);
    expect(yieldCalls).toBe(1);
  });

  it('creates a yield barrier when a search joins an externally scheduled in-flight scan', async () => {
    const textGate = deferred<void>();
    const yieldGate = deferred<void>();
    const yieldStarted = deferred<void>();
    const query = 'the external extraction target appears on page two';
    const fake = createFakePdf([
      {
        items: [textItem('unrelated first page')],
        textContentGate: textGate.promise,
      },
      { items: [textItem(query)] },
    ]);
    let yieldCalls = 0;
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, {
      yieldToBrowser: () => {
        yieldCalls += 1;
        yieldStarted.resolve();
        return yieldGate.promise;
      },
    });

    const externalPage = session.ensurePage(1);
    const findResult = session.find(query);
    const firstContinuation = Promise.race([
      yieldStarted.promise.then(() => 'yield' as const),
      findResult.then(() => 'result' as const),
    ]);
    textGate.resolve();
    await externalPage;

    expect(await firstContinuation).toBe('yield');
    expect(fake.getPageCalls).toEqual([1]);
    expect(yieldCalls).toBe(1);

    yieldGate.resolve();
    expect((await findResult)?.startPage).toBe(2);
    expect(fake.getPageCalls).toEqual([1, 2]);
    expect(yieldCalls).toBe(1);
  });

  it('ignores a stale barrier when replaying a scan that completed an earlier search', async () => {
    const firstQuery = 'the first search completes immediately on page one';
    const secondQuery = 'the later search target appears only on page two';
    const fake = createFakePdf([
      { items: [textItem(firstQuery)] },
      { items: [textItem(secondQuery)] },
    ]);
    let yieldCalls = 0;
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, {
      yieldToBrowser: async () => {
        yieldCalls += 1;
      },
    });

    expect((await session.find(firstQuery))?.startPage).toBe(1);
    expect(fake.getPageCalls).toEqual([1]);
    expect(yieldCalls).toBe(0);

    expect((await session.find(secondQuery))?.startPage).toBe(2);
    expect(fake.getPageCalls).toEqual([1, 2]);
    expect(yieldCalls).toBe(0);
  });

  it('replays cached pages for a later narrow query without duplicate PDF calls', async () => {
    const fake = createFakePdf([
      { items: [textItem('unrelated first page')] },
      { items: [textItem('the later narrow quote appears on page two')] },
      { items: [textItem('the original citation appears only on page three')] },
    ]);
    let yieldCalls = 0;
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, {
      yieldToBrowser: async () => {
        yieldCalls += 1;
      },
    });

    expect((await session.find('the original citation appears only on page three'))?.startPage).toBe(3);
    expect(fake.getPageCalls).toEqual([1, 2, 3]);
    expect(fake.getTextContentCalls).toEqual([1, 2, 3]);
    expect(yieldCalls).toBe(2);

    yieldCalls = 0;
    expect((await session.find('the later narrow quote appears on page two'))?.startPage).toBe(2);
    expect(fake.getPageCalls).toEqual([1, 2, 3]);
    expect(fake.getTextContentCalls).toEqual([1, 2, 3]);
    expect(yieldCalls).toBe(0);
  });

  it('returns null without scheduling work when already aborted', async () => {
    const fake = createFakePdf([{ items: [textItem('the target would otherwise match this page')] }]);
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, { yieldToBrowser: noYield });
    const controller = new AbortController();
    controller.abort();

    await expect(session.find('the target would otherwise match this page', controller.signal))
      .resolves.toBeNull();
    expect(fake.getPageCalls).toEqual([]);
    expect(fake.getTextContentCalls).toEqual([]);
  });

  it('suppresses a result after an extraction await is aborted while keeping the scan cached', async () => {
    const textGate = deferred<void>();
    const fake = createFakePdf([
      {
        items: [textItem('the target completes on the in flight first page')],
        textContentGate: textGate.promise,
      },
      { items: [textItem('a later page must never be requested')] },
    ]);
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, { yieldToBrowser: noYield });
    const controller = new AbortController();

    const resultPromise = session.find(
      'the target completes on the in flight first page',
      controller.signal,
    );
    const inFlightPage = session.ensurePage(1);
    controller.abort();
    textGate.resolve();

    await expect(resultPromise).resolves.toBeNull();
    const scan = await inFlightPage;
    expect(session.ensurePage(1)).toBe(inFlightPage);
    expect(session.getCached(1)).toBe(scan);
    expect(fake.getPageCalls).toEqual([1]);
    expect(fake.getTextContentCalls).toEqual([1]);
  });

  it('ignores a stale barrier when replaying a scan completed after abort', async () => {
    const textGate = deferred<void>();
    const laterQuery = 'the replay target appears only on the second page';
    const fake = createFakePdf([
      {
        items: [textItem('the first page completes after its search aborts')],
        textContentGate: textGate.promise,
      },
      { items: [textItem(laterQuery)] },
    ]);
    let yieldCalls = 0;
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, {
      yieldToBrowser: async () => {
        yieldCalls += 1;
      },
    });
    const controller = new AbortController();

    const abortedFind = session.find(
      'the first page completes after its search aborts',
      controller.signal,
    );
    controller.abort();
    textGate.resolve();

    await expect(abortedFind).resolves.toBeNull();
    expect(session.getCached(1)?.pageNumber).toBe(1);
    expect(yieldCalls).toBe(0);

    expect((await session.find(laterQuery))?.startPage).toBe(2);
    expect(fake.getPageCalls).toEqual([1, 2]);
    expect(yieldCalls).toBe(0);
  });

  it('returns null when an in-flight extraction rejects after the search is aborted', async () => {
    const extractionGate = deferred<void>();
    const fake = createFakePdf([
      {
        items: [textItem('the first page extraction will fail after abort')],
        getPageGate: extractionGate.promise,
      },
      { items: [textItem('a later page must never be requested')] },
    ]);
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, { yieldToBrowser: noYield });
    const controller = new AbortController();

    const resultPromise = session.find(
      'the first page extraction will fail after abort',
      controller.signal,
    );
    controller.abort();
    extractionGate.reject(new Error('page extraction failed'));

    await expect(resultPromise).resolves.toBeNull();
    expect(fake.getPageCalls).toEqual([1]);
    expect(fake.getTextContentCalls).toEqual([]);
  });

  it('does not schedule the next page or return a result after a browser yield is aborted', async () => {
    const yieldGate = deferred<void>();
    const yieldStarted = deferred<void>();
    const fake = createFakePdf([
      { items: [textItem('unrelated first page')] },
      { items: [textItem('the target on page two must be suppressed after abort')] },
    ]);
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, {
      yieldToBrowser: () => {
        yieldStarted.resolve();
        return yieldGate.promise;
      },
    });
    const controller = new AbortController();

    const resultPromise = session.find(
      'the target on page two must be suppressed after abort',
      controller.signal,
    );
    await yieldStarted.promise;
    controller.abort();
    yieldGate.resolve();

    await expect(resultPromise).resolves.toBeNull();
    expect(fake.getPageCalls).toEqual([1]);
    expect(fake.getTextContentCalls).toEqual([1]);
    expect(session.getCached(1)?.pageNumber).toBe(1);
  });

  it('extracts an arbitrary requested page on demand without scanning earlier pages', async () => {
    const fake = createFakePdf([
      { items: [textItem('page one')] },
      { items: [textItem('page two')] },
      { items: [textItem('page three')] },
    ]);
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, { yieldToBrowser: noYield });

    const scan = await session.ensurePage(3);

    expect(scan.pageNumber).toBe(3);
    expect(fake.getPageCalls).toEqual([3]);
    expect(fake.getTextContentCalls).toEqual([3]);
    expect(session.getCached(3)).toBe(scan);
    expect(session.getCached(1)).toBeUndefined();
  });

  it('calls finish only at EOF so a missing-end match falls back after every page', async () => {
    const longStart = 'start anchor material '.repeat(12);
    const longEnd = 'missing ending material '.repeat(12);
    const fake = createFakePdf([
      { items: [textItem('preface'), textItem(longStart), textItem('tail of start page')] },
      { items: [textItem('middle page without the requested ending')] },
      { items: [textItem('final page still without the requested ending')] },
    ]);
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, { yieldToBrowser: noYield });

    const result = await session.find(`${longStart} ${longEnd}`);

    expect(fake.getPageCalls).toEqual([1, 2, 3]);
    expect(result).toEqual({
      startPage: 1,
      pageHighlights: new Map([[1, { firstItem: 1, lastItem: 2 }]]),
    });
  });

  it('publishes each successful scan once, supports unsubscribe, and never replays to late subscribers', async () => {
    const fake = createFakePdf([
      { items: [textItem('page one')] },
      { items: [textItem('page two')] },
      { items: [textItem('page three')] },
    ]);
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, { yieldToBrowser: noYield });
    const seenByFirst: PageScan[] = [];
    const seenBySecond: PageScan[] = [];
    const seenByLateSubscriber: PageScan[] = [];
    const unsubscribeFirst = session.subscribe((scan) => seenByFirst.push(scan));
    session.subscribe((scan) => seenBySecond.push(scan));

    const pageOne = await session.ensurePage(1);
    await session.ensurePage(1);
    expect(seenByFirst).toEqual([pageOne]);
    expect(seenBySecond).toEqual([pageOne]);

    unsubscribeFirst();
    const pageTwo = await session.ensurePage(2);
    expect(seenByFirst).toEqual([pageOne]);
    expect(seenBySecond).toEqual([pageOne, pageTwo]);

    session.subscribe((scan) => seenByLateSubscriber.push(scan));
    expect(seenByLateSubscriber).toEqual([]);
    expect(session.getCached(1)).toBe(pageOne);
    expect(session.getCached(2)).toBe(pageTwo);

    const pageThree = await session.ensurePage(3);
    expect(seenBySecond).toEqual([pageOne, pageTwo, pageThree]);
    expect(seenByLateSubscriber).toEqual([pageThree]);
    expect(fake.getPageCalls).toEqual([1, 2, 3]);
  });

  it('isolates throwing subscribers and snapshots delivery against reentrant additions', async () => {
    const fake = createFakePdf([
      { items: [textItem('page one')] },
      { items: [textItem('page two')] },
    ]);
    const session = createPdfPageExtractionSession(fake.pdfDoc, 600, { yieldToBrowser: noYield });
    const seenByLaterSubscriber: number[] = [];
    const seenByReentrantSubscriber: number[] = [];

    session.subscribe((scan) => {
      if (scan.pageNumber === 1) {
        session.subscribe((laterScan) => seenByReentrantSubscriber.push(laterScan.pageNumber));
      }
      throw new Error('listener failed');
    });
    session.subscribe((scan) => seenByLaterSubscriber.push(scan.pageNumber));

    await expect(session.ensurePage(1)).resolves.toMatchObject({ pageNumber: 1 });
    expect(seenByLaterSubscriber).toEqual([1]);
    expect(seenByReentrantSubscriber).toEqual([]);

    await expect(session.ensurePage(2)).resolves.toMatchObject({ pageNumber: 2 });
    expect(seenByLaterSubscriber).toEqual([1, 2]);
    expect(seenByReentrantSubscriber).toEqual([2]);
  });
});
