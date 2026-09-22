import { describe, it, expect } from 'vitest';
import {
  createIncrementalDocumentLocator,
  locateInPage,
  locateAcrossDocument,
  type PDFTextItemLike,
} from './pdfTextMatch';

const items = (...strs: string[]): PDFTextItemLike[] => strs.map((str) => ({ str }));

const longStart =
  'the transformer architecture relies entirely on self attention mechanisms ' +
  'that allow the model to weigh the relative importance of many different ' +
  'tokens appearing throughout the entire input sequence at once';
const longEnd =
  'this particular design choice enables highly parallel computation across ' +
  'the whole sequence and over time has become the single dominant modeling ' +
  'approach for large scale natural language processing systems deployed today';

describe('locateInPage', () => {
  it('matches a query split across items joined with spaces', () => {
    const page = items('The quick ', 'brown fox ', 'jumps over ', 'the lazy dog');
    const res = locateInPage(page, 'The quick brown fox jumps over the lazy dog');
    expect(res).not.toBeNull();
    expect(res!.itemIndices).toEqual([0, 1, 2, 3]);
  });

  it('matches a word hyphen-split across adjacent items (tight join, no space)', () => {
    // pdfjs frequently splits a single word into adjacent items with no space.
    const page = items('repre', 'sentation learning improves accuracy');
    const res = locateInPage(page, 'representation learning improves accuracy');
    expect(res).not.toBeNull();
    expect(res!.itemIndices).toEqual([0, 1]);
  });

  it('falls back to letters-only matching when punctuation drifts', () => {
    // Spaced/tight joins carry stray punctuation the query lacks; only the
    // letters-only index lines up.
    const page = items('the, model.', 'achieves; state of', 'the art results');
    const res = locateInPage(page, 'the model achieves state of the art results');
    expect(res).not.toBeNull();
    expect(res!.itemIndices).toEqual([0, 1, 2]);
  });

  it('returns null for a query below the minimum anchor length', () => {
    const page = items('the quick brown fox');
    expect(locateInPage(page, 'too short')).toBeNull();
  });

  it('returns null when the query is not present', () => {
    const page = items('the quick brown fox jumps over the lazy dog');
    expect(locateInPage(page, 'entirely unrelated sentence about something else')).toBeNull();
  });

  it('returns null for an empty page', () => {
    expect(locateInPage([], 'a sufficiently long query string here')).toBeNull();
  });
});

describe('locateAcrossDocument', () => {
  // Each page item is > 150 chars so the start anchor lands wholly on page 1
  // and the end anchor wholly on page 2 (dual-anchor cross-page case).
  it('spans a chunk whose anchors fall on different pages', () => {
    const pages = [
      { pageNumber: 1, items: items(longStart) },
      { pageNumber: 2, items: items(longEnd) },
    ];
    const res = locateAcrossDocument(pages, `${longStart} ${longEnd}`);
    expect(res).not.toBeNull();
    expect(res!.startPage).toBe(1);
    expect(res!.pageHighlights.get(1)).toEqual({ firstItem: 0, lastItem: 0 });
    expect(res!.pageHighlights.get(2)).toEqual({ firstItem: 0, lastItem: 0 });
  });

  it('returns null when the start anchor matches no page', () => {
    const pages = [{ pageNumber: 1, items: items(longStart) }];
    expect(locateAcrossDocument(pages, 'a phrase that does not occur anywhere in the document text at all')).toBeNull();
  });
});

describe('createIncrementalDocumentLocator', () => {
  it('completes a short query on its start page and protects completed results from mutation', () => {
    const locator = createIncrementalDocumentLocator('The quick brown fox jumps over the lazy dog');

    const result = locator.pushPage({
      pageNumber: 4,
      items: items('unrelated preface', 'The quick ', 'brown fox jumps over the lazy dog', 'unrelated tail'),
    });

    expect(result).toEqual({
      startPage: 4,
      pageHighlights: new Map([[4, { firstItem: 1, lastItem: 2 }]]),
    });

    result!.pageHighlights.get(4)!.firstItem = 99;
    result!.pageHighlights.set(99, { firstItem: 99, lastItem: 99 });

    expect(locator.finish()).toEqual({
      startPage: 4,
      pageHighlights: new Map([[4, { firstItem: 1, lastItem: 2 }]]),
    });
    expect(locator.pushPage({ pageNumber: 5, items: items('ignored after completion') })).toEqual({
      startPage: 4,
      pageHighlights: new Map([[4, { firstItem: 1, lastItem: 2 }]]),
    });
  });

  it('completes a long dual-anchor query on one page', () => {
    const locator = createIncrementalDocumentLocator(`${longStart} ${longEnd}`);

    const result = locator.pushPage({
      pageNumber: 2,
      items: items('unrelated preface', longStart, 'content between the anchors', longEnd, 'unrelated tail'),
    });

    expect(result).toEqual({
      startPage: 2,
      pageHighlights: new Map([[2, { firstItem: 1, lastItem: 3 }]]),
    });
  });

  it('completes a cross-page query immediately when its end anchor arrives', () => {
    const locator = createIncrementalDocumentLocator(`${longStart} ${longEnd}`);

    expect(locator.pushPage({
      pageNumber: 3,
      items: items('unrelated preface', longStart, 'remaining text on the start page'),
    })).toBeNull();
    expect(locator.pushPage({
      pageNumber: 4,
      items: items('first middle-page item', 'second middle-page item'),
    })).toBeNull();

    const result = locator.pushPage({
      pageNumber: 5,
      items: items('opening text on the end page', longEnd, 'unrelated tail'),
    });

    expect(result).toEqual({
      startPage: 3,
      pageHighlights: new Map([
        [3, { firstItem: 1, lastItem: 2 }],
        [4, { firstItem: 0, lastItem: 1 }],
        [5, { firstItem: 0, lastItem: 1 }],
      ]),
    });
  });

  it('matches despite punctuation and whitespace drift', () => {
    const locator = createIncrementalDocumentLocator('the model achieves state of the art results');

    const result = locator.pushPage({
      pageNumber: 6,
      items: items('unrelated preface', 'the,\n model.', 'achieves;\tstate of', 'the art results', 'unrelated tail'),
    });

    expect(result).toEqual({
      startPage: 6,
      pageHighlights: new Map([[6, { firstItem: 1, lastItem: 3 }]]),
    });
  });

  it('waits until finish before returning the missing-end fallback', () => {
    const locator = createIncrementalDocumentLocator(`${longStart} ${longEnd}`);

    expect(locator.pushPage({
      pageNumber: 7,
      items: items('unrelated preface', longStart, 'remaining text on the start page'),
    })).toBeNull();
    expect(locator.pushPage({
      pageNumber: 8,
      items: items('the expected end anchor never appears on this page'),
    })).toBeNull();

    expect(locator.finish()).toEqual({
      startPage: 7,
      pageHighlights: new Map([[7, { firstItem: 1, lastItem: 2 }]]),
    });
    expect(locator.finish()).toEqual({
      startPage: 7,
      pageHighlights: new Map([[7, { firstItem: 1, lastItem: 2 }]]),
    });
  });

  it('returns null at EOF when no start anchor matches', () => {
    const locator = createIncrementalDocumentLocator(
      'a phrase that does not occur anywhere in the document text at all',
    );

    expect(locator.pushPage({ pageNumber: 1, items: items('unrelated first page text') })).toBeNull();
    expect(locator.pushPage({ pageNumber: 2, items: items('unrelated second page text') })).toBeNull();
    expect(locator.finish()).toBeNull();
    expect(locator.pushPage({ pageNumber: 3, items: items('ignored after EOF') })).toBeNull();
  });
});
