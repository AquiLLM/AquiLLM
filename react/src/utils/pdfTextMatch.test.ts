import { describe, it, expect } from 'vitest';
import { locateInPage, locateAcrossDocument, type PDFTextItemLike } from './pdfTextMatch';

const items = (...strs: string[]): PDFTextItemLike[] => strs.map((str) => ({ str }));

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
  const pageOne =
    'the transformer architecture relies entirely on self attention mechanisms ' +
    'that allow the model to weigh the relative importance of many different ' +
    'tokens appearing throughout the entire input sequence at once';
  const pageTwo =
    'this particular design choice enables highly parallel computation across ' +
    'the whole sequence and over time has become the single dominant modeling ' +
    'approach for large scale natural language processing systems deployed today';

  it('spans a chunk whose anchors fall on different pages', () => {
    const pages = [
      { pageNumber: 1, items: items(pageOne) },
      { pageNumber: 2, items: items(pageTwo) },
    ];
    const res = locateAcrossDocument(pages, `${pageOne} ${pageTwo}`);
    expect(res).not.toBeNull();
    expect(res!.startPage).toBe(1);
    expect(res!.pageHighlights.get(1)).toEqual({ firstItem: 0, lastItem: 0 });
    expect(res!.pageHighlights.get(2)).toEqual({ firstItem: 0, lastItem: 0 });
  });

  it('returns null when the start anchor matches no page', () => {
    const pages = [{ pageNumber: 1, items: items(pageOne) }];
    expect(locateAcrossDocument(pages, 'a phrase that does not occur anywhere in the document text at all')).toBeNull();
  });
});
