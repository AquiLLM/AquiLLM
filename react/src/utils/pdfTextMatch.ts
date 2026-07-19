/**
 * Locate an LLM-supplied citation snippet inside a PDF.js text layer.
 *
 * Strategy:
 *   1. Build three indexed views of the page's text-layer items:
 *        - "tight"   = items joined with no separator (matches PDF text
 *                      flow where pdfjs splits a sentence into adjacent
 *                      items with no space between them)
 *        - "spaced"  = items joined with a single space (matches PDF text
 *                      flow where there is whitespace between items)
 *        - "letters" = only [a-z0-9], for whitespace/punctuation drift
 *      All normalised: NFKC, collapsed whitespace, smart quotes/dashes
 *      unified to ASCII, soft hyphens stripped, lowercased.
 *   2. Locate the chunk's *start anchor* (first ~150 chars) in the page.
 *   3. Locate the chunk's *end anchor* (last ~150 chars) in the page,
 *      searching from at-or-after the start anchor's items.
 *   4. Return every item index in [start, end] inclusive — that's the
 *      full extent of the chunk on this page.
 *
 * If only the start anchor matches, highlight just that anchor (chunk
 * may span pages or partially fail to match further on).
 */

export interface PDFTextItemLike {
  str: string;
}

export interface PDFMatchResult {
  /** Indices into `items[]` that overlap the matched range. */
  itemIndices: number[];
}

const SOFT_HYPHEN = '­';
const SMART_QUOTES = /[‘’‚‛]/g;
const SMART_DQUOTES = /[“”„‟]/g;
const SMART_DASHES = /[‐‑‒–—―]/g;

const MIN_ANCHOR_LEN = 20;
const MAX_ANCHOR_LEN = 150;

function normalise(s: string): string {
  return s
    .normalize('NFKC')
    .replace(new RegExp(SOFT_HYPHEN, 'g'), '')
    .replace(SMART_QUOTES, "'")
    .replace(SMART_DQUOTES, '"')
    .replace(SMART_DASHES, '-')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

function lettersOnly(s: string): string {
  return normalise(s).replace(/[^a-z0-9]+/g, '');
}

/** Pick the leading anchor: first N word-bounded chars of the query. */
function pickStartAnchor(query: string): string {
  const trimmed = query.trim();
  if (trimmed.length <= MAX_ANCHOR_LEN) return trimmed;
  const slice = trimmed.slice(0, MAX_ANCHOR_LEN);
  const lastSpace = slice.lastIndexOf(' ');
  return lastSpace > MAX_ANCHOR_LEN / 2 ? slice.slice(0, lastSpace) : slice;
}

/** Pick the trailing anchor: last N word-bounded chars of the query. */
function pickEndAnchor(query: string): string {
  const trimmed = query.trim();
  if (trimmed.length <= MAX_ANCHOR_LEN) return trimmed;
  const slice = trimmed.slice(trimmed.length - MAX_ANCHOR_LEN);
  const firstSpace = slice.indexOf(' ');
  return firstSpace >= 0 && firstSpace < MAX_ANCHOR_LEN / 2
    ? slice.slice(firstSpace + 1)
    : slice;
}

interface ItemRange {
  start: number;
  end: number;
}

interface PageIndex {
  itemCount: number;
  tightText: string;
  tightRanges: ItemRange[];
  spacedText: string;
  spacedRanges: ItemRange[];
  letters: string;
  letterIdxToItem: number[];
}

function buildPageIndex(items: PDFTextItemLike[]): PageIndex {
  const tightRanges: ItemRange[] = [];
  const spacedRanges: ItemRange[] = [];
  const letterIdxToItem: number[] = [];
  let tight = '';
  let spaced = '';
  let letters = '';

  for (let i = 0; i < items.length; i++) {
    const piece = normalise(items[i].str);

    const tightStart = tight.length;
    tight += piece;
    tightRanges.push({ start: tightStart, end: tight.length });

    const spacedStart = spaced.length;
    spaced += piece;
    spacedRanges.push({ start: spacedStart, end: spaced.length });
    if (i < items.length - 1) spaced += ' ';

    const lettersPiece = piece.replace(/[^a-z0-9]+/g, '');
    for (let k = 0; k < lettersPiece.length; k++) {
      letters += lettersPiece[k];
      letterIdxToItem.push(i);
    }
  }

  return {
    itemCount: items.length,
    tightText: tight,
    tightRanges,
    spacedText: spaced,
    spacedRanges,
    letters,
    letterIdxToItem,
  };
}

interface AnchorMatch {
  /** First item index covered by the match (0-indexed). */
  firstItem: number;
  /** Last item index covered by the match (0-indexed, inclusive). */
  lastItem: number;
}

function rangeToItemSpan(ranges: ItemRange[], hitStart: number, hitEnd: number): AnchorMatch | null {
  let first = -1;
  let last = -1;
  for (let i = 0; i < ranges.length; i++) {
    const r = ranges[i];
    if (r.start < hitEnd && r.end > hitStart) {
      if (first === -1) first = i;
      last = i;
    }
  }
  return first === -1 ? null : { firstItem: first, lastItem: last };
}

/**
 * Locate an anchor in a built PageIndex.
 *
 * `minItemIndex` lets the caller require the match to be at or after a
 * particular item (useful for the end-anchor pass: find the second
 * anchor at or after the start anchor's last item).
 */
function findAnchor(
  idx: PageIndex,
  rawAnchor: string,
  minItemIndex = 0,
): AnchorMatch | null {
  const needle = normalise(rawAnchor);
  if (needle.length < MIN_ANCHOR_LEN) return null;

  // Compute starting offsets corresponding to minItemIndex.
  const tightFrom = minItemIndex > 0 && minItemIndex < idx.tightRanges.length
    ? idx.tightRanges[minItemIndex].start
    : 0;
  const spacedFrom = minItemIndex > 0 && minItemIndex < idx.spacedRanges.length
    ? idx.spacedRanges[minItemIndex].start
    : 0;

  const tightHit = idx.tightText.indexOf(needle, tightFrom);
  if (tightHit >= 0) {
    const span = rangeToItemSpan(idx.tightRanges, tightHit, tightHit + needle.length);
    if (span) return span;
  }

  const spacedHit = idx.spacedText.indexOf(needle, spacedFrom);
  if (spacedHit >= 0) {
    const span = rangeToItemSpan(idx.spacedRanges, spacedHit, spacedHit + needle.length);
    if (span) return span;
  }

  // Letters-only fallback. Find the letters-onky position that corresponds
  // to minItemIndex (first letterIdx whose mapped item >= minItemIndex).
  const lettersNeedle = lettersOnly(rawAnchor);
  if (lettersNeedle.length >= MIN_ANCHOR_LEN) {
    let lettersFrom = 0;
    if (minItemIndex > 0) {
      while (lettersFrom < idx.letterIdxToItem.length && idx.letterIdxToItem[lettersFrom] < minItemIndex) {
        lettersFrom++;
      }
    }
    const lettersHit = idx.letters.indexOf(lettersNeedle, lettersFrom);
    if (lettersHit >= 0) {
      const first = idx.letterIdxToItem[lettersHit];
      const last = idx.letterIdxToItem[lettersHit + lettersNeedle.length - 1];
      if (first !== undefined && last !== undefined) {
        return { firstItem: first, lastItem: last };
      }
    }
  }

  return null;
}

/**
 * Try to locate `query` inside the text layer of one PDF page.
 * Returns null if no match passes the length / fidelity gates.
 *
 * Implementation: dual-anchor matching. Anchors the first and last ~150
 * chars of the query separately and highlights every item between (and
 * including) them, so the full extent of the chunk is highlighted —
 * not just the leading portion.
 */
export function locateInPage(
  items: PDFTextItemLike[],
  query: string,
): PDFMatchResult | null {
  if (items.length === 0) return null;
  const trimmed = query.trim();
  if (trimmed.length < MIN_ANCHOR_LEN) return null;

  const idx = buildPageIndex(items);

  const startAnchor = pickStartAnchor(trimmed);
  const startMatch = findAnchor(idx, startAnchor);
  if (!startMatch) return null;

  // Short queries: start anchor already covers the whole thing.
  if (trimmed.length <= MAX_ANCHOR_LEN) {
    return collectItemIndices(startMatch.firstItem, startMatch.lastItem);
  }

  const endAnchor = pickEndAnchor(trimmed);
  // Search for end anchor at or after the start anchor's last item.
  const endMatch = findAnchor(idx, endAnchor, startMatch.lastItem);

  const firstItem = startMatch.firstItem;
  const lastItem = endMatch
    ? Math.max(startMatch.lastItem, endMatch.lastItem)
    : startMatch.lastItem;
  return collectItemIndices(firstItem, lastItem);
}

export interface PageHighlightRange {
  firstItem: number;
  lastItem: number;
}

export interface DocumentMatchResult {
  /** First page (1-indexed) that has any highlighted items — scroll target. */
  startPage: number;
  /** Map of pageNumber → inclusive item-index range to highlight on that page. */
  pageHighlights: Map<number, PageHighlightRange>;
}

interface DocumentPage {
  pageNumber: number;
  items: PDFTextItemLike[];
}

interface IndexedPage {
  pageNumber: number;
  index: PageIndex;
}

interface DocumentAnchorMatch extends IndexedPage {
  match: AnchorMatch;
}

export interface IncrementalDocumentLocator {
  /** Add the next extracted page, returning as soon as the full range is known. */
  pushPage(page: DocumentPage): DocumentMatchResult | null;
  /** Signal EOF and return the missing-end fallback when only the start matched. */
  finish(): DocumentMatchResult | null;
}

class StatefulIncrementalDocumentLocator implements IncrementalDocumentLocator {
  private readonly startAnchor: string;
  private readonly endAnchor: string | null;
  private startInfo: DocumentAnchorMatch | null = null;
  private readonly pagesFromStart: IndexedPage[] = [];
  private completed = false;
  private completedResult: DocumentMatchResult | null = null;

  constructor(query: string) {
    const trimmed = query.trim();
    this.startAnchor = pickStartAnchor(trimmed);
    this.endAnchor = trimmed.length > MAX_ANCHOR_LEN ? pickEndAnchor(trimmed) : null;

    if (trimmed.length < MIN_ANCHOR_LEN) {
      this.completed = true;
    }
  }

  pushPage(page: DocumentPage): DocumentMatchResult | null {
    if (this.completed) return cloneDocumentMatchResult(this.completedResult);

    const indexedPage: IndexedPage = {
      pageNumber: page.pageNumber,
      index: buildPageIndex(page.items),
    };

    if (!this.startInfo) {
      const startMatch = findAnchor(indexedPage.index, this.startAnchor);
      if (!startMatch) return null;

      this.startInfo = { ...indexedPage, match: startMatch };
      this.pagesFromStart.push(indexedPage);

      if (!this.endAnchor) {
        return this.complete({
          startPage: page.pageNumber,
          pageHighlights: new Map([[page.pageNumber, {
            firstItem: startMatch.firstItem,
            lastItem: startMatch.lastItem,
          }]]),
        });
      }

      const endMatch = findAnchor(indexedPage.index, this.endAnchor, startMatch.lastItem);
      if (endMatch) {
        return this.complete(this.buildCompletedResult({ ...indexedPage, match: endMatch }));
      }
      return null;
    }

    if (page.pageNumber < this.startInfo.pageNumber) return null;

    this.pagesFromStart.push(indexedPage);
    const endMatch = findAnchor(indexedPage.index, this.endAnchor!, 0);
    if (!endMatch) return null;

    return this.complete(this.buildCompletedResult({ ...indexedPage, match: endMatch }));
  }

  finish(): DocumentMatchResult | null {
    if (this.completed) return cloneDocumentMatchResult(this.completedResult);

    if (!this.startInfo) return this.complete(null);

    return this.complete({
      startPage: this.startInfo.pageNumber,
      pageHighlights: new Map([[this.startInfo.pageNumber, {
        firstItem: this.startInfo.match.firstItem,
        lastItem: this.startInfo.index.itemCount - 1,
      }]]),
    });
  }

  private buildCompletedResult(endInfo: DocumentAnchorMatch): DocumentMatchResult {
    const startInfo = this.startInfo!;
    const pageHighlights = new Map<number, PageHighlightRange>();

    if (endInfo.pageNumber === startInfo.pageNumber) {
      pageHighlights.set(startInfo.pageNumber, {
        firstItem: startInfo.match.firstItem,
        lastItem: Math.max(startInfo.match.lastItem, endInfo.match.lastItem),
      });
      return { startPage: startInfo.pageNumber, pageHighlights };
    }

    pageHighlights.set(startInfo.pageNumber, {
      firstItem: startInfo.match.firstItem,
      lastItem: startInfo.index.itemCount - 1,
    });
    for (const page of this.pagesFromStart) {
      if (page.pageNumber > startInfo.pageNumber && page.pageNumber < endInfo.pageNumber) {
        pageHighlights.set(page.pageNumber, {
          firstItem: 0,
          lastItem: page.index.itemCount - 1,
        });
      }
    }
    pageHighlights.set(endInfo.pageNumber, {
      firstItem: 0,
      lastItem: endInfo.match.lastItem,
    });

    return { startPage: startInfo.pageNumber, pageHighlights };
  }

  private complete(result: DocumentMatchResult | null): DocumentMatchResult | null {
    this.completed = true;
    this.completedResult = result;
    return cloneDocumentMatchResult(result);
  }
}

function cloneDocumentMatchResult(result: DocumentMatchResult | null): DocumentMatchResult | null {
  if (!result) return null;

  return {
    startPage: result.startPage,
    pageHighlights: new Map(
      Array.from(result.pageHighlights, ([pageNumber, range]) => [pageNumber, { ...range }]),
    ),
  };
}

export function createIncrementalDocumentLocator(query: string): IncrementalDocumentLocator {
  return new StatefulIncrementalDocumentLocator(query);
}

/**
 * Locate a chunk across an entire document and return per-page highlight
 * ranges. Handles the case where a chunk spans page boundaries:
 *
 *  - start page  → from startAnchor.firstItem through end of page items
 *  - middle pages → every item
 *  - end page    → from item 0 through endAnchor.lastItem
 *  - single page → from startAnchor.firstItem through endAnchor.lastItem
 *
 * If only the start anchor matches (e.g. extraction drift), highlights
 * extend to the end of the start page only.
 */
export function locateAcrossDocument(
  pages: DocumentPage[],
  query: string,
): DocumentMatchResult | null {
  const locator = createIncrementalDocumentLocator(query);
  for (const page of pages) {
    const result = locator.pushPage(page);
    if (result) return result;
  }
  return locator.finish();
}

function collectItemIndices(first: number, last: number): PDFMatchResult {
  const itemIndices: number[] = [];
  for (let i = first; i <= last; i++) itemIndices.push(i);
  return { itemIndices };
}

/**
 * Locate `query` across multiple pages' text contents. Returns the first
 * page (1-indexed) that produces a match plus the matching item indices,
 * or null if no page matches.
 */
export function locateInDocument(
  pages: Array<{ pageNumber: number; items: PDFTextItemLike[] }>,
  query: string,
): { pageNumber: number; itemIndices: number[] } | null {
  for (const page of pages) {
    const m = locateInPage(page.items, query);
    if (m) return { pageNumber: page.pageNumber, itemIndices: m.itemIndices };
  }
  return null;
}
