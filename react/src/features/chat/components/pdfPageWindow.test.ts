import { describe, expect, it } from 'vitest';

import {
  MAX_RENDERED_PAGES,
  calculateRenderedPages,
  findViewportPage,
} from './pdfPageWindow';

describe('calculateRenderedPages', () => {
  it('caps the render window at seven pages', () => {
    expect(MAX_RENDERED_PAGES).toBe(7);

    const pages = calculateRenderedPages({
      numPages: 100,
      visiblePage: 80,
      citationStartPage: 20,
      initialJumpComplete: false,
    });

    expect(pages.length).toBeLessThanOrEqual(MAX_RENDERED_PAGES);
  });

  it('keeps the citation page and its two-page neighborhood while the initial jump is pending', () => {
    expect(
      calculateRenderedPages({
        numPages: 30,
        visiblePage: 19,
        citationStartPage: 19,
        initialJumpComplete: false,
      }),
    ).toEqual([17, 18, 19, 20, 21]);
  });

  it('retains the citation start but otherwise favors pages nearest the viewport when the union is too large', () => {
    const pages = calculateRenderedPages({
      numPages: 100,
      visiblePage: 80,
      citationStartPage: 20,
      initialJumpComplete: false,
    });

    expect(pages).toContain(20);
    expect(pages).toEqual([19, 20, 21, 22, 79, 80, 81]);
  });

  it('lets the viewport fully control the window after the initial jump', () => {
    expect(
      calculateRenderedPages({
        numPages: 100,
        visiblePage: 80,
        citationStartPage: 20,
        initialJumpComplete: true,
      }),
    ).toEqual([79, 80, 81]);
  });

  it('clamps page numbers at document boundaries and returns sorted unique pages', () => {
    expect(
      calculateRenderedPages({
        numPages: 3,
        visiblePage: -50,
        citationStartPage: 99,
        initialJumpComplete: false,
      }),
    ).toEqual([1, 2, 3]);
  });

  it('normalizes invalid and fractional document inputs before building a window', () => {
    expect(
      calculateRenderedPages({
        numPages: Number.POSITIVE_INFINITY,
        visiblePage: 1,
        citationStartPage: 1,
        initialJumpComplete: false,
      }),
    ).toEqual([]);
    expect(
      calculateRenderedPages({
        numPages: 3.8,
        visiblePage: Number.NaN,
        citationStartPage: Number.POSITIVE_INFINITY,
        initialJumpComplete: false,
      }),
    ).toEqual([1, 2, 3]);
  });

  it('exposes later pages of a long citation as the viewport advances', () => {
    const visited = new Set<number>();

    for (let visiblePage = 19; visiblePage <= 28; visiblePage += 1) {
      const pages = calculateRenderedPages({
        numPages: 30,
        visiblePage,
        citationStartPage: 19,
        initialJumpComplete: true,
      });
      expect(pages.length).toBeLessThanOrEqual(MAX_RENDERED_PAGES);
      pages.forEach((page) => visited.add(page));
    }

    for (let page = 19; page <= 28; page += 1) {
      expect(visited.has(page)).toBe(true);
    }
  });
});

describe('findViewportPage', () => {
  const measurements = [
    { pageNumber: 1, top: 0, height: 700 },
    { pageNumber: 2, top: 716, height: 700 },
    { pageNumber: 3, top: 1432, height: 700 },
    { pageNumber: 4, top: 2148, height: 700 },
  ];

  it('selects the page containing the viewport midpoint', () => {
    expect(findViewportPage(measurements, 800, 600)).toBe(2);
    expect(findViewportPage(measurements, 1500, 400)).toBe(3);
  });

  it('selects the nearest page when the midpoint falls in a gap', () => {
    expect(findViewportPage(measurements, 406, 600)).toBe(1);
  });

  it('handles page edges and equal-distance gap ties deterministically', () => {
    expect(findViewportPage(measurements, 400, 600)).toBe(1);
    expect(findViewportPage(measurements, 416, 600)).toBe(2);
    expect(findViewportPage(measurements, 408, 600)).toBe(1);
  });

  it('clamps before and after the measured document', () => {
    expect(findViewportPage(measurements, -1000, 100)).toBe(1);
    expect(findViewportPage(measurements, 9000, 100)).toBe(4);
  });

  it('returns page one when measurements are not available yet', () => {
    expect(findViewportPage([], 0, 600)).toBe(1);
  });
});
