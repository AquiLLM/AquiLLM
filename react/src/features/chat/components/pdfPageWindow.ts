export const MAX_RENDERED_PAGES = 7;

interface RenderedPageWindowInput {
  numPages: number;
  visiblePage: number;
  citationStartPage: number | null;
  initialJumpComplete: boolean;
}

export interface PageMeasurement {
  pageNumber: number;
  top: number;
  height: number;
}

function normalizePageCount(numPages: number): number {
  return Number.isFinite(numPages) ? Math.max(0, Math.floor(numPages)) : 0;
}

function clampPage(pageNumber: number, numPages: number, fallback: number): number {
  const normalized = Number.isFinite(pageNumber) ? Math.round(pageNumber) : fallback;
  return Math.min(numPages, Math.max(1, normalized));
}

function addNeighborhood(
  pages: Set<number>,
  center: number,
  radius: number,
  numPages: number,
): void {
  for (let page = center - radius; page <= center + radius; page += 1) {
    if (page >= 1 && page <= numPages) pages.add(page);
  }
}

export function calculateRenderedPages({
  numPages,
  visiblePage,
  citationStartPage,
  initialJumpComplete,
}: RenderedPageWindowInput): number[] {
  const pageCount = normalizePageCount(numPages);
  if (pageCount === 0) return [];

  const visible = clampPage(visiblePage, pageCount, 1);
  const pages = new Set<number>();
  addNeighborhood(pages, visible, 1, pageCount);

  if (citationStartPage === null || initialJumpComplete) {
    return Array.from(pages).sort((a, b) => a - b);
  }

  const citation = clampPage(citationStartPage, pageCount, visible);
  addNeighborhood(pages, citation, 2, pageCount);
  if (pages.size <= MAX_RENDERED_PAGES) {
    return Array.from(pages).sort((a, b) => a - b);
  }

  const prioritized = Array.from(pages)
    .filter((page) => page !== citation)
    .sort((a, b) => {
      const viewportDistance = Math.abs(a - visible) - Math.abs(b - visible);
      if (viewportDistance !== 0) return viewportDistance;

      const citationDistance = Math.abs(a - citation) - Math.abs(b - citation);
      return citationDistance !== 0 ? citationDistance : a - b;
    });

  return [citation, ...prioritized.slice(0, MAX_RENDERED_PAGES - 1)].sort((a, b) => a - b);
}

/**
 * Return the page nearest the viewport midpoint.
 * Measurements must be sorted by ascending `top` and built outside the scroll hot path.
 */
export function findViewportPage(
  measurements: ReadonlyArray<PageMeasurement>,
  scrollTop: number,
  viewportHeight: number,
): number {
  if (measurements.length === 0) return 1;

  const viewportMidpoint = scrollTop + viewportHeight / 2;
  let low = 0;
  let high = measurements.length - 1;

  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const measurement = measurements[middle];
    const bottom = measurement.top + measurement.height;

    if (viewportMidpoint < measurement.top) {
      high = middle - 1;
    } else if (viewportMidpoint > bottom) {
      low = middle + 1;
    } else {
      return measurement.pageNumber;
    }
  }

  if (high < 0) return measurements[0].pageNumber;
  if (low >= measurements.length) return measurements[measurements.length - 1].pageNumber;

  const before = measurements[high];
  const after = measurements[low];
  const distanceFromBefore = viewportMidpoint - (before.top + before.height);
  const distanceFromAfter = after.top - viewportMidpoint;
  return distanceFromBefore <= distanceFromAfter ? before.pageNumber : after.pageNumber;
}
