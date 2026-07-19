# PDF Citation Performance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make large PDF citations open and highlight responsively while preserving the current side panel, cited-page jump, yellow transform-based highlights, full-document scrolling, image/text citation routing, unavailable state, source links, and pin controls.

**Architecture:** Keep Django and React as the delivery path. Django returns compact citation metadata and streams authorized PDFs. React uses one reusable page-extraction session per mounted PDF, feeds an incremental matcher, and renders a seven-canvas virtual window over full-height page placeholders. Existing APIs remain backward compatible and existing PDFs benefit without reingestion.

**Tech Stack:** Django, Django REST-style `JsonResponse` views, React 19, TypeScript, React-PDF/PDF.js, Vitest, Testing Library, pytest.

**Design reference:** `docs/superpowers/specs/2026-07-18-pdf-citation-performance-design.md`

**Visual acceptance reference:** `C:\Users\jackj\AppData\Local\Temp\codex-clipboard-6456fc42-cc44-4e07-a053-f5a56783053d.png`

**Branch baseline:** `codex/pdf-citation-performance` is based directly on `origin/development` at `3746ddb9`. Do not fix the eight unrelated TypeScript errors present at this baseline.

---

## Chunk 1: Lightweight Django delivery

### Task 1: Add a backward-compatible compact chunk response

**Files:**

- Modify: `aquillm/apps/documents/tests/test_citation_api.py`
- Modify: `aquillm/apps/documents/views/api.py`

- [ ] **Step 1: Write failing compact-response tests**

Extend `test_citation_api.py` with tests that prove:

1. The default request still includes `document.full_text` and `document.text_offset`.
2. `?include_full_text=0` omits both fields.
3. Compact text, image, and PDF-shaped responses retain the current fields: chunk positions, `modality`, `image_url`, document identity/title/type, `has_pdf`, and `source_url`.
4. A compact request has the same 403 and 404 behavior as the default request.

Use the existing `_make_chunk` helper. For the compact image test, give the document a `source_url` and assert both it and `image_url` survive. For the PDF test, override default storage with a temporary `FileSystemStorage`, give `PDFDocument` non-empty `full_text`, and save a small `SimpleUploadedFile`; this prevents `PDFDocument.save()` from trying to parse fixture bytes or contacting the configured S3 backend. The failing phase must fail on compact-contract assertions, not fixture setup.

- [ ] **Step 2: Run the focused tests and confirm the new assertions fail**

Run from the repository root:

```powershell
python -m pytest aquillm/apps/documents/tests/test_citation_api.py -q
```

Expected: compact-response tests fail because `full_text` and `text_offset` are still present.

- [ ] **Step 3: Implement the compact contract without calculating the text window**

In `chunk_detail`, parse the flag once:

```python
include_full_text = request.GET.get("include_full_text", "1").lower() not in {
    "0",
    "false",
    "no",
}
```

Build the document payload with only the summary fields first:

```python
document_payload = {
    "id": str(doc.id),
    "title": doc.title,
    "type": doc.__class__.__name__,
    "has_pdf": has_pdf,
    "source_url": getattr(doc, "source_url", None),
}
```

Only read, window, and attach `doc.full_text` when `include_full_text` is true. Preserve the current 500,000-character window and `text_offset` behavior for callers that omit the parameter. Return the unchanged top-level chunk fields and `document_payload`.

- [ ] **Step 4: Re-run the focused tests**

```powershell
python -m pytest aquillm/apps/documents/tests/test_citation_api.py -q
```

Expected: all citation API tests pass.

- [ ] **Step 5: Commit the API contract**

```powershell
git add aquillm/apps/documents/tests/test_citation_api.py aquillm/apps/documents/views/api.py
git commit -m "perf: add compact citation metadata response"
```

### Task 2: Stream authorized S3/MinIO PDF bodies with private caching

**Files:**

- Create: `aquillm/apps/documents/tests/test_pdf_response.py`
- Modify: `aquillm/apps/documents/views/pages.py`

- [ ] **Step 1: Write failing PDF response tests**

Create a Django test module with two storage paths.

For a fake S3/MinIO-shaped storage, provide a fake SDK body whose `read(size)` records each read. Use a `PDFDocument` with a stored field name and assert:

- the response is a `StreamingHttpResponse`;
- requesting only the first item from `response.streaming_content` reads only the first object chunk, not the whole object;
- consuming the remainder yields the original bytes and closes the object body;
- calling `response.close()` before consuming a chunk still closes the object body;
- `Content-Length` and available S3 `ETag`/`Last-Modified` validators are forwarded;
- `Content-Type`, a safely encoded inline `Content-Disposition`, and `Cache-Control: private, max-age=300` match the filesystem path;
- a missing-key `ClientError` becomes 404;
- a zero-length object closes its body and becomes 404;
- an unrelated `ClientError` is not misreported as a missing file.

For a temporary `FileSystemStorage`, create an authorized `PDFDocument` containing known bytes and assert:

- the authorized response is a `FileResponse`;
- `response.streaming` is true;
- `Content-Type` is `application/pdf`;
- `Content-Disposition` is inline and contains a stable filename;
- `Cache-Control` contains `private` and `max-age=300`;
- joining `response.streaming_content` yields the original bytes;
- a document without a file returns 404;
- a missing storage object returns 404;
- a zero-byte file returns 404;
- a user without collection view permission remains denied.

Add one authorized `RawTextDocument.rendered_pdf` case to cover the alternate field name used for crawled web citations. Give `PDFDocument` fixtures non-empty `full_text` or call `save(skip_text_extraction=True)` so setup never invokes PDF parsing.

Close each streaming response after consuming it so the temporary file can be cleaned up on Windows.

- [ ] **Step 2: Run the tests and confirm streaming assertions fail**

```powershell
python -m pytest aquillm/apps/documents/tests/test_pdf_response.py -q
```

Expected: the response is currently buffered and lacks both incremental object-body delivery and the private cache contract.

- [ ] **Step 3: Add an incremental S3 object-body path and a filesystem fallback**

In `pages.py`, keep `get_doc` and the route unchanged. Import `ClientError`, `S3Storage`, `clean_name`, Django's `content_disposition_header`, and `http_date`. Add small private helpers so the storage-specific behavior is isolated and unit-testable. Use a closeable iterator object, not a generator, because closing an unstarted generator does not execute its `finally` block:

```python
PDF_STREAM_CHUNK_SIZE = 64 * 1024

class _ObjectBodyIterator:
    def __init__(self, body):
        self.body = body

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.body.read(PDF_STREAM_CHUNK_SIZE)
        if chunk:
            return chunk
        self.close()
        raise StopIteration

    def close(self):
        if self.body is not None:
            self.body.close()
            self.body = None
```

For `isinstance(pdf_field.storage, S3Storage)`, mirror django-storages' lookup exactly:

```python
key = storage._normalize_name(clean_name(pdf_field.name))
object_response = storage.bucket.Object(key).get()
```

Catch `botocore.exceptions.ClientError` and translate it to `Http404` only when `error.response["ResponseMetadata"]["HTTPStatusCode"] == 404`; re-raise other errors. Reject a zero `ContentLength`, closing `Body` before raising. Build `StreamingHttpResponse(_ObjectBodyIterator(body), content_type="application/pdf")`; Django registers the iterator's `close()` as a response resource closer. Forward `ContentLength`, `ETag`, and `LastModified` when present, formatting the date with `http_date`.

For other storages, check size, open the field, and use:

```python
FileResponse(
    pdf_field,
    as_attachment=False,
    filename=Path(pdf_field.name).name,
    content_type="application/pdf",
)
```

Apply the same `Content-Disposition` generated by `content_disposition_header(False, Path(pdf_field.name).name)` and `Cache-Control: private, max-age=300` contract to both paths. Do not add a partial HTTP Range implementation in this pass; the design intentionally defers complete Range semantics.

- [ ] **Step 4: Re-run the response tests**

```powershell
python -m pytest aquillm/apps/documents/tests/test_pdf_response.py -q
```

Expected: all PDF response tests pass.

- [ ] **Step 5: Commit the streaming response**

```powershell
git add aquillm/apps/documents/tests/test_pdf_response.py aquillm/apps/documents/views/pages.py
git commit -m "perf: stream private PDF responses"
```

### Task 3: Stop document-subtype lookup after the first match

**Files:**

- Create: `aquillm/apps/documents/tests/test_text_chunk_document.py`
- Modify: `aquillm/apps/documents/models/chunks.py`

- [ ] **Step 1: Write a failing short-circuit test**

Patch `_get_descended_from_document` to return three fake model classes whose managers record calls. Make the second manager return a sentinel document. Assert `chunk.document` returns the sentinel and the third manager is never called. Add a second test asserting that no match still raises the existing `ValidationError`.

- [ ] **Step 2: Run the model tests and confirm the short-circuit assertion fails**

```powershell
python -m pytest aquillm/apps/documents/tests/test_text_chunk_document.py -q
```

Expected: the third manager is called by the current loop.

- [ ] **Step 3: Return immediately on the first matching subtype**

Replace the accumulator with:

```python
for document_type in _get_descended_from_document():
    document = document_type.objects.filter(id=self.doc_id).first()
    if document:
        return document
raise ValidationError(f"TextChunk {self.pk} is not associated with a document!")
```

- [ ] **Step 4: Run the model and citation API tests**

```powershell
python -m pytest aquillm/apps/documents/tests/test_text_chunk_document.py aquillm/apps/documents/tests/test_citation_api.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the lookup cleanup**

```powershell
git add aquillm/apps/documents/tests/test_text_chunk_document.py aquillm/apps/documents/models/chunks.py
git commit -m "perf: short circuit chunk document lookup"
```

## Chunk 2: Frontend request and extraction foundations

### Task 4: Split citation summary/detail types and preserve development routing

**Files:**

- Modify: `react/package.json`
- Modify: `react/package-lock.json`
- Modify: `react/vitest.config.ts`
- Modify: `react/src/features/chat/components/citationTypes.ts`
- Create: `react/src/features/chat/components/CitationModalProvider.test.tsx`
- Modify: `react/src/features/chat/components/CitationModalProvider.tsx`
- Modify: `react/src/features/chat/components/PDFCitationModal.tsx`
- Modify: `react/src/features/chat/components/ImageCitationModal.tsx`
- Modify: `react/src/features/chat/components/TextCitationModal.tsx`

- [ ] **Step 1: Enable focused React component tests**

From `react/`, install the test-only DOM dependencies:

```powershell
npm install --save-dev @testing-library/react jsdom
```

Update `vitest.config.ts` so `include` accepts both `src/**/*.test.ts` and `src/**/*.test.tsx`. Keep the default environment `node`; put `// @vitest-environment jsdom` at the top of component test files only.

- [ ] **Step 2: Write failing dispatcher component tests**

Mock the four lazy-loaded modal modules, `formatUrl`, and `fetch`. Render `CitationModalProvider` plus a small harness that calls `openCitation`. Cover:

- a compact PDF response makes one `chunk_detail?include_full_text=0` request, mounts the PDF modal, and passes `pinned`/`onTogglePin`;
- a compact image response makes one request, mounts the image modal, and retains `modality`, `image_url`, and `source_url`;
- a compact non-PDF text response makes exactly one second request without the compact parameter, then mounts the text modal with required `full_text`/`text_offset`;
- a 404 from either compact or full-detail request mounts `CitationUnavailable`;
- non-404 failures retain the current friendly error route;
- toggling the provider-owned pin state continues to reach every modal as props.
- in development mode, a click records `aquillm:citation:open-start` and compact completion records `aquillm:citation:compact-ready` plus the `aquillm:citation:compact-metadata` measure.

Restore `window.apiUrls`, `window.localStorage`, and `globalThis.fetch` between tests.

- [ ] **Step 3: Run the test and confirm the compact routing cases fail**

```powershell
cd react
npx vitest run src/features/chat/components/CitationModalProvider.test.tsx
```

Expected: the current provider requests the full detail once and cannot distinguish summary/detail loading.

- [ ] **Step 4: Introduce explicit summary and detail contracts**

In `citationTypes.ts`, define:

```typescript
export interface CitationDocumentSummary {
  id: string;
  title: string;
  type: string;
  has_pdf: boolean;
  source_url: string | null;
}

export interface CitationChunkSummary {
  content: string;
  chunk_number: number;
  start_position: number;
  end_position: number;
  start_time: number | null;
  modality: string;
  image_url: string | null;
  document: CitationDocumentSummary;
}

export interface CitationChunkDetail extends CitationChunkSummary {
  document: CitationDocumentSummary & {
    full_text: string;
    text_offset: number;
  };
}
```

Type PDF and image modal `preloadedChunk` props as `CitationChunkSummary`; keep the text modal on `CitationChunkDetail`.

- [ ] **Step 5: Fetch compact metadata first and full text only for text citations**

In `CitationDispatcher`:

1. Fetch the formatted detail URL with `include_full_text=0` using `URL`/`URLSearchParams` so existing query strings are preserved.
2. Route image and PDF summaries immediately.
3. For a non-PDF text summary, perform exactly one full-detail fetch, replace the loading state only after it resolves, and pass the detail to `TextCitationModal`.
4. Use one cancellation guard/abort controller across both stages.
5. Preserve `CitationUnavailable`, the transient error fallback, source data, and provider-owned pin props.

Render image, PDF, and text branches explicitly rather than assigning the lazy components to one `Modal` union. This lets TypeScript prove that only the text branch receives `CitationChunkDetail` while the other branches receive `CitationChunkSummary`.

Behind `import.meta.env.DEV` and a `typeof performance !== 'undefined'` guard, clear the prior citation marks when a new target opens, mark `aquillm:citation:open-start`, mark `aquillm:citation:compact-ready` when the compact response is accepted, and measure `aquillm:citation:compact-metadata` between them. Instrumentation must not alter production behavior.

Do not put pin state into either API type.

- [ ] **Step 6: Re-run component and existing unit tests**

```powershell
npx vitest run src/features/chat/components/CitationModalProvider.test.tsx
npm test
npm run typecheck
```

Expected: dispatcher tests and the existing PDF/text matching tests pass. Typecheck reports only the eight recorded `development` baseline errors and no error in the citation files or new tests.

- [ ] **Step 7: Commit compact frontend routing**

```powershell
git add package.json package-lock.json vitest.config.ts src/features/chat/components/citationTypes.ts src/features/chat/components/CitationModalProvider.test.tsx src/features/chat/components/CitationModalProvider.tsx src/features/chat/components/PDFCitationModal.tsx src/features/chat/components/ImageCitationModal.tsx src/features/chat/components/TextCitationModal.tsx
git commit -m "perf: load compact citation metadata first"
```

### Task 5: Make PDF text matching incremental without changing semantics

**Files:**

- Modify: `react/src/utils/pdfTextMatch.test.ts`
- Modify: `react/src/utils/pdfTextMatch.ts`

- [ ] **Step 1: Add parity and early-completion tests**

Add tests for a `createIncrementalDocumentLocator(query)` API. Feed pages one at a time and assert explicit golden `startPage` plus page-to-item ranges for:

- a single-page short query;
- a same-page dual-anchor query;
- a cross-page query;
- punctuation/whitespace drift;
- a missing end anchor;
- no start anchor.

Keep the legacy `locateAcrossDocument` assertions too, but do not use parity between two functions as the only oracle because Step 3 makes them share an implementation. Also assert `pushPage` returns a complete match immediately when the end anchor arrives, while `finish()` preserves the explicit start-item-to-end-of-start-page fallback only at end-of-document.

- [ ] **Step 2: Run the matcher tests and confirm the new API is missing**

```powershell
cd react
npx vitest run src/utils/pdfTextMatch.test.ts
```

Expected: compilation/test failure because the incremental locator is not implemented.

- [ ] **Step 3: Implement one stateful locator**

Export an `IncrementalDocumentLocator` interface with `pushPage(page)` and `finish()` methods. Move start-anchor, end-anchor, and page-range accumulation state into `createIncrementalDocumentLocator`.

Rules:

- before the start anchor, each page is indexed once and discarded from locator state;
- after the start anchor, retain the indexed pages needed to construct cross-page ranges;
- short queries complete on the start page;
- long queries complete only when the end anchor is found;
- `finish()` emits the existing start-to-end-of-start-page fallback when only the start anchor was found;
- calls after completion return the same immutable logical result.

Refactor `locateAcrossDocument` to instantiate the incremental locator, push its input pages, return on completion, and otherwise call `finish()`. This keeps one matching implementation.

- [ ] **Step 4: Run all matcher tests**

```powershell
npx vitest run src/utils/pdfTextMatch.test.ts src/utils/textOffsetMatch.test.ts
```

Expected: all matcher tests pass without changing legacy results.

- [ ] **Step 5: Commit the incremental matcher**

```powershell
git add src/utils/pdfTextMatch.test.ts src/utils/pdfTextMatch.ts
git commit -m "perf: match PDF citations incrementally"
```

### Task 6: Add a single-flight PDF page extraction session

**Files:**

- Create: `react/src/features/chat/components/pdfPageExtraction.test.ts`
- Create: `react/src/features/chat/components/pdfPageExtraction.ts`

- [ ] **Step 1: Write failing extraction-session tests**

Use a fake `PDFDocumentProxy`/page shape with counters around `getPage()` and `getTextContent()`. Test that:

- two callers asking for the same page share one promise;
- a sequential `find(query)` stops after a complete match;
- the session yields between sequential pages via an injected `yieldToBrowser` function;
- a narrow query arriving later replays cached scans and does not duplicate PDF.js calls;
- aborting one query stops its state updates but does not evict or cancel shared page promises;
- after abort, no later page number is scheduled or extracted, while an already-shared in-flight page promise is allowed to settle and remain cached;
- `ensurePage(pageNumber)` extracts an approached, previously unscanned page through the same cache;
- end-of-document invokes the matcher's missing-end fallback.

- [ ] **Step 2: Run the tests and confirm the module is missing**

```powershell
cd react
npx vitest run src/features/chat/components/pdfPageExtraction.test.ts
```

Expected: module-not-found failure.

- [ ] **Step 3: Implement the extraction session**

Export the PDF text item and `PageScan` types currently local to `PDFCitationModal`. Define the injection point and session API exactly:

```typescript
interface PdfPageExtractionSession {
  getCached(pageNumber: number): PageScan | undefined;
  ensurePage(pageNumber: number): Promise<PageScan>;
  find(query: string, signal?: AbortSignal): Promise<DocumentMatchResult | null>;
  subscribe(listener: (scan: PageScan) => void): () => void;
}

type YieldToBrowser = () => Promise<void>;

export function createPdfPageExtractionSession(
  pdfDoc: PDFDocumentProxy,
  pageWidth: number,
  options: { yieldToBrowser?: YieldToBrowser } = {},
): PdfPageExtractionSession;
```

Use a `Map<number, Promise<PageScan>>` as the single-flight source of truth. `ensurePage` must cache the promise before awaiting it. `find` must:

1. create an incremental locator;
2. walk page numbers in ascending order, checking `signal.aborted` before scheduling every page;
3. await `ensurePage`, check the signal again, and only then push the scan into the locator;
4. return as soon as the locator completes;
5. yield between newly extracted pages, then check the signal again before continuing;
6. never request a later page after abort, but never evict or cancel a page promise already shared by another caller;
7. call `finish()` only after the last page.

The default `yieldToBrowser` returns a promise resolved by one `requestAnimationFrame` callback when rAF exists; otherwise it uses one `setTimeout(resolve, 0)`. It registers only one mechanism, so there is no losing timer/callback to clean up. Unit tests inject a deterministic promise-returning spy.

Publish each successfully extracted `PageScan` to current subscribers exactly once, regardless of how many callers awaited its promise. `subscribe` returns an unsubscribe function and does not replay old scans; consumers call `getCached` to seed initial state. Add subscriber assertions to Step 1, including unsubscribe and no duplicate publication.

The session lifetime is the mounted `PDFDocumentProxy`; changing documents creates a new session.

- [ ] **Step 4: Run extraction and matcher tests**

```powershell
npx vitest run src/features/chat/components/pdfPageExtraction.test.ts src/utils/pdfTextMatch.test.ts
```

Expected: all tests pass and counter assertions show at-most-once extraction per page.

- [ ] **Step 5: Commit the extraction session**

```powershell
git add src/features/chat/components/pdfPageExtraction.test.ts src/features/chat/components/pdfPageExtraction.ts
git commit -m "perf: cache PDF page extraction promises"
```

## Chunk 3: Seven-canvas virtual PDF viewer

### Task 7: Define and test the bounded page-window policy

**Files:**

- Create: `react/src/features/chat/components/pdfPageWindow.test.ts`
- Create: `react/src/features/chat/components/pdfPageWindow.ts`

- [ ] **Step 1: Write failing window-calculator tests**

Cover these exact invariants:

- `MAX_RENDERED_PAGES` is seven;
- initial rendering includes the citation start page and prefers up to two pages on either side;
- the visible page and one page on either side are prioritized;
- a union larger than seven remains capped and keeps the citation page until the initial highlighted scroll completes;
- after the initial scroll, pages nearest the viewport win;
- page numbers are clamped to `[1, numPages]`, unique, and sorted;
- moving the viewport over a long citation eventually exposes each later cited page without mounting the full citation range;
- viewport-page calculation works from placeholder top/height measurements.

- [ ] **Step 2: Run the focused test and confirm the module is missing**

```powershell
cd react
npx vitest run src/features/chat/components/pdfPageWindow.test.ts
```

Expected: module-not-found failure.

- [ ] **Step 3: Implement pure window helpers**

Export:

```typescript
export const MAX_RENDERED_PAGES = 7;

export function calculateRenderedPages(input: {
  numPages: number;
  visiblePage: number;
  citationStartPage: number | null;
  initialJumpComplete: boolean;
}): number[];

export function findViewportPage(
  measurements: Array<{ pageNumber: number; top: number; height: number }>,
  scrollTop: number,
  viewportHeight: number,
): number;
```

Keep these functions independent of React and DOM globals.

- [ ] **Step 4: Run the focused tests**

```powershell
npx vitest run src/features/chat/components/pdfPageWindow.test.ts
```

Expected: all window-policy tests pass.

- [ ] **Step 5: Commit the page-window policy**

```powershell
git add src/features/chat/components/pdfPageWindow.test.ts src/features/chat/components/pdfPageWindow.ts
git commit -m "perf: bound the PDF render window"
```

### Task 8: Render a full-height virtual page list with both visibility paths

**Files:**

- Create: `react/src/features/chat/components/VirtualizedPdfPages.test.tsx`
- Create: `react/src/features/chat/components/VirtualizedPdfPages.tsx`

- [ ] **Step 1: Write failing virtual-list component tests**

Use `// @vitest-environment jsdom`, mock React-PDF's `Page`, and supply controlled page measurements. Assert:

- there is one lightweight placeholder per PDF page;
- no more than seven mocked `<Page>` components are mounted;
- every mounted page receives `renderTextLayer={false}` and `renderAnnotationLayer={false}`;
- each placeholder keeps its height while its canvas is unmounted;
- an `IntersectionObserver` rooted at the scroll container updates the visible page;
- with `IntersectionObserver` deleted, a request-animation-frame-throttled scroll handler calls the same window calculator;
- approaching an unscanned page requests it through `ensurePage`;
- every placeholder has a nonzero estimated height and the complete document scroll range before citation matching resolves;
- scans completed by a background `find()` reach the list through the extraction-session subscription and replace estimated heights;
- replacing an estimated height above the viewport adjusts `scrollTop` by the same delta;
- page-height updates do not remove the overall document scroll range.

Stub `requestAnimationFrame` and restore browser globals after each test.

- [ ] **Step 2: Run the component test and confirm the module is missing**

```powershell
cd react
npx vitest run src/features/chat/components/VirtualizedPdfPages.test.tsx
```

Expected: module-not-found failure.

- [ ] **Step 3: Implement the virtual list**

The component must:

1. render one wrapper/placeholder for every page;
2. mount `<Page>` only for `calculateRenderedPages(...)` results;
3. set both React-PDF layers to false;
4. initialize every unknown height to `pageWidth * (11 / 8.5)`, seed any existing scans through `getCached`, then subscribe to new scans and replace estimates with `scan.viewport.height`;
5. use `IntersectionObserver` when available;
6. always register the rAF-throttled scroll fallback when the observer is unavailable;
7. call the extraction session for a visible page without starting a second extraction path;
8. expose page/highlight data attributes needed for scrolling and tests.

Keep highlight rendering as absolutely positioned children of the page wrapper so its transform-derived coordinates remain relative to the rendered canvas.

When a measured height replaces an estimate for a placeholder wholly above the current viewport, add the height delta to `scrollContainer.scrollTop`. Unsubscribe from the extraction session on session change/unmount. Seed cached scans before listening and immediately check the cache again after subscribing so a scan completing across that boundary is not lost.

- [ ] **Step 4: Re-run the virtual-list and window tests**

```powershell
npx vitest run src/features/chat/components/VirtualizedPdfPages.test.tsx src/features/chat/components/pdfPageWindow.test.ts
```

Expected: both suites pass, including the seven-page cap and no-observer fallback.

- [ ] **Step 5: Commit the virtual page list**

```powershell
git add src/features/chat/components/VirtualizedPdfPages.test.tsx src/features/chat/components/VirtualizedPdfPages.tsx
git commit -m "perf: virtualize PDF citation pages"
```

### Task 9: Integrate cached matching, virtualization, and existing highlights

**Files:**

- Create: `react/src/features/chat/components/PDFCitationModal.test.tsx`
- Modify: `react/src/features/chat/components/PDFCitationModal.tsx`

- [ ] **Step 1: Write failing modal integration tests**

Mock `react-pdf`, the extraction session factory, and the narrow endpoint. Use a 30-page fixture whose citation is on page 19, matching the supplied acceptance screenshot. Assert:

- PDF loading and citation narrowing begin without waiting on one another;
- the full-chunk query begins immediately after the PDF proxy is ready;
- a narrowed quote reuses the same extraction session and never restarts `getTextContent()` for cached pages;
- full and narrow results resolving in either order always select the successful narrow result; if it targets a different page after the full result jumped, the modal promotes and scrolls the narrow page too;
- locating page 19 promotes its placeholder, mounts its page, then scrolls the first yellow highlight into view;
- the header reports `highlighted on page 19`;
- the yellow overlay uses the existing transform-derived rectangles and `data-citation-hit` marker;
- the rendered React-PDF page count never exceeds seven;
- scrolling near page 30 mounts it and unmounts distant pages while all 30 placeholders remain;
- the source link, external document link, close button, Escape behavior, and pin control still work;
- PDF load and no-match errors retain their current fallbacks;
- switching directly from one PDF citation to another resets the old proxy/session/geometry/jump state and ignores late results from the first document.
- in development mode, the modal records `aquillm:citation:pdf-ready`, `aquillm:citation:first-match`, `aquillm:citation:canvas-mounted`, and `aquillm:citation:highlight-scrolled`, plus matching duration measures from `aquillm:citation:open-start`.

- [ ] **Step 2: Run the modal integration test and confirm it fails**

```powershell
cd react
npx vitest run src/features/chat/components/PDFCitationModal.test.tsx
```

Expected: the current modal scans all pages, re-runs on `narrowQuote`, and mounts all 30 pages.

- [ ] **Step 3: Replace the all-pages scan with one extraction session**

In `PDFCitationModal`:

1. Reset `pdfDoc`, `numPages`, the extraction session, full/narrow matches, page geometry, highlights, and jump state when `docId` or `pdfUrl` changes; key the React-PDF `<Document>` by `pdfUrl` and guard async publication with a document generation token.
2. Create the extraction session with `useMemo` only when the current generation's `pdfDoc` changes.
3. Start a full-chunk `find(chunk.content)` with an abort controller and store its result in `fullMatch` only.
4. When a non-empty narrow quote arrives, start `find(narrowQuote)` against the same session and store its result in `narrowMatch` only.
5. Derive `preferredMatch = narrowMatch ?? fullMatch`; never let promise resolution order write directly to the displayed match.
6. Give each query a generation identity so a result from an old document or query cannot publish. Abort query publication on close/document change without clearing the current session's promise cache.
7. Store `PageScan` data by page for highlight-rectangle calculation.
8. Remove `scannedPagesRef` and the all-document extraction loop.

If the preferred match's start page or serialized page ranges change, set `initialJumpComplete=false`, clear the prior one-shot scroll marker, and run the promote/mount/highlight/scroll sequence for the new preferred match. This covers a full result that jumps before a later, more precise narrow result resolves on another page.

Add a development-only helper local to the citation components (or a small exported helper if tests need it) that safely calls `performance.mark`/`performance.measure`. Record:

- `aquillm:citation:pdf-ready` in `onDocumentLoad` and measure `aquillm:citation:pdf-load`;
- `aquillm:citation:first-match` once for the first successful preferred match and measure `aquillm:citation:match`;
- `aquillm:citation:canvas-mounted` when the preferred start page reports its React-PDF page render success and measure `aquillm:citation:canvas`;
- `aquillm:citation:highlight-scrolled` immediately after the highlighted element is scrolled and measure `aquillm:citation:highlight`.

All measures start at `aquillm:citation:open-start`. Component tests stub `performance.mark`/`measure` and assert the milestone order; repeated narrow-result jumps may update marks but must not duplicate the first-match measure.

- [ ] **Step 4: Replace the all-pages map with `VirtualizedPdfPages`**

Pass the number of pages, extraction session, current match, page width, and highlight-render callback to the virtual list. Preserve the current PDF transform math and yellow style. Remove the React-PDF text-layer CSS import once `renderTextLayer={false}` is enforced.

When a match arrives:

1. set the citation start page and highlighted ranges;
2. let the virtual window mount the start page;
3. wait until a `[data-citation-hit]` descendant exists on that page;
4. call the existing centered `scrollIntoView` once;
5. mark the initial jump complete so viewport pages can fully control the window.

For a citation spanning more than seven pages, retain all `pageHighlights` metadata but mount each later highlighted page only when the viewport reaches it.

- [ ] **Step 5: Run all frontend tests**

```powershell
npx vitest run src/features/chat/components/PDFCitationModal.test.tsx
npm test
```

Expected: all frontend tests pass; the modal fixture has 30 placeholders and at most seven mounted pages.

- [ ] **Step 6: Run the production build**

```powershell
npm run build
```

Expected: Vite production build succeeds.

- [ ] **Step 7: Check the TypeScript baseline without expanding scope**

```powershell
npm run typecheck
```

Expected at the recorded development baseline: eight unrelated errors in `ChatFileUpload.tsx`, `SearchPage.tsx`, `useCollectionViewMoveBatch.ts`, `FileSystemViewer.tsx`, and `uiUtils.ts`. Confirm this change introduces no additional files/errors; do not fix those baseline issues in this branch.

- [ ] **Step 8: Commit the modal integration**

```powershell
git add src/features/chat/components/PDFCitationModal.test.tsx src/features/chat/components/PDFCitationModal.tsx
git commit -m "perf: integrate virtual PDF citation viewer"
```

### Task 10: Verify the end-to-end acceptance behavior

**Files:**

- Modify only if an assertion gap is found in the files above.

- [ ] **Step 1: Run the complete focused verification set**

From the repository root:

```powershell
python -m pytest aquillm/apps/documents/tests/test_citation_api.py aquillm/apps/documents/tests/test_pdf_response.py aquillm/apps/documents/tests/test_text_chunk_document.py -q
cd react
npm test
npm run build
```

Expected: Django tests, all Vitest suites, and the Vite build pass.

- [ ] **Step 2: Start the repository's local stack**

From the repository root, use the documented no-GPU development compose file so Django, the built React assets, PostgreSQL, and MinIO use the same paths as the feature:

```powershell
docker compose -f deploy/compose/no_gpu_dev.yml up -d --build
docker compose -f deploy/compose/no_gpu_dev.yml ps
```

Expected: `web`, `db`, `storage`, `redis`, and the supporting services are healthy/running; the application responds at `http://localhost:8080` (or the `.env` `PORT`).

- [ ] **Step 3: Manually verify the screenshot citation**

Open `http://localhost:8080`, sign in, and use the existing `2504.19874v1` / Chunk 28 citation from the supplied screenshot. If that development database does not contain it, upload `https://arxiv.org/pdf/2504.19874`, wait for ingestion, and use a citation containing “Near Neighbour Search Experiments” so the target is around page 19. Click the citation and confirm:

- the side panel opens with its existing loading state;
- the cited page and yellow highlight appear and the header reports the page;
- surrounding pages remain visible and the whole document remains scrollable;
- image citations, text citations, unavailable citations, source links, and pinning still behave as on `development`;
- the browser stays responsive while page text is being located.

In DevTools, record:

```javascript
({
  canvases: document.querySelectorAll('.react-pdf__Page canvas').length,
  textLayers: document.querySelectorAll('.react-pdf__Page__textContent').length,
  placeholders: document.querySelectorAll('[data-pdf-page-placeholder]').length,
  measures: performance.getEntriesByType('measure')
    .filter((entry) => entry.name.startsWith('aquillm:citation:'))
    .map(({ name, duration }) => ({ name, duration })),
})
```

Expected: canvas count is at most seven, text-layer count is zero, placeholder count equals the PDF page count, and measures include `compact-metadata`, `pdf-load`, `match`, `canvas`, and `highlight` with finite nonnegative durations. Scroll to the end of the PDF and run the DOM query again; later canvases mount, distant canvases unmount, and the count remains at most seven.

- [ ] **Step 4: Confirm the performance invariants with the 30-page fixture**

Verify from test spies or development-only marks that:

- each page's `getTextContent()` is called at most once per mounted PDF document;
- the full-chunk scan stops after the complete citation range is found;
- an arriving narrow quote reuses cached pages;
- compact PDF metadata omits `full_text` and does not scale with document length;
- the authenticated PDF response begins streaming and carries `private, max-age=300`.

- [ ] **Step 5: Inspect the final diff and history**

```powershell
git diff --check origin/development...HEAD
git status --short
git log --oneline origin/development..HEAD
```

Expected: no whitespace errors, no unintended files, and small task-oriented commits matching this plan.

- [ ] **Step 6: Request code review before integration**

Use `superpowers:requesting-code-review` against `origin/development`, address any correctness findings, then use `superpowers:finishing-a-development-branch` to present merge/PR options.

## Deferred follow-ups

Do not add these to the first implementation unless measurement shows the low-hanging changes are insufficient and the user expands scope:

- a storage-aware HTTP Range proxy or short-lived S3/MinIO URL;
- ingestion-time chunk-to-page hints and backfill;
- sharing a `PDFDocumentProxy` across independently mounted citation panels;
- cross-click text-index persistence;
- Nginx or object-storage deployment changes.
