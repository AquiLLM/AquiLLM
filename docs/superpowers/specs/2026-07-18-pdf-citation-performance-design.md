# PDF Citation Performance Design

## Summary

Clicking a chat citation must continue to open the side panel, locate the cited passage, scroll to the cited PDF page, and draw the existing transform-based yellow highlight over the source text. The implementation will reduce lag by avoiding full-document canvas and text-layer rendering, making text discovery progressive and non-restarting, and removing avoidable backend buffering and payload work.

The screenshot supplied by the user is the visual acceptance criterion: the cited page and nearby PDF content remain fully rendered and scrollable, the header continues to report the highlighted page, and the citation overlay remains aligned with the rendered glyphs.

## Goals

- Keep the current citation-panel interaction and highlight fidelity.
- Keep the entire PDF browsable from the panel.
- Bound the number of mounted PDF canvases independently of total page count.
- Stop scanning once the citation's complete page range is known.
- Keep the browser responsive while text extraction is in progress.
- Ensure an arriving LLM-narrowed quote reuses extracted page text instead of restarting PDF work.
- Reduce click-time metadata payloads for PDFs.
- Stream S3/MinIO PDF object bodies incrementally instead of spooling the complete object before Django yields its first response chunk.
- Improve existing PDFs immediately, without reingestion.

## Non-goals for the first pass

- A storage-specific HTTP Range proxy or Nginx/MinIO deployment changes.
- Persisting page hints during ingestion or backfilling existing documents.
- Sharing a `PDFDocumentProxy` across independently mounted React panels.
- Changing the citation syntax, LLM narrowing prompt, or highlight appearance.
- Fixing unrelated pre-existing TypeScript errors.

These remain possible follow-ups after measuring the first-pass improvements.

## Current bottlenecks

1. `CitationDispatcher` waits for `chunk_detail` before mounting a viewer, while `chunk_detail` includes up to 500,000 characters of `full_text` even for PDFs.
2. The PDF page view passes an S3-backed `FieldFile` to `HttpResponse`, which materializes iterable content. Replacing it with `FileResponse` alone is insufficient because django-storages may first spool the complete S3/MinIO object into a temporary file.
3. `PDFCitationModal` extracts text from every page before invoking the document locator.
4. It mounts a React-PDF `<Page>` for every page at once. Each page creates a canvas and, by default, a text layer.
5. `narrowQuote` is a dependency of the scan effect. A quote arriving mid-scan cancels partial work and starts again from page one.
6. Closing the panel discards all page scan state.
7. `TextChunk.document` continues querying every concrete document table after it has found the matching document.

## Proposed architecture

### 1. Compact citation metadata

`chunk_detail` will accept an `include_full_text` query parameter that defaults to the current behavior for compatibility. The citation dispatcher will request `include_full_text=0`. For a PDF citation the response will contain the chunk, title, type, PDF availability, and offsets, but omit `document.full_text`.

The frontend will model these as two explicit contracts:

- `CitationChunkSummary`, returned by the compact request, contains all existing top-level chunk fields and document metadata except `full_text` and `text_offset`.
- `CitationChunkDetail` extends the summary with required `document.full_text` and `document.text_offset` fields.

The summary retains the `development` branch's `modality`, `image_url`, and `source_url` fields. Pin state remains provider-owned UI state, separate from the API response, and continues flowing to each modal as props. If `modality=image`, the dispatcher mounts `ImageCitationModal` directly with the summary. Otherwise, if `has_pdf=true`, it mounts the PDF modal with the summary; those modals only need `chunk.content` and document identity/title. If the summary describes a non-PDF text citation, the dispatcher performs exactly one full-detail request and mounts `TextCitationModal` only after receiving `CitationChunkDetail`. A 404 at either stage still mounts `CitationUnavailable`; other failures use the existing citation error state. Existing API consumers that omit `include_full_text` continue receiving the full response.

### 2. Streaming PDF response

The authenticated PDF view will keep the existing URL and authorization. For the configured django-storages S3/MinIO backend, it will obtain the object body and return a `StreamingHttpResponse` whose closeable iterator pulls fixed-size chunks from the SDK `StreamingBody`; closing the response closes the body even if iteration never starts. Local/test filesystem storage will use `FileResponse`. Both paths return inline PDF content, `Cache-Control: private, max-age=300`, a safe stable filename, and available `Content-Length`/validator headers. This avoids waiting for django-storages to spool the entire object before the first response chunk. A missing storage object or a zero-byte file returns HTTP 404 through the same user-facing PDF load failure path.

Full byte-range proxying is intentionally deferred. The configured S3 endpoint is internal, so redirecting browsers to a presigned URL would require deployment/CORS changes, while a correct Range proxy needs conditional requests and complete range semantics. The incremental object-body adapter improves time-to-first-byte without pretending to provide Range support; the frontend changes are still expected to remove the dominant lag in this pass.

### 3. Progressive, single-flight page extraction

The modal will own one extraction session per loaded PDF. The session will:

- extract pages in ascending order;
- publish each completed page to query-specific locators;
- yield to the event loop between pages;
- retain extracted text items and viewports for reuse during the mounted panel lifetime;
- stop extracting once the active citation locator has a complete match, unless the user scrolls to an unscanned page;
- stop scheduling new pages and ignore stale results after the modal closes or the document changes, while allowing an already-shared page promise to settle into the session cache.

The text matcher will gain a stateful incremental document locator. It will preserve the existing dual-anchor semantics, including citations that cross page boundaries. The existing `locateAcrossDocument` function will be expressed through the same stateful implementation so legacy and incremental matching share one behavior.

If an LLM-narrowed quote arrives during extraction, a second locator will replay already-extracted pages in memory and then subscribe to subsequent pages. It will not cancel or repeat `pdfDoc.getPage()` or `getTextContent()` calls. The full-chunk locator remains available as the fallback.

Full-chunk and narrow results are stored separately. The derived preferred result is a successful narrow result when available, otherwise the full result; promise completion order never changes that priority. If a later narrow result moves the preferred citation to a different page or range, the pending highlighted-jump state resets so the new target is promoted, mounted, and scrolled. Document/citation generations prevent either result from publishing after a direct switch to another PDF.

### 4. Bounded page rendering

The document scroll area will keep one lightweight placeholder per PDF page, preserving the full scroll range. At most seven React-PDF `<Page>` canvases may be mounted at once. The window calculator prioritizes:

- the currently visible page and one page on either side;
- the citation start page and up to two pages on either side while the initial jump is pending.

If those sets exceed seven pages, pages nearest the current viewport win after retaining the citation start page until the first highlighted scroll completes. A citation spanning more than seven pages does not mount its complete range simultaneously: its highlight metadata is retained, and later highlighted pages mount normally when the user scrolls to them. This preserves cross-page highlighting without violating the canvas cap.

An `IntersectionObserver` rooted at the panel scroll container will normally update the visible window. A required request-animation-frame-throttled scroll handler will calculate visible placeholder indices from their offsets when `IntersectionObserver` is unavailable. Both paths use the same seven-page window calculator. Distant canvases unmount, while their placeholders keep the scroll position stable, so every page remains reachable and renders when approached. Before page metadata is available, every placeholder uses a nonzero letter-page estimate based on the fixed render width. The extraction session publishes each completed page viewport to subscribers; exact heights replace estimates as scans finish. When a height above the current viewport changes, the scroll container is adjusted by the same delta so visible content does not jump.

Both `renderTextLayer` and `renderAnnotationLayer` will be false. The existing highlight rectangles already use PDF text transforms and the page viewport, so disabling the text layer does not change selection or highlight geometry.

When a citation match is found, the target placeholder will first be promoted into the rendered window. Scrolling occurs only after its highlight overlay mounts, preserving the current centered-scroll behavior shown in the acceptance screenshot.

### 5. Small backend cleanup

`TextChunk.document` will return as soon as it finds the matching concrete document. This reduces the compact metadata request from eight document-table lookups to the number required to reach the matching subtype.

## Data flow

1. The user clicks a citation and the side panel opens immediately with its existing loading state.
2. The dispatcher requests compact chunk metadata.
3. The dispatcher preserves the `development` branch's image/PDF/text routing, deleted-citation state, and pin controls. For PDFs, the modal starts the authenticated PDF request and the citation-narrow request in parallel.
4. PDF.js loads the document; the virtual page list creates placeholders without mounting every canvas.
5. The single extraction session feeds pages to the full-chunk locator and, when available, the narrow-quote locator.
6. Once a complete match is known, the cited page window mounts, highlight rectangles are computed, and the panel scrolls to the first rectangle.
7. Scrolling through the PDF mounts nearby pages and unmounts distant pages. Missing page text is extracted on demand through the same session/cache.

## Error and fallback behavior

- A compact metadata failure keeps the existing friendly error and document-page link.
- A PDF load failure keeps the existing fallback link.
- If the narrow quote fails or cannot be located, the full chunk remains the locator input.
- If no complete match is found by end-of-document, the current start-anchor fallback is retained.
- If `IntersectionObserver` is unavailable, the required throttled scroll-position fallback updates the same bounded seven-page window; the app never mounts every page and never leaves an approached page as a permanent blank placeholder.
- Cancellation prevents state updates after closing or switching citations, and changing `docId`/PDF URL resets the old proxy, extraction session, geometry, matches, and pending jump before the new document can publish. Cached page promises are never duplicated within one mounted document.

## Testing strategy

### Python

- Verify `chunk_detail?include_full_text=0` omits `document.full_text` while the default response remains compatible.
- In Python, verify compact image citations retain `modality`, `image_url`, and `source_url`; verify compact PDF citations retain `source_url`; and verify compact/full requests preserve their HTTP status behavior.
- In TypeScript component tests, verify non-PDF text citations receive one follow-up full-detail request, compact or full-detail 404 selects `CitationUnavailable`, routing selects the correct modal, and provider-owned pin props continue flowing to every modal.
- Verify the S3-like PDF response yields its first chunk without reading the complete object, the filesystem fallback remains streaming, `rendered_pdf` remains supported, and both paths are inline, private-cacheable, and authorized.
- Verify `TextChunk.document` stops querying after a match.

### TypeScript unit tests

- Verify incremental matching produces the same single-page, cross-page, short-query, missing-end, and not-found results as `locateAcrossDocument`.
- Verify replaying previously scanned pages for a newly arrived narrow quote does not call page extraction again.
- Verify the rendered page-window calculation never exceeds seven pages, retains the citation start page until the initial jump, prioritizes viewport pages afterward, and exposes later pages of a long cross-page citation as the user scrolls.

### UI/build verification

- Use a multi-page PDF fixture and assert that the number of mounted canvases remains bounded while the placeholder count equals the document page count.
- Assert the panel reports the cited page, scrolls the highlighted element into view, and retains a highlight overlay after virtualization.
- Verify scrolling mounts later pages and removes distant canvases without losing the document scroll range.
- Run the production Vite build. The repository's existing unrelated TypeScript errors are recorded as baseline and are not part of this change.

## Performance acceptance criteria

- Mounted PDF canvases remain at or below seven for documents and citation ranges of any size.
- React-PDF text-layer nodes are absent from the citation panel.
- Each PDF page's `getTextContent()` runs at most once per mounted document.
- A matching page stops sequential extraction once the complete citation range is known.
- An LLM narrow result never restarts extraction from page one.
- PDF metadata responses do not scale with `document.full_text` size.
- The highlighted page and surrounding PDF remain visually equivalent to the supplied screenshot.

## Rollout and follow-up measurement

The change requires no data migration and applies to existing citations immediately. Add development-only performance marks around compact metadata completion, PDF document load, first match, cited-page canvas mount, and highlight scroll. Compare large-document timings before deciding whether page-hint ingestion, a storage-aware Range proxy, or cross-click PDF proxy caching is necessary.
