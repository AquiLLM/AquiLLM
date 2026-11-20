// @vitest-environment jsdom

import { cleanup, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { MessageBubble } from './MessageBubble';
import fixtures from '../../../../tests/fixtures/scientific-excerpts.json';

vi.mock('../../../main', () => ({ getCsrfCookie: () => 'csrf-token' }));
vi.mock('./CitationModalProvider', () => ({
  useCitationModal: () => ({ openCitation: () => {} }),
}));

beforeEach(() => {
  Object.defineProperty(window, 'apiUrls', { configurable: true, value: {
    api_citation_sources: '/api/citations/sources/',
  } });
  Object.defineProperty(window, 'pageUrls', { configurable: true, value: {
    document: '/document/%(doc_id)s/',
  } });
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ sources: [] })));
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it.each(fixtures)('renders actual backend output literally: $name', async ({ source, published }) => {
  // Python's test_rag_excerpt_dom_fixtures checks these generated outputs against
  // the real fallback renderer. Exercise linkification, Markdown and raw HTML.
  const { container } = render(<MessageBubble
    message={{ role: 'assistant', content: published }}
    onRate={() => {}} onFeedback={() => {}}
  />);
  await waitFor(() => expect(container.querySelector('.rag-citation-link')).not.toBeNull());
  const quote = container.querySelector('blockquote')!;
  for (const line of source.split('\n').filter((line) => line.trim())) {
    expect(quote.textContent).toContain(line.trim());
  }
  for (const link of quote.querySelectorAll('a')) {
    expect(link.getAttribute('data-chunk-id')).toBe('1');
  }
  expect(quote.querySelector('img, script, iframe, pre, code, strong, .katex')).toBeNull();
  await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/citations/sources/',
    expect.objectContaining({ body: JSON.stringify({ chunk_ids: [1] }) })));
});
