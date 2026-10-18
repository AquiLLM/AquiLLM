// @vitest-environment jsdom

import { cleanup, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { MessageBubble } from './MessageBubble';
import fixtures from '../../../../tests/fixtures/model-presentation.json';

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

it.each(fixtures)('renders the publication boundary output safely: $name', async ({
  published, visible, hasCitation, hasMath,
}) => {
  // The backend companion test finalizes each fixture's draft and checks its
  // exact published value. Keep real citation linkification and Markdown here:
  // inserted citation HTML can otherwise close an incomplete authored tag.
  const { container } = render(<MessageBubble
    message={{ role: 'assistant', content: published }}
    onRate={() => {}} onFeedback={() => {}}
  />);
  const bubble = container.querySelector('.markdown-cell')!;
  await waitFor(() => expect(bubble.textContent).toContain(visible));
  expect(bubble.querySelector('img, iframe, script, style, object, embed')).toBeNull();
  expect(bubble.querySelector('[hidden], [style*="display:none"], [style*="display: none"]')).toBeNull();
  expect(bubble.textContent).not.toContain('unacquired.example');
  expect(Boolean(bubble.querySelector('.katex'))).toBe(hasMath);
  expect(Boolean(bubble.querySelector('.rag-citation-link'))).toBe(hasCitation);
  for (const link of bubble.querySelectorAll('a')) {
    expect(link.getAttribute('data-doc-id')).toBe('12345678-1234-1234-1234-123456789abc');
    expect(link.getAttribute('data-chunk-id')).toBe('1');
  }
});
