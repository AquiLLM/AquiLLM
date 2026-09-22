// @vitest-environment jsdom

import { cleanup, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import MessageSources from './MessageSources';

vi.mock('../../../main', () => ({ getCsrfCookie: () => 'csrf-token' }));
vi.mock('./CitationModalProvider', () => ({
  useCitationModal: () => ({ openCitation: vi.fn() }),
}));

describe('MessageSources', () => {
  beforeEach(() => {
    Object.defineProperty(window, 'apiUrls', {
      configurable: true,
      value: { api_citation_sources: '/api/citations/sources/' },
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('sends the CSRF token when loading citation sources', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          sources: [
            {
              chunk_id: 17,
              doc_id: '11111111-1111-4111-8111-111111111111',
              title: 'Paper A',
              modality: 'text',
            },
          ],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );

    render(
      <MessageSources
        content="Evidence [doc:11111111-1111-4111-8111-111111111111 chunk:17]."
      />,
    );

    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/citations/sources/',
      expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': 'csrf-token',
        },
      }),
    );
  });
});
