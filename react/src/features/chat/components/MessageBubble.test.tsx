// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MessageBubble } from './MessageBubble';

const openCitation = vi.hoisted(() => vi.fn());
vi.mock('../../../main', () => ({ getCsrfCookie: () => 'csrf-token' }));
vi.mock('./CitationModalProvider', () => ({
  useCitationModal: () => ({ openCitation }),
}));

const docA = '11111111-1111-4111-8111-111111111111';
const docB = '22222222-2222-4222-8222-222222222222';
const cite = (doc: string, chunk: number) => `[doc:${doc} chunk:${chunk}]`;
const bubble = (content: string) => (
  <MessageBubble message={{ role: 'assistant', content, message_uuid: 'answer-1' }}
    onRate={() => {}} onFeedback={() => {}} />
);
const response = (sources: unknown[]) => new Response(JSON.stringify({ sources }), {
  headers: { 'Content-Type': 'application/json' },
});

beforeEach(() => {
  Object.defineProperty(window, 'apiUrls', { configurable: true, value: {
    api_citation_sources: '/api/citations/sources/',
  } });
  Object.defineProperty(window, 'pageUrls', { configurable: true, value: {
    document: '/document/%(doc_id)s/',
  } });
  openCitation.mockClear();
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('readable message citations', () => {
  it('shares one authenticated title lookup with the footer and keeps exact passage navigation', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response([
      { doc_id: docA, chunk_id: 17, title: 'HSC survey overview', modality: 'text' },
      { doc_id: docA, chunk_id: 23, title: 'HSC survey overview', modality: 'text' },
      { doc_id: docB, chunk_id: 42, title: 'Photometric redshift estimates', modality: 'image' },
    ]));
    const content = `Evidence ${cite(docA, 17)} and ${cite(docB, 42)}. More ${cite(docA, 23)}.\n\nSources:\n- ${cite(docA, 17)}`;
    const { container, rerender } = render(bubble(content));
    const links = await screen.findAllByRole('link', { name: /HSC survey overview/ });
    expect(links).toHaveLength(2);
    expect(screen.queryByText('Sources:')).toBeNull();
    expect(links[0].getAttribute('href')).toBe(`/document/${docA}/?chunk=17`);
    expect(links[1].getAttribute('href')).toBe(`/document/${docA}/?chunk=23`);
    expect(container.textContent).not.toContain(docA);
    expect(container.textContent).not.toContain('chunk:');
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith('/api/citations/sources/', expect.objectContaining({
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': 'csrf-token' },
      body: JSON.stringify({ chunk_ids: [17, 42, 23] }),
    }));

    fireEvent.click(links[1]);
    expect(openCitation).toHaveBeenLastCalledWith({ docId: docA, chunkId: '23', messageUuid: 'answer-1' });
    openCitation.mockClear();
    for (const modifiers of [{ ctrlKey: true }, { metaKey: true }, { shiftKey: true }, { button: 1 }]) {
      expect(fireEvent.click(links[0], modifiers)).toBe(true);
    }
    expect(openCitation).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Sources · 2 documents · 3 passages' }));
    fireEvent.click(screen.getByRole('button', { name: 'HSC survey overview · Passage 2' }));
    expect(openCitation).toHaveBeenLastCalledWith({ docId: docA, chunkId: '23', messageUuid: 'answer-1' });
    expect(container.textContent).not.toContain('chunk 23');
    rerender(bubble(content + ' More streamed prose.'));
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it('uses stable source numbers while loading and when metadata is unavailable', async () => {
    let resolve!: (value: Response) => void;
    vi.spyOn(globalThis, 'fetch').mockReturnValue(new Promise((r) => { resolve = r; }));
    render(bubble(`${cite(docA, 17)} ${cite(docB, 42)} ${cite(docA, 23)}`));
    expect(screen.getAllByRole('link', { name: /Source 1/ })).toHaveLength(2);
    expect(screen.getByRole('link', { name: /Source 2/ })).not.toBeNull();
    await act(async () => { resolve(new Response(null, { status: 503 })); });
    const link = screen.getAllByRole('link', { name: /Source 1/ })[1];
    fireEvent.click(link);
    expect(openCitation).toHaveBeenLastCalledWith({ docId: docA, chunkId: '23', messageUuid: 'answer-1' });
  });

  it('keeps citations found only in the trailing list available in the dropdown', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(response([
      { doc_id: docA, chunk_id: 17, title: 'HSC survey overview', modality: 'text' },
      { doc_id: docB, chunk_id: 42, title: 'Additional source', modality: 'text' },
    ]));
    render(bubble(`Evidence ${cite(docA, 17)}.\n\n**Sources:**\n- ${cite(docA, 17)}\n- ${cite(docB, 42)}`));
    await screen.findByRole('link', { name: /HSC survey overview/ });
    expect(screen.getAllByRole('link')).toHaveLength(1);
    expect(screen.queryByText('Sources:')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Sources · 2 documents · 2 passages' }));
    fireEvent.click(screen.getByRole('button', { name: 'Additional source · Passage 1' }));
    expect(openCitation).toHaveBeenLastCalledWith({ docId: docB, chunkId: '42', messageUuid: 'answer-1' });
  });

  it('renders source titles as text, never as HTML or markdown', async () => {
    const title = '<img src=x onerror="alert(1)"> & [a paper](https://example.com)';
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(response([
      { doc_id: docA, chunk_id: 17, title, modality: 'text' },
    ]));
    const { container } = render(bubble(cite(docA, 17)));
    const link = await screen.findByRole('link', { name: `${title} · Passage 1` });
    expect(link.textContent).toContain(title);
    expect(container.querySelector('img[src="x"]')).toBeNull();
    expect(screen.getAllByRole('link')).toHaveLength(1);
  });

  it('does not label a citation with metadata belonging to a different document', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response([
      { doc_id: docB, chunk_id: 17, title: 'Wrong document', modality: 'text' },
      { doc_id: docB, chunk_id: 42, title: '  ', modality: 'text' },
    ]));
    render(bubble(`${cite(docA, 17)} ${cite(docB, 42)}`));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    await act(async () => {});
    expect(screen.queryByText('Wrong document')).toBeNull();
    expect(screen.getByRole('link', { name: /Source 1/ })).not.toBeNull();
    expect(screen.getByRole('link', { name: /Source 2/ })).not.toBeNull();
  });

  it('ignores stale title responses after the cited message changes', async () => {
    let resolveOld!: (value: Response) => void;
    const fetchMock = vi.spyOn(globalThis, 'fetch')
      .mockReturnValueOnce(new Promise((r) => { resolveOld = r; }))
      .mockResolvedValueOnce(response([
        { doc_id: docB, chunk_id: 42, title: 'Current paper', modality: 'text' },
      ]));
    const { rerender } = render(bubble(cite(docA, 17)));
    rerender(bubble(cite(docB, 42)));
    await screen.findByRole('link', { name: /Current paper/ });
    await act(async () => { resolveOld(response([
      { doc_id: docA, chunk_id: 17, title: 'Old paper', modality: 'text' },
    ])); });
    expect(screen.queryByText('Old paper')).toBeNull();
    expect(screen.getByRole('link', { name: /Current paper/ })).not.toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
