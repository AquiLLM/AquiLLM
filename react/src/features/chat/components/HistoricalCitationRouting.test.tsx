// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { CitationModalProvider, CitationPanelSlot, useCitationModal } from './CitationModalProvider';

vi.mock('./PDFCitationModal', () => ({ default: () => <div>Historical PDF</div> }));
vi.mock('./ImageCitationModal', () => ({ default: () => <div>Historical image</div> }));
vi.mock('./TextCitationModal', () => ({ default: () => <div>Historical text</div> }));
const summary = { content: 'caption', chunk_number: 0, start_position: 0, end_position: 7,
  start_time: null, modality: 'text', source_context_status: 'verified', image_url: null,
  document: { id: 'doc-1', title: 'Saved source', type: 'DocumentFigure', has_pdf: false, source_url: null } };

function Open() {
  const { openCitation } = useCitationModal();
  return <><button onClick={() => openCitation({ docId: 'doc-1', chunkId: '42' })}>Open</button><CitationPanelSlot /></>;
}

beforeEach(() => {
  window.apiUrls = { api_chunk_detail: '/chunk/%(chunk_id)s' } as typeof window.apiUrls;
  vi.stubGlobal('fetch', vi.fn());
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

it.each(['image', 'pdf', 'unavailable'])('routes the saved %s binding without requiring image modality', async kind => {
  const payload = { ...summary, image_url: kind === 'image' ? '/document_image/doc-1/?chunk_id=42' : null,
    source_context_status: kind === 'unavailable' ? 'unavailable' : 'verified',
    document: { ...summary.document, has_pdf: kind === 'pdf' } };
  vi.mocked(fetch).mockResolvedValue(new Response(JSON.stringify(payload), { status: 200 }));
  render(<CitationModalProvider><Open /></CitationModalProvider>);
  fireEvent.click(screen.getByText('Open'));
  expect(await screen.findByText(kind === 'image' ? 'Historical image' : kind === 'pdf' ? 'Historical PDF' : 'Historical text')).toBeTruthy();
  expect(fetch).toHaveBeenCalledTimes(kind === 'unavailable' ? 2 : 1);
});
