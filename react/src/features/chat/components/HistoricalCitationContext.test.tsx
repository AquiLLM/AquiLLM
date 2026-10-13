// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { TextCitationModal } from './TextCitationModal';
import { PDFCitationModal } from './PDFCitationModal';
import type { CitationChunkDetail } from './citationTypes';

vi.mock('../../../main', () => ({ getCsrfCookie: () => 'csrf' }));
vi.mock('../../../utils/pdfWorker', () => ({ configurePdfWorker: vi.fn() }));
vi.mock('react-pdf', () => ({
  Document: ({ file }: { file: string }) => <div data-testid="pdf-file" data-file={file} />,
  Page: () => null,
}));

const base: CitationChunkDetail = {
  content: 'same', chunk_number: 1, start_position: 13, end_position: 17,
  start_time: null, modality: 'text', image_url: null,
  document: { id: 'doc-1', title: 'Historical source', type: 'RawTextDocument',
    has_pdf: false, source_url: null, full_text: '😀😀 same then same', text_offset: 0 },
};

beforeEach(() => {
  HTMLElement.prototype.scrollIntoView = vi.fn();
  window.pageUrls = { pdf: '/pdf/%(doc_id)s?download=1#page=2' } as typeof window.pageUrls;
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it('highlights the exact second occurrence with codepoint offsets after astral text', () => {
  const chunk = { ...base, source_context_status: 'verified' as const };
  const { container } = render(<TextCitationModal docId="doc-1" chunkId="42" preloadedChunk={chunk} onClose={() => {}} />);
  const mark = container.querySelector('mark');
  expect(mark?.textContent).toBe('same');
  expect(mark?.previousSibling?.textContent).toBe('😀😀 same then ');
});

it.each(['unavailable', 'verified'] as const)('shows saved bytes without relocating a %s context mismatch', status => {
  const chunk = { ...base, start_position: 0, end_position: 4, source_context_status: status,
    document: { ...base.document, full_text: status === 'unavailable' ? '' : 'other same then same' } };
  const { container } = render(<TextCitationModal docId="doc-1" chunkId="42" preloadedChunk={chunk} onClose={() => {}} />);
  expect(container.querySelector('mark')).toBeNull();
  expect(screen.getByText('same')).toBeTruthy();
  expect(screen.getByText(/original source context is unavailable/i)).toBeTruthy();
});

it('keeps the historical undefined-status normalized fallback', () => {
  const chunk = { ...base, start_position: 0, end_position: 4 };
  const { container } = render(<TextCitationModal docId="doc-1" chunkId="42" preloadedChunk={chunk} onClose={() => {}} />);
  expect(container.querySelector('mark')?.textContent).toBe('same');
});

it('binds the PDF resource and component identity to each exact cited chunk preserving query/hash', () => {
  const chunk = { ...base, document: { ...base.document, has_pdf: true } };
  const { rerender } = render(<PDFCitationModal docId="doc-1" chunkId="42" preloadedChunk={chunk} onClose={() => {}} />);
  expect(screen.getByTestId('pdf-file').getAttribute('data-file')).toBe('/pdf/doc-1?download=1&chunk_id=42#page=2');
  rerender(<PDFCitationModal docId="doc-1" chunkId="84" preloadedChunk={chunk} onClose={() => {}} />);
  expect(screen.getByTestId('pdf-file').getAttribute('data-file')).toBe('/pdf/doc-1?download=1&chunk_id=84#page=2');
});
