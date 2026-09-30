// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import IngestRowsContainer from './IngestRowsContainer';
import CollectionView from '../../collections/components/CollectionView';

vi.mock('../../../main', () => ({ getCsrfCookie: () => 'test-csrf' }));
const response = (body: unknown, status = 200) => ({ ok: status < 400, status, json: async () => body });
const urls = {
  ingestUploadsUrl: '/uploads/', ingestArxivUrl: '/arxiv/', ingestPdfUrl: '/pdf/',
  ingestVttUrl: '/vtt/', ingestWebpageUrl: '/webpage/', ingestHandwrittenUrl: '/handwritten/',
};
const collection = {
  collection: { id: 7, name: 'Research', parent: null, path: '/research', created_at: '2026-01-01', updated_at: '2026-01-01' },
  documents: [], children: [], can_edit: true, can_manage: true,
};
function chooseFiles(container: HTMLElement, files: File[], row = 0) {
  fireEvent.change(container.querySelectorAll('input[type=file]')[row], { target: { files } });
}
function queued(batchId: number, names: string[], rejected: unknown[] = []) {
  return response({ batch_id: batchId, queued_count: names.length, rejected_count: rejected.length,
    items: names.map((filename, index) => ({ id: batchId * 10 + index, filename, status: 'queued' })), rejected }, 202);
}
beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  vi.spyOn(console, 'error').mockImplementation(() => {});
  window.apiUrls = {
    api_collection: '/api/collection/%(col_id)s/', api_collections: '/api/collections/',
    api_ingest_uploads: '/uploads/', api_ingest_arxiv: '/arxiv/', api_ingest_pdf: '/pdf/',
    api_ingest_vtt: '/vtt/', api_ingest_webpage: '/webpage/', api_ingest_handwritten_notes: '/handwritten/',
  };
  window.pageUrls = { document: '/document/%(doc_id)s/', collection: '/collection/%(col_id)s/' };
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('upload recovery', () => {
  it('retries the correct same-name file using the source index returned with acceptance', async () => {
    const accepted = new File(['accepted'], 'same.txt');
    const rejected = new File(['rejected'], 'same.txt');
    let posts = 0;
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        posts++;
        return Promise.resolve(posts === 1 ? response({ batch_id: 12, queued_count: 1, rejected_count: 1,
          items: [{ id: 120, filename: 'same.txt', source_index: 0, status: 'queued' }],
          rejected: [{ filename: 'same.txt', source_index: 1, error: 'Rejected second file' }] }, 202)
          : queued(13, ['same.txt']));
      }
      return posts === 1 ? Promise.resolve(response({ counts: { success: 1, error: 0 },
        items: [{ id: 120, original_filename: 'same.txt', status: 'success' }] })) : new Promise(() => {});
    });
    vi.stubGlobal('fetch', fetchMock);
    const view = render(<IngestRowsContainer {...urls} collectionId="7" />);
    chooseFiles(view.container, [accepted, rejected]);
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    await screen.findByText(/Rejected second file/);
    await waitFor(() => expect(screen.queryByText('Batch ingestion queued...')).toBeNull());
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    await waitFor(() => expect(posts).toBe(2));
    const second = fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST')[1][1]!;
    expect((second.body as FormData).getAll('files')).toEqual([rejected]);
  });

  it('retains the accepted batch on a status outage and retries status without another upload', async () => {
    let gets = 0;
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      if (init?.method === 'POST') return Promise.resolve(queued(12, ['ok.txt']));
      gets++;
      return gets === 1 ? Promise.reject(new Error('Network unavailable')) : Promise.resolve(response({
        counts: { success: 1, error: 0 }, items: [{ id: 120, original_filename: 'ok.txt', status: 'success' }],
      }));
    });
    vi.stubGlobal('fetch', fetchMock);
    const view = render(<IngestRowsContainer {...urls} collectionId="7" />);
    chooseFiles(view.container, [new File(['x'], 'ok.txt')]);
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    await screen.findByText(/Network unavailable/);
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    fireEvent.click(screen.getByRole('button', { name: 'Retry status' }));
    await screen.findByText('Submission successful!');
    expect(fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1);
  });

  it('retains initial rejections through processing and retries only unsuccessful files', async () => {
    let finishPoll!: (value: unknown) => void;
    let posts = 0;
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        posts++;
        return Promise.resolve(posts === 1
          ? queued(12, ['ok.txt'], [{ filename: 'rejected.txt', error: 'File too large.' }])
          : queued(13, ['rejected.txt']));
      }
      return new Promise(resolve => { finishPoll = resolve; });
    });
    vi.stubGlobal('fetch', fetchMock);
    const view = render(<IngestRowsContainer {...urls} collectionId="7" />);
    chooseFiles(view.container, [new File(['x'], 'ok.txt'), new File(['x'], 'rejected.txt')]);
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    await screen.findByText(/rejected.txt.*File too large/);
    expect(screen.queryByText('Submission successful!')).toBeNull();
    await act(async () => finishPoll(response({ counts: { success: 1, error: 0, queued: 0, processing: 0 },
      items: [{ id: 120, original_filename: 'ok.txt', status: 'success' }] })));
    expect(screen.getByText(/rejected.txt.*File too large/)).toBeTruthy();
    expect(screen.queryByText('Submission successful!')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    await waitFor(() => expect(posts).toBe(2));
    const second = fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST')[1][1]!;
    expect((second.body as FormData).getAll('files').map(file => (file as File).name)).toEqual(['rejected.txt']);
  });

  it('skips active batches while permitting a new row to be submitted', async () => {
    let posts = 0;
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        posts++;
        return Promise.resolve(queued(posts, [posts === 1 ? 'first.txt' : 'second.txt']));
      }
      return Promise.resolve(response({ counts: { queued: 1, processing: 0, success: 0, error: 0 }, items: [] }));
    });
    vi.stubGlobal('fetch', fetchMock);
    const view = render(<IngestRowsContainer {...urls} collectionId="7" />);
    chooseFiles(view.container, [new File(['x'], 'first.txt')]);
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    await screen.findByText('Batch ingestion queued...');
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    await act(async () => {});
    expect(posts).toBe(1);
    fireEvent.click(screen.getByRole('button', { name: 'Add Another' }));
    chooseFiles(view.container, [new File(['x'], 'second.txt')], 1);
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    await waitFor(() => expect(posts).toBe(2));
    const second = fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST')[1][1]!;
    expect((second.body as FormData).getAll('files').map(file => (file as File).name)).toEqual(['second.txt']);
  });

  it('keeps processing failures and unsent rows mounted through a collection refresh', async () => {
    let finishPoll!: (value: unknown) => void;
    let finishRefresh!: (value: unknown) => void;
    let collectionRequests = 0;
    vi.stubGlobal('fetch', vi.fn((url: string, init?: RequestInit) => {
      if (url.startsWith('/api/collection/')) {
        collectionRequests++;
        return collectionRequests === 1 ? Promise.resolve(response(collection)) : new Promise(resolve => { finishRefresh = resolve; });
      }
      if (url === '/api/collections/') return Promise.resolve(response({ collections: [] }));
      if (url === '/uploads/' && init?.method === 'POST') return Promise.resolve(queued(12, ['ok.txt', 'bad.txt']));
      if (url === '/uploads/12/') return new Promise(resolve => { finishPoll = resolve; });
      return Promise.reject(new Error(`Unexpected fetch ${url}`));
    }));
    const view = render(<CollectionView collectionId="7" />);
    await screen.findByRole('heading', { name: 'Research' });
    chooseFiles(view.container, [new File(['x'], 'ok.txt'), new File(['x'], 'bad.txt')]);
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    await screen.findByText('Batch ingestion queued...');
    fireEvent.click(screen.getByRole('button', { name: 'Add Another' }));
    chooseFiles(view.container, [new File(['unsent'], 'draft.txt')], 1);
    await act(async () => finishPoll(response({ counts: { success: 1, error: 1 }, items: [
      { id: 120, original_filename: 'ok.txt', status: 'success' },
      { id: 121, original_filename: 'bad.txt', status: 'error', error_message: 'Unreadable file' },
    ] })));
    expect(screen.getByText(/Unreadable file/)).toBeTruthy();
    expect(screen.getByText('draft.txt')).toBeTruthy();
    await act(async () => finishRefresh(response(collection)));
    expect(screen.getByText(/Unreadable file/)).toBeTruthy();
    expect(screen.getByText('draft.txt')).toBeTruthy();
    expect(view.container.querySelectorAll('input[type=file]')).toHaveLength(2);
  });

  it('keeps an unsent row after a temporary refresh failure and offers a usable retry', async () => {
    let finishPoll!: (value: unknown) => void;
    let collectionRequests = 0;
    vi.stubGlobal('fetch', vi.fn((url: string, init?: RequestInit) => {
      if (url.startsWith('/api/collection/')) {
        collectionRequests++;
        return collectionRequests === 2 ? Promise.reject(new Error('Temporary network outage')) : Promise.resolve(response(collection));
      }
      if (url === '/api/collections/') return Promise.resolve(response({ collections: [] }));
      if (url === '/uploads/' && init?.method === 'POST') return Promise.resolve(queued(12, ['ok.txt']));
      if (url === '/uploads/12/') return new Promise(resolve => { finishPoll = resolve; });
      return Promise.reject(new Error(`Unexpected fetch ${url}`));
    }));
    const view = render(<CollectionView collectionId="7" />);
    await screen.findByRole('heading', { name: 'Research' });
    chooseFiles(view.container, [new File(['x'], 'ok.txt')]);
    fireEvent.click(screen.getByRole('button', { name: 'Submit All' }));
    await screen.findByText('Batch ingestion queued...');
    fireEvent.click(screen.getByRole('button', { name: 'Add Another' }));
    chooseFiles(view.container, [new File(['unsent'], 'draft.txt')], 1);
    await act(async () => finishPoll(response({ counts: { success: 1, error: 0 }, items: [
      { id: 120, original_filename: 'ok.txt', status: 'success' },
    ] })));
    await screen.findByText(/Temporary network outage/);
    expect(screen.getByText('draft.txt')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Retry collection' }));
    await waitFor(() => expect(screen.queryByText(/Temporary network outage/)).toBeNull());
    expect(screen.getByText('draft.txt')).toBeTruthy();
    expect(collectionRequests).toBe(3);
  });
});
