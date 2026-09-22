// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CitationModalProvider,
  CitationPanelSlot,
  useCitationModal,
} from './CitationModalProvider';

interface ModalProbeProps {
  docId?: string;
  chunkId?: string;
  messageUuid?: string;
  pinned?: boolean;
  onTogglePin?: () => void;
  preloadedChunk?: unknown;
}

const formatUrlMock = vi.hoisted(() =>
  vi.fn((_pattern: string, values: Record<string, string>) =>
    `/api/chunks/${values.chunk_id}?existing=1`,
  ),
);
const modalRenderMocks = vi.hoisted(() => ({
  pdf: vi.fn(),
  image: vi.fn(),
  text: vi.fn(),
}));

vi.mock('../../../utils/formatUrl', () => ({ default: formatUrlMock }));

vi.mock('./PDFCitationModal', () => ({
  default: (props: ModalProbeProps) => {
    modalRenderMocks.pdf(props);
    return (
      <button
        type="button"
        data-testid="pdf-modal"
        data-pinned={String(props.pinned)}
        data-chunk={JSON.stringify(props.preloadedChunk)}
        onClick={props.onTogglePin}
      >
        PDF modal
      </button>
    );
  },
}));

vi.mock('./ImageCitationModal', () => ({
  default: (props: ModalProbeProps) => {
    modalRenderMocks.image(props);
    return (
      <button
        type="button"
        data-testid="image-modal"
        data-pinned={String(props.pinned)}
        data-chunk={JSON.stringify(props.preloadedChunk)}
        onClick={props.onTogglePin}
      >
        Image modal
      </button>
    );
  },
}));

vi.mock('./TextCitationModal', () => ({
  default: (props: ModalProbeProps) => {
    modalRenderMocks.text(props);
    return (
      <button
        type="button"
        data-testid="text-modal"
        data-pinned={String(props.pinned)}
        data-chunk={JSON.stringify(props.preloadedChunk)}
        onClick={props.onTogglePin}
      >
        Text modal
      </button>
    );
  },
}));

vi.mock('./CitationUnavailable', () => ({
  default: ({ pinned, onTogglePin }: ModalProbeProps) => (
    <button
      type="button"
      data-testid="unavailable-modal"
      data-pinned={String(pinned)}
      onClick={onTogglePin}
    >
      Citation unavailable
    </button>
  ),
}));

const documentSummary = {
  id: 'doc-1',
  title: 'Source document',
  type: 'RawTextDocument',
  has_pdf: false,
  source_url: 'https://source.example/article',
};

const textSummary = {
  content: 'cited passage',
  chunk_number: 3,
  start_position: 7,
  end_position: 20,
  start_time: null,
  modality: 'text',
  image_url: null,
  document: documentSummary,
};

const textDetail = {
  ...textSummary,
  document: {
    ...documentSummary,
    full_text: 'prefix cited passage suffix',
    text_offset: 2,
  },
};

const pdfSummary = {
  ...textSummary,
  document: {
    ...documentSummary,
    type: 'PDFDocument',
    has_pdf: true,
  },
};

const imageSummary = {
  ...textSummary,
  modality: 'image',
  image_url: '/media/figures/chart.png',
  document: {
    ...documentSummary,
    type: 'PDFDocument',
    has_pdf: true,
  },
};

function Harness() {
  const { openCitation } = useCitationModal();
  return (
    <>
      <button
        type="button"
        onClick={() =>
          openCitation({ docId: 'doc-1', chunkId: '42', messageUuid: 'message-1' })
        }
      >
        Open citation
      </button>
      <button
        type="button"
        onClick={() =>
          openCitation({ docId: 'doc-2', chunkId: '84', messageUuid: 'message-2' })
        }
      >
        Switch citation
      </button>
      <button
        type="button"
        onClick={() =>
          openCitation({ docId: 'doc-3', chunkId: '126', messageUuid: 'message-3' })
        }
      >
        Switch citation again
      </button>
      <CitationPanelSlot />
    </>
  );
}

function renderProvider() {
  render(
    <CitationModalProvider>
      <Harness />
    </CitationModalProvider>,
  );
  fireEvent.click(screen.getByRole('button', { name: 'Open citation' }));
}

function response(body: unknown, status = 200): Response {
  return new Response(body === null ? null : JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function createStorage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() {
      return values.size;
    },
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => Array.from(values.keys())[index] ?? null,
    removeItem: (key) => values.delete(key),
    setItem: (key, value) => values.set(key, String(value)),
  };
}

const originalApiUrlsDescriptor = Object.getOwnPropertyDescriptor(window, 'apiUrls');
const originalLocalStorageDescriptor = Object.getOwnPropertyDescriptor(window, 'localStorage');
const originalFetchDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'fetch');
const originalPerformanceDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'performance');

let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>;

beforeEach(() => {
  fetchMock = vi.fn<typeof fetch>();
  Object.defineProperty(window, 'apiUrls', {
    configurable: true,
    writable: true,
    value: { api_chunk_detail: '/api/chunks/{chunk_id}?existing=1' },
  });
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: createStorage(),
  });
  Object.defineProperty(globalThis, 'fetch', {
    configurable: true,
    writable: true,
    value: fetchMock,
  });
  formatUrlMock.mockClear();
  modalRenderMocks.pdf.mockClear();
  modalRenderMocks.image.mockClear();
  modalRenderMocks.text.mockClear();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  if (originalApiUrlsDescriptor) {
    Object.defineProperty(window, 'apiUrls', originalApiUrlsDescriptor);
  } else {
    Reflect.deleteProperty(window, 'apiUrls');
  }
  if (originalLocalStorageDescriptor) {
    Object.defineProperty(window, 'localStorage', originalLocalStorageDescriptor);
  } else {
    Reflect.deleteProperty(window, 'localStorage');
  }
  if (originalFetchDescriptor) {
    Object.defineProperty(globalThis, 'fetch', originalFetchDescriptor);
  } else {
    Reflect.deleteProperty(globalThis, 'fetch');
  }
  if (originalPerformanceDescriptor) {
    Object.defineProperty(globalThis, 'performance', originalPerformanceDescriptor);
  } else {
    Reflect.deleteProperty(globalThis, 'performance');
  }
});

describe('CitationModalProvider compact routing', () => {
  it('isolates modal state and aborts outstanding work when the complete target changes', async () => {
    const firstCompact = deferred<Response>();
    const secondCompact = deferred<Response>();
    const secondDetail = deferred<Response>();
    const thirdCompact = deferred<Response>();
    const observedSignals: {
      firstCompact?: AbortSignal | null;
      secondDetail?: AbortSignal | null;
    } = {};
    fetchMock
      .mockImplementationOnce((_input, init) => {
        observedSignals.firstCompact = init?.signal ?? null;
        return firstCompact.promise;
      })
      .mockImplementationOnce(() => secondCompact.promise)
      .mockImplementationOnce((_input, init) => {
        observedSignals.secondDetail = init?.signal ?? null;
        return secondDetail.promise;
      })
      .mockImplementationOnce(() => thirdCompact.promise);

    renderProvider();
    await act(async () => {
      firstCompact.resolve(response(pdfSummary));
      await firstCompact.promise;
    });
    expect(await screen.findByTestId('pdf-modal')).not.toBeNull();
    modalRenderMocks.pdf.mockClear();

    fireEvent.click(screen.getByRole('button', { name: 'Switch citation' }));

    expect(screen.queryByTestId('pdf-modal')).toBeNull();
    expect(observedSignals.firstCompact?.aborted).toBe(true);
    expect(modalRenderMocks.pdf).not.toHaveBeenCalled();

    const secondSummary = {
      ...textSummary,
      document: { ...documentSummary, id: 'doc-2', title: 'Second document' },
    };
    await act(async () => {
      secondCompact.resolve(response(secondSummary));
      await secondCompact.promise;
    });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));

    fireEvent.click(screen.getByRole('button', { name: 'Switch citation again' }));

    expect(observedSignals.secondDetail?.aborted).toBe(true);
    expect(screen.queryByTestId('text-modal')).toBeNull();
    await act(async () => {
      secondDetail.resolve(
        response({
          ...textDetail,
          document: {
            ...textDetail.document,
            id: 'doc-2',
            title: 'Second document',
          },
        }),
      );
      await secondDetail.promise;
    });
    expect(screen.queryByTestId('text-modal')).toBeNull();

    const thirdSummary = {
      ...imageSummary,
      image_url: '/media/figures/third.png',
      document: { ...imageSummary.document, id: 'doc-3', title: 'Third document' },
    };
    await act(async () => {
      thirdCompact.resolve(response(thirdSummary));
      await thirdCompact.promise;
    });
    const imageModal = await screen.findByTestId('image-modal');
    expect(JSON.parse(imageModal.getAttribute('data-chunk') ?? 'null')).toEqual(
      thirdSummary,
    );
    expect(screen.queryByTestId('pdf-modal')).toBeNull();
    expect(screen.queryByTestId('text-modal')).toBeNull();
  });

  it('loads compact PDF metadata once and mounts the PDF modal with pin props', async () => {
    fetchMock.mockResolvedValueOnce(response(pdfSummary));

    renderProvider();

    const modal = await screen.findByTestId('pdf-modal');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const compactUrl = new URL(String(fetchMock.mock.calls[0][0]), window.location.href);
    expect(compactUrl.searchParams.get('existing')).toBe('1');
    expect(compactUrl.searchParams.get('include_full_text')).toBe('0');
    expect(JSON.parse(modal.getAttribute('data-chunk') ?? 'null')).toEqual(pdfSummary);
    expect(modal.getAttribute('data-pinned')).toBe('false');
    fireEvent.click(modal);
    await waitFor(() => expect(modal.getAttribute('data-pinned')).toBe('true'));
  });

  it('loads compact image metadata once and preserves image and source fields', async () => {
    fetchMock.mockResolvedValueOnce(response(imageSummary));

    renderProvider();

    const modal = await screen.findByTestId('image-modal');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const chunk = JSON.parse(modal.getAttribute('data-chunk') ?? 'null');
    expect(chunk.modality).toBe('image');
    expect(chunk.image_url).toBe('/media/figures/chart.png');
    expect(chunk.document.source_url).toBe('https://source.example/article');
    expect(modal.getAttribute('data-pinned')).toBe('false');
    fireEvent.click(modal);
    await waitFor(() => expect(modal.getAttribute('data-pinned')).toBe('true'));
  });

  it('loads full detail exactly once after compact non-PDF text metadata', async () => {
    fetchMock
      .mockResolvedValueOnce(response(textSummary))
      .mockResolvedValueOnce(response(textDetail));

    renderProvider();

    const modal = await screen.findByTestId('text-modal');
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const compactUrl = new URL(String(fetchMock.mock.calls[0][0]), window.location.href);
    const detailUrl = new URL(String(fetchMock.mock.calls[1][0]), window.location.href);
    expect(compactUrl.searchParams.get('include_full_text')).toBe('0');
    expect(detailUrl.searchParams.has('include_full_text')).toBe(false);
    expect(detailUrl.searchParams.get('existing')).toBe('1');
    const chunk = JSON.parse(modal.getAttribute('data-chunk') ?? 'null');
    expect(chunk.document.full_text).toBe('prefix cited passage suffix');
    expect(chunk.document.text_offset).toBe(2);
    expect(modal.getAttribute('data-pinned')).toBe('false');
    fireEvent.click(modal);
    await waitFor(() => expect(modal.getAttribute('data-pinned')).toBe('true'));
  });

  it('routes a compact 404 to CitationUnavailable', async () => {
    fetchMock.mockResolvedValueOnce(response(null, 404));

    renderProvider();

    const modal = await screen.findByTestId('unavailable-modal');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(modal.getAttribute('data-pinned')).toBe('false');
    fireEvent.click(modal);
    await waitFor(() => expect(modal.getAttribute('data-pinned')).toBe('true'));
  });

  it('routes a full-detail 404 to CitationUnavailable', async () => {
    fetchMock
      .mockResolvedValueOnce(response(textSummary))
      .mockResolvedValueOnce(response(null, 404));

    renderProvider();

    expect(await screen.findByTestId('unavailable-modal')).not.toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('retains the friendly text-modal route for non-404 failures', async () => {
    fetchMock.mockResolvedValueOnce(response(null, 500));

    renderProvider();

    const modal = await screen.findByTestId('text-modal');
    expect(modal.getAttribute('data-chunk')).toBe('null');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('retains the friendly text-modal route for full-detail failures', async () => {
    fetchMock
      .mockResolvedValueOnce(response(textSummary))
      .mockResolvedValueOnce(response(null, 503));

    renderProvider();

    const modal = await screen.findByTestId('text-modal');
    expect(modal.getAttribute('data-chunk')).toBe('null');
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('records development compact-metadata performance marks and measurement', async () => {
    const performanceMock = {
      clearMarks: vi.fn(),
      clearMeasures: vi.fn(),
      mark: vi.fn(),
      measure: vi.fn(),
    };
    Object.defineProperty(globalThis, 'performance', {
      configurable: true,
      value: performanceMock,
    });
    fetchMock.mockResolvedValueOnce(response(pdfSummary));

    renderProvider();

    await screen.findByTestId('pdf-modal');
    expect(performanceMock.clearMarks).toHaveBeenCalledWith(
      'aquillm:citation:open-start',
    );
    expect(performanceMock.clearMarks).toHaveBeenCalledWith(
      'aquillm:citation:compact-ready',
    );
    expect(performanceMock.clearMeasures).toHaveBeenCalledWith(
      'aquillm:citation:compact-metadata',
    );
    expect(performanceMock.mark).toHaveBeenCalledWith('aquillm:citation:open-start');
    expect(performanceMock.mark).toHaveBeenCalledWith('aquillm:citation:compact-ready');
    expect(performanceMock.measure).toHaveBeenCalledWith(
      'aquillm:citation:compact-metadata',
      'aquillm:citation:open-start',
      'aquillm:citation:compact-ready',
    );
  });

  it('keeps citation routing available when User Timing methods are absent', async () => {
    Object.defineProperty(globalThis, 'performance', {
      configurable: true,
      value: {},
    });
    fetchMock.mockResolvedValueOnce(response(pdfSummary));

    expect(() => renderProvider()).not.toThrow();

    expect(await screen.findByTestId('pdf-modal')).not.toBeNull();
  });

  it('keeps citation routing available when User Timing methods throw', async () => {
    const timingFailure = () => {
      throw new Error('User Timing unavailable');
    };
    Object.defineProperty(globalThis, 'performance', {
      configurable: true,
      value: {
        clearMarks: timingFailure,
        clearMeasures: timingFailure,
        mark: timingFailure,
        measure: timingFailure,
      },
    });
    fetchMock.mockResolvedValueOnce(response(pdfSummary));

    expect(() => renderProvider()).not.toThrow();

    expect(await screen.findByTestId('pdf-modal')).not.toBeNull();
  });
});
