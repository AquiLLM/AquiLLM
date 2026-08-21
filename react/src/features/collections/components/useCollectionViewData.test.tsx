// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useCollectionViewData } from './useCollectionViewData';

const collectionPayload = {
  collection: {
    id: 7,
    name: 'Research',
    parent: null,
    path: '/research',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-02T00:00:00Z',
  },
  children: [],
  documents: [],
  can_edit: true,
  can_manage: false,
};

function mockFetchResponses() {
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo) => {
    const url = String(input);
    if (url.includes('/api/collection/')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(collectionPayload),
      });
    }
    if (url.includes('/api/collections')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ collections: [] }),
      });
    }
    return Promise.reject(new Error(`Unexpected fetch: ${url}`));
  }));
}

beforeEach(() => {
  window.apiUrls = {
    api_collection: '/api/collection/%(col_id)s/',
    api_collections: '/api/collections/',
  };
  window.pageUrls = {};
  mockFetchResponses();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useCollectionViewData', () => {
  it('maps can_edit and can_manage from the collection response', async () => {
    const { result } = renderHook(() => useCollectionViewData('7'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.initialCanEdit).toBe(true);
    expect(result.current.initialCanManage).toBe(false);
    expect(result.current.collection?.name).toBe('Research');
  });

  it('defaults can_edit and can_manage to false when absent', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo) => {
      const url = String(input);
      if (url.includes('/api/collection/')) {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({
              ...collectionPayload,
              can_edit: undefined,
              can_manage: undefined,
            }),
        });
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ collections: [] }),
      });
    }));

    const { result } = renderHook(() => useCollectionViewData('7'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.initialCanEdit).toBe(false);
    expect(result.current.initialCanManage).toBe(false);
  });

  it('resets permissions when collectionId changes', async () => {
    const { result, rerender } = renderHook(
      ({ id }: { id: string }) => useCollectionViewData(id),
      { initialProps: { id: '7' } }
    );

    await waitFor(() => {
      expect(result.current.initialCanEdit).toBe(true);
    });

    vi.stubGlobal('fetch', vi.fn((input: RequestInfo) => {
      const url = String(input);
      if (url.includes('/api/collection/12')) {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({
              ...collectionPayload,
              collection: { ...collectionPayload.collection, id: 12, name: 'Other' },
              can_edit: false,
              can_manage: true,
            }),
        });
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ collections: [] }),
      });
    }));

    rerender({ id: '12' });

    await waitFor(() => {
      expect(result.current.collection?.id).toBe(12);
    });

    expect(result.current.initialCanEdit).toBe(false);
    expect(result.current.initialCanManage).toBe(true);
  });

  it('clears session state on fetch error', async () => {
    const { result } = renderHook(() => useCollectionViewData('7'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        ok: false,
        json: () => Promise.resolve({ error: 'Forbidden' }),
      })
    ));

    await act(async () => {
      result.current.fetchCollectionData();
    });

    await waitFor(() => {
      expect(result.current.error).toBe('Forbidden');
    });

    expect(result.current.collection).toBeNull();
    expect(result.current.contents).toEqual([]);
    expect(result.current.initialCanEdit).toBe(false);
    expect(result.current.initialCanManage).toBe(false);
  });
});
