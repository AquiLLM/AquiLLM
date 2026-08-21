// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { CollectionSchemaApi } from './collectionSchemaApi';
import {
  editDraftEnvelope,
  historyPageFixture,
  manageDraftEnvelope,
  validationResultFixture,
} from './schemaTestFixtures';
import { useCollectionSchemaEditor } from './useCollectionSchemaEditor';

function createMockApi(overrides: Partial<CollectionSchemaApi> = {}): CollectionSchemaApi {
  return {
    loadWorkspace: vi.fn().mockResolvedValue({ ok: true, data: manageDraftEnvelope }),
    createDraft: vi.fn().mockResolvedValue({ ok: true, data: editDraftEnvelope }),
    upsertEntity: vi.fn().mockResolvedValue({ ok: true, data: manageDraftEnvelope }),
    deleteEntity: vi.fn().mockResolvedValue({ ok: true, data: manageDraftEnvelope }),
    upsertRelation: vi.fn().mockResolvedValue({ ok: true, data: manageDraftEnvelope }),
    deleteRelation: vi.fn().mockResolvedValue({ ok: true, data: manageDraftEnvelope }),
    validate: vi.fn().mockResolvedValue({ ok: true, data: validationResultFixture }),
    fetchDiff: vi.fn(),
    publish: vi.fn().mockResolvedValue({ ok: true, data: { ...manageDraftEnvelope, draft: null } }),
    discardDraft: vi.fn().mockResolvedValue({ ok: true, data: { ...manageDraftEnvelope, draft: null } }),
    listVersions: vi.fn().mockResolvedValue({ ok: true, data: historyPageFixture }),
    fetchVersionDiff: vi.fn(),
    restoreVersion: vi.fn().mockResolvedValue({ ok: true, data: editDraftEnvelope }),
    restoreReplace: vi.fn(),
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useCollectionSchemaEditor', () => {
  it('loads workspace on mount', async () => {
    const api = createMockApi();
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    expect(api.loadWorkspace).toHaveBeenCalledWith('col-manage');
    expect(result.current.editorState.envelope?.draft?.draft_id).toBe('draft-manage-1');
  });

  it('reloads when collection id changes', async () => {
    const api = createMockApi();
    const { result, rerender } = renderHook(
      ({ collectionId }) => useCollectionSchemaEditor({ collectionId, api }),
      { initialProps: { collectionId: '1' } },
    );
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    rerender({ collectionId: '2' });
    await waitFor(() => expect(api.loadWorkspace).toHaveBeenCalledWith('2'));
  });

  it('enters unavailable when routes are missing', async () => {
    const api = createMockApi({
      loadWorkspace: vi.fn().mockResolvedValue({ ok: false, kind: 'schema_unavailable' }),
    });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: '1', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('unavailable'));
  });

  it('enters session_expired on auth failures', async () => {
    const api = createMockApi({
      loadWorkspace: vi.fn().mockResolvedValue({ ok: false, kind: 'session_expired' }),
    });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: '1', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('session_expired'));
  });

  it('creates a draft through the adapter', async () => {
    const api = createMockApi();
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-edit', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      await result.current.onCreateDraft?.();
    });
    expect(api.createDraft).toHaveBeenCalledWith('col-edit');
  });

  it('validates using the loaded draft identity', async () => {
    const api = createMockApi();
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      await result.current.onValidate?.();
    });
    expect(api.validate).toHaveBeenCalledWith('col-manage', 'draft-manage-1', 5);
    expect(result.current.editorState.validation.status).toBe('valid');
  });

  it('discards draft through the adapter', async () => {
    const api = createMockApi();
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      await result.current.onDiscardDraft?.();
    });
    expect(api.discardDraft).toHaveBeenCalledWith('col-manage', 'draft-manage-1', 5);
  });

  it('loads history when requested', async () => {
    const api = createMockApi();
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      await result.current.onLoadHistory?.();
    });
    expect(api.listVersions).toHaveBeenCalledWith('col-manage');
    expect(result.current.history?.versions).toHaveLength(2);
  });

  it('ignores stale load responses after collection change', async () => {
    let resolveFirst: ((value: unknown) => void) | undefined;
    const api = createMockApi({
      loadWorkspace: vi
        .fn()
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              resolveFirst = resolve;
            }),
        )
        .mockResolvedValue({ ok: true, data: manageDraftEnvelope }),
    });
    const { result, rerender } = renderHook(
      ({ collectionId }) => useCollectionSchemaEditor({ collectionId, api }),
      { initialProps: { collectionId: 'slow' } },
    );
    rerender({ collectionId: 'fast' });
    await waitFor(() => expect(result.current.editorState.collectionId).toBe('fast'));
    await act(async () => {
      resolveFirst?.({ ok: true, data: editDraftEnvelope });
    });
    await waitFor(() => expect(result.current.editorState.envelope?.draft?.draft_id).toBe('draft-manage-1'));
  });
});
