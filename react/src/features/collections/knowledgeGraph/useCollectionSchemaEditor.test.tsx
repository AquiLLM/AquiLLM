// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { CollectionSchemaApi } from './collectionSchemaApi';
import {
  editDraftEnvelope,
  emptyEditableEnvelope,
  historyPageFixture,
  manageDraftEnvelope,
  validationResultFixture,
  viewPublishedEnvelope,
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
    startGeneration: vi.fn().mockResolvedValue({
      ok: true,
      data: { run_id: 'run-1', status: 'queued', status_url: '/api/collection/col-empty/schema/generation/run-1/' },
    }),
    getGenerationStatus: vi.fn().mockResolvedValue({
      ok: true,
      data: { run_id: 'run-1', status: 'succeeded', error_code: null, statistics: {} },
    }),
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

  it('automatically starts exactly one generation run for an empty editable workspace', async () => {
    const api = createMockApi({
      loadWorkspace: vi.fn().mockResolvedValue(emptyEditableEnvelope).mockResolvedValue({ ok: true, data: emptyEditableEnvelope }),
      getGenerationStatus: vi.fn(() => new Promise(() => undefined)) as unknown as CollectionSchemaApi['getGenerationStatus'],
    });
    const { result, rerender } = renderHook(
      ({ collectionName }) => useCollectionSchemaEditor({ collectionId: 'col-empty', collectionName, api }),
      { initialProps: { collectionName: 'Empty' } },
    );

    await waitFor(() => expect(api.startGeneration).toHaveBeenCalledWith('col-empty'));
    rerender({ collectionName: 'Renamed empty collection' });
    expect(api.startGeneration).toHaveBeenCalledTimes(1);
    expect(result.current.generation.status).toBe('queued');
  });

  it.each([
    ['VIEW workspace', viewPublishedEnvelope],
    ['existing draft', editDraftEnvelope],
    ['existing published schema', manageDraftEnvelope],
  ])('does not auto-generate for a %s', async (_label, envelope) => {
    const api = createMockApi({ loadWorkspace: vi.fn().mockResolvedValue({ ok: true, data: envelope }) });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: envelope.collection_id, api }));

    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    expect(api.startGeneration).not.toHaveBeenCalled();
  });

  it('reloads the workspace when generation succeeds', async () => {
    const api = createMockApi({
      loadWorkspace: vi
        .fn()
        .mockResolvedValueOnce({ ok: true, data: emptyEditableEnvelope })
        .mockResolvedValueOnce({ ok: true, data: editDraftEnvelope }),
    });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-empty', api }));

    await waitFor(() => expect(api.loadWorkspace).toHaveBeenCalledTimes(2));
    expect(result.current.generation.status).toBe('succeeded');
    expect(result.current.editorState.envelope?.draft?.draft_id).toBe('draft-edit-1');
  });

  it('exposes manual retry after a failed generation', async () => {
    const api = createMockApi({
      loadWorkspace: vi.fn().mockResolvedValue({ ok: true, data: emptyEditableEnvelope }),
      getGenerationStatus: vi
        .fn()
        .mockResolvedValueOnce({ ok: true, data: { run_id: 'run-1', status: 'failed', error_code: 'no_collection_text', statistics: {} } })
        .mockResolvedValueOnce({ ok: true, data: { run_id: 'run-2', status: 'failed', error_code: 'no_collection_text', statistics: {} } }),
      startGeneration: vi
        .fn()
        .mockResolvedValueOnce({ ok: true, data: { run_id: 'run-1', status: 'queued', status_url: '/run-1/' } })
        .mockResolvedValueOnce({ ok: true, data: { run_id: 'run-2', status: 'queued', status_url: '/run-2/' } }),
    });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-empty', api }));

    await waitFor(() => expect(result.current.generation).toMatchObject({ status: 'failed', errorCode: 'no_collection_text' }));
    await act(async () => {
      await result.current.onGenerateSchema();
    });
    expect(api.startGeneration).toHaveBeenCalledTimes(2);
  });

  it('ignores a generation start response after the collection changes', async () => {
    let resolveStart: ((value: unknown) => void) | undefined;
    const api = createMockApi({
      loadWorkspace: vi
        .fn()
        .mockResolvedValueOnce({ ok: true, data: { ...emptyEditableEnvelope, collection_id: 'slow' } })
        .mockResolvedValueOnce({ ok: true, data: manageDraftEnvelope }),
      startGeneration: vi.fn(
        () =>
          new Promise((resolve) => {
            resolveStart = resolve;
          }),
      ) as unknown as CollectionSchemaApi['startGeneration'],
    });
    const { rerender } = renderHook(
      ({ collectionId }) => useCollectionSchemaEditor({ collectionId, api }),
      { initialProps: { collectionId: 'slow' } },
    );

    await waitFor(() => expect(api.startGeneration).toHaveBeenCalledWith('slow'));
    rerender({ collectionId: 'fast' });
    await act(async () => {
      resolveStart?.({ ok: true, data: { run_id: 'slow-run', status: 'queued', status_url: '/slow-run/' } });
    });
    await waitFor(() => expect(api.loadWorkspace).toHaveBeenCalledWith('fast'));
    expect(api.getGenerationStatus).not.toHaveBeenCalled();
  });

  it('does not create a poll when a deferred generation start resolves after unmount', async () => {
    let resolveStart: ((value: unknown) => void) | undefined;
    const api = createMockApi({
      loadWorkspace: vi.fn().mockResolvedValue({ ok: true, data: emptyEditableEnvelope }),
      startGeneration: vi.fn(
        () =>
          new Promise((resolve) => {
            resolveStart = resolve;
          }),
      ) as unknown as CollectionSchemaApi['startGeneration'],
    });
    const { unmount } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-empty', api }));

    await waitFor(() => expect(api.startGeneration).toHaveBeenCalledWith('col-empty'));
    unmount();
    await act(async () => {
      resolveStart?.({ ok: true, data: { run_id: 'late-run', status: 'queued', status_url: '/late-run/' } });
    });

    expect(api.getGenerationStatus).not.toHaveBeenCalled();
  });

  it('aborts and ignores a deferred generation status response after unmount', async () => {
    let resolveStatus: ((value: unknown) => void) | undefined;
    let statusSignal: AbortSignal | undefined;
    const api = createMockApi({
      loadWorkspace: vi.fn().mockResolvedValue({ ok: true, data: emptyEditableEnvelope }),
      getGenerationStatus: vi.fn((_, __, signal) => {
        statusSignal = signal;
        return new Promise((resolve) => {
          resolveStatus = resolve;
        });
      }) as unknown as CollectionSchemaApi['getGenerationStatus'],
    });
    const { unmount } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-empty', api }));

    await waitFor(() => expect(api.getGenerationStatus).toHaveBeenCalledTimes(1));
    unmount();
    expect(statusSignal?.aborted).toBe(true);
    await act(async () => {
      resolveStatus?.({ ok: true, data: { run_id: 'run-1', status: 'succeeded', error_code: null, statistics: {} } });
    });

    expect(api.loadWorkspace).toHaveBeenCalledTimes(1);
  });
});
