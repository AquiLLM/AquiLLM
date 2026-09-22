// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { CollectionSchemaApi } from './collectionSchemaApi';

import {
  createMockApi,
  editDraftEnvelope,
  emptyDraftEnvelope,
  emptyEditableEnvelope,
  manageDraftEnvelope,
  viewPublishedEnvelope,
} from './schemaTestFixtures';
import { useCollectionSchemaEditor } from './useCollectionSchemaEditor';

afterEach(() => { vi.restoreAllMocks(); });

describe('useCollectionSchemaEditor generation', () => {
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

  it('automatically starts generation for an unchanged empty draft', async () => {
    const api = createMockApi({
      loadWorkspace: vi.fn().mockResolvedValue({ ok: true, data: emptyDraftEnvelope }),
      getGenerationStatus: vi.fn(() => new Promise(() => undefined)) as unknown as CollectionSchemaApi['getGenerationStatus'],
    });

    renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-empty-draft', api }));

    await waitFor(() => expect(api.startGeneration).toHaveBeenCalledWith('col-empty-draft'));
    expect(api.startGeneration).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['VIEW workspace', viewPublishedEnvelope],
    ['existing nonempty draft', editDraftEnvelope],
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
    expect(result.current.editorState.envelope?.draft?.draft_id).toBe('10000000-0000-4000-8000-000000000002');
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
