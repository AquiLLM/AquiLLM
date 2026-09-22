// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  createMockApi,
  editDraftEnvelope,
  manageDraftEnvelope,
} from './schemaTestFixtures';
import { useCollectionSchemaEditor } from './useCollectionSchemaEditor';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useCollectionSchemaEditor', () => {
  it('keeps form draft UUID and revision when the workspace receives a replacement draft', async () => {
    const original = manageDraftEnvelope.draft!;
    const replacement = { ...manageDraftEnvelope, draft: { ...original, draft_id: '00000000-0000-0000-0000-000000000002' } };
    const api = createMockApi({ createDraft: vi.fn().mockResolvedValue({ ok: true, data: replacement }) });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    act(() => result.current.onSelectDefinition('entity', 'person'));
    act(() => result.current.onFieldChange('description', 'local edit'));
    await act(async () => { await result.current.onCreateDraft(); });
    await act(async () => { await result.current.onSaveDefinition(); });
    expect(api.upsertEntity).toHaveBeenCalledWith('col-manage', 'person', original.draft_id, original.revision,
      expect.objectContaining({ description: 'local edit' }));
  });
  it('loads workspace on mount', async () => {
    const api = createMockApi();
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    expect(api.loadWorkspace).toHaveBeenCalledWith('col-manage');
    expect(result.current.editorState.envelope?.draft?.draft_id).toBe('10000000-0000-4000-8000-000000000001');
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

  it('opens a published definition for inspection when no draft exists', async () => {
    const publishedOnlyEnvelope = { ...editDraftEnvelope, draft: null };
    const api = createMockApi({
      loadWorkspace: vi.fn().mockResolvedValue({ ok: true, data: publishedOnlyEnvelope }),
    });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-edit', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));

    act(() => {
      result.current.onSelectDefinition('entity', 'person');
    });

    expect(result.current.editorState.selection).toEqual({ kind: 'entity', key: 'person' });
    expect(result.current.formBuffer).toMatchObject({
      open: true,
      definitionKind: 'entity',
      definitionKey: 'person',
      baseRevision: null,
    });
    expect(result.current.formBuffer.currentValues?.description).toBe('A person entity');
  });

  it('validates using the loaded draft identity', async () => {
    const api = createMockApi();
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      await result.current.onValidate?.();
    });
    expect(api.validate).toHaveBeenCalledWith('col-manage', '10000000-0000-4000-8000-000000000001', 5);
    expect(result.current.editorState.validation.status).toBe('valid');
  });

  it('discards draft through the adapter', async () => {
    const api = createMockApi();
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      await result.current.onDiscardDraft?.();
    });
    expect(api.discardDraft).toHaveBeenCalledWith('col-manage', '10000000-0000-4000-8000-000000000001', 5);
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
    await waitFor(() => expect(result.current.editorState.envelope?.draft?.draft_id).toBe('10000000-0000-4000-8000-000000000001'));
  });
});
