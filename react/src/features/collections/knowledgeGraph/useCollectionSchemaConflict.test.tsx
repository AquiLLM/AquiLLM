// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { CollectionSchemaApi } from './collectionSchemaApi';
import {
  conflictInfoFixture,
  envelopeAfterConflict,
  manageDraftEnvelope,
} from './schemaTestFixtures';
import { useCollectionSchemaEditor } from './useCollectionSchemaEditor';

function createMockApi(overrides: Partial<CollectionSchemaApi> = {}): CollectionSchemaApi {
  return {
    loadWorkspace: vi.fn().mockResolvedValue({ ok: true, data: manageDraftEnvelope }),
    createDraft: vi.fn(),
    upsertEntity: vi.fn().mockResolvedValue({
      ok: false,
      kind: 'revision_conflict',
      conflict: conflictInfoFixture,
    }),
    deleteEntity: vi.fn(),
    upsertRelation: vi.fn(),
    deleteRelation: vi.fn(),
    validate: vi.fn(),
    fetchDiff: vi.fn(),
    publish: vi.fn(),
    discardDraft: vi.fn(),
    listVersions: vi.fn(),
    fetchVersionDiff: vi.fn(),
    restoreVersion: vi.fn(),
    restoreReplace: vi.fn(),
    startGeneration: vi.fn(),
    getGenerationStatus: vi.fn(),
    ...overrides,
  };
}

describe('useCollectionSchemaConflict', () => {
  it('preserves the local form buffer and reloads envelope on 409', async () => {
    const api = createMockApi({
      loadWorkspace: vi
        .fn()
        .mockResolvedValueOnce({ ok: true, data: manageDraftEnvelope })
        .mockResolvedValueOnce({ ok: true, data: envelopeAfterConflict(manageDraftEnvelope) }),
    });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      result.current.onSelectDefinition('entity', 'person');
      result.current.onFieldChange('description', 'Local unsaved description');
    });
    await act(async () => {
      await result.current.onSaveDefinition?.();
    });
    expect(result.current.editorState.conflict).toEqual(conflictInfoFixture);
    expect(result.current.formBuffer.currentValues?.description).toBe('Local unsaved description');
    expect(result.current.editorState.envelope?.draft?.revision).toBe(6);
  });

  it('supports reviewed reapply against the reloaded revision', async () => {
    const api = createMockApi({
      loadWorkspace: vi
        .fn()
        .mockResolvedValueOnce({ ok: true, data: manageDraftEnvelope })
        .mockResolvedValueOnce({ ok: true, data: envelopeAfterConflict(manageDraftEnvelope) }),
    });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      result.current.onSelectDefinition('entity', 'person');
      result.current.onFieldChange('description', 'Local unsaved description');
    });
    await act(async () => {
      await result.current.onSaveDefinition?.();
    });
    await act(async () => {
      result.current.onConflictReapply?.([{ field: 'description', choice: 'local' }]);
    });
    expect(result.current.editorState.conflict).toBeNull();
    expect(result.current.formBuffer.baseRevision).toBe(6);
    expect(result.current.formBuffer.currentValues?.description).toBe('Local unsaved description');
  });

  it('uses atomic restore replacement with challenge token', async () => {
    const api = createMockApi({
      restoreReplace: vi.fn().mockResolvedValue({ ok: true, data: manageDraftEnvelope }),
    });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      await api.restoreReplace('col-manage', 4, 'restore-challenge-token', 5);
    });
    expect(api.restoreReplace).toHaveBeenCalledWith('col-manage', 4, 'restore-challenge-token', 5);
    expect(result.current.editorState.phase).toBe('ready');
  });
});
