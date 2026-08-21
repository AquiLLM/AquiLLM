// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { CollectionSchemaApi } from './collectionSchemaApi';
import { manageDraftEnvelope, validationResultFixture } from './schemaTestFixtures';
import { useCollectionSchemaEditor } from './useCollectionSchemaEditor';

function createMockApi(overrides: Partial<CollectionSchemaApi> = {}): CollectionSchemaApi {
  return {
    loadWorkspace: vi.fn().mockResolvedValue({ ok: true, data: manageDraftEnvelope }),
    createDraft: vi.fn(),
    upsertEntity: vi.fn(),
    deleteEntity: vi.fn(),
    upsertRelation: vi.fn(),
    deleteRelation: vi.fn(),
    validate: vi.fn().mockResolvedValue({ ok: true, data: validationResultFixture }),
    fetchDiff: vi.fn(),
    publish: vi.fn().mockResolvedValue({ ok: true, data: { ...manageDraftEnvelope, draft: null } }),
    discardDraft: vi.fn(),
    listVersions: vi.fn(),
    fetchVersionDiff: vi.fn(),
    restoreVersion: vi.fn(),
    restoreReplace: vi.fn(),
    ...overrides,
  };
}

describe('useCollectionSchemaPublishing', () => {
  it('publishes with validated draft identity and checksum', async () => {
    const api = createMockApi();
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      await result.current.onValidate?.();
    });
    await act(async () => {
      await result.current.onPublish?.();
    });
    expect(api.publish).toHaveBeenCalledWith(
      'col-manage',
      {
        draft_id: 'draft-manage-1',
        revision: 5,
        candidate_checksum: 'candidate-checksum-v5',
        validation_result_id: 'validation-result-1',
      },
      5,
    );
    expect(result.current.editorState.publish.status).toBe('succeeded');
  });

  it('ignores stale validation responses', async () => {
    let resolveValidate: ((value: unknown) => void) | undefined;
    const api = createMockApi({
      validate: vi
        .fn()
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              resolveValidate = resolve;
            }),
        )
        .mockResolvedValue({ ok: true, data: validationResultFixture }),
    });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      void result.current.onValidate?.();
      await result.current.onValidate?.();
    });
    await act(async () => {
      resolveValidate?.({ ok: true, data: validationResultFixture });
    });
    await waitFor(() => expect(result.current.editorState.validation.status).toBe('valid'));
  });

  it('keeps draft intact when publish fails', async () => {
    const api = createMockApi({
      publish: vi.fn().mockResolvedValue({ ok: false, kind: 'server_error' }),
    });
    const { result } = renderHook(() => useCollectionSchemaEditor({ collectionId: 'col-manage', api }));
    await waitFor(() => expect(result.current.editorState.phase).toBe('ready'));
    await act(async () => {
      await result.current.onValidate?.();
      await result.current.onPublish?.();
    });
    expect(result.current.editorState.publish.status).toBe('failed');
    expect(result.current.editorState.envelope?.draft).not.toBeNull();
  });
});
