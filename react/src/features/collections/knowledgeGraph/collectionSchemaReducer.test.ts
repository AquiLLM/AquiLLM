import { describe, expect, it } from 'vitest';

import {
  canPublish,
  canValidate,
  collectionSchemaReducer,
  createInitialCollectionSchemaState,
  selectedDefinition,
  type CollectionSchemaAction,
  type CollectionSchemaEditorState,
} from './collectionSchemaReducer';
import {
  conflictInfoFixture,
  editDraftEnvelope,
  envelopeAfterConflict,
  manageDraftEnvelope,
  validationResultFixture,
  viewPublishedEnvelope,
} from './schemaTestFixtures';
import type { CollectionSchemaEnvelope } from './schemaTypes';

type State = CollectionSchemaEditorState;

const reduce = (state: State, action: CollectionSchemaAction) => collectionSchemaReducer(state, action);

function loadEnvelope(collectionId: string, envelope: CollectionSchemaEnvelope, generation = 1): State {
  const started = reduce(createInitialCollectionSchemaState(), {
    type: 'load/started',
    collectionId,
    requestGeneration: generation,
  });
  return reduce(started, { type: 'load/succeeded', envelope, requestGeneration: generation });
}

function withValidation(state: State, generation: number, result = validationResultFixture): State {
  return reduce(reduce(state, { type: 'validate/started', requestGeneration: generation }), {
    type: 'validate/succeeded',
    result,
    requestGeneration: generation,
  });
}

const publishOperation = {
  draft_id: 'draft-manage-1',
  revision: 5,
  candidate_checksum: 'candidate-checksum-v5',
  validation_result_id: 'validation-result-1',
};

describe('collectionSchemaReducer initial/loading', () => {
  it('starts in initial phase with no envelope', () => {
    const state = createInitialCollectionSchemaState();
    expect(state.phase).toBe('initial');
    expect(state.envelope).toBeNull();
  });

  it('enters loading on load/started and tracks request generation', () => {
    const next = reduce(createInitialCollectionSchemaState(), {
      type: 'load/started',
      collectionId: 'col-view',
      requestGeneration: 1,
    });
    expect(next.phase).toBe('loading');
    expect(next.requestGeneration).toBe(1);
    expect(next.pendingOperation).toBe('load');
  });
});

describe('collectionSchemaReducer published/no-draft', () => {
  it('loads a VIEW published envelope without draft content', () => {
    const state = loadEnvelope('col-view', viewPublishedEnvelope);
    expect(state.envelope?.draft).toBeNull();
    expect(state.envelope?.permissions.level).toBe('VIEW');
  });
});

describe('collectionSchemaReducer draft/clean', () => {
  it('loads an EDIT shared draft envelope', () => {
    const state = loadEnvelope('col-edit', editDraftEnvelope);
    expect(state.envelope?.draft?.revision).toBe(2);
    expect(state.validation.status).toBe('idle');
  });
});

describe('collectionSchemaReducer mutation/pending', () => {
  it('marks mutation pending and clears validation identity', () => {
    const pending = reduce(withValidation(loadEnvelope('col-manage', manageDraftEnvelope), 2), {
      type: 'mutation/started',
      requestGeneration: 3,
    });
    expect(pending.pendingOperation).toBe('mutation');
    expect(pending.validation.result).toBeNull();
  });

  it('replaces the complete normalized envelope on accepted mutation', () => {
    const updated: CollectionSchemaEnvelope = {
      ...manageDraftEnvelope,
      draft: { ...manageDraftEnvelope.draft!, revision: 6, updated_at: '2026-08-21T11:00:00Z' },
    };
    const next = reduce(reduce(loadEnvelope('col-manage', manageDraftEnvelope), { type: 'mutation/started', requestGeneration: 2 }), {
      type: 'mutation/succeeded',
      envelope: updated,
      requestGeneration: 2,
    });
    expect(next.envelope).toEqual(updated);
    expect(next.validation.result).toBeNull();
  });
});

describe('collectionSchemaReducer validating/valid/invalid', () => {
  it('tracks validating then valid state with exact identity', () => {
    const validating = reduce(loadEnvelope('col-manage', manageDraftEnvelope), { type: 'validate/started', requestGeneration: 2 });
    const valid = withValidation(loadEnvelope('col-manage', manageDraftEnvelope), 2);
    expect(validating.validation.status).toBe('pending');
    expect(valid.validation.result?.identity).toEqual(validationResultFixture.identity);
  });

  it('tracks invalid validation results', () => {
    const invalid = withValidation(loadEnvelope('col-manage', manageDraftEnvelope), 2, {
      ...validationResultFixture,
      issues: [{ code: 'required_field', location: 'entity.person.name', message: 'Name is required', severity: 'error' }],
    });
    expect(invalid.validation.status).toBe('invalid');
    expect(canPublish(invalid)).toBe(false);
  });
});

describe('collectionSchemaReducer publishing/publish-polling', () => {
  it('tracks publish pending and polling without clearing the draft', () => {
    const publishing = reduce(withValidation(loadEnvelope('col-manage', manageDraftEnvelope), 2), {
      type: 'publish/started',
      operation: publishOperation,
      requestGeneration: 3,
    });
    const polling = reduce(publishing, {
      type: 'publish/polling',
      operation: { ...publishOperation, status_url: '/status/publish-1' },
      requestGeneration: 3,
    });
    expect(publishing.publish.status).toBe('pending');
    expect(polling.publish.status).toBe('polling');
    expect(polling.envelope?.draft).not.toBeNull();
  });

  it('clears draft and validation on publish success', () => {
    const publishing = reduce(loadEnvelope('col-manage', manageDraftEnvelope), {
      type: 'publish/started',
      operation: publishOperation,
      requestGeneration: 2,
    });
    const next = reduce(publishing, {
      type: 'publish/succeeded',
      envelope: {
        ...manageDraftEnvelope,
        draft: null,
        published: { ...manageDraftEnvelope.published, version: 5, checksum: 'candidate-checksum-v5' },
      },
      requestGeneration: 2,
    });
    expect(next.envelope?.draft).toBeNull();
    expect(next.publish.status).toBe('succeeded');
  });
});

describe('collectionSchemaReducer conflict', () => {
  it('reloads latest envelope, records conflict, and clears validation', () => {
    const conflicted = reduce(reduce(loadEnvelope('col-manage', manageDraftEnvelope), { type: 'mutation/started', requestGeneration: 2 }), {
      type: 'conflict/received',
      conflict: conflictInfoFixture,
      envelope: envelopeAfterConflict(manageDraftEnvelope),
      requestGeneration: 2,
    });
    expect(conflicted.conflict).toEqual(conflictInfoFixture);
    expect(conflicted.envelope?.draft?.revision).toBe(6);
  });
});

describe('collectionSchemaReducer unavailable/forbidden/read-only', () => {
  it('enters unavailable on schema_unavailable failures', () => {
    const next = reduce(reduce(createInitialCollectionSchemaState(), { type: 'load/started', collectionId: 'col-view', requestGeneration: 1 }), {
      type: 'load/failed',
      kind: 'schema_unavailable',
      requestGeneration: 1,
    });
    expect(next.phase).toBe('unavailable');
    expect(next.envelope).toBeNull();
  });

  it('enters forbidden on forbidden failures', () => {
    const next = reduce(reduce(createInitialCollectionSchemaState(), { type: 'load/started', collectionId: 'col-view', requestGeneration: 1 }), {
      type: 'load/failed',
      kind: 'forbidden',
      requestGeneration: 1,
    });
    expect(next.phase).toBe('forbidden');
  });

  it('drops retained draft when VIEW reload removes draft permissions', () => {
    const viewReload = reduce(
      reduce(loadEnvelope('col-edit', editDraftEnvelope), { type: 'load/started', collectionId: 'col-view', requestGeneration: 2 }),
      { type: 'load/succeeded', envelope: viewPublishedEnvelope, requestGeneration: 2 },
    );
    expect(viewReload.envelope?.draft).toBeNull();
    expect(viewReload.envelope?.permissions.level).toBe('VIEW');
  });
});

describe('collectionSchemaReducer selectors', () => {
  it('canValidate requires draft permissions and no pending validation', () => {
    const loaded = loadEnvelope('col-edit', editDraftEnvelope);
    expect(canValidate(loaded)).toBe(true);
    expect(canValidate(reduce(loaded, { type: 'validate/started', requestGeneration: 2 }))).toBe(false);
  });

  it('canPublish requires exact validation identity match for MANAGE', () => {
    expect(canPublish(loadEnvelope('col-manage', manageDraftEnvelope))).toBe(false);
    expect(canPublish(withValidation(loadEnvelope('col-manage', manageDraftEnvelope), 2))).toBe(true);
  });

  it('selectedDefinition survives when the definition still exists', () => {
    const selected = reduce(loadEnvelope('col-edit', editDraftEnvelope), {
      type: 'selection/changed',
      selection: { kind: 'entity', key: 'person' },
    });
    const next = reduce(reduce(selected, { type: 'mutation/started', requestGeneration: 2 }), {
      type: 'mutation/succeeded',
      envelope: { ...editDraftEnvelope, draft: { ...editDraftEnvelope.draft!, revision: 3 } },
      requestGeneration: 2,
    });
    expect(selectedDefinition(next)?.definition.key).toBe('person');
  });

  it('selectedDefinition falls back to null when removed', () => {
    const selected = reduce(loadEnvelope('col-edit', editDraftEnvelope), {
      type: 'selection/changed',
      selection: { kind: 'entity', key: 'person' },
    });
    const next = reduce(reduce(selected, { type: 'mutation/started', requestGeneration: 2 }), {
      type: 'mutation/succeeded',
      envelope: { ...editDraftEnvelope, draft: { ...editDraftEnvelope.draft!, revision: 3, entities: [] } },
      requestGeneration: 2,
    });
    expect(selectedDefinition(next)).toBeNull();
  });
});

describe('collectionSchemaReducer stale async responses', () => {
  it('ignores load success from an older request generation', () => {
    const stale = reduce(reduce(createInitialCollectionSchemaState(), { type: 'load/started', collectionId: 'col-view', requestGeneration: 2 }), {
      type: 'load/succeeded',
      envelope: viewPublishedEnvelope,
      requestGeneration: 1,
    });
    expect(stale.envelope).toBeNull();
    expect(stale.phase).toBe('loading');
  });

  it('ignores mutation success from an older request generation', () => {
    const stale = reduce(loadEnvelope('col-edit', editDraftEnvelope), {
      type: 'mutation/succeeded',
      envelope: { ...editDraftEnvelope, draft: { ...editDraftEnvelope.draft!, revision: 99 } },
      requestGeneration: 0,
    });
    expect(stale.envelope?.draft?.revision).toBe(2);
  });

  it('clears validation on discard and restore successes', () => {
    const validated = withValidation(loadEnvelope('col-manage', manageDraftEnvelope), 2);
    const discarded = reduce(reduce(validated, { type: 'discard/started', requestGeneration: 3 }), {
      type: 'discard/succeeded',
      envelope: { ...manageDraftEnvelope, draft: null },
      requestGeneration: 3,
    });
    const restored = reduce(reduce(validated, { type: 'restore/started', requestGeneration: 4 }), {
      type: 'restore/succeeded',
      envelope: editDraftEnvelope,
      requestGeneration: 4,
    });
    expect(discarded.validation.result).toBeNull();
    expect(restored.validation.result).toBeNull();
  });
});
