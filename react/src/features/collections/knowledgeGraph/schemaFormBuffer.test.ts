import { describe, expect, it } from 'vitest';

import {
  applyReviewedRebase,
  createInitialSchemaFormBufferState,
  getDirtyFields,
  previewReviewedRebase,
  schemaFormBufferReducer,
} from './schemaFormBuffer';

describe('schemaFormBuffer opening and editing', () => {
  it('opens a definition with base revision and initial values', () => {
    const next = schemaFormBufferReducer(createInitialSchemaFormBufferState(), {
      type: 'form/open',
      kind: 'entity',
      key: 'person',
      baseRevision: 2,
      values: { description: 'Draft description', aliases: ['individual'] },
    });
    expect(next.open).toBe(true);
    expect(next.definitionKey).toBe('person');
    expect(next.baseRevision).toBe(2);
    expect(next.currentValues).toEqual({
      description: 'Draft description',
      aliases: ['individual'],
    });
    expect(getDirtyFields(next)).toEqual([]);
  });

  it('tracks dirty fields while editing', () => {
    const opened = schemaFormBufferReducer(createInitialSchemaFormBufferState(), {
      type: 'form/open',
      kind: 'entity',
      key: 'person',
      baseRevision: 2,
      values: { description: 'Draft description' },
    });
    const edited = schemaFormBufferReducer(opened, {
      type: 'form/edit',
      field: 'description',
      value: 'Local unsaved description',
    });
    expect(getDirtyFields(edited)).toEqual(['description']);
    expect(edited.currentValues?.description).toBe('Local unsaved description');
  });
});

describe('schemaFormBuffer revert and discard', () => {
  it('reverts edited fields back to the opened initial values', () => {
    const opened = schemaFormBufferReducer(createInitialSchemaFormBufferState(), {
      type: 'form/open',
      kind: 'entity',
      key: 'person',
      baseRevision: 2,
      values: { description: 'Draft description' },
    });
    const edited = schemaFormBufferReducer(opened, {
      type: 'form/edit',
      field: 'description',
      value: 'Local unsaved description',
    });
    const reverted = schemaFormBufferReducer(edited, { type: 'form/revert' });
    expect(reverted.currentValues).toEqual(opened.initialValues);
    expect(getDirtyFields(reverted)).toEqual([]);
  });

  it('discards local buffer state on discard-local', () => {
    const opened = schemaFormBufferReducer(createInitialSchemaFormBufferState(), {
      type: 'form/open',
      kind: 'entity',
      key: 'person',
      baseRevision: 2,
      values: { description: 'Draft description' },
    });
    const edited = schemaFormBufferReducer(opened, {
      type: 'form/edit',
      field: 'description',
      value: 'Local unsaved description',
    });
    const discarded = schemaFormBufferReducer(edited, { type: 'form/discard' });
    expect(discarded.open).toBe(false);
    expect(discarded.currentValues).toBeNull();
  });
});

describe('schemaFormBuffer save transitions', () => {
  it('marks pending on save start and resets on accepted save', () => {
    const opened = schemaFormBufferReducer(createInitialSchemaFormBufferState(), {
      type: 'form/open',
      kind: 'entity',
      key: 'person',
      baseRevision: 2,
      values: { description: 'Draft description' },
    });
    const edited = schemaFormBufferReducer(opened, {
      type: 'form/edit',
      field: 'description',
      value: 'Saved description',
    });
    const pending = schemaFormBufferReducer(edited, { type: 'form/save/started' });
    expect(pending.pending).toBe(true);
    const saved = schemaFormBufferReducer(pending, {
      type: 'form/save/succeeded',
      baseRevision: 3,
      values: { description: 'Saved description' },
    });
    expect(saved.pending).toBe(false);
    expect(saved.baseRevision).toBe(3);
    expect(getDirtyFields(saved)).toEqual([]);
  });

  it('preserves local edits when save is rejected', () => {
    const opened = schemaFormBufferReducer(createInitialSchemaFormBufferState(), {
      type: 'form/open',
      kind: 'entity',
      key: 'person',
      baseRevision: 2,
      values: { description: 'Draft description' },
    });
    const edited = schemaFormBufferReducer(opened, {
      type: 'form/edit',
      field: 'description',
      value: 'Local unsaved description',
    });
    const pending = schemaFormBufferReducer(edited, { type: 'form/save/started' });
    const rejected = schemaFormBufferReducer(pending, {
      type: 'form/save/rejected',
      conflictFields: ['description'],
    });
    expect(rejected.pending).toBe(false);
    expect(rejected.currentValues?.description).toBe('Local unsaved description');
    expect(rejected.conflictFields).toEqual(['description']);
  });
});

describe('schemaFormBuffer navigation and reload', () => {
  it('reports dirty state for navigation guards', () => {
    const opened = schemaFormBufferReducer(createInitialSchemaFormBufferState(), {
      type: 'form/open',
      kind: 'entity',
      key: 'person',
      baseRevision: 2,
      values: { description: 'Draft description' },
    });
    const dirty = schemaFormBufferReducer(opened, {
      type: 'form/edit',
      field: 'description',
      value: 'Local unsaved description',
    });
    expect(getDirtyFields(dirty).length).toBeGreaterThan(0);
  });

  it('closes the buffer without preserving dirty values', () => {
    const opened = schemaFormBufferReducer(createInitialSchemaFormBufferState(), {
      type: 'form/open',
      kind: 'entity',
      key: 'person',
      baseRevision: 2,
      values: { description: 'Draft description' },
    });
    const closed = schemaFormBufferReducer(opened, { type: 'form/close' });
    expect(closed.open).toBe(false);
    expect(closed.definitionKey).toBeNull();
  });

  it('reloads server values while keeping local dirty fields when requested', () => {
    const opened = schemaFormBufferReducer(createInitialSchemaFormBufferState(), {
      type: 'form/open',
      kind: 'entity',
      key: 'person',
      baseRevision: 4,
      values: { description: 'Draft description', aliases: ['individual'] },
    });
    const edited = schemaFormBufferReducer(opened, {
      type: 'form/edit',
      field: 'description',
      value: 'Local unsaved description',
    });
    const reloaded = schemaFormBufferReducer(edited, {
      type: 'form/reload',
      baseRevision: 6,
      values: { description: 'Server accepted description', aliases: ['individual'] },
      preserveDirty: true,
    });
    expect(reloaded.baseRevision).toBe(6);
    expect(reloaded.initialValues?.description).toBe('Server accepted description');
    expect(reloaded.currentValues?.description).toBe('Local unsaved description');
    expect(getDirtyFields(reloaded)).toEqual(['description']);
  });
});

describe('schemaFormBuffer reviewed rebase', () => {
  it('auto-stages non-overlapping local changes', () => {
    const preview = previewReviewedRebase(
      { description: 'Draft description', aliases: ['individual'] },
      { description: 'Draft description', aliases: ['individual', 'person'] },
      { description: 'Server accepted description', aliases: ['individual'] },
    );
    expect(preview.autoStaged).toEqual({ aliases: ['individual', 'person'] });
    expect(preview.conflicts).toEqual([]);
  });

  it('requires explicit resolution for overlapping fields and rebases to latest revision', () => {
    const initial = { description: 'Draft description', aliases: ['individual'] };
    const current = { description: 'Local unsaved description', aliases: ['individual', 'person'] };
    const latest = { description: 'Server accepted description', aliases: ['individual'] };
    const preview = previewReviewedRebase(initial, current, latest);
    expect(preview.conflicts).toEqual([
      {
        field: 'description',
        baseValue: 'Draft description',
        localValue: 'Local unsaved description',
        serverValue: 'Server accepted description',
      },
    ]);
    const rebased = applyReviewedRebase(
      preview,
      [{ field: 'description', choice: 'local' }],
      7,
      latest,
    );
    expect(rebased.baseRevision).toBe(7);
    expect(rebased.values).toEqual({
      description: 'Local unsaved description',
      aliases: ['individual', 'person'],
    });
  });

  it('rejects incomplete conflict resolution', () => {
    const preview = previewReviewedRebase(
      { description: 'Draft description' },
      { description: 'Local unsaved description' },
      { description: 'Server accepted description' },
    );
    expect(() => applyReviewedRebase(preview, [], 7, { description: 'Server accepted description' })).toThrow(
      /conflict/i,
    );
  });
});
