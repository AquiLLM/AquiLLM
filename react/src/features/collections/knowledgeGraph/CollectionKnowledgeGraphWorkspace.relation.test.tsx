// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import type { ComponentProps } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import CollectionKnowledgeGraphWorkspace from './CollectionKnowledgeGraphWorkspace';
import {
  createInitialCollectionSchemaState,
  type CollectionSchemaEditorState,
} from './collectionSchemaReducer';
import { createInitialSchemaFormBufferState } from './schemaFormBuffer';
import { editDraftEnvelope } from './schemaTestFixtures';

afterEach(() => cleanup());

describe('CollectionKnowledgeGraphWorkspace relation actions', () => {
  it('selects relation and exposes save for editable fields', () => {
    const onSaveDefinition = vi.fn();
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: editDraftEnvelope,
      selection: { kind: 'relation', key: 'works_for' },
    };
    const relation = editDraftEnvelope.draft!.relations[0];
    const props: ComponentProps<typeof CollectionKnowledgeGraphWorkspace> = {
      collectionId: 'col-edit',
      collectionName: 'Demo Collection',
      initialCanEdit: true,
      initialCanManage: false,
      editorState: state,
      formBuffer: {
        ...createInitialSchemaFormBufferState(),
        open: true,
        definitionKind: 'relation',
        definitionKey: 'works_for',
        baseRevision: 2,
        initialValues: relation.values as unknown as Record<string, unknown>,
        currentValues: relation.values as unknown as Record<string, unknown>,
      dirtyFields: ['description'],
    },
      history: null,
      historyLoading: false,
      historyError: null,
      conflictPreview: null,
      onSelectDefinition: vi.fn(),
      onFieldChange: vi.fn(),
      onSaveDefinition,
    };

    render(<CollectionKnowledgeGraphWorkspace {...props} />);
    expect(screen.getByRole('listbox', { name: 'Allowed head entity types' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(onSaveDefinition).toHaveBeenCalled();
  });
});
