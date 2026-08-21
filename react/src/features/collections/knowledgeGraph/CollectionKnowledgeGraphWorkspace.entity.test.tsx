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

const removableEntityEnvelope = {
  ...editDraftEnvelope,
  draft: {
    ...editDraftEnvelope.draft!,
    entities: [
      {
        ...editDraftEnvelope.draft!.entities[0],
        capabilities: {
          ...editDraftEnvelope.draft!.entities[0].capabilities,
          removable: true,
        },
      },
    ],
  },
};

afterEach(() => cleanup());

function renderWorkspace(overrides: Partial<ComponentProps<typeof CollectionKnowledgeGraphWorkspace>> = {}) {
  const state: CollectionSchemaEditorState = {
    ...createInitialCollectionSchemaState(),
    phase: 'ready',
    envelope: editDraftEnvelope,
    selection: { kind: 'entity', key: 'person' },
  };
  const props: ComponentProps<typeof CollectionKnowledgeGraphWorkspace> = {
    collectionId: 'col-edit',
    collectionName: 'Demo Collection',
    initialCanEdit: true,
    initialCanManage: false,
    editorState: state,
    formBuffer: {
      ...createInitialSchemaFormBufferState(),
      open: true,
      definitionKind: 'entity',
      definitionKey: 'person',
      baseRevision: 2,
      initialValues: editDraftEnvelope.draft!.entities[0].values as unknown as Record<string, unknown>,
      currentValues: {
        ...(editDraftEnvelope.draft!.entities[0].values as unknown as Record<string, unknown>),
        description: 'Edited locally',
      },
      dirtyFields: ['description'],
    },
    history: null,
    historyLoading: false,
    historyError: null,
    conflictPreview: null,
    onSelectDefinition: vi.fn(),
    onFieldChange: vi.fn(),
    onSaveDefinition: vi.fn(),
    onRemoveDefinition: vi.fn(),
    ...overrides,
  };
  return render(<CollectionKnowledgeGraphWorkspace {...props} />);
}

describe('CollectionKnowledgeGraphWorkspace entity actions', () => {
  it('calls onSaveDefinition from entity editor', () => {
    const onSaveDefinition = vi.fn();
    renderWorkspace({ onSaveDefinition });
    fireEvent.change(screen.getByLabelText('Description'), { target: { value: 'Edited locally' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(onSaveDefinition).toHaveBeenCalled();
  });

  it('requires confirmation before remove', () => {
    const onRemoveDefinition = vi.fn();
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: removableEntityEnvelope,
      selection: { kind: 'entity', key: 'person' },
    };
    renderWorkspace({
      onRemoveDefinition,
      editorState: state,
    });
    const removeButtons = screen.getAllByRole('button', { name: 'Remove' });
    fireEvent.click(removeButtons[removeButtons.length - 1]);
    expect(onRemoveDefinition).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Confirm removal' }));
    expect(onRemoveDefinition).toHaveBeenCalled();
  });
});
