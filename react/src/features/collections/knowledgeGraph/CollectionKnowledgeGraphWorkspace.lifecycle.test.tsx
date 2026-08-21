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
import { manageDraftEnvelope, validationResultFixture } from './schemaTestFixtures';

afterEach(() => cleanup());

describe('CollectionKnowledgeGraphWorkspace lifecycle', () => {
  it('disables publish until validation succeeds', () => {
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: manageDraftEnvelope,
      selection: null,
      validation: { status: 'idle', result: null, requestGeneration: 0 },
    };

    const props: ComponentProps<typeof CollectionKnowledgeGraphWorkspace> = {
      collectionId: 'col-manage',
      collectionName: 'Manage Collection',
      initialCanEdit: true,
      initialCanManage: true,
      editorState: state,
      formBuffer: createInitialSchemaFormBufferState(),
      history: null,
      historyLoading: false,
      historyError: null,
      conflictPreview: null,
      onSelectDefinition: vi.fn(),
      onFieldChange: vi.fn(),
      onValidate: vi.fn(),
      onPublish: vi.fn(),
    };

    render(<CollectionKnowledgeGraphWorkspace {...props} />);
    expect((screen.getByRole('button', { name: 'Publish' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('enables publish after valid validation identity matches draft', () => {
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: manageDraftEnvelope,
      selection: null,
      validation: { status: 'valid', result: validationResultFixture, requestGeneration: 1 },
    };

    render(
      <CollectionKnowledgeGraphWorkspace
        collectionId="col-manage"
        collectionName="Manage Collection"
        initialCanEdit={true}
        initialCanManage={true}
        editorState={state}
        formBuffer={createInitialSchemaFormBufferState()}
        history={null}
        historyLoading={false}
        historyError={null}
        conflictPreview={null}
        onSelectDefinition={vi.fn()}
        onFieldChange={vi.fn()}
        onPublish={vi.fn()}
      />,
    );

    expect((screen.getByRole('button', { name: 'Publish' }) as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Publish' }));
    expect(screen.getByRole('heading', { name: 'Publish schema draft' })).toBeTruthy();
  });
});
