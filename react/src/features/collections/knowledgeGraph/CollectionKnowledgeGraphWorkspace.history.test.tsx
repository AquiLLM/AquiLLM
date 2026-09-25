// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import CollectionKnowledgeGraphWorkspace from './CollectionKnowledgeGraphWorkspace';
import {
  createInitialCollectionSchemaState,
  type CollectionSchemaEditorState,
} from './collectionSchemaReducer';
import { createInitialSchemaFormBufferState } from './schemaFormBuffer';
import { historyPageFixture, manageDraftEnvelope } from './schemaTestFixtures';

afterEach(() => cleanup());

describe('CollectionKnowledgeGraphWorkspace history', () => {
  it('loads history when panel opens and offers restore for MANAGE', () => {
    const onLoadHistory = vi.fn();
    const onRestoreVersion = vi.fn();
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: manageDraftEnvelope,
      selection: null,
    };

    render(
      <CollectionKnowledgeGraphWorkspace
        collectionId="col-manage"
        collectionName="Manage Collection"
        initialCanEdit={true}
        initialCanManage={true}
        editorState={state}
        formBuffer={createInitialSchemaFormBufferState()}
        history={historyPageFixture}
        historyLoading={false}
        historyError={null}
        conflictPreview={null}
        onSelectDefinition={vi.fn()}
        onFieldChange={vi.fn()}
        onLoadHistory={onLoadHistory}
        onRestoreVersion={onRestoreVersion}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'History' }));
    expect(onLoadHistory).toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: /Version 3/i }));
    fireEvent.click(screen.getByRole('button', { name: 'Restore version' }));
    expect(onRestoreVersion).toHaveBeenCalled();
  });
});
