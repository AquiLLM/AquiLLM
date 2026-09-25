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
import {
  editDraftEnvelope,
  emptyDraftEnvelope,
  emptyEditableEnvelope,
  historyPageFixture,
  manageDraftEnvelope,
  viewPublishedEnvelope,
} from './schemaTestFixtures';

afterEach(() => cleanup());

function renderWorkspace(
  editorState: CollectionSchemaEditorState,
  overrides: Partial<ComponentProps<typeof CollectionKnowledgeGraphWorkspace>> = {},
) {
  const props = {
    collectionId: 'col-1',
    collectionName: 'Demo Collection',
    initialCanEdit: true,
    initialCanManage: false,
    editorState,
    formBuffer: createInitialSchemaFormBufferState(),
    history: null,
    historyLoading: false,
    historyError: null,
    conflictPreview: null,
    onSelectDefinition: vi.fn(),
    onFieldChange: vi.fn(),
    ...overrides,
  };
  return render(<CollectionKnowledgeGraphWorkspace {...props} />);
}

describe('CollectionKnowledgeGraphWorkspace states', () => {
  it('shows loading status', () => {
    renderWorkspace({ ...createInitialCollectionSchemaState(), phase: 'loading' });
    expect(screen.getByTestId('schema-workspace-status').textContent).toMatch(/Loading schema workspace/i);
  });

  it('shows unavailable and forbidden messages', () => {
    renderWorkspace({ ...createInitialCollectionSchemaState(), phase: 'unavailable' });
    expect(screen.getByText(/unavailable/i)).toBeTruthy();

    cleanup();
    renderWorkspace({ ...createInitialCollectionSchemaState(), phase: 'forbidden' });
    expect(screen.getByText(/do not have permission/i)).toBeTruthy();
  });

  it('renders VIEW workspace without mutation controls even when initialCanEdit is true', () => {
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: viewPublishedEnvelope,
      selection: { kind: 'entity', key: 'person' },
    };

    renderWorkspace(state, { initialCanEdit: true });
    expect(screen.getByText(/View-only access/i)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Add entity type' })).toBeNull();
    expect(screen.queryByText(/Last editor/i)).toBeNull();
  });

  it('keeps a published definition read-only when an editor has no active draft', () => {
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: { ...editDraftEnvelope, draft: null },
      selection: { kind: 'entity', key: 'person' },
    };
    const values: Record<string, unknown> = { ...editDraftEnvelope.published.entities[0].values };

    renderWorkspace(state, {
      formBuffer: {
        ...createInitialSchemaFormBufferState(),
        open: true,
        definitionKind: 'entity',
        definitionKey: 'person',
        baseRevision: null,
        initialValues: values,
        currentValues: values,
      },
    });

    expect(screen.getByRole('heading', { name: 'Entity type: person' })).toBeTruthy();
    expect((screen.getByLabelText('Description') as HTMLTextAreaElement).disabled).toBe(true);
    expect(screen.queryByRole('button', { name: 'Save' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Add entity type' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Add relation type' })).toBeNull();
  });

  it('renders draft toolbar and navigation for EDIT draft', () => {
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: editDraftEnvelope,
      selection: { kind: 'entity', key: 'person' },
    };

    renderWorkspace(state);
    expect(screen.getByTestId('collection-knowledge-graph-workspace')).toBeTruthy();
    expect(screen.getByTestId('schema-nav-entity-person')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Validate' })).toBeTruthy();
  });

  it('filters definitions via search input', () => {
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: manageDraftEnvelope,
      selection: null,
    };

    renderWorkspace(state);
    fireEvent.change(screen.getByLabelText('Search definitions'), { target: { value: 'works_for' } });
    expect(screen.getByTestId('schema-nav-relation-works_for')).toBeTruthy();
    expect(screen.queryByTestId('schema-nav-entity-person')).toBeNull();
  });

  it('treats generated definitions as collection-scoped in the origin filter', () => {
    const generatedEntity = {
      ...manageDraftEnvelope.draft!.entities[0],
      key: 'generated_topic',
      origin: 'generated' as const,
      values: {
        ...manageDraftEnvelope.draft!.entities[0].values,
        name: 'generated_topic',
      },
    };
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: {
        ...manageDraftEnvelope,
        draft: {
          ...manageDraftEnvelope.draft!,
          entities: [...manageDraftEnvelope.draft!.entities, generatedEntity],
        },
      },
      selection: null,
    };

    renderWorkspace(state);
    fireEvent.change(screen.getByLabelText('Filter by origin'), { target: { value: 'collection' } });

    expect(screen.getByTestId('schema-nav-entity-generated_topic')).toBeTruthy();
    expect(screen.queryByTestId('schema-nav-entity-person')).toBeNull();
  });

  it('opens history panel when History is clicked', () => {
    const onLoadHistory = vi.fn();
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: manageDraftEnvelope,
      selection: null,
    };

    renderWorkspace(state, { history: historyPageFixture, onLoadHistory });
    fireEvent.click(screen.getByRole('button', { name: 'History' }));
    expect(onLoadHistory).toHaveBeenCalled();
    expect(screen.getByText('Version 4')).toBeTruthy();
  });

  it('announces status messages through aria-live', () => {
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: editDraftEnvelope,
      selection: null,
    };

    renderWorkspace(state, { statusMessage: 'Validation succeeded.' });
    expect(screen.getByText('Validation succeeded.')).toBeTruthy();
  });

  it('renders an editor-only Generate from collection action', () => {
    const onGenerateSchema = vi.fn();
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: emptyEditableEnvelope,
      selection: null,
    };

    renderWorkspace(state, { onGenerateSchema });
    fireEvent.click(screen.getByRole('button', { name: 'Generate from collection' }));
    expect(onGenerateSchema).toHaveBeenCalledTimes(1);
  });

  it('renders Generate from collection for an unchanged empty draft', () => {
    const onGenerateSchema = vi.fn();
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: emptyDraftEnvelope,
      selection: null,
    };

    renderWorkspace(state, { onGenerateSchema });
    fireEvent.click(screen.getByRole('button', { name: 'Generate from collection' }));
    expect(onGenerateSchema).toHaveBeenCalledTimes(1);
  });

  it('disables Generate from collection and announces queued progress', () => {
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: emptyEditableEnvelope,
      selection: null,
    };

    renderWorkspace(state, { generation: { status: 'queued', runId: 'run-1' }, onGenerateSchema: vi.fn() });
    expect(screen.getByRole('button', { name: 'Generate from collection' }).hasAttribute('disabled')).toBe(true);
    expect(screen.getByText('Schema generation queued.')).toBeTruthy();
  });

  it('announces generation failure and exposes manual retry', () => {
    const onGenerateSchema = vi.fn();
    const state: CollectionSchemaEditorState = {
      ...createInitialCollectionSchemaState(),
      phase: 'ready',
      envelope: emptyEditableEnvelope,
      selection: null,
    };

    renderWorkspace(state, {
      generation: { status: 'failed', runId: 'run-1', errorCode: 'no_collection_text' },
      onGenerateSchema,
    });
    expect(screen.getByText('Schema generation failed: no_collection_text.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Retry generation' }));
    expect(onGenerateSchema).toHaveBeenCalledTimes(1);
  });
});
