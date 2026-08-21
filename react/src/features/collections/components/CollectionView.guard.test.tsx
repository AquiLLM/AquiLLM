// @vitest-environment jsdom

import React, { useEffect } from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import CollectionViewGuardedContent from './collectionViewGuardedContent';
import CollectionSchemaNavigationGuard, {
  useSchemaNavigationGuard,
} from './CollectionSchemaNavigationGuard';
import type { Collection } from '../../../components/CollectionsTree';

vi.mock('./CollectionViewShell', () => ({
  default: ({
    guardNavigation,
    knowledgeGraphContent,
    onActiveModeChange,
  }: {
    guardNavigation?: (intent: { type: string; mode: string }) => boolean;
    knowledgeGraphContent: React.ReactNode;
    onActiveModeChange: (mode: string) => void;
  }) => (
    <div>
      <button
        type="button"
        data-testid="switch-to-files"
        onClick={() => {
          if (guardNavigation?.({ type: 'mode', mode: 'files' }) === false) return;
          onActiveModeChange('files');
        }}
      >
        Switch to Files
      </button>
      {knowledgeGraphContent}
    </div>
  ),
}));

vi.mock('../knowledgeGraph/CollectionKnowledgeGraphWorkspace', () => ({
  default: () => <div data-testid="kg-workspace" />,
}));

vi.mock('../knowledgeGraph/useCollectionSchemaEditor', () => ({
  useCollectionSchemaEditor: () => ({
    editorState: { phase: 'ready' },
    formBuffer: { open: false, dirtyFields: [] },
    onSelectDefinition: vi.fn(),
    onFieldChange: vi.fn(),
  }),
}));

const collection: Collection = {
  id: 1,
  name: 'Demo',
  parent: null,
  collection: 1,
  path: '/demo',
  children: [],
  document_count: 0,
  children_count: 0,
  created_at: '2026-01-01',
  updated_at: '2026-01-01',
};

function DirtyRegistrar({ dirty }: { dirty: boolean }) {
  const { registerDirtyState } = useSchemaNavigationGuard();
  useEffect(() => {
    registerDirtyState({ isDirty: dirty, discard: vi.fn() });
    return () => registerDirtyState(null);
  }, [dirty, registerDirtyState]);
  return null;
}

const baseProps = {
  collection,
  collectionId: '1',
  breadcrumbs: [],
  contents: [],
  permissionSource: null,
  allCollections: [collection],
  activeMode: 'knowledge-graph' as const,
  onActiveModeChange: vi.fn(),
  initialCanEdit: true,
  initialCanManage: true,
  movingItem: null,
  isMoveModalOpen: false,
  batchMovingItems: [],
  isBatchMoveModalOpen: false,
  isCreateSubcollectionOpen: false,
  successMessage: null,
  isBatchOperationLoading: false,
  isUserManagementModalOpen: false,
  onBack: vi.fn(),
  onManageCollaborators: vi.fn(),
  onDelete: vi.fn(),
  onOpenCollectionSettingsMove: vi.fn(),
  onOpenCreateSubcollection: vi.fn(),
  onCloseCreateSubcollection: vi.fn(),
  onSubmitCreateSubcollection: vi.fn(),
  onCloseMoveModal: vi.fn(),
  onMoveSubmit: vi.fn(),
  onCloseBatchMoveModal: vi.fn(),
  onBatchMoveSubmit: vi.fn(),
  fetchCollectionData: vi.fn(),
  onOpenItem: vi.fn(),
  onRemoveItem: vi.fn(),
  onContextMove: vi.fn(),
  onRenameItem: vi.fn(),
  onBatchMove: vi.fn(),
  onBatchRemove: vi.fn(),
  onCloseUserManagement: vi.fn(),
  onUserManagementSave: vi.fn(),
};

afterEach(() => cleanup());

describe('CollectionView guard coordinator', () => {
  it('blocks mode switch when schema form is dirty', () => {
    const onActiveModeChange = vi.fn();
    render(
      <CollectionSchemaNavigationGuard onProceedNavigation={vi.fn()}>
        <DirtyRegistrar dirty />
        <CollectionViewGuardedContent {...baseProps} onActiveModeChange={onActiveModeChange} />
      </CollectionSchemaNavigationGuard>,
    );

    fireEvent.click(screen.getByTestId('switch-to-files'));
    expect(screen.getByText('Unsaved schema changes')).toBeTruthy();
    expect(onActiveModeChange).not.toHaveBeenCalled();
  });

  it('completes navigation after discard confirmation', () => {
    const onProceedNavigation = vi.fn();
    render(
      <CollectionSchemaNavigationGuard onProceedNavigation={onProceedNavigation}>
        <DirtyRegistrar dirty />
        <CollectionViewGuardedContent {...baseProps} />
      </CollectionSchemaNavigationGuard>,
    );

    fireEvent.click(screen.getByTestId('switch-to-files'));
    fireEvent.click(screen.getByRole('button', { name: 'Discard unsaved form' }));
    expect(onProceedNavigation).toHaveBeenCalledWith({ type: 'mode', mode: 'files' });
  });
});
