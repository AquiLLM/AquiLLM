import React, { useCallback, useState } from 'react';
import type { Collection } from '../../../components/CollectionsTree';
import type { FileSystemItem } from '../../../types/FileSystemItem';
import { getCookie } from '../../../utils/csrf';
import formatUrl from '../../../utils/formatUrl';
import CollectionSchemaNavigationGuard from './CollectionSchemaNavigationGuard';
import { buildCollectionBreadcrumbs } from './collectionViewBreadcrumbs';
import CollectionViewGuardedContent from './collectionViewGuardedContent';
import type { CollectionViewProps } from './collectionViewTypes';
import {
  buildCollectionViewUrl,
  parseCollectionViewMode,
  type CollectionViewNavigationIntent,
} from './collectionViewTypes';
import { useCollectionViewMoveBatch } from './useCollectionViewMoveBatch';
import { useCollectionViewData } from './useCollectionViewData';

const CollectionView: React.FC<CollectionViewProps> = ({ collectionId, onBack }) => {
  const {
    collection,
    contents,
    loading,
    error,
    permissionSource,
    allCollections,
    initialCanEdit,
    initialCanManage,
    fetchCollectionData,
    refreshAllCollections,
    setContents,
  } = useCollectionViewData(collectionId);

  const [activeMode, setActiveMode] = useState(() =>
    parseCollectionViewMode(window.location.search)
  );
  const [movingItem, setMovingItem] = useState<FileSystemItem | Collection | null>(null);
  const [isMoveModalOpen, setIsMoveModalOpen] = useState(false);
  const [batchMovingItems, setBatchMovingItems] = useState<FileSystemItem[]>([]);
  const [isBatchMoveModalOpen, setIsBatchMoveModalOpen] = useState(false);
  const [isBatchOperationLoading, setIsBatchOperationLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [isCreateSubcollectionOpen, setIsCreateSubcollectionOpen] = useState(false);
  const [isUserManagementModalOpen, setIsUserManagementModalOpen] = useState(false);

  const proceedNavigation = useCallback((intent: CollectionViewNavigationIntent) => {
    const nextUrl = buildCollectionViewUrl(
      window.location.pathname,
      window.location.search,
      window.location.hash,
      intent.mode,
    );
    window.history.pushState(null, '', nextUrl);
    setActiveMode(intent.mode);
  }, []);

  const {
    handleMoveSubmit,
    handleBatchMove,
    handleBatchMoveSubmit,
    handleBatchRemoveItems,
  } = useCollectionViewMoveBatch({
    movingItem,
    setMovingItem,
    setIsMoveModalOpen,
    batchMovingItems,
    setBatchMovingItems,
    setIsBatchMoveModalOpen,
    setIsBatchOperationLoading,
    setSuccessMessage,
    setContents,
    allCollections,
  });

  const handleDelete = () => {
    if (collection && window.confirm(`Are you sure you want to delete "${collection.name}"?`)) {
      fetch(formatUrl(window.apiUrls.api_delete_collection, { collection_id: collection.id }), {
        method: 'DELETE',
        headers: { 'X-CSRFToken': getCookie('csrftoken') ?? '' },
        credentials: 'include',
      })
        .then((res) => {
          if (!res.ok) throw new Error('Failed to delete collection');
          if (onBack) onBack();
          else window.location.href = window.pageUrls.user_collections;
        })
        .catch((err) => {
          console.error('Error:', err);
          alert('Failed to delete collection. Please try again.');
        });
    }
  };

  const handleRemoveItem = (item: FileSystemItem) => {
    if (window.confirm(`Are you sure you want to remove "${item.name}"?`)) {
      const endpoint =
        item.type === 'collection'
          ? formatUrl(window.apiUrls.api_delete_collection, { collection_id: item.id })
          : formatUrl(window.apiUrls.api_delete_document, { doc_id: item.id });

      fetch(endpoint, {
        method: 'DELETE',
        headers: { 'X-CSRFToken': getCookie('csrftoken') ?? '' },
        credentials: 'include',
      })
        .then((res) => {
          if (!res.ok) throw new Error(`Failed to remove ${item.type}`);
          setContents((prev) => prev.filter((contentItem) => contentItem.id !== item.id));
        })
        .catch((err) => {
          console.error('Error:', err);
          alert(`Failed to remove ${item.type}. Please try again.`);
        });
    }
  };

  const handleCreateSubcollection = (newCollection: Collection) => {
    if (!collection) return;
    fetch(window.apiUrls.api_collections, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken') ?? '',
      },
      body: JSON.stringify({ name: newCollection.name, parent_id: collection.id }),
      credentials: 'include',
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to create subcollection');
        setIsCreateSubcollectionOpen(false);
        setSuccessMessage(`Created subcollection "${newCollection.name}"`);
        setTimeout(() => setSuccessMessage(null), 3000);
        fetchCollectionData();
        refreshAllCollections();
      })
      .catch((err) => {
        console.error('Error creating subcollection:', err);
        alert('Failed to create subcollection. Please try again.');
      });
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!collection) return <div>Collection not found</div>;

  const breadcrumbs = buildCollectionBreadcrumbs(collection, allCollections);

  return (
    <CollectionSchemaNavigationGuard onProceedNavigation={proceedNavigation}>
      <CollectionViewGuardedContent
        collection={collection}
        collectionId={collectionId}
        breadcrumbs={breadcrumbs}
        contents={contents}
        permissionSource={permissionSource}
        allCollections={allCollections}
        activeMode={activeMode}
        onActiveModeChange={setActiveMode}
        initialCanEdit={initialCanEdit}
        initialCanManage={initialCanManage}
        movingItem={movingItem}
        isMoveModalOpen={isMoveModalOpen}
        batchMovingItems={batchMovingItems}
        isBatchMoveModalOpen={isBatchMoveModalOpen}
        isCreateSubcollectionOpen={isCreateSubcollectionOpen}
        successMessage={successMessage}
        isBatchOperationLoading={isBatchOperationLoading}
        isUserManagementModalOpen={isUserManagementModalOpen}
        onBack={() => (onBack ? onBack() : window.history.back())}
        onManageCollaborators={() => setIsUserManagementModalOpen(true)}
        onDelete={handleDelete}
        onOpenCollectionSettingsMove={() => {
          setMovingItem({ id: collection.id, type: 'collection', name: collection.name });
          setIsMoveModalOpen(true);
        }}
        onOpenCreateSubcollection={() => setIsCreateSubcollectionOpen(true)}
        onCloseCreateSubcollection={() => setIsCreateSubcollectionOpen(false)}
        onSubmitCreateSubcollection={handleCreateSubcollection}
        onCloseMoveModal={() => {
          setIsMoveModalOpen(false);
          setMovingItem(null);
        }}
        onMoveSubmit={handleMoveSubmit}
        onCloseBatchMoveModal={() => {
          setIsBatchMoveModalOpen(false);
          setBatchMovingItems([]);
        }}
        onBatchMoveSubmit={handleBatchMoveSubmit}
        fetchCollectionData={fetchCollectionData}
        onOpenItem={(item) => {
          if (item.type === 'collection') {
            window.location.href = formatUrl(window.pageUrls.collection, { col_id: item.id });
          } else {
            window.location.href = formatUrl(window.pageUrls.document, { doc_id: item.id });
          }
        }}
        onRemoveItem={handleRemoveItem}
        onContextMove={(item) => {
          setMovingItem(item);
          setIsMoveModalOpen(true);
        }}
        onRenameItem={() => undefined}
        onBatchMove={handleBatchMove}
        onBatchRemove={handleBatchRemoveItems}
        onCloseUserManagement={() => setIsUserManagementModalOpen(false)}
        onUserManagementSave={() => {
          setSuccessMessage('Permissions updated successfully!');
          setTimeout(() => setSuccessMessage(null), 3000);
          fetchCollectionData();
        }}
      />
    </CollectionSchemaNavigationGuard>
  );
};

export default CollectionView;
