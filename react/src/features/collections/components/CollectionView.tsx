import React, { useState, useEffect, useCallback } from 'react';
import type { Collection } from '../../../components/CollectionsTree';
import type { FileSystemItem } from '../../../types/FileSystemItem';
import { getCookie } from '../../../utils/csrf';
import formatUrl from '../../../utils/formatUrl';
import { buildCollectionBreadcrumbs } from './collectionViewBreadcrumbs';
import { buildOrderedCollectionContents } from './collectionViewContents';
import type { CollectionContent, CollectionViewProps } from './collectionViewTypes';
import CollectionViewShell from './CollectionViewShell';
import { useCollectionViewMoveBatch } from './useCollectionViewMoveBatch';

const CollectionView: React.FC<CollectionViewProps> = ({ collectionId, onBack }) => {
  const [collection, setCollection] = useState<Collection | null>(null);
  const [contents, setContents] = useState<CollectionContent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [movingItem, setMovingItem] = useState<FileSystemItem | Collection | null>(null);
  const [isMoveModalOpen, setIsMoveModalOpen] = useState(false);
  const [allCollections, setAllCollections] = useState<Collection[]>([]);
  const [batchMovingItems, setBatchMovingItems] = useState<FileSystemItem[]>([]);
  const [isBatchMoveModalOpen, setIsBatchMoveModalOpen] = useState(false);
  const [isBatchOperationLoading, setIsBatchOperationLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [permissionSource, setPermissionSource] = useState<{
    direct: boolean;
    source_collection_id: number | null;
    source_collection_name: string | null;
    permission_level: string | null;
  } | null>(null);
  const [isUserManagementModalOpen, setIsUserManagementModalOpen] = useState(false);

  const fetchCollectionData = useCallback(() => {
    setLoading(true);
    fetch(formatUrl(window.apiUrls.api_collection, { col_id: collectionId }), {
      headers: { Accept: 'application/json' },
    })
      .then((res) => {
        if (!res.ok) {
          return res.json().then((err) => {
            throw new Error(err.error || 'Failed to fetch collection');
          });
        }
        return res.json();
      })
      .then((data) => {
        if (!data.collection) throw new Error('Invalid response format');
        if (data.permission_source) setPermissionSource(data.permission_source);
        setCollection({
          id: data.collection.id,
          name: data.collection.name,
          parent: data.collection.parent,
          collection: data.collection.id,
          path: data.collection.path,
          children: data.children || [],
          document_count: data.documents?.length || 0,
          children_count: data.children?.length || 0,
          created_at: data.collection.created_at
            ? new Date(data.collection.created_at).toLocaleString()
            : new Date().toLocaleString(),
          updated_at: data.collection.updated_at
            ? new Date(data.collection.updated_at).toISOString()
            : new Date().toISOString(),
        });
        const orderedContents = buildOrderedCollectionContents(
          data.documents || [],
          data.children || []
        );
        setContents(orderedContents);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error refetching collection:', err);
        setError(err.message);
        setLoading(false);
      });
  }, [collectionId]);

  useEffect(() => {
    fetchCollectionData();
  }, [fetchCollectionData]);

  useEffect(() => {
    fetch(window.apiUrls.api_collections, {
      headers: {
        Accept: 'application/json',
      },
      credentials: 'include',
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error('Failed to fetch available collections');
        }
        return res.json();
      })
      .then((data) => {
        const collectionsData = data.collections || [];
        const parsed = collectionsData.map((col: any) => ({
          id: col.id,
          name: col.name,
          parent: col.parent,
          document_count: col.document_count,
          path: col.path,
          children: [],
          children_count: 0,
          created_at: '',
          updated_at: '',
        }));
        setAllCollections(parsed);
      })
      .catch((err) => {
        console.error('Error fetching all collections:', err);
      });
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

  const handleRenameItem = () => {
    console.log('Rename item clicked');
  };

  const handleBack = () => {
    if (onBack) {
      onBack();
    } else {
      window.history.back();
    }
  };

  const handleManageCollaborators = () => {
    setIsUserManagementModalOpen(true);
  };

  const handleDelete = () => {
    if (collection && window.confirm(`Are you sure you want to delete "${collection.name}"?`)) {
      fetch(formatUrl(window.apiUrls.api_delete_collection, { collection_id: collection.id }), {
        method: 'DELETE',
        headers: { 'X-CSRFToken': getCookie('csrftoken') },
        credentials: 'include',
      })
        .then((res) => {
          if (!res.ok) throw new Error('Failed to delete collection');
          if (onBack) {
            onBack();
          } else {
            window.location.href = window.pageUrls.user_collections;
          }
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
        headers: { 'X-CSRFToken': getCookie('csrftoken') },
        credentials: 'include',
      })
        .then((res) => {
          if (!res.ok) throw new Error(`Failed to remove ${item.type}`);
          setContents((prevContents) => prevContents.filter((contentItem) => contentItem.id !== item.id));
        })
        .catch((err) => {
          console.error('Error:', err);
          alert(`Failed to remove ${item.type}. Please try again.`);
        });
    }
  };

  const handleOpenItem = (item: FileSystemItem) => {
    if (item.type === 'collection') {
      window.location.href = formatUrl(window.pageUrls.collection, { col_id: item.id });
    } else {
      window.location.href = formatUrl(window.pageUrls.document, { doc_id: item.id });
    }
  };

  const handleContextMove = (item: FileSystemItem) => {
    setMovingItem(item);
    setIsMoveModalOpen(true);
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!collection) return <div>Collection not found</div>;

  const breadcrumbs = buildCollectionBreadcrumbs(collection, allCollections);

  return (
    <CollectionViewShell
      collection={collection}
      collectionId={collectionId}
      breadcrumbs={breadcrumbs}
      contents={contents}
      permissionSource={permissionSource}
      allCollections={allCollections}
      movingItem={movingItem}
      isMoveModalOpen={isMoveModalOpen}
      batchMovingItems={batchMovingItems}
      isBatchMoveModalOpen={isBatchMoveModalOpen}
      successMessage={successMessage}
      isBatchOperationLoading={isBatchOperationLoading}
      isUserManagementModalOpen={isUserManagementModalOpen}
      onBack={handleBack}
      onManageCollaborators={handleManageCollaborators}
      onDelete={handleDelete}
      onOpenCollectionSettingsMove={() => {
        setMovingItem({ id: collection.id, type: 'collection', name: collection.name });
        setIsMoveModalOpen(true);
      }}
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
      onOpenItem={handleOpenItem}
      onRemoveItem={handleRemoveItem}
      onContextMove={handleContextMove}
      onRenameItem={handleRenameItem}
      onBatchMove={handleBatchMove}
      onBatchRemove={handleBatchRemoveItems}
      onCloseUserManagement={() => setIsUserManagementModalOpen(false)}
      onUserManagementSave={() => {
        setSuccessMessage('Permissions updated successfully!');
        setTimeout(() => setSuccessMessage(null), 3000);
        fetchCollectionData();
      }}
    />
  );
};

export default CollectionView;
