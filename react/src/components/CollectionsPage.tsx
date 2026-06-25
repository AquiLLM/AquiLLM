import React, { useState, useEffect } from 'react';
import { Collection } from './CollectionsTree';
import CollectionsPageContent from './CollectionsPageContent';
import { getCookie } from '../utils/csrf';
import formatUrl from '../utils/formatUrl';
import { mapCollectionFromApi, rootCollectionsFromParsed } from './collectionsPageMap';

const CollectionsPage: React.FC = () => {
  const [collections, setCollectionsToView] = useState<Collection[]>([]);
  const [allCollections, setAllCollections] = useState<Collection[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [folderToMove, setFolderToMove] = useState<Collection | null>(null);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState<boolean>(false);
  const [selectedCollection, setSelectedCollection] = useState<Collection | null>(null);
  const [isUserManagementModalOpen, setIsUserManagementModalOpen] = useState<boolean>(false);

  const apiUrl = window.apiUrls.api_collections;
  const detailUrlBase = window.pageUrls.collection;

  useEffect(() => {
    fetch(apiUrl, {
      credentials: 'include',
      headers: { Accept: 'application/json' },
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error('Failed to fetch collections');
        }
        return res.json();
      })
      .then((data) => {
        const collectionsData = data.collections || [];
        const parsedCollections = collectionsData.map((col: Parameters<typeof mapCollectionFromApi>[0]) =>
          mapCollectionFromApi(col)
        );
        setAllCollections(parsedCollections);
        setCollectionsToView(rootCollectionsFromParsed(parsedCollections));
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError('Error fetching collections');
        setLoading(false);
      });
  }, [apiUrl]);

  const handleMoveClick = (folder: Collection) => {
    setFolderToMove(folder);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setFolderToMove(null);
    setIsModalOpen(false);
  };

  const handleOpenCreateModal = () => setIsCreateModalOpen(true);
  const handleCloseCreateModal = () => setIsCreateModalOpen(false);

  const handleSubmitCreate = (newCollection: Collection) => {
    fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken'),
      },
      body: JSON.stringify({ name: newCollection.name }),
      credentials: 'include',
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error('Failed to create collection');
        }
        return fetch(apiUrl, { credentials: 'include' });
      })
      .then((res) => res.json())
      .then((data) => {
        const collectionsData = data.collections || [];
        const parsedCollections = collectionsData.map((col: Record<string, unknown>) =>
          mapCollectionFromApi(col as Parameters<typeof mapCollectionFromApi>[0])
        );
        setAllCollections(parsedCollections);
        setCollectionsToView(rootCollectionsFromParsed(parsedCollections));
        setIsCreateModalOpen(false);
      })
      .catch(() => {
        alert('Failed to create collection. Please try again.');
      });
  };

  const handleDeleteCollection = (collection: Collection) => {
    if (window.confirm(`Are you sure you want to delete "${collection.name}"?`)) {
      fetch(formatUrl(window.apiUrls.api_delete_collection, { collection_id: collection.id }), {
        method: 'DELETE',
        headers: { 'X-CSRFToken': getCookie('csrftoken') },
        credentials: 'include',
      })
        .then((res) => {
          if (!res.ok) throw new Error('Failed to delete collection');
          setCollectionsToView((prev) => prev.filter((c) => c.id !== collection.id));
        })
        .catch(() => {
          alert('Failed to delete collection. Please try again.');
        });
    }
  };

  const handleManageCollaborators = (folder: Collection) => {
    setSelectedCollection(folder);
    setIsUserManagementModalOpen(true);
  };

  const handleCloseUserManagementModal = () => {
    setIsUserManagementModalOpen(false);
    setSelectedCollection(null);
  };

  const handleUserManagementSave = () => {
    setSuccessMessage(`Collaborators for "${selectedCollection?.name}" updated successfully!`);
    setTimeout(() => setSuccessMessage(null), 5000);
    fetch(apiUrl, {
      credentials: 'include',
      headers: { Accept: 'application/json' },
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error('Failed to refresh collections');
        }
        return res.json();
      })
      .then((data) => {
        const collectionsData = data.collections || [];
        const parsedCollections = collectionsData.map((col: Record<string, unknown>) =>
          mapCollectionFromApi(col as Parameters<typeof mapCollectionFromApi>[0])
        );
        setAllCollections(parsedCollections);
        setCollectionsToView(rootCollectionsFromParsed(parsedCollections));
      })
      .catch(() => {
        setError('Failed to refresh collections after updating permissions');
        setTimeout(() => setError(null), 5000);
      });
  };

  const handleCollectionClick = (collection: Collection) => {
    if (detailUrlBase) {
      window.location.href = formatUrl(detailUrlBase, { col_id: collection.id });
    }
  };

  const handleMoveCollection = (folderId: number, newParentId: number | null) => {
    fetch(formatUrl(window.apiUrls.api_move_collection, { collection_id: folderId }), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken'),
      },
      body: JSON.stringify({ new_parent_id: newParentId }),
      credentials: 'include',
    })
      .then((res) => {
        if (!res.ok) {
          return res.json().then((data: { error?: string }) => {
            throw new Error(data.error || 'Failed to move collection');
          });
        }
        return res.json();
      })
      .then(() => {
        handleCloseModal();
      })
      .catch((err: Error) => {
        alert(`Error moving collection: ${err.message}`);
      });
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <CollectionsPageContent
      loading={loading}
      error={error}
      successMessage={successMessage}
      collections={collections}
      allCollections={allCollections}
      folderToMove={folderToMove}
      isModalOpen={isModalOpen}
      isCreateModalOpen={isCreateModalOpen}
      isUserManagementModalOpen={isUserManagementModalOpen}
      selectedCollection={selectedCollection}
      onDismissSuccess={() => setSuccessMessage(null)}
      onDismissError={() => setError(null)}
      onOpenCreateModal={handleOpenCreateModal}
      onCollectionCardClick={handleCollectionClick}
      onMove={handleMoveClick}
      onDelete={handleDeleteCollection}
      onManageCollaborators={handleManageCollaborators}
      onCloseMoveModal={handleCloseModal}
      onCloseCreateModal={handleCloseCreateModal}
      onSubmitCreate={handleSubmitCreate}
      onMoveSubmit={handleMoveCollection}
      onCloseUserModal={handleCloseUserManagementModal}
      onUserManagementSave={handleUserManagementSave}
    />
  );
};

export default CollectionsPage;
