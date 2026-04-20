import React from 'react';
import { Collection } from './CollectionsTree';
import MoveCollectionModal from './MoveCollectionModal';
import CreateCollectionModal from './CreateCollectionModal';
import CollectionSettingsMenu from './CollectionSettingsMenu';
import UserManagementModal from './UserManagementModal';

export interface CollectionsPageContentProps {
  loading: boolean;
  error: string | null;
  successMessage: string | null;
  collections: Collection[];
  allCollections: Collection[];
  folderToMove: Collection | null;
  isModalOpen: boolean;
  isCreateModalOpen: boolean;
  isUserManagementModalOpen: boolean;
  selectedCollection: Collection | null;
  onDismissSuccess: () => void;
  onDismissError: () => void;
  onOpenCreateModal: () => void;
  onCollectionCardClick: (folder: Collection) => void;
  onMove: (folder: Collection) => void;
  onDelete: (folder: Collection) => void;
  onManageCollaborators: (folder: Collection) => void;
  onCloseMoveModal: () => void;
  onCloseCreateModal: () => void;
  onSubmitCreate: (c: Collection) => void;
  onMoveSubmit: (folderId: number, newParentId: number | null) => void;
  onCloseUserModal: () => void;
  onUserManagementSave: () => void;
}

const CollectionsPageContent: React.FC<CollectionsPageContentProps> = ({
  loading,
  error,
  successMessage,
  collections,
  allCollections,
  folderToMove,
  isModalOpen,
  isCreateModalOpen,
  isUserManagementModalOpen,
  selectedCollection,
  onDismissSuccess,
  onDismissError,
  onOpenCreateModal,
  onCollectionCardClick,
  onMove,
  onDelete,
  onManageCollaborators,
  onCloseMoveModal,
  onCloseCreateModal,
  onSubmitCreate,
  onMoveSubmit,
  onCloseUserModal,
  onUserManagementSave,
}) => (
  <div className="p-4">
    {successMessage && (
      <div className="bg-green-600 text-white p-4 mb-4 rounded flex items-center justify-between">
        <span>{successMessage}</span>
        <button type="button" onClick={onDismissSuccess} className="text-white">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    )}
    {error && (
      <div className="bg-red-600 text-white p-4 mb-4 rounded flex items-center justify-between">
        <span>{error}</span>
        <button type="button" onClick={onDismissError} className="text-white">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    )}
    {loading ? (
      <div>Loading collections...</div>
    ) : (
      <>
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold">My Collections</h1>
          <button
            type="button"
            onClick={onOpenCreateModal}
            className="bg-accent hover:bg-accent-dark text-text-normal px-4 py-2 rounded"
          >
            New Collection
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {collections.map((folder) => (
            <div
              key={folder.id}
              className="bg-scheme-shade_3 hover:bg-opacity-100 rounded-lg p-4 cursor-pointer transition duration-200 relative element-border"
              onClick={() => onCollectionCardClick(folder)}
            >
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-xl font-semibold mb-2">{folder.name}</h2>
                  <p className="text-text-normal mb-2">
                    {folder.document_count} documents • {folder.children_count} subcollections
                  </p>
                  <p className="text-text-normal text-sm">
                    Created: {new Date(folder.created_at).toLocaleDateString()}
                  </p>
                </div>
                <CollectionSettingsMenu
                  collection={folder}
                  onMove={onMove}
                  onDelete={onDelete}
                  onManageCollaborators={onManageCollaborators}
                />
              </div>
            </div>
          ))}
        </div>
      </>
    )}
    {folderToMove && (
      <MoveCollectionModal
        isOpen={isModalOpen}
        onClose={onCloseMoveModal}
        folder={folderToMove}
        collections={allCollections.filter((c) => c.id !== folderToMove.id)}
        onSubmit={onMoveSubmit}
      />
    )}
    <CreateCollectionModal isOpen={isCreateModalOpen} onClose={onCloseCreateModal} onSubmit={onSubmitCreate} />
    <UserManagementModal
      isOpen={isUserManagementModalOpen}
      onClose={onCloseUserModal}
      onSave={onUserManagementSave}
      collection={selectedCollection}
    />
  </div>
);

export default CollectionsPageContent;
