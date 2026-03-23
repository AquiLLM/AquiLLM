import React from 'react';
import type { Collection } from '../../../components/CollectionsTree';
import CollectionSettingsMenu from '../../../components/CollectionSettingsMenu';
import FileSystemViewer from '../../documents/components/FileSystemViewer';
import MoveCollectionModal from '../../../components/MoveCollectionModal';
import UserManagementModal from '../../platform_admin/components/UserManagementModal';
import type { FileSystemItem } from '../../../types/FileSystemItem';
import IngestRowContainer from '../../../components/IngestRow';
import formatUrl from '../../../utils/formatUrl';
import type { CollectionBreadcrumb, CollectionContent } from './collectionViewTypes';

export interface CollectionViewShellProps {
  collection: Collection;
  collectionId: string;
  breadcrumbs: CollectionBreadcrumb[];
  contents: CollectionContent[];
  permissionSource: {
    direct: boolean;
    source_collection_id: number | null;
    source_collection_name: string | null;
    permission_level: string | null;
  } | null;
  allCollections: Collection[];
  movingItem: FileSystemItem | Collection | null;
  isMoveModalOpen: boolean;
  batchMovingItems: FileSystemItem[];
  isBatchMoveModalOpen: boolean;
  successMessage: string | null;
  isBatchOperationLoading: boolean;
  isUserManagementModalOpen: boolean;
  onBack: () => void;
  onManageCollaborators: () => void;
  onDelete: () => void;
  onOpenCollectionSettingsMove: () => void;
  onCloseMoveModal: () => void;
  onMoveSubmit: (itemId: number, newParentId: number | null) => void;
  onCloseBatchMoveModal: () => void;
  onBatchMoveSubmit: (newParentId: number | null) => void;
  fetchCollectionData: () => void;
  onOpenItem: (item: FileSystemItem) => void;
  onRemoveItem: (item: FileSystemItem) => void;
  onContextMove: (item: FileSystemItem) => void;
  onRenameItem: () => void;
  onBatchMove: (items: FileSystemItem[]) => void;
  onBatchRemove: (items: FileSystemItem[]) => void;
  onCloseUserManagement: () => void;
  onUserManagementSave: () => void;
}

const CollectionViewShell: React.FC<CollectionViewShellProps> = ({
  collection,
  collectionId,
  breadcrumbs,
  contents,
  permissionSource,
  allCollections,
  movingItem,
  isMoveModalOpen,
  batchMovingItems,
  isBatchMoveModalOpen,
  successMessage,
  isBatchOperationLoading,
  isUserManagementModalOpen,
  onBack,
  onManageCollaborators,
  onDelete,
  onOpenCollectionSettingsMove,
  onCloseMoveModal,
  onMoveSubmit,
  onCloseBatchMoveModal,
  onBatchMoveSubmit,
  fetchCollectionData,
  onOpenItem,
  onRemoveItem,
  onContextMove,
  onRenameItem,
  onBatchMove,
  onBatchRemove,
  onCloseUserManagement,
  onUserManagementSave,
}) => (
  <div className="p-[24px] md:p-[32px]">
    <div className="px-[8px] md:px-[12px] mb-[24px]">
      <button
        onClick={onBack}
        className="h-[36px] px-3 rounded-[18px] bg-scheme-shade_4 text-text-slightly_less_contrast border border-border-mid_contrast hover:bg-scheme-shade_5 hover:text-text-normal transition-colors cursor-pointer inline-flex items-center justify-center mb-[12px]"
      >
        {'← Back'}
      </button>

      <nav className="flex mb-[8px]" aria-label="Breadcrumb">
        <ol className="inline-flex items-center space-x-1 md:space-x-3">
          {breadcrumbs.map((crumb, index) => (
            <li key={`${crumb.name}-${index}`} className="inline-flex items-center">
              {index > 0 && (
                <svg
                  className="w-3 h-3 mx-1 text-text-lower_contrast"
                  aria-hidden="true"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 6 10"
                >
                  <path
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="m1 9 4-4-4-4"
                  />
                </svg>
              )}

              {crumb.id !== null ? (
                <a
                  href={formatUrl(window.pageUrls.collection, { col_id: crumb.id })}
                  className={`ml-1 text-sm ${
                    index === breadcrumbs.length - 1
                      ? 'text-accent font-medium'
                      : 'text-text-slightly_less_contrast hover:text-accent'
                  }`}
                >
                  {crumb.name}
                </a>
              ) : (
                <a
                  href={window.pageUrls.user_collections}
                  className="ml-1 text-sm text-text-slightly_less_contrast hover:text-accent"
                >
                  {crumb.name}
                </a>
              )}
            </li>
          ))}
        </ol>
      </nav>

      <div className="flex items-center justify-between gap-4 border-b border-border-low_contrast pb-[10px]">
        <h1 className="text-[2.05rem] font-semibold leading-none text-text-normal">{collection.name}</h1>
        <CollectionSettingsMenu
          collection={collection}
          onManageCollaborators={onManageCollaborators}
          onDelete={onDelete}
          triggerLabel="Collection Settings"
          onMove={onOpenCollectionSettingsMove}
        />
      </div>
    </div>
    {permissionSource && !permissionSource.direct && permissionSource.source_collection_name && (
      <div className="mb-4 p-3 bg-accent bg-opacity-15 text-accent-light rounded-md flex items-center">
        <svg
          className="w-5 h-5 mr-2"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <span>
          You have access to this collection through{' '}
          <strong>{permissionSource.permission_level}</strong> permission inherited from parent collection:{' '}
          <strong>{permissionSource.source_collection_name}</strong>
        </span>
      </div>
    )}

    <div className="mb-[24px] bg-scheme-shade_4 border border-border-low_contrast rounded-[20px] p-[14px]">
      <IngestRowContainer
        ingestUploadsUrl={window.apiUrls.api_ingest_uploads}
        ingestArxivUrl={window.apiUrls.api_ingest_arxiv}
        ingestPdfUrl={window.apiUrls.api_ingest_pdf}
        ingestVttUrl={window.apiUrls.api_ingest_vtt}
        ingestWebpageUrl={window.apiUrls.api_ingest_webpage}
        ingestHandwrittenUrl={window.apiUrls.api_ingest_handwritten_notes}
        collectionId={collectionId}
        onUploadSuccess={fetchCollectionData}
        layout="compact"
      />
    </div>

    <div className="relative flex items-center mb-[16px]">
      <div className="flex-grow border-t border-border-low_contrast" />
      <span className="text-xs px-[8px] bg-dark-mode-background text-text-lower_contrast">Browse</span>
      <div className="flex-grow border-t border-border-low_contrast" />
    </div>

    <FileSystemViewer
      mode="browse"
      items={contents}
      collection={collection}
      onOpenItem={onOpenItem}
      onRemoveItem={onRemoveItem}
      onMove={onContextMove}
      onContextMenuRename={onRenameItem}
      onBatchMove={onBatchMove}
      onRemoveBatch={onBatchRemove}
    />

    <MoveCollectionModal
      folder={movingItem as unknown as Collection}
      collections={allCollections.filter((c) => c.id !== movingItem?.id)}
      isOpen={isMoveModalOpen}
      onClose={onCloseMoveModal}
      onSubmit={onMoveSubmit}
    />

    {batchMovingItems.length > 0 && (
      <MoveCollectionModal
        folder={{
          id: -1,
          name: `${batchMovingItems.length} selected item${batchMovingItems.length > 1 ? 's' : ''}`,
          parent: null,
          collection: collection.id,
          path: '',
          children: [],
          document_count: 0,
          children_count: 0,
          created_at: '',
          updated_at: '',
        }}
        collections={allCollections}
        isOpen={isBatchMoveModalOpen}
        onClose={onCloseBatchMoveModal}
        onSubmit={(_, newParentId) => onBatchMoveSubmit(newParentId)}
      />
    )}

    {successMessage && (
      <div className="fixed top-4 right-4 bg-green-600 text-white px-4 py-2 rounded shadow-lg z-50 animate-fade-in">
        {successMessage}
      </div>
    )}

    {isBatchOperationLoading && (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-scheme-shade_3 p-6 rounded-lg shadow-xl flex flex-col items-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4" />
          <p className="text-white text-lg">Processing items...</p>
        </div>
      </div>
    )}

    <UserManagementModal
      collection={collection}
      isOpen={isUserManagementModalOpen}
      onClose={onCloseUserManagement}
      onSave={onUserManagementSave}
    />
  </div>
);

export default CollectionViewShell;
