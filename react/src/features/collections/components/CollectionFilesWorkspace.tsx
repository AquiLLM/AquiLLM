import React from 'react';
import type { Collection } from '../../../components/CollectionsTree';
import FileSystemViewer from '../../documents/components/FileSystemViewer';
import MoveCollectionModal from '../../../components/MoveCollectionModal';
import CreateCollectionModal from '../../../components/CreateCollectionModal';
import type { FileSystemItem } from '../../../types/FileSystemItem';
import IngestRowContainer from '../../../components/IngestRow';

export interface CollectionFilesWorkspaceProps {
  collection: Collection;
  collectionId: string;
  contents: import('./collectionViewTypes').CollectionContent[];
  allCollections: Collection[];
  movingItem: FileSystemItem | Collection | null;
  isMoveModalOpen: boolean;
  batchMovingItems: FileSystemItem[];
  isBatchMoveModalOpen: boolean;
  isCreateSubcollectionOpen: boolean;
  successMessage: string | null;
  isBatchOperationLoading: boolean;
  fetchCollectionData: () => void;
  onOpenItem: (item: FileSystemItem) => void;
  onRemoveItem: (item: FileSystemItem) => void;
  onContextMove: (item: FileSystemItem) => void;
  onRenameItem: () => void;
  onBatchMove: (items: FileSystemItem[]) => void;
  onBatchRemove: (items: FileSystemItem[]) => void;
  onCloseCreateSubcollection: () => void;
  onSubmitCreateSubcollection: (collection: Collection) => void;
  onCloseMoveModal: () => void;
  onMoveSubmit: (itemId: number, newParentId: number | null) => void;
  onCloseBatchMoveModal: () => void;
  onBatchMoveSubmit: (newParentId: number | null) => void;
}

const CollectionFilesWorkspace: React.FC<CollectionFilesWorkspaceProps> = ({
  collection,
  collectionId,
  contents,
  allCollections,
  movingItem,
  isMoveModalOpen,
  batchMovingItems,
  isBatchMoveModalOpen,
  isCreateSubcollectionOpen,
  successMessage,
  isBatchOperationLoading,
  fetchCollectionData,
  onOpenItem,
  onRemoveItem,
  onContextMove,
  onRenameItem,
  onBatchMove,
  onBatchRemove,
  onCloseCreateSubcollection,
  onSubmitCreateSubcollection,
  onCloseMoveModal,
  onMoveSubmit,
  onCloseBatchMoveModal,
  onBatchMoveSubmit,
}) => (
  <>
    <div
      className="mb-[24px] bg-scheme-shade_4 border border-border-low_contrast rounded-[20px] p-[14px]"
      data-testid="collection-files-ingest"
    >
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

    <div className="relative flex items-center mb-[16px]" data-testid="collection-files-browse-separator">
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

    <CreateCollectionModal
      isOpen={isCreateSubcollectionOpen}
      onClose={onCloseCreateSubcollection}
      onSubmit={onSubmitCreateSubcollection}
      parentCollection={collection}
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
  </>
);

export default CollectionFilesWorkspace;
