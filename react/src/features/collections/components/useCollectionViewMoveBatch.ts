import { useCallback } from 'react';
import type { Collection } from '../../../components/CollectionsTree';
import type { FileSystemItem } from '../../../types/FileSystemItem';
import { getCookie } from '../../../utils/csrf';
import formatUrl from '../../../utils/formatUrl';
import type { CollectionContent } from './collectionViewTypes';

export interface UseCollectionViewMoveBatchParams {
  movingItem: FileSystemItem | Collection | null;
  setMovingItem: React.Dispatch<React.SetStateAction<FileSystemItem | Collection | null>>;
  setIsMoveModalOpen: React.Dispatch<React.SetStateAction<boolean>>;
  batchMovingItems: FileSystemItem[];
  setBatchMovingItems: React.Dispatch<React.SetStateAction<FileSystemItem[]>>;
  setIsBatchMoveModalOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setIsBatchOperationLoading: React.Dispatch<React.SetStateAction<boolean>>;
  setSuccessMessage: React.Dispatch<React.SetStateAction<string | null>>;
  setContents: React.Dispatch<React.SetStateAction<CollectionContent[]>>;
  allCollections: Collection[];
}

export function useCollectionViewMoveBatch({
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
}: UseCollectionViewMoveBatchParams) {
  const handleMoveSubmit = useCallback(
    (itemId: number, newParentId: number | null) => {
      if (!movingItem) return;

      const isFileSystemItem = (item: unknown): item is FileSystemItem =>
        Boolean(item && typeof item === 'object' && 'type' in item);

      if (isFileSystemItem(movingItem) && movingItem.type === 'collection') {
        fetch(formatUrl(window.apiUrls.api_move_collection, { collection_id: itemId }), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
          },
          body: JSON.stringify({ new_parent_id: newParentId }),
          credentials: 'include',
        })
          .then((res) =>
            res.ok ? res.json() : res.json().then((data) => {
              throw new Error(data.error || 'Failed to move collection');
            })
          )
          .then((data) => {
            console.log('Collection moved:', data);
            setIsMoveModalOpen(false);
            setMovingItem(null);
          })
          .catch((err) => {
            console.error('Error moving collection:', err);
            alert(`Error: ${err.message}`);
          });
      } else {
        fetch(formatUrl(window.apiUrls.api_move_document, { doc_id: movingItem.id }), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
          },
          body: JSON.stringify({ new_collection_id: newParentId }),
          credentials: 'include',
        })
          .then((res) =>
            res.ok ? res.json() : res.json().then((data) => {
              throw new Error(data.error || 'Failed to move document');
            })
          )
          .then((data) => {
            console.log('Document moved:', data);
            setIsMoveModalOpen(false);
            setMovingItem(null);
          })
          .catch((err) => {
            console.error('Error moving document:', err);
            alert(`Error: ${err.message}`);
          });
      }
    },
    [movingItem, setIsMoveModalOpen, setMovingItem]
  );

  const handleBatchMove = useCallback(
    (items: FileSystemItem[]) => {
      setBatchMovingItems(items);
      setIsBatchMoveModalOpen(true);
    },
    [setBatchMovingItems, setIsBatchMoveModalOpen]
  );

  const handleBatchMoveSubmit = useCallback(
    (newParentId: number | null) => {
      const documents = batchMovingItems.filter((item) => item.type !== 'collection');
      const collections = batchMovingItems.filter((item) => item.type === 'collection');
      const totalCount = batchMovingItems.length;
      let processedCount = 0;
      let errorCount = 0;

      setIsBatchOperationLoading(true);

      collections.forEach((col) => {
        fetch(formatUrl(window.apiUrls.api_move_collection, { collection_id: col.id }), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
          },
          body: JSON.stringify({ new_parent_id: newParentId }),
          credentials: 'include',
        })
          .then((res) =>
            res.ok ? res.json() : res.json().then((data) => {
              throw new Error(data.error || 'Failed to move collection');
            })
          )
          .then(() => {
            processedCount++;
            checkIfComplete();
          })
          .catch((err) => {
            console.error(`Error moving collection ${col.id}:`, err);
            errorCount++;
            processedCount++;
            checkIfComplete();
          });
      });

      documents.forEach((document) => {
        fetch(formatUrl(window.apiUrls.api_move_document, { doc_id: document.id }), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
          },
          body: JSON.stringify({ new_collection_id: newParentId }),
          credentials: 'include',
        })
          .then((res) =>
            res.ok ? res.json() : res.json().then((data) => {
              throw new Error(data.error || 'Failed to move document');
            })
          )
          .then(() => {
            processedCount++;
            checkIfComplete();
          })
          .catch((err) => {
            console.error(`Error moving document ${document.id}:`, err);
            errorCount++;
            processedCount++;
            checkIfComplete();
          });
      });

      function checkIfComplete() {
        if (processedCount === totalCount) {
          setIsBatchOperationLoading(false);
          setIsBatchMoveModalOpen(false);
          setBatchMovingItems([]);

          if (errorCount > 0) {
            setSuccessMessage(`Move completed with ${errorCount} errors. The page will refresh.`);
          } else {
            setSuccessMessage(
              `Successfully moved ${totalCount} item${totalCount !== 1 ? 's' : ''}. The page will refresh.`
            );
          }

          setTimeout(() => {
            window.location.reload();
          }, 2000);
        }
      }
    },
    [
      batchMovingItems,
      setBatchMovingItems,
      setIsBatchMoveModalOpen,
      setIsBatchOperationLoading,
      setSuccessMessage,
    ]
  );

  const handleBatchRemoveItems = useCallback(
    (items: FileSystemItem[]) => {
      if (!window.confirm(`Are you sure you want to delete ${items.length} selected items?`)) {
        return;
      }

      setIsBatchOperationLoading(true);
      let processedCount = 0;
      let errorCount = 0;
      const totalCount = items.length;
      const removedItemKeys = new Set<string>();

      items.forEach((item) => {
        const url =
          item.type === 'collection'
            ? formatUrl(window.apiUrls.api_delete_collection, { collection_id: item.id })
            : formatUrl(window.apiUrls.api_delete_document, { doc_id: item.id });

        fetch(url, {
          method: 'DELETE',
          headers: {
            'X-CSRFToken': getCookie('csrftoken'),
          },
          credentials: 'include',
        })
          .then((res) => {
            if (!res.ok) {
              throw new Error(`Failed to delete ${item.type} ${item.id}`);
            }
            removedItemKeys.add(`${item.type}:${item.id}`);
            processedCount++;
            checkIfComplete();
          })
          .catch((err) => {
            console.error(`Error deleting ${item.type} ${item.id}:`, err);
            errorCount++;
            processedCount++;
            checkIfComplete();
          });
      });

      function checkIfComplete() {
        if (processedCount === totalCount) {
          setIsBatchOperationLoading(false);
          if (removedItemKeys.size > 0) {
            setContents((prevContents) =>
              prevContents.filter(
                (contentItem) => !removedItemKeys.has(`${contentItem.type}:${contentItem.id}`)
              )
            );
          }

          const successCount = removedItemKeys.size;
          if (errorCount > 0) {
            setSuccessMessage(
              `Deleted ${successCount} item${successCount !== 1 ? 's' : ''}; ${errorCount} failed.`
            );
          } else {
            setSuccessMessage(
              `Successfully deleted ${successCount} item${successCount !== 1 ? 's' : ''}.`
            );
          }

          setTimeout(() => {
            setSuccessMessage(null);
          }, 3000);
        }
      }
    },
    [setContents, setIsBatchOperationLoading, setSuccessMessage]
  );

  return {
    handleMoveSubmit,
    handleBatchMove,
    handleBatchMoveSubmit,
    handleBatchRemoveItems,
  };
}
