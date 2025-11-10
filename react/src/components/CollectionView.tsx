import React, { useState, useEffect, useCallback } from 'react';
import { Collection } from '../components/CollectionsTree';
// import { Folder } from '../components/CollectionsTree';
import CollectionSettingsMenu from '../components/CollectionSettingsMenu';
import FileSystemViewer from '../components/FileSystemViewer';
import MoveCollectionModal from '../components/MoveCollectionModal';
import UserManagementModal from '../components/UserManagementModal';
import { FileSystemItem } from '../types/FileSystemItem';
import { getCookie } from '../utils/csrf';
import IngestRowContainer from '../components/IngestRow';
import IngestionDashboard from '../components/IngestionDashboard'; // Import the dashboard
import formatUrl from '../utils/formatUrl';

interface CollectionContent {
  id: number;
  type: string;
  name: string;
  created_at: string;
  document_count?: number;
}

interface CollectionViewProps {
  collectionId: string;
  onBack?: () => void; // optional callback when the back button is pressed
}

const CollectionView: React.FC<CollectionViewProps> = ({ collectionId, onBack }) => {
  // State for current collection details (children, documents)
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

  // Function to refetch collection data
  const fetchCollectionData = useCallback(() => {
    setLoading(true); // Show loading indicator during refetch
    fetch(formatUrl(window.apiUrls.api_collection, { col_id: collectionId }), {
      headers: { 'Accept': 'application/json' }
    })
      .then(res => {
        if (!res.ok) {
          return res.json().then(err => { throw new Error(err.error || 'Failed to fetch collection'); });
        }
        return res.json();
      })
      .then(data => {
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
          created_at: data.collection.created_at ? new Date(data.collection.created_at).toLocaleString() : new Date().toLocaleString(),
          updated_at: data.collection.updated_at ? new Date(data.collection.updated_at).toISOString() : new Date().toISOString(),
        });
        const transformedDocuments = (data.documents || []).map((doc: any) => ({
          id: doc.id, type: doc.type || 'document', name: doc.title || 'Untitled',
          created_at: doc.ingestion_date ? new Date(doc.ingestion_date).toLocaleString() : (doc.created_at ? new Date(doc.created_at).toLocaleString() : new Date().toLocaleString()), document_count: 0,
        }));
        const transformedChildren = (data.children || []).map((child: any) => ({
          id: child.id, type: 'collection', name: child.name,
          created_at: new Date(child.created_at || new Date()).toLocaleString(), document_count: child.document_count,
        }));
        setContents([...transformedChildren, ...transformedDocuments]);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error refetching collection:', err);
        setError(err.message);
        setLoading(false);
      });
  }, [collectionId]); // Dependency array includes collectionId

  // Initial fetch
  useEffect(() => {
    fetchCollectionData();
  }, [fetchCollectionData]); // Use the memoized fetch function

  // Fetch all available collections (for moving purposes)
  useEffect(() => {
    fetch(window.apiUrls.api_collections, {
      headers: {
        'Accept': 'application/json'
      },
      credentials: 'include'
    })
      .then(res => {
        if (!res.ok) {
          throw new Error('Failed to fetch available collections');
        }
        return res.json();
      })
      .then(data => {
        // Expecting data.collections to be an array
        const collectionsData = data.collections || [];
        // We assume each collection has id, name, parent, etc.
        const parsed = collectionsData.map((col: any) => ({
          id: col.id,
          name: col.name,
          parent: col.parent, // either null or parent ID
          document_count: col.document_count,
          path: col.path, // Include path for breadcrumb navigation
          children: [],
          children_count: 0,
          created_at: '',
          updated_at: ''
        }));
        setAllCollections(parsed);
      })
      .catch(err => {
        console.error('Error fetching all collections:', err);
      });
  }, []);

  const handleRenameItem = () => {
    // TODO: Implement rename functionality
    console.log("Rename item clicked");
  }

  // Navigation handlers
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
    // Add null check for collection before accessing id
    if (collection && window.confirm(`Are you sure you want to delete "${collection.name}"?`)) {
      fetch(formatUrl(window.apiUrls.api_delete_collection, { collection_id: collection.id }), {
        method: 'DELETE',
        headers: { 'X-CSRFToken': getCookie('csrftoken') },
        credentials: 'include'
      })
        .then((res) => {
          if (!res.ok) throw new Error('Failed to delete collection');
          // After successful deletion, navigate back
          if (onBack) {
            onBack();
          } else {
             // If no onBack prop, maybe redirect to parent or collections page
             window.location.href = window.pageUrls.user_collections;
          }
        })
        .catch((err) => {
          console.error('Error:', err);
          alert('Failed to delete collection. Please try again.');
        });
    }
  };
  // End of re-inserted functions

  const handleRemoveItem = (item: FileSystemItem) => {
    if (window.confirm(`Are you sure you want to remove "${item.name}"?`)) {
      // Different endpoints for collections vs documents
      const endpoint = item.type === 'collection' 
        ? formatUrl(window.apiUrls.api_delete_collection, { collection_id: item.id })
        : formatUrl(window.apiUrls.api_delete_document, { doc_id: item.id });
      
      fetch(endpoint, {
        method: 'DELETE',
        headers: { 'X-CSRFToken': getCookie('csrftoken') },
        credentials: 'include'
      })
        .then((res) => {
          if (!res.ok) throw new Error(`Failed to remove ${item.type}`);
          
          // Remove the item from the content list
          setContents(prevContents => prevContents.filter(contentItem => contentItem.id !== item.id));
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

  // Context Menu onMove handler (from FileSystemViewer)
  const handleContextMove = (item: FileSystemItem) => {
    // When user right-clicks on an item and selects "Move"
    setMovingItem(item);
    setIsMoveModalOpen(true);
  };

  // Handler for moving the current collection.
  const handleMoveSubmit = (itemId: number, newParentId: number | null) => {
    if (!movingItem) return;

    // Type guard to check if movingItem is a FileSystemItem (which has 'type')
    const isFileSystemItem = (item: any): item is FileSystemItem => {
      return item && typeof item === 'object' && 'type' in item;
    };

    if (isFileSystemItem(movingItem) && movingItem.type === 'collection') {
      // Call collection move endpoint:
      fetch(formatUrl(window.apiUrls.api_move_collection, { collection_id: itemId }), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({ new_parent_id: newParentId }),
        credentials: 'include',
      })
        .then(res => res.ok ? res.json() : res.json().then(data => { throw new Error(data.error || 'Failed to move collection'); }))
        .then(data => {
          console.log('Collection moved:', data);
          setIsMoveModalOpen(false);
          setMovingItem(null);
          // Optionally, refresh the collection details.
        })
        .catch(err => {
          console.error('Error moving collection:', err);
          alert(`Error: ${err.message}`);
        });
    } else {
      // Assume it's a document
      fetch(formatUrl(window.apiUrls.api_move_document, { doc_id: movingItem.id }), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({ new_collection_id: newParentId }),
        credentials: 'include',
      })
        .then(res => res.ok ? res.json() : res.json().then(data => { throw new Error(data.error || 'Failed to move document'); }))
        .then(data => {
          console.log('Document moved:', data);
          setIsMoveModalOpen(false);
          setMovingItem(null);
          // Optionally, refresh the document list.
        })
        .catch(err => {
          console.error('Error moving document:', err);
          alert(`Error: ${err.message}`);
        });
    }
  };

  // Handler for batch move
  const handleBatchMove = (items: FileSystemItem[]) => {
    setBatchMovingItems(items);
    setIsBatchMoveModalOpen(true);
  };

  // Submit handler for batch move
  const handleBatchMoveSubmit = (newParentId: number | null) => {
    // Process documents and collections separately
    const documents = batchMovingItems.filter(item => item.type !== 'collection');
    const collections = batchMovingItems.filter(item => item.type === 'collection');
    const totalCount = batchMovingItems.length;
    let processedCount = 0;
    let errorCount = 0;

    // Show loading state
    setIsBatchOperationLoading(true);

    // Move collections
    collections.forEach(collection => {
      fetch(formatUrl(window.apiUrls.api_move_collection, { collection_id: collection.id }), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({ new_parent_id: newParentId }),
        credentials: 'include',
      })
        .then(res => res.ok ? res.json() : res.json().then(data => { throw new Error(data.error || 'Failed to move collection'); }))
        .then(() => {
          processedCount++;
          checkIfComplete();
        })
        .catch(err => {
          console.error(`Error moving collection ${collection.id}:`, err);
          errorCount++;
          processedCount++;
          checkIfComplete();
        });
    });

    // Move documents
    documents.forEach(document => {
      fetch(formatUrl(window.apiUrls.api_move_document, { doc_id: document.id }), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({ new_collection_id: newParentId }),
        credentials: 'include',
      })
        .then(res => res.ok ? res.json() : res.json().then(data => { throw new Error(data.error || 'Failed to move document'); }))
        .then(() => {
          processedCount++;
          checkIfComplete();
        })
        .catch(err => {
          console.error(`Error moving document ${document.id}:`, err);
          errorCount++;
          processedCount++;
          checkIfComplete();
        });
    });

    // Function to check if all operations are complete
    function checkIfComplete() {
      if (processedCount === totalCount) {
        setIsBatchOperationLoading(false);
        setIsBatchMoveModalOpen(false);
        setBatchMovingItems([]);
        
        if (errorCount > 0) {
          setSuccessMessage(`Move completed with ${errorCount} errors. The page will refresh.`);
        } else {
          setSuccessMessage(`Successfully moved ${totalCount} item${totalCount !== 1 ? 's' : ''}. The page will refresh.`);
        }
        
        // Set a timeout to reload the page after showing the message
        setTimeout(() => {
          window.location.reload();
        }, 2000);
      }
    }
  };

  // Handler for batch remove
  const handleBatchRemoveItems = (items: FileSystemItem[]) => {
    if (!window.confirm(`Are you sure you want to delete ${items.length} selected items?`)) {
      return;
    }
    
    setIsBatchOperationLoading(true);
    let processedCount = 0;
    let errorCount = 0;
    const totalCount = items.length;
    
    // Process each item for deletion
    items.forEach(item => {
      // Using a modified version of handleRemoveItem that tracks completion
      const url = item.type === 'collection' 
        ? formatUrl(window.apiUrls.api_delete_collection, { collection_id: item.id })
        : formatUrl(window.apiUrls.api_delete_document, { doc_id: item.id });
      
      fetch(url, {
        method: 'DELETE',
        headers: {
          'X-CSRFToken': getCookie('csrftoken'),
        },
        credentials: 'include',
      })
        .then(res => {
          if (!res.ok) {
            throw new Error(`Failed to delete ${item.type} ${item.id}`);
          }
          processedCount++;
          checkIfComplete();
        })
        .catch(err => {
          console.error(`Error deleting ${item.type} ${item.id}:`, err);
          errorCount++;
          processedCount++;
          checkIfComplete();
        });
    });
    
    function checkIfComplete() {
      if (processedCount === totalCount) {
        setIsBatchOperationLoading(false);
        
        if (errorCount > 0) {
          setSuccessMessage(`Deletion completed with ${errorCount} errors. The page will refresh.`);
        } else {
          setSuccessMessage(`Successfully deleted ${totalCount} item${totalCount !== 1 ? 's' : ''}. The page will refresh.`);
        }
        
        // Set a timeout to reload the page after showing the message
        setTimeout(() => {
          window.location.reload();
        }, 2000);
      }
    }
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!collection) return <div>Collection not found</div>;

  // Parse the collection path into breadcrumb segments
  const parseBreadcrumbs = () => {
    if (!collection || !collection.path) return [];
    
    // Split the path into segments
    const segments = collection.path.split('/').filter(segment => segment.trim() !== '');
    const breadcrumbs = [];
    
    // Add "Root" as the first breadcrumb
    breadcrumbs.push({
      name: 'Root',
      id: null, // No ID for root
      path: '',
      fullPath: ''
    });
    
    // Build the breadcrumb links with proper IDs
    let currentFullPath = '';
    const pathToCollectionMap = new Map();
    
    // First, build a map of paths to collection IDs
    allCollections.forEach(col => {
      if (col.path) {
        pathToCollectionMap.set(col.path, col.id);
      }
    });
    
    // Now build the breadcrumbs
    for (let i = 0; i < segments.length; i++) {
      const segmentName = segments[i];
      currentFullPath = currentFullPath ? `${currentFullPath}/${segmentName}` : segmentName;
      
      // First try to get the collection ID from our path map
      const collectionId = pathToCollectionMap.get(currentFullPath);
      
      // If we couldn't find an exact path match, try to find a collection with matching name and path ending
      let matchingCollection = null;
      if (!collectionId) {
        matchingCollection = allCollections.find(col => 
          col.name === segmentName && col.path && col.path.endsWith(currentFullPath)
        );
      }
      
      if (collectionId || matchingCollection) {
        breadcrumbs.push({
          name: segmentName,
          id: collectionId || (matchingCollection ? matchingCollection.id : null),
          path: segmentName,
          fullPath: currentFullPath
        });
      } else {
        // If we still can't find a match, add the segment without an ID
        breadcrumbs.push({
          name: segmentName,
          id: null,
          path: segmentName,
          fullPath: currentFullPath
        });
      }
    }
    
    return breadcrumbs;
  };
  
  const breadcrumbs = parseBreadcrumbs();

  // Construct WebSocket URL base
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsBaseUrl = `${wsProtocol}//${window.location.host}`;

  return (
    <div style={{ padding: '2rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'end', justifyContent: 'center' }} className='mb-[32px] px-[40px]'>
        <button
          onClick={handleBack}
          className="bg-transparent border-none cursor-pointer flex items-center text-text-lower_contrast"
        >
          ← Back
        </button>
        
        {/* 
            The right padding here is a little weird, because the padding on the the element to the right of it
            has a padding of 0.5rem that slightly offsets this title off-center.  This padding re-centers it with respect
            to the page. 
        */}
        <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', flex: 1, textAlign: 'center', paddingRight: '0.5rem' }}>
          {collection.name}
        </h1>

        <CollectionSettingsMenu
          collection={collection}
          onManageCollaborators={handleManageCollaborators}
          onDelete={handleDelete}
          onMove={() => {
            setMovingItem({ id: collection!.id, type: 'collection', name: collection!.name });
            setIsMoveModalOpen(true);
          }}
        />
      </div>

      {/* Breadcrumb Navigation */}
      <div className="mb-4 px-[40px]">
        <nav className="flex" aria-label="Breadcrumb">
          <ol className="inline-flex items-center space-x-1 md:space-x-3">
            {breadcrumbs.map((crumb, index) => (
              <li key={`${crumb.name}-${index}`} className="inline-flex items-center">
                {index > 0 && (
                  <svg className="w-3 h-3 mx-1 text-text-lower_contrast" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 6 10">
                    <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m1 9 4-4-4-4"/>
                  </svg>
                )}
                
                {crumb.id !== null ? (
                  <a 
                    href={formatUrl(window.pageUrls.collection, {col_id: crumb.id})}
                    className={`ml-1 text-sm ${index === breadcrumbs.length - 1
                      ? 'text-accent font-medium'
                      : 'text-text-slightly_less_contrast hover:text-accent'}`}
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
      </div>

      {/* Permission Source Indicator */}
      {permissionSource && !permissionSource.direct && permissionSource.source_collection_name && (
        <div className="mb-4 p-3 bg-accent bg-opacity-15 text-accent-light rounded-md flex items-center">
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>
            You have access to this collection through <strong>{permissionSource.permission_level}</strong> permission 
            inherited from parent collection: <strong>{permissionSource.source_collection_name}</strong>
          </span>
        </div>
      )}

      <div className="relative flex items-center mb-[24px]"> 
          <div className="flex-grow border-t border-border-low_contrast"></div>
            <span className="text-xs px-[8px] bg-dark-mode-background text-text-lower_contrast">Add Content</span>
          <div className="flex-grow border-t border-border-low_contrast"></div>
      </div>

      <IngestRowContainer
        ingestArxivUrl={window.apiUrls.api_ingest_arxiv}
        ingestPdfUrl={window.apiUrls.api_ingest_pdf}
        ingestVttUrl={window.apiUrls.api_ingest_vtt}
        ingestWebpageUrl={window.apiUrls.api_ingest_webpage}
        ingestHandwrittenUrl={window.apiUrls.api_ingest_handwritten_notes}
        collectionId={collectionId}
      />

      <div className="relative flex items-center mb-[24px]"> 
          <div className="flex-grow border-t border-border-low_contrast"></div>
            <span className="text-xs px-[8px] bg-dark-mode-background text-text-lower_contrast">Browse</span>
          <div className="flex-grow border-t border-border-low_contrast"></div>
      </div>

      <FileSystemViewer
        mode="browse"
        items={contents}
        collection={collection}
        onOpenItem={handleOpenItem}
        onRemoveItem={handleRemoveItem}
        onMove={handleContextMove}
        onContextMenuRename={handleRenameItem}
        onBatchMove={handleBatchMove}
        onRemoveBatch={handleBatchRemoveItems}
      />

      {/* Move Modal for moving the current collection */}
      <MoveCollectionModal
        folder={movingItem as unknown as Collection}       // the collection or document being moved
        collections={allCollections.filter(collection => collection.id !== movingItem?.id)} // full list of collections with parent information
        isOpen={isMoveModalOpen}
        onClose={() => { setIsMoveModalOpen(false); setMovingItem(null); }}
        onSubmit={handleMoveSubmit} // your move handler that calls the API
      />

      {/* Batch Move Modal */}
      {batchMovingItems.length > 0 && (
        <MoveCollectionModal
          folder={{ 
            id: -1, // dummy id
            name: `${batchMovingItems.length} selected item${batchMovingItems.length > 1 ? 's' : ''}`,
            parent: null,
            collection: collection.id,
            path: '',
            children: [],
            document_count: 0,
            children_count: 0,
            created_at: '',
            updated_at: ''
          }}
          collections={allCollections}
          isOpen={isBatchMoveModalOpen}
          onClose={() => { 
            setIsBatchMoveModalOpen(false);
            setBatchMovingItems([]);
          }}
          onSubmit={(_, newParentId) => handleBatchMoveSubmit(newParentId)}
        />
      )}

      {/* Success Message */}
      {successMessage && (
        <div className="fixed top-4 right-4 bg-green-600 text-white px-4 py-2 rounded shadow-lg z-50 animate-fade-in">
          {successMessage}
        </div>
      )}

      {/* Loading Indicator for Batch Operations */}
      {isBatchOperationLoading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-scheme-shade_3 p-6 rounded-lg shadow-xl flex flex-col items-center">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
            <p className="text-white text-lg">Processing items...</p>
          </div>
        </div>
      )}

      {/* User Management Modal */}
      <UserManagementModal
        collection={collection}
        isOpen={isUserManagementModalOpen}
        onClose={() => setIsUserManagementModalOpen(false)}
        onSave={() => {
          setSuccessMessage('Permissions updated successfully!');
          setTimeout(() => setSuccessMessage(null), 3000);
          // Optionally refresh collection data to reflect new permissions
          fetchCollectionData();
        }}
      />
    </div>
  );
};

export default CollectionView;
