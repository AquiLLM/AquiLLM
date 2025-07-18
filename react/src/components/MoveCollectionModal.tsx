import React, { useState, useMemo } from 'react';
import { Collection } from './CollectionsTree';
import { FileSystemItem } from '../types/FileSystemItem';
import { FolderIcon } from '../icons/folder.tsx';

interface MoveCollectionModalProps {
  folder: Collection | null; // The collection being moved
  collections: Collection[]; // All available collections (including nested ones)
  isOpen: boolean;       // Modal visibility
  onClose: () => void;
  onSubmit: (folderId: number, newParentId: number | null) => void;
}

/* Example inline modal styles */
const modalStyles = {
  title: {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    marginBottom: '1rem',
    color: '#eeeeee',
  },
  breadcrumb: {
    marginBottom: '1rem',
    fontSize: '1rem',
    color: '#555',
    width: '100%'
  },
  list: {
    listStyle: 'none',
    padding: 0,
    margin: 0,
    borderRadius: '8px',
    border: '1px solid #777777',
    maxHeight: '300px',
    overflowy: 'auto',
  },
  listItem: {
    padding: '0.75rem 1rem',
    cursor: 'pointer',
  },
  listItemHover: {
    backgroundColor: '#f7fafc',
  },
  buttonContainer: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '0.75rem',
    marginTop: '1.5rem',
    marginBottom: '.5rem',
  },
  button: {
    padding: '0.5rem 1rem',
    borderRadius: '20px',
    fontSize: '0.875rem',
    cursor: 'pointer',
  },
  cancelButton: {
    color: '#eeeeee',
    border: 'none',
  },
  submitButton: {
    color: 'eeeeee',
    border: 'none',
  },
  moveToRootButton: {
    padding: '0.5rem 1rem',
    borderRadius: '20px',
    fontSize: '0.875rem',
    backgroundColor: '#2d3748',
    color: '#eeeeee',
    border: 'none',
    cursor: 'pointer',
    marginBottom: '1rem',
  }
};

const MoveCollectionModal: React.FC<MoveCollectionModalProps> = ({
  folder,
  collections,
  isOpen,
  onClose,
  onSubmit,
}) => {
  // currentParentId represents the folder whose children we're browsing.
  // null means we're at the root level.
  const [currentParentId, setCurrentParentId] = useState<number | null>(null);

  // Compute the list of collections that have currentParentId as their parent.
  const currentItems = useMemo(() => {
    return collections.filter(col => col.parent === currentParentId);
  }, [collections, currentParentId]);

  // Build breadcrumb from currentParentId up to the root.
  const buildBreadcrumb = (parentId: number | null): Collection[] => {
    if (parentId === null) return [];
    const parentFolder = collections.find(col => col.id === parentId);
    if (!parentFolder) return [];
    return [...buildBreadcrumb(parentFolder.parent), parentFolder];
  };
  const breadcrumb = buildBreadcrumb(currentParentId);

  // Handler when user clicks on a folder in the list to drill down.
  const handleItemClick = (item: FileSystemItem) => {
    // Only allow drilling down if the item is a collection.
    if (item.type === 'collection') {
      setCurrentParentId(item.id);
    }
  };

  const insertFolderIcon = () => {
    return <FolderIcon />;
  }

  // "Select This Folder" button will select the current folder (i.e. currentParentId)
  const handleSelectCurrent = () => {
    onSubmit(folder!.id, currentParentId);
  };

  // "Move to Root" button handler
  const handleMoveToRoot = () => {
    onSubmit(folder!.id, null);
  };

  // Back button to go up one level.
  const handleGoBack = () => {
    if (currentParentId !== null) {
      const parentFolder = collections.find(col => col.id === currentParentId);
      setCurrentParentId(parentFolder ? parentFolder.parent : null);
    }
  };

  if (!isOpen || !folder) return null;

  return (
    <div className="fixed top-0 left-0 right-0 bottom-0 flex items-center justify-center bg-black bg-opacity-75 z-[100] backdrop-blur-[10px]">
      <div className='border-border-high_contrast border flex flex-col items-center justify-left bg-scheme-shade_3 p-[1rem] rounded-[32px] w-full max-w-[600px] position-relative box-shadow-[0_4px_6px_rgba(0,0,0,0.1)] text-text-normal'> 

        <h3 style={modalStyles.title}>Move Collection</h3>

        {/* Breadcrumb Navigation */}
        <div style={modalStyles.breadcrumb} className='flex flex-col items-center justify-center'>
          <span className='text-text-less_contrast'>
            <strong>{folder.name}</strong> will be moved to:
          </span>
          <div>          
            {breadcrumb.length > 0 ? (
            <>      
              {breadcrumb.map((b, idx) => (
                <span key={b.id} className="text-accent-light">
                    {b.name}{idx < breadcrumb.length - 1 ? ' / ' : ''}
                </span>
              ))}
            </>
            ) : (
              <span className='text-accent-light'>Root</span>
            )}
          </div>
        </div>

        {/* Move to Root button */}
        {folder.parent !== null && (
          <button
            type="button"
            onClick={handleMoveToRoot}
            style={modalStyles.moveToRootButton}
            className='bg-scheme-shade_4 hover:bg-scheme-shade_6 transition-all w-full'
          >
            Move to Root Level
          </button>
        )}

        <div className='w-full flex justify-left'>
          <button onClick={handleGoBack} style={{ marginRight: '32px' }} className='text-text-less_contrast hover:bg-scheme-shade_6 transition-all rounded-[8px] p-[4px] mb-[8px]'>
                  ← Back
          </button>
        </div>

        {/* List of collections at the current level */}
        <ul style={modalStyles.list} className='w-full'>
          {currentItems.map(item => (
            <li
              key={item.id}
              style={modalStyles.listItem}
              className='hover:bg-scheme-shade_4 transition-all rounded-[8px] text-accent-light flex items-center justify-left gap-[16px]'
              onClick={() => handleItemClick({ id: item.id, type: 'collection', name: item.name })}
            >
              {insertFolderIcon()}
              {item.name}
            </li>
          ))}
          {currentItems.length === 0 && (
            <li style={{ padding: '0.75rem 1rem', color: '#999' }}>
              No sub-collections here.
            </li>
          )}
        </ul>

        {/* Action Buttons */}
        <div style={modalStyles.buttonContainer}>
          <button
            type="button"
            onClick={onClose}
            className='bg-scheme-shade_6'
            style={{ ...modalStyles.button, ...modalStyles.cancelButton }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSelectCurrent}
            className='bg-accent'
            style={{ ...modalStyles.button, ...modalStyles.submitButton }}
          >
            Select This Location
          </button>
        </div>
      </div>
    </div>
  );
};

export default MoveCollectionModal;
