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

        <h3 className="text-2xl font-bold mb-4 text-text-normal">Move Collection</h3>

        {/* Breadcrumb Navigation */}
        <div className='flex flex-col items-center justify-center mb-4 text-base text-text-low_contrast w-full'>
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
            className='bg-scheme-shade_4 hover:bg-scheme-shade_6 transition-all w-full py-2 px-4 rounded-[20px] text-sm text-text-normal border-none cursor-pointer mb-4'
          >
            Move to Root Level
          </button>
        )}

        <div className='w-full flex justify-left'>
          <button onClick={handleGoBack} className='text-text-less_contrast hover:bg-scheme-shade_6 transition-all rounded-[8px] p-[4px] mb-[8px] mr-8'>
                  ← Back
          </button>
        </div>

        {/* List of collections at the current level */}
        <ul className='w-full list-none p-0 m-0 rounded-lg border border-border-mid_contrast max-h-[300px] overflow-y-auto'>
          {currentItems.map(item => (
            <li
              key={item.id}
              className='p-3 cursor-pointer hover:bg-scheme-shade_4 transition-all rounded-[8px] text-accent-light flex items-center justify-left gap-[16px]'
              onClick={() => handleItemClick({ id: item.id, type: 'collection', name: item.name })}
            >
              {insertFolderIcon()}
              {item.name}
            </li>
          ))}
          {currentItems.length === 0 && (
            <li className="p-3 text-text-lower_contrast">
              No sub-collections here.
            </li>
          )}
        </ul>

        {/* Action Buttons */}
        <div className="flex justify-end gap-3 mt-6 mb-2">
          <button
            type="button"
            onClick={onClose}
            className='bg-scheme-shade_6 hover:bg-scheme-shade_7 text-text-normal py-2 px-4 rounded-[20px] text-sm cursor-pointer border-none'
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSelectCurrent}
            className='bg-accent hover:bg-accent-dark text-text-normal py-2 px-4 rounded-[20px] text-sm cursor-pointer border-none'
          >
            Select This Location
          </button>
        </div>
      </div>
    </div>
  );
};

export default MoveCollectionModal;
