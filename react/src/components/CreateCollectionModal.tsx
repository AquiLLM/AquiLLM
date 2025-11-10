/**
 * Modal component for creating new collections
 * @module CreateCollectionModal
 */

import React, { useState } from 'react';
import { Collection } from './CollectionsTree';

interface CreateCollectionModalProps {
  isOpen: boolean;                          // Controls modal visibility
  onClose: () => void;                      // Callback when modal is closed
  onSubmit: (newCollection: Collection) => void; // Callback when new collection is created
}

/**
 * Modal dialog for creating a new collection
 * @param props - Component properties
 * @param props.isOpen - Whether the modal is visible
 * @param props.onClose - Function to call when modal should close
 * @param props.onSubmit - Function to call with new collection data
 */
const CreateCollectionModal: React.FC<CreateCollectionModalProps> = ({ isOpen, onClose, onSubmit }) => {
  const [name, setName] = useState('');

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    
    // Create temporary collection object
    const newCollection: Collection = {
      id: Date.now(), // Temporary ID, will be replaced by server
      name: name.trim(),
      parent: null,
      collection: 0,
      path: name.trim(),
      children: [],
      document_count: 0,
      children_count: 0,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    
    onSubmit(newCollection);
    setName('');
  };


  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 backdrop-blur-[10px] flex justify-center items-center z-[1000]">
      <div className="bg-scheme-shade_3 p-6 rounded-[32px] border border-border-mid_contrast w-full max-w-[400px] relative shadow-lg">
        <h3 className="text-2xl font-bold mb-6 text-text-normal">Create New Collection</h3>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div>
            <label htmlFor="collectionName" className="block mb-2 text-text-low_contrast">
              Collection Name
            </label>
            <input
              id="collectionName"
              type="text"
              value={name}
              className="w-full p-3 bg-scheme-shade_4 border border-border-mid_contrast rounded-md text-base text-text-normal focus:outline-none focus:ring-2 focus:ring-accent"
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter collection name"
              autoFocus
            />
          </div>
          <div className="flex justify-end gap-3 mt-4">
            <button
              type="button"
              onClick={onClose}
              className="py-2 px-4 rounded bg-scheme-shade_6 hover:bg-scheme-shade_7 text-text-normal text-sm font-medium cursor-pointer border-none"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="py-2 px-4 rounded bg-accent hover:bg-accent-dark text-text-normal text-sm font-medium cursor-pointer border-none"
            >
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default CreateCollectionModal; 