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

  const modalStyles = {
    overlay: {
      position: 'fixed' as const,
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.75)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 1000,
    },
    content: {
      backgroundColor: '#333333',
      padding: '1.5rem',
      borderRadius: '32px',
      border: '1px solid #777777',
      width: '100%',
      maxWidth: '400px',
      position: 'relative' as const,
      boxShadow: '0 0px 10px rgba(0, 0, 0, 0.5)',
    },
    title: {
      fontSize: '1.5rem',
      fontWeight: 'bold',
      marginBottom: '1.5rem',
    },
    form: {
      display: 'flex',
      flexDirection: 'column' as const,
      gap: '1rem',
    },
    label: {
      display: 'block',
      marginBottom: '0.5rem',
    },
    input: {
      width: '100%',
      padding: '0.75rem',
      backgound: '#444444',
      border: '1px solid #777777',
      borderRadius: '0.375rem',
      fontSize: '1rem',
      color: '#eeeeee',
    },
    buttonContainer: {
      display: 'flex',
      justifyContent: 'flex-end',
      gap: '0.75rem',
      marginTop: '1rem',
    },
    button: {
      padding: '0.5rem 1rem',
      borderRadius: '0.375rem',
      fontSize: '0.875rem',
      fontWeight: 500,
      cursor: 'pointer',
    },
    cancelButton: {
      backgroundColor: '#666666',
      color: '#eeeeee',
      border: 'none',
    },
    submitButton: {
      backgroundColor: '#1C79D8',
      color: '#eeeeee',
      border: 'none',
    }
  };

  return (
    <div style={modalStyles.overlay}>
      <div style={modalStyles.content}>
        <h3 style={modalStyles.title}>Create New Collection</h3>
        <form onSubmit={handleSubmit} style={modalStyles.form}>
          <div>
            <label htmlFor="collectionName" style={modalStyles.label}>
              Collection Name
            </label>
            <input
              id="collectionName"
              type="text"
              value={name}
              className="bg-scheme-shade_4 border border-border-high_contrast rounded-md p-2 text-gray-shade_1"
              onChange={(e) => setName(e.target.value)}
              style={modalStyles.input}
              placeholder="Enter collection name"
              autoFocus
            />
          </div>
          <div style={modalStyles.buttonContainer}>
            <button
              type="button"
              onClick={onClose}
              style={{ ...modalStyles.button, ...modalStyles.cancelButton }}
            >
              Cancel
            </button>
            <button
              type="submit"
              style={{ ...modalStyles.button, ...modalStyles.submitButton }}
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