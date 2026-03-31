import React from 'react';
import type { Collection } from '../types';

export interface ChatCollectionsModalProps {
  open: boolean;
  onClose: () => void;
  searchTerm: string;
  onSearchTermChange: (value: string) => void;
  filteredCollections: Collection[];
  selectedCollections: Set<string>;
  onToggleCollection: (collectionId: string) => void;
}

const ChatCollectionsModal: React.FC<ChatCollectionsModalProps> = ({
  open,
  onClose,
  searchTerm,
  onSearchTermChange,
  filteredCollections,
  selectedCollections,
  onToggleCollection,
}) => {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
      onClick={onClose}
    >
      <div
        className="bg-scheme-shade_2 rounded-lg p-6 w-[90%] max-w-md max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-text-normal">Select Collections</h2>
          <button
            onClick={onClose}
            className="text-text-normal hover:text-text-slightly_less_contrast"
            type="button"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="search-container mb-4">
          <input
            type="text"
            placeholder="Search..."
            value={searchTerm}
            onChange={(e) => onSearchTermChange(e.target.value)}
            className="text-sm h-[36px] w-full p-2 border rounded-lg text-text-normal bg-scheme-shade_3 border-border-mid_contrast"
          />
        </div>

        <div className="p-2 border rounded-lg bg-scheme-shade_3 border-border-mid_contrast max-h-[400px] overflow-y-auto">
          {filteredCollections.map((collection) => {
            const collectionId = String(collection.id);
            const isSelected = selectedCollections.has(collectionId);

            return (
              <div key={collectionId} className="flex items-start gap-2 p-2 hover:bg-scheme-shade_4 rounded">
                <input
                  type="checkbox"
                  id={`collection-${collectionId}`}
                  checked={isSelected}
                  onChange={() => onToggleCollection(collectionId)}
                  className={`w-4 h-4 mt-[3px] shrink-0 rounded cursor-pointer relative border ${
                    isSelected
                      ? "bg-accent border-accent after:content-[''] after:absolute after:left-[4px] after:top-[1px] after:h-[8px] after:w-[4px] after:rotate-45 after:border-r-2 after:border-b-2 after:border-white"
                      : 'bg-scheme-shade_5 border-border-mid_contrast'
                  }`}
                  style={{
                    appearance: 'none',
                    WebkitAppearance: 'none',
                    MozAppearance: 'none',
                  }}
                />
                <label htmlFor={`collection-${collectionId}`} className="text-sm leading-6 text-text-normal cursor-pointer">
                  {collection.name}
                </label>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ChatCollectionsModal;
