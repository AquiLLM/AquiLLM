import React from 'react';
import { FileSystemItem } from '../../../types/FileSystemItem';

export interface FileSystemViewerBatchBarProps {
  mode: 'browse' | 'select';
  selectedCount: number;
  selectedItems: FileSystemItem[];
  onBatchMove: (items: FileSystemItem[]) => void;
  onBatchRemove: () => void;
  onClearSelection: () => void;
}

const FileSystemViewerBatchBar: React.FC<FileSystemViewerBatchBarProps> = ({
  mode,
  selectedCount,
  selectedItems,
  onBatchMove,
  onBatchRemove,
  onClearSelection,
}) => {
  if (selectedCount <= 0 || mode !== 'browse') return null;
  return (
    <div className="flex items-center justify-between bg-scheme-shade_3 p-3 mb-2 rounded-md border border-border-mid_contrast transition-all duration-300 ease-in-out">
      <div className="flex items-center">
        <span className="text-text-very_slightly_less_contrast mr-4">
          <strong>{selectedCount}</strong> {selectedCount === 1 ? 'item' : 'items'} selected
        </span>
        <button
          type="button"
          className="bg-accent hover:bg-accent-dark text-text-normal py-1 px-3 mr-2 rounded flex items-center transition-colors duration-200"
          onClick={() => onBatchMove(selectedItems)}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 7h12m0 0l-4-4m4 4l-4 4m-8 6H4m0 0l4 4m-4-4l4-4"
            />
          </svg>
          Move
        </button>
        <button
          type="button"
          className="bg-red-600 hover:bg-red-700 text-white py-1 px-3 rounded flex items-center transition-colors duration-200"
          onClick={onBatchRemove}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
            />
          </svg>
          Delete
        </button>
      </div>
      <button
        type="button"
        className="text-text-slightly_less_contrast hover:text-text-very_slightly_less_contrast transition-colors duration-200"
        onClick={onClearSelection}
        title="Clear selection"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
};

export default FileSystemViewerBatchBar;
