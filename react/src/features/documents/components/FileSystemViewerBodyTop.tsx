import React from 'react';
import { Collection } from '../../../components/CollectionsTree';

export interface FileSystemViewerBodyTopProps {
  collection: Collection;
  searchQuery: string;
  onSearchChange: (value: string) => void;
  onClearSearch: () => void;
}

const FileSystemViewerBodyTop: React.FC<FileSystemViewerBodyTopProps> = ({
  collection,
  searchQuery,
  onSearchChange,
  onClearSearch,
}) => (
  <div style={{ display: 'flex', justifyContent: 'space-between' }} className="bg-scheme-shade_4 p-[16px] border-b border-b-scheme-shade_6">
    <div className="flex gap-[16px]">
      <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
        <div style={{ position: 'absolute', left: '10px', pointerEvents: 'none' }}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-text-slightly_less_contrast"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </div>
        <input
          type="text"
          placeholder="Search items..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="bg-scheme-shade_4 border border-border-high_contrast placeholder:text-text-slightly_less_contrast rounded-[20px] w-[220px]"
          style={{
            padding: '0.5rem',
            paddingLeft: '2rem',
            paddingRight: searchQuery ? '2rem' : '0.5rem',
            outline: 'none',
          }}
        />
        {searchQuery && (
          <div
            style={{ position: 'absolute', right: '10px', cursor: 'pointer' }}
            onClick={onClearSearch}
            title="Clear search"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="text-text-slightly_less_contrast hover:text-text-very_slightly_less_contrast"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </div>
        )}
      </div>
      <div className="h-full border-r border-border-mid_contrast" />
      <span className="flex items-center text-align-center text-text-less_contrast text-sm">
        Path: Root/{collection.path}
      </span>
    </div>
  </div>
);

export default FileSystemViewerBodyTop;
