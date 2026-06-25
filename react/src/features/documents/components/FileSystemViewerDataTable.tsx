import React from 'react';
import { FileSystemItem } from '../../../types/FileSystemItem';
import ContextMenu from '../../../components/CustomContextMenu';
import { Folder, File } from 'lucide-react';
import { typeToTextColorClass } from './fileSystemTypeColors';

export interface FileSystemViewerDataTableProps {
  paginatedItems: FileSystemItem[];
  selectedIds: Set<number | string>;
  searchQuery: string;
  onClearSearch: () => void;
  onToggleSort: (key: 'name' | 'type' | 'details') => void;
  sortSuffix: (key: 'name' | 'type' | 'details') => string;
  allDisplayedSelected: boolean;
  onToggleSelectAll: () => void;
  onToggleSelect: (id: number | string) => void;
  onOpenItem?: (item: FileSystemItem) => void;
  onContextMenu: (e: React.MouseEvent, item: FileSystemItem) => void;
  contextMenu: { visible: boolean; x: number; y: number; item: FileSystemItem | null };
  onCloseContextMenu: () => void;
  onRemoveItem?: (item: FileSystemItem) => void;
  onMove?: (item: FileSystemItem) => void;
  onContextMenuRename?: (item: FileSystemItem) => void;
}

const getIconForType = (type: string) => {
  switch (type) {
    case 'collection':
      return <Folder size={20} />;
    default:
      return <File size={20} />;
  }
};

const FileSystemViewerDataTable: React.FC<FileSystemViewerDataTableProps> = ({
  paginatedItems,
  selectedIds,
  searchQuery,
  onClearSearch,
  onToggleSort,
  sortSuffix,
  allDisplayedSelected,
  onToggleSelectAll,
  onToggleSelect,
  onOpenItem,
  onContextMenu,
  contextMenu,
  onCloseContextMenu,
  onRemoveItem,
  onMove,
  onContextMenuRename,
}) => (
  <div style={{ overflow: 'auto' }}>
    <table style={{ width: '100%', height: '100%', borderCollapse: 'collapse' }}>
      <thead className="bg-scheme-shade_4">
        <tr className="border-b border-l border-r border-border-mid_contrast h-[40px] max-h-[40px]">
          <th style={{ textAlign: 'center', width: '64px' }} className="h-full align-middle">
            <div
              style={{
                padding: '4px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              className="hover:bg-scheme-shade_5 rounded"
              onClick={(e) => {
                e.stopPropagation();
                onToggleSelectAll();
              }}
            >
              <input
                type="checkbox"
                checked={allDisplayedSelected}
                onChange={(e) => {
                  e.stopPropagation();
                  onToggleSelectAll();
                }}
                onClick={(e) => e.stopPropagation()}
                className={`w-4 h-4 rounded cursor-pointer relative border ${
                  allDisplayedSelected
                    ? 'bg-accent border-accent after:content-["✓"] after:absolute after:text-text-normal after:text-xs after:top-[-1px] after:left-[3px]'
                    : 'bg-scheme-shade_5 border-border-mid_contrast'
                }`}
                style={{
                  zIndex: 10,
                  appearance: 'none',
                  WebkitAppearance: 'none',
                  MozAppearance: 'none',
                }}
              />
            </div>
          </th>
          <th style={{ textAlign: 'left' }}>
            <button
              type="button"
              onClick={() => onToggleSort('name')}
              className="bg-transparent border-none p-0 cursor-pointer text-text-normal font-inherit"
            >
              Name{sortSuffix('name')}
            </button>
          </th>
          <th style={{ textAlign: 'left' }}>
            <button
              type="button"
              onClick={() => onToggleSort('type')}
              className="bg-transparent border-none p-0 cursor-pointer text-text-normal font-inherit"
            >
              Type{sortSuffix('type')}
            </button>
          </th>
          <th style={{ textAlign: 'left' }}>
            <button
              type="button"
              onClick={() => onToggleSort('details')}
              className="bg-transparent border-none p-0 cursor-pointer text-text-normal font-inherit"
            >
              Details{sortSuffix('details')}
            </button>
          </th>
        </tr>
      </thead>
      <tbody className="bg-scheme-shade_3">
        {paginatedItems.length > 0 ? (
          paginatedItems.map((item) => {
            const isSelected = selectedIds.has(item.id);
            return (
              <tr
                key={item.id}
                onContextMenu={(e) => onContextMenu(e, item)}
                className={`h-[40px] max-h-[40px] hover:bg-scheme-shade_3 transition-colors border border-border-mid_contrast ${
                  isSelected ? 'bg-accent bg-opacity-10' : 'bg-transparent'
                } ${typeToTextColorClass[item.type as keyof typeof typeToTextColorClass] || ''}`}
                style={{ cursor: 'pointer' }}
                onClick={() => onOpenItem?.(item)}
              >
                <td
                  style={{ textAlign: 'center', width: '64px' }}
                  onClick={(e) => {
                    e.stopPropagation();
                  }}
                >
                  <div
                    style={{
                      padding: '4px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                    className="hover:bg-scheme-shade_5 rounded"
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleSelect(item.id);
                    }}
                  >
                    <input
                      type="checkbox"
                      className={`w-4 h-4 rounded cursor-pointer relative border ${
                        isSelected
                          ? "bg-accent border-accent after:content-['✓'] after:absolute after:text-white after:text-xs after:top-[-1px] after:left-[3px]"
                          : 'bg-scheme-shade_5 border-border-mid_contrast'
                      }`}
                      style={{
                        zIndex: 10,
                        appearance: 'none',
                        WebkitAppearance: 'none',
                        MozAppearance: 'none',
                      }}
                      checked={isSelected}
                      onChange={(e) => {
                        e.stopPropagation();
                        onToggleSelect(item.id);
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                      }}
                    />
                  </div>
                </td>
                <td style={{ textAlign: 'left', fontSize: '14px' }}>{item.name}</td>
                <td
                  style={{ textAlign: 'left', fontSize: '14px' }}
                  className="flex justify-left items-center gap-[16px] h-full"
                >
                  {getIconForType(item.type)}
                  {item.type}
                </td>
                <td style={{ textAlign: 'left', fontSize: '14px' }}>{item.created_at}</td>
              </tr>
            );
          })
        ) : (
          <tr>
            <td colSpan={4} className="text-center py-8 text-text-less_contrast">
              {searchQuery ? (
                <div className="flex flex-col items-center justify-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="mb-2"
                  >
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  </svg>
                  <p>No items match your search for &quot;{searchQuery}&quot;</p>
                  <button type="button" onClick={onClearSearch} className="mt-2 text-accent-light hover:underline">
                    Clear search
                  </button>
                </div>
              ) : (
                <p>No items in this collection</p>
              )}
            </td>
          </tr>
        )}
      </tbody>
    </table>
    {contextMenu.visible && contextMenu.item && (
      <ContextMenu
        x={contextMenu.x}
        y={contextMenu.y}
        item={contextMenu.item}
        onClose={onCloseContextMenu}
        onViewDetails={(item) => {
          console.log('View details for', item);
        }}
        onRename={(item) => {
          console.log('Rename', item);
          onContextMenuRename?.(item);
        }}
        onMove={(item) => {
          console.log('Move', item);
          onMove?.(item);
        }}
        onRemove={(item) => {
          onRemoveItem?.(item);
        }}
      />
    )}
  </div>
);

export default FileSystemViewerDataTable;
