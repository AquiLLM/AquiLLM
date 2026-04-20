import React, { useState, useMemo, useEffect } from 'react';
import { FileSystemItem } from '../../../types/FileSystemItem';
import { Collection } from '../../../components/CollectionsTree';
import FileSystemViewerBodyTop from './FileSystemViewerBodyTop';
import FileSystemViewerBatchBar from './FileSystemViewerBatchBar';
import FileSystemViewerDataTable from './FileSystemViewerDataTable';
import FileSystemViewerPaginationBar from './FileSystemViewerPaginationBar';

interface FileSystemViewerProps {
  mode: 'browse' | 'select';
  items: FileSystemItem[];
  collection: Collection;
  onOpenItem?: (item: FileSystemItem) => void;
  onRemoveItem?: (item: FileSystemItem) => void;
  onSelectCollection?: (item: FileSystemItem) => void;
  onMove?: (item: FileSystemItem) => void;
  onContextMenuRename?: (item: FileSystemItem) => void;
  onBatchMove?: (items: FileSystemItem[]) => void;
  onRemoveBatch?: (items: FileSystemItem[]) => void;
}

const FileSystemViewer: React.FC<FileSystemViewerProps> = ({
  mode,
  items,
  collection,
  onOpenItem,
  onRemoveItem,
  onSelectCollection,
  onMove,
  onContextMenuRename,
  onBatchMove,
  onRemoveBatch,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIds, setSelectedIds] = useState<Set<number | string>>(new Set());
  const [sortKey, setSortKey] = useState<'name' | 'type' | 'details' | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [rowsPerPage, setRowsPerPage] = useState<number>(10);
  const [currentPage, setCurrentPage] = useState<number>(1);

  const filteredItems = useMemo(() => {
    if (!searchQuery.trim()) return items;
    const normalizedQuery = searchQuery.trim().toLowerCase();
    return items.filter((item) => item.name.toLowerCase().includes(normalizedQuery));
  }, [items, searchQuery]);

  const sortedItems = useMemo(() => {
    if (!sortKey) return filteredItems;
    const sorted = [...filteredItems];
    const directionFactor = sortDirection === 'asc' ? 1 : -1;
    const asTimestamp = (value?: string) => {
      const parsed = Date.parse(value || '');
      return Number.isNaN(parsed) ? 0 : parsed;
    };
    sorted.sort((a, b) => {
      if (sortKey === 'name') {
        return directionFactor * a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
      }
      if (sortKey === 'type') {
        return directionFactor * a.type.localeCompare(b.type, undefined, { sensitivity: 'base' });
      }
      return directionFactor * (asTimestamp(a.created_at) - asTimestamp(b.created_at));
    });
    return sorted;
  }, [filteredItems, sortKey, sortDirection]);

  const totalPages = Math.max(1, Math.ceil(sortedItems.length / rowsPerPage));
  const activePage = Math.min(currentPage, totalPages);
  const pageStart = (activePage - 1) * rowsPerPage;
  const pageEnd = pageStart + rowsPerPage;
  const paginatedItems = sortedItems.slice(pageStart, pageEnd);

  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, rowsPerPage, items.length, sortKey, sortDirection]);

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [currentPage, totalPages]);

  const [contextMenu, setContextMenu] = useState<{
    visible: boolean;
    x: number;
    y: number;
    item: FileSystemItem | null;
  }>({ visible: false, x: 0, y: 0, item: null });

  const handleContextMenu = (e: React.MouseEvent, item: FileSystemItem) => {
    e.preventDefault();
    setContextMenu({
      visible: true,
      x: e.clientX,
      y: e.clientY,
      item,
    });
  };

  useEffect(() => {
    const handleClickOutside = () => {
      if (contextMenu.visible) {
        setContextMenu({ visible: false, x: 0, y: 0, item: null });
      }
    };
    if (contextMenu.visible) {
      document.addEventListener('click', handleClickOutside);
    }
    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, [contextMenu.visible]);

  const handleToggleSelect = (itemId: number | string) => {
    setSelectedIds((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  const allDisplayedSelected = paginatedItems.length > 0 && paginatedItems.every((item) => selectedIds.has(item.id));

  const selectedItems = useMemo(() => items.filter((item) => selectedIds.has(item.id)), [items, selectedIds]);

  useEffect(() => {
    const remainingIds = new Set(items.map((item) => item.id));
    setSelectedIds((prev) => {
      const next = new Set<number | string>();
      prev.forEach((id) => {
        if (remainingIds.has(id)) {
          next.add(id);
        }
      });
      return next;
    });
  }, [items]);

  const handleToggleSelectAll = () => {
    if (allDisplayedSelected) {
      setSelectedIds((prev) => {
        const newSet = new Set(prev);
        paginatedItems.forEach((item) => newSet.delete(item.id));
        return newSet;
      });
    } else {
      setSelectedIds((prev) => {
        const newSet = new Set(prev);
        paginatedItems.forEach((item) => newSet.add(item.id));
        return newSet;
      });
    }
  };

  const toggleSort = (nextKey: 'name' | 'type' | 'details') => {
    if (sortKey === nextKey) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
      return;
    }
    setSortKey(nextKey);
    setSortDirection('asc');
  };

  const sortSuffix = (key: 'name' | 'type' | 'details') => {
    if (sortKey !== key) return '';
    return sortDirection === 'asc' ? ' (asc)' : ' (desc)';
  };

  const handleBatchMove = () => {
    const sel = items.filter((i) => selectedIds.has(i.id));
    if (sel.length === 0) {
      alert('No items selected');
      return;
    }
    onBatchMove?.(sel);
  };

  const handleBatchRemove = () => {
    const sel = items.filter((i) => selectedIds.has(i.id));
    if (sel.length === 0) {
      alert('No items selected');
      return;
    }
    if (typeof onRemoveBatch === 'function') {
      onRemoveBatch(sel);
    } else if (window.confirm(`Are you sure you want to delete ${sel.length} selected items?`)) {
      sel.forEach((item) => onRemoveItem?.(item));
      setSelectedIds(new Set());
    }
  };

  return (
    <div className="bg-scheme-shade_4 rounded-[36px] border border-border-mid_contrast overflow-hidden">
      <FileSystemViewerBodyTop
        collection={collection}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onClearSearch={() => setSearchQuery('')}
      />
      <FileSystemViewerBatchBar
        mode={mode}
        selectedCount={selectedIds.size}
        selectedItems={selectedItems}
        onBatchMove={(sel) => onBatchMove?.(sel)}
        onBatchRemove={handleBatchRemove}
        onClearSelection={() => setSelectedIds(new Set())}
      />
      <FileSystemViewerDataTable
        paginatedItems={paginatedItems}
        selectedIds={selectedIds}
        searchQuery={searchQuery}
        onClearSearch={() => setSearchQuery('')}
        onToggleSort={toggleSort}
        sortSuffix={sortSuffix}
        allDisplayedSelected={allDisplayedSelected}
        onToggleSelectAll={handleToggleSelectAll}
        onToggleSelect={handleToggleSelect}
        onOpenItem={onOpenItem}
        onContextMenu={handleContextMenu}
        contextMenu={contextMenu}
        onCloseContextMenu={() => setContextMenu({ visible: false, x: 0, y: 0, item: null })}
        onRemoveItem={onRemoveItem}
        onMove={onMove}
        onContextMenuRename={onContextMenuRename}
      />
      <FileSystemViewerPaginationBar
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={setRowsPerPage}
        sortedLength={sortedItems.length}
        pageStart={pageStart}
        pageEnd={pageEnd}
        activePage={activePage}
        totalPages={totalPages}
        onPrevPage={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
        onNextPage={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
      />
    </div>
  );
};

export default FileSystemViewer;
