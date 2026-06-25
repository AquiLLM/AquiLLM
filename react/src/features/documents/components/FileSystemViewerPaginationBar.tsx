import React from 'react';

export interface FileSystemViewerPaginationBarProps {
  rowsPerPage: number;
  onRowsPerPageChange: (n: number) => void;
  sortedLength: number;
  pageStart: number;
  pageEnd: number;
  activePage: number;
  totalPages: number;
  onPrevPage: () => void;
  onNextPage: () => void;
}

const FileSystemViewerPaginationBar: React.FC<FileSystemViewerPaginationBarProps> = ({
  rowsPerPage,
  onRowsPerPageChange,
  sortedLength,
  pageStart,
  pageEnd,
  activePage,
  totalPages,
  onPrevPage,
  onNextPage,
}) => (
  <div
    className="text-text-normal bg-scheme-shade_3"
    style={{
      display: 'flex',
      justifyContent: 'flex-end',
      alignItems: 'center',
      gap: '1rem',
      padding: '1rem',
    }}
  >
    <label className="text-text-normal flex items-center gap-2">
      <span>Rows per page:</span>
      <select
        value={rowsPerPage}
        onChange={(e) => onRowsPerPageChange(Number(e.target.value))}
        className="bg-scheme-shade_4 border border-border-mid_contrast rounded px-2 py-1 text-text-normal"
      >
        <option value={10}>10</option>
        <option value={25}>25</option>
        <option value={50}>50</option>
      </select>
    </label>
    <span className="text-text-normal">
      {sortedLength === 0 ? '0-0 of 0' : `${pageStart + 1}-${Math.min(pageEnd, sortedLength)} of ${sortedLength}`}
    </span>
    <button
      type="button"
      style={{ background: 'none', border: 'none' }}
      className="text-text-normal disabled:opacity-40 disabled:cursor-not-allowed"
      onClick={onPrevPage}
      disabled={activePage <= 1}
      aria-label="Previous page"
    >
      {'<'}
    </button>
    <button
      type="button"
      style={{ background: 'none', border: 'none' }}
      className="text-text-normal disabled:opacity-40 disabled:cursor-not-allowed"
      onClick={onNextPage}
      disabled={activePage >= totalPages}
      aria-label="Next page"
    >
      {'>'}
    </button>
  </div>
);

export default FileSystemViewerPaginationBar;
