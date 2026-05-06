import React, { useEffect, useState } from 'react';
import type { QueryResultRow } from './types';

type Props = {
  columns: string[];
  rows: QueryResultRow[];
  isRowLevel: boolean;
  onOpenThread: (conversationId: string, messageUuid: string) => void;
};

function cellDisplay(v: unknown): string {
  if (v === null || v === undefined) return '—';
  const s = String(v);
  // Collapse ISO-ish timestamps to YYYY-MM-DD HH:MM. Full value still opens in the modal on click.
  const ts = s.match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/);
  if (ts) return `${ts[1]} ${ts[2]}`;
  // Collapse UUIDs to first 8 chars + ellipsis. Full UUID opens in the modal on click.
  if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(s)) {
    return s.slice(0, 8) + '…';
  }
  return s;
}

// Parse a raw filter string into a matcher. Supports:
//   ">3", ">=3", "<3", "<=3", "!=3"   numeric comparisons
//   "1-3"                              inclusive numeric range
//   "3"                                exact number OR substring fallback
//   anything else                      case-insensitive substring match
type Filter =
  | { kind: 'substring'; needle: string }
  | { kind: 'op'; op: '>' | '>=' | '<' | '<=' | '!='; value: number }
  | { kind: 'eq'; value: number; raw: string }
  | { kind: 'range'; min: number; max: number };

function parseFilter(raw: string): Filter | null {
  const s = raw.trim();
  if (!s) return null;
  const opMatch = s.match(/^(>=|<=|!=|>|<)\s*(-?\d+(?:\.\d+)?)$/);
  if (opMatch) {
    return { kind: 'op', op: opMatch[1] as Filter extends { kind: 'op' } ? Filter['op'] : never, value: Number(opMatch[2]) };
  }
  const rangeMatch = s.match(/^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$/);
  if (rangeMatch) {
    const a = Number(rangeMatch[1]);
    const b = Number(rangeMatch[2]);
    return { kind: 'range', min: Math.min(a, b), max: Math.max(a, b) };
  }
  if (/^-?\d+(?:\.\d+)?$/.test(s)) {
    return { kind: 'eq', value: Number(s), raw: s };
  }
  return { kind: 'substring', needle: s.toLowerCase() };
}

function cellMatches(cell: unknown, filter: Filter): boolean {
  if (filter.kind === 'substring') {
    if (cell === null || cell === undefined) return false;
    return String(cell).toLowerCase().includes(filter.needle);
  }
  const asNumber = typeof cell === 'number' ? cell : Number(cell);
  const numeric = !Number.isNaN(asNumber) && cell !== null && cell !== undefined && cell !== '';
  if (filter.kind === 'eq') {
    if (numeric && asNumber === filter.value) return true;
    // fall back to substring so "3" matches textual cells containing "3"
    if (cell === null || cell === undefined) return false;
    return String(cell).toLowerCase().includes(filter.raw.toLowerCase());
  }
  if (!numeric) return false;
  if (filter.kind === 'range') return asNumber >= filter.min && asNumber <= filter.max;
  switch (filter.op) {
    case '>': return asNumber > filter.value;
    case '>=': return asNumber >= filter.value;
    case '<': return asNumber < filter.value;
    case '<=': return asNumber <= filter.value;
    case '!=': return asNumber !== filter.value;
  }
}

const ResultsTable: React.FC<Props> = ({ columns, rows, isRowLevel, onOpenThread }) => {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [pageSize, setPageSize] = useState<number>(10);
  const [page, setPage] = useState<number>(1);
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [openFilterCol, setOpenFilterCol] = useState<string | null>(null);

  // Close the filter modal on Escape
  useEffect(() => {
    if (!openFilterCol) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpenFilterCol(null);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [openFilterCol]);

  useEffect(() => {
    if (expanded === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpanded(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [expanded]);

  // Reset to page 1 and drop column filters when a new query arrives
  useEffect(() => {
    setPage(1);
    setFilters({});
  }, [rows]);

  useEffect(() => {
    setPage(1);
  }, [pageSize, filters]);

  const activeFilters = Object.entries(filters)
    .map(([col, raw]) => ({ col, raw, parsed: parseFilter(raw) }))
    .filter((f): f is { col: string; raw: string; parsed: Filter } => f.parsed !== null);

  const filteredRows = activeFilters.length === 0
    ? rows
    : rows.filter((row) =>
        activeFilters.every(({ col, parsed }) => {
          const idx = columns.indexOf(col);
          if (idx === -1) return true;
          return cellMatches(row.cells[idx], parsed);
        }),
      );

  const unfilteredTotal = rows.length;
  const total = filteredRows.length;
  const effectivePageSize = pageSize === 0 ? total : pageSize;
  const totalPages = effectivePageSize > 0 ? Math.max(1, Math.ceil(total / effectivePageSize)) : 1;
  const safePage = Math.min(page, totalPages);
  const startIdx = pageSize === 0 ? 0 : (safePage - 1) * pageSize;
  const endIdx = pageSize === 0 ? total : Math.min(startIdx + pageSize, total);
  const visibleRows = filteredRows.slice(startIdx, endIdx);

  const updateFilter = (col: string, value: string) => {
    setFilters((prev) => {
      const next = { ...prev };
      if (value.trim() === '') delete next[col];
      else next[col] = value;
      return next;
    });
  };

  const clearFilter = (col: string) => {
    setFilters((prev) => {
      const next = { ...prev };
      delete next[col];
      return next;
    });
  };

  const clearAllFilters = () => setFilters({});

  return (
    <>
      <div className="flex flex-wrap items-center gap-3 mb-2 text-xs text-text-muted">
        <label className="flex items-center gap-2">
          Rows per page
          <input
            type="number"
            min={1}
            value={pageSize === 0 ? '' : pageSize}
            placeholder="10"
            onChange={(e) => {
              const v = e.target.value;
              if (v === '') { setPageSize(10); return; }
              const n = parseInt(v, 10);
              if (!Number.isNaN(n) && n > 0) setPageSize(n);
            }}
            className="w-20 px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono"
          />
          <button
            onClick={() => setPageSize(pageSize === 0 ? 20 : 0)}
            title={pageSize === 0 ? 'Switch to paged view' : 'Show all rows on one page'}
            className={`px-2 py-1 rounded element-border ${
              pageSize === 0 ? 'bg-blue-600/40 text-text-normal' : 'bg-scheme-shade_3 hover:bg-scheme-shade_5'
            }`}
          >
            All
          </button>
        </label>
        <span>
          Showing {total === 0 ? 0 : startIdx + 1}–{endIdx} of {total}
          {activeFilters.length > 0 && ` (filtered from ${unfilteredTotal})`}
        </span>
        {pageSize !== 0 && totalPages > 1 && (
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={safePage <= 1}
              className="px-2 py-1 rounded bg-scheme-shade_3 hover:bg-scheme-shade_5 disabled:opacity-40 disabled:cursor-not-allowed element-border"
            >
              Prev
            </button>
            <span className="font-mono">
              Page {safePage} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={safePage >= totalPages}
              className="px-2 py-1 rounded bg-scheme-shade_3 hover:bg-scheme-shade_5 disabled:opacity-40 disabled:cursor-not-allowed element-border"
            >
              Next
            </button>
          </div>
        )}
      </div>
      {activeFilters.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 mb-2 text-xs">
          {activeFilters.map(({ col, raw }) => (
            <span
              key={col}
              className="inline-flex items-center gap-1 px-2 py-1 rounded bg-blue-600/30 element-border font-mono"
            >
              <span className="text-text-normal">{col}</span>
              <span className="text-text-muted">=</span>
              <span className="text-text-normal">{raw}</span>
              <button
                onClick={() => clearFilter(col)}
                title="Clear filter"
                className="ml-1 text-text-muted hover:text-red-400"
              >
                ×
              </button>
            </span>
          ))}
          <button
            onClick={clearAllFilters}
            className="px-2 py-1 rounded bg-scheme-shade_3 hover:bg-scheme-shade_5 element-border"
          >
            Clear all
          </button>
        </div>
      )}
      <div className="overflow-x-auto rounded-lg element-border">
        <table className="w-full text-sm">
          <thead className="bg-scheme-shade_3">
            <tr>
              {isRowLevel && <th className="px-2 py-2 w-8"></th>}
              {columns.map((col) => {
                const active = !!filters[col]?.trim();
                const isOpen = openFilterCol === col;
                return (
                  <th key={col} className="px-3 py-2 text-left font-semibold whitespace-nowrap relative">
                    <div className="flex items-center gap-2">
                      <span>{col}</span>
                      <button
                        type="button"
                        onClick={() => setOpenFilterCol(isOpen ? null : col)}
                        title={active ? `Filter: ${filters[col]}` : 'Filter this column'}
                        className={`transition-colors ${
                          active ? 'text-blue-400' : 'text-slate-400 hover:text-blue-400'
                        }`}
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="w-4 h-4"
                          viewBox="0 0 24 24"
                          fill="currentColor"
                        >
                          <path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z" />
                        </svg>
                      </button>
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, rowIdx) => (
              <tr key={rowIdx} className="border-t border-scheme-contrast/20 hover:bg-scheme-shade_3/50">
                {isRowLevel && (
                  <td className="px-2 py-2 text-center">
                    <button
                      className="text-text-low_contrast hover:text-blue-500 transition-colors"
                      title="View full conversation"
                      onClick={() =>
                        onOpenThread(row.conversation_id || '', row.message_uuid || '')
                      }
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="w-4 h-4"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth={2}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                      </svg>
                    </button>
                  </td>
                )}
                {row.cells.map((cell, cellIdx) => {
                  const display = cellDisplay(cell);
                  const raw = cell === null || cell === undefined ? '' : String(cell);
                  return (
                    <td
                      key={cellIdx}
                      className="px-3 py-2 cursor-pointer hover:text-blue-500"
                      onClick={() => raw && setExpanded(raw)}
                    >
                      <div className="truncate" style={{ maxWidth: '12rem' }}>
                        {display}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {openFilterCol && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
        >
          <div
            className="bg-scheme-shade_2 element-border rounded-lg w-full max-w-md p-5 shadow-2xl"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm">
                Filter column <span className="font-mono font-semibold text-text-normal">{openFilterCol}</span>
              </div>
              <button
                onClick={() => setOpenFilterCol(null)}
                title="Close"
                className="w-7 h-7 flex items-center justify-center rounded hover:bg-scheme-shade_5 text-text-muted hover:text-text-normal"
              >
                ×
              </button>
            </div>
            <input
              autoFocus
              type="text"
              value={filters[openFilterCol] || ''}
              onChange={(e) => updateFilter(openFilterCol, e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') setOpenFilterCol(null); }}
              placeholder="filter value"
              className="w-full px-3 py-2 rounded bg-scheme-shade_3 element-border text-sm font-mono placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            />
            <p className="text-xs text-text-muted mt-3 leading-relaxed">
              Substring match (case-insensitive). Numeric cells also accept{' '}
              <span className="font-mono text-text-normal">&gt;3</span>,{' '}
              <span className="font-mono text-text-normal">&lt;=2</span>,{' '}
              <span className="font-mono text-text-normal">!=0</span>, or a range like{' '}
              <span className="font-mono text-text-normal">1-3</span>.
            </p>
            <div className="flex justify-end gap-2 mt-4">
              {filters[openFilterCol]?.trim() && (
                <button
                  onClick={() => { clearFilter(openFilterCol); setOpenFilterCol(null); }}
                  className="px-3 py-1.5 rounded text-xs bg-scheme-shade_3 hover:bg-scheme-shade_5 element-border"
                >
                  Clear
                </button>
              )}
              <button
                onClick={() => setOpenFilterCol(null)}
                className="px-3 py-1.5 rounded text-xs bg-blue-600 hover:bg-blue-700 text-white font-semibold"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {expanded !== null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
          onClick={() => setExpanded(null)}
        >
          <div
            className="bg-scheme-shade_2 element-border rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <pre className="whitespace-pre-wrap text-sm text-text-normal font-mono break-words">
              {expanded}
            </pre>
            <button
              onClick={() => setExpanded(null)}
              className="mt-4 px-4 py-1.5 rounded-lg bg-scheme-shade_5 hover:bg-scheme-shade_6 text-sm transition-all"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default ResultsTable;
