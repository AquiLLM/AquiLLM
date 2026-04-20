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

const ResultsTable: React.FC<Props> = ({ columns, rows, isRowLevel, onOpenThread }) => {
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    if (expanded === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpanded(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [expanded]);

  return (
    <>
      <div className="overflow-x-auto rounded-lg element-border">
        <table className="w-full text-sm">
          <thead className="bg-scheme-shade_3">
            <tr>
              {isRowLevel && <th className="px-2 py-2 w-8"></th>}
              {columns.map((col) => (
                <th key={col} className="px-3 py-2 text-left font-semibold whitespace-nowrap">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => (
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
