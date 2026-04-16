import React, { useState } from 'react';
import { FeedbackRow } from './types';

interface FeedbackTableProps {
  rows: FeedbackRow[];
  loading: boolean;
  error: string | null;
  totalCount: number;
  totalPages: number;
  currentPage: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString(undefined, {
      year:   'numeric',
      month:  'short',
      day:    'numeric',
      hour:   '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

function ratingBadge(rating: number | null): React.ReactNode {
  if (rating === null) return <span className="text-text-low_contrast">—</span>;
  const color =
    rating >= 4 ? 'text-green bg-green/10' :
    rating === 3 ? 'text-accent bg-accent/10' :
                   'text-red bg-red/10';
  return (
    <span className={`inline-flex items-center px-2 py-[2px] rounded-full text-xs font-semibold ${color}`}>
      {rating}★
    </span>
  );
}

function roleBadge(role: string): React.ReactNode {
  const color =
    role === 'assistant' ? 'bg-accent/10 text-accent' :
    role === 'user'      ? 'bg-scheme-shade_6 text-text-normal' :
                           'bg-scheme-shade_5 text-text-less_contrast';
  return (
    <span className={`inline-flex items-center px-2 py-[2px] rounded-full text-xs ${color}`}>
      {role}
    </span>
  );
}

// RowDetail renders the full expanded view when a row is clicked
interface RowDetailProps {
  row: FeedbackRow;
  onClose: () => void;
}

const RowDetail: React.FC<RowDetailProps> = ({ row, onClose }) => {
  return (
    <tr>
      <td
        colSpan={10}
        className="bg-scheme-shade_4 border-b border-border-mid_contrast px-6 py-4"
      >
        <div className="flex items-start justify-between mb-3">
          <span className="text-sm font-semibold text-text-normal">Row detail</span>
          <button
            onClick={onClose}
            className="text-text-low_contrast hover:text-text-normal transition-colors text-lg leading-none"
            aria-label="close detail"
          >
            ×
          </button>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-2 text-sm">
          <DetailField label="Message UUID"       value={row.message_uuid} />
          <DetailField label="Conversation ID"    value={String(row.conversation_id)} />
          <DetailField label="Conversation name"  value={row.conversation_name ?? '—'} />
          <DetailField label="User ID"            value={String(row.user_id)} />
          <DetailField label="Username"           value={row.username} />
          <DetailField label="Role"               value={row.role} />
          <DetailField label="Rating"             value={row.rating !== null ? `${row.rating}` : '—'} />
          <DetailField label="Model"              value={row.model ?? '—'} />
          <DetailField label="Tool call name"     value={row.tool_call_name ?? '—'} />
          <DetailField label="Usage (tokens)"     value={String(row.usage)} />
          <DetailField label="Has feedback text"  value={row.has_feedback_text ? 'yes' : 'no'} />
          <DetailField label="Created at"         value={formatDate(row.created_at)} />
          <DetailField label="Feedback submitted" value={formatDate(row.feedback_submitted_at)} />
          <DetailField label="Effective date"     value={formatDate(row.effective_date)} />
        </div>
        {row.feedback_text && (
          <div className="mt-3">
            <span className="text-xs text-text-low_contrast uppercase tracking-wide block mb-1">
              Feedback text
            </span>
            <p className="text-sm text-text-normal whitespace-pre-wrap bg-scheme-shade_3 rounded-[8px] p-3 border border-border-low_contrast">
              {row.feedback_text}
            </p>
          </div>
        )}
        {row.content_snippet && (
          <div className="mt-3">
            <span className="text-xs text-text-low_contrast uppercase tracking-wide block mb-1">
              Content snippet
            </span>
            <p className="text-sm text-text-less_contrast whitespace-pre-wrap bg-scheme-shade_3 rounded-[8px] p-3 border border-border-low_contrast">
              {row.content_snippet}
            </p>
          </div>
        )}
      </td>
    </tr>
  );
};

interface DetailFieldProps {
  label: string;
  value: string;
}

const DetailField: React.FC<DetailFieldProps> = ({ label, value }) => (
  <div className="flex flex-col">
    <span className="text-xs text-text-low_contrast">{label}</span>
    <span className="text-text-normal font-mono text-xs break-all">{value}</span>
  </div>
);

// column header with consistent styling
interface ThProps {
  children: React.ReactNode;
  className?: string;
}

const Th: React.FC<ThProps> = ({ children, className = '' }) => (
  <th
    className={`px-3 py-2 text-left text-xs font-semibold text-text-low_contrast uppercase tracking-wide whitespace-nowrap ${className}`}
  >
    {children}
  </th>
);

// data cell
interface TdProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
}

const Td: React.FC<TdProps> = ({ children, className = '', title }) => (
  <td
    className={`px-3 py-2 text-sm text-text-normal align-top ${className}`}
    title={title}
  >
    {children}
  </td>
);

const FeedbackTable: React.FC<FeedbackTableProps> = ({
  rows,
  loading,
  error,
  totalCount,
  totalPages,
  currentPage,
  pageSize,
  onPageChange,
}) => {
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const toggleRow = (id: number) => {
    setExpandedId(prev => (prev === id ? null : id));
  };

  if (error) {
    return (
      <div className="rounded-[12px] bg-scheme-shade_3 border border-border-mid_contrast p-4 text-red">
        failed to load rows: {error}
      </div>
    );
  }

  const startRow = totalCount === 0 ? 0 : (currentPage - 1) * pageSize + 1;
  const endRow   = Math.min(currentPage * pageSize, totalCount);

  return (
    <div className="flex flex-col gap-3">
      {/* table header info + pagination */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-text-less_contrast">
          {loading
            ? 'Loading…'
            : totalCount === 0
            ? 'No results'
            : `${startRow}–${endRow} of ${totalCount.toLocaleString()} rows`}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage <= 1 || loading}
            className="px-3 py-1 rounded-[8px] text-sm bg-scheme-shade_4 border border-border-mid_contrast text-text-normal disabled:opacity-40 hover:bg-scheme-shade_5 transition-colors disabled:cursor-not-allowed"
          >
            ← Prev
          </button>
          <span className="text-sm text-text-normal px-2">
            {currentPage} / {totalPages}
          </span>
          <button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage >= totalPages || loading}
            className="px-3 py-1 rounded-[8px] text-sm bg-scheme-shade_4 border border-border-mid_contrast text-text-normal disabled:opacity-40 hover:bg-scheme-shade_5 transition-colors disabled:cursor-not-allowed"
          >
            Next →
          </button>
        </div>
      </div>

      {/* table */}
      <div className="rounded-[12px] border border-border-mid_contrast overflow-hidden overflow-x-auto">
        <table className="min-w-full border-collapse">
          <thead className="bg-scheme-shade_4">
            <tr>
              <Th>Date</Th>
              <Th>User</Th>
              <Th>Conversation</Th>
              <Th>Rating</Th>
              <Th>Role</Th>
              <Th>Model</Th>
              <Th>Tool</Th>
              <Th className="max-w-[200px]">Feedback text</Th>
              <Th className="max-w-[200px]">Content snippet</Th>
              <Th>{/* expand chevron column */}</Th>
            </tr>
          </thead>
          <tbody>
            {loading && rows.length === 0 && (
              <tr>
                <td colSpan={10} className="px-3 py-8 text-center text-text-low_contrast text-sm">
                  Loading…
                </td>
              </tr>
            )}
            {!loading && rows.length === 0 && (
              <tr>
                <td colSpan={10} className="px-3 py-8 text-center text-text-low_contrast text-sm">
                  No feedback matching current filters.
                </td>
              </tr>
            )}
            {rows.map(row => {
              const isExpanded = expandedId === row.id;
              return (
                <React.Fragment key={row.id}>
                  <tr
                    onClick={() => toggleRow(row.id)}
                    className={`border-b border-border-low_contrast cursor-pointer transition-colors ${
                      isExpanded
                        ? 'bg-scheme-shade_5'
                        : 'hover:bg-scheme-shade_4'
                    } ${loading ? 'opacity-50' : ''}`}
                  >
                    <Td className="whitespace-nowrap">
                      {formatDate(row.effective_date)}
                    </Td>
                    <Td className="whitespace-nowrap font-medium">
                      {row.username}
                    </Td>
                    <Td
                      className="max-w-[160px] truncate"
                      title={row.conversation_name ?? ''}
                    >
                      {row.conversation_name ?? <span className="text-text-low_contrast">—</span>}
                    </Td>
                    <Td>{ratingBadge(row.rating)}</Td>
                    <Td>{roleBadge(row.role)}</Td>
                    <Td
                      className="whitespace-nowrap text-text-less_contrast"
                      title={row.model ?? ''}
                    >
                      {row.model
                        ? row.model.length > 20
                          ? row.model.slice(0, 20) + '…'
                          : row.model
                        : <span className="text-text-low_contrast">—</span>}
                    </Td>
                    <Td className="whitespace-nowrap text-text-less_contrast">
                      {row.tool_call_name ?? <span className="text-text-low_contrast">—</span>}
                    </Td>
                    <Td className="max-w-[200px]" title={row.feedback_text ?? ''}>
                      {row.feedback_text
                        ? <span className="line-clamp-2">{row.feedback_text}</span>
                        : <span className="text-text-low_contrast">—</span>}
                    </Td>
                    <Td className="max-w-[200px]" title={row.content_snippet ?? ''}>
                      {row.content_snippet
                        ? <span className="line-clamp-2 text-text-less_contrast">{row.content_snippet}</span>
                        : <span className="text-text-low_contrast">—</span>}
                    </Td>
                    <Td>
                      <span
                        className="text-text-low_contrast transition-transform inline-block select-none"
                        style={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
                      >
                        ▾
                      </span>
                    </Td>
                  </tr>
                  {isExpanded && (
                    <RowDetail row={row} onClose={() => setExpandedId(null)} />
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* bottom pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 flex-wrap">
          <button
            onClick={() => onPageChange(1)}
            disabled={currentPage <= 1 || loading}
            className="px-3 py-1 rounded-[8px] text-sm bg-scheme-shade_4 border border-border-mid_contrast text-text-normal disabled:opacity-40 hover:bg-scheme-shade_5 transition-colors disabled:cursor-not-allowed"
          >
            « First
          </button>
          <button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage <= 1 || loading}
            className="px-3 py-1 rounded-[8px] text-sm bg-scheme-shade_4 border border-border-mid_contrast text-text-normal disabled:opacity-40 hover:bg-scheme-shade_5 transition-colors disabled:cursor-not-allowed"
          >
            ← Prev
          </button>

          {Array.from({ length: totalPages }, (_, i) => i + 1)
            .filter(p =>
              p === 1 ||
              p === totalPages ||
              Math.abs(p - currentPage) <= 2
            )
            .reduce<Array<number | '…'>>((acc, p, idx, arr) => {
              if (
                idx > 0 &&
                typeof arr[idx - 1] === 'number' &&
                (p as number) - (arr[idx - 1] as number) > 1
              ) {
                acc.push('…');
              }
              acc.push(p);
              return acc;
            }, [])
            .map((p, idx) =>
              p === '…' ? (
                <span key={`ellipsis-${idx}`} className="px-2 text-text-low_contrast text-sm">…</span>
              ) : (
                <button
                  key={p}
                  onClick={() => onPageChange(p as number)}
                  disabled={loading}
                  className={`px-3 py-1 rounded-[8px] text-sm border transition-colors disabled:cursor-not-allowed ${
                    p === currentPage
                      ? 'bg-accent text-slight_muted_white border-accent font-semibold'
                      : 'bg-scheme-shade_4 border-border-mid_contrast text-text-normal hover:bg-scheme-shade_5'
                  }`}
                >
                  {p}
                </button>
              )
            )}

          <button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage >= totalPages || loading}
            className="px-3 py-1 rounded-[8px] text-sm bg-scheme-shade_4 border border-border-mid_contrast text-text-normal disabled:opacity-40 hover:bg-scheme-shade_5 transition-colors disabled:cursor-not-allowed"
          >
            Next →
          </button>
          <button
            onClick={() => onPageChange(totalPages)}
            disabled={currentPage >= totalPages || loading}
            className="px-3 py-1 rounded-[8px] text-sm bg-scheme-shade_4 border border-border-mid_contrast text-text-normal disabled:opacity-40 hover:bg-scheme-shade_5 transition-colors disabled:cursor-not-allowed"
          >
            Last »
          </button>
        </div>
      )}
    </div>
  );
};

export default FeedbackTable;