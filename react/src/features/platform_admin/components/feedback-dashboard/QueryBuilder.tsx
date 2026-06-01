import React, { useState } from 'react';
import type { AdvancedState, WhereRow, SummarizeRow, OrderByRow } from './useAdvancedBuilder';

// ---------------------------------------------------------------------------
// Field / operator constants
// ---------------------------------------------------------------------------

const FIELDS = [
  'rating', 'feedback_text', 'feedback_submitted_at', 'model', 'role', 'content',
  'sequence_number', 'created_at', 'message_uuid', 'tool_call_name', 'user_id',
  'conversation_id', 'conversation_tool',
] as const;

const SELECTABLE_FIELDS = FIELDS.filter(f => f !== 'conversation_tool');

const OPERATORS = [
  { value: '==',     label: '== (equals)'      },
  { value: '!=',     label: '!= (not equals)'   },
  { value: '<',      label: '< (less than)'     },
  { value: '>',      label: '> (greater than)'  },
  { value: '<=',     label: '<= (less or equal)'},
  { value: '>=',     label: '>= (greater or equal)'},
  { value: 'contains',   label: 'contains'      },
  { value: 'startswith', label: 'startswith'    },
  { value: 'in',         label: 'in [list]'     },
  { value: '== null',    label: 'is null'       },
  { value: '!= null',    label: 'is not null'   },
] as const;

const AGG_FUNCTIONS = ['count', 'avg', 'min', 'max', 'sum', 'median'] as const;

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export type QueryBuilderAdvancedProps = {
  state: AdvancedState;
  onAddWhere:       (field: string, op: string, value: string) => void;
  onRemoveWhere:    (id: string) => void;
  onAddSummarize:   (alias: string, func: string, field: string, byField: string) => void;
  onRemoveSummarize:(id: string) => void;
  onToggleSelect:   (field: string) => void;
  onRemoveSelect:   (field: string) => void;
  onAddOrderBy:     (field: string, dir: 'asc' | 'desc') => void;
  onRemoveOrderBy:  (id: string) => void;
  onSetLimit:       (n: number) => void;
};

// ---------------------------------------------------------------------------
// Small shared components
// ---------------------------------------------------------------------------

const Trash: React.FC<{ onClick: () => void; title?: string }> = ({ onClick, title = 'Remove' }) => (
  <button
    onClick={onClick}
    title={title}
    className="text-text-muted hover:text-red-400 transition-colors flex-shrink-0 p-0.5"
  >
    <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  </button>
);

const selectCls =
  'px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono text-xs';
const inputCls =
  'px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono text-xs';
const addBtnCls =
  'px-3 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs active:scale-95 transition-all';

// Row pill displayed under each section
const ClausePill: React.FC<{ label: string; onRemove: () => void }> = ({ label, onRemove }) => (
  <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono text-xs w-full">
    <span className="flex-1 text-text-normal truncate">{label}</span>
    <Trash onClick={onRemove} />
  </div>
);

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const QueryBuilderAdvanced: React.FC<QueryBuilderAdvancedProps> = ({
  state,
  onAddWhere, onRemoveWhere,
  onAddSummarize, onRemoveSummarize,
  onToggleSelect, onRemoveSelect,
  onAddOrderBy, onRemoveOrderBy,
  onSetLimit,
}) => {
  // ── where input state ──────────────────────────────────────────────────
  const [wField, setWField] = useState<string>('rating');
  const [wOp,    setWOp]    = useState<string>('==');
  const [wValue, setWValue] = useState<string>('');

  // ── summarize input state ──────────────────────────────────────────────
  const [summAlias,   setSummAlias]   = useState<string>('n');
  const [summFunc,    setSummFunc]    = useState<string>('count');
  const [summField,   setSummField]   = useState<string>('rating');
  const [summByField, setSummByField] = useState<string>('');

  // ── order by input state ───────────────────────────────────────────────
  const [orderField, setOrderField] = useState<string>('');
  const [orderDir,   setOrderDir]   = useState<'asc' | 'desc'>('desc');

  // ── handlers ──────────────────────────────────────────────────────────
  const handleAddWhere = () => {
    if (wOp !== '== null' && wOp !== '!= null' && !wValue.trim()) return;
    onAddWhere(wField, wOp, wValue.trim());
    setWValue('');
  };

  const handleAddSummarize = () => {
    if (summFunc !== 'count' && !summField) return;
    onAddSummarize(summAlias.trim() || 'n', summFunc, summField, summByField);
  };

  const handleAddOrderBy = () => {
    if (!orderField.trim()) return;
    onAddOrderBy(orderField.trim(), orderDir);
    setOrderField('');
  };

  // ── human-readable label for a where row ──────────────────────────────
  const whereLabel = (row: WhereRow) => {
    if (row.op === '== null' || row.op === '!= null') return `${row.field} ${row.op}`;
    if (!row.value) return `${row.field} ${row.op}`;
    return `${row.field} ${row.op} ${row.value}`;
  };

  const summarizeLabel = (row: SummarizeRow) => {
    const inner = row.func === 'count' ? '' : row.field;
    let s = `${row.alias} = ${row.func}(${inner})`;
    if (row.byField) s += ` by ${row.byField}`;
    return s;
  };

  const orderByLabel = (row: OrderByRow) => `${row.field} ${row.dir}`;

  // ── render ─────────────────────────────────────────────────────────────
  return (
    <div className="px-4 pt-2 space-y-5 text-xs pb-6">
      <p className="text-text-muted leading-relaxed">
        Add clauses in order:{' '}
        <span className="font-mono text-text-normal">
          where → summarize → select → order by → limit
        </span>.
        Each clause stacks below its section and appears instantly in Textual Query.
        The builder only generates{' '}
        <span className="font-mono text-text-normal">messages</span>-stream queries.
      </p>

      {/* ── WHERE ─────────────────────────────────────────────────────── */}
      <div className="p-4 rounded bg-scheme-shade_2 element-border space-y-3">
        <div className="font-semibold text-text-normal">Filter rows (where)</div>

        {/* Existing where rows */}
        {state.whereRows.length > 0 && (
          <div className="space-y-1">
            {state.whereRows.map(row => (
              <ClausePill
                key={row.id}
                label={`where ${whereLabel(row)}`}
                onRemove={() => onRemoveWhere(row.id)}
              />
            ))}
          </div>
        )}

        {/* Add where controls */}
        <div className="flex flex-wrap gap-2 items-center">
          <select value={wField} onChange={e => setWField(e.target.value)} className={selectCls}>
            {FIELDS.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
          <select value={wOp} onChange={e => setWOp(e.target.value)} className={selectCls}>
            {OPERATORS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
          {wOp !== '== null' && wOp !== '!= null' && (
            <input
              type="text"
              placeholder={wOp === 'in' ? 'value1, value2, …' : 'value'}
              value={wValue}
              onChange={e => setWValue(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); handleAddWhere(); } }}
              className={`${inputCls} w-40`}
            />
          )}
          <button onClick={handleAddWhere} className={addBtnCls}>Add where</button>
        </div>

        {wOp === 'in' && (
          <p className="text-text-muted/80 leading-relaxed">
            Comma-separated values, e.g.{' '}
            <span className="font-mono text-text-normal">1, 2, 3</span> or{' '}
            <span className="font-mono text-text-normal">claude-3, gpt-4o</span>.
          </p>
        )}
        {(wOp === '== null' || wOp === '!= null') && (
          <p className="text-text-muted/80 leading-relaxed">
            No value needed — matches rows where the field is (or isn't) empty.
          </p>
        )}
      </div>

      {/* ── SUMMARIZE ─────────────────────────────────────────────────── */}
      <div className="p-4 rounded bg-scheme-shade_2 element-border space-y-3">
        <div className="font-semibold text-text-normal">Group &amp; aggregate (summarize)</div>

        {state.summarizeRows.length > 0 && (
          <div className="space-y-1">
            {state.summarizeRows.map(row => (
              <ClausePill
                key={row.id}
                label={`summarize ${summarizeLabel(row)}`}
                onRemove={() => onRemoveSummarize(row.id)}
              />
            ))}
          </div>
        )}

        <div className="flex flex-wrap gap-2 items-center">
          <input
            value={summAlias}
            onChange={e => setSummAlias(e.target.value)}
            placeholder="alias"
            className={`${inputCls} w-20`}
          />
          <span className="font-mono">=</span>
          <select value={summFunc} onChange={e => setSummFunc(e.target.value)} className={selectCls}>
            {AGG_FUNCTIONS.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
          {summFunc !== 'count' && (
            <select value={summField} onChange={e => setSummField(e.target.value)} className={selectCls}>
              {SELECTABLE_FIELDS.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          )}
          <span className="font-mono">by</span>
          <select value={summByField} onChange={e => setSummByField(e.target.value)} className={selectCls}>
            <option value="">— none —</option>
            {SELECTABLE_FIELDS.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
          <button onClick={handleAddSummarize} className={addBtnCls}>Add summarize</button>
        </div>
      </div>

      {/* ── SELECT ────────────────────────────────────────────────────── */}
      <div className="p-4 rounded bg-scheme-shade_2 element-border space-y-3">
        <div className="font-semibold text-text-normal">Choose columns (select)</div>

        {/* Selected field tags */}
        {state.selectFields.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {state.selectFields.map(f => (
              <span
                key={f}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-600/30 element-border font-mono text-xs"
              >
                {f}
                <Trash onClick={() => onRemoveSelect(f)} title={`Remove ${f}`} />
              </span>
            ))}
          </div>
        )}

        {/* Checkbox grid for toggling */}
        <div className="flex flex-wrap gap-2">
          {SELECTABLE_FIELDS.map(f => {
            const checked = state.selectFields.includes(f);
            return (
              <label
                key={f}
                className={`flex items-center gap-1.5 font-mono cursor-pointer px-2 py-1 rounded element-border text-xs ${
                  checked ? 'bg-blue-600/30' : 'bg-scheme-shade_3 hover:bg-scheme-shade_5'
                }`}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => onToggleSelect(f)}
                  className="w-3 h-3"
                />
                {f}
              </label>
            );
          })}
        </div>
        {state.selectFields.length === 0 && (
          <p className="text-text-muted/80 leading-relaxed">
            Check fields above to add a <span className="font-mono text-text-normal">select</span> clause.
          </p>
        )}
      </div>

      {/* ── ORDER BY ──────────────────────────────────────────────────── */}
      <div className="p-4 rounded bg-scheme-shade_2 element-border space-y-3">
        <div className="font-semibold text-text-normal">Sort (order by)</div>

        {state.orderByRows.length > 0 && (
          <div className="space-y-1">
            {state.orderByRows.map(row => (
              <ClausePill
                key={row.id}
                label={`order by ${orderByLabel(row)}`}
                onRemove={() => onRemoveOrderBy(row.id)}
              />
            ))}
          </div>
        )}

        <div className="flex flex-wrap gap-2 items-center">
          <input
            value={orderField}
            onChange={e => setOrderField(e.target.value)}
            placeholder="field or alias"
            className={`${inputCls} w-36`}
          />
          <select
            value={orderDir}
            onChange={e => setOrderDir(e.target.value as 'asc' | 'desc')}
            className={selectCls}
          >
            <option value="asc">asc</option>
            <option value="desc">desc</option>
          </select>
          <button onClick={handleAddOrderBy} className={addBtnCls}>Add order by</button>
        </div>
      </div>

      {/* ── LIMIT ─────────────────────────────────────────────────────── */}
      <div className="p-4 rounded bg-scheme-shade_2 element-border">
        <div className="font-semibold text-text-normal mb-2">Row cap (limit)</div>
        <div className="flex flex-wrap gap-2 items-center">
          <input
            type="number"
            min={1}
            value={state.limitValue}
            onChange={e => {
              const n = parseInt(e.target.value, 10);
              if (!Number.isNaN(n) && n > 0) onSetLimit(n);
            }}
            className={`${inputCls} w-24`}
          />
          <span className="text-text-muted">rows</span>
        </div>
      </div>
    </div>
  );
};

// Default export for legacy imports
const QueryBuilder = QueryBuilderAdvanced;
export default QueryBuilder;