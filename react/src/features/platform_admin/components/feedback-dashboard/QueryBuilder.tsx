import React, { useState } from 'react';

const FIELDS = [
  'rating', 'feedback_text', 'feedback_submitted_at', 'model', 'role', 'content',
  'sequence_number', 'created_at', 'message_uuid', 'tool_call_name', 'user_id',
  'conversation_id', 'conversation_tool',
] as const;

const NUMERIC_FIELDS = new Set(['rating', 'sequence_number', 'user_id', 'conversation_id']);

const SELECTABLE_FIELDS = FIELDS.filter((f) => f !== 'conversation_tool');

const OPERATORS = [
  { value: '==', label: '== (equals)' },
  { value: '!=', label: '!= (not equals)' },
  { value: '<', label: '< (less than)' },
  { value: '>', label: '> (greater than)' },
  { value: '<=', label: '<= (less or equal)' },
  { value: '>=', label: '>= (greater or equal)' },
  { value: 'contains', label: 'contains' },
  { value: 'startswith', label: 'startswith' },
  { value: 'in', label: 'in [list]' },
  { value: '== null', label: 'is null' },
  { value: '!= null', label: 'is not null' },
] as const;

const AGG_FUNCTIONS = ['count', 'avg', 'min', 'max', 'sum', 'median'] as const;

type Props = {
  value: string;
  onChange: (v: string) => void;
};

function quoteIfString(field: string, raw: string): string {
  const trimmed = raw.trim();
  if (NUMERIC_FIELDS.has(field)) return trimmed;
  return `"${trimmed.replace(/"/g, '\\"')}"`;
}

function formatValueList(field: string, raw: string): string {
  const parts = raw.split(',').map((s) => s.trim()).filter(Boolean);
  const items = parts.map((p) => {
    if (NUMERIC_FIELDS.has(field)) return p;
    return `"${p.replace(/"/g, '\\"')}"`;
  });
  return `[${items.join(', ')}]`;
}

function appendStage(current: string, stage: string): string {
  const trimmed = current.trim();
  if (!trimmed) return `messages\n| ${stage}`;
  return `${trimmed}\n| ${stage}`;
}

const QueryBuilder: React.FC<Props> = ({ value, onChange }) => {
  // WHERE
  const [wField, setWField] = useState<string>('rating');
  const [wOp, setWOp] = useState<string>('==');
  const [wValue, setWValue] = useState<string>('');

  // SELECT
  const [selectedFields, setSelectedFields] = useState<string[]>([]);

  // SUMMARIZE
  const [summAlias, setSummAlias] = useState<string>('n');
  const [summFunc, setSummFunc] = useState<string>('count');
  const [summField, setSummField] = useState<string>('rating');
  const [summByField, setSummByField] = useState<string>('');

  // ORDER BY
  const [orderField, setOrderField] = useState<string>('');
  const [orderDir, setOrderDir] = useState<string>('desc');

  // LIMIT
  const [limitValue, setLimitValue] = useState<string>('20');

  const addWhere = () => {
    if (wOp === '== null' || wOp === '!= null') {
      onChange(appendStage(value, `where ${wField} ${wOp}`));
      return;
    }
    if (wOp === 'in') {
      if (!wValue.trim()) return;
      onChange(appendStage(value, `where ${wField} in ${formatValueList(wField, wValue)}`));
      setWValue('');
      return;
    }
    if (!wValue.trim()) return;
    onChange(appendStage(value, `where ${wField} ${wOp} ${quoteIfString(wField, wValue)}`));
    setWValue('');
  };

  const toggleSelectField = (field: string) => {
    setSelectedFields((prev) =>
      prev.includes(field) ? prev.filter((f) => f !== field) : [...prev, field],
    );
  };

  const addSelect = () => {
    if (selectedFields.length === 0) return;
    onChange(appendStage(value, `select ${selectedFields.join(', ')}`));
    setSelectedFields([]);
  };

  const addSummarize = () => {
    const alias = summAlias.trim() || 'n';
    if (summFunc !== 'count' && !summField) return;
    const inner = summFunc === 'count' ? '' : summField;
    let stage = `summarize ${alias} = ${summFunc}(${inner})`;
    if (summByField) stage += ` by ${summByField}`;
    onChange(appendStage(value, stage));
  };

  const addOrderBy = () => {
    const f = orderField.trim();
    if (!f) return;
    onChange(appendStage(value, `order by ${f} ${orderDir}`));
  };

  const addLimit = () => {
    const n = parseInt(limitValue, 10);
    if (!n || n <= 0) return;
    onChange(appendStage(value, `limit ${n}`));
  };

  return (
    <details className="mb-5 rounded-lg bg-scheme-shade_3 element-border text-sm">
      <summary className="px-4 py-2.5 cursor-pointer font-semibold select-none hover:text-blue-500 transition-colors">
        Visual query builder
      </summary>
      <div
        className="px-4 pt-2 space-y-5 text-xs"
        style={{ paddingBottom: '48px' }}
      >
        <p className="text-text-muted leading-relaxed">
          Click the buttons below to build a query. Each click appends a clause to the raw query box,
          and you can still edit the raw text by hand. Clauses should be added in this order:{' '}
          <span className="font-mono text-text-normal">where → summarize → select → order by → limit</span>.
        </p>
        <p className="text-text-muted/80 leading-relaxed">
          The builder only generates <span className="font-mono text-text-normal">messages</span>-stream
          queries. To query at the conversation level, type{' '}
          <span className="font-mono text-text-normal">conversations | …</span> directly in the editor —
          see the syntax reference for the full conversations-stream field list.
        </p>

        <div className="p-4 rounded bg-scheme-shade_2 element-border">
          <div className="font-semibold text-text-normal mb-2">Filter rows (where)</div>
          <div className="flex flex-wrap gap-2 items-center">
            <select
              value={wField}
              onChange={(e) => setWField(e.target.value)}
              className="px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono"
            >
              {FIELDS.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
            <select
              value={wOp}
              onChange={(e) => setWOp(e.target.value)}
              className="px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono"
            >
              {OPERATORS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
            {wOp !== '== null' && wOp !== '!= null' && (
              <input
                type="text"
                placeholder={wOp === 'in' ? 'value1, value2, …' : 'value'}
                value={wValue}
                onChange={(e) => setWValue(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addWhere(); } }}
                className="px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono w-48"
              />
            )}
            <button
              onClick={addWhere}
              className="px-3 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white font-semibold active:scale-95 transition-all"
            >
              Add where
            </button>
          </div>
          {wOp === 'in' ? (
            <p className="text-text-muted/80 mt-2 leading-relaxed">
              Type the list of allowed values separated by commas, for example{' '}
              <span className="font-mono text-text-normal">1, 2, 3</span> or{' '}
              <span className="font-mono text-text-normal">claude-3, gpt-4o</span>.
            </p>
          ) : wOp === '== null' || wOp === '!= null' ? (
            <p className="text-text-muted/80 mt-2 leading-relaxed">
              No value needed. Matches rows where the field is (or isn't) empty.
            </p>
          ) : null}
        </div>

        <div className="p-4 rounded bg-scheme-shade_2 element-border">
          <div className="font-semibold text-text-normal mb-2">Choose columns (select)</div>
          <div className="flex flex-wrap gap-3">
            {SELECTABLE_FIELDS.map((f) => {
              const checked = selectedFields.includes(f);
              return (
                <label
                  key={f}
                  className={`flex items-center gap-2 font-mono cursor-pointer px-3 py-1.5 rounded element-border ${
                    checked ? 'bg-blue-600/30' : 'bg-scheme-shade_3 hover:bg-scheme-shade_5'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleSelectField(f)}
                  />
                  {f}
                </label>
              );
            })}
          </div>
          <button
            onClick={addSelect}
            className="mt-4 px-3 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white font-semibold active:scale-95 transition-all"
          >
            Add select
          </button>
        </div>

        <div className="p-4 rounded bg-scheme-shade_2 element-border">
          <div className="font-semibold text-text-normal mb-2">Group &amp; aggregate (summarize)</div>
          <div className="flex flex-wrap gap-2 items-center">
            <input
              value={summAlias}
              onChange={(e) => setSummAlias(e.target.value)}
              placeholder="name"
              className="px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono w-24"
            />
            <span className="font-mono">=</span>
            <select
              value={summFunc}
              onChange={(e) => setSummFunc(e.target.value)}
              className="px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono"
            >
              {AGG_FUNCTIONS.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
            {summFunc !== 'count' && (
              <select
                value={summField}
                onChange={(e) => setSummField(e.target.value)}
                className="px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono"
              >
                {SELECTABLE_FIELDS.map((f) => <option key={f} value={f}>{f}</option>)}
              </select>
            )}
            <span className="font-mono">by</span>
            <select
              value={summByField}
              onChange={(e) => setSummByField(e.target.value)}
              className="px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono"
            >
              <option value="">— none —</option>
              {SELECTABLE_FIELDS.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
            <button
              onClick={addSummarize}
              className="px-3 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white font-semibold active:scale-95 transition-all"
            >
              Add summarize
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="p-4 rounded bg-scheme-shade_2 element-border">
            <div className="font-semibold text-text-normal mb-2">Sort (order by)</div>
            <div className="flex flex-wrap gap-2 items-center">
              <input
                value={orderField}
                onChange={(e) => setOrderField(e.target.value)}
                placeholder="field or alias"
                className="px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono w-40"
              />
              <select
                value={orderDir}
                onChange={(e) => setOrderDir(e.target.value)}
                className="px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono"
              >
                <option value="asc">asc</option>
                <option value="desc">desc</option>
              </select>
              <button
                onClick={addOrderBy}
                className="px-3 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white font-semibold active:scale-95 transition-all"
              >
                Add order by
              </button>
            </div>
          </div>
          <div className="p-4 rounded bg-scheme-shade_2 element-border">
            <div className="font-semibold text-text-normal mb-2">Row cap (limit)</div>
            <div className="flex flex-wrap gap-2 items-center">
              <input
                type="number"
                min={1}
                value={limitValue}
                onChange={(e) => setLimitValue(e.target.value)}
                className="px-2 py-1 rounded bg-scheme-shade_3 element-border font-mono w-24"
              />
              <button
                onClick={addLimit}
                className="px-3 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white font-semibold active:scale-95 transition-all"
              >
                Add limit
              </button>
            </div>
          </div>
        </div>
      </div>
    </details>
  );
};

export default QueryBuilder;
