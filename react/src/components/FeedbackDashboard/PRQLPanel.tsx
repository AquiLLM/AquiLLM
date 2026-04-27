import React, { useState, useRef } from 'react';
import { getCsrfCookie } from '../../main';

interface PRQLResult {
  columns: string[];
  rows: Array<Array<string | null>>;
  row_count: number;
  truncated: boolean;
  sql: string;
}

interface PRQLError {
  error: string;
  type: 'compilation' | 'execution' | 'permission' | 'parse';
}

interface PRQLPanelProps {
  apiPrql: string;
}

const DEFAULT_QUERY = `from feedback
sort {-effective_date}
take 25
select {
  username,
  rating,
  feedback_text,
  model,
  effective_date,
}`;

const GUIDE_EXAMPLES: Array<{ title: string; description: string; prql: string }> = [
  {
    title: "Most recent feedback",
    description: "The 25 most recent feedback rows, newest first.",
    prql: `from feedback
sort {-effective_date}
take 25
select {
  username,
  rating,
  feedback_text,
  model,
  effective_date,
}`,
  },
  {
    title: "Count by rating",
    description: "How many rows exist for each rating value.",
    prql: `from feedback
filter rating != null
group rating (
  aggregate {
    count = count id,
  }
)
sort rating`,
  },
  {
    title: "Average rating per user",
    description: "Each user's average rating, highest first.",
    prql: `from feedback
filter rating != null
group username (
  aggregate {
    avg_rating = average rating,
    submissions = count id,
  }
)
sort {-avg_rating}`,
  },
  {
    title: "Only rows with written feedback",
    description: "Rows where the user left a written comment.",
    prql: `from feedback
filter has_feedback_text == true
sort {-effective_date}
take 50
select {
  username,
  rating,
  feedback_text,
  conversation_name,
  effective_date,
}`,
  },
  {
    title: "Low rated with text",
    description: "Rating 1 or 2 where the user also wrote a comment.",
    prql: `from feedback
filter rating <= 2
filter has_feedback_text == true
sort {-effective_date}
select {
  username,
  rating,
  feedback_text,
  model,
  effective_date,
}`,
  },
  {
    title: "Count by model",
    description: "How many feedback rows per model, most common first.",
    prql: `from feedback
filter model != null
group model (
  aggregate {
    count = count id,
    avg_rating = average rating,
  }
)
sort {-count}`,
  },
  {
    title: "Tool usage in feedback",
    description: "Which tools appear most often in feedback-bearing messages.",
    prql: `from feedback
filter tool_call_name != null
group tool_call_name (
  aggregate {
    count = count id,
  }
)
sort {-count}`,
  },
  {
    title: "High volume users",
    description: "Users who have submitted the most feedback.",
    prql: `from feedback
group username (
  aggregate {
    total = count id,
    avg_rating = average rating,
  }
)
sort {-total}`,
  },
  {
    title: "Feedback with no rating",
    description: "Text-only feedback with no numeric score.",
    prql: `from feedback
filter rating == null
filter has_feedback_text == true
sort {-effective_date}
take 50
select {
  username,
  feedback_text,
  model,
  effective_date,
}`,
  },
  {
    title: "See all columns",
    description: "Every available column on a single row.",
    prql: `from feedback
take 1`,
  },
];

const PRQL_REFERENCE = [
  { keyword: "from feedback",             note: "Required first line — the only available source" },
  { keyword: "filter col == value",       note: "Equality filter. Use == not =" },
  { keyword: "filter col != null",        note: "Exclude null values" },
  { keyword: "filter col >= value",       note: "Comparisons: >= <= > < == !=" },
  { keyword: "filter a == x && b == y",   note: "Combine conditions with && or ||" },
  { keyword: "sort col",                  note: "Sort ascending" },
  { keyword: "sort {-col}",              note: "Sort descending — wrap in {-}" },
  { keyword: "sort {col1, -col2}",       note: "Sort by multiple columns" },
  { keyword: "take N",                    note: "Return first N rows" },
  { keyword: "select {c1, c2}",          note: "Choose columns to return" },
  { keyword: "group col (aggregate {})", note: "Group and aggregate" },
  { keyword: "count id",                  note: "Count rows (inside aggregate block)" },
  { keyword: "count col",                 note: "Count non-null values" },
  { keyword: "average col",              note: "Mean of a numeric column" },
  { keyword: "sum col",                   note: "Sum of a numeric column" },
  { keyword: "min col / max col",        note: "Min or max value" },
];

const AVAILABLE_COLUMNS = [
  { name: "id",                    type: "int" },
  { name: "message_uuid",          type: "uuid" },
  { name: "conversation_id",       type: "int" },
  { name: "conversation_name",     type: "text|null" },
  { name: "user_id",               type: "int" },
  { name: "username",              type: "text" },
  { name: "rating",                type: "int|null" },
  { name: "feedback_text",         type: "text|null" },
  { name: "feedback_submitted_at", type: "timestamp|null" },
  { name: "created_at",            type: "timestamp" },
  { name: "effective_date",        type: "timestamp" },
  { name: "role",                  type: "text" },
  { name: "content",               type: "text" },
  { name: "content_snippet",       type: "text" },
  { name: "model",                 type: "text|null" },
  { name: "tool_call_name",        type: "text|null" },
  { name: "usage",                 type: "int" },
  { name: "has_feedback_text",     type: "bool" },
];

const Code: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <code className="font-mono text-xs px-1 py-0.5 bg-scheme-shade_5 border border-border-low_contrast rounded text-accent">
    {children}
  </code>
);

const PRQLPanel: React.FC<PRQLPanelProps> = ({ apiPrql }) => {
  const [prql, setPrql] = useState(DEFAULT_QUERY);
  const [result, setResult] = useState<PRQLResult | null>(null);
  const [error, setError] = useState<PRQLError | null>(null);
  const [loading, setLoading] = useState(false);
  const [showGuide, setShowGuide] = useState(false);
  const [showSql, setShowSql] = useState(false);
  const [guideTab, setGuideTab] = useState<'tutorial' | 'reference' | 'columns' | 'examples'>('tutorial');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleRun = async () => {
    if (!prql.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const resp = await fetch(apiPrql, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCsrfCookie(),
        },
        body: JSON.stringify({ prql }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data as PRQLError);
      } else {
        setResult(data as PRQLResult);
      }
    } catch (err) {
      setError({ error: String(err), type: 'execution' });
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (ex: typeof GUIDE_EXAMPLES[0]) => {
    setPrql(ex.prql);
    setResult(null);
    setError(null);
    setTimeout(() => textareaRef.current?.focus(), 50);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleRun();
    }
    if (e.key === 'Tab') {
      e.preventDefault();
      const el = e.currentTarget;
      const start = el.selectionStart;
      const end = el.selectionEnd;
      const newVal = prql.substring(0, start) + '  ' + prql.substring(end);
      setPrql(newVal);
      requestAnimationFrame(() => {
        el.selectionStart = start + 2;
        el.selectionEnd = start + 2;
      });
    }
  };

  return (
    <div className="flex flex-col gap-4">

      {/* header */}
      <div>
        <h2 className="text-base font-semibold text-text-normal">PRQL Query Console</h2>
        <p className="text-xs text-text-low_contrast mt-0.5">
          Query the feedback dataset using PRQL. Must start with <Code>from feedback</Code>.
          {' '}<kbd className="px-1 py-0.5 bg-scheme-shade_5 rounded text-xs font-mono">Ctrl+Enter</kbd> runs the query.
          {' '}<strong>Descending sort:</strong> use <Code>sort {'{-col}'}</Code> not <Code>sort [-col]</Code>.
        </p>
      </div>

      {/* editor */}
      <div className="flex flex-col gap-2">
        <textarea
          ref={textareaRef}
          value={prql}
          onChange={e => setPrql(e.target.value)}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          rows={10}
          className="w-full font-mono text-sm px-4 py-3 rounded-[10px] bg-scheme-shade_4 border border-border-mid_contrast text-text-normal focus:outline-none focus:border-border-high_contrast transition-colors resize-y"
        />
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={handleRun}
            disabled={loading || !prql.trim()}
            className="px-5 py-2 rounded-[8px] text-sm font-semibold bg-accent text-slight_muted_white hover:bg-accent-dark transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {loading ? 'Running…' : 'Run query'}
          </button>
          <button
            onClick={() => { setPrql(DEFAULT_QUERY); setResult(null); setError(null); }}
            className="px-3 py-2 rounded-[8px] text-sm border border-border-mid_contrast bg-scheme-shade_4 hover:bg-scheme-shade_5 text-text-normal transition-colors"
          >
            Reset
          </button>
          <button
            onClick={() => { setPrql(''); setResult(null); setError(null); }}
            className="px-3 py-2 rounded-[8px] text-sm border border-border-mid_contrast bg-scheme-shade_4 hover:bg-scheme-shade_5 text-text-normal transition-colors"
          >
            Clear
          </button>
          {result && (
            <span className="text-xs text-text-low_contrast ml-auto">
              {result.row_count.toLocaleString()} row{result.row_count !== 1 ? 's' : ''}
              {result.truncated ? ' (capped at 500)' : ''}
            </span>
          )}
        </div>
      </div>

      {/* error */}
      {error && (
        <div className="rounded-[10px] border border-red bg-red/10 p-4">
          <div className="text-sm font-semibold text-red mb-1">
            {error.type === 'compilation' ? 'Compilation error' :
             error.type === 'execution'   ? 'Execution error' :
             error.type === 'parse'       ? 'Request error' : 'Error'}
          </div>
          <pre className="text-xs text-red whitespace-pre-wrap font-mono leading-relaxed">
            {error.error}
          </pre>
          {error.type === 'compilation' && (
            <p className="mt-2 text-xs text-text-less_contrast">
              Common causes: using <Code>=</Code> instead of <Code>==</Code>,
              using <Code>sort [-col]</Code> instead of <Code>sort {'{-col}'}</Code>,
              or a missing <Code>from feedback</Code> at the top.
            </p>
          )}
          {error.type === 'execution' && (
            <p className="mt-2 text-xs text-text-less_contrast">
              Most execution errors are caused by <Code>sort [-col]</Code> which PostgreSQL
              rejects for timestamp columns. Use <Code>sort {'{-col}'}</Code> instead.
            </p>
          )}
        </div>
      )}

      {/* results */}
      {result && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-low_contrast">
              {result.row_count.toLocaleString()} row{result.row_count !== 1 ? 's' : ''}
              {result.truncated ? ' — capped at 500. Add take N to be explicit.' : ''}
            </span>
            <button
              onClick={() => setShowSql(s => !s)}
              className="text-xs text-text-low_contrast hover:text-text-normal transition-colors"
            >
              {showSql ? '▾ hide SQL' : '▸ show compiled SQL'}
            </button>
          </div>
          {showSql && (
            <pre className="text-xs font-mono text-text-less_contrast bg-scheme-shade_4 border border-border-low_contrast rounded-[8px] p-3 overflow-x-auto whitespace-pre-wrap">
              {result.sql}
            </pre>
          )}
          {result.columns.length > 0 ? (
            <div className="rounded-[10px] border border-border-mid_contrast overflow-hidden overflow-x-auto">
              <table className="min-w-full border-collapse text-sm">
                <thead className="bg-scheme-shade_4">
                  <tr>
                    {result.columns.map(col => (
                      <th key={col} className="px-3 py-2 text-left text-xs font-semibold text-text-low_contrast uppercase tracking-wide whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, ri) => (
                    <tr key={ri} className="border-t border-border-low_contrast hover:bg-scheme-shade_4 transition-colors">
                      {row.map((cell, ci) => (
                        <td key={ci} className="px-3 py-2 text-text-normal align-top font-mono text-xs max-w-[300px] truncate" title={cell ?? 'null'}>
                          {cell === null
                            ? <span className="text-text-low_contrast italic">null</span>
                            : cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-sm text-text-low_contrast p-4 bg-scheme-shade_3 border border-border-low_contrast rounded-[10px]">
              Query returned no rows.
            </div>
          )}
        </div>
      )}

      {/* guide — below editor and results */}
      <div className="border border-border-mid_contrast rounded-[12px] overflow-hidden">
        <button
          onClick={() => setShowGuide(g => !g)}
          className="w-full flex items-center justify-between px-4 py-3 bg-scheme-shade_4 hover:bg-scheme-shade_5 transition-colors text-left"
        >
          <span className="text-sm font-medium text-text-normal">PRQL Guide</span>
          <span className="text-text-low_contrast text-sm">{showGuide ? '▾' : '▸'}</span>
        </button>

        {showGuide && (
          <div className="bg-scheme-shade_3">
            {/* guide tab bar */}
            <div className="flex border-b border-border-mid_contrast">
              {(['tutorial', 'reference', 'columns', 'examples'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setGuideTab(tab)}
                  className={`px-4 py-2 text-xs font-medium capitalize transition-colors ${
                    guideTab === tab
                      ? 'text-accent border-b-2 border-accent bg-scheme-shade_3'
                      : 'text-text-low_contrast hover:text-text-normal bg-scheme-shade_4'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            <div className="p-4">

              {/* TUTORIAL */}
              {guideTab === 'tutorial' && (
                <div className="flex flex-col gap-4 text-sm text-text-less_contrast leading-relaxed">
                  <div>
                    <p className="mb-2">
                      PRQL is a pipeline query language. Each line transforms the previous result.
                      Every query must start with <Code>from feedback</Code>.
                    </p>
                    <pre className="font-mono text-xs bg-scheme-shade_4 border border-border-low_contrast rounded-[8px] p-3 text-accent leading-6 overflow-x-auto">{`from feedback           -- start with all feedback rows
filter rating >= 4      -- keep only 4 and 5 star rows
sort {-effective_date}  -- newest first  ← use {-col} not [-col]
take 10                 -- first 10 only
select {                -- choose columns
  username,
  rating,
  feedback_text,
}`}</pre>
                  </div>

                  <div>
                    <p className="font-semibold text-text-normal mb-1">Critical syntax rules</p>
                    <div className="overflow-x-auto rounded-[8px] border border-border-low_contrast">
                      <table className="min-w-full text-xs">
                        <thead className="bg-scheme-shade_4">
                          <tr>
                            <th className="px-3 py-2 text-left text-text-low_contrast uppercase font-semibold">Wrong</th>
                            <th className="px-3 py-2 text-left text-text-low_contrast uppercase font-semibold">Correct</th>
                            <th className="px-3 py-2 text-left text-text-low_contrast uppercase font-semibold">Why</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[
                            ["sort [-col]",      "sort {-col}",         "[-col] → ORDER BY -col which PostgreSQL rejects for timestamps"],
                            ["filter x = 5",     "filter x == 5",       "= is assignment, == is comparison"],
                            ["filter x = NULL",  "filter x == null",    "null must be lowercase"],
                            ["from auth_user",   "from feedback",       "only the feedback source is available"],
                          ].map(([w, r, why], i) => (
                            <tr key={i} className="border-t border-border-low_contrast">
                              <td className="px-3 py-2 font-mono text-red">{w}</td>
                              <td className="px-3 py-2 font-mono text-green">{r}</td>
                              <td className="px-3 py-2 text-text-less_contrast">{why}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div>
                    <p className="font-semibold text-text-normal mb-1">Grouping example</p>
                    <pre className="font-mono text-xs bg-scheme-shade_4 border border-border-low_contrast rounded-[8px] p-3 text-accent leading-6 overflow-x-auto">{`from feedback
filter rating != null
group username (
  aggregate {
    total = count id,
    avg_rating = average rating,
  }
)
sort {-avg_rating}`}</pre>
                  </div>
                </div>
              )}

              {/* REFERENCE */}
              {guideTab === 'reference' && (
                <div className="overflow-x-auto rounded-[8px] border border-border-low_contrast">
                  <table className="min-w-full text-sm">
                    <thead className="bg-scheme-shade_4">
                      <tr>
                        <th className="px-3 py-2 text-left text-xs font-semibold text-text-low_contrast uppercase">Syntax</th>
                        <th className="px-3 py-2 text-left text-xs font-semibold text-text-low_contrast uppercase">What it does</th>
                      </tr>
                    </thead>
                    <tbody>
                      {PRQL_REFERENCE.map((ref, i) => (
                        <tr key={i} className="border-t border-border-low_contrast hover:bg-scheme-shade_4">
                          <td className="px-3 py-2 font-mono text-accent text-xs whitespace-nowrap">{ref.keyword}</td>
                          <td className="px-3 py-2 text-text-less_contrast text-xs">{ref.note}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* COLUMNS */}
              {guideTab === 'columns' && (
                <div className="overflow-x-auto rounded-[8px] border border-border-low_contrast">
                  <table className="min-w-full text-sm">
                    <thead className="bg-scheme-shade_4">
                      <tr>
                        <th className="px-3 py-2 text-left text-xs font-semibold text-text-low_contrast uppercase">Column</th>
                        <th className="px-3 py-2 text-left text-xs font-semibold text-text-low_contrast uppercase">Type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {AVAILABLE_COLUMNS.map((col, i) => (
                        <tr key={i} className="border-t border-border-low_contrast hover:bg-scheme-shade_4">
                          <td className="px-3 py-2 font-mono text-accent text-xs">{col.name}</td>
                          <td className="px-3 py-2 font-mono text-text-less_contrast text-xs">{col.type}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* EXAMPLES */}
              {guideTab === 'examples' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {GUIDE_EXAMPLES.map((ex, i) => (
                    <div key={i} className="border border-border-mid_contrast rounded-[10px] overflow-hidden">
                      <div className="px-3 py-2 bg-scheme-shade_4">
                        <div className="text-xs font-medium text-text-normal">{ex.title}</div>
                        <div className="text-xs text-text-low_contrast">{ex.description}</div>
                      </div>
                      <pre className="px-3 py-2 text-xs font-mono text-accent bg-scheme-shade_3 overflow-x-auto leading-5 max-h-[120px]">{ex.prql}</pre>
                      <div className="px-3 py-2 bg-scheme-shade_4 border-t border-border-low_contrast">
                        <button
                          onClick={() => handleExampleClick(ex)}
                          className="text-xs px-2 py-1 rounded-[6px] bg-accent text-slight_muted_white hover:bg-accent-dark transition-colors"
                        >
                          Load →
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

            </div>
          </div>
        )}
      </div>

    </div>
  );
};

export default PRQLPanel;