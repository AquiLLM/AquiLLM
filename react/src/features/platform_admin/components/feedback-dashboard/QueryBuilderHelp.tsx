import React, { useEffect } from 'react';

type Props = {
  onClose: () => void;
};

const CLAUSES = [
  { name: 'messages', required: true, summary: 'Starts every query. Returns rows from the messages table.' },
  { name: 'where', required: false, summary: 'Filter rows by a condition.' },
  { name: 'summarize', required: false, summary: 'Group rows and compute aggregates (average, count, etc.).' },
  { name: 'select', required: false, summary: 'Pick which columns to return.' },
  { name: 'order by', required: false, summary: 'Sort the results by a field.' },
  { name: 'limit', required: false, summary: 'Cap the number of rows returned.' },
];

const EXAMPLES = [
  {
    query: 'messages | limit 20',
    desc: 'The simplest query, returns the first 20 messages with no filtering or sorting.',
  },
  {
    query: 'messages\n| where rating < 3\n| select rating, model, feedback_text',
    desc: 'Low-rated messages showing only the columns you care about.',
  },
  {
    query: 'messages\n| where rating != null\n| summarize avg_r = avg(rating), n = count() by model\n| order by avg_r desc',
    desc: 'Average rating per model, sorted best to worst. Renders as a bar chart.',
  },
];

const QueryBuilderHelp: React.FC<Props> = ({ onClose }) => {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <div
        className="bg-scheme-shade_2 element-border rounded-lg w-full max-w-2xl overflow-y-auto shadow-2xl my-8"
        style={{ maxHeight: '600px' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 flex justify-between items-center px-5 py-3 bg-scheme-shade_2 border-b border-scheme-contrast/30">
          <h2 className="text-base font-bold">How the query language works</h2>
          <button
            onClick={onClose}
            title="Close"
            className="w-7 h-7 flex items-center justify-center rounded hover:bg-scheme-shade_5 text-text-muted hover:text-text-normal"
          >
            ×
          </button>
        </div>

        <div className="px-5 py-4 space-y-5 text-xs">
          <p className="text-text-muted leading-relaxed">
            A query is a <strong className="text-text-normal">pipeline</strong>. You start with{' '}
            <span className="font-mono text-text-normal">messages</span> and pass the results through
            clauses separated by <span className="font-mono text-text-normal">|</span>. Only{' '}
            <span className="font-mono text-text-normal">messages</span> is required, so skip any
            clause you don't need.
          </p>
          <p className="text-text-muted leading-relaxed">
            You can build a query by <strong className="text-text-normal">typing it directly</strong>{' '}
            into the query box or by using the <strong className="text-text-normal">visual builder</strong>{' '}
            (dropdowns and buttons above the box). Both produce the same thing, the builder just fills
            the box for you so you can learn the syntax as you go.
          </p>

          <div>
            <p className="font-semibold text-text-normal mb-2 text-sm">Clauses (must be in this order)</p>
            <div className="space-y-1">
              {CLAUSES.map((c) => (
                <div key={c.name} className="flex items-baseline gap-3 py-1 border-b border-scheme-contrast/10 last:border-b-0">
                  <span className="font-mono font-semibold text-text-normal w-24 shrink-0">{c.name}</span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wide shrink-0 ${
                      c.required ? 'bg-red-600/30 text-red-300' : 'bg-scheme-shade_5 text-text-muted'
                    }`}
                  >
                    {c.required ? 'required' : 'optional'}
                  </span>
                  <span className="text-text-muted leading-relaxed">{c.summary}</span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <p className="font-semibold text-text-normal mb-2 text-sm">Examples, from simple to complex</p>
            <div className="space-y-3">
              {EXAMPLES.map((e, i) => (
                <div key={i} className="p-2.5 rounded bg-scheme-shade_3 element-border">
                  <pre className="font-mono text-text-normal leading-relaxed whitespace-pre-wrap">
                    {e.query}
                  </pre>
                  <p className="text-text-muted mt-1.5 leading-relaxed">{e.desc}</p>
                </div>
              ))}
            </div>
          </div>

          <p className="text-text-muted leading-relaxed">
            For the full list of fields, operators, and aggregation functions, open the{' '}
            <strong className="text-text-normal">Syntax quick reference</strong> panel below the query box.
            The language is modeled on{' '}
            <a
              href="https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/"
              target="_blank"
              rel="noreferrer"
              className="text-blue-400 hover:underline"
            >
              KQL (Kusto Query Language)
            </a>
            , so online KQL examples mostly translate.
          </p>
        </div>
      </div>
    </div>
  );
};

export default QueryBuilderHelp;
