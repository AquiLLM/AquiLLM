import React, { useState } from 'react';

interface PRQLPanelProps {
  currentPrql?: string;
  loading?: boolean;
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
    title: 'Most recent feedback',
    description: 'The 25 most recent feedback rows, newest first.',
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
    title: 'Count by rating',
    description: 'How many rows exist for each rating value.',
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
    title: 'Average rating per user',
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
    title: 'Only rows with written feedback',
    description: 'Rows where the user left a written comment.',
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
    title: 'Low rated with text',
    description: 'Rating 1 or 2 where the user also wrote a comment.',
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
    title: 'Count by model',
    description: 'How many feedback rows per model, most common first.',
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
];

const PRQL_REFERENCE = [
  { keyword: 'from feedback', note: 'Required first line. The feedback dashboard exposes this source.' },
  { keyword: 'filter col == value', note: 'Equality filter. Use == for comparison.' },
  { keyword: 'filter col != null', note: 'Exclude null values.' },
  { keyword: 'filter col >= value', note: 'Numeric or date comparison.' },
  { keyword: 'filter a == x && b == y', note: 'Combine conditions with && or ||.' },
  { keyword: 'sort col', note: 'Sort ascending.' },
  { keyword: 'sort {-col}', note: 'Sort descending.' },
  { keyword: 'sort {col1, -col2}', note: 'Sort by multiple columns.' },
  { keyword: 'take N', note: 'Return the first N rows.' },
  { keyword: 'select {c1, c2}', note: 'Choose fields to display.' },
  { keyword: 'group col (aggregate {})', note: 'Group rows and calculate aggregate values.' },
  { keyword: 'count id', note: 'Count rows inside an aggregate block.' },
  { keyword: 'average col', note: 'Mean of a numeric column.' },
  { keyword: 'min col / max col', note: 'Minimum or maximum value.' },
];

const AVAILABLE_COLUMNS = [
  { name: 'id', type: 'int' },
  { name: 'message_uuid', type: 'uuid' },
  { name: 'conversation_id', type: 'int' },
  { name: 'conversation_name', type: 'text|null' },
  { name: 'user_id', type: 'int' },
  { name: 'username', type: 'text' },
  { name: 'rating', type: 'int|null' },
  { name: 'feedback_text', type: 'text|null' },
  { name: 'feedback_submitted_at', type: 'timestamp|null' },
  { name: 'created_at', type: 'timestamp' },
  { name: 'effective_date', type: 'timestamp' },
  { name: 'role', type: 'text' },
  { name: 'content_snippet', type: 'text' },
  { name: 'model', type: 'text|null' },
  { name: 'tool_call_name', type: 'text|null' },
  { name: 'usage', type: 'int' },
  { name: 'has_feedback_text', type: 'bool' },
];

const Code: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <code className="font-mono text-xs px-1 py-0.5 bg-scheme-shade_5 border border-border-low_contrast rounded text-accent">
    {children}
  </code>
);

const PRQLPanel: React.FC<PRQLPanelProps> = ({
  currentPrql = DEFAULT_QUERY,
  loading = false,
}) => {
  const [showGuide, setShowGuide] = useState(false);
  const [guideTab, setGuideTab] = useState<'tutorial' | 'reference' | 'columns' | 'examples'>('tutorial');

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-scheme-shade_3 border border-border-mid_contrast rounded-[12px] overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-border-mid_contrast bg-scheme-shade_4">
          <div>
            <h2 className="text-sm font-semibold text-text-normal">Current PRQL</h2>
            <p className="text-xs text-text-low_contrast mt-0.5">
              Display-only query text generated from the current dashboard filters.
            </p>
          </div>
          <span className="text-xs text-text-low_contrast">
            {loading ? 'updating…' : 'ready'}
          </span>
        </div>

        <pre className="m-0 px-4 py-3 text-xs font-mono text-accent bg-scheme-shade_3 overflow-x-auto whitespace-pre leading-6 max-h-[260px]">
          {currentPrql || 'from feedback'}
        </pre>
      </div>

      <div className="border border-border-mid_contrast rounded-[12px] overflow-hidden">
        <button
          onClick={() => setShowGuide(value => !value)}
          className="w-full flex items-center justify-between px-4 py-3 bg-scheme-shade_4 hover:bg-scheme-shade_5 transition-colors text-left"
        >
          <span className="text-sm font-medium text-text-normal">PRQL Guide</span>
          <span className="text-text-low_contrast text-sm">{showGuide ? '▾' : '▸'}</span>
        </button>

        {showGuide && (
          <div className="bg-scheme-shade_3">
            <div className="flex border-b border-border-mid_contrast overflow-x-auto">
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
              {guideTab === 'tutorial' && (
                <div className="flex flex-col gap-4 text-sm text-text-less_contrast leading-relaxed">
                  <p>
                    PRQL is a pipeline-style query language. Each line transforms the previous result.
                    Dashboard PRQL starts with <Code>from feedback</Code>, then adds filter, sort,
                    take, select, group, or aggregate steps.
                  </p>

                  <pre className="font-mono text-xs bg-scheme-shade_4 border border-border-low_contrast rounded-[8px] p-3 text-accent leading-6 overflow-x-auto">{`from feedback
filter rating >= 4
sort {-effective_date}
take 10
select {
  username,
  rating,
  feedback_text,
}`}</pre>

                  <div>
                    <p className="font-semibold text-text-normal mb-1">Important syntax rules</p>
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
                            ['sort [-col]', 'sort {-col}', 'Use braces for descending sort.'],
                            ['filter x = 5', 'filter x == 5', 'Use == for comparison.'],
                            ['filter x = NULL', 'filter x == null', 'Use lowercase null.'],
                            ['from other_table', 'from feedback', 'The dashboard exposes the feedback source.'],
                          ].map(([wrong, right, why], index) => (
                            <tr key={index} className="border-t border-border-low_contrast">
                              <td className="px-3 py-2 font-mono text-red">{wrong}</td>
                              <td className="px-3 py-2 font-mono text-green">{right}</td>
                              <td className="px-3 py-2 text-text-less_contrast">{why}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

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
                      {PRQL_REFERENCE.map((ref, index) => (
                        <tr key={index} className="border-t border-border-low_contrast hover:bg-scheme-shade_4">
                          <td className="px-3 py-2 font-mono text-accent text-xs whitespace-nowrap">{ref.keyword}</td>
                          <td className="px-3 py-2 text-text-less_contrast text-xs">{ref.note}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

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
                      {AVAILABLE_COLUMNS.map((col, index) => (
                        <tr key={index} className="border-t border-border-low_contrast hover:bg-scheme-shade_4">
                          <td className="px-3 py-2 font-mono text-accent text-xs">{col.name}</td>
                          <td className="px-3 py-2 font-mono text-text-less_contrast text-xs">{col.type}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {guideTab === 'examples' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {GUIDE_EXAMPLES.map((example, index) => (
                    <div key={index} className="border border-border-mid_contrast rounded-[10px] overflow-hidden">
                      <div className="px-3 py-2 bg-scheme-shade_4">
                        <div className="text-xs font-medium text-text-normal">{example.title}</div>
                        <div className="text-xs text-text-low_contrast">{example.description}</div>
                      </div>
                      <pre className="px-3 py-2 text-xs font-mono text-accent bg-scheme-shade_3 overflow-x-auto leading-5 max-h-[160px]">
                        {example.prql}
                      </pre>
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
