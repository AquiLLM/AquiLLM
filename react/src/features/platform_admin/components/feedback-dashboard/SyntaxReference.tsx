import React from 'react';

type FieldType = 'text' | 'number' | 'time' | 'special';

type Field = { name: string; type: FieldType; desc: string };

const MESSAGES_FIELDS: Field[] = [
  { name: 'rating', type: 'number', desc: '1–5 star rating' },
  { name: 'feedback_text', type: 'text', desc: 'Written comment' },
  { name: 'feedback_submitted_at', type: 'time', desc: 'When rating/feedback was submitted' },
  { name: 'model', type: 'text', desc: 'AI model used' },
  { name: 'role', type: 'text', desc: 'user / assistant / tool' },
  { name: 'content', type: 'text', desc: 'Message text' },
  { name: 'created_at', type: 'time', desc: 'Message timestamp' },
  { name: 'user_id', type: 'number', desc: 'User account ID' },
  { name: 'conversation_id', type: 'number', desc: 'Conversation ID' },
  { name: 'sequence_number', type: 'number', desc: 'Position in conversation' },
  { name: 'message_uuid', type: 'text', desc: 'Unique message ID' },
  { name: 'tool_call_name', type: 'text', desc: 'Tool used (intermediate messages only)' },
  { name: 'conversation_tool', type: 'special', desc: 'Tools used anywhere in this conversation (see below).' },
];

const CONVERSATIONS_FIELDS: Field[] = [
  { name: 'conversation_id', type: 'number', desc: 'Conversation ID' },
  { name: 'user_id', type: 'number', desc: 'Owner user ID' },
  { name: 'name', type: 'text', desc: 'Auto-generated title' },
  { name: 'created_at', type: 'time', desc: 'When conversation started' },
  { name: 'updated_at', type: 'time', desc: 'Last activity time' },
  { name: 'message_count', type: 'number', desc: 'Total message turns' },
  { name: 'rated_count', type: 'number', desc: 'Number of rated messages' },
  { name: 'avg_rating', type: 'number', desc: 'Mean rating across messages' },
  { name: 'min_rating', type: 'number', desc: 'Lowest rating in conv.' },
  { name: 'max_rating', type: 'number', desc: 'Highest rating in conv.' },
  { name: 'last_rated_at', type: 'time', desc: 'Most recent rating time' },
  { name: 'tools_used', type: 'special', desc: 'Comma-list of distinct tools used (see below).' },
];

const TYPE_STYLES: Record<FieldType, string> = {
  number: 'bg-sky-600/20 text-sky-300',
  text: 'bg-emerald-600/20 text-emerald-300',
  time: 'bg-amber-600/20 text-amber-300',
  special: 'bg-purple-600/20 text-purple-300',
};

const OPERATORS: { syntax: string; desc: string }[] = [
  { syntax: '== / !=', desc: 'Equal / not equal' },
  { syntax: '< > <= >=', desc: 'Numeric comparisons' },
  { syntax: 'startswith "text"', desc: 'Prefix match' },
  { syntax: 'contains "text"', desc: 'Substring match' },
  { syntax: 'in [1, 2, 3]', desc: 'Membership test' },
  { syntax: '== null / != null', desc: 'Missing / present' },
];

const FUNCTIONS: { name: string; desc: string }[] = [
  { name: 'avg(field)', desc: 'Average of values' },
  { name: 'count()', desc: 'Rows in group' },
  { name: 'min(field)', desc: 'Lowest value' },
  { name: 'max(field)', desc: 'Highest value' },
  { name: 'sum(field)', desc: 'Total' },
  { name: 'median(field)', desc: 'Middle value' },
];

const TOOLS = [
  'vector_search',
  'more_context',
  'document_ids',
  'whole_document',
  'search_single_document',
  'sky_subtraction',
  'flat_fielding',
  'detect_point_sources',
];

const SectionTitle: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <h3 className="font-semibold text-text-normal mb-2 text-sm">{children}</h3>
);

const Card: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = '' }) => (
  <div className={`p-3 rounded-lg bg-scheme-shade_2 element-border ${className}`}>{children}</div>
);

const TypeBadge: React.FC<{ type: FieldType }> = ({ type }) => (
  <span className={`text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wide font-semibold whitespace-nowrap ${TYPE_STYLES[type]}`}>
    {type}
  </span>
);

const FieldCells: React.FC<{ field: Field | undefined; border: boolean }> = ({ field, border }) => {
  const b = border ? 'border-b border-scheme-contrast/10' : '';
  return (
    <>
      <td className={`py-1.5 pr-2 align-top font-mono text-text-normal whitespace-nowrap ${b}`}>
        {field?.name}
      </td>
      <td className={`py-1.5 pr-2 align-top ${b}`}>{field && <TypeBadge type={field.type} />}</td>
      <td className={`py-1.5 pr-4 text-text-muted align-top ${b}`}>{field?.desc}</td>
    </>
  );
};

const StreamFieldsTable: React.FC<{ fields: Field[] }> = ({ fields }) => (
  <table className="w-full">
    <tbody>
      {fields.map((f, i) => {
        const isLast = i === fields.length - 1;
        const border = !isLast;
        return (
          <tr key={f.name}>
            <FieldCells field={f} border={border} />
          </tr>
        );
      })}
    </tbody>
  </table>
);

const SyntaxReference: React.FC = () => {
  return (
    <details className="mb-5 rounded-lg bg-scheme-shade_3 element-border text-sm">
      <summary className="px-4 py-2.5 cursor-pointer font-semibold select-none hover:text-blue-500 transition-colors">
        Syntax quick reference
      </summary>
      <div className="px-4 pb-5 pt-3 space-y-4 text-xs">

        <Card>
          <SectionTitle>Streams &amp; their fields</SectionTitle>
          <p className="text-text-muted mb-2 leading-relaxed">
            Every query starts with a <em>stream</em>, which determines what each result row
            represents and which fields are available.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="font-mono text-text-normal mb-1">messages</p>
              <p className="text-text-muted/80 mb-1.5 leading-relaxed">
                One row per message turn (user, assistant, or tool call).
              </p>
              <StreamFieldsTable fields={MESSAGES_FIELDS} />
            </div>
            <div>
              <p className="font-mono text-text-normal mb-1">conversations</p>
              <p className="text-text-muted/80 mb-1.5 leading-relaxed">
                One row per conversation, with derived aggregates across its messages.
              </p>
              <StreamFieldsTable fields={CONVERSATIONS_FIELDS} />
            </div>
          </div>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <SectionTitle>Comparison operators</SectionTitle>
            <table className="w-full">
              <tbody>
                {OPERATORS.map((o) => (
                  <tr key={o.syntax} className="border-b border-scheme-contrast/10 last:border-b-0">
                    <td className="py-1.5 pr-3 align-top font-mono text-text-normal whitespace-nowrap">{o.syntax}</td>
                    <td className="py-1.5 text-text-muted align-top">{o.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card>
            <SectionTitle>Aggregation functions</SectionTitle>
            <table className="w-full">
              <tbody>
                {FUNCTIONS.map((fn) => (
                  <tr key={fn.name} className="border-b border-scheme-contrast/10 last:border-b-0">
                    <td className="py-1.5 pr-3 align-top font-mono text-text-normal whitespace-nowrap">{fn.name}</td>
                    <td className="py-1.5 text-text-muted align-top">{fn.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>

        <Card>
          <SectionTitle>
            Tool use (<code className="font-mono">conversation_tool</code> &amp; <code className="font-mono">tools_used</code>)
          </SectionTitle>
          <p className="text-text-muted leading-relaxed mb-2.5">
            Tool calls (e.g. document search) live on a separate message from the rated response.
            Both streams expose the conversation's tool history so you can ask about it directly:{' '}
            <code className="text-text-normal">conversation_tool</code> on the messages stream
            (per-row) and <code className="text-text-normal">tools_used</code> on the conversations
            stream (per-conversation).
          </p>
          <table className="w-full mb-3">
            <tbody>
              <tr>
                <td className="py-1 pr-3 font-mono text-text-normal whitespace-nowrap">where conversation_tool == "vector_search"</td>
                <td className="py-1 text-text-muted">messages from convs that used it</td>
              </tr>
              <tr>
                <td className="py-1 pr-3 font-mono text-text-normal whitespace-nowrap">select rating, conversation_tool</td>
                <td className="py-1 text-text-muted">tools per row (comma-list, null when none)</td>
              </tr>
              <tr>
                <td className="py-1 pr-3 font-mono text-text-normal whitespace-nowrap">summarize n = count() by conversation_tool</td>
                <td className="py-1 text-text-muted">group by tool (each msg counts once per tool)</td>
              </tr>
              <tr>
                <td className="py-1 pr-3 font-mono text-text-normal whitespace-nowrap">where tools_used contains "vector_search"</td>
                <td className="py-1 text-text-muted">conversations that used it (use contains, not ==)</td>
              </tr>
            </tbody>
          </table>
          <p className="text-text-muted leading-relaxed mb-2.5">
            <code className="text-text-normal">tools_used</code> is a comma-separated string. Use{' '}
            <code className="text-text-normal">contains</code> rather than{' '}
            <code className="text-text-normal">==</code> — equality only matches conversations that
            used <em>exactly</em> that one tool, silently dropping any multi-tool ones.
          </p>
          <p className="text-text-muted mb-1.5">Available tool names:</p>
          <div className="flex flex-wrap gap-1.5">
            {TOOLS.map((t) => (
              <code key={t} className="font-mono text-text-normal bg-scheme-shade_3 px-2 py-0.5 rounded element-border">
                {t}
              </code>
            ))}
          </div>
        </Card>

        <div className="p-3 rounded-lg bg-scheme-shade_2 element-border text-text-muted leading-relaxed">
          <div className="font-semibold text-text-normal mb-1">Modeled on KQL</div>
          This language is a simplified subset of{' '}
          <a
            href="https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/"
            target="_blank"
            rel="noreferrer"
            className="text-blue-500 hover:underline"
          >
            Kusto Query Language
          </a>
          , so most KQL examples online will translate directly. Key differences: use{' '}
          <span className="font-mono text-text-normal">select</span> not{' '}
          <span className="font-mono text-text-normal">project</span>; lists use square brackets{' '}
          <span className="font-mono text-text-normal">[ ]</span> not parentheses;{' '}
          <span className="font-mono text-text-normal">median()</span> is built in; no joins,
          let-statements, time functions, or cross-table queries.
        </div>
      </div>
    </details>
  );
};

export default SyntaxReference;
