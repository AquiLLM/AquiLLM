import React from 'react';

const SyntaxReference: React.FC = () => {
  return (
    <details className="mb-5 rounded-lg bg-scheme-shade_3 element-border text-sm">
      <summary className="px-4 py-2.5 cursor-pointer font-semibold select-none hover:text-blue-500 transition-colors">
        Syntax quick reference
      </summary>
      <div className="px-4 pb-5 pt-3 space-y-5 text-xs">

        <div className="p-3 rounded bg-scheme-shade_2 element-border text-text-muted leading-relaxed">
          <span className="font-semibold text-text-normal">Note: </span>
          This query language is modeled on{' '}
          <a
            href="https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/"
            target="_blank"
            rel="noreferrer"
            className="text-blue-500 hover:underline"
          >
            KQL (Kusto Query Language)
          </a>{' '}
          but is a simplified subset with a few differences noted below. KQL documentation and
          examples online will mostly apply here, but not all KQL features exist.
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono">
          <div>
            <p className="font-sans font-semibold text-text-normal mb-1 text-xs">Row query example</p>
            <pre className="leading-relaxed text-text-muted bg-scheme-shade_2 rounded p-2">
{`messages
| where rating < 3
| select rating, model, feedback_text
| order by rating asc
| limit 50`}
            </pre>
            <p className="font-sans text-text-muted mt-2 text-xs leading-relaxed">
              Finds the worst-rated messages, showing which model produced them and what the user
              said in their feedback. Useful for quickly identifying pain points — the messages
              users were most unhappy with.
            </p>
          </div>
          <div>
            <p className="font-sans font-semibold text-text-normal mb-1 text-xs">Aggregate query example</p>
            <pre className="leading-relaxed text-text-muted bg-scheme-shade_2 rounded p-2">
{`messages
| where rating != null
| summarize n = count() by rating
| order by rating asc`}
            </pre>
            <p className="font-sans text-text-muted mt-2 text-xs leading-relaxed">
              Produces a rating distribution — how many messages received each star score (1
              through 5). Useful for understanding overall user sentiment at a glance, and will
              produce a histogram chart. Note: always filter out null models when grouping by
              model (<span className="text-text-normal">where model != null</span>), since user
              messages have no model and would otherwise appear as a separate group.
            </p>
          </div>
        </div>

        <div className="space-y-3 text-text-muted">
          <div>
            <p className="font-semibold text-text-normal mb-1">Streams &amp; their fields</p>
            <p className="text-text-muted mb-2 leading-relaxed">
              Every query starts with a <em>stream</em>, which determines what each result row represents and which fields you can use.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="font-mono text-text-normal mb-1">messages</p>
                <p className="text-text-muted/80 mb-1.5 leading-relaxed">One row per message turn (user message, assistant reply, or tool call).</p>
                <div className="font-mono grid grid-cols-2 gap-x-3 gap-y-0.5">
                  <span>rating</span><span className="text-text-muted/60">— 1–5 stars</span>
                  <span>feedback_text</span><span className="text-text-muted/60">— written comment</span>
                  <span>feedback_submitted_at</span><span className="text-text-muted/60">— rating timestamp</span>
                  <span>model</span><span className="text-text-muted/60">— AI model</span>
                  <span>role</span><span className="text-text-muted/60">— user / assistant / tool</span>
                  <span>content</span><span className="text-text-muted/60">— message text</span>
                  <span>created_at</span><span className="text-text-muted/60">— message timestamp</span>
                  <span>user_id</span><span className="text-text-muted/60">— user account ID</span>
                  <span>conversation_id</span><span className="text-text-muted/60">— conversation ID</span>
                  <span>sequence_number</span><span className="text-text-muted/60">— position in conv.</span>
                  <span>message_uuid</span><span className="text-text-muted/60">— unique message ID</span>
                  <span>tool_call_name</span><span className="text-text-muted/60">— tool used (intermediate)</span>
                  <span>conversation_tool</span><span className="text-text-muted/60">— tools used in conv. (see below)</span>
                </div>
              </div>
              <div>
                <p className="font-mono text-text-normal mb-1">conversations</p>
                <p className="text-text-muted/80 mb-1.5 leading-relaxed">One row per conversation, with derived aggregates across its messages.</p>
                <div className="font-mono grid grid-cols-2 gap-x-3 gap-y-0.5">
                  <span>conversation_id</span><span className="text-text-muted/60">— conversation ID</span>
                  <span>user_id</span><span className="text-text-muted/60">— owner</span>
                  <span>name</span><span className="text-text-muted/60">— auto title</span>
                  <span>created_at</span><span className="text-text-muted/60">— start time</span>
                  <span>updated_at</span><span className="text-text-muted/60">— last activity</span>
                  <span>message_count</span><span className="text-text-muted/60">— total turns</span>
                  <span>rated_count</span><span className="text-text-muted/60">— turns with a rating</span>
                  <span>avg_rating</span><span className="text-text-muted/60">— mean rating</span>
                  <span>min_rating</span><span className="text-text-muted/60">— lowest rating</span>
                  <span>max_rating</span><span className="text-text-muted/60">— highest rating</span>
                  <span>last_rated_at</span><span className="text-text-muted/60">— most recent rating time</span>
                  <span>tools_used</span><span className="text-text-muted/60">— comma-list of tools used</span>
                </div>
              </div>
            </div>
          </div>

          <div>
            <p className="font-semibold text-text-normal mb-1">Clauses (in order)</p>
            <div className="space-y-1 font-mono">
              <div><span className="text-text-normal">messages</span> / <span className="text-text-normal">conversations</span> — every query must start with one of these</div>
              <div><span className="text-text-normal">| where</span> <em>condition</em> — filter rows; chain with <span className="text-text-normal">and</span> / <span className="text-text-normal">or</span></div>
              <div><span className="text-text-normal">| select</span> <em>field, field, …</em> — choose which columns to return (omit to get all)</div>
              <div><span className="text-text-normal">| summarize</span> <em>alias = func(field), …</em> <span className="text-text-normal">by</span> <em>field</em> — aggregate; omit <span className="text-text-normal">by</span> for a global total</div>
              <div><span className="text-text-normal">| order by</span> <em>field</em> <span className="text-text-normal">asc</span>/<span className="text-text-normal">desc</span> — sort results</div>
              <div><span className="text-text-normal">| limit</span> <em>n</em> — cap number of rows returned</div>
            </div>
          </div>

          <div>
            <p className="font-semibold text-text-normal mb-1">Comparison operators</p>
            <div className="font-mono space-y-0.5">
              <div><span className="text-text-normal">{'==  !=  <  >  <=  >='}</span> — standard comparisons</div>
              <div><span className="text-text-normal">field startswith &quot;text&quot;</span> — prefix match (case-insensitive)</div>
              <div><span className="text-text-normal">field contains &quot;text&quot;</span> — substring match (case-insensitive)</div>
              <div>
                <span className="text-text-normal">field in [1, 2, 3]</span> — membership test
                <span className="text-text-muted/60"> — note: uses [ ] not ( ) unlike standard KQL</span>
              </div>
              <div><span className="text-text-normal">field == null</span> — matches rows where the field has no value</div>
              <div><span className="text-text-normal">field != null</span> — matches rows where the field has a value</div>
            </div>
          </div>

          <div>
            <p className="font-semibold text-text-normal mb-1">Aggregation functions</p>
            <div className="font-mono space-y-0.5">
              <div><span className="text-text-normal">avg(field)</span> — average of non-null values</div>
              <div><span className="text-text-normal">count()</span> — total number of rows in group</div>
              <div><span className="text-text-normal">min(field)</span> / <span className="text-text-normal">max(field)</span> — lowest / highest value</div>
              <div><span className="text-text-normal">sum(field)</span> — total of all values</div>
              <div>
                <span className="text-text-normal">median(field)</span> — middle value
                <span className="text-text-muted/60"> — not in standard KQL (use percentile() there instead)</span>
              </div>
            </div>
          </div>

          <div>
            <p className="font-semibold text-text-normal mb-1">
              Working with tool use (<code>conversation_tool</code>)
            </p>
            <p className="text-text-muted mb-1.5 leading-relaxed">
              When the AI uses a tool (e.g. document search), the tool call is stored on a separate
              message from the rated response.{' '}
              <code className="text-text-normal">conversation_tool</code> describes the tools used
              anywhere in this message's conversation, so you can ask questions across the whole
              conversation, not just the rated turn.
            </p>
            <div className="font-mono space-y-0.5">
              <div><span className="text-text-normal">where conversation_tool == &quot;vector_search&quot;</span> <span className="text-text-muted/60">— conversation used vector search</span></div>
              <div><span className="text-text-normal">where conversation_tool != null</span> <span className="text-text-muted/60">— conversation used any tool</span></div>
              <div><span className="text-text-normal">where conversation_tool == null</span> <span className="text-text-muted/60">— conversation used no tools</span></div>
              <div><span className="text-text-normal">select rating, conversation_tool</span> <span className="text-text-muted/60">— show tools per row (comma-separated, null when none)</span></div>
              <div><span className="text-text-normal">summarize n = count() by conversation_tool</span> <span className="text-text-muted/60">— group by tool (each message counts once per tool used)</span></div>
            </div>
            <p className="text-text-muted mt-2 leading-relaxed">
              On the <code className="text-text-normal">conversations</code> stream the equivalent
              field is <code className="text-text-normal">tools_used</code> — a comma-separated
              string of every distinct tool the conversation used. Filter it with{' '}
              <code className="text-text-normal">contains</code> rather than{' '}
              <code className="text-text-normal">==</code>, e.g.{' '}
              <code className="text-text-normal">where tools_used contains &quot;vector_search&quot;</code>.{' '}
              <code className="text-text-normal">==</code> only matches conversations that used
              <em> exactly</em> that one tool, so any multi-tool conversation gets silently dropped.
            </p>
            <p className="text-text-muted mt-1.5 leading-relaxed">
              Available tool names: <code className="text-text-normal">vector_search</code>,{' '}
              <code className="text-text-normal">more_context</code>,{' '}
              <code className="text-text-normal">document_ids</code>,{' '}
              <code className="text-text-normal">whole_document</code>,{' '}
              <code className="text-text-normal">search_single_document</code>,{' '}
              <code className="text-text-normal">sky_subtraction</code>,{' '}
              <code className="text-text-normal">flat_fielding</code>,{' '}
              <code className="text-text-normal">detect_point_sources</code>
            </p>
          </div>

          <div className="font-mono p-2 rounded bg-scheme-shade_2 text-text-muted/70 leading-relaxed">
            <span className="font-sans font-semibold text-text-normal">Differences from KQL: </span>
            Use <span className="text-text-normal">select</span> instead of{' '}
            <span className="text-text-normal">project</span> · Use{' '}
            <span className="text-text-normal">in [  ]</span> with square brackets ·{' '}
            <span className="text-text-normal">median()</span> is built-in · No joins, let
            statements, time functions, or cross-table queries · Only the fields listed above are
            accessible
          </div>
        </div>
      </div>
    </details>
  );
};

export default SyntaxReference;
