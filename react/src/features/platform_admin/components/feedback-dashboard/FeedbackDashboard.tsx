import React, { useEffect, useRef, useState } from 'react';
import FilterBar from './FilterBar';
import { QueryBuilderAdvanced } from './QueryBuilder';
import QueryBuilderHelp from './QueryBuilderHelp';
import ResultsChart from './ResultsChart';
import ResultsTable from './ResultsTable';
import ThreadModal from './ThreadModal';
import { b64decode, b64encode, runQuery } from './api';
import { useFilterBar } from './useFilterBar';
import { useFilterOptions } from './useFilterOptions';
import type { FilterState } from './useFilterBar';
import type { QueryResponse } from './types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isPageReload(): boolean {
  try {
    const entries = window.performance?.getEntriesByType('navigation') ?? [];
    if (entries.length > 0) {
      return (entries[0] as PerformanceNavigationTiming).type === 'reload';
    }
  } catch { /* ignore */ }
  return false;
}

function readQueryFromUrl(): string {
  if (isPageReload()) {
    if (window.location.search)
      window.history.replaceState({}, '', window.location.pathname);
    return '';
  }
  const params = new URLSearchParams(window.location.search);
  const q = params.get('q');
  return q ? b64decode(q) : '';
}

function buildShareableUrl(queryText: string): string {
  const encoded = b64encode(queryText);
  return `${window.location.origin}${window.location.pathname}?q=${encodeURIComponent(encoded)}`;
}

function extractBadToken(errorMessage: string): string | null {
  const match = errorMessage.match(/'([^']+)'|"([^"]+)"/);
  if (!match) return null;
  return match[1] ?? match[2] ?? null;
}

// ---------------------------------------------------------------------------
// Parse a KQL string back into FilterState fields.
// Only recognises clauses that buildKQLFromFilters produces — anything else
// (e.g. custom where clauses) is silently ignored. Missing clauses leave the
// corresponding field at its EMPTY_FILTER_STATE default.
// ---------------------------------------------------------------------------
function parseKQLToFilters(kql: string): Partial<FilterState> {
  const result: Partial<FilterState> = {};
  // Split on pipe separators handling both inline (|) and multi-line (newline | )
  const clauses = kql
    .split(/\n?\s*\|\s*/)
    .map((s) => s.trim())
    .filter(Boolean);

  for (const clause of clauses) {
    let m: RegExpMatchArray | null;

    // date range
    m = clause.match(/^where feedback_submitted_at >= "(\d{4}-\d{2}-\d{2})"/);
    if (m) { result.date_from = m[1]; continue; }

    m = clause.match(/^where feedback_submitted_at <= "(\d{4}-\d{2}-\d{2})"/);
    if (m) { result.date_to = m[1]; continue; }

    // user_id
    m = clause.match(/^where user_id == (\d+)/);
    if (m) { result.user_id = m[1]; continue; }

    // rating (exact before range so exact_rating wins)
    m = clause.match(/^where rating == (\d+)/);
    if (m) { result.exact_rating = m[1]; continue; }

    m = clause.match(/^where rating >= (\d+)/);
    if (m) { result.min_rating = m[1]; continue; }

    m = clause.match(/^where rating <= (\d+)/);
    if (m) { result.max_rating = m[1]; continue; }

    // role
    m = clause.match(/^where role == "([^"\\]*)"/);
    if (m) { result.role = m[1]; continue; }

    // model
    m = clause.match(/^where model == "([^"\\]*)"/);
    if (m) { result.model = m[1]; continue; }

    // tool_call_name
    m = clause.match(/^where tool_call_name == "([^"\\]*)"/);
    if (m) { result.tool_call_name = m[1]; continue; }

    // has_feedback_text
    if (/^where feedback_text != null/.test(clause)) {
      result.has_feedback_text = 'true'; continue;
    }
    if (/^where feedback_text == null/.test(clause)) {
      result.has_feedback_text = 'false'; continue;
    }

    // feedback_text_search (must come after the null checks above)
    m = clause.match(/^where feedback_text contains "([^"\\]*)"/);
    if (m) { result.feedback_text_search = m[1]; continue; }

    // limit
    m = clause.match(/^limit (\d+)/);
    if (m) { result.limit = parseInt(m[1], 10); continue; }
  }

  return result;
}

// ---------------------------------------------------------------------------
// DashboardSection — collapsible panel with a Run button in the header
// ---------------------------------------------------------------------------

type SectionProps = {
  title: string;
  defaultOpen?: boolean;
  onRun: () => void;
  extraActions?: React.ReactNode;
  children: React.ReactNode;
};

const DashboardSection: React.FC<SectionProps> = ({
  title,
  defaultOpen = false,
  onRun,
  extraActions,
  children,
}) => {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="mb-5 rounded-lg bg-scheme-shade_3 element-border text-sm overflow-hidden">
      <div
        className="flex items-center justify-between px-4 py-2.5 cursor-pointer select-none hover:bg-scheme-shade_5/40 transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="flex items-center gap-2 font-semibold text-text-normal">
          <span className="font-mono text-text-muted text-xs">{open ? '▼' : '▶'}</span>
          {title}
        </span>
        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
          {extraActions}
          <button
            onClick={onRun}
            className="px-4 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold active:scale-95 transition-all shadow-sm"
          >
            Run
          </button>
        </div>
      </div>
      {open && <div>{children}</div>}
    </div>
  );
};

// ---------------------------------------------------------------------------
// FeedbackDashboard
// ---------------------------------------------------------------------------

const FeedbackDashboard: React.FC = () => {
  const [queryText, setQueryText] = useState<string>(() => readQueryFromUrl());
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [thread, setThread] = useState<{ conversationId: string; messageUuid: string } | null>(null);
  const [showHelp, setShowHelp] = useState<boolean>(false);
  const [copied, setCopied] = useState(false);

  const {
    filters,
    kqlForExecution,
    hasActiveFilters,
    setFilter,
    setAllFilters,
    resetFilters,
  } = useFilterBar();

  const { options, loading: optionsLoading } = useFilterOptions();

  // When true, the kqlForExecution effect should NOT overwrite queryText because
  // the filter change was itself triggered by a textarea edit.
  const filtersDrivenByTextarea = useRef(false);

  const executeQuery = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setLoading(true);
    setResponse(null);
    try {
      const data = await runQuery(b64encode(trimmed));
      setResponse(data);
    } catch (err) {
      setResponse({
        query_text: trimmed,
        rows: [],
        columns: [],
        is_row_level: true,
        chart_data: null,
        row_count: 0,
        error: err instanceof Error ? err.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------------------------------------------------
  // Filter bar auto-execution (debounced via useFilterBar)
  // When kqlForExecution changes and it was driven by the FilterBar (not by a
  // textarea edit), update the textarea and run the query automatically.
  // -------------------------------------------------------------------------
  const isFirstRender = useRef(true);

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    if (filtersDrivenByTextarea.current) {
      // This kqlForExecution update came from parsing the textarea — don't
      // overwrite what the user typed, and don't auto-run.
      filtersDrivenByTextarea.current = false;
      return;
    }
    // FilterBar drove this change — update textarea to show generated KQL
    setQueryText(kqlForExecution);
    window.history.replaceState({}, '', window.location.pathname);
    void executeQuery(kqlForExecution.trim());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kqlForExecution]);

  // -------------------------------------------------------------------------
  // Initial load from URL ?q=
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (queryText.trim()) {
      void executeQuery(queryText);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -------------------------------------------------------------------------
  // Browser back/forward
  // -------------------------------------------------------------------------
  useEffect(() => {
    const onPopState = () => {
      const next = readQueryFromUrl();
      setQueryText(next);
      if (next.trim()) void executeQuery(next);
      else setResponse(null);
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  // -------------------------------------------------------------------------
  // Shared Run handler (used by all three section Run buttons)
  // Parses the current textarea KQL back into filters so all sections stay
  // in sync after a manual edit.
  // -------------------------------------------------------------------------
  const handleRun = () => {
    const trimmed = queryText.trim();
    if (!trimmed) return;
    // Sync textarea → filters without triggering auto-execution loop
    filtersDrivenByTextarea.current = true;
    setAllFilters(parseKQLToFilters(queryText));
    // Execute
    window.history.pushState({}, '', buildShareableUrl(trimmed));
    void executeQuery(trimmed);
  };

  // -------------------------------------------------------------------------
  // Textarea change — update filters live so the Basic Query Builder reflects
  // what is typed.
  // -------------------------------------------------------------------------
  const handleQueryChange = (v: string) => {
    setQueryText(v);
    // Mark that the coming kqlForExecution update is caused by us, not FilterBar
    filtersDrivenByTextarea.current = true;
    setAllFilters(parseKQLToFilters(v));
  };

  // -------------------------------------------------------------------------
  // FilterBar control change — clear the textarea-source flag so the next
  // kqlForExecution update correctly overwrites the textarea.
  // -------------------------------------------------------------------------
  const handleFilterChange = (key: keyof FilterState, value: string | number) => {
    filtersDrivenByTextarea.current = false;
    setFilter(key, value as string);
  };

  const handleReset = () => {
    filtersDrivenByTextarea.current = false;
    resetFilters();
    setQueryText('');
    setResponse(null);
    window.history.pushState({}, '', window.location.pathname);
  };

  const handleClear = () => {
    filtersDrivenByTextarea.current = false;
    resetFilters();
    setQueryText('');
    setResponse(null);
    window.history.pushState({}, '', window.location.pathname);
  };

  const handleCopyLink = () => {
    const trimmed = queryText.trim();
    if (!trimmed) { alert('Type a query first.'); return; }
    void navigator.clipboard.writeText(buildShareableUrl(trimmed));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Advanced Query Builder appends clauses — also clear the textarea-source flag
  const handleAdvancedChange = (v: string) => {
    filtersDrivenByTextarea.current = false;
    setQueryText(v);
    // Parse back to update Basic filters too
    filtersDrivenByTextarea.current = true;
    setAllFilters(parseKQLToFilters(v));
  };

  const hasQueryRun = response !== null;
  const badToken = response?.error ? extractBadToken(response.error) : null;

  return (
    <div className="container mx-auto p-6 text-text-normal max-w-7xl">
      {/* Page header */}
      <div className="flex items-center gap-3 mb-6">
        <h1 className="text-2xl font-bold">Feedback Dashboard</h1>
        <button
          onClick={() => setShowHelp(true)}
          className="w-6 h-6 rounded-full bg-scheme-shade_5 hover:bg-scheme-shade_6 text-text-muted hover:text-text-normal text-xs font-bold flex items-center justify-center transition-colors"
          title="How the query language works"
        >
          ?
        </button>
      </div>

      {showHelp && <QueryBuilderHelp onClose={() => setShowHelp(false)} />}

      {/* ----------------------------------------------------------------- */}
      {/* Section 1: Basic Query Builder (FilterBar dropdowns)               */}
      {/* ----------------------------------------------------------------- */}
      <DashboardSection title="Basic Query Builder" defaultOpen={true} onRun={handleRun}>
        {/* Render FilterBar content without its built-in outer wrapper so
            the DashboardSection header is the only header shown */}
        <FilterBar
          filters={filters}
          options={options}
          optionsLoading={optionsLoading}
          hasActiveFilters={hasActiveFilters}
          onFilterChange={handleFilterChange}
          onReset={handleReset}
          hideHeader
        />
      </DashboardSection>

      {/* ----------------------------------------------------------------- */}
      {/* Section 2: Advanced Query Builder (visual clause builder)          */}
      {/* ----------------------------------------------------------------- */}
      <DashboardSection title="Advanced Query Builder" onRun={handleRun}>
        <QueryBuilderAdvanced value={queryText} onChange={handleAdvancedChange} />
      </DashboardSection>

      {/* ----------------------------------------------------------------- */}
      {/* Section 3: Textual Query — editable KQL textarea                   */}
      {/* ----------------------------------------------------------------- */}
      <DashboardSection
        title="Textual Query"
        defaultOpen={true}
        onRun={handleRun}
        extraActions={
          <>
            <button
              onClick={handleCopyLink}
              className="px-3 py-1 rounded bg-scheme-shade_5 hover:bg-scheme-shade_6 text-xs transition-all"
            >
              {copied ? 'Copied!' : 'Copy link'}
            </button>
            <button
              onClick={handleClear}
              className="px-3 py-1 rounded bg-scheme-shade_5 hover:bg-scheme-shade_6 text-xs transition-all"
            >
              Clear
            </button>
          </>
        }
      >
        <div className="px-4 py-3">
          <textarea
            rows={4}
            spellCheck={false}
            placeholder="messages | where rating < 3 | limit 20"
            className="w-full font-mono text-sm rounded-lg px-4 py-3 bg-scheme-shade_2 element-border text-text-normal placeholder-text-muted resize-y focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            value={queryText}
            onChange={(e) => handleQueryChange(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                handleRun();
              }
            }}
          />
          <p className="text-xs text-text-muted mt-1.5">
            <span className="font-mono">Ctrl+Enter</span> /{' '}
            <span className="font-mono">⌘+Enter</span> to run.
            Editing here updates the Basic Query Builder filters automatically.
            Clauses added above appear here instantly.
          </p>
        </div>
      </DashboardSection>

      {/* Loading */}
      {loading && (
        <div className="flex items-center gap-3 text-sm text-text-muted mb-4">
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent" />
          Running query…
        </div>
      )}

      {/* Error */}
      {response?.error && (
        <div className="mb-4 px-4 py-3 rounded-lg bg-red-900/30 border border-red-500/40 text-sm text-red-300 font-mono">
          {response.error}
          {badToken && <span className="ml-2 text-red-400 font-bold">← {badToken}</span>}
        </div>
      )}

      {/* Notice */}
      {response?.notice && (
        <div className="mb-4 px-4 py-3 rounded-lg bg-yellow-900/20 border border-yellow-500/30 text-sm text-yellow-300">
          {response.notice}
        </div>
      )}

      {/* Results */}
      {hasQueryRun && !response?.error && (
        <>
          <div className="mb-3 flex items-center gap-3 text-xs text-text-muted">
            <span>
              {response!.row_count} row{response!.row_count !== 1 ? 's' : ''}
            </span>
            {response!.query_text && (
              <span className="font-mono truncate max-w-xl opacity-60">
                {response!.query_text}
              </span>
            )}
          </div>
          {response!.chart_data && <ResultsChart data={response!.chart_data} />}
          {response!.rows.length > 0 ? (
            <ResultsTable
              columns={response!.columns}
              rows={response!.rows}
              isRowLevel={response!.is_row_level}
              onOpenThread={(conversationId, messageUuid) =>
                setThread({ conversationId, messageUuid })
              }
            />
          ) : (
            <div className="py-10 text-center text-text-muted text-sm">No rows returned.</div>
          )}
        </>
      )}

      {/* Syntax quick reference */}
      <details className="mt-6 rounded-lg bg-scheme-shade_3 element-border text-sm">
        <summary className="px-4 py-2.5 cursor-pointer font-semibold select-none hover:text-blue-500 transition-colors">
          Syntax quick reference
        </summary>
        <div className="px-5 py-4 text-xs space-y-4 text-text-muted">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="font-mono font-bold text-text-normal mb-2">messages stream</p>
              {[
                ['rating', 'NUMBER', '1–5 star rating'],
                ['feedback_text', 'TEXT', 'Written comment'],
                ['feedback_submitted_at', 'TIME', 'When rating was submitted'],
                ['model', 'TEXT', 'AI model used'],
                ['role', 'TEXT', 'user / assistant / tool'],
                ['content', 'TEXT', 'Message text'],
                ['user_id', 'NUMBER', 'User account ID'],
                ['conversation_id', 'NUMBER', 'Conversation ID'],
                ['tool_call_name', 'TEXT', 'Tool used (intermediate only)'],
                ['conversation_tool', 'ARRAY', 'Tools used in this conversation'],
              ].map(([f, t, d]) => (
                <div key={f} className="flex items-baseline gap-2 py-0.5 border-b border-scheme-contrast/10 last:border-0">
                  <span className="font-mono text-text-normal w-44 shrink-0">{f}</span>
                  <span className={`text-[10px] px-1 rounded shrink-0 ${
                    t === 'NUMBER' ? 'bg-blue-600/30 text-blue-300' :
                    t === 'TIME' ? 'bg-yellow-600/30 text-yellow-300' :
                    t === 'ARRAY' ? 'bg-purple-600/30 text-purple-300' :
                    'bg-green-600/30 text-green-300'
                  }`}>{t}</span>
                  <span>{d}</span>
                </div>
              ))}
            </div>
            <div>
              <p className="font-mono font-bold text-text-normal mb-2">conversations stream</p>
              {[
                ['conversation_id', 'NUMBER', 'Conversation ID'],
                ['user_id', 'NUMBER', 'Owner user ID'],
                ['name', 'TEXT', 'Auto-generated title'],
                ['created_at', 'TIME', 'When conversation started'],
                ['message_count', 'NUMBER', 'Total message turns'],
                ['rated_count', 'NUMBER', 'Number of rated messages'],
                ['avg_rating', 'NUMBER', 'Mean rating across messages'],
                ['min_rating', 'NUMBER', 'Lowest rating in conv.'],
                ['max_rating', 'NUMBER', 'Highest rating in conv.'],
                ['last_rated_at', 'TIME', 'Most recent rating time'],
                ['tools_used', 'ARRAY', 'Comma-list of distinct tools'],
              ].map(([f, t, d]) => (
                <div key={f} className="flex items-baseline gap-2 py-0.5 border-b border-scheme-contrast/10 last:border-0">
                  <span className="font-mono text-text-normal w-36 shrink-0">{f}</span>
                  <span className={`text-[10px] px-1 rounded shrink-0 ${
                    t === 'NUMBER' ? 'bg-blue-600/30 text-blue-300' :
                    t === 'TIME' ? 'bg-yellow-600/30 text-yellow-300' :
                    t === 'ARRAY' ? 'bg-purple-600/30 text-purple-300' :
                    'bg-green-600/30 text-green-300'
                  }`}>{t}</span>
                  <span>{d}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="font-semibold text-text-normal mb-1">Operators</p>
              {['== / !=  equal / not equal', '< > <= >=  numeric', 'contains "x"  substring', 'startswith "x"  prefix', 'in [1,2,3]  membership', '== null / != null  missing/present'].map(o => (
                <div key={o} className="font-mono text-text-normal/80 py-0.5 text-[11px]">{o}</div>
              ))}
            </div>
            <div>
              <p className="font-semibold text-text-normal mb-1">Aggregation functions</p>
              {['avg(field)', 'count()', 'min(field)', 'max(field)', 'sum(field)', 'median(field)'].map(f => (
                <div key={f} className="font-mono text-text-normal/80 py-0.5 text-[11px]">{f}</div>
              ))}
            </div>
          </div>
        </div>
      </details>

      {thread && (
        <ThreadModal
          conversationId={thread.conversationId}
          focalMessageUuid={thread.messageUuid}
          onClose={() => setThread(null)}
        />
      )}
    </div>
  );
};

export default FeedbackDashboard;