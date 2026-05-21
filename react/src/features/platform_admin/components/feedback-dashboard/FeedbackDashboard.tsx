import React, { useEffect, useMemo, useRef, useState } from 'react';
import FilterBar from './FilterBar';
import { QueryBuilderAdvanced } from './QueryBuilder';
import QueryBuilderHelp from './QueryBuilderHelp';
import ResultsChart from './ResultsChart';
import ResultsTable from './ResultsTable';
import ThreadModal from './ThreadModal';
import { b64decode, b64encode, runQuery } from './api';
import { useFilterBar, buildBasicClauses, EMPTY_FILTER_STATE } from './useFilterBar';
import { useAdvancedBuilder, buildAdvancedClauses } from './useAdvancedBuilder';
import { useFilterOptions } from './useFilterOptions';
import type { FilterState } from './useFilterBar';
import type { ParsedAdvanced } from './useAdvancedBuilder';
import type { QueryResponse } from './types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isPageReload(): boolean {
  try {
    const entries = window.performance?.getEntriesByType('navigation') ?? [];
    if (entries.length > 0)
      return (entries[0] as PerformanceNavigationTiming).type === 'reload';
  } catch { /* ignore */ }
  return false;
}

function readQueryFromUrl(): string {
  if (isPageReload()) {
    if (window.location.search) window.history.replaceState({}, '', window.location.pathname);
    return '';
  }
  const params = new URLSearchParams(window.location.search);
  const q = params.get('q');
  return q ? b64decode(q) : '';
}

function buildShareableUrl(text: string): string {
  return `${window.location.origin}${window.location.pathname}?q=${encodeURIComponent(b64encode(text))}`;
}

function extractBadToken(msg: string): string | null {
  const m = msg.match(/'([^']+)'|"([^"]+)"/);
  return m ? (m[1] ?? m[2] ?? null) : null;
}

// Build the full combined KQL from Basic filters + Advanced clause list.
// Basic where clauses come first; Advanced appends after.
function buildCombinedKQL(basicClauses: string[], advancedClauses: string[]): string {
  const all = [...basicClauses, ...advancedClauses];
  if (all.length === 0) return 'messages';
  return `messages\n${all.map(c => `| ${c}`).join('\n')}`;
}

// ---------------------------------------------------------------------------
// Full KQL parser — splits each clause between Basic and Advanced.
//
// Basic patterns: the exact operators that FilterBar dropdowns can represent.
// Everything else (different operators, summarize, select, order by, limit)
// goes into Advanced rows.
// ---------------------------------------------------------------------------
function parseKQLFull(kql: string): { filters: Partial<FilterState>; advanced: ParsedAdvanced } {
  const filters: Partial<FilterState> = {};
  const advanced: ParsedAdvanced = {
    whereRows: [],
    summarizeRows: [],
    selectFields: [],
    orderByRows: [],
    limitValue: null,
  };

  const clauses = kql
    .split(/\n?\s*\|\s*/)
    .map(s => s.trim())
    .filter(Boolean);

  for (const clause of clauses) {
    // Skip the stream identifier
    if (clause === 'messages' || clause.startsWith('messages ')) continue;

    // ── summarize ──────────────────────────────────────────────────────
    {
      const m = clause.match(/^summarize (\w+)\s*=\s*(\w+)\(([^)]*)\)(?:\s+by\s+(\S+))?/);
      if (m) {
        advanced.summarizeRows.push({
          alias: m[1],
          func: m[2],
          field: m[3]?.trim() ?? '',
          byField: m[4]?.trim() ?? '',
        });
        continue;
      }
    }

    // ── select ─────────────────────────────────────────────────────────
    {
      const m = clause.match(/^select (.+)/);
      if (m) {
        advanced.selectFields = m[1].split(',').map(f => f.trim()).filter(Boolean);
        continue;
      }
    }

    // ── order by ───────────────────────────────────────────────────────
    {
      const m = clause.match(/^order by (\S+)(?:\s+(asc|desc))?/i);
      if (m) {
        advanced.orderByRows.push({
          field: m[1],
          dir: (m[2]?.toLowerCase() ?? 'desc') as 'asc' | 'desc',
        });
        continue;
      }
    }

    // ── limit ──────────────────────────────────────────────────────────
    {
      const m = clause.match(/^limit (\d+)/);
      if (m) { advanced.limitValue = parseInt(m[1], 10); continue; }
    }

    // ── where — try Basic patterns first, fall through to Advanced ──────
    if (clause.startsWith('where ')) {
      let matched = false;

      // date range (Basic only handles >= / <=)
      let m = clause.match(/^where feedback_submitted_at >= "(\d{4}-\d{2}-\d{2})"/);
      if (m) { filters.date_from = m[1]; matched = true; }

      if (!matched) {
        m = clause.match(/^where feedback_submitted_at <= "(\d{4}-\d{2}-\d{2})"/);
        if (m) { filters.date_to = m[1]; matched = true; }
      }

      // user_id (Basic only handles ==)
      if (!matched) {
        m = clause.match(/^where user_id == (\d+)/);
        if (m) { filters.user_id = m[1]; matched = true; }
      }

      // rating == (Basic exact_rating)
      if (!matched) {
        m = clause.match(/^where rating == (\d+)/);
        if (m) { filters.exact_rating = m[1]; matched = true; }
      }

      // rating >= (Basic min_rating)
      if (!matched) {
        m = clause.match(/^where rating >= (\d+)/);
        if (m) { filters.min_rating = m[1]; matched = true; }
      }

      // rating <= (Basic max_rating)
      if (!matched) {
        m = clause.match(/^where rating <= (\d+)/);
        if (m) { filters.max_rating = m[1]; matched = true; }
      }

      // role == (Basic)
      if (!matched) {
        m = clause.match(/^where role == "([^"\\]*)"/);
        if (m) { filters.role = m[1]; matched = true; }
      }

      // model == (Basic)
      if (!matched) {
        m = clause.match(/^where model == "([^"\\]*)"/);
        if (m) { filters.model = m[1]; matched = true; }
      }

      // tool_call_name == (Basic)
      if (!matched) {
        m = clause.match(/^where tool_call_name == "([^"\\]*)"/);
        if (m) { filters.tool_call_name = m[1]; matched = true; }
      }

      // feedback_text != null (Basic has_feedback_text)
      if (!matched && /^where feedback_text != null/.test(clause)) {
        filters.has_feedback_text = 'true'; matched = true;
      }

      // feedback_text == null (Basic has_feedback_text)
      if (!matched && /^where feedback_text == null/.test(clause)) {
        filters.has_feedback_text = 'false'; matched = true;
      }

      // feedback_text contains (Basic text search)
      if (!matched) {
        m = clause.match(/^where feedback_text contains "([^"\\]*)"/);
        if (m) { filters.feedback_text_search = m[1]; matched = true; }
      }

      if (!matched) {
        // Doesn't match any Basic pattern → Advanced where row
        const rest = clause.slice('where '.length).trim();

        // Parse: field op value
        // Try null operators first
        const nullM = rest.match(/^(\S+)\s+(==|!=) null$/);
        if (nullM) {
          advanced.whereRows.push({ field: nullM[1], op: `${nullM[2]} null`, value: '' });
          continue;
        }

        // contains / startswith
        const strOpM = rest.match(/^(\S+)\s+(contains|startswith) "([^"]*)"/);
        if (strOpM) {
          advanced.whereRows.push({ field: strOpM[1], op: strOpM[2], value: strOpM[3] });
          continue;
        }

        // in [...]
        const inM = rest.match(/^(\S+)\s+in \[([^\]]*)\]/);
        if (inM) {
          advanced.whereRows.push({ field: inM[1], op: 'in', value: inM[2] });
          continue;
        }

        // comparison operators
        const compM = rest.match(/^(\S+)\s+(==|!=|>=|<=|>|<)\s+(.+)/);
        if (compM) {
          const rawVal = compM[3].trim();
          // Strip surrounding quotes if present
          const val = rawVal.startsWith('"') && rawVal.endsWith('"')
            ? rawVal.slice(1, -1)
            : rawVal;
          advanced.whereRows.push({ field: compM[1], op: compM[2], value: val });
          continue;
        }

        // Unrecognised — skip
      }
    }
  }

  return { filters, advanced };
}

// ---------------------------------------------------------------------------
// DashboardSection
// ---------------------------------------------------------------------------

type SectionProps = {
  title: string;
  defaultOpen?: boolean;
  onRun: () => void;
  extraActions?: React.ReactNode;
  children: React.ReactNode;
};

const DashboardSection: React.FC<SectionProps> = ({
  title, defaultOpen = false, onRun, extraActions, children,
}) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mb-5 rounded-lg bg-scheme-shade_3 element-border text-sm overflow-hidden">
      <div
        className="flex items-center justify-between px-4 py-2.5 cursor-pointer select-none hover:bg-scheme-shade_5/40 transition-colors"
        onClick={() => setOpen(o => !o)}
      >
        <span className="flex items-center gap-2 font-semibold text-text-normal">
          <span className="font-mono text-text-muted text-xs">{open ? '▼' : '▶'}</span>
          {title}
        </span>
        <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
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
  const [showHelp, setShowHelp] = useState(false);
  const [copied, setCopied] = useState(false);

  const { filters, basicKQL, hasActiveFilters, setFilter, setAllFilters, resetFilters } =
    useFilterBar();
  const adv = useAdvancedBuilder();
  const { options, loading: optionsLoading } = useFilterOptions();

  // ── Combined KQL (derived) ────────────────────────────────────────────
  // basicKQL already encodes the debounced basic where clauses.
  // We extract just the clause strings (everything after "messages\n| ").
  const basicClauseStrings = useMemo(() => {
    if (basicKQL === 'messages') return [];
    return basicKQL
      .split(/\n?\s*\|\s*/)
      .map(s => s.trim())
      .filter(s => s && s !== 'messages');
  }, [basicKQL]);

  const advancedClauseStrings = useMemo(
    () => buildAdvancedClauses(adv.state),
    [adv.state],
  );

  const combinedKQL = useMemo(
    () => buildCombinedKQL(basicClauseStrings, advancedClauseStrings),
    [basicClauseStrings, advancedClauseStrings],
  );

  // ── Sync control ──────────────────────────────────────────────────────
  // true  → state changed via UI; combinedKQL effect should update textarea + run
  // false → textarea changed state; combinedKQL effect should be suppressed
  const stateChangedByUI = useRef(false);
  const isFirstRender = useRef(true);

  // ── Auto-run when combined KQL changes due to UI interaction ──────────
  useEffect(() => {
    if (isFirstRender.current) { isFirstRender.current = false; return; }
    if (!stateChangedByUI.current) return;
    stateChangedByUI.current = false;
    setQueryText(combinedKQL);
    window.history.replaceState({}, '', window.location.pathname);
    void executeQuery(combinedKQL);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [combinedKQL]);

  // ── Execution ─────────────────────────────────────────────────────────
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
        rows: [], columns: [], is_row_level: true, chart_data: null, row_count: 0,
        error: err instanceof Error ? err.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  };

  // ── Initial URL load ──────────────────────────────────────────────────
  useEffect(() => {
    if (queryText.trim()) {
      const { filters: f, advanced: a } = parseKQLFull(queryText);
      setAllFilters(f);
      adv.setFromParsed(a);
      void executeQuery(queryText);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Back/forward ──────────────────────────────────────────────────────
  useEffect(() => {
    const onPopState = () => {
      const next = readQueryFromUrl();
      setQueryText(next);
      if (next.trim()) {
        const { filters: f, advanced: a } = parseKQLFull(next);
        setAllFilters(f);
        adv.setFromParsed(a);
        void executeQuery(next);
      } else {
        setResponse(null);
      }
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Shared Run ────────────────────────────────────────────────────────
  const handleRun = () => {
    const trimmed = queryText.trim();
    if (!trimmed) return;
    // Sync textarea → state (don't trigger auto-run loop)
    stateChangedByUI.current = false;
    const { filters: f, advanced: a } = parseKQLFull(trimmed);
    setAllFilters(f);
    adv.setFromParsed(a);
    // Execute with what the user sees
    window.history.pushState({}, '', buildShareableUrl(trimmed));
    void executeQuery(trimmed);
  };

  // ── Textarea change → sync both Basic and Advanced ────────────────────
  const handleQueryChange = (v: string) => {
    setQueryText(v);
    stateChangedByUI.current = false; // suppress combinedKQL auto-run
    const { filters: f, advanced: a } = parseKQLFull(v);
    setAllFilters(f);
    adv.setFromParsed(a);
  };

  // ── Basic filter control change ───────────────────────────────────────
  const handleFilterChange = (key: keyof FilterState, value: string) => {
    stateChangedByUI.current = true;
    setFilter(key, value);
  };

  const handleReset = () => {
    stateChangedByUI.current = true;
    resetFilters();
    adv.reset();
    setQueryText('');
    setResponse(null);
    window.history.pushState({}, '', window.location.pathname);
  };

  const handleClear = () => {
    stateChangedByUI.current = false;
    resetFilters();
    adv.reset();
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

  // ── Advanced builder actions (each sets stateChangedByUI = true) ──────
  const advAction = <T extends unknown[]>(fn: (...args: T) => void) =>
    (...args: T) => { stateChangedByUI.current = true; fn(...args); };

  const hasQueryRun = response !== null;
  const badToken = response?.error ? extractBadToken(response.error) : null;

  return (
    <div className="container mx-auto p-6 text-text-normal max-w-7xl">
      <div className="flex items-center gap-3 mb-6">
        <h1 className="text-2xl font-bold">Feedback Dashboard</h1>
        <button
          onClick={() => setShowHelp(true)}
          className="w-6 h-6 rounded-full bg-scheme-shade_5 hover:bg-scheme-shade_6 text-text-muted hover:text-text-normal text-xs font-bold flex items-center justify-center transition-colors"
          title="How the query language works"
        >?</button>
      </div>

      {showHelp && <QueryBuilderHelp onClose={() => setShowHelp(false)} />}

      {/* Section 1: Basic Query Builder */}
      <DashboardSection
        title="Basic Query Builder"
        defaultOpen={true}
        onRun={handleRun}
        extraActions={
          <button
            onClick={handleClear}
            className="px-3 py-1 rounded bg-scheme-shade_5 hover:bg-scheme-shade_6 text-xs transition-all"
          >Clear</button>
        }
      >
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

      {/* Section 2: Advanced Query Builder */}
      <DashboardSection
        title="Advanced Query Builder"
        onRun={handleRun}
        extraActions={
          <button
            onClick={handleClear}
            className="px-3 py-1 rounded bg-scheme-shade_5 hover:bg-scheme-shade_6 text-xs transition-all"
          >Clear</button>
        }
      >
        <QueryBuilderAdvanced
          state={adv.state}
          onAddWhere={advAction(adv.addWhere)}
          onRemoveWhere={advAction(adv.removeWhere)}
          onAddSummarize={advAction(adv.addSummarize)}
          onRemoveSummarize={advAction(adv.removeSummarize)}
          onToggleSelect={advAction(adv.toggleSelect)}
          onRemoveSelect={advAction(adv.removeSelect)}
          onAddOrderBy={advAction(adv.addOrderBy)}
          onRemoveOrderBy={advAction(adv.removeOrderBy)}
          onSetLimit={advAction(adv.setLimitValue)}
        />
      </DashboardSection>

      {/* Section 3: Textual Query */}
      <DashboardSection
        title="Textual Query"
        defaultOpen={true}
        onRun={handleRun}
        extraActions={
          <>
            <button
              onClick={handleCopyLink}
              className="px-3 py-1 rounded bg-scheme-shade_5 hover:bg-scheme-shade_6 text-xs transition-all"
            >{copied ? 'Copied!' : 'Copy link'}</button>
            <button
              onClick={handleClear}
              className="px-3 py-1 rounded bg-scheme-shade_5 hover:bg-scheme-shade_6 text-xs transition-all"
            >Clear</button>
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
            onChange={e => handleQueryChange(e.target.value)}
            onKeyDown={e => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                handleRun();
              }
            }}
          />
          <p className="text-xs text-text-muted mt-1.5">
            <span className="font-mono">Ctrl+Enter</span> /{' '}
            <span className="font-mono">⌘+Enter</span> to run.
            Editing here updates both builders automatically.
            Clauses added in either builder appear here instantly.
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
            <span>{response!.row_count} row{response!.row_count !== 1 ? 's' : ''}</span>
            {response!.query_text && (
              <span className="font-mono truncate max-w-xl opacity-60">{response!.query_text}</span>
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
                ['rating','NUMBER','1–5 star rating'],
                ['feedback_text','TEXT','Written comment'],
                ['feedback_submitted_at','TIME','When rating was submitted'],
                ['model','TEXT','AI model used'],
                ['role','TEXT','user / assistant / tool'],
                ['content','TEXT','Message text'],
                ['user_id','NUMBER','User account ID'],
                ['conversation_id','NUMBER','Conversation ID'],
                ['tool_call_name','TEXT','Tool used (intermediate only)'],
                ['conversation_tool','ARRAY','Tools used in this conversation'],
              ].map(([f,t,d]) => (
                <div key={f} className="flex items-baseline gap-2 py-0.5 border-b border-scheme-contrast/10 last:border-0">
                  <span className="font-mono text-text-normal w-44 shrink-0">{f}</span>
                  <span className={`text-[10px] px-1 rounded shrink-0 ${
                    t==='NUMBER'?'bg-blue-600/30 text-blue-300':
                    t==='TIME'?'bg-yellow-600/30 text-yellow-300':
                    t==='ARRAY'?'bg-purple-600/30 text-purple-300':
                    'bg-green-600/30 text-green-300'}`}>{t}</span>
                  <span>{d}</span>
                </div>
              ))}
            </div>
            <div>
              <p className="font-mono font-bold text-text-normal mb-2">conversations stream</p>
              {[
                ['conversation_id','NUMBER','Conversation ID'],
                ['user_id','NUMBER','Owner user ID'],
                ['name','TEXT','Auto-generated title'],
                ['created_at','TIME','When conversation started'],
                ['message_count','NUMBER','Total message turns'],
                ['rated_count','NUMBER','Number of rated messages'],
                ['avg_rating','NUMBER','Mean rating across messages'],
                ['min_rating','NUMBER','Lowest rating in conv.'],
                ['max_rating','NUMBER','Highest rating in conv.'],
                ['last_rated_at','TIME','Most recent rating time'],
                ['tools_used','ARRAY','Comma-list of distinct tools'],
              ].map(([f,t,d]) => (
                <div key={f} className="flex items-baseline gap-2 py-0.5 border-b border-scheme-contrast/10 last:border-0">
                  <span className="font-mono text-text-normal w-36 shrink-0">{f}</span>
                  <span className={`text-[10px] px-1 rounded shrink-0 ${
                    t==='NUMBER'?'bg-blue-600/30 text-blue-300':
                    t==='TIME'?'bg-yellow-600/30 text-yellow-300':
                    t==='ARRAY'?'bg-purple-600/30 text-purple-300':
                    'bg-green-600/30 text-green-300'}`}>{t}</span>
                  <span>{d}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="font-semibold text-text-normal mb-1">Operators</p>
              {['== / !=  equal / not equal','< > <= >=  numeric','contains "x"  substring','startswith "x"  prefix','in [1,2,3]  membership','== null / != null  missing/present'].map(o => (
                <div key={o} className="font-mono text-text-normal/80 py-0.5 text-[11px]">{o}</div>
              ))}
            </div>
            <div>
              <p className="font-semibold text-text-normal mb-1">Aggregation functions</p>
              {['avg(field)','count()','min(field)','max(field)','sum(field)','median(field)'].map(f => (
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