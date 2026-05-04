import React, { useEffect, useRef, useState } from 'react';
import FilterBar from './FilterBar';
import FilterKQLPreview from './FilterKQLPreview';
import QueryBuilder from './QueryBuilder';
import QueryBuilderHelp from './QueryBuilderHelp';
import QueryEditor from './QueryEditor';
import SyntaxReference from './SyntaxReference';
import ResultsChart from './ResultsChart';
import ResultsTable from './ResultsTable';
import ThreadModal from './ThreadModal';
import { b64decode, b64encode, runQuery } from './api';
import { useFilterBar } from './useFilterBar';
import { useFilterOptions } from './useFilterOptions';
import type { QueryResponse } from './types';

function isPageReload(): boolean {
  const entries = performance.getEntriesByType('navigation') as PerformanceNavigationTiming[];
  return entries.length > 0 && entries[0].type === 'reload';
}

function readQueryFromUrl(): string {
  if (isPageReload()) {
    if (window.location.search) {
      window.history.replaceState({}, '', window.location.pathname);
    }
    return '';
  }
  const params = new URLSearchParams(window.location.search);
  const q = params.get('q');
  return q ? b64decode(q) : '';
}

function buildShareableUrl(queryText: string): string {
  const encoded = b64encode(queryText.trim());
  return `${window.location.origin}${window.location.pathname}?q=${encodeURIComponent(encoded)}`;
}

function extractBadToken(errorMessage: string): string | null {
  const match = errorMessage.match(/'([^']+)'|"([^"]+)"/);
  if (!match) return null;
  return match[1] ?? match[2] ?? null;
}

function renderQueryWithHighlight(queryText: string, badToken: string | null): React.ReactNode {
  if (!badToken) return queryText;
  const idx = queryText.indexOf(badToken);
  if (idx === -1) return queryText;
  return (
    <>
      {queryText.slice(0, idx)}
      <span className="bg-red-500/30 rounded px-0.5 underline decoration-red-400 decoration-wavy">
        {queryText.slice(idx, idx + badToken.length)}
      </span>
      {queryText.slice(idx + badToken.length)}
    </>
  );
}

const FeedbackDashboard: React.FC = () => {
  const [queryText, setQueryText] = useState<string>(() => readQueryFromUrl());
  const [response, setResponse]   = useState<QueryResponse | null>(null);
  const [loading, setLoading]     = useState(false);
  const [thread, setThread]       = useState<{ conversationId: string; messageUuid: string } | null>(null);
  const [showHelp, setShowHelp]   = useState(false);

  const {
    filters,
    kqlPreview,
    kqlForExecution,
    hasActiveFilters,
    setFilter,
    resetFilters,
  } = useFilterBar();

  const { options, loading: optionsLoading } = useFilterOptions();

  // track whether the last query came from the filter bar or the editor
  // so we know whether to show the filter KQL preview or the editor KQL
  const [filterBarActive, setFilterBarActive] = useState(false);

  const executeQuery = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) {
      setResponse(null);
      return;
    }
    setLoading(true);
    try {
      const result = await runQuery(b64encode(trimmed));
      setResponse(result);
    } catch (err) {
      setResponse({
        query_text: trimmed,
        rows: [],
        columns: [],
        is_row_level: true,
        chart_data: null,
        row_count: 0,
        error: err instanceof Error ? err.message : 'Request failed',
      });
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------------------------------------------------
  // filter bar auto-execution
  // when kqlForExecution changes (debounced), run the filter-generated query
  // also load the generated KQL into the query editor so the user can see
  // and further edit it
  // -------------------------------------------------------------------------
  const isFirstRender = useRef(true);

  useEffect(() => {
    // skip on first render — only run when filters actually change
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    const trimmed = kqlForExecution.trim();
    setFilterBarActive(true);
    setQueryText(kqlForExecution);
    window.history.replaceState({}, '', window.location.pathname);
    void executeQuery(trimmed);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kqlForExecution]);

  // -------------------------------------------------------------------------
  // manual editor run (Run button or Ctrl+Enter)
  // when the user manually runs a query, mark filter bar as inactive so the
  // preview shows the editor content, not the filter-generated KQL
  // -------------------------------------------------------------------------
  const handleRun = () => {
    const trimmed = queryText.trim();
    if (!trimmed) return;
    setFilterBarActive(false);
    const url = buildShareableUrl(trimmed);
    window.history.pushState({}, '', url);
    void executeQuery(trimmed);
  };

  const handleClear = () => {
    setQueryText('');
    setResponse(null);
    setFilterBarActive(false);
    window.history.pushState({}, '', window.location.pathname);
  };

  const handleCopyLink = () => {
    const trimmed = queryText.trim();
    if (!trimmed) { alert('Type a query first.'); return; }
    void navigator.clipboard.writeText(buildShareableUrl(trimmed));
  };

  useEffect(() => {
    if (queryText.trim()) {
      void executeQuery(queryText);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const onPopState = () => {
      const next = readQueryFromUrl();
      setQueryText(next);
      setFilterBarActive(false);
      if (next.trim()) void executeQuery(next);
      else setResponse(null);
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const hasQueryRun = response !== null;

  // the KQL shown in the live preview — either the filter-generated KQL
  // (when filter bar is the active source) or the raw editor text
  const previewKQL = filterBarActive ? kqlPreview : queryText;

  return (
    <div className="container mx-auto p-6 text-text-normal max-w-7xl">
      <div className="flex items-center gap-3 mb-6">
        <h1 className="text-3xl font-bold">Feedback Dashboard</h1>
        <button
          type="button"
          onClick={() => setShowHelp(true)}
          title="How does the query language work?"
          className="w-8 h-8 flex items-center justify-center rounded-full bg-scheme-shade_3 hover:bg-blue-600 text-text-normal font-bold text-sm element-border transition-colors"
        >
          ?
        </button>
      </div>

      <FilterBar
        filters={filters}
        options={options}
        optionsLoading={optionsLoading}
        hasActiveFilters={hasActiveFilters}
        onFilterChange={(key, value) => {
          setFilter(key, value as string);
        }}
        onReset={() => {
          resetFilters();
          setFilterBarActive(false);
          setQueryText('');
          setResponse(null);
        }}
      />

      <FilterKQLPreview kql={previewKQL} loading={loading && filterBarActive} />

      <QueryBuilder
        value={queryText}
        onChange={(v) => {
          setQueryText(v);
          setFilterBarActive(false);
        }}
      />

      <QueryEditor
        value={queryText}
        onChange={(v) => {
          setQueryText(v);
          setFilterBarActive(false);
        }}
        onRun={handleRun}
        onCopyLink={handleCopyLink}
        onClear={handleClear}
      />

      <SyntaxReference />

      {response?.error && (
        <div className="mb-5 p-4 rounded-lg bg-red-900/40 border border-red-500 text-red-300 text-sm space-y-2">
          {response.query_text && (
            <pre className="font-mono text-text-normal whitespace-pre-wrap bg-scheme-shade_2 p-2 rounded element-border">
              {renderQueryWithHighlight(response.query_text, extractBadToken(response.error))}
            </pre>
          )}
          <div><strong>Error:</strong> {response.error}</div>
        </div>
      )}

      {response?.notice && (
        <div className="mb-5 p-4 rounded-lg bg-amber-900/30 border border-amber-500/60 text-amber-200 text-sm">
          {response.notice}
        </div>
      )}

      {loading && (
        <p className="mb-3 text-sm text-text-muted">Running query…</p>
      )}

      {!loading && hasQueryRun && !response?.error && (
        <>
          {response.rows.length > 0 ? (
            <p className="mb-3 text-sm text-text-muted">
              {response.row_count} row{response.row_count === 1 ? '' : 's'}
            </p>
          ) : (
            <p className="mb-3 text-sm text-text-muted">No results.</p>
          )}

          {response.chart_data && <ResultsChart data={response.chart_data} />}

          {response.rows.length > 0 && (
            <ResultsTable
              columns={response.columns}
              rows={response.rows}
              isRowLevel={response.is_row_level}
              onOpenThread={(conversationId, messageUuid) =>
                setThread({ conversationId, messageUuid })
              }
            />
          )}
        </>
      )}

      {thread && (
        <ThreadModal
          conversationId={thread.conversationId}
          focalMessageUuid={thread.messageUuid}
          onClose={() => setThread(null)}
        />
      )}

      {showHelp && <QueryBuilderHelp onClose={() => setShowHelp(false)} />}
    </div>
  );
};

export default FeedbackDashboard;