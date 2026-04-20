import React, { useEffect, useState } from 'react';
import QueryEditor from './QueryEditor';
import SyntaxReference from './SyntaxReference';
import ResultsChart from './ResultsChart';
import ResultsTable from './ResultsTable';
import ThreadModal from './ThreadModal';
import { b64decode, b64encode, runQuery } from './api';
import type { QueryResponse } from './types';

function readQueryFromUrl(): string {
  const params = new URLSearchParams(window.location.search);
  const q = params.get('q');
  return q ? b64decode(q) : '';
}

function buildShareableUrl(queryText: string): string {
  const encoded = b64encode(queryText.trim());
  return `${window.location.origin}${window.location.pathname}?q=${encodeURIComponent(encoded)}`;
}

const FeedbackDashboard: React.FC = () => {
  const [queryText, setQueryText] = useState<string>(() => readQueryFromUrl());
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [thread, setThread] = useState<{ conversationId: string; messageUuid: string } | null>(null);

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

  // Run initial query if URL has ?q=
  useEffect(() => {
    if (queryText.trim()) {
      void executeQuery(queryText);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // React to browser back/forward — reparse the URL and rerun
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

  const handleRun = () => {
    const trimmed = queryText.trim();
    if (!trimmed) return;
    const url = buildShareableUrl(trimmed);
    window.history.pushState({}, '', url);
    void executeQuery(trimmed);
  };

  const handleCopyLink = () => {
    const trimmed = queryText.trim();
    if (!trimmed) {
      alert('Type a query first.');
      return;
    }
    void navigator.clipboard.writeText(buildShareableUrl(trimmed));
  };

  const hasQueryRun = response !== null;

  return (
    <div className="container mx-auto p-6 text-text-normal max-w-7xl">
      <h1 className="text-3xl font-bold mb-6">Feedback Dashboard</h1>

      <QueryEditor
        value={queryText}
        onChange={setQueryText}
        onRun={handleRun}
        onCopyLink={handleCopyLink}
      />

      <SyntaxReference />

      {response?.error && (
        <div className="mb-5 p-4 rounded-lg bg-red-900/40 border border-red-500 text-red-300 text-sm">
          <strong>Error:</strong> {response.error}
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
    </div>
  );
};

export default FeedbackDashboard;
