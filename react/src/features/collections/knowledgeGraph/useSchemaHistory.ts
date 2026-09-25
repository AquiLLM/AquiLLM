import { useCallback, useState } from 'react';
import type { CollectionSchemaApi } from './collectionSchemaApi';
import type { SchemaHistoryPage } from './schemaTypes';

export function useSchemaHistory(api: CollectionSchemaApi, collectionId: string) {
  const [history, setHistory] = useState<SchemaHistoryPage | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const onLoadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    const result = await api.listVersions(collectionId);
    setHistoryLoading(false);
    if (!result.ok) {
      setHistoryError('Unable to load schema history.');
      return;
    }
    setHistory(result.data);
  }, [api, collectionId]);

  const onLoadMoreHistory = useCallback(async () => {
    if (!history?.next_cursor) return;
    setHistoryLoading(true);
    const result = await api.listVersions(collectionId, history.next_cursor);
    setHistoryLoading(false);
    if (!result.ok) {
      setHistoryError('Unable to load more schema history.');
      return;
    }
    setHistory({
      versions: [...history.versions, ...result.data.versions],
      next_cursor: result.data.next_cursor,
      has_more: result.data.has_more,
    });
  }, [api, collectionId, history]);

  return { history, historyLoading, historyError, onLoadHistory, onLoadMoreHistory };
}
