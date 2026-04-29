import { useState, useEffect, useRef } from 'react';
import {
  FilterState,
  FilterOptions,
  SummaryMetrics,
  RowsResponse,
  FeedbackRow,
} from './types';

// text search fields are debounced so we don't fire a request on every keystroke
const DEBOUNCE_MS = 400;

function buildQueryString(filters: FilterState): string {
  const params = new URLSearchParams();

  if (filters.start_date)            params.set('start_date',             filters.start_date);
  if (filters.end_date)              params.set('end_date',               filters.end_date);
  if (filters.user_id)               params.set('user_id',                filters.user_id);
  if (filters.exact_rating)          params.set('exact_rating',           filters.exact_rating);
  else {
    if (filters.min_rating)          params.set('min_rating',             filters.min_rating);
    if (filters.max_rating)          params.set('max_rating',             filters.max_rating);
  }
  if (filters.feedback_text_search)  params.set('feedback_text_search',   filters.feedback_text_search);
  if (filters.conversation_name_search) params.set('conversation_name_search', filters.conversation_name_search);
  if (filters.role)                  params.set('role',                   filters.role);
  if (filters.model)                 params.set('model',                  filters.model);
  if (filters.tool_call_name)        params.set('tool_call_name',         filters.tool_call_name);
  if (filters.has_feedback_text)     params.set('has_feedback_text',      filters.has_feedback_text);

  return params.toString();
}

export interface UseFilteredDataResult {
  summary: SummaryMetrics | null;
  rows: FeedbackRow[];
  filterOptions: FilterOptions | null;
  totalCount: number;
  totalPages: number;
  currentPage: number;
  // canonical PRQL string returned by the rows endpoint
  // represents the current active filter state
  prql: string;
  loadingRows: boolean;
  loadingSummary: boolean;
  loadingOptions: boolean;
  errorRows: string | null;
  errorSummary: string | null;
  errorOptions: string | null;
  exportQueryString: string;
}

export function useFilteredData(
  filters: FilterState,
  apiRows: string,
  apiSummary: string,
  apiFilters: string,
): UseFilteredDataResult {
  const [summary, setSummary]             = useState<SummaryMetrics | null>(null);
  const [rows, setRows]                   = useState<FeedbackRow[]>([]);
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [totalCount, setTotalCount]       = useState(0);
  const [totalPages, setTotalPages]       = useState(1);
  const [currentPage, setCurrentPage]     = useState(1);
  const [prql, setPrql]                   = useState('');

  const [loadingRows, setLoadingRows]         = useState(false);
  const [loadingSummary, setLoadingSummary]   = useState(false);
  const [loadingOptions, setLoadingOptions]   = useState(false);
  const [errorRows, setErrorRows]             = useState<string | null>(null);
  const [errorSummary, setErrorSummary]       = useState<string | null>(null);
  const [errorOptions, setErrorOptions]       = useState<string | null>(null);

  // debounced version of the text search fields so we only fire after the
  // user stops typing
  const [debouncedTextSearch, setDebouncedTextSearch] = useState(filters.feedback_text_search);
  const [debouncedConvoSearch, setDebouncedConvoSearch] = useState(filters.conversation_name_search);

  const textTimer  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const convoTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (textTimer.current) clearTimeout(textTimer.current);
    textTimer.current = setTimeout(() => {
      setDebouncedTextSearch(filters.feedback_text_search);
    }, DEBOUNCE_MS);
    return () => { if (textTimer.current) clearTimeout(textTimer.current); };
  }, [filters.feedback_text_search]);

  useEffect(() => {
    if (convoTimer.current) clearTimeout(convoTimer.current);
    convoTimer.current = setTimeout(() => {
      setDebouncedConvoSearch(filters.conversation_name_search);
    }, DEBOUNCE_MS);
    return () => { if (convoTimer.current) clearTimeout(convoTimer.current); };
  }, [filters.conversation_name_search]);

  // the effective filters object — swaps raw text fields for debounced ones
  // so pagination, dropdowns, and date pickers still fire immediately while
  // text inputs wait for the user to stop typing
  const effectiveFilters: FilterState = {
    ...filters,
    feedback_text_search: debouncedTextSearch,
    conversation_name_search: debouncedConvoSearch,
  };

  const queryString = buildQueryString(effectiveFilters);

  // fetch rows — also captures the canonical PRQL from the response
  useEffect(() => {
    let cancelled = false;
    setLoadingRows(true);
    setErrorRows(null);

    const rowsParams = new URLSearchParams(queryString);
    rowsParams.set('page',      String(filters.page));
    rowsParams.set('page_size', String(filters.page_size));

    fetch(`${apiRows}?${rowsParams.toString()}`, { credentials: 'include' })
      .then(r => {
        if (!r.ok) throw new Error(`rows fetch failed: ${r.status}`);
        return r.json() as Promise<RowsResponse>;
      })
      .then(data => {
        if (cancelled) return;
        setRows(data.rows);
        setTotalCount(data.total_count);
        setTotalPages(data.total_pages);
        setCurrentPage(data.page);
        // store the canonical PRQL returned by the backend
        // this is what gets displayed live under the filters
        if (data.prql) setPrql(data.prql);
      })
      .catch(err => {
        if (cancelled) return;
        setErrorRows(err.message ?? 'error loading rows');
      })
      .finally(() => { if (!cancelled) setLoadingRows(false); });

    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryString, filters.page, filters.page_size, apiRows]);


  useEffect(() => {
    let cancelled = false;
    setLoadingSummary(true);
    setErrorSummary(null);

    fetch(`${apiSummary}?${queryString}`, { credentials: 'include' })
      .then(r => {
        if (!r.ok) throw new Error(`summary fetch failed: ${r.status}`);
        return r.json() as Promise<SummaryMetrics>;
      })
      .then(data => {
        if (cancelled) return;
        setSummary(data);
      })
      .catch(err => {
        if (cancelled) return;
        setErrorSummary(err.message ?? 'error loading summary');
      })
      .finally(() => { if (!cancelled) setLoadingSummary(false); });

    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryString, apiSummary]);

  // fetch filter options once on mount — never changes
  useEffect(() => {
    let cancelled = false;
    setLoadingOptions(true);
    setErrorOptions(null);

    fetch(apiFilters, { credentials: 'include' })
      .then(r => {
        if (!r.ok) throw new Error(`filter options fetch failed: ${r.status}`);
        return r.json() as Promise<FilterOptions>;
      })
      .then(data => {
        if (cancelled) return;
        setFilterOptions(data);
      })
      .catch(err => {
        if (cancelled) return;
        setErrorOptions(err.message ?? 'error loading filter options');
      })
      .finally(() => { if (!cancelled) setLoadingOptions(false); });

    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiFilters]);

  return {
    summary,
    rows,
    filterOptions,
    totalCount,
    totalPages,
    currentPage,
    prql,
    loadingRows,
    loadingSummary,
    loadingOptions,
    errorRows,
    errorSummary,
    errorOptions,
    exportQueryString: queryString,
  };
}
