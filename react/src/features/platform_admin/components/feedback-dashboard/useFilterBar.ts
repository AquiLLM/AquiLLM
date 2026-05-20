// useFilterBar.ts
// owns filter state for the filter bar
// builds a KQL query string from active filters
// debounces text inputs so we do not fire a query on every keystroke
// the generated KQL is used both for live display and for actual execution
import { useState, useEffect, useRef, useCallback } from 'react';

const DEBOUNCE_MS = 400;

export interface FilterState {
  date_from: string;
  date_to: string;
  user_id: string;
  exact_rating: string;
  min_rating: string;
  max_rating: string;
  feedback_text_search: string;
  role: string;
  model: string;
  tool_call_name: string;
  has_feedback_text: string;
  limit: number;
}

export const EMPTY_FILTER_STATE: FilterState = {
  date_from: '',
  date_to: '',
  user_id: '',
  exact_rating: '',
  min_rating: '',
  max_rating: '',
  feedback_text_search: '',
  role: '',
  model: '',
  tool_call_name: '',
  has_feedback_text: '',
  limit: 200,
};

export function buildKQLFromFilters(filters: FilterState): string {
  const clauses: string[] = ['messages'];

  if (filters.date_from)
    clauses.push(`where feedback_submitted_at >= "${filters.date_from}"`);
  if (filters.date_to)
    clauses.push(`where feedback_submitted_at <= "${filters.date_to}"`);

  if (filters.user_id) {
    const uid = parseInt(filters.user_id, 10);
    if (!Number.isNaN(uid)) clauses.push(`where user_id == ${uid}`);
  }

  if (filters.exact_rating) {
    const r = parseInt(filters.exact_rating, 10);
    if (!Number.isNaN(r)) clauses.push(`where rating == ${r}`);
  } else {
    if (filters.min_rating) {
      const r = parseInt(filters.min_rating, 10);
      if (!Number.isNaN(r)) clauses.push(`where rating >= ${r}`);
    }
    if (filters.max_rating) {
      const r = parseInt(filters.max_rating, 10);
      if (!Number.isNaN(r)) clauses.push(`where rating <= ${r}`);
    }
  }

  if (filters.role) clauses.push(`where role == "${filters.role}"`);
  if (filters.model)
    clauses.push(`where model == "${filters.model.replace(/"/g, '\\"')}"`);
  if (filters.tool_call_name)
    clauses.push(`where tool_call_name == "${filters.tool_call_name.replace(/"/g, '\\"')}"`);

  if (filters.has_feedback_text === 'true')
    clauses.push('where feedback_text != null');
  else if (filters.has_feedback_text === 'false')
    clauses.push('where feedback_text == null');

  if (filters.feedback_text_search.trim()) {
    const escaped = filters.feedback_text_search.trim().replace(/"/g, '\\"');
    clauses.push(`where feedback_text contains "${escaped}"`);
  }

  clauses.push('order by feedback_submitted_at desc');
  clauses.push(`limit ${filters.limit}`);

  const [first, ...rest] = clauses;
  if (rest.length === 0) return first;
  return `${first}\n${rest.map(c => `| ${c}`).join('\n')}`;
}

export function useFilterBar(): {
  filters: FilterState;
  kqlPreview: string;
  kqlForExecution: string;
  hasActiveFilters: boolean;
  setFilter: (key: keyof FilterState, value: string | number) => void;
  setAllFilters: (partial: Partial<FilterState>) => void;
  resetFilters: () => void;
} {
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTER_STATE);
  const [debouncedTextSearch, setDebouncedTextSearch] = useState('');
  const textTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (textTimer.current) clearTimeout(textTimer.current);
    textTimer.current = setTimeout(() => {
      setDebouncedTextSearch(filters.feedback_text_search);
    }, DEBOUNCE_MS);
    return () => { if (textTimer.current) clearTimeout(textTimer.current); };
  }, [filters.feedback_text_search]);

  const kqlPreview = buildKQLFromFilters(filters);

  const effectiveFilters: FilterState = {
    ...filters,
    feedback_text_search: debouncedTextSearch,
  };
  const kqlForExecution = buildKQLFromFilters(effectiveFilters);

  const setFilter = useCallback((key: keyof FilterState, value: string | number) => {
    setFilters(prev => {
      const next = { ...prev, [key]: value };
      if (key === 'exact_rating' && value !== '') {
        next.min_rating = '';
        next.max_rating = '';
      }
      if ((key === 'min_rating' || key === 'max_rating') && value !== '') {
        next.exact_rating = '';
      }
      return next;
    });
  }, []);

  // Set all filters at once (used when parsing KQL back into filter state)
  const setAllFilters = useCallback((partial: Partial<FilterState>) => {
    const next = { ...EMPTY_FILTER_STATE, ...partial };
    setFilters(next);
    // Immediately sync debounced text search to avoid a 400ms lag
    setDebouncedTextSearch(partial.feedback_text_search ?? '');
  }, []);

  const resetFilters = useCallback(() => {
    setFilters(EMPTY_FILTER_STATE);
    setDebouncedTextSearch('');
  }, []);

  const hasActiveFilters =
    filters.date_from !== '' ||
    filters.date_to !== '' ||
    filters.user_id !== '' ||
    filters.exact_rating !== '' ||
    filters.min_rating !== '' ||
    filters.max_rating !== '' ||
    filters.feedback_text_search !== '' ||
    filters.role !== '' ||
    filters.model !== '' ||
    filters.tool_call_name !== '' ||
    filters.has_feedback_text !== '';

  return {
    filters,
    kqlPreview,
    kqlForExecution,
    hasActiveFilters,
    setFilter,
    setAllFilters,
    resetFilters,
  };
}