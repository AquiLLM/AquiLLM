// useFilterBar.ts
// owns filter state for the filter bar
// builds a KQL query string from active filters
// debounces text inputs so we do not fire a query on every keystroke
// the generated KQL is used both for live display and for actual execution

import { useState, useEffect, useRef, useCallback } from 'react';

const DEBOUNCE_MS = 400;

export interface FilterState {
  // date range — applied to feedback_submitted_at
  date_from: string;        // YYYY-MM-DD or ''
  date_to: string;          // YYYY-MM-DD or ''
  // user
  user_id: string;          // numeric string or ''
  // rating
  exact_rating: string;     // '1'–'5' or ''
  min_rating: string;       // '1'–'5' or ''
  max_rating: string;       // '1'–'5' or ''
  // text search
  feedback_text_search: string;
  // categorical
  role: string;
  model: string;
  tool_call_name: string;
  has_feedback_text: string; // 'true' | 'false' | ''
  // result cap
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

// ---------------------------------------------------------------------------
// KQL builder
// ---------------------------------------------------------------------------
// builds a valid KQL query string from the current filter state
// always starts with `messages` stream
// date, user, rating, role, model, tool filters become `where` clauses
// feedback_text_search becomes a `where feedback_text contains "..."` clause
// ordering is always by feedback_submitted_at desc so most recent is first
// limit is always applied

export function buildKQLFromFilters(filters: FilterState): string {
  const clauses: string[] = ['messages'];

  // date range — feedback_submitted_at is when the user actually rated
  if (filters.date_from) {
    clauses.push(`where feedback_submitted_at >= "${filters.date_from}"`);
  }
  if (filters.date_to) {
    clauses.push(`where feedback_submitted_at <= "${filters.date_to}"`);
  }

  // user
  if (filters.user_id) {
    const uid = parseInt(filters.user_id, 10);
    if (!Number.isNaN(uid)) {
      clauses.push(`where user_id == ${uid}`);
    }
  }

  // rating — exact takes precedence over range
  if (filters.exact_rating) {
    const r = parseInt(filters.exact_rating, 10);
    if (!Number.isNaN(r)) {
      clauses.push(`where rating == ${r}`);
    }
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

  // role
  if (filters.role) {
    clauses.push(`where role == "${filters.role}"`);
  }

  // model
  if (filters.model) {
    // model names may contain hyphens and dots — safe to quote directly
    clauses.push(`where model == "${filters.model.replace(/"/g, '\\"')}"`);
  }

  // tool call name
  if (filters.tool_call_name) {
    clauses.push(`where tool_call_name == "${filters.tool_call_name.replace(/"/g, '\\"')}"`);
  }

  // has feedback text
  if (filters.has_feedback_text === 'true') {
    clauses.push('where feedback_text != null');
  } else if (filters.has_feedback_text === 'false') {
    clauses.push('where feedback_text == null');
  }

  // feedback text search — contains is case-insensitive substring match
  if (filters.feedback_text_search.trim()) {
    const escaped = filters.feedback_text_search.trim().replace(/"/g, '\\"');
    clauses.push(`where feedback_text contains "${escaped}"`);
  }

  // always sort newest rated first, cap results
  clauses.push('order by feedback_submitted_at desc');
  clauses.push(`limit ${filters.limit}`);

  // format as a readable multi-line pipeline
  const [first, ...rest] = clauses;
  if (rest.length === 0) return first;
  return `${first}\n${rest.map(c => `| ${c}`).join('\n')}`;
}

// ---------------------------------------------------------------------------
// hook
// ---------------------------------------------------------------------------

export function useFilterBar(): {
  filters: FilterState;
  kqlPreview: string;        // live KQL string that updates as user types (instant)
  kqlForExecution: string;   // debounced KQL string — only changes after pause in typing
  hasActiveFilters: boolean;
  setFilter: (key: keyof FilterState, value: string | number) => void;
  resetFilters: () => void;
} {
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTER_STATE);

  // debounced text search — only changes after user stops typing
  const [debouncedTextSearch, setDebouncedTextSearch] = useState('');
  const textTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (textTimer.current) clearTimeout(textTimer.current);
    textTimer.current = setTimeout(() => {
      setDebouncedTextSearch(filters.feedback_text_search);
    }, DEBOUNCE_MS);
    return () => { if (textTimer.current) clearTimeout(textTimer.current); };
  }, [filters.feedback_text_search]);

  // kqlPreview uses the raw (instant) text search so the display updates as you type
  const kqlPreview = buildKQLFromFilters(filters);

  // kqlForExecution uses the debounced text search so queries do not fire on every keystroke
  const effectiveFilters: FilterState = {
    ...filters,
    feedback_text_search: debouncedTextSearch,
  };
  const kqlForExecution = buildKQLFromFilters(effectiveFilters);

  const setFilter = useCallback((key: keyof FilterState, value: string | number) => {
    setFilters(prev => {
      const next = { ...prev, [key]: value };
      // when exact_rating is set, clear range rating to avoid conflict
      if (key === 'exact_rating' && value !== '') {
        next.min_rating = '';
        next.max_rating = '';
      }
      // when range rating is set, clear exact_rating
      if ((key === 'min_rating' || key === 'max_rating') && value !== '') {
        next.exact_rating = '';
      }
      return next;
    });
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
    resetFilters,
  };
}