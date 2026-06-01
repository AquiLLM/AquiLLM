// useFilterBar.ts
// owns filter state for the basic query builder dropdowns
// builds the basic KQL clauses (where only — order by and limit live in useAdvancedBuilder)
// debounces feedback_text_search so queries don't fire on every keystroke
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
};

// Builds only the Basic where clauses — no order by, no limit.
// order by and limit are owned by useAdvancedBuilder.
export function buildBasicClauses(filters: FilterState, feedbackTextSearch: string): string[] {
  const clauses: string[] = [];

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

  if (feedbackTextSearch.trim()) {
    const escaped = feedbackTextSearch.trim().replace(/"/g, '\\"');
    clauses.push(`where feedback_text contains "${escaped}"`);
  }

  return clauses;
}

export function useFilterBar(): {
  filters: FilterState;
  basicKQL: string;          // debounced basic where clauses only ("messages\n| where …")
  hasActiveFilters: boolean;
  setFilter: (key: keyof FilterState, value: string) => void;
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

  // Debounced basic clauses — safe to use as auto-run trigger
  const basicClauses = buildBasicClauses(filters, debouncedTextSearch);
  const basicKQL =
    basicClauses.length === 0
      ? 'messages'
      : `messages\n${basicClauses.map(c => `| ${c}`).join('\n')}`;

  const setFilter = useCallback((key: keyof FilterState, value: string) => {
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

  const setAllFilters = useCallback((partial: Partial<FilterState>) => {
    const next = { ...EMPTY_FILTER_STATE, ...partial };
    setFilters(next);
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

  return { filters, basicKQL, hasActiveFilters, setFilter, setAllFilters, resetFilters };
}