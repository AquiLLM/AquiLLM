// useFilterOptions.ts
// fetches available filter option values from the existing ORM-based filter
// options endpoint — users, models, tool names, ratings
// this endpoint is kept from the previous dashboard implementation and still works
// it is only used to populate dropdowns, not to execute queries

import { useState, useEffect } from 'react';

export interface FilterOptions {
  users: Array<{ id: number; username: string }>;
  models: string[];
  tool_names: string[];
  ratings: number[];
  roles: string[];
}

const EMPTY_OPTIONS: FilterOptions = {
  users: [],
  models: [],
  tool_names: [],
  ratings: [1, 2, 3, 4, 5],
  roles: ['assistant', 'user', 'tool'],
};

export function useFilterOptions(): {
  options: FilterOptions;
  loading: boolean;
  error: string | null;
} {
  const [options, setOptions] = useState<FilterOptions>(EMPTY_OPTIONS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const url = window.apiUrls?.api_feedback_filter_options;
    if (!url) {
      // endpoint not registered in this deployment — use hardcoded defaults
      setOptions(EMPTY_OPTIONS);
      setLoading(false);
      return;
    }

    fetch(url, { credentials: 'include' })
      .then(r => {
        if (!r.ok) throw new Error(`filter options fetch failed: ${r.status}`);
        return r.json();
      })
      .then(data => {
        if (cancelled) return;
        setOptions({
          users:      Array.isArray(data.users)      ? data.users      : [],
          models:     Array.isArray(data.models)     ? data.models     : [],
          tool_names: Array.isArray(data.tool_names) ? data.tool_names : [],
          ratings:    Array.isArray(data.ratings)    ? data.ratings    : [1, 2, 3, 4, 5],
          roles:      Array.isArray(data.roles)      ? data.roles      : ['assistant', 'user', 'tool'],
        });
      })
      .catch(err => {
        if (cancelled) return;
        // non-fatal — dropdowns fall back to hardcoded defaults
        setError(err.message ?? 'could not load filter options');
        setOptions(EMPTY_OPTIONS);
      })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, []);

  return { options, loading, error };
}