// useAdvancedBuilder.ts
// owns the Advanced Query Builder's clause state
// each clause type (where, summarize, select, order by, limit) is a proper list with IDs
// the hook is the source of truth for the Advanced section;
// the Textual Query textarea is derived from Basic + Advanced combined.
import { useState, useCallback } from 'react';

const NUMERIC_FIELDS = new Set(['rating', 'sequence_number', 'user_id', 'conversation_id']);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type WhereRow = {
  id: string;
  field: string;
  op: string;
  value: string;
};

export type SummarizeRow = {
  id: string;
  alias: string;
  func: string;
  field: string;    // empty string for count()
  byField: string;  // empty string for no grouping
};

export type OrderByRow = {
  id: string;
  field: string;
  dir: 'asc' | 'desc';
};

export type AdvancedState = {
  whereRows: WhereRow[];
  summarizeRows: SummarizeRow[];
  selectFields: string[];
  orderByRows: OrderByRow[];
  limitValue: number;
};

// Used when parsing KQL back into Advanced state
export type ParsedAdvanced = {
  whereRows: Omit<WhereRow, 'id'>[];
  summarizeRows: Omit<SummarizeRow, 'id'>[];
  selectFields: string[];
  orderByRows: Omit<OrderByRow, 'id'>[];
  limitValue: number | null;
};

// ---------------------------------------------------------------------------
// ID generator
// ---------------------------------------------------------------------------

let _idCounter = 0;
function nextId(): string {
  return `adv-${++_idCounter}`;
}

// ---------------------------------------------------------------------------
// KQL builders
// ---------------------------------------------------------------------------

function buildWhereClause(row: WhereRow): string {
  if (row.op === '== null' || row.op === '!= null') {
    return `where ${row.field} ${row.op}`;
  }
  if (row.op === 'in') {
    const parts = row.value.split(',').map(s => s.trim()).filter(Boolean);
    const items = parts.map(p =>
      NUMERIC_FIELDS.has(row.field) ? p : `"${p.replace(/"/g, '\\"')}"`
    );
    return `where ${row.field} in [${items.join(', ')}]`;
  }
  if (row.op === 'contains' || row.op === 'startswith') {
    return `where ${row.field} ${row.op} "${row.value.replace(/"/g, '\\"')}"`;
  }
  const val = NUMERIC_FIELDS.has(row.field)
    ? row.value.trim()
    : `"${row.value.trim().replace(/"/g, '\\"')}"`;
  return `where ${row.field} ${row.op} ${val}`;
}

function buildSummarizeClause(row: SummarizeRow): string {
  const inner = row.func === 'count' ? '' : row.field;
  let clause = `summarize ${row.alias} = ${row.func}(${inner})`;
  if (row.byField) clause += ` by ${row.byField}`;
  return clause;
}

// Returns an array of pipe-clause strings (no "messages" prefix, no leading "|")
export function buildAdvancedClauses(state: AdvancedState): string[] {
  const clauses: string[] = [];

  for (const row of state.whereRows) {
    clauses.push(buildWhereClause(row));
  }
  for (const row of state.summarizeRows) {
    clauses.push(buildSummarizeClause(row));
  }
  if (state.selectFields.length > 0) {
    clauses.push(`select ${state.selectFields.join(', ')}`);
  }
  // After summarize, original stream fields like feedback_submitted_at no longer
  // exist in the result set. Replace any order by on that field with an order by
  // the first summarize alias (e.g. "n") so results are still meaningfully sorted.
  let effectiveOrderBy = state.orderByRows;
  if (state.summarizeRows.length > 0) {
    const filtered = state.orderByRows.filter(r => r.field !== 'feedback_submitted_at');
    if (filtered.length === 0) {
      // No valid order by left — default to the first summarize alias descending
      const defaultAlias = state.summarizeRows[0]?.alias || 'n';
      effectiveOrderBy = [{ id: 'auto-order', field: defaultAlias, dir: 'desc' }];
    } else {
      effectiveOrderBy = filtered;
    }
  }

  for (const row of effectiveOrderBy) {
    clauses.push(`order by ${row.field} ${row.dir}`);
  }
  clauses.push(`limit ${state.limitValue}`);

  return clauses;
}

// ---------------------------------------------------------------------------
// Default state — always includes a sensible order by and limit so queries
// are never accidentally unordered or unbounded.
// ---------------------------------------------------------------------------

const DEFAULT_ORDER_ROW: OrderByRow = {
  id: 'default-order',
  field: 'feedback_submitted_at',
  dir: 'desc',
};

const INITIAL_STATE: AdvancedState = {
  whereRows: [],
  summarizeRows: [],
  selectFields: [],
  orderByRows: [DEFAULT_ORDER_ROW],
  limitValue: 200,
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAdvancedBuilder() {
  const [state, setState] = useState<AdvancedState>(INITIAL_STATE);

  // WHERE
  const addWhere = useCallback((field: string, op: string, value: string) => {
    setState(prev => ({
      ...prev,
      whereRows: [...prev.whereRows, { id: nextId(), field, op, value }],
    }));
  }, []);

  const removeWhere = useCallback((id: string) => {
    setState(prev => ({ ...prev, whereRows: prev.whereRows.filter(r => r.id !== id) }));
  }, []);

  // SUMMARIZE
  const addSummarize = useCallback(
    (alias: string, func: string, field: string, byField: string) => {
      setState(prev => ({
        ...prev,
        summarizeRows: [...prev.summarizeRows, { id: nextId(), alias, func, field, byField }],
      }));
    },
    [],
  );

  const removeSummarize = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      summarizeRows: prev.summarizeRows.filter(r => r.id !== id),
    }));
  }, []);

  // SELECT (toggle individual fields)
  const toggleSelect = useCallback((field: string) => {
    setState(prev => ({
      ...prev,
      selectFields: prev.selectFields.includes(field)
        ? prev.selectFields.filter(f => f !== field)
        : [...prev.selectFields, field],
    }));
  }, []);

  const removeSelect = useCallback((field: string) => {
    setState(prev => ({
      ...prev,
      selectFields: prev.selectFields.filter(f => f !== field),
    }));
  }, []);

  // ORDER BY
  const addOrderBy = useCallback((field: string, dir: 'asc' | 'desc') => {
    setState(prev => ({
      ...prev,
      orderByRows: [...prev.orderByRows, { id: nextId(), field, dir }],
    }));
  }, []);

  const removeOrderBy = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      orderByRows: prev.orderByRows.filter(r => r.id !== id),
    }));
  }, []);

  // LIMIT
  const setLimitValue = useCallback((n: number) => {
    setState(prev => ({ ...prev, limitValue: n }));
  }, []);

  // Set entire Advanced state from parsed KQL (used when textarea is edited)
  const setFromParsed = useCallback((parsed: ParsedAdvanced) => {
    setState({
      whereRows: parsed.whereRows.map(r => ({ ...r, id: nextId() })),
      summarizeRows: parsed.summarizeRows.map(r => ({ ...r, id: nextId() })),
      selectFields: parsed.selectFields,
      orderByRows:
        parsed.orderByRows.length > 0
          ? parsed.orderByRows.map(r => ({ ...r, id: nextId() }))
          : [{ ...DEFAULT_ORDER_ROW, id: nextId() }],
      limitValue: parsed.limitValue ?? 200,
    });
  }, []);

  const reset = useCallback(() => {
    setState({
      ...INITIAL_STATE,
      orderByRows: [{ ...DEFAULT_ORDER_ROW, id: nextId() }],
    });
  }, []);

  return {
    state,
    addWhere,
    removeWhere,
    addSummarize,
    removeSummarize,
    toggleSelect,
    removeSelect,
    addOrderBy,
    removeOrderBy,
    setLimitValue,
    setFromParsed,
    reset,
  };
}