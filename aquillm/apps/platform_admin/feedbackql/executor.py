"""
FeedbackQL Executor
===================

This module takes the structured Query object produced by the parser and
actually runs it against the database, returning results as a list of dicts.

HOW IT WORKS
--------------------------------
The executor walks clauses in pipeline order, just like KQL: each stage
operates on whatever the previous stage produced. The pipeline starts as a
Django queryset against the Message table (joined to WSConversation and
User), then transforms into a list of dicts as soon as a `summarize` runs.

  - `where`     before summarize → adds a DB filter on the queryset
  - `summarize`                   → fetches rows, groups them, returns dicts
  - `where`     after summarize  → filters the in-memory dicts (can reference
                                   aggregate aliases like `avg_r` and group-by
                                   fields, since the queryset is gone)
  - `order by`  before summarize → ORM order_by on the queryset
  - `order by`  after summarize  → Python sort on the dict list (alias-aware)
  - `limit`     before summarize → DB LIMIT
  - `limit`     after summarize  → slice the dict list
  - `select`   only valid in row-level queries (no summarize)

SECURITY
--------
The executor only ever queries the Message table via Django ORM — no raw SQL.
Pre-summarize clauses must reference fields in ALLOWED_FIELDS (validated
here, since the parser now defers that check to allow alias references in
post-summarize `where` clauses). Post-summarize clauses are restricted to
aliases and group-by fields produced by the summarize stage.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from apps.chat.models import Message

from .exceptions import FeedbackQLSyntaxError
from .parser import (
    ALLOWED_FIELDS,
    Condition,
    LimitClause,
    OrderByClause,
    Query,
    SelectClause,
    SummarizeClause,
    WhereClause,
)

# ---------------------------------------------------------------------------
# Field mapping — user-facing name → Django ORM lookup path
# ---------------------------------------------------------------------------
# Most fields map directly to a column name on the Message table. Two
# exceptions need a JOIN path using Django's __ notation:
#   user_id        → conversation__owner_id  (Message → WSConversation → User)
#   conversation_id → conversation_id        (FK on Message, same name)

FIELD_MAP: dict[str, str] = {
    'rating':                'rating',
    'feedback_text':         'feedback_text',
    'feedback_submitted_at': 'feedback_submitted_at',
    'model':                 'model',
    'role':                  'role',
    'content':               'content',
    'sequence_number':       'sequence_number',
    'created_at':            'created_at',
    'message_uuid':          'message_uuid',
    'tool_call_name':        'tool_call_name',
    'user_id':               'conversation__owner_id',  # requires JOIN
    'conversation_id':       'conversation_id',
}

# Reverse mapping — used to rename ORM keys back to user-facing names in output.
# e.g. 'conversation__owner_id' → 'user_id'
_ORM_TO_FIELD = {v: k for k, v in FIELD_MAP.items()}

# Safety cap — even without a LIMIT clause we never return more than this many rows.
# Prevents accidentally dumping the entire table.
MAX_ROWS = 10_000


def _orm(field: str) -> str:
    """Convert a user-facing field name to its Django ORM lookup path."""
    return FIELD_MAP[field]  # always safe — parser already validated the name


# ---------------------------------------------------------------------------
# WHERE clause → Django Q objects
# ---------------------------------------------------------------------------
# Django's Q objects let you build filter conditions programmatically.
# We convert each Condition into a Q, then chain them with & (and) or | (or).

def _condition_to_q(cond: Condition):
    from django.db.models import Q

    if cond.field not in ALLOWED_FIELDS:
        raise FeedbackQLSyntaxError(
            f"Unknown field {cond.field!r} in pre-summarize where clause. "
            f"Allowed fields: {', '.join(sorted(ALLOWED_FIELDS))}"
        )
    path = _orm(cond.field)
    op = cond.op

    # null comparisons map to IS NULL / IS NOT NULL
    if cond.value is None:
        if op == '==':
            return Q(**{f'{path}__isnull': True})
        if op == '!=':
            return Q(**{f'{path}__isnull': False})
        raise FeedbackQLSyntaxError(f"Operator {op!r} cannot be used with null")

    if op == '==':
        return Q(**{path: cond.value})
    if op == '!=':
        return ~Q(**{path: cond.value})
    if op == '<':
        return Q(**{f'{path}__lt': cond.value})
    if op == '>':
        return Q(**{f'{path}__gt': cond.value})
    if op == '<=':
        return Q(**{f'{path}__lte': cond.value})
    if op == '>=':
        return Q(**{f'{path}__gte': cond.value})
    if op == 'startswith':
        return Q(**{f'{path}__istartswith': cond.value})   # case-insensitive
    if op == 'contains':
        return Q(**{f'{path}__icontains': cond.value})     # case-insensitive
    if op == 'in':
        return Q(**{f'{path}__in': cond.value})

    raise FeedbackQLSyntaxError(f"Unknown operator: {op!r}")  # shouldn't reach here


def _apply_where(qs, clause: WhereClause):
    """
    Apply a WhereClause to a queryset by building a chain of Q objects.

    The clause.parts list alternates: [Condition, 'and'/'or', Condition, ...]
    We combine them left-to-right using & for 'and' and | for 'or'.
    """
    parts = clause.parts
    q = _condition_to_q(parts[0])
    for i in range(1, len(parts), 2):
        connector = parts[i]
        next_q = _condition_to_q(parts[i + 1])
        q = q & next_q if connector == 'and' else q | next_q
    return qs.filter(q)


# ---------------------------------------------------------------------------
# Python-side WHERE evaluation (used for post-summarize where clauses)
# ---------------------------------------------------------------------------
# Once we've run a summarize, the queryset is gone — we have a list of dicts
# in memory. A `where` after that point has to filter the dicts directly,
# and is allowed to reference aggregate aliases like `avg_r` in addition to
# the group-by fields.

def _eval_condition_py(cond: Condition, row: dict, allowed_keys: set[str]) -> bool:
    """
    Evaluate a single Condition against an in-memory dict.

    `allowed_keys` is the set of field names this condition is allowed to
    reference (group-by fields + aggregation aliases produced by summarize).
    """
    if cond.field not in allowed_keys:
        raise FeedbackQLSyntaxError(
            f"Unknown field {cond.field!r} in post-summarize where clause. "
            f"Available fields: {', '.join(sorted(allowed_keys))}"
        )
    lhs = row.get(cond.field)
    op = cond.op
    rhs = cond.value

    # null comparisons
    if rhs is None:
        if op == '==':
            return lhs is None
        if op == '!=':
            return lhs is not None
        raise FeedbackQLSyntaxError(f"Operator {op!r} cannot be used with null")

    # null on the left side never matches anything except the null comparisons above
    if lhs is None:
        return False

    if op == '==':
        return lhs == rhs
    if op == '!=':
        return lhs != rhs
    if op == '<':
        return lhs < rhs
    if op == '>':
        return lhs > rhs
    if op == '<=':
        return lhs <= rhs
    if op == '>=':
        return lhs >= rhs
    if op == 'startswith':
        return isinstance(lhs, str) and lhs.lower().startswith(str(rhs).lower())
    if op == 'contains':
        return isinstance(lhs, str) and str(rhs).lower() in lhs.lower()
    if op == 'in':
        return lhs in rhs

    raise FeedbackQLSyntaxError(f"Unknown operator: {op!r}")


def _apply_where_py(rows: list[dict], clause: WhereClause, allowed_keys: set[str]) -> list[dict]:
    """
    Filter an in-memory list of dicts by a WhereClause.

    Mirrors `_apply_where` (which works on a Django queryset) but operates
    on Python objects so it can reference aggregate aliases.
    """
    parts = clause.parts

    def matches(row: dict) -> bool:
        result = _eval_condition_py(parts[0], row, allowed_keys)
        for i in range(1, len(parts), 2):
            connector = parts[i]
            next_result = _eval_condition_py(parts[i + 1], row, allowed_keys)
            result = (result and next_result) if connector == 'and' else (result or next_result)
        return result

    return [r for r in rows if matches(r)]


# ---------------------------------------------------------------------------
# Python-side aggregation
# ---------------------------------------------------------------------------
# We fetch rows from the DB and compute aggregations in Python. This is
# slightly less efficient than doing it all in SQL, but it means we can
# support `median` without needing PostgreSQL-specific functions, and the
# code stays simple and easy to reason about.
#
# For our use case (feedback data, not millions of rows) this is fine.

def _compute_agg(func: str, values: list) -> Any:
    """
    Compute an aggregation over a list of values.

    `count` counts all values including None.
    All other functions ignore None values (consistent with SQL behaviour).
    Returns None if there are no non-null values to aggregate.
    """
    if func == 'count':
        return len(values)
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    if func == 'avg':
        return sum(cleaned) / len(cleaned)
    if func == 'min':
        return min(cleaned)
    if func == 'max':
        return max(cleaned)
    if func == 'sum':
        return sum(cleaned)
    if func == 'median':
        return statistics.median(cleaned)
    raise FeedbackQLSyntaxError(f"Unknown aggregation: {func!r}")


# ---------------------------------------------------------------------------
# Row-level finalisation (no summarize ran)
# ---------------------------------------------------------------------------

# These fields are always fetched and included in row-level results so the
# conversation viewer can open the full thread for any result row, even when
# the user's query didn't explicitly select them.
_THREAD_META_FIELDS: frozenset[str] = frozenset({'conversation_id', 'message_uuid'})


def _finalise_row_level(qs, select: SelectClause | None) -> list[dict]:
    """
    Materialise the queryset into a list of dicts with user-facing field names.

    By the time we get here, all where / order_by / limit operations on the
    queryset have already been applied. We just need to project the right
    columns and remap ORM paths back to user-facing names.
    """
    output_fields = select.fields if select else sorted(ALLOWED_FIELDS)

    # Always fetch thread metadata fields for the conversation viewer
    fetch_fields = set(output_fields) | _THREAD_META_FIELDS
    orm_fields = [_orm(f) for f in fetch_fields]

    # If no LIMIT was applied to the queryset, cap at MAX_ROWS as a safety net
    rows = list(qs.values(*orm_fields)[:MAX_ROWS])

    output_field_set = set(output_fields)
    result = []
    for row in rows:
        remapped = {_ORM_TO_FIELD.get(k, k): v for k, v in row.items()}
        result.append({
            k: v for k, v in remapped.items()
            if k in output_field_set or k in _THREAD_META_FIELDS
        })

    return result


# ---------------------------------------------------------------------------
# Summarize (aggregate) execution
# ---------------------------------------------------------------------------

def _run_summarize(qs, summarize: SummarizeClause) -> list[dict]:
    """
    Fetch the queryset, group it in Python, and compute aggregations per group.

    Returns a list of dicts where each dict's keys are the group-by field
    names plus the aggregation aliases. From this point onward in the
    pipeline we work with this list rather than the queryset.
    """
    by_fields = summarize.by
    agg_fields_needed = {agg.agg_field for agg in summarize.aggregations if agg.agg_field}
    fetch_fields = set(by_fields) | agg_fields_needed
    orm_fields = [_orm(f) for f in fetch_fields]

    rows = list(qs.values(*orm_fields)[:MAX_ROWS])

    if not rows:
        return []

    if by_fields:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            key = tuple(row[_orm(f)] for f in by_fields)
            groups[key].append(row)

        result = []
        for key, group_rows in groups.items():
            out: dict = {by_fields[i]: key[i] for i in range(len(by_fields))}
            for agg in summarize.aggregations:
                if agg.func == 'count' and not agg.agg_field:
                    out[agg.alias] = len(group_rows)
                else:
                    values = [r[_orm(agg.agg_field)] for r in group_rows]  # type: ignore[arg-type]
                    out[agg.alias] = _compute_agg(agg.func, values)
            result.append(out)
        return result

    # No 'by' fields — one global aggregate row over all data
    out = {}
    for agg in summarize.aggregations:
        if agg.func == 'count' and not agg.agg_field:
            out[agg.alias] = len(rows)
        else:
            values = [r[_orm(agg.agg_field)] for r in rows]  # type: ignore[arg-type]
            out[agg.alias] = _compute_agg(agg.func, values)
    return [out]


# ---------------------------------------------------------------------------
# Public entry point — pipeline-order execution
# ---------------------------------------------------------------------------

def execute(query: Query) -> list[dict]:
    """
    Execute a parsed FeedbackQL Query by walking its clauses in pipeline order.

    The pipeline starts as a Django queryset and stays a queryset until a
    `summarize` runs, at which point it becomes a list of in-memory dicts.
    Subsequent `where`/`order by`/`limit` clauses operate on whatever shape
    the previous stage left behind.

    Always queries the Message table, joined to WSConversation and User.
    Never executes raw SQL — all database access goes through Django ORM.
    """
    # Pre-summarize state: a Django queryset
    qs = Message.objects.select_related('conversation', 'conversation__owner')

    # Post-summarize state: a list of dicts (None until summarize runs)
    rows: list[dict] | None = None

    # The set of field names a post-summarize where/order can reference
    # (group-by fields + aggregation aliases). Empty until summarize runs.
    post_summarize_keys: set[str] = set()

    select_clause: SelectClause | None = None

    for clause in query.clauses:
        if isinstance(clause, WhereClause):
            if rows is None:
                qs = _apply_where(qs, clause)
            else:
                rows = _apply_where_py(rows, clause, post_summarize_keys)

        elif isinstance(clause, SelectClause):
            if rows is not None:
                raise FeedbackQLSyntaxError(
                    "Cannot use 'select' after 'summarize' — use the aggregation "
                    "aliases in the summarize clause to control output columns"
                )
            if select_clause is not None:
                raise FeedbackQLSyntaxError("Only one 'select' clause is allowed")
            select_clause = clause

        elif isinstance(clause, SummarizeClause):
            if rows is not None:
                raise FeedbackQLSyntaxError("Only one 'summarize' clause is allowed")
            if select_clause is not None:
                raise FeedbackQLSyntaxError(
                    "Cannot use both 'select' and 'summarize' in the same query"
                )
            rows = _run_summarize(qs, clause)
            post_summarize_keys = set(clause.by) | {agg.alias for agg in clause.aggregations}

        elif isinstance(clause, OrderByClause):
            if rows is None:
                if clause.field not in ALLOWED_FIELDS:
                    raise FeedbackQLSyntaxError(
                        f"Unknown field {clause.field!r} in order by. "
                        f"Allowed fields: {', '.join(sorted(ALLOWED_FIELDS))}"
                    )
                prefix = '-' if clause.direction == 'desc' else ''
                qs = qs.order_by(f'{prefix}{_orm(clause.field)}')
            else:
                if clause.field not in post_summarize_keys:
                    raise FeedbackQLSyntaxError(
                        f"Unknown field {clause.field!r} in post-summarize order by. "
                        f"Available fields: {', '.join(sorted(post_summarize_keys))}"
                    )
                reverse = clause.direction == 'desc'
                rows.sort(
                    # Put None values last regardless of direction
                    key=lambda r: (r.get(clause.field) is None, r.get(clause.field)),
                    reverse=reverse,
                )

        elif isinstance(clause, LimitClause):
            if rows is None:
                qs = qs[:clause.n]
            else:
                rows = rows[:clause.n]

    if rows is not None:
        return rows

    return _finalise_row_level(qs, select_clause)
