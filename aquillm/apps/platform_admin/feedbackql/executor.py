"""
FeedbackQL Executor
===================

This module takes the structured Query object produced by the parser and
actually runs it against the database, returning results as a list of dicts.

HOW IT WORKS  
--------------------------------
The executor walks the list of clauses in the Query and translates each one
into Django ORM calls against the Message table (joined to WSConversation
and User as needed).

There are two execution paths:

  1. Row-level queries (no summarize clause)
     Each result dict is one message row, e.g.:
       {'rating': 3, 'model': 'gemma-2', 'user_id': 42, ...}

  2. Aggregate queries (with a summarize clause)
     Each result dict is a computed summary, e.g.:
       {'model': 'gemma-2', 'avg_rating': 2.7, 'count': 15}

     Aggregations are computed in Python after fetching the raw rows.
     This handles `median` cleanly (PostgreSQL's PERCENTILE_DISC is awkward
     to use with Django ORM), and our dataset is small enough that it's fine.

SECURITY
--------
The executor only ever queries the Message table via Django ORM — no raw SQL.
Field validation already happened in the parser (before we got here), so we
can trust that every field name in the AST is in ALLOWED_FIELDS.
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
# Row-level execution path (no summarize clause)
# ---------------------------------------------------------------------------

def _execute_row_level(
    qs,
    select: SelectClause | None,
    order: OrderByClause | None,
    limit: LimitClause | None,
) -> list[dict]:
    """
    Fetch individual message rows, applying select / order / limit.

    If no select clause, all allowed fields are returned.
    The order field is included in the DB fetch even if not in the select list
    (so the DB can sort correctly), then stripped from the output if needed.
    """
    output_fields = select.fields if select else sorted(ALLOWED_FIELDS)

    # Make sure the order field is fetched from the DB even if it's not in
    # the select list — we need it for sorting, then strip it from output.
    fetch_fields = set(output_fields)
    if order:
        fetch_fields.add(order.field)

    orm_fields = [_orm(f) for f in fetch_fields]

    # .values() tells Django to return dicts instead of model objects,
    # fetching only the columns we actually need.
    qs = qs.values(*orm_fields)

    if order:
        prefix = '-' if order.direction == 'desc' else ''
        qs = qs.order_by(f'{prefix}{_orm(order.field)}')

    cap = min(limit.n, MAX_ROWS) if limit else MAX_ROWS
    rows = list(qs[:cap])

    # Django returns ORM paths as keys (e.g. 'conversation__owner_id').
    # Remap them to user-facing names (e.g. 'user_id'), then drop any
    # fields that were fetched for ordering but not in the select list.
    output_field_set = set(output_fields)
    result = []
    for row in rows:
        remapped = {_ORM_TO_FIELD.get(k, k): v for k, v in row.items()}
        result.append({k: v for k, v in remapped.items() if k in output_field_set})

    return result


# ---------------------------------------------------------------------------
# Summarize (aggregate) execution path
# ---------------------------------------------------------------------------

def _execute_summarize(
    qs,
    summarize: SummarizeClause,
    order: OrderByClause | None,
    limit: LimitClause | None,
) -> list[dict]:
    """
    Fetch raw rows, group them in Python, and compute aggregations per group.

    If `summarize.by` is empty, all rows are treated as one group and we
    return a single result dict with global aggregates (e.g. overall avg rating).

    If `summarize.by` has fields, we group by those fields (like SQL GROUP BY)
    and return one result dict per group.
    """
    by_fields = summarize.by
    # Collect only the fields we actually need from the DB
    agg_fields_needed = {agg.agg_field for agg in summarize.aggregations if agg.agg_field}
    fetch_fields = set(by_fields) | agg_fields_needed
    orm_fields = [_orm(f) for f in fetch_fields]

    rows = list(qs.values(*orm_fields)[:MAX_ROWS])

    if not rows:
        return []

    if by_fields:
        # Group rows by the 'by' fields using a dict keyed on a tuple of values
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            key = tuple(row[_orm(f)] for f in by_fields)
            groups[key].append(row)

        result = []
        for key, group_rows in groups.items():
            # Start with the group-by field values
            out: dict = {by_fields[i]: key[i] for i in range(len(by_fields))}
            # Add each aggregation
            for agg in summarize.aggregations:
                if agg.func == 'count' and not agg.agg_field:
                    out[agg.alias] = len(group_rows)
                else:
                    values = [r[_orm(agg.agg_field)] for r in group_rows]  # type: ignore[arg-type]
                    out[agg.alias] = _compute_agg(agg.func, values)
            result.append(out)
    else:
        # No 'by' fields — compute one global aggregate over all rows
        out = {}
        for agg in summarize.aggregations:
            if agg.func == 'count' and not agg.agg_field:
                out[agg.alias] = len(rows)
            else:
                values = [r[_orm(agg.agg_field)] for r in rows]  # type: ignore[arg-type]
                out[agg.alias] = _compute_agg(agg.func, values)
        result = [out]

    # Apply ordering — the order field may be a summarize alias, not a DB field,
    # so we sort the Python list rather than using ORM order_by.
    if order:
        reverse = order.direction == 'desc'
        result.sort(
            # Put None values last regardless of direction
            key=lambda r: (r.get(order.field) is None, r.get(order.field)),
            reverse=reverse,
        )

    if limit:
        result = result[:limit.n]

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute(query: Query) -> list[dict]:
    """
    Execute a parsed FeedbackQL Query and return results as a list of dicts.

    Always queries the Message table, joined to WSConversation and User.
    Never executes raw SQL — all database access goes through Django ORM.

    Returns:
        For row-level queries: one dict per message row.
        For summarize queries: one dict per group (or one dict total for global aggs).
    """
    # Base queryset — always Message, always with the conversation + owner joined
    qs = Message.objects.select_related('conversation', 'conversation__owner')

    # Collect clauses by type
    where_clauses: list[WhereClause] = []
    select_clause: SelectClause | None = None
    summarize_clause: SummarizeClause | None = None
    order_clause: OrderByClause | None = None
    limit_clause: LimitClause | None = None

    for clause in query.clauses:
        if isinstance(clause, WhereClause):
            where_clauses.append(clause)       # multiple where clauses are ANDed together
        elif isinstance(clause, SelectClause):
            select_clause = clause
        elif isinstance(clause, SummarizeClause):
            summarize_clause = clause
        elif isinstance(clause, OrderByClause):
            order_clause = clause
        elif isinstance(clause, LimitClause):
            limit_clause = clause

    # Apply all where filters to the queryset
    for w in where_clauses:
        qs = _apply_where(qs, w)

    if summarize_clause and select_clause:
        raise FeedbackQLSyntaxError(
            "Cannot use both 'select' and 'summarize' in the same query"
        )

    if summarize_clause:
        return _execute_summarize(qs, summarize_clause, order_clause, limit_clause)

    return _execute_row_level(qs, select_clause, order_clause, limit_clause)
