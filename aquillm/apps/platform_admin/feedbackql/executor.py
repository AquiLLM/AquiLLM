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
from dataclasses import dataclass, field
from typing import Any, Callable

from apps.chat.models import Message, WSConversation

from .exceptions import FeedbackQLSyntaxError
from .parser import (
    ALLOWED_FIELDS,
    CONVERSATIONS_FIELDS,
    MESSAGES_FIELDS,
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

# Field mapping for the `messages` stream — most fields map directly to a
# column on the Message table. Two need a JOIN path with Django's __ notation:
#   user_id        → conversation__owner_id  (Message → WSConversation → User)
#   conversation_id → conversation_id        (FK on Message, same name)
MESSAGES_FIELD_MAP: dict[str, str] = {
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

# Field mapping for the `conversations` stream. Native columns on WSConversation
# map to themselves; derived fields (computed via .annotate() in the queryset
# builder) are also referenced by their annotation name.
CONVERSATIONS_FIELD_MAP: dict[str, str] = {
    # Native columns
    'conversation_id': 'id',
    'user_id':         'owner_id',
    'name':            'name',
    'created_at':      'created_at',
    'updated_at':      'updated_at',
    # Derived (annotation names — set by _build_conversations_queryset)
    'message_count':   'message_count',
    'rated_count':     'rated_count',
    'avg_rating':      'avg_rating',
    'min_rating':      'min_rating',
    'max_rating':      'max_rating',
    'last_rated_at':   'last_rated_at',
    'tools_used':      'tools_used',
}

# Backwards-compat: a couple of imports still expect FIELD_MAP referring to messages.
FIELD_MAP = MESSAGES_FIELD_MAP

# Virtual fields exist in ALLOWED_FIELDS but have no DB column. They are
# computed in Python after fetching by looking up values per conversation.
# Currently only the `messages` stream has one — `conversation_tool`.
MESSAGES_VIRTUAL_FIELDS = frozenset({'conversation_tool'})
CONVERSATIONS_VIRTUAL_FIELDS: frozenset[str] = frozenset()
# Backwards-compat for code paths expecting the old global name.
VIRTUAL_FIELDS = MESSAGES_VIRTUAL_FIELDS

# Safety cap — even without a LIMIT clause we never return more than this many rows.
# Prevents accidentally dumping the entire table.
MAX_ROWS = 10_000


# ---------------------------------------------------------------------------
# Stream configuration
# ---------------------------------------------------------------------------
# Each stream packages up the per-stream knobs the executor needs: the field
# whitelist, the user-name → ORM-path map, which fields are virtual, and a
# callable that builds the base queryset (with annotations for derived fields).
# All other helpers in this module are stream-agnostic — they receive a
# StreamConfig and look up what they need on it.

@dataclass
class StreamConfig:
    name: str
    allowed_fields: frozenset
    field_map: dict[str, str]
    virtual_fields: frozenset
    base_qs: Callable[[], Any]
    # Reverse mapping from ORM lookup path → user-facing field name, used to
    # rename keys back when materialising results.
    orm_to_field: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.orm_to_field = {v: k for k, v in self.field_map.items()}

    def orm(self, name: str) -> str:
        """user-facing field name → Django ORM lookup path."""
        return self.field_map[name]


def _build_messages_queryset():
    return Message.objects.select_related('conversation', 'conversation__owner')


def _build_conversations_queryset():
    """
    Base queryset for the `conversations` stream — one row per conversation,
    with derived aggregate fields annotated from the messages relation.

    All annotations are pure Django ORM (Count/Avg/Min/Max/Subquery + a
    PostgreSQL StringAgg for the tools list) so the resulting SQL stays
    parameterised and safe.
    """
    from django.db.models import Avg, Count, Max, Min, OuterRef, Q, Subquery, Value
    from django.db.models.aggregates import StringAgg

    # Per-conversation distinct list of tools used, comma-separated.
    # Using a correlated subquery so each conversation gets its own list,
    # rather than mixing aggregations that would interact poorly with the
    # other Count/Avg annotations.
    tools_subquery = (
        Message.objects
        .filter(conversation_id=OuterRef('pk'), tool_call_name__isnull=False)
        .order_by()
        .values('conversation_id')
        .annotate(tools=StringAgg('tool_call_name', delimiter=Value(', '), distinct=True))
        .values('tools')
    )

    return WSConversation.objects.annotate(
        message_count=Count('db_messages', distinct=True),
        rated_count=Count(
            'db_messages',
            filter=Q(db_messages__rating__isnull=False),
            distinct=True,
        ),
        avg_rating=Avg('db_messages__rating'),
        min_rating=Min('db_messages__rating'),
        max_rating=Max('db_messages__rating'),
        last_rated_at=Max('db_messages__feedback_submitted_at'),
        tools_used=Subquery(tools_subquery),
    )


MESSAGES_STREAM = StreamConfig(
    name='messages',
    allowed_fields=MESSAGES_FIELDS,
    field_map=MESSAGES_FIELD_MAP,
    virtual_fields=MESSAGES_VIRTUAL_FIELDS,
    base_qs=_build_messages_queryset,
)

CONVERSATIONS_STREAM = StreamConfig(
    name='conversations',
    allowed_fields=CONVERSATIONS_FIELDS,
    field_map=CONVERSATIONS_FIELD_MAP,
    virtual_fields=CONVERSATIONS_VIRTUAL_FIELDS,
    base_qs=_build_conversations_queryset,
)

STREAMS: dict[str, StreamConfig] = {
    'messages': MESSAGES_STREAM,
    'conversations': CONVERSATIONS_STREAM,
}


def _orm(field: str) -> str:
    """Backwards-compat: ORM lookup for the messages stream."""
    return MESSAGES_FIELD_MAP[field]


# ---------------------------------------------------------------------------
# WHERE clause → Django Q objects
# ---------------------------------------------------------------------------
# Django's Q objects let you build filter conditions programmatically.
# We convert each Condition into a Q, then chain them with & (and) or | (or).

def _condition_to_q(cond: Condition, stream: StreamConfig):
    from django.db.models import Q

    if cond.field not in stream.allowed_fields:
        raise FeedbackQLSyntaxError(
            f"Unknown field {cond.field!r} in pre-summarize where clause on "
            f"the {stream.name!r} stream. "
            f"Allowed fields: {', '.join(sorted(stream.allowed_fields))}"
        )
    op = cond.op

    # conversation_tool is a virtual field with no DB column on the messages
    # stream. We translate it to a conversation_id __in subquery so it
    # composes cleanly with other Q objects using & and |.
    if stream.name == 'messages' and cond.field == 'conversation_tool':
        if op not in ('==', '!='):
            raise FeedbackQLSyntaxError(
                f"conversation_tool only supports == and != operators, got {op!r}"
            )
        if cond.value is None:
            conv_ids = Message.objects.filter(
                tool_call_name__isnull=False
            ).values('conversation_id')
            q = Q(conversation_id__in=conv_ids)
            return ~q if op == '==' else q
        conv_ids = Message.objects.filter(
            tool_call_name=cond.value
        ).values('conversation_id')
        q = Q(conversation_id__in=conv_ids)
        return q if op == '==' else ~q

    path = stream.orm(cond.field)

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


def _apply_where(qs, clause: WhereClause, stream: StreamConfig):
    """
    Apply a WhereClause to a queryset by building a chain of Q objects.

    The clause.parts list alternates: [Condition, 'and'/'or', Condition, ...]
    We combine them left-to-right using & for 'and' and | for 'or'.
    """
    parts = clause.parts
    q = _condition_to_q(parts[0], stream)
    for i in range(1, len(parts), 2):
        connector = parts[i]
        next_q = _condition_to_q(parts[i + 1], stream)
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
# Virtual field resolution (multi-value lookups)
# ---------------------------------------------------------------------------
# Virtual fields like `conversation_tool` have no DB column on Message — they
# describe something about the message's *conversation*. To display them in
# select or group by them in summarize, we look up the values per conversation
# and attach them to message rows after fetching.

def _conversation_tool_lookup(conv_ids) -> dict:
    """
    Build a {conversation_id: sorted list of distinct tool names} map.

    Conversations with no tool calls are absent from the map (callers should
    treat absence as the empty list / a null group).
    """
    conv_id_set = {c for c in conv_ids if c is not None}
    if not conv_id_set:
        return {}
    pairs = (
        Message.objects
        .filter(conversation_id__in=conv_id_set, tool_call_name__isnull=False)
        .values_list('conversation_id', 'tool_call_name')
    )
    grouped: dict = defaultdict(set)
    for conv_id, tool in pairs:
        grouped[conv_id].add(tool)
    return {cid: sorted(tools) for cid, tools in grouped.items()}


# ---------------------------------------------------------------------------
# Row-level finalisation (no summarize ran)
# ---------------------------------------------------------------------------

# Per-stream "always include" fields — the messages stream always carries
# conversation_id + message_uuid so the conversation thread viewer can open
# the full thread for any row. Conversations stream rows are already keyed
# by conversation_id (it's a regular field), so nothing extra is needed.
_THREAD_META_FIELDS_BY_STREAM: dict[str, frozenset[str]] = {
    'messages':      frozenset({'conversation_id', 'message_uuid'}),
    'conversations': frozenset(),
}
# Backwards-compat name for any code path still expecting the messages stream.
_THREAD_META_FIELDS: frozenset[str] = _THREAD_META_FIELDS_BY_STREAM['messages']


def _finalise_row_level(qs, select: SelectClause | None, stream: StreamConfig) -> list[dict]:
    """
    Materialise the queryset into a list of dicts with user-facing field names.

    By the time we get here, all where / order_by / limit operations on the
    queryset have already been applied. We just need to project the right
    columns and remap ORM paths back to user-facing names.

    Virtual fields (no DB column) are computed in Python after fetching by
    looking up values per conversation. For `conversation_tool` (messages
    stream only) we join the list of distinct tools used in each conversation
    into a comma-separated string, or leave it null when no tools were used.
    """
    thread_meta = _THREAD_META_FIELDS_BY_STREAM[stream.name]
    output_fields = (
        select.fields if select
        else sorted(stream.allowed_fields - stream.virtual_fields)
    )
    output_field_set = set(output_fields)

    # Virtual fields can't be fetched from the DB — drop them from orm_fields
    # but keep them in the desired output set. They get populated below.
    db_fields = (set(output_fields) - stream.virtual_fields) | thread_meta
    orm_fields = [stream.orm(f) for f in db_fields]

    # If no LIMIT was applied to the queryset, cap at MAX_ROWS as a safety net
    rows = list(qs.values(*orm_fields)[:MAX_ROWS])

    # Resolve any virtual field values needed for output (messages stream only)
    tool_map: dict = {}
    if stream.name == 'messages' and 'conversation_tool' in output_field_set:
        tool_map = _conversation_tool_lookup({r.get('conversation_id') for r in rows})

    result = []
    for row in rows:
        remapped = {stream.orm_to_field.get(k, k): v for k, v in row.items()}
        if stream.name == 'messages' and 'conversation_tool' in output_field_set:
            tools = tool_map.get(remapped.get('conversation_id'))
            remapped['conversation_tool'] = ', '.join(tools) if tools else None
        result.append({
            k: v for k, v in remapped.items()
            if k in output_field_set or k in thread_meta
        })

    return result


# ---------------------------------------------------------------------------
# Summarize (aggregate) execution
# ---------------------------------------------------------------------------

def _run_summarize(qs, summarize: SummarizeClause, stream: StreamConfig) -> list[dict]:
    """
    Fetch the queryset, group it in Python, and compute aggregations per group.

    Returns a list of dicts where each dict's keys are the group-by field
    names plus the aggregation aliases. From this point onward in the
    pipeline we work with this list rather than the queryset.

    Multi-value virtual fields (currently just `conversation_tool` on the
    messages stream) are unnested before grouping when used in `by`:
    a message in a conversation that used N tools contributes N rows,
    one per tool. Conversations with no tool calls contribute one row with
    the value None.
    """
    by_fields = summarize.by
    agg_fields_needed = {agg.agg_field for agg in summarize.aggregations if agg.agg_field}
    fetch_fields = (set(by_fields) | agg_fields_needed) - stream.virtual_fields
    # If a virtual field is in `by`, we need conversation_id to look up its values
    if stream.name == 'messages' and 'conversation_tool' in by_fields:
        fetch_fields.add('conversation_id')
    orm_fields = [stream.orm(f) for f in fetch_fields]

    rows = list(qs.values(*orm_fields)[:MAX_ROWS])

    if not rows:
        return []

    # Unnest virtual fields in `by` before grouping (messages stream only)
    if stream.name == 'messages' and 'conversation_tool' in by_fields:
        conv_orm = stream.orm('conversation_id')
        tool_map = _conversation_tool_lookup({r.get(conv_orm) for r in rows})
        expanded = []
        for row in rows:
            tools = tool_map.get(row.get(conv_orm))
            if tools:
                for tool in tools:
                    new_row = dict(row)
                    new_row['conversation_tool'] = tool
                    expanded.append(new_row)
            else:
                new_row = dict(row)
                new_row['conversation_tool'] = None
                expanded.append(new_row)
        rows = expanded

    # Virtual fields live under their own key after unnest; real fields are
    # under the ORM lookup path returned by qs.values()
    def _row_key(row: dict, fname: str):
        if fname in stream.virtual_fields:
            return row.get(fname)
        return row[stream.orm(fname)]

    if by_fields:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            key = tuple(_row_key(row, f) for f in by_fields)
            groups[key].append(row)

        result = []
        for key, group_rows in groups.items():
            out: dict = {by_fields[i]: key[i] for i in range(len(by_fields))}
            for agg in summarize.aggregations:
                if agg.func == 'count' and not agg.agg_field:
                    out[agg.alias] = len(group_rows)
                else:
                    values = [r[stream.orm(agg.agg_field)] for r in group_rows]  # type: ignore[arg-type]
                    out[agg.alias] = _compute_agg(agg.func, values)
            result.append(out)
        return result

    # No 'by' fields — one global aggregate row over all data
    out = {}
    for agg in summarize.aggregations:
        if agg.func == 'count' and not agg.agg_field:
            out[agg.alias] = len(rows)
        else:
            values = [r[stream.orm(agg.agg_field)] for r in rows]  # type: ignore[arg-type]
            out[agg.alias] = _compute_agg(agg.func, values)
    return [out]


# ---------------------------------------------------------------------------
# Public entry point — pipeline-order execution
# ---------------------------------------------------------------------------

def execute(query: Query) -> list[dict]:
    """
    Execute a parsed FeedbackQL Query by walking its clauses in pipeline order.

    The pipeline starts as a Django queryset (built from the source stream —
    `messages` or `conversations`) and stays a queryset until a `summarize`
    runs, at which point it becomes a list of in-memory dicts. Subsequent
    where/order by/limit clauses operate on whatever shape the previous
    stage left behind.

    Never executes raw SQL — all database access goes through Django ORM.
    """
    stream = STREAMS.get(query.stream)
    if stream is None:
        raise FeedbackQLSyntaxError(f"Unknown stream {query.stream!r}")

    # Pre-summarize state: a Django queryset built per stream
    qs = stream.base_qs()

    # Post-summarize state: a list of dicts (None until summarize runs)
    rows: list[dict] | None = None

    # The set of field names a post-summarize where/order can reference
    # (group-by fields + aggregation aliases). Empty until summarize runs.
    post_summarize_keys: set[str] = set()

    select_clause: SelectClause | None = None

    for clause in query.clauses:
        if isinstance(clause, WhereClause):
            if rows is None:
                qs = _apply_where(qs, clause, stream)
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
            rows = _run_summarize(qs, clause, stream)
            post_summarize_keys = set(clause.by) | {agg.alias for agg in clause.aggregations}

        elif isinstance(clause, OrderByClause):
            if rows is None:
                if clause.field not in stream.allowed_fields:
                    raise FeedbackQLSyntaxError(
                        f"Unknown field {clause.field!r} in order by on the "
                        f"{stream.name!r} stream. "
                        f"Allowed fields: {', '.join(sorted(stream.allowed_fields))}"
                    )
                prefix = '-' if clause.direction == 'desc' else ''
                qs = qs.order_by(f'{prefix}{stream.orm(clause.field)}')
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

    return _finalise_row_level(qs, select_clause, stream)
