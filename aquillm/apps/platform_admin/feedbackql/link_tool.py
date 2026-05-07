"""Feedback dashboard link builder tool.

Exposes the `build_feedback_link` LLM tool. The LLM passes a structured
JSON spec describing the user's intent (stream, where conditions,
aggregations, etc.) as a string. This module parses the JSON, builds the
FeedbackQL query string deterministically, validates it through the
existing parser, base64-encodes the URL, and returns it.

Why structured JSON instead of a free-form query string:
  Asking an LLM to produce strict FeedbackQL syntax (pipes, the 'where'
  keyword, null literals, etc.) is unreliable. By making the tool take
  structured pieces and assemble the query in Python, we make every
  syntactic guarantee deterministic — the LLM can only describe intent,
  never produce a broken query string.

Why a single JSON-string param rather than multiple typed params:
  The project's @llm_tool decorator only supports primitive types and
  lists of primitives in tool signatures, not nested dicts. Taking the
  whole spec as one JSON string keeps the structured-input architecture
  without modifying the boss's decorator.
"""
from __future__ import annotations

import json
from typing import Any

from lib.llm.decorators import llm_tool
from lib.llm.types import LLMTool, ToolResultDict

from .exceptions import FeedbackQLFieldError, FeedbackQLSyntaxError
from .parser import Condition, OrderByClause, Query, SummarizeClause, WhereClause, parse
from .token_store import mint_token


_DASHBOARD_PATH = "/aquillm/feedback-dashboard/"

_VALID_OPS = frozenset({
    "==", "!=", "<", ">", "<=", ">=",
    "startswith", "contains", "in",
})

_VALID_AGG_FUNCS = frozenset({"avg", "count", "min", "max", "sum", "median"})

_VALID_DIRECTIONS = frozenset({"asc", "desc"})

# Fields that are null on a meaningful fraction of rows. When the LLM
# sorts/aggregates/groups by one of these without a where filter on it,
# the null group typically dominates and the answer is wrong (e.g. the
# top group is `(none)` because most messages aren't tool calls).
# Per stream because the field set differs.
_NULL_PRONE_FIELDS = {
    "messages": frozenset({
        "rating",
        "feedback_text",
        "feedback_submitted_at",
        "tool_call_name",
    }),
    "conversations": frozenset({
        "avg_rating",
        "min_rating",
        "max_rating",
        "last_rated_at",
        "tools_used",
    }),
}


def _query_filters_on(query: Query, field: str) -> bool:
    """True if any where clause in the query has a Condition on `field`."""
    for clause in query.clauses:
        if not isinstance(clause, WhereClause):
            continue
        for part in clause.parts:
            if isinstance(part, Condition) and part.field == field:
                return True
    return False


def _query_aggregates_on(query: Query, field: str) -> bool:
    """True if any summarize clause aggregates on the given field."""
    for clause in query.clauses:
        if isinstance(clause, SummarizeClause):
            for agg in clause.aggregations:
                if agg.agg_field == field:
                    return True
    return False


def _query_orders_on(query: Query, field: str) -> bool:
    """True if any order_by clause sorts on the given field."""
    for clause in query.clauses:
        if isinstance(clause, OrderByClause) and clause.field == field:
            return True
    return False


def _query_groups_by(query: Query, field: str) -> bool:
    """True if any summarize clause groups by the given field."""
    for clause in query.clauses:
        if isinstance(clause, SummarizeClause) and field in clause.by:
            return True
    return False


# ---------------------------------------------------------------------------
# Server-side auto-fixes
# ---------------------------------------------------------------------------
# Some interpretation rules have only one correct completion (add the null
# filter, add role==assistant for rating queries). These were originally
# implemented as advisory hints, then as exceptions that asked the LLM to
# retry. Both depended on the LLM doing follow-up work, which weaker models
# can't reliably do. So for these unambiguous cases we just apply the
# correction server-side and ship a working URL. The fix is visible to the
# user via the query string the dashboard renders.
#
# Cases where this is NOT safe (and should stay as advisory or LLM choice):
# stream selection, comparative-words sort-vs-threshold, sample-size count
# alongside aggregations. Those involve genuine judgment calls.

def _spec_filtered_fields(spec: dict) -> set[str]:
    """Fields that appear in any where condition in the JSON spec."""
    fields: set[str] = set()
    for cond in spec.get("where") or []:
        if isinstance(cond, dict) and "field" in cond:
            fields.add(cond["field"])
    return fields


def _spec_aggregated_or_sorted_fields(spec: dict) -> set[str]:
    """Fields referenced by order_by, summarize.aggregations, or summarize.by."""
    referenced: set[str] = set()
    ob = spec.get("order_by")
    if isinstance(ob, dict) and "field" in ob:
        referenced.add(ob["field"])
    s = spec.get("summarize")
    if isinstance(s, dict):
        for agg in s.get("aggregations") or []:
            if isinstance(agg, dict) and agg.get("field"):
                referenced.add(agg["field"])
        for f in s.get("by") or []:
            referenced.add(f)
    return referenced


_RATING_FEEDBACK_FIELDS = frozenset({
    "rating", "feedback_text", "feedback_submitted_at",
})


def _apply_safe_fixes(spec: dict) -> None:
    """Mutate spec in place to add unambiguous safety filters.

    Two fixes:
      1. Null filter on null-prone fields referenced in summarize/order_by.
         The user clearly doesn't want the (none) group dominating the
         result; if they did, they'd have filtered explicitly.
      2. Role filter on the messages stream when rating/feedback fields
         are referenced. In AquiLLM, ratings and feedback only attach to
         assistant messages; without the filter the result includes
         user/tool messages that aren't part of what the user asked.

    Both fixes are no-ops if the user already filtered the field in any
    way (any operator). Only adds the filter when no constraint exists.
    """
    stream = spec.get("stream")
    if stream not in _NULL_PRONE_FIELDS:
        return

    where = spec.setdefault("where", [])
    if not isinstance(where, list):
        return  # Malformed spec — let downstream validation catch it.

    filtered = _spec_filtered_fields(spec)
    sorted_or_aggregated = _spec_aggregated_or_sorted_fields(spec)

    # Fix 1: null filter on null-prone fields used in summarize/order_by
    for field in sorted(_NULL_PRONE_FIELDS[stream] & sorted_or_aggregated):
        if field not in filtered:
            where.append({"field": field, "op": "!=", "value": None})
            filtered.add(field)

    # Fix 2: role filter on messages stream when rating/feedback referenced
    if stream == "messages":
        all_field_refs = filtered | sorted_or_aggregated
        if all_field_refs & _RATING_FEEDBACK_FIELDS and "role" not in filtered:
            where.append({"field": "role", "op": "==", "value": "assistant"})


def _format_value(v: Any) -> str:
    """Render a Python value as a FeedbackQL literal."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        raise ValueError("FeedbackQL has no boolean literal; use a comparison instead")
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if '"' in v and "'" not in v:
            return f"'{v}'"
        if '"' in v and "'" in v:
            raise ValueError(f"String value contains both quote types: {v!r}")
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_format_value(x) for x in v) + "]"
    raise ValueError(f"Unsupported value type: {type(v).__name__}")


def _format_condition(cond: dict) -> str:
    if not isinstance(cond, dict):
        raise ValueError(f"Each where condition must be an object, got {type(cond).__name__}")
    if "field" not in cond:
        raise ValueError(f"Where condition missing 'field': {cond}")
    if "op" not in cond:
        raise ValueError(f"Where condition missing 'op': {cond}")
    field = cond["field"]
    op = cond["op"]
    if op not in _VALID_OPS:
        raise ValueError(
            f"Invalid op {op!r}. Valid: {', '.join(sorted(_VALID_OPS))}"
        )
    value = cond.get("value")
    if op == "in" and not isinstance(value, list):
        raise ValueError("'in' requires a list value")
    return f"{field} {op} {_format_value(value)}"


def _format_where(conditions: list) -> str:
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("'where' must be a non-empty list of conditions")
    return "where " + " and ".join(_format_condition(c) for c in conditions)


def _format_aggregation(agg: dict) -> str:
    if not isinstance(agg, dict):
        raise ValueError(f"Each aggregation must be an object, got {type(agg).__name__}")
    if "alias" not in agg or "func" not in agg:
        raise ValueError(f"Aggregation needs 'alias' and 'func': {agg}")
    func = agg["func"]
    if func not in _VALID_AGG_FUNCS:
        raise ValueError(
            f"Invalid aggregation func {func!r}. Valid: {', '.join(sorted(_VALID_AGG_FUNCS))}"
        )
    alias = agg["alias"]
    field = agg.get("field")
    if func == "count" and field is None:
        return f"{alias} = count()"
    if field is None:
        raise ValueError(f"Aggregation func {func!r} requires 'field'")
    return f"{alias} = {func}({field})"


def _format_summarize(summarize: dict) -> str:
    if not isinstance(summarize, dict):
        raise ValueError(f"'summarize' must be an object, got {type(summarize).__name__}")
    aggs = summarize.get("aggregations")
    if not isinstance(aggs, list) or not aggs:
        raise ValueError("'summarize.aggregations' must be a non-empty list")
    out = "summarize " + ", ".join(_format_aggregation(a) for a in aggs)
    by = summarize.get("by")
    if by:
        if not isinstance(by, list):
            raise ValueError("'summarize.by' must be a list of field names")
        out += " by " + ", ".join(by)
    return out


def _format_select(fields: list) -> str:
    if not isinstance(fields, list) or not fields:
        raise ValueError("'select' must be a non-empty list of field names")
    return "select " + ", ".join(fields)


def _format_order_by(order_by: dict) -> str:
    if not isinstance(order_by, dict) or "field" not in order_by:
        raise ValueError("'order_by' must be an object with 'field'")
    direction = order_by.get("direction", "asc")
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"order_by direction must be 'asc' or 'desc', got {direction!r}")
    return f"order by {order_by['field']} {direction}"


def _format_limit(n: Any) -> str:
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"'limit' must be a positive integer, got {n!r}")
    return f"limit {n}"


def _build_query(
    stream: str,
    where: list | None,
    summarize: dict | None,
    having: list | None,
    select: list | None,
    order_by: dict | None,
    limit: int | None,
) -> str:
    if stream not in ("messages", "conversations"):
        raise ValueError(
            f"'stream' must be 'messages' or 'conversations', got {stream!r}"
        )
    if select and summarize:
        raise ValueError("'select' and 'summarize' cannot both be used")
    if having and not summarize:
        raise ValueError(
            "'having' filters aggregate output, so it requires a 'summarize' clause"
        )
    parts = [stream]
    if where:
        parts.append(_format_where(where))
    if summarize:
        parts.append(_format_summarize(summarize))
        if having:
            # Post-summarize where: same syntax as pre-summarize where, but
            # the parser scopes valid fields to the summarize aliases + by fields.
            parts.append(_format_where(having))
    if select:
        parts.append(_format_select(select))
    if order_by:
        parts.append(_format_order_by(order_by))
    if limit is not None:
        parts.append(_format_limit(limit))
    return "\n| ".join(parts)


def build_feedback_link_tool(base_url: str = "") -> LLMTool:
    """Build the build_feedback_link tool. See module docstring."""

    base = base_url.rstrip("/") if base_url else ""

    @llm_tool(
        for_whom="assistant",
        description=(
            "Build a shareable feedback-dashboard URL from a structured JSON "
            "spec. ALWAYS use this tool to produce dashboard links — never "
            "construct URLs or query strings by hand. You only describe "
            "intent in JSON; the tool assembles the FeedbackQL syntax "
            "internally so you cannot make pipe/keyword/operator mistakes. "
            "If the tool returns an `exception`, the message tells you "
            "exactly what to fix — rebuild the JSON to address it and "
            "call the tool again before replying to the user. Never "
            "paste exception text into your final response."
        ),
        required=["query_spec"],
        param_descs={
            "query_spec": (
                'JSON object as a string. Schema:\n'
                '{\n'
                '  "stream": "messages" | "conversations",  // REQUIRED\n'
                '  "where": [                               // optional, AND-joined\n'
                '    {"field": <name>, "op": <op>, "value": <v>},\n'
                '    ...\n'
                '  ],\n'
                '  "summarize": {                           // optional\n'
                '    "aggregations": [\n'
                '      {"alias": <out-name>, "func": <func>, "field": <field>},\n'
                '      ...\n'
                '    ],\n'
                '    "by": [<field>, ...]                   // optional\n'
                '  },\n'
                '  "having": [                              // optional, requires summarize; filter on aggregate aliases + by fields\n'
                '    {"field": <alias-or-by-field>, "op": <op>, "value": <v>},\n'
                '    ...\n'
                '  ],\n'
                '  "select": [<field>, ...],                // optional, mutually exclusive with summarize\n'
                '  "order_by": {"field": <name>, "direction": "asc"|"desc"},\n'
                '  "limit": <positive int>\n'
                '}\n\n'
                'Valid ops: ==, !=, <, >, <=, >=, startswith, contains, in.\n'
                'Use value: null with op == or != for null checks.\n'
                'Valid funcs: avg, count, min, max, sum, median (count needs no field).\n'
                'Date values are ISO strings, e.g. "2026-04-01".\n\n'
                'ROLE FILTER RULE: when the user mentions answers, responses, '
                'outputs, the assistant, the model, the AI, ratings, feedback, '
                'comments, or stars, include '
                '{"field": "role", "op": "==", "value": "assistant"} in `where` '
                '— these always attach to assistant messages in AquiLLM.\n\n'
                'COUNT RULE: when using `summarize` with avg/min/max/sum/median, '
                'always include a count aggregation alongside so the user can '
                'see sample sizes.\n\n'
                'HAVING RULE: questions like "for X who did Y at least N times" '
                'or "groups whose aggregate is above/below threshold" need the '
                '`having` field, NOT `where`. `having` filters the aggregate '
                'output and can reference aggregation aliases. Example: '
                '"users who left at least 3 ratings, what is their average" → '
                'summarize by user_id with count alias `n` and avg alias `avg_r`, '
                'then having: [{"field":"n","op":">=","value":3}].\n\n'
                'EXAMPLE — "lowest-rated assistant answers from last month":\n'
                '{"stream": "messages", '
                '"where": ['
                '{"field":"role","op":"==","value":"assistant"},'
                '{"field":"rating","op":"<=","value":2},'
                '{"field":"feedback_submitted_at","op":">=","value":"2026-04-01"},'
                '{"field":"feedback_submitted_at","op":"<","value":"2026-05-01"}'
                '], '
                '"order_by": {"field":"rating","direction":"asc"}, '
                '"limit": 50}'
            ),
        },
    )
    def build_feedback_link(query_spec: str) -> ToolResultDict:
        try:
            spec = json.loads(query_spec)
        except json.JSONDecodeError as exc:
            return {"exception": f"query_spec is not valid JSON: {exc}"}
        if not isinstance(spec, dict):
            return {"exception": "query_spec must be a JSON object"}

        # Apply unambiguous safety corrections before building the query.
        # See _apply_safe_fixes for details.
        _apply_safe_fixes(spec)

        try:
            query_text = _build_query(
                stream=spec.get("stream"),
                where=spec.get("where"),
                summarize=spec.get("summarize"),
                having=spec.get("having"),
                select=spec.get("select"),
                order_by=spec.get("order_by"),
                limit=spec.get("limit"),
            )
        except ValueError as exc:
            return {"exception": f"Invalid query_spec: {exc}"}

        # Parse the generated query as a sanity check. If this fails it's
        # almost certainly a bug in our query builder — but we'd rather
        # surface a clear error than ship a broken URL.
        try:
            parsed = parse(query_text)
        except (FeedbackQLSyntaxError, FeedbackQLFieldError) as exc:
            return {
                "exception": (
                    f"Generated query failed validation: {exc}. "
                    f"Query was: {query_text}"
                )
            }

        # Use a short token instead of base64-in-URL: gpt-4o (and other LLMs)
        # occasionally drop characters when transcribing long opaque strings,
        # which silently corrupts the query. A 7-character token is short
        # enough to copy reliably.
        token = mint_token(query_text)
        url = f"{base}{_DASHBOARD_PATH}?t={token}"
        result: dict = {"url": url, "query": query_text}

        # The role filter and null filter — the most common LLM omissions —
        # are auto-corrected up-front by _apply_safe_fixes. The hints list
        # below is for future ambiguous interpretation rules that warrant
        # advisory feedback (where auto-fix would be presumptuous).
        hints: list[str] = []

        # If we detected an interpretation problem, return an error rather
        # than the URL. Returning a URL + hint lets the LLM optionally
        # forward the buggy URL to the user (we've observed this with
        # gpt-4o); returning an error forces a real retry. Wording is
        # imperative and matches parser-error style — gpt-4o reliably
        # retries on parser errors; softer "needs adjustment" wording
        # was sometimes met with a "let me adjust" stub that didn't
        # actually retry.
        if hints:
            return {"exception": " ".join(hints)}
        return {"result": result}

    return build_feedback_link
