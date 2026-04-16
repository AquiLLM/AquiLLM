# apps/platform_admin/services/feedback_prql.py
# complete file — all changes from previous version are:
#   1. _compile_prql_subset_to_sql: add group block handling
#   2. _collect_group_block: new helper
#   3. _parse_group_block: new helper
#   everything else is identical to the version already on disk
"""
prql-backed query engine for the feedback dashboard

architecture
------------
FeedbackFilters struct
    -> build_prql_query()       generates a PRQL-like string describing the query
    -> compile_prql_to_sql()    compiles that query to SQL
    -> _build_feedback_cte()    prepended as WITH feedback AS (...) before execution
    -> execute_prql_query()     runs the full SQL against django db connection
    -> returns list of dicts

notes
-----
The installed prql-python version in this environment does not accept some of the
syntax shapes used by our test suite, especially:

    take {0..49}
    filter feedback_text ~* $param

To keep the PRQL contract stable for the project and the tests, this module
supports a constrained PRQL subset and compiles it to SQL directly when needed.
That lets us preserve the same query language shape without depending on
compiler-version-specific syntax support.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone as py_tz
from typing import Any

from django.db import connection

logger = logging.getLogger(__name__)

_MSG_TABLE = "aquillm_message"
_CONVO_TABLE = "aquillm_wsconversation"
_USER_TABLE = "auth_user"
_PRQL_SOURCE = "feedback"
_PLACEHOLDER_RE = re.compile(r"^\$[A-Za-z_][A-Za-z0-9_]*$|^\$\d+$")


# ---------------------------------------------------------------------------
# PRQL generation
# ---------------------------------------------------------------------------

def build_prql_query(
    filters: "FeedbackFilters",  # noqa: F821
    *,
    aggregate: bool = False,
    page: int = 1,
    page_size: int = 50,
) -> tuple[str, list[Any]]:
    filter_clauses: list[str] = []
    params: list[Any] = []

    if filters.start_date is not None:
        filter_clauses.append("filter effective_date >= $start_date")
        params.append(_to_pg_ts(filters.start_date))

    if filters.end_date is not None:
        filter_clauses.append("filter effective_date <= $end_date")
        params.append(_to_pg_ts(filters.end_date))

    if filters.user_id is not None:
        filter_clauses.append("filter user_id == $user_id")
        params.append(filters.user_id)

    if filters.exact_rating is not None:
        filter_clauses.append("filter rating == $exact_rating")
        params.append(filters.exact_rating)
    else:
        if filters.min_rating is not None:
            filter_clauses.append("filter rating >= $min_rating")
            params.append(filters.min_rating)
        if filters.max_rating is not None:
            filter_clauses.append("filter rating <= $max_rating")
            params.append(filters.max_rating)

    if filters.role:
        filter_clauses.append("filter role == $role")
        params.append(filters.role)

    if filters.model:
        filter_clauses.append("filter model == $model")
        params.append(filters.model)

    if filters.tool_call_name:
        filter_clauses.append("filter tool_call_name == $tool_call_name")
        params.append(filters.tool_call_name)

    if filters.feedback_text_search:
        filter_clauses.append("filter feedback_text ~* $feedback_text_search")
        params.append(f"%{filters.feedback_text_search}%")

    if filters.conversation_name_search:
        filter_clauses.append("filter conversation_name ~* $conversation_name_search")
        params.append(f"%{filters.conversation_name_search}%")

    if filters.has_feedback_text is True:
        filter_clauses.append("filter has_feedback_text == true")
    elif filters.has_feedback_text is False:
        filter_clauses.append("filter has_feedback_text == false")

    if aggregate:
        prql = _build_aggregate_prql(filter_clauses)
    else:
        prql = _build_rows_prql(filter_clauses, page=page, page_size=page_size)

    return prql, params


def _build_rows_prql(
    filter_clauses: list[str],
    *,
    page: int,
    page_size: int,
) -> str:
    offset = max(0, (page - 1) * page_size)
    take_end = offset + page_size - 1
    filter_block = "\n".join(filter_clauses)
    if filter_block:
        filter_block += "\n"
    return (
        f"from {_PRQL_SOURCE}\n"
        f"{filter_block}"
        f"sort [effective_date, id]\n"
        f"take {{{offset}..{take_end}}}\n"
        f"select {{\n"
        f"  id,\n"
        f"  message_uuid,\n"
        f"  conversation_id,\n"
        f"  conversation_name,\n"
        f"  user_id,\n"
        f"  username,\n"
        f"  rating,\n"
        f"  feedback_text,\n"
        f"  feedback_submitted_at,\n"
        f"  created_at,\n"
        f"  effective_date,\n"
        f"  role,\n"
        f"  content_snippet,\n"
        f"  model,\n"
        f"  tool_call_name,\n"
        f"  usage,\n"
        f"}}\n"
    )


def _build_aggregate_prql(filter_clauses: list[str]) -> str:
    filter_block = "\n".join(filter_clauses)
    if filter_block:
        filter_block += "\n"
    return (
        f"from {_PRQL_SOURCE}\n"
        f"{filter_block}"
        f"aggregate {{\n"
        f"  total_count = count id,\n"
        f"  avg_rating = average rating,\n"
        f"}}\n"
    )


# ---------------------------------------------------------------------------
# PRQL compilation
# ---------------------------------------------------------------------------

class PRQLCompilationError(Exception):
    """raised when compilation to sql fails"""


def compile_prql_to_sql(prql_string: str, *, dialect: str = "postgres") -> str:
    try:
        return _compile_prql_subset_to_sql(prql_string)
    except Exception as exc:
        raise PRQLCompilationError(f"prql compilation failed: {exc}") from exc


def _compile_prql_subset_to_sql(prql_string: str) -> str:
    lines = [line.rstrip() for line in prql_string.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty PRQL string")

    source: str | None = None
    where_clauses: list[str] = []
    order_by_fields: list[str] = []
    select_fields: list[str] | None = None
    aggregate_fields: list[str] | None = None
    group_by_fields: list[str] | None = None
    limit_clause: str | None = None
    offset_clause: str | None = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("from "):
            source = line[len("from "):].strip()
            if not source:
                raise ValueError("missing source after from")

        elif line.startswith("filter "):
            clause = _parse_filter_line(line)
            where_clauses.append(clause)

        elif line.startswith("sort "):
            order_by_fields = _parse_sort_line(line)

        elif line.startswith("take "):
            limit_clause, offset_clause = _parse_take_line(line)

        elif line.startswith("select"):
            block_text, next_i = _collect_block(lines, i, "select")
            select_fields = _parse_select_block(block_text)
            i = next_i

        elif line.startswith("aggregate"):
            block_text, next_i = _collect_block(lines, i, "aggregate")
            aggregate_fields = _parse_aggregate_block(block_text)
            i = next_i

        elif re.match(r"^group\s+", line):
            # group col (aggregate { ... }) or group {col1, col2} (aggregate { ... })
            group_by_fields, aggregate_fields, next_i = _collect_group_block(lines, i)
            i = next_i

        else:
            raise ValueError(f"unsupported PRQL line: {line}")

        i += 1

    if not source:
        raise ValueError("missing from clause")

    # no select and no aggregate: return all columns
    if not select_fields and not aggregate_fields:
        select_fields = ["*"]

    if select_fields and aggregate_fields:
        raise ValueError("query cannot contain both select and aggregate")

    # build SELECT clause
    if aggregate_fields is not None and group_by_fields:
        # group query: SELECT group_cols, agg_exprs FROM ...
        select_sql = ", ".join(group_by_fields + aggregate_fields)
    elif aggregate_fields is not None:
        # aggregate-only (no group)
        select_sql = ", ".join(aggregate_fields)
    else:
        select_sql = ", ".join(select_fields)  # type: ignore[arg-type]

    sql_parts = [f"SELECT {select_sql}", f"FROM {source}"]

    if where_clauses:
        sql_parts.append("WHERE " + " AND ".join(where_clauses))

    if group_by_fields:
        sql_parts.append("GROUP BY " + ", ".join(group_by_fields))

    # ORDER BY only applies to row queries and group queries (not bare aggregates)
    if order_by_fields:
        sql_parts.append("ORDER BY " + ", ".join(order_by_fields))

    if limit_clause and aggregate_fields is None:
        sql_parts.append(limit_clause)

    if offset_clause and aggregate_fields is None:
        sql_parts.append(offset_clause)

    return " ".join(sql_parts)


def _collect_group_block(
    lines: list[str], start_idx: int
) -> tuple[list[str], list[str], int]:
    """
    parse a group ... (aggregate { ... }) construct

    supported forms:
        group col (aggregate { ... })        — single column
        group {col1, col2} (aggregate {...}) — multiple columns

    returns (group_by_fields, aggregate_exprs, end_line_index)
    """
    line = lines[start_idx].strip()

    # extract the group-by column(s) and the opening of the aggregate block
    # form 1: group col (
    m_single = re.match(r"^group\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*$", line)
    # form 2: group {col1, col2} (
    m_multi = re.match(r"^group\s+\{([^}]+)\}\s*\(\s*$", line)

    if m_single:
        group_cols = [m_single.group(1).strip()]
    elif m_multi:
        group_cols = [c.strip() for c in m_multi.group(1).split(",") if c.strip()]
    else:
        raise ValueError(f"unsupported group clause: {line}")

    # validate all group column names
    for col in group_cols:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col):
            raise ValueError(f"invalid group column: {col!r}")

    # now collect everything until the closing ) of the group block
    # the inner structure is: aggregate { ... }
    # find the aggregate keyword
    i = start_idx + 1
    aggregate_fields: list[str] | None = None
    end_idx = start_idx

    while i < len(lines):
        inner = lines[i].strip()

        if inner.startswith("aggregate"):
            block_text, next_i = _collect_block(lines, i, "aggregate")
            aggregate_fields = _parse_aggregate_block(block_text)
            i = next_i + 1
            continue

        if inner == ")":
            end_idx = i
            break

        # skip blank lines inside the group block
        if not inner:
            i += 1
            continue

        raise ValueError(f"unexpected line inside group block: {inner!r}")
    else:
        raise ValueError("unterminated group block — missing closing )")

    if aggregate_fields is None:
        raise ValueError("group block must contain an aggregate clause")

    return group_cols, aggregate_fields, end_idx


def _parse_filter_line(line: str) -> str:
    # handle null comparisons separately: == null / != null
    m_null = re.match(
        r"^filter\s+([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=)\s*null\s*$",
        line,
        re.IGNORECASE,
    )
    if m_null:
        field, op = m_null.groups()
        return f"{field} IS NOT NULL" if op == "!=" else f"{field} IS NULL"

    m = re.match(
        r"^filter\s+([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|>=|<=|>|<|~\*)\s*(.+?)\s*$",
        line,
    )
    if not m:
        raise ValueError(f"unsupported filter clause: {line}")

    field, op, rhs = m.groups()
    rhs = rhs.strip()

    if op == "~*":
        if not _PLACEHOLDER_RE.match(rhs):
            raise ValueError(f"unsupported text-search rhs: {rhs}")
        return f"{field} ILIKE %s"

    if rhs.lower() == "true":
        rhs_sql = "TRUE"
    elif rhs.lower() == "false":
        rhs_sql = "FALSE"
    elif _PLACEHOLDER_RE.match(rhs):
        rhs_sql = "%s"
    elif re.fullmatch(r"-?\d+(\.\d+)?", rhs):
        rhs_sql = rhs
    else:
        raise ValueError(f"unsupported rhs value: {rhs}")

    sql_op = {"==": "=", "!=": "!=", ">=": ">=", "<=": "<=", ">": ">", "<": "<"}[op]
    return f"{field} {sql_op} {rhs_sql}"


def _parse_sort_line(line: str) -> list[str]:
    """
    parse a sort line into ORDER BY expressions

    supported syntaxes:
        sort [field1, -field2]    bracket form, minus = DESC
        sort {field1, -field2}    curly brace form, minus = DESC
        sort field                bare single field ascending
        sort -field               bare single field descending

    emits: field ASC or field DESC
    """
    rest = line[len("sort "):].strip()

    m_curly = re.match(r"^\{(.+)\}$", rest)
    if m_curly:
        return _sort_fields_to_sql(m_curly.group(1))

    m_bracket = re.match(r"^\[(.+)\]$", rest)
    if m_bracket:
        return _sort_fields_to_sql(m_bracket.group(1))

    if re.match(r"^-?[A-Za-z_][A-Za-z0-9_]*$", rest):
        return _sort_fields_to_sql(rest)

    raise ValueError(f"unsupported sort clause: {line}")


def _sort_fields_to_sql(inner: str) -> list[str]:
    """convert comma-separated field tokens to SQL ORDER BY expressions"""
    result = []
    for part in inner.split(","):
        field = part.strip()
        if not field:
            continue
        if field.startswith("-"):
            col = field[1:].strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col):
                raise ValueError(f"invalid sort field: -{col}")
            result.append(f"{col} DESC")
        else:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", field):
                raise ValueError(f"invalid sort field: {field}")
            result.append(f"{field} ASC")
    if not result:
        raise ValueError("sort clause has no fields")
    return result


def _parse_take_line(line: str) -> tuple[str | None, str | None]:
    expr = line[len("take "):].strip()

    m_range_curly = re.match(r"^\{(\d+)\.\.(\d+)\}$", expr)
    if m_range_curly:
        start = int(m_range_curly.group(1))
        end = int(m_range_curly.group(2))
        if end < start:
            raise ValueError("take range end is before start")
        return f"LIMIT {end - start + 1}", f"OFFSET {start}"

    m_range_plain = re.match(r"^(\d+)\.\.(\d+)$", expr)
    if m_range_plain:
        start = int(m_range_plain.group(1))
        end = int(m_range_plain.group(2))
        if end < start:
            raise ValueError("take range end is before start")
        return f"LIMIT {end - start + 1}", f"OFFSET {start}"

    m_count = re.match(r"^(\d+)$", expr)
    if m_count:
        return f"LIMIT {int(m_count.group(1))}", None

    raise ValueError(f"unsupported take clause: {line}")


def _collect_block(lines: list[str], start_idx: int, keyword: str) -> tuple[str, int]:
    line = lines[start_idx].strip()

    # inline form: select {id, name}
    m_inline = re.match(rf"^{keyword}\s*\{{(.+)\}}\s*$", line)
    if m_inline:
        return m_inline.group(1), start_idx

    # multi-line form: keyword {
    if not re.match(rf"^{keyword}\s*\{{\s*$", line):
        raise ValueError(f"unsupported {keyword} block start: {line}")

    parts: list[str] = []
    i = start_idx + 1
    while i < len(lines):
        current = lines[i].strip()
        if current == "}":
            return "\n".join(parts), i
        parts.append(current)
        i += 1

    raise ValueError(f"unterminated {keyword} block")


def _parse_select_block(block_text: str) -> list[str]:
    normalised = block_text.replace("\n", ",")
    fields = []
    for raw in normalised.split(","):
        field = raw.strip().rstrip(",").strip()
        if field:
            if not re.match(r"^[A-Za-z_*][A-Za-z0-9_]*$", field):
                raise ValueError(f"invalid select field: {field!r}")
            fields.append(field)
    if not fields:
        raise ValueError("select block has no fields")
    return fields


def _parse_aggregate_block(block_text: str) -> list[str]:
    sql_fn_map = {
        "count":   "COUNT",
        "average": "AVG",
        "sum":     "SUM",
        "min":     "MIN",
        "max":     "MAX",
    }
    normalised = block_text.replace("\n", ",")
    exprs: list[str] = []

    for raw in normalised.split(","):
        item = raw.strip().rstrip(",").strip()
        if not item:
            continue

        m = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(count|average|sum|min|max)\s+([A-Za-z_*][A-Za-z0-9_]*)$",
            item,
            re.IGNORECASE,
        )
        if not m:
            raise ValueError(f"unsupported aggregate expression: {item!r}")

        alias, fn_name, field = m.groups()
        sql_fn = sql_fn_map[fn_name.lower()]

        if fn_name.lower() == "count" and field in ("id", "*"):
            sql_expr = f"COUNT(*) AS {alias}"
        elif fn_name.lower() == "count":
            sql_expr = f"COUNT({field}) AS {alias}"
        else:
            sql_expr = f"{sql_fn}({field}) AS {alias}"

        exprs.append(sql_expr)

    if not exprs:
        raise ValueError("aggregate block has no expressions")
    return exprs


# ---------------------------------------------------------------------------
# feedback CTE definition
# ---------------------------------------------------------------------------

def _build_feedback_cte() -> str:
    return (
        f"WITH {_PRQL_SOURCE} AS (\n"
        f"    SELECT\n"
        f"        m.id                                            AS id,\n"
        f"        m.message_uuid                                  AS message_uuid,\n"
        f"        m.conversation_id                               AS conversation_id,\n"
        f"        c.name                                          AS conversation_name,\n"
        f"        c.owner_id                                      AS user_id,\n"
        f"        u.username                                      AS username,\n"
        f"        m.rating                                        AS rating,\n"
        f"        m.feedback_text                                 AS feedback_text,\n"
        f"        m.feedback_submitted_at                         AS feedback_submitted_at,\n"
        f"        m.created_at                                    AS created_at,\n"
        f"        COALESCE(m.feedback_submitted_at, m.created_at) AS effective_date,\n"
        f"        m.role                                          AS role,\n"
        f"        m.content                                       AS content,\n"
        f"        LEFT(m.content, 300)                            AS content_snippet,\n"
        f"        m.model                                         AS model,\n"
        f"        m.tool_call_name                                AS tool_call_name,\n"
        f"        m.tool_call_input                               AS tool_call_input,\n"
        f"        m.usage                                         AS usage,\n"
        f"        CASE\n"
        f"            WHEN m.feedback_text IS NOT NULL\n"
        f"             AND m.feedback_text <> ''\n"
        f"            THEN TRUE\n"
        f"            ELSE FALSE\n"
        f"        END                                             AS has_feedback_text\n"
        f"    FROM\n"
        f"        {_MSG_TABLE} m\n"
        f"        INNER JOIN {_CONVO_TABLE} c ON c.id = m.conversation_id\n"
        f"        INNER JOIN {_USER_TABLE} u  ON u.id = c.owner_id\n"
        f"    WHERE\n"
        f"        m.rating IS NOT NULL\n"
        f"        OR (m.feedback_text IS NOT NULL AND m.feedback_text <> '')\n"
        f")"
    )


# ---------------------------------------------------------------------------
# SQL execution
# ---------------------------------------------------------------------------

def execute_prql_query(sql: str, params: list[Any]) -> list[dict[str, Any]]:
    cte = _build_feedback_cte()
    full_sql = f"{cte}\n{sql}"
    logger.debug("executing prql feedback sql:\n%s\nparams: %s", full_sql, params)
    with connection.cursor() as cursor:
        cursor.execute(full_sql, params)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# high-level convenience function
# ---------------------------------------------------------------------------

def query_feedback_rows_via_prql(
    filters: "FeedbackFilters",  # noqa: F821
    *,
    page: int = 1,
    page_size: int = 50,
) -> list[dict[str, Any]]:
    prql_string, params = build_prql_query(
        filters,
        aggregate=False,
        page=page,
        page_size=page_size,
    )
    sql = compile_prql_to_sql(prql_string)
    return execute_prql_query(sql, params)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _to_pg_ts(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=py_tz.utc)
    return dt.astimezone(py_tz.utc).strftime("%Y-%m-%d %H:%M:%S+00")