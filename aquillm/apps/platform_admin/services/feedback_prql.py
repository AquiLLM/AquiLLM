"""
prql-backed query engine for the feedback dashboard

architecture
------------
FeedbackFilters
    -> build_prql_query()          generates valid PRQL for prql-python 0.11.2
    -> compile_prql_to_sql()       compiles via prql_python.compile()
    -> _merge_ctes_with_feedback() merges feedback CTE with any prql-python CTEs
    -> _add_pagination_sql()       appends LIMIT/OFFSET after compilation
    -> _add_text_search_sql()      appends ILIKE WHERE clauses after compilation
    -> _build_feedback_cte()       defines the feedback virtual table as a CTE
    -> execute_prql_query()        runs full SQL against django db connection

prql-python 0.11.2 compatibility constraints
--------------------------------------------
DOES NOT support:
    take {0..49}        curly-brace ranges rejected
    take 0..49          zero-start ranges rejected
    filter col ~* $1    ~* operator not recognised
    filter col == $1    positional placeholders not supported
    filter col == $name named placeholders not supported
    sort [col1, col2]   bracket arrays not supported in sort

DOES support:
    take N              plain integer limit
    sort {col1}         curly brace sort ascending
    sort {-col1}        curly brace sort descending
    sort {col1, col2}   multi-column sort
    filter col == val   literal values inline
    filter col != null  null checks
    filter col == true  boolean literals
    aggregate {alias = count id, alias = average col}

CTE conflict handling
---------------------
when prql-python compiles a query with sort or take, it may emit its own
WITH clause:
    WITH table_0 AS (SELECT ... FROM feedback ...) SELECT ...

prepending WITH feedback AS (...) would produce two WITH clauses, which
postgresql does not allow. _merge_ctes_with_feedback() detects this and
merges both sets of CTEs into a single WITH clause:
    WITH feedback AS (...), table_0 AS (...) SELECT ...
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone as py_tz
from typing import Any

from django.db import connection

logger = logging.getLogger(__name__)

_MSG_TABLE   = "aquillm_message"
_CONVO_TABLE = "aquillm_wsconversation"
_USER_TABLE  = "auth_user"
_PRQL_SOURCE = "feedback"

# safe string pattern — only these characters are allowed in inlined string literals
_SAFE_STR_RE = re.compile(r"^[A-Za-z0-9 ._\-:/]+$")


# ---------------------------------------------------------------------------
# value inlining helpers
# ---------------------------------------------------------------------------

def _inline_str(value: str) -> str:
    """
    inline a string value safely into PRQL as a quoted literal.
    rejects any value containing characters outside a safe whitelist.
    """
    if not _SAFE_STR_RE.match(value):
        raise ValueError(
            f"filter string value contains unsafe characters: {value!r}. "
            "only letters, digits, spaces, and ._-:/ are allowed."
        )
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _inline_int(value: int) -> str:
    """inline an integer value into PRQL as a literal"""
    return str(int(value))


def _inline_ts(dt: datetime) -> str:
    """inline a datetime as a quoted ISO timestamp string in PRQL"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=py_tz.utc)
    ts = dt.astimezone(py_tz.utc).strftime("%Y-%m-%d %H:%M:%S")
    return f"@{ts}"


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
    """
    build a PRQL string from FeedbackFilters.

    returns (prql_string, params) where params is always [] because all
    filter values are inlined as safe literals in the PRQL string.
    pagination and text search are applied as SQL post-compilation.
    text search filters are shown as PRQL comments for display accuracy.
    """
    filter_clauses: list[str] = []

    if filters.start_date is not None:
        ts = _inline_ts(filters.start_date)
        filter_clauses.append(f"filter effective_date >= {ts}")

    if filters.end_date is not None:
        ts = _inline_ts(filters.end_date)
        filter_clauses.append(f"filter effective_date <= {ts}")

    if filters.user_id is not None:
        filter_clauses.append(f"filter user_id == {_inline_int(filters.user_id)}")

    if filters.exact_rating is not None:
        filter_clauses.append(f"filter rating == {_inline_int(filters.exact_rating)}")
    else:
        if filters.min_rating is not None:
            filter_clauses.append(f"filter rating >= {_inline_int(filters.min_rating)}")
        if filters.max_rating is not None:
            filter_clauses.append(f"filter rating <= {_inline_int(filters.max_rating)}")

    if filters.role:
        filter_clauses.append(f"filter role == {_inline_str(filters.role)}")

    if filters.model:
        filter_clauses.append(f"filter model == {_inline_str(filters.model)}")

    if filters.tool_call_name:
        filter_clauses.append(
            f"filter tool_call_name == {_inline_str(filters.tool_call_name)}"
        )

    if filters.has_feedback_text is True:
        filter_clauses.append("filter has_feedback_text == true")
    elif filters.has_feedback_text is False:
        filter_clauses.append("filter has_feedback_text == false")

    text_comments: list[str] = []
    if filters.feedback_text_search:
        safe = filters.feedback_text_search.replace("'", "''")
        text_comments.append(f"# text search: feedback_text ILIKE '%{safe}%'")

    if filters.conversation_name_search:
        safe = filters.conversation_name_search.replace("'", "''")
        text_comments.append(f"# text search: conversation_name ILIKE '%{safe}%'")

    offset   = max(0, (page - 1) * page_size)
    take_end = offset + page_size - 1
    if offset == 0:
        page_comment = f"# pagination: take {{{offset}..{take_end}}}  (LIMIT {page_size})"
    else:
        page_comment = (
            f"# pagination: take {{{offset}..{take_end}}}  "
            f"(LIMIT {page_size} OFFSET {offset})"
        )

    if aggregate:
        prql = _build_aggregate_prql(filter_clauses, text_comments)
    else:
        prql = _build_rows_prql(filter_clauses, text_comments, page_comment)

    return prql, []


def _build_rows_prql(
    filter_clauses: list[str],
    text_comments: list[str],
    page_comment: str,
) -> str:
    filter_block = "\n".join(filter_clauses)
    if filter_block:
        filter_block += "\n"

    comment_block = "\n".join(text_comments)
    if comment_block:
        comment_block += "\n"

    return (
        f"from {_PRQL_SOURCE}\n"
        f"{filter_block}"
        f"{comment_block}"
        f"{page_comment}\n"
        f"sort {{effective_date, id}}\n"
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


def _build_aggregate_prql(
    filter_clauses: list[str],
    text_comments: list[str],
) -> str:
    filter_block = "\n".join(filter_clauses)
    if filter_block:
        filter_block += "\n"

    comment_block = "\n".join(text_comments)
    if comment_block:
        comment_block += "\n"

    return (
        f"from {_PRQL_SOURCE}\n"
        f"{filter_block}"
        f"{comment_block}"
        f"aggregate {{\n"
        f"  total_count = count id,\n"
        f"  avg_rating = average rating,\n"
        f"}}\n"
    )


# ---------------------------------------------------------------------------
# PRQL compilation
# ---------------------------------------------------------------------------

class PRQLCompilationError(Exception):
    """raised when prql-python fails to compile a PRQL string"""


def compile_prql_to_sql(prql_string: str, *, dialect: str = "postgres") -> str:
    """
    compile a PRQL string to SQL using prql-python.

    strips comment lines (starting with #) before compilation because our
    pagination and text-search comments are for display only and prql-python
    does not support # comments in all positions.

    replaces $N positional placeholders with %s for psycopg2 compatibility,
    retained for callers that pass hand-written PRQL with positional params.

    raises PRQLCompilationError if prql-python rejects the input.
    """
    try:
        import prql_python as prql_lib
    except ImportError as exc:
        raise PRQLCompilationError(
            "prql-python is not installed — add prql-python to requirements.txt"
        ) from exc

    clean_lines = [
        line for line in prql_string.splitlines()
        if not line.strip().startswith("#")
    ]
    clean_prql = "\n".join(clean_lines)

    try:
        options = prql_lib.CompileOptions(
            target=f"sql.{dialect}",
            signature_comment=False,
            format=False,
        )
        sql = prql_lib.compile(clean_prql, options)
    except Exception as exc:
        raise PRQLCompilationError(f"prql compilation failed: {exc}") from exc

    sql = re.sub(r"\$\d+", "%s", sql)
    return sql


# ---------------------------------------------------------------------------
# CTE merge
# ---------------------------------------------------------------------------

def _merge_ctes_with_feedback(compiled_sql: str) -> str:
    """
    merge the feedback CTE with any CTEs already emitted by prql-python.

    prql-python 0.11.2 wraps sorted/paginated queries in its own WITH clause:
        WITH table_0 AS (SELECT ... FROM feedback ORDER BY ...) SELECT ...

    prepending our WITH feedback AS (...) would produce two WITH clauses,
    which postgresql rejects. this function merges them into one:
        WITH feedback AS (...), table_0 AS (...) SELECT ...

    when the compiled SQL has no WITH clause, the feedback CTE is simply
    prepended as-is.
    """
    feedback_cte_body = _build_feedback_cte()
    # feedback_cte_body is: "WITH feedback AS (\n    SELECT ...\n)"
    # extract "feedback AS (\n    SELECT ...\n)" — everything after "WITH "
    feedback_def = feedback_cte_body[len("WITH "):].strip()

    compiled_stripped = compiled_sql.strip()

    if not compiled_stripped.upper().startswith("WITH "):
        # no CTE in compiled output — simple prepend
        return f"{feedback_cte_body}\n{compiled_sql}"

    # compiled SQL has its own WITH clause — merge
    # compiled looks like: "WITH table_0 AS (...) SELECT ..."
    # we want:             "WITH feedback AS (...), table_0 AS (...) SELECT ..."
    rest_of_compiled = compiled_stripped[len("WITH "):].strip()
    merged = f"WITH {feedback_def},\n{rest_of_compiled}"
    return merged


# ---------------------------------------------------------------------------
# post-compilation SQL transformations
# ---------------------------------------------------------------------------

def _add_pagination_sql(
    base_sql: str,
    page: int,
    page_size: int,
    existing_params: list[Any],
) -> tuple[str, list[Any]]:
    """
    wrap the compiled SQL with LIMIT and OFFSET for pagination.

    we cannot express offset pagination in prql-python 0.11.2 because
    take 0..N is rejected for zero-start ranges. we apply it as SQL here.
    the base_sql passed here already has the feedback CTE merged in via
    _merge_ctes_with_feedback, so we must NOT add another CTE wrapper.
    we wrap the whole thing in a subquery instead.
    """
    offset = max(0, (page - 1) * page_size)
    final_params = list(existing_params) + [page_size, offset]
    final_sql = (
        f"SELECT * FROM ({base_sql}) AS _prql_page\n"
        f"LIMIT %s OFFSET %s"
    )
    return final_sql, final_params


def _add_text_search_sql(
    base_sql: str,
    filters: "FeedbackFilters",  # noqa: F821
    existing_params: list[Any],
) -> tuple[str, list[Any]]:
    """
    wrap the compiled SQL with ILIKE WHERE clauses for text search.

    prql-python 0.11.2 does not support the ~* operator or any ILIKE
    equivalent. these are applied as parameterised SQL post-compilation.

    call this before _add_pagination_sql so text search is in the inner
    subquery and pagination is applied to the already-filtered result set.
    """
    clauses: list[str] = []
    params: list[Any] = list(existing_params)

    if filters.feedback_text_search:
        clauses.append("feedback_text ILIKE %s")
        params.append(f"%{filters.feedback_text_search}%")

    if filters.conversation_name_search:
        clauses.append("conversation_name ILIKE %s")
        params.append(f"%{filters.conversation_name_search}%")

    if not clauses:
        return base_sql, existing_params

    where = " AND ".join(clauses)
    final_sql = (
        f"SELECT * FROM ({base_sql}) AS _prql_search\n"
        f"WHERE {where}"
    )
    return final_sql, params


# ---------------------------------------------------------------------------
# feedback CTE definition
# ---------------------------------------------------------------------------

def _build_feedback_cte() -> str:
    """
    build the postgresql WITH clause that defines the feedback virtual table.

    prql compiles `from feedback` to `SELECT ... FROM feedback`.
    this CTE is merged into the compiled SQL so that reference resolves.
    the field set matches the dataset contract in feedback_dataset.py exactly.
    """
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

def execute_prql_query(
    sql: str,
    params: list[Any],
) -> list[dict[str, Any]]:
    """
    execute compiled SQL, merging the feedback CTE with any CTEs emitted
    by prql-python to avoid duplicate WITH clause errors in postgresql.

    the sql argument must have been produced by compile_prql_to_sql() and
    optionally further modified by _add_text_search_sql() or
    _add_pagination_sql(). the feedback CTE is injected here via
    _merge_ctes_with_feedback() rather than simple string prepending.
    """
    full_sql = _merge_ctes_with_feedback(sql)
    logger.debug("executing prql feedback sql:\n%s\nparams: %s", full_sql, params)
    with connection.cursor() as cursor:
        cursor.execute(full_sql, params)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# high-level convenience functions
# ---------------------------------------------------------------------------

def query_feedback_rows_via_prql(
    filters: "FeedbackFilters",  # noqa: F821
    *,
    page: int = 1,
    page_size: int = 50,
) -> list[dict[str, Any]]:
    """
    build PRQL from filters, compile via prql-python, apply text search
    and pagination as SQL post-compilation, execute with merged CTE, return rows.
    """
    prql_string, _ = build_prql_query(
        filters,
        aggregate=False,
        page=page,
        page_size=page_size,
    )
    sql = compile_prql_to_sql(prql_string)

    # text search first (inner subquery), then pagination (outermost)
    sql, params = _add_text_search_sql(sql, filters, [])
    sql, params = _add_pagination_sql(sql, page, page_size, params)

    return execute_prql_query(sql, params)


def get_prql_string_for_filters(
    filters: "FeedbackFilters",  # noqa: F821
    *,
    page: int = 1,
    page_size: int = 50,
) -> str:
    """
    return the canonical PRQL string for the current filter state.
    used by the rows API endpoint to return live PRQL to the frontend.
    does not compile or execute — display only.
    """
    prql_string, _ = build_prql_query(
        filters,
        aggregate=False,
        page=page,
        page_size=page_size,
    )
    return prql_string


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _to_pg_ts(dt: datetime) -> str:
    """convert a datetime to a postgresql-compatible iso timestamp string"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=py_tz.utc)
    return dt.astimezone(py_tz.utc).strftime("%Y-%m-%d %H:%M:%S+00")