"""
tests for the prql query engine

covers:
    build_prql_query generation for rows and aggregates
    each filter type produces the correct prql clause
    exact_rating takes precedence over min and max
    compile_prql_to_sql produces valid sql with %s placeholders
    _replace placeholder logic via re.sub
    _build_feedback_cte structure and required fields
    execute_prql_query round-trip against real db
    query_feedback_rows_via_prql end-to-end against real data
    PRQLCompilationError raised on invalid prql
"""
import re

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.chat.models import Message, WSConversation
from apps.platform_admin.services.feedback_dataset import FeedbackFilters
from apps.platform_admin.services.feedback_prql import (
    PRQLCompilationError,
    _build_feedback_cte,
    _CONVO_TABLE,
    _MSG_TABLE,
    _USER_TABLE,
    build_prql_query,
    compile_prql_to_sql,
    execute_prql_query,
    query_feedback_rows_via_prql,
)

User = get_user_model()


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def make_user(username: str) -> User:
    return User.objects.create_user(
        username=username,
        email=f"{username}@test.com",
        password="testpass",
    )


def make_conversation(owner: User, name: str = "test convo") -> WSConversation:
    return WSConversation.objects.create(owner=owner, name=name)


def make_message(conversation: WSConversation, seq: int = 1, **kwargs) -> Message:
    defaults = dict(
        role="assistant",
        content="test content",
        sequence_number=seq,
        rating=None,
        feedback_text=None,
    )
    defaults.update(kwargs)
    return Message.objects.create(conversation=conversation, **defaults)


def _try_compile(prql_string: str) -> str | None:
    """
    attempt to compile prql, return sql string on success,
    return none if prql-python is not installed so tests skip gracefully
    """
    try:
        import prql_python  # noqa: F401
    except ImportError:
        return None
    return compile_prql_to_sql(prql_string)


# ---------------------------------------------------------------------------
# build_prql_query generation tests — no db needed
# ---------------------------------------------------------------------------

class BuildPRQLQueryRowsTests(TestCase):

    def test_empty_filters_generates_from_feedback(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        self.assertIn("from feedback", prql)
        self.assertEqual(params, [])

    def test_empty_filters_generates_sort(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        self.assertIn("sort", prql)

    def test_empty_filters_generates_take(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        self.assertIn("take", prql)

    def test_empty_filters_generates_select(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        self.assertIn("select", prql)

    def test_empty_filters_no_params(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        self.assertEqual(params, [])

    def test_user_id_filter_adds_clause_and_param(self):
        prql, params = build_prql_query(FeedbackFilters(user_id=42), aggregate=False)
        self.assertIn("user_id", prql)
        self.assertIn(42, params)

    def test_min_rating_filter_adds_clause_and_param(self):
        prql, params = build_prql_query(
            FeedbackFilters(min_rating=3), aggregate=False
        )
        self.assertIn("rating", prql)
        self.assertIn(3, params)

    def test_max_rating_filter_adds_clause_and_param(self):
        prql, params = build_prql_query(
            FeedbackFilters(max_rating=4), aggregate=False
        )
        self.assertIn("rating", prql)
        self.assertIn(4, params)

    def test_exact_rating_adds_clause_and_param(self):
        prql, params = build_prql_query(
            FeedbackFilters(exact_rating=5), aggregate=False
        )
        self.assertIn("rating", prql)
        self.assertIn(5, params)

    def test_exact_rating_excludes_min_and_max_from_params(self):
        # when exact_rating is set, min and max should not appear in params
        prql, params = build_prql_query(
            FeedbackFilters(exact_rating=5, min_rating=1, max_rating=3),
            aggregate=False,
        )
        self.assertIn(5, params)
        self.assertNotIn(1, params)
        self.assertNotIn(3, params)

    def test_feedback_text_search_adds_clause_and_wrapped_param(self):
        prql, params = build_prql_query(
            FeedbackFilters(feedback_text_search="good"), aggregate=False
        )
        self.assertIn("feedback_text", prql)
        # the value is wrapped in % for ilike matching
        self.assertIn("%good%", params)

    def test_conversation_name_search_adds_clause_and_wrapped_param(self):
        prql, params = build_prql_query(
            FeedbackFilters(conversation_name_search="alice"), aggregate=False
        )
        self.assertIn("conversation_name", prql)
        self.assertIn("%alice%", params)

    def test_role_filter_adds_clause_and_param(self):
        prql, params = build_prql_query(
            FeedbackFilters(role="assistant"), aggregate=False
        )
        self.assertIn("role", prql)
        self.assertIn("assistant", params)

    def test_model_filter_adds_clause_and_param(self):
        prql, params = build_prql_query(
            FeedbackFilters(model="gpt-4"), aggregate=False
        )
        self.assertIn("model", prql)
        self.assertIn("gpt-4", params)

    def test_tool_call_name_filter_adds_clause_and_param(self):
        prql, params = build_prql_query(
            FeedbackFilters(tool_call_name="search"), aggregate=False
        )
        self.assertIn("tool_call_name", prql)
        self.assertIn("search", params)

    def test_has_feedback_text_true_adds_literal_clause_no_param(self):
        prql, params = build_prql_query(
            FeedbackFilters(has_feedback_text=True), aggregate=False
        )
        self.assertIn("has_feedback_text", prql)
        self.assertIn("true", prql)
        # no param added since it is a literal boolean in the prql
        self.assertEqual(params, [])

    def test_has_feedback_text_false_adds_literal_clause_no_param(self):
        prql, params = build_prql_query(
            FeedbackFilters(has_feedback_text=False), aggregate=False
        )
        self.assertIn("has_feedback_text", prql)
        self.assertIn("false", prql)
        self.assertEqual(params, [])

    def test_has_feedback_text_none_adds_no_clause(self):
        prql, params = build_prql_query(
            FeedbackFilters(has_feedback_text=None), aggregate=False
        )
        self.assertNotIn("has_feedback_text", prql)

    def test_multiple_filters_produce_multiple_params(self):
        prql, params = build_prql_query(
            FeedbackFilters(user_id=1, min_rating=3, role="assistant"),
            aggregate=False,
        )
        self.assertEqual(len(params), 3)

    def test_pagination_page_1_offset_is_zero(self):
        prql, params = build_prql_query(
            FeedbackFilters(), aggregate=False, page=1, page_size=50
        )
        # page 1 with page_size 50 should be take {0..49}
        self.assertIn("{0..49}", prql)

    def test_pagination_page_2_offset_is_page_size(self):
        prql, params = build_prql_query(
            FeedbackFilters(), aggregate=False, page=2, page_size=50
        )
        # page 2 with page_size 50 should be take {50..99}
        self.assertIn("{50..99}", prql)

    def test_pagination_page_3_custom_size(self):
        prql, params = build_prql_query(
            FeedbackFilters(), aggregate=False, page=3, page_size=10
        )
        # page 3 with page_size 10 should be take {20..29}
        self.assertIn("{20..29}", prql)


class BuildPRQLQueryAggregateTests(TestCase):

    def test_aggregate_contains_aggregate_keyword(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=True)
        self.assertIn("aggregate", prql)

    def test_aggregate_does_not_contain_take(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=True)
        self.assertNotIn("take", prql)

    def test_aggregate_does_not_contain_select(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=True)
        self.assertNotIn("select", prql)

    def test_aggregate_contains_count(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=True)
        self.assertIn("count", prql)

    def test_aggregate_filters_still_applied(self):
        prql, params = build_prql_query(
            FeedbackFilters(min_rating=4), aggregate=True
        )
        self.assertIn("rating", prql)
        self.assertIn(4, params)


# ---------------------------------------------------------------------------
# compile_prql_to_sql tests
# ---------------------------------------------------------------------------

class CompilePRQLToSQLTests(TestCase):

    def test_simple_select_compiles(self):
        sql = _try_compile("from my_table\nselect {id, name}\n")
        if sql is None:
            self.skipTest("prql-python not installed")
        self.assertIn("SELECT", sql.upper())
        self.assertIn("my_table", sql)

    def test_compiled_sql_uses_percent_s_not_dollar_n(self):
        sql = _try_compile("from my_table\nfilter id == $1\nselect {id}\n")
        if sql is None:
            self.skipTest("prql-python not installed")
        self.assertIn("%s", sql)
        self.assertNotIn("$1", sql)
        self.assertNotIn("$2", sql)

    def test_multiple_placeholders_all_replaced(self):
        prql = (
            "from my_table\n"
            "filter id == $1\n"
            "filter name == $2\n"
            "select {id, name}\n"
        )
        sql = _try_compile(prql)
        if sql is None:
            self.skipTest("prql-python not installed")
        # count %s occurrences — should match number of placeholders
        count = sql.count("%s")
        self.assertEqual(count, 2)
        # no dollar placeholders should remain
        self.assertFalse(re.search(r"\$\d+", sql))

    def test_invalid_prql_raises_compilation_error(self):
        try:
            import prql_python  # noqa: F401
        except ImportError:
            self.skipTest("prql-python not installed")
        with self.assertRaises(PRQLCompilationError):
            compile_prql_to_sql("this is not valid prql %%%")

    def test_empty_filters_prql_compiles_without_error(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        sql = _try_compile(prql)
        if sql is None:
            self.skipTest("prql-python not installed")
        self.assertIsInstance(sql, str)
        self.assertGreater(len(sql), 0)

    def test_filtered_prql_compiles_without_error(self):
        prql, params = build_prql_query(
            FeedbackFilters(user_id=1, min_rating=3), aggregate=False
        )
        sql = _try_compile(prql)
        if sql is None:
            self.skipTest("prql-python not installed")
        self.assertIsInstance(sql, str)
        self.assertIn("%s", sql)

    def test_aggregate_prql_compiles_without_error(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=True)
        sql = _try_compile(prql)
        if sql is None:
            self.skipTest("prql-python not installed")
        self.assertIsInstance(sql, str)


# ---------------------------------------------------------------------------
# _build_feedback_cte structure tests
# ---------------------------------------------------------------------------

class FeedbackCTETests(TestCase):

    def test_cte_starts_with_with_keyword(self):
        cte = _build_feedback_cte()
        self.assertTrue(cte.strip().upper().startswith("WITH"))

    def test_cte_references_message_table(self):
        cte = _build_feedback_cte()
        self.assertIn(_MSG_TABLE, cte)

    def test_cte_references_conversation_table(self):
        cte = _build_feedback_cte()
        self.assertIn(_CONVO_TABLE, cte)

    def test_cte_references_user_table(self):
        cte = _build_feedback_cte()
        self.assertIn(_USER_TABLE, cte)

    def test_cte_defines_feedback_source(self):
        cte = _build_feedback_cte()
        self.assertIn("feedback", cte)

    def test_cte_exposes_required_fields(self):
        cte = _build_feedback_cte()
        required = [
            "message_uuid",
            "conversation_id",
            "conversation_name",
            "user_id",
            "username",
            "rating",
            "feedback_text",
            "effective_date",
            "role",
            "content_snippet",
            "has_feedback_text",
        ]
        for field in required:
            self.assertIn(field, cte, msg=f"cte missing field: {field}")

    def test_cte_filters_feedback_bearing_rows(self):
        cte = _build_feedback_cte()
        # the where clause should filter to rating not null or feedback_text
        self.assertIn("rating IS NOT NULL", cte)
        self.assertIn("feedback_text IS NOT NULL", cte)

    def test_cte_uses_coalesce_for_effective_date(self):
        cte = _build_feedback_cte()
        self.assertIn("COALESCE", cte)
        self.assertIn("feedback_submitted_at", cte)

    def test_cte_uses_left_for_content_snippet(self):
        cte = _build_feedback_cte()
        self.assertIn("LEFT", cte)
        self.assertIn("300", cte)

    def test_cte_uses_case_for_has_feedback_text(self):
        cte = _build_feedback_cte()
        self.assertIn("CASE", cte)
        self.assertIn("has_feedback_text", cte)


# ---------------------------------------------------------------------------
# end-to-end integration tests against real db
# ---------------------------------------------------------------------------

class ExecutePRQLQueryIntegrationTests(TestCase):
    """
    these tests execute real sql against the test database,
    they verify that the CTE + compiled sql round-trip works correctly
    """

    def setUp(self):
        self.user = make_user("prqluser")
        self.convo = make_conversation(self.user, name="prql test convo")
        self.msg1 = make_message(
            self.convo, seq=1, rating=5, feedback_text="excellent"
        )
        self.msg2 = make_message(
            self.convo, seq=2, rating=3, feedback_text="okay"
        )
        self.msg3 = make_message(
            self.convo, seq=3, rating=None, feedback_text="no rating"
        )

    def _skip_if_no_prql(self):
        try:
            import prql_python  # noqa: F401
        except ImportError:
            self.skipTest("prql-python not installed")

    def test_empty_filters_returns_all_feedback_rows(self):
        self._skip_if_no_prql()
        rows = query_feedback_rows_via_prql(FeedbackFilters())
        ids = {r["id"] for r in rows}
        self.assertIn(self.msg1.id, ids)
        self.assertIn(self.msg2.id, ids)
        self.assertIn(self.msg3.id, ids)

    def test_rows_have_expected_fields(self):
        self._skip_if_no_prql()
        rows = query_feedback_rows_via_prql(FeedbackFilters())
        self.assertGreater(len(rows), 0)
        row = rows[0]
        for field in (
            "id", "message_uuid", "conversation_id", "conversation_name",
            "user_id", "username", "rating", "feedback_text",
            "effective_date", "role", "content_snippet",
        ):
            self.assertIn(field, row, msg=f"missing field: {field}")

    def test_username_in_rows_matches_owner(self):
        self._skip_if_no_prql()
        rows = query_feedback_rows_via_prql(FeedbackFilters())
        for row in rows:
            self.assertEqual(row["username"], "prqluser")

    def test_rating_filter_narrows_results(self):
        self._skip_if_no_prql()
        rows = query_feedback_rows_via_prql(FeedbackFilters(exact_rating=5))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rating"], 5)

    def test_min_rating_filter_works(self):
        self._skip_if_no_prql()
        rows = query_feedback_rows_via_prql(FeedbackFilters(min_rating=4))
        for row in rows:
            self.assertGreaterEqual(row["rating"], 4)

    def test_feedback_text_search_works(self):
        self._skip_if_no_prql()
        rows = query_feedback_rows_via_prql(
            FeedbackFilters(feedback_text_search="excellent")
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], self.msg1.id)

    def test_pagination_page_size_one_returns_one_row(self):
        self._skip_if_no_prql()
        rows = query_feedback_rows_via_prql(
            FeedbackFilters(), page=1, page_size=1
        )
        self.assertEqual(len(rows), 1)

    def test_pagination_page_2_returns_different_row(self):
        self._skip_if_no_prql()
        rows_p1 = query_feedback_rows_via_prql(
            FeedbackFilters(), page=1, page_size=1
        )
        rows_p2 = query_feedback_rows_via_prql(
            FeedbackFilters(), page=2, page_size=1
        )
        self.assertNotEqual(rows_p1[0]["id"], rows_p2[0]["id"])

    def test_user_filter_works(self):
        self._skip_if_no_prql()
        other_user = make_user("otherprqluser")
        other_convo = make_conversation(other_user)
        make_message(other_convo, seq=1, rating=1, feedback_text="bad")

        rows = query_feedback_rows_via_prql(
            FeedbackFilters(user_id=self.user.id)
        )
        ids = {r["id"] for r in rows}
        # all rows belong to self.user
        self.assertIn(self.msg1.id, ids)
        # other user's row should not appear
        for row in rows:
            self.assertEqual(row["user_id"], self.user.id)

    def test_execute_prql_query_direct(self):
        self._skip_if_no_prql()
        # build and compile a simple query manually
        prql = (
            f"from feedback\n"
            f"sort [effective_date, id]\n"
            f"take {{0..99}}\n"
            f"select {{id, username, rating}}\n"
        )
        sql = compile_prql_to_sql(prql)
        rows = execute_prql_query(sql, [])
        self.assertIsInstance(rows, list)
        if rows:
            self.assertIn("id", rows[0])
            self.assertIn("username", rows[0])
            self.assertIn("rating", rows[0])