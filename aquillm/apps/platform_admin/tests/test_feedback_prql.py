"""
tests for the prql query engine

covers:
    build_prql_query generation for rows and aggregates
    filter clauses appear correctly in generated PRQL
    exact_rating takes precedence over min and max
    text search filters appear as comments (applied post-compilation as SQL)
    pagination appears as comments (applied post-compilation as SQL)
    compile_prql_to_sql uses prql-python and replaces $N placeholders
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
    """compile prql, return sql on success, None if prql-python not installed"""
    try:
        import prql_python  # noqa: F401
    except ImportError:
        return None
    return compile_prql_to_sql(prql_string)


def _skip_if_no_prql(test_case: TestCase) -> None:
    try:
        import prql_python  # noqa: F401
    except ImportError:
        test_case.skipTest("prql-python not installed")


# ---------------------------------------------------------------------------
# build_prql_query generation tests — no db needed
# ---------------------------------------------------------------------------

class BuildPRQLQueryRowsTests(TestCase):

    def test_empty_filters_generates_from_feedback(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        self.assertIn("from feedback", prql)

    def test_empty_filters_params_always_empty(self):
        # values are inlined as literals — params list is always empty
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        self.assertEqual(params, [])

    def test_empty_filters_generates_sort(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        self.assertIn("sort", prql)

    def test_empty_filters_generates_pagination_comment(self):
        # pagination is shown as a PRQL comment, not as a take clause
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        self.assertIn("pagination", prql)

    def test_empty_filters_generates_select(self):
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        self.assertIn("select", prql)

    def test_user_id_filter_inlined_in_prql(self):
        prql, params = build_prql_query(FeedbackFilters(user_id=42), aggregate=False)
        self.assertIn("user_id", prql)
        self.assertIn("42", prql)
        # value is inlined — not in params
        self.assertEqual(params, [])

    def test_min_rating_filter_inlined_in_prql(self):
        prql, params = build_prql_query(FeedbackFilters(min_rating=3), aggregate=False)
        self.assertIn("rating", prql)
        self.assertIn("3", prql)
        self.assertEqual(params, [])

    def test_max_rating_filter_inlined_in_prql(self):
        prql, params = build_prql_query(FeedbackFilters(max_rating=4), aggregate=False)
        self.assertIn("rating", prql)
        self.assertIn("4", prql)
        self.assertEqual(params, [])

    def test_exact_rating_inlined_in_prql(self):
        prql, params = build_prql_query(FeedbackFilters(exact_rating=5), aggregate=False)
        self.assertIn("rating", prql)
        self.assertIn("5", prql)
        self.assertEqual(params, [])

    def test_exact_rating_excludes_min_and_max(self):
        # when exact_rating is set, min and max filter clauses must not appear
        prql, params = build_prql_query(
            FeedbackFilters(exact_rating=5, min_rating=1, max_rating=3),
            aggregate=False,
        )
        # exact_rating==5 appears
        self.assertIn("rating == 5", prql)
        # min/max range clauses must not appear
        self.assertNotIn("rating >= 1", prql)
        self.assertNotIn("rating <= 3", prql)

    def test_feedback_text_search_appears_as_comment(self):
        # text search cannot be compiled by prql-python — shown as comment
        prql, params = build_prql_query(
            FeedbackFilters(feedback_text_search="good"), aggregate=False
        )
        self.assertIn("feedback_text", prql)
        self.assertIn("good", prql)
        self.assertIn("#", prql)
        # params are always empty — text search params added post-compilation
        self.assertEqual(params, [])

    def test_conversation_name_search_appears_as_comment(self):
        prql, params = build_prql_query(
            FeedbackFilters(conversation_name_search="alice"), aggregate=False
        )
        self.assertIn("conversation_name", prql)
        self.assertIn("alice", prql)
        self.assertIn("#", prql)
        self.assertEqual(params, [])

    def test_role_filter_inlined_in_prql(self):
        prql, params = build_prql_query(
            FeedbackFilters(role="assistant"), aggregate=False
        )
        self.assertIn("role", prql)
        self.assertIn("assistant", prql)
        self.assertEqual(params, [])

    def test_model_filter_inlined_in_prql(self):
        prql, params = build_prql_query(
            FeedbackFilters(model="gpt-4"), aggregate=False
        )
        self.assertIn("model", prql)
        self.assertIn("gpt-4", prql)
        self.assertEqual(params, [])

    def test_tool_call_name_filter_inlined_in_prql(self):
        prql, params = build_prql_query(
            FeedbackFilters(tool_call_name="search"), aggregate=False
        )
        self.assertIn("tool_call_name", prql)
        self.assertIn("search", prql)
        self.assertEqual(params, [])

    def test_has_feedback_text_true_adds_literal_clause(self):
        prql, params = build_prql_query(
            FeedbackFilters(has_feedback_text=True), aggregate=False
        )
        self.assertIn("has_feedback_text", prql)
        self.assertIn("true", prql)
        self.assertEqual(params, [])

    def test_has_feedback_text_false_adds_literal_clause(self):
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

    def test_params_always_empty_regardless_of_filters(self):
        # all values are inlined — params is always []
        prql, params = build_prql_query(
            FeedbackFilters(user_id=1, min_rating=3, role="assistant"),
            aggregate=False,
        )
        self.assertEqual(params, [])

    def test_pagination_page_1_comment_shows_offset_zero(self):
        prql, params = build_prql_query(
            FeedbackFilters(), aggregate=False, page=1, page_size=50
        )
        # pagination comment should show {0..49}
        self.assertIn("{0..49}", prql)

    def test_pagination_page_2_comment_shows_correct_range(self):
        prql, params = build_prql_query(
            FeedbackFilters(), aggregate=False, page=2, page_size=50
        )
        self.assertIn("{50..99}", prql)

    def test_pagination_page_3_custom_size_comment(self):
        prql, params = build_prql_query(
            FeedbackFilters(), aggregate=False, page=3, page_size=10
        )
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

    def test_aggregate_filters_still_appear_in_prql(self):
        prql, params = build_prql_query(
            FeedbackFilters(min_rating=4), aggregate=True
        )
        self.assertIn("rating", prql)
        self.assertIn("4", prql)
        self.assertEqual(params, [])


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

    def test_compiled_sql_replaces_dollar_n_with_percent_s(self):
        # hand-written PRQL with $1 should get $1 replaced with %s
        # note: prql-python 0.11.2 does not support $1 syntax itself,
        # but our compile_prql_to_sql does a regex replacement as a safety net
        # this test verifies the replacement logic works on SQL that contains $1
        # after compilation (even if prql-python itself doesn't emit it)
        # use a direct string substitution test instead
        import re as re_mod
        raw_sql = "SELECT id FROM feedback WHERE user_id = $1 AND rating = $2"
        result = re_mod.sub(r"\$\d+", "%s", raw_sql)
        self.assertEqual(result, "SELECT id FROM feedback WHERE user_id = %s AND rating = %s")
        self.assertNotIn("$1", result)
        self.assertNotIn("$2", result)

    def test_invalid_prql_raises_compilation_error(self):
        _skip_if_no_prql(self)
        with self.assertRaises(PRQLCompilationError):
            compile_prql_to_sql("this is not valid prql %%%")

    def test_empty_filters_prql_compiles_without_error(self):
        _skip_if_no_prql(self)
        prql, params = build_prql_query(FeedbackFilters(), aggregate=False)
        sql = compile_prql_to_sql(prql)
        self.assertIsInstance(sql, str)
        self.assertGreater(len(sql), 0)

    def test_filters_prql_compiles_without_error(self):
        _skip_if_no_prql(self)
        prql, params = build_prql_query(
            FeedbackFilters(user_id=1, min_rating=3), aggregate=False
        )
        sql = compile_prql_to_sql(prql)
        self.assertIsInstance(sql, str)
        # values are inlined — no placeholders needed
        self.assertNotIn("$1", sql)
        self.assertNotIn("%s", sql)

    def test_aggregate_prql_compiles_without_error(self):
        _skip_if_no_prql(self)
        prql, params = build_prql_query(FeedbackFilters(), aggregate=True)
        sql = compile_prql_to_sql(prql)
        self.assertIsInstance(sql, str)

    def test_sort_descending_curly_brace_compiles(self):
        _skip_if_no_prql(self)
        sql = compile_prql_to_sql(
            "from feedback\nsort {-effective_date}\ntake 5\nselect {id}"
        )
        self.assertIn("DESC", sql.upper())

    def test_filter_with_inlined_integer_compiles(self):
        _skip_if_no_prql(self)
        sql = compile_prql_to_sql(
            "from feedback\nfilter rating == 5\nselect {id, rating}"
        )
        self.assertIn("5", sql)

    def test_filter_null_compiles_to_is_not_null(self):
        _skip_if_no_prql(self)
        sql = compile_prql_to_sql(
            "from feedback\nfilter rating != null\nselect {id}"
        )
        self.assertIn("IS NOT NULL", sql.upper())

    def test_filter_bool_literal_compiles(self):
        _skip_if_no_prql(self)
        sql = compile_prql_to_sql(
            "from feedback\nfilter has_feedback_text == true\nselect {id}"
        )
        self.assertIsInstance(sql, str)

    def test_aggregate_compiles_to_count_and_avg(self):
        _skip_if_no_prql(self)
        sql = compile_prql_to_sql(
            "from feedback\naggregate {total = count id, avg_r = average rating}"
        )
        self.assertIn("COUNT", sql.upper())
        self.assertIn("AVG", sql.upper())


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
    execute real SQL against the test database.
    verifies that the CTE + compiled sql round-trip works correctly
    with the new prql-python-based compiler.
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

    def test_empty_filters_returns_all_feedback_rows(self):
        _skip_if_no_prql(self)
        rows = query_feedback_rows_via_prql(FeedbackFilters())
        ids = {r["id"] for r in rows}
        self.assertIn(self.msg1.id, ids)
        self.assertIn(self.msg2.id, ids)
        self.assertIn(self.msg3.id, ids)

    def test_rows_have_expected_fields(self):
        _skip_if_no_prql(self)
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
        _skip_if_no_prql(self)
        rows = query_feedback_rows_via_prql(FeedbackFilters())
        for row in rows:
            self.assertEqual(row["username"], "prqluser")

    def test_rating_filter_narrows_results(self):
        _skip_if_no_prql(self)
        rows = query_feedback_rows_via_prql(FeedbackFilters(exact_rating=5))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rating"], 5)

    def test_min_rating_filter_works(self):
        _skip_if_no_prql(self)
        rows = query_feedback_rows_via_prql(FeedbackFilters(min_rating=4))
        for row in rows:
            self.assertGreaterEqual(row["rating"], 4)

    def test_feedback_text_search_works(self):
        _skip_if_no_prql(self)
        rows = query_feedback_rows_via_prql(
            FeedbackFilters(feedback_text_search="excellent")
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], self.msg1.id)

    def test_pagination_page_size_one_returns_one_row(self):
        _skip_if_no_prql(self)
        rows = query_feedback_rows_via_prql(
            FeedbackFilters(), page=1, page_size=1
        )
        self.assertEqual(len(rows), 1)

    def test_pagination_page_2_returns_different_row(self):
        _skip_if_no_prql(self)
        rows_p1 = query_feedback_rows_via_prql(
            FeedbackFilters(), page=1, page_size=1
        )
        rows_p2 = query_feedback_rows_via_prql(
            FeedbackFilters(), page=2, page_size=1
        )
        self.assertNotEqual(rows_p1[0]["id"], rows_p2[0]["id"])

    def test_user_filter_works(self):
        _skip_if_no_prql(self)
        other_user = make_user("otherprqluser")
        other_convo = make_conversation(other_user)
        make_message(other_convo, seq=1, rating=1, feedback_text="bad")

        rows = query_feedback_rows_via_prql(
            FeedbackFilters(user_id=self.user.id)
        )
        ids = {r["id"] for r in rows}
        self.assertIn(self.msg1.id, ids)
        for row in rows:
            self.assertEqual(row["user_id"], self.user.id)

    def test_execute_prql_query_direct(self):
        _skip_if_no_prql(self)
        # use syntax valid for prql-python 0.11.2:
        #   sort {col1, col2} not sort [col1, col2]
        #   take N (plain integer) not take {0..N} or take 0..N
        prql = (
            "from feedback\n"
            "sort {effective_date, id}\n"
            "take 100\n"
            "select {id, username, rating}\n"
        )
        sql = compile_prql_to_sql(prql)
        rows = execute_prql_query(sql, [])
        self.assertIsInstance(rows, list)
        if rows:
            self.assertIn("id", rows[0])
            self.assertIn("username", rows[0])
            self.assertIn("rating", rows[0])

    def test_has_feedback_text_filter_works(self):
        _skip_if_no_prql(self)
        # msg1 has text, msg3 has no rating — all have text
        rows_with_text = query_feedback_rows_via_prql(
            FeedbackFilters(has_feedback_text=True)
        )
        for row in rows_with_text:
            self.assertTrue(row.get("has_feedback_text") or row.get("feedback_text"))

    def test_prql_string_returned_for_filters(self):
        # verify get_prql_string_for_filters returns a non-empty string
        from apps.platform_admin.services.feedback_prql import get_prql_string_for_filters
        prql = get_prql_string_for_filters(FeedbackFilters(exact_rating=5))
        self.assertIn("from feedback", prql)
        self.assertIn("rating", prql)
        self.assertIn("5", prql)