"""Unit tests for the feedback dashboard link tool.

These tests cover the server-side pipeline that the LLM's JSON spec
flows through:
  - Auto-fix application (null filter, role filter)
  - Query-string assembly from JSON
  - Parser validation propagation
  - JSON validation and error messages
  - Token URL minting

They do NOT cover the LLM's natural-language → JSON step — that's
non-deterministic and tested manually (see skills/feedback/MANUAL_TESTS.md).
"""
from __future__ import annotations

import json

import pytest

from apps.platform_admin.feedbackql.link_tool import build_feedback_link_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tool(db):
    """Build a feedback link tool with a fixed base URL. The `db` fixture is
    pulled in because mint_token writes to Django's cache; without a Django
    DB fixture loaded the cache backend may not be initialised in some
    configurations."""
    return build_feedback_link_tool(base_url="http://test.example.org")


def _call(tool, **spec) -> dict:
    """Convenience: serialise the kwargs as JSON and invoke the tool."""
    return tool(query_spec=json.dumps(spec))


def _query(tool, **spec) -> str:
    """Return the generated query string, asserting no exception."""
    r = _call(tool, **spec)
    assert "exception" not in r, f"Unexpected exception: {r.get('exception')}"
    return r["result"]["query"]


# ---------------------------------------------------------------------------
# Query-string assembly
# ---------------------------------------------------------------------------

class TestQueryAssembly:
    def test_simplest_query(self, tool):
        q = _query(tool, stream="messages")
        assert q == "messages"

    def test_single_where(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "user_id", "op": "==", "value": 7}],
        )
        assert q == 'messages\n| where user_id == 7'

    def test_multiple_where_anded(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[
                {"field": "user_id", "op": "==", "value": 7},
                {"field": "sequence_number", "op": ">", "value": 0},
            ],
        )
        assert "user_id == 7 and sequence_number > 0" in q

    def test_in_operator_with_list(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "user_id", "op": "in", "value": [1, 2, 3]}],
        )
        assert "user_id in [1, 2, 3]" in q

    def test_null_value_equality(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "tool_call_name", "op": "==", "value": None}],
        )
        assert "tool_call_name == null" in q

    def test_string_value_double_quoted(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "model", "op": "==", "value": "gpt-4o"}],
        )
        assert 'model == "gpt-4o"' in q

    def test_summarize_with_by(self, tool):
        q = _query(
            tool,
            stream="conversations",
            summarize={
                "aggregations": [{"alias": "n", "func": "count"}],
                "by": ["user_id"],
            },
        )
        assert "summarize n = count() by user_id" in q

    def test_summarize_global_no_by(self, tool):
        q = _query(
            tool,
            stream="conversations",
            summarize={
                "aggregations": [{"alias": "n", "func": "count"}],
            },
        )
        assert "summarize n = count()" in q
        assert " by " not in q

    def test_having_appears_after_summarize(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "rating", "op": "!=", "value": None}],
            summarize={
                "aggregations": [{"alias": "n", "func": "count"}],
                "by": ["user_id"],
            },
            having=[{"field": "n", "op": ">=", "value": 3}],
        )
        # Order matters: summarize first, then post-summarize where for having
        summarize_pos = q.find("summarize")
        having_pos = q.find("n >= 3")
        assert summarize_pos < having_pos

    def test_select_with_multiple_fields(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "user_id", "op": "==", "value": 7}],
            select=["rating", "content"],
        )
        assert "select rating, content" in q

    def test_order_by_default_direction_is_asc(self, tool):
        # When direction is omitted, asc is the default — query is valid.
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "rating", "op": "!=", "value": None}],
            order_by={"field": "rating"},
        )
        assert "order by rating asc" in q

    def test_order_by_bad_direction_rejected(self, tool):
        r = _call(
            tool,
            stream="messages",
            order_by={"field": "rating", "direction": "ascending"},
        )
        assert "exception" in r

    def test_limit_appears_at_end(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "user_id", "op": "==", "value": 7}],
            limit=25,
        )
        assert q.rstrip().endswith("| limit 25")


# ---------------------------------------------------------------------------
# Auto-fix: null filter on null-prone fields
# ---------------------------------------------------------------------------

class TestNullAutoFix:
    def test_order_by_rating_adds_null_filter(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "role", "op": "==", "value": "assistant"}],
            order_by={"field": "rating", "direction": "desc"},
            limit=10,
        )
        assert "rating != null" in q

    def test_group_by_tool_call_name_adds_null_filter(self, tool):
        q = _query(
            tool,
            stream="messages",
            summarize={
                "aggregations": [{"alias": "n", "func": "count"}],
                "by": ["tool_call_name"],
            },
        )
        assert "tool_call_name != null" in q

    def test_aggregate_on_rating_adds_null_filter(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "role", "op": "==", "value": "assistant"}],
            summarize={
                "aggregations": [{"alias": "avg_r", "func": "avg", "field": "rating"}]
            },
        )
        assert "rating != null" in q

    def test_explicit_rating_filter_prevents_autofix(self, tool):
        """User explicitly wants rating == null — auto-fix should NOT override."""
        q = _query(
            tool,
            stream="messages",
            where=[
                {"field": "role", "op": "==", "value": "assistant"},
                {"field": "rating", "op": "==", "value": None},
            ],
            order_by={"field": "rating", "direction": "desc"},
        )
        # The explicit filter stays; auto-fix doesn't add a contradicting != null
        assert "rating == null" in q
        assert "rating != null" not in q

    def test_explicit_rating_range_filter_prevents_autofix(self, tool):
        """A range filter (e.g. rating >= 4) implicitly excludes nulls."""
        q = _query(
            tool,
            stream="messages",
            where=[
                {"field": "role", "op": "==", "value": "assistant"},
                {"field": "rating", "op": ">=", "value": 4},
            ],
            order_by={"field": "rating", "direction": "desc"},
        )
        # No redundant != null
        assert "rating != null" not in q

    def test_select_only_does_not_trigger_null_filter(self, tool):
        """select-only references don't trigger auto-fix (too aggressive)."""
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "user_id", "op": "==", "value": 7}],
            select=["rating", "content"],
        )
        # rating is in select but not in order_by/summarize → no auto-add
        assert "rating != null" not in q

    def test_conversations_avg_rating_order_adds_null_filter(self, tool):
        """avg_rating on conversations stream is also null-prone."""
        q = _query(
            tool,
            stream="conversations",
            order_by={"field": "avg_rating", "direction": "asc"},
            limit=10,
        )
        assert "avg_rating != null" in q


# ---------------------------------------------------------------------------
# Auto-fix: role filter for rating/feedback queries
# ---------------------------------------------------------------------------

class TestRoleAutoFix:
    def test_rating_filter_adds_role_filter(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "rating", "op": "<=", "value": 2}],
        )
        assert 'role == "assistant"' in q

    def test_feedback_text_filter_adds_role_filter(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "feedback_text", "op": "contains", "value": "wrong"}],
        )
        assert 'role == "assistant"' in q

    def test_rating_aggregation_adds_role_filter(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "rating", "op": "!=", "value": None}],
            summarize={
                "aggregations": [{"alias": "avg_r", "func": "avg", "field": "rating"}]
            },
        )
        assert 'role == "assistant"' in q

    def test_explicit_role_user_filter_not_overridden(self, tool):
        """User explicitly asked for user-message ratings — don't override."""
        q = _query(
            tool,
            stream="messages",
            where=[
                {"field": "role", "op": "==", "value": "user"},
                {"field": "rating", "op": "!=", "value": None},
            ],
            order_by={"field": "rating", "direction": "desc"},
        )
        assert 'role == "user"' in q
        assert 'role == "assistant"' not in q

    def test_no_role_autofix_when_no_rating_feedback_referenced(self, tool):
        """Tool-only query shouldn't get a role filter — tool_call_name lives
        on assistant and tool messages both."""
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "tool_call_name", "op": "==", "value": "vector_search"}],
            summarize={"aggregations": [{"alias": "n", "func": "count"}]},
        )
        assert 'role ==' not in q

    def test_conversations_stream_no_role_autofix(self, tool):
        """Conversations stream has no role column — auto-fix shouldn't fire."""
        q = _query(
            tool,
            stream="conversations",
            where=[{"field": "avg_rating", "op": "<", "value": 3}],
        )
        assert 'role ==' not in q


# ---------------------------------------------------------------------------
# JSON validation
# ---------------------------------------------------------------------------

class TestJsonValidation:
    def test_invalid_json_is_rejected(self, tool):
        r = tool(query_spec="{not json}")
        assert "exception" in r
        assert "JSON" in r["exception"]

    def test_non_object_json_rejected(self, tool):
        r = tool(query_spec="[1, 2, 3]")
        assert "exception" in r

    def test_missing_stream_rejected(self, tool):
        r = _call(tool, where=[{"field": "rating", "op": "==", "value": 5}])
        assert "exception" in r
        assert "stream" in r["exception"]

    def test_invalid_stream_rejected(self, tool):
        r = _call(tool, stream="users")
        assert "exception" in r

    def test_invalid_operator_rejected(self, tool):
        r = _call(
            tool,
            stream="messages",
            where=[{"field": "rating", "op": "like", "value": 1}],
        )
        assert "exception" in r
        assert "Invalid op" in r["exception"]

    def test_invalid_aggregation_func_rejected(self, tool):
        r = _call(
            tool,
            stream="messages",
            where=[{"field": "rating", "op": "!=", "value": None}],
            summarize={
                "aggregations": [{"alias": "x", "func": "stddev", "field": "rating"}]
            },
        )
        assert "exception" in r

    def test_select_and_summarize_together_rejected(self, tool):
        r = _call(
            tool,
            stream="messages",
            where=[{"field": "rating", "op": "!=", "value": None}],
            select=["rating"],
            summarize={"aggregations": [{"alias": "n", "func": "count"}]},
        )
        assert "exception" in r

    def test_having_without_summarize_rejected(self, tool):
        r = _call(
            tool,
            stream="messages",
            where=[{"field": "user_id", "op": "==", "value": 7}],
            having=[{"field": "n", "op": ">=", "value": 3}],
        )
        assert "exception" in r
        assert "summarize" in r["exception"]

    def test_negative_limit_rejected(self, tool):
        r = _call(tool, stream="messages", limit=-1)
        assert "exception" in r

    def test_in_operator_requires_list(self, tool):
        r = _call(
            tool,
            stream="messages",
            where=[{"field": "user_id", "op": "in", "value": 7}],
        )
        assert "exception" in r


# ---------------------------------------------------------------------------
# Parser-error propagation (relies on _build_query → parse round-trip)
# ---------------------------------------------------------------------------

class TestParserValidation:
    """The parser validates field names inside `select`, `summarize.by`, and
    `summarize.aggregations` — but defers validation of `where` and
    `order_by` field names to execution time. These tests cover the parts
    the parser validates synchronously."""

    def test_unknown_field_in_select_rejected(self, tool):
        r = _call(
            tool,
            stream="messages",
            select=["feedback_message_text"],
        )
        assert "exception" in r

    def test_wrong_stream_field_in_summarize_rejected(self, tool):
        """avg_rating exists on the conversations stream — using it in a
        summarize aggregation against the messages stream is rejected."""
        r = _call(
            tool,
            stream="messages",
            summarize={
                "aggregations": [{"alias": "x", "func": "avg", "field": "avg_rating"}]
            },
        )
        assert "exception" in r

    def test_unknown_field_in_summarize_by_rejected(self, tool):
        r = _call(
            tool,
            stream="messages",
            summarize={
                "aggregations": [{"alias": "n", "func": "count"}],
                "by": ["nonexistent_field"],
            },
        )
        assert "exception" in r

    def test_post_summarize_order_by_on_stream_field_rejected(self, tool):
        """After summarize, order_by may only reference aggregation aliases
        or by-fields. Stream fields are no longer available. The executor
        catches this at run time; our validator should catch it up-front
        so the LLM gets a clear error instead of a click-time failure."""
        r = _call(
            tool,
            stream="conversations",
            where=[{"field": "rated_count", "op": ">", "value": 0}],
            summarize={
                "aggregations": [{"alias": "min_r", "func": "min", "field": "min_rating"}]
            },
            order_by={"field": "updated_at", "direction": "desc"},
        )
        assert "exception" in r
        assert "after summarize" in r["exception"].lower()

    def test_conversation_tool_contains_rejected(self, tool):
        """conversation_tool is virtual and only supports == and !=.
        The executor catches contains/startswith at run time; we surface
        the same error up front so the LLM can retry instead of shipping
        a broken URL."""
        r = _call(
            tool,
            stream="messages",
            where=[{"field": "conversation_tool", "op": "contains", "value": "vector_search"}],
        )
        assert "exception" in r
        assert "conversation_tool" in r["exception"]

    def test_conversation_tool_equality_works(self, tool):
        q = _query(
            tool,
            stream="messages",
            where=[{"field": "conversation_tool", "op": "==", "value": "vector_search"}],
        )
        assert 'conversation_tool == "vector_search"' in q

    def test_post_summarize_having_field_validates(self, tool):
        """`having` on an alias is valid; `having` on a stream field is not."""
        # Valid: filter on aggregation alias
        r_ok = _call(
            tool,
            stream="messages",
            where=[{"field": "rating", "op": "!=", "value": None}],
            summarize={
                "aggregations": [
                    {"alias": "n", "func": "count"},
                    {"alias": "avg_r", "func": "avg", "field": "rating"},
                ],
                "by": ["user_id"],
            },
            having=[{"field": "n", "op": ">=", "value": 3}],
        )
        assert "exception" not in r_ok
        # Invalid: filter on stream field (rating) post-summarize
        r_bad = _call(
            tool,
            stream="messages",
            where=[{"field": "rating", "op": "!=", "value": None}],
            summarize={
                "aggregations": [{"alias": "avg_r", "func": "avg", "field": "rating"}],
                "by": ["user_id"],
            },
            having=[{"field": "rating", "op": ">=", "value": 4}],
        )
        assert "exception" in r_bad


# ---------------------------------------------------------------------------
# Token URL
# ---------------------------------------------------------------------------

class TestTokenUrl:
    def test_url_uses_token_param(self, tool):
        r = _call(tool, stream="messages")
        assert "exception" not in r
        url = r["result"]["url"]
        assert "?t=" in url
        # Should NOT use the base64 ?q= form
        assert "?q=" not in url

    def test_url_uses_base_url(self, tool):
        r = _call(tool, stream="messages")
        url = r["result"]["url"]
        assert url.startswith("http://test.example.org/aquillm/feedback-dashboard/")

    def test_url_token_resolves_to_query(self, tool):
        from apps.platform_admin.feedbackql.token_store import resolve_token
        r = _call(
            tool,
            stream="messages",
            where=[{"field": "user_id", "op": "==", "value": 42}],
        )
        url = r["result"]["url"]
        token = url.split("?t=")[-1]
        resolved = resolve_token(token)
        assert resolved is not None
        assert "user_id == 42" in resolved

    def test_relative_url_when_no_base(self, db):
        relative_tool = build_feedback_link_tool(base_url="")
        r = relative_tool(query_spec=json.dumps({"stream": "messages"}))
        assert "exception" not in r
        url = r["result"]["url"]
        assert url.startswith("/aquillm/feedback-dashboard/")


# ---------------------------------------------------------------------------
# End-to-end regression tests for specific real-world prompts
# ---------------------------------------------------------------------------
# Each one mirrors a prompt from MANUAL_TESTS.md by submitting the JSON spec
# a competent LLM would emit, and asserting the final query is what we expect.

class TestRealWorldPromptRegressions:
    def test_top_10_highest_rated_autofixes(self, tool):
        """LLM might forget rating != null AND role filter — auto-fix both."""
        q = _query(
            tool,
            stream="messages",
            order_by={"field": "rating", "direction": "desc"},
            limit=10,
        )
        assert "rating != null" in q
        assert 'role == "assistant"' in q
        assert "order by rating desc" in q
        assert q.rstrip().endswith("| limit 10")

    def test_users_with_3_plus_ratings(self, tool):
        """The 'having' canonical example."""
        q = _query(
            tool,
            stream="messages",
            where=[
                {"field": "role", "op": "==", "value": "assistant"},
                {"field": "rating", "op": "!=", "value": None},
            ],
            summarize={
                "aggregations": [
                    {"alias": "n", "func": "count"},
                    {"alias": "avg_r", "func": "avg", "field": "rating"},
                ],
                "by": ["user_id"],
            },
            having=[{"field": "n", "op": ">=", "value": 3}],
            order_by={"field": "avg_r", "direction": "asc"},
        )
        assert "summarize n = count(), avg_r = avg(rating) by user_id" in q
        # Post-summarize where for the having clause
        assert "| where n >= 3" in q
        assert "| order by avg_r asc" in q

    def test_tool_usage_breakdown(self, tool):
        q = _query(
            tool,
            stream="messages",
            summarize={
                "aggregations": [{"alias": "n", "func": "count"}],
                "by": ["tool_call_name"],
            },
            order_by={"field": "n", "direction": "desc"},
        )
        assert "tool_call_name != null" in q
        assert "summarize n = count() by tool_call_name" in q

    def test_rating_histogram(self, tool):
        q = _query(
            tool,
            stream="messages",
            summarize={
                "aggregations": [{"alias": "n", "func": "count"}],
                "by": ["rating"],
            },
            order_by={"field": "rating", "direction": "asc"},
        )
        assert "rating != null" in q
        assert 'role == "assistant"' in q
        assert "summarize n = count() by rating" in q
