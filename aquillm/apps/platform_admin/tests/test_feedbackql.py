"""Tests for the FeedbackQL parser and executor."""
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model

from apps.chat.models import Message, WSConversation
from apps.platform_admin.feedbackql import run
from apps.platform_admin.feedbackql.exceptions import FeedbackQLFieldError, FeedbackQLSyntaxError
from apps.platform_admin.feedbackql.parser import parse

User = get_user_model()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def user_a(db):
    return User.objects.create_user(username='alice', password='x')


@pytest.fixture
def user_b(db):
    return User.objects.create_user(username='bob', password='x')


def _msg(conversation, role, content, seq, **kwargs):
    return Message.objects.create(
        conversation=conversation,
        message_uuid=uuid4(),
        role=role,
        content=content,
        sequence_number=seq,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Parser-only tests (no DB)
# ---------------------------------------------------------------------------

class TestParser:
    def test_minimal_query(self):
        q = parse('messages')
        assert q.clauses == []

    def test_where_eq(self):
        from apps.platform_admin.feedbackql.parser import WhereClause, Condition
        q = parse('messages | where rating == 5')
        assert len(q.clauses) == 1
        clause = q.clauses[0]
        assert isinstance(clause, WhereClause)
        assert clause.parts[0] == Condition(field='rating', op='==', value=5)

    def test_where_lt_and_eq(self):
        from apps.platform_admin.feedbackql.parser import WhereClause, Condition
        q = parse('messages | where rating < 3 and user_id == 7')
        clause = q.clauses[0]
        assert isinstance(clause, WhereClause)
        assert len(clause.parts) == 3
        assert clause.parts[1] == 'and'

    def test_where_startswith(self):
        from apps.platform_admin.feedbackql.parser import Condition
        q = parse('messages | where model startswith "gemma"')
        cond = q.clauses[0].parts[0]
        assert cond == Condition(field='model', op='startswith', value='gemma')

    def test_where_contains(self):
        from apps.platform_admin.feedbackql.parser import Condition
        q = parse('messages | where content contains "hello"')
        cond = q.clauses[0].parts[0]
        assert cond.op == 'contains'
        assert cond.value == 'hello'

    def test_where_in(self):
        from apps.platform_admin.feedbackql.parser import Condition
        q = parse('messages | where rating in [1, 2, 3]')
        cond = q.clauses[0].parts[0]
        assert cond.op == 'in'
        assert cond.value == [1, 2, 3]

    def test_select(self):
        from apps.platform_admin.feedbackql.parser import SelectClause
        q = parse('messages | select rating, model, user_id')
        clause = q.clauses[0]
        assert isinstance(clause, SelectClause)
        assert clause.fields == ['rating', 'model', 'user_id']

    def test_summarize_without_by(self):
        from apps.platform_admin.feedbackql.parser import SummarizeClause, Aggregation
        q = parse('messages | summarize total = count()')
        clause = q.clauses[0]
        assert isinstance(clause, SummarizeClause)
        assert clause.aggregations[0] == Aggregation(alias='total', func='count', agg_field=None)
        assert clause.by == []

    def test_summarize_with_by(self):
        from apps.platform_admin.feedbackql.parser import SummarizeClause
        q = parse('messages | summarize avg_r = avg(rating), n = count() by model')
        clause = q.clauses[0]
        assert isinstance(clause, SummarizeClause)
        assert len(clause.aggregations) == 2
        assert clause.by == ['model']

    def test_order_by_desc(self):
        from apps.platform_admin.feedbackql.parser import OrderByClause
        q = parse('messages | order by created_at desc')
        clause = q.clauses[0]
        assert isinstance(clause, OrderByClause)
        assert clause.field == 'created_at'
        assert clause.direction == 'desc'

    def test_order_by_default_asc(self):
        from apps.platform_admin.feedbackql.parser import OrderByClause
        q = parse('messages | order by rating')
        assert q.clauses[0].direction == 'asc'

    def test_limit(self):
        from apps.platform_admin.feedbackql.parser import LimitClause
        q = parse('messages | limit 25')
        assert q.clauses[0] == LimitClause(n=25)

    def test_multiline_query(self):
        q = parse("""
            messages
            | where user_id == 1
            | select user_id, rating
            | order by rating desc
            | limit 10
        """)
        assert len(q.clauses) == 4

    # --- Error cases ---

    def test_unknown_field_in_select_raises(self):
        # select still validates field names at parse time (no aliases possible)
        with pytest.raises(FeedbackQLFieldError):
            parse('messages | select password')

    def test_unknown_field_in_where_raises_at_execute(self, db):
        # where defers field validation to the executor so it can support
        # aliases in post-summarize where clauses. Pre-summarize wheres still
        # error — just at execute time, with a syntax error rather than a field error.
        with pytest.raises(FeedbackQLSyntaxError):
            run('messages | where password == "secret"')

    def test_must_start_with_messages(self):
        with pytest.raises(FeedbackQLSyntaxError):
            parse('users | where rating == 1')

    def test_unknown_clause_raises(self):
        with pytest.raises(FeedbackQLSyntaxError):
            parse('messages | drop table messages')

    def test_unknown_agg_function_raises(self):
        with pytest.raises(FeedbackQLSyntaxError):
            parse('messages | summarize x = mode(rating)')

    def test_empty_query_raises(self):
        with pytest.raises(FeedbackQLSyntaxError):
            parse('')

    def test_negative_limit_raises(self):
        with pytest.raises(FeedbackQLSyntaxError):
            parse('messages | limit -1')


# ---------------------------------------------------------------------------
# Executor tests (require DB)
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestExecutorRowLevel:
    def test_returns_all_messages_no_filter(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'user', 'hello', 0)
        _msg(conv, 'assistant', 'hi', 1, model='gpt-4', stop_reason='end_turn')

        results = run('messages')
        assert len(results) >= 2

    def test_where_rating_filter(self, user_a, user_b):
        conv_a = WSConversation.objects.create(owner=user_a)
        conv_b = WSConversation.objects.create(owner=user_b)
        _msg(conv_a, 'assistant', 'good', 1, model='m', stop_reason='end_turn', rating=5)
        _msg(conv_b, 'assistant', 'bad', 1, model='m', stop_reason='end_turn', rating=2)

        results = run('messages | where rating == 5')
        assert all(r['rating'] == 5 for r in results)

    def test_where_user_id_filter(self, user_a, user_b):
        conv_a = WSConversation.objects.create(owner=user_a)
        conv_b = WSConversation.objects.create(owner=user_b)
        _msg(conv_a, 'assistant', 'a msg', 1, model='m', stop_reason='end_turn', rating=3)
        _msg(conv_b, 'assistant', 'b msg', 1, model='m', stop_reason='end_turn', rating=3)

        results = run(f'messages | where user_id == {user_a.id}')
        assert all(r['user_id'] == user_a.id for r in results)
        assert len(results) >= 1

    def test_where_and_combines_filters(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'x', 1, model='gemma-2', stop_reason='end_turn', rating=2)
        _msg(conv, 'assistant', 'y', 2, model='gemma-2', stop_reason='end_turn', rating=5)
        _msg(conv, 'assistant', 'z', 3, model='claude', stop_reason='end_turn', rating=2)

        results = run('messages | where model startswith "gemma" and rating < 3')
        assert len(results) == 1
        assert results[0]['rating'] == 2

    def test_where_or_combines_filters(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'x', 1, model='m', stop_reason='end_turn', rating=1)
        _msg(conv, 'assistant', 'y', 2, model='m', stop_reason='end_turn', rating=3)
        _msg(conv, 'assistant', 'z', 3, model='m', stop_reason='end_turn', rating=5)

        results = run('messages | where rating == 1 or rating == 5')
        ratings = {r['rating'] for r in results}
        assert ratings == {1, 5}

    def test_select_limits_output_fields(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'hi', 1, model='m', stop_reason='end_turn', rating=4)

        results = run('messages | select rating, model')
        assert results
        # conversation_id and message_uuid are always included as thread-viewer metadata
        assert {'rating', 'model'}.issubset(results[0].keys())
        assert results[0].keys() <= {'rating', 'model', 'conversation_id', 'message_uuid'}

    def test_order_by_rating_asc(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='m', stop_reason='end_turn', rating=3)
        _msg(conv, 'assistant', 'b', 2, model='m', stop_reason='end_turn', rating=1)
        _msg(conv, 'assistant', 'c', 3, model='m', stop_reason='end_turn', rating=5)

        results = run(f'messages | where user_id == {user_a.id} | select rating | order by rating asc')
        ratings = [r['rating'] for r in results]
        assert ratings == sorted(ratings)

    def test_order_by_rating_desc(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='m', stop_reason='end_turn', rating=3)
        _msg(conv, 'assistant', 'b', 2, model='m', stop_reason='end_turn', rating=1)
        _msg(conv, 'assistant', 'c', 3, model='m', stop_reason='end_turn', rating=5)

        results = run(f'messages | where user_id == {user_a.id} | select rating | order by rating desc')
        ratings = [r['rating'] for r in results]
        assert ratings == sorted(ratings, reverse=True)

    def test_limit(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        for i in range(5):
            _msg(conv, 'user', f'msg {i}', i)

        results = run(f'messages | where user_id == {user_a.id} | limit 2')
        assert len(results) == 2

    def test_where_in_list(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='m', stop_reason='end_turn', rating=1)
        _msg(conv, 'assistant', 'b', 2, model='m', stop_reason='end_turn', rating=3)
        _msg(conv, 'assistant', 'c', 3, model='m', stop_reason='end_turn', rating=5)

        results = run(f'messages | where user_id == {user_a.id} and rating in [1, 3]')
        ratings = {r['rating'] for r in results}
        assert ratings == {1, 3}

    def test_where_contains(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'user', 'what is vector search?', 0)
        _msg(conv, 'user', 'tell me about embeddings', 1)

        results = run(f'messages | where user_id == {user_a.id} and content contains "vector"')
        assert len(results) == 1
        assert 'vector' in results[0]['content']


@pytest.mark.django_db
class TestExecutorSummarize:
    def test_global_count(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='m', stop_reason='end_turn', rating=4)
        _msg(conv, 'assistant', 'b', 2, model='m', stop_reason='end_turn', rating=2)

        results = run(f'messages | where user_id == {user_a.id} | summarize total = count()')
        assert len(results) == 1
        assert results[0]['total'] == 2

    def test_global_avg(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='m', stop_reason='end_turn', rating=4)
        _msg(conv, 'assistant', 'b', 2, model='m', stop_reason='end_turn', rating=2)

        results = run(f'messages | where user_id == {user_a.id} | summarize avg_r = avg(rating)')
        assert results[0]['avg_r'] == pytest.approx(3.0)

    def test_global_median(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        for rating in [1, 2, 3, 4, 5]:
            _msg(conv, 'assistant', f'msg {rating}', rating, model='m', stop_reason='end_turn', rating=rating)

        results = run(f'messages | where user_id == {user_a.id} | summarize med = median(rating)')
        assert results[0]['med'] == 3

    def test_group_by_model(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='gemma', stop_reason='end_turn', rating=2)
        _msg(conv, 'assistant', 'b', 2, model='gemma', stop_reason='end_turn', rating=4)
        _msg(conv, 'assistant', 'c', 3, model='claude', stop_reason='end_turn', rating=5)

        results = run(f'messages | where user_id == {user_a.id} | summarize avg_r = avg(rating) by model')
        by_model = {r['model']: r['avg_r'] for r in results}
        assert by_model['gemma'] == pytest.approx(3.0)
        assert by_model['claude'] == pytest.approx(5.0)

    def test_multiple_aggs_with_by(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='gemma', stop_reason='end_turn', rating=2)
        _msg(conv, 'assistant', 'b', 2, model='gemma', stop_reason='end_turn', rating=4)

        results = run(
            f'messages | where user_id == {user_a.id} '
            f'| summarize n = count(), avg_r = avg(rating) by model'
        )
        assert len(results) == 1
        assert results[0]['n'] == 2
        assert results[0]['avg_r'] == pytest.approx(3.0)

    def test_summarize_ordered(self, user_a):
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='z-model', stop_reason='end_turn', rating=1)
        _msg(conv, 'assistant', 'b', 2, model='a-model', stop_reason='end_turn', rating=5)

        results = run(
            f'messages | where user_id == {user_a.id} '
            f'| summarize avg_r = avg(rating) by model '
            f'| order by avg_r asc'
        )
        avgs = [r['avg_r'] for r in results]
        assert avgs == sorted(avgs)

    def test_select_and_summarize_raises(self):
        with pytest.raises(FeedbackQLSyntaxError):
            run('messages | select rating | summarize n = count()')

    def test_where_null_eq(self, user_a):
        """rating == null returns only messages with no rating."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, rating=None)
        _msg(conv, 'assistant', 'b', 2, rating=4)
        results = run('messages | where rating == null | select rating')
        assert all(r['rating'] is None for r in results)
        assert len(results) == 1

    def test_where_null_neq(self, user_a):
        """rating != null returns only messages that have a rating."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, rating=None)
        _msg(conv, 'assistant', 'b', 2, rating=4)
        _msg(conv, 'assistant', 'c', 3, rating=2)
        results = run('messages | where rating != null | select rating')
        assert all(r['rating'] is not None for r in results)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Post-summarize pipeline tests
# ---------------------------------------------------------------------------
# A `where`, `order by`, or `limit` after a `summarize` operates on the
# aggregated rows in memory and may reference aggregation aliases (avg_r, n)
# or group-by fields. This is the KQL pipeline semantic.

@pytest.mark.django_db
class TestExecutorPostSummarize:
    def test_where_on_aggregate_alias(self, user_a):
        """where avg_r < 3.0 after summarize filters groups by aggregate value."""
        conv = WSConversation.objects.create(owner=user_a)
        # gpt-4: avg rating 2 (below threshold)
        _msg(conv, 'assistant', 'a', 1, model='gpt-4', stop_reason='end_turn', rating=1)
        _msg(conv, 'assistant', 'b', 2, model='gpt-4', stop_reason='end_turn', rating=3)
        # claude: avg rating 5 (above threshold)
        _msg(conv, 'assistant', 'c', 3, model='claude', stop_reason='end_turn', rating=5)
        _msg(conv, 'assistant', 'd', 4, model='claude', stop_reason='end_turn', rating=5)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'| summarize avg_r = avg(rating), n = count() by model '
            f'| where avg_r < 3.0 '
            f'| order by avg_r asc'
        )
        models = [r['model'] for r in results]
        assert models == ['gpt-4']
        assert results[0]['avg_r'] < 3.0

    def test_where_on_count_alias(self, user_a):
        """where n >= 2 keeps only groups with at least 2 messages."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='m1', stop_reason='end_turn', rating=3)
        _msg(conv, 'assistant', 'b', 2, model='m1', stop_reason='end_turn', rating=4)
        _msg(conv, 'assistant', 'c', 3, model='m2', stop_reason='end_turn', rating=5)

        results = run(
            f'messages | where user_id == {user_a.id} '
            f'| summarize n = count() by model '
            f'| where n >= 2'
        )
        assert len(results) == 1
        assert results[0]['model'] == 'm1'
        assert results[0]['n'] == 2

    def test_where_on_group_by_field_after_summarize(self, user_a):
        """A post-summarize where can reference group-by fields too."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='gpt-4', stop_reason='end_turn', rating=4)
        _msg(conv, 'assistant', 'b', 2, model='claude', stop_reason='end_turn', rating=5)

        results = run(
            f'messages | where user_id == {user_a.id} '
            f'| summarize avg_r = avg(rating) by model '
            f'| where model == "gpt-4"'
        )
        assert len(results) == 1
        assert results[0]['model'] == 'gpt-4'

    def test_where_after_summarize_unknown_alias_raises(self, user_a):
        """Referencing a name that isn't a group-by field or alias errors clearly."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='m', stop_reason='end_turn', rating=3)
        with pytest.raises(FeedbackQLSyntaxError):
            run(
                f'messages | where user_id == {user_a.id} '
                f'| summarize avg_r = avg(rating) by model '
                f'| where unknown_alias < 3'
            )

    def test_limit_after_summarize(self, user_a):
        """limit after summarize slices the aggregated dict list."""
        conv = WSConversation.objects.create(owner=user_a)
        for i, m in enumerate(['m1', 'm2', 'm3', 'm4']):
            _msg(conv, 'assistant', f'msg {i}', i, model=m, stop_reason='end_turn', rating=3)

        results = run(
            f'messages | where user_id == {user_a.id} '
            f'| summarize n = count() by model '
            f'| limit 2'
        )
        assert len(results) == 2

    def test_where_and_or_on_aliases(self, user_a):
        """and/or both work on alias-referencing where clauses."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, model='m1', stop_reason='end_turn', rating=2)
        _msg(conv, 'assistant', 'b', 2, model='m2', stop_reason='end_turn', rating=4)
        _msg(conv, 'assistant', 'c', 3, model='m3', stop_reason='end_turn', rating=5)

        results = run(
            f'messages | where user_id == {user_a.id} '
            f'| summarize avg_r = avg(rating), n = count() by model '
            f'| where avg_r < 3 or avg_r > 4.5'
        )
        models = sorted(r['model'] for r in results)
        assert models == ['m1', 'm3']


@pytest.mark.django_db
class TestConversationTool:
    """Tests for the conversation_tool virtual field."""

    def test_eq_specific_tool_matches_conversations_using_that_tool(self, user_a):
        """conversation_tool == "vector_search" returns messages from convs that used it."""
        conv_with_tool = WSConversation.objects.create(owner=user_a)
        _msg(conv_with_tool, 'assistant', 'searching...', 1, tool_call_name='vector_search')
        rated = _msg(conv_with_tool, 'assistant', 'answer', 2, rating=3)

        conv_without_tool = WSConversation.objects.create(owner=user_a)
        _msg(conv_without_tool, 'assistant', 'plain answer', 1, rating=4)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'and conversation_tool == "vector_search"'
        )
        assert len(results) == 1
        assert results[0]['message_uuid'] == rated.message_uuid

    def test_neq_specific_tool_excludes_conversations_using_that_tool(self, user_a):
        """conversation_tool != "vector_search" excludes messages from convs that used it."""
        conv_with_tool = WSConversation.objects.create(owner=user_a)
        _msg(conv_with_tool, 'assistant', 'searching...', 1, tool_call_name='vector_search')
        _msg(conv_with_tool, 'assistant', 'answer', 2, rating=3)

        conv_without_tool = WSConversation.objects.create(owner=user_a)
        plain = _msg(conv_without_tool, 'assistant', 'plain answer', 1, rating=4)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'and conversation_tool != "vector_search"'
        )
        assert len(results) == 1
        assert results[0]['message_uuid'] == plain.message_uuid

    def test_neq_null_matches_conversations_with_any_tool(self, user_a):
        """conversation_tool != null returns messages from convs that used any tool."""
        conv_tools = WSConversation.objects.create(owner=user_a)
        _msg(conv_tools, 'assistant', 'searching...', 1, tool_call_name='more_context')
        rated = _msg(conv_tools, 'assistant', 'answer', 2, rating=2)

        conv_plain = WSConversation.objects.create(owner=user_a)
        _msg(conv_plain, 'assistant', 'plain answer', 1, rating=5)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'and conversation_tool != null'
        )
        assert len(results) == 1
        assert results[0]['message_uuid'] == rated.message_uuid

    def test_eq_null_matches_conversations_with_no_tools(self, user_a):
        """conversation_tool == null returns messages from convs that used no tools."""
        conv_tools = WSConversation.objects.create(owner=user_a)
        _msg(conv_tools, 'assistant', 'searching...', 1, tool_call_name='vector_search')
        _msg(conv_tools, 'assistant', 'answer', 2, rating=3)

        conv_plain = WSConversation.objects.create(owner=user_a)
        plain = _msg(conv_plain, 'assistant', 'plain answer', 1, rating=5)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'and conversation_tool == null'
        )
        assert len(results) == 1
        assert results[0]['message_uuid'] == plain.message_uuid

    def test_multiple_tools_still_matches(self, user_a):
        """conversation_tool == "vector_search" still matches when multiple tools were used."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'step 1', 1, tool_call_name='vector_search')
        _msg(conv, 'assistant', 'step 2', 2, tool_call_name='more_context')
        rated = _msg(conv, 'assistant', 'final answer', 3, rating=4)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'and conversation_tool == "vector_search"'
        )
        assert len(results) == 1
        assert results[0]['message_uuid'] == rated.message_uuid

    def test_select_returns_comma_separated_tools(self, user_a):
        """select conversation_tool returns the tools used in each message's conversation."""
        conv_with_tools = WSConversation.objects.create(owner=user_a)
        _msg(conv_with_tools, 'assistant', 'lookup', 1, tool_call_name='vector_search')
        _msg(conv_with_tools, 'assistant', 'more', 2, tool_call_name='more_context')
        _msg(conv_with_tools, 'assistant', 'answer', 3, rating=4)

        conv_no_tools = WSConversation.objects.create(owner=user_a)
        _msg(conv_no_tools, 'assistant', 'plain', 1, rating=5)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'| select rating, conversation_tool'
        )
        by_rating = {r['rating']: r['conversation_tool'] for r in results}
        # multi-tool conversation joins distinct tools alphabetically
        assert by_rating[4] == 'more_context, vector_search'
        # tool-free conversation shows null
        assert by_rating[5] is None

    def test_select_null_when_no_tools_used(self, user_a):
        """conversation_tool is null for messages whose conversation used no tools."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'plain', 1, rating=3)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'| select rating, conversation_tool'
        )
        assert results[0]['conversation_tool'] is None

    def test_summarize_by_unnests_each_tool(self, user_a):
        """A message in a conversation with N tools contributes N rows to the group-by."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'lookup', 1, tool_call_name='vector_search')
        _msg(conv, 'assistant', 'more', 2, tool_call_name='more_context')
        _msg(conv, 'assistant', 'answer', 3, rating=4)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'| summarize n = count() by conversation_tool'
        )
        by_tool = {r['conversation_tool']: r['n'] for r in results}
        # The single rated message contributes once per tool used in its conversation
        assert by_tool == {'vector_search': 1, 'more_context': 1}

    def test_summarize_by_groups_tool_free_conversations_under_none(self, user_a):
        """Conversations with no tools form a single (None) group, not an error."""
        conv_no_tools = WSConversation.objects.create(owner=user_a)
        _msg(conv_no_tools, 'assistant', 'plain1', 1, rating=3)
        _msg(conv_no_tools, 'assistant', 'plain2', 2, rating=5)

        conv_with_tool = WSConversation.objects.create(owner=user_a)
        _msg(conv_with_tool, 'assistant', 'lookup', 1, tool_call_name='vector_search')
        _msg(conv_with_tool, 'assistant', 'answer', 2, rating=4)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'| summarize n = count(), avg_r = avg(rating) by conversation_tool'
        )
        by_tool = {r['conversation_tool']: r for r in results}
        assert by_tool[None]['n'] == 2
        assert by_tool[None]['avg_r'] == pytest.approx(4.0)
        assert by_tool['vector_search']['n'] == 1
        assert by_tool['vector_search']['avg_r'] == pytest.approx(4.0)

    def test_summarize_by_with_post_summarize_filter_on_alias(self, user_a):
        """Unnested groups compose with post-summarize where on aggregate aliases."""
        conv1 = WSConversation.objects.create(owner=user_a)
        _msg(conv1, 'assistant', 'lookup', 1, tool_call_name='vector_search')
        _msg(conv1, 'assistant', 'r1', 2, rating=2)
        _msg(conv1, 'assistant', 'r2', 3, rating=3)

        conv2 = WSConversation.objects.create(owner=user_a)
        _msg(conv2, 'assistant', 'morecontext', 1, tool_call_name='more_context')
        _msg(conv2, 'assistant', 'r3', 2, rating=5)

        results = run(
            f'messages | where user_id == {user_a.id} and rating != null '
            f'| summarize avg_r = avg(rating) by conversation_tool '
            f'| where avg_r < 4'
        )
        tools = [r['conversation_tool'] for r in results]
        assert tools == ['vector_search']


# ---------------------------------------------------------------------------
# Conversations stream tests
# ---------------------------------------------------------------------------
# A query starting with `conversations` returns one row per conversation,
# with derived fields (message_count, avg_rating, tools_used, etc.) computed
# across that conversation's messages. Filtering, summarizing, ordering and
# limiting work the same as on the messages stream — just over a different
# base set with a different field whitelist.

@pytest.mark.django_db
class TestConversationsStream:
    def test_basic_returns_one_row_per_conversation(self, user_a):
        """A bare `conversations` query yields one row per WSConversation."""
        c1 = WSConversation.objects.create(owner=user_a)
        _msg(c1, 'assistant', 'a', 1, rating=4)
        c2 = WSConversation.objects.create(owner=user_a)
        _msg(c2, 'assistant', 'b', 1, rating=5)

        results = run('conversations')
        ids = {r['conversation_id'] for r in results}
        assert c1.id in ids and c2.id in ids

    def test_derived_message_count(self, user_a):
        """message_count counts each message belonging to the conversation."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'user', 'q1', 0)
        _msg(conv, 'assistant', 'a1', 1, rating=4)
        _msg(conv, 'assistant', 'a2', 2, rating=3)

        results = run(
            f'conversations | where conversation_id == {conv.id} | select message_count'
        )
        assert results[0]['message_count'] == 3

    def test_derived_avg_min_max_rating(self, user_a):
        """avg_rating, min_rating, max_rating reflect aggregate of message ratings."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'a', 1, rating=2)
        _msg(conv, 'assistant', 'b', 2, rating=4)
        _msg(conv, 'assistant', 'c', 3, rating=5)

        results = run(
            f'conversations | where conversation_id == {conv.id} '
            f'| select avg_rating, min_rating, max_rating, rated_count'
        )
        assert results[0]['avg_rating'] == pytest.approx(11/3)
        assert results[0]['min_rating'] == 2
        assert results[0]['max_rating'] == 5
        assert results[0]['rated_count'] == 3

    def test_where_on_derived_avg_rating(self, user_a):
        """Filter conversations by their aggregate rating."""
        good = WSConversation.objects.create(owner=user_a)
        _msg(good, 'assistant', 'g', 1, rating=5)
        bad = WSConversation.objects.create(owner=user_a)
        _msg(bad, 'assistant', 'b', 1, rating=2)

        results = run(
            f'conversations | where user_id == {user_a.id} and avg_rating < 3'
        )
        ids = {r['conversation_id'] for r in results}
        assert bad.id in ids
        assert good.id not in ids

    def test_tools_used_contains(self, user_a):
        """tools_used surfaces a comma-joined list filterable with contains."""
        with_tool = WSConversation.objects.create(owner=user_a)
        _msg(with_tool, 'assistant', 'lookup', 1, tool_call_name='vector_search')
        _msg(with_tool, 'assistant', 'reply', 2, rating=4)
        without = WSConversation.objects.create(owner=user_a)
        _msg(without, 'assistant', 'plain', 1, rating=5)

        results = run(
            f'conversations | where user_id == {user_a.id} '
            f'and tools_used contains "vector_search"'
        )
        ids = {r['conversation_id'] for r in results}
        assert ids == {with_tool.id}

    def test_summarize_aggregates_across_conversations(self, user_a):
        """Aggregate aggregates — sum of message_count across conversations."""
        c1 = WSConversation.objects.create(owner=user_a)
        for i in range(3):
            _msg(c1, 'assistant', f'm{i}', i, rating=4)
        c2 = WSConversation.objects.create(owner=user_a)
        for i in range(2):
            _msg(c2, 'assistant', f'm{i}', i, rating=5)

        results = run(
            f'conversations | where user_id == {user_a.id} '
            f'| summarize total_msgs = sum(message_count), n = count()'
        )
        assert results[0]['total_msgs'] == 5
        assert results[0]['n'] == 2

    def test_post_summarize_where_on_alias(self, user_a):
        """HAVING-style filter on aggregation alias still works on conversations stream."""
        c1 = WSConversation.objects.create(owner=user_a)
        _msg(c1, 'assistant', 'a', 1, rating=2)
        c2 = WSConversation.objects.create(owner=user_a)
        _msg(c2, 'assistant', 'b', 1, rating=5)

        results = run(
            f'conversations | where user_id == {user_a.id} '
            f'| summarize avg_overall = avg(avg_rating) by user_id '
            f'| where avg_overall > 3'
        )
        # avg of (2, 5) = 3.5, passes the > 3 filter
        assert len(results) == 1
        assert results[0]['avg_overall'] == pytest.approx(3.5)

    def test_unknown_field_on_conversations_stream_raises(self):
        """Fields valid on messages but not on conversations stream are rejected."""
        with pytest.raises(FeedbackQLFieldError):
            parse('conversations | select content')

    def test_messages_field_not_on_conversations_select(self):
        """`rating` is a messages-stream field; bare conversations stream doesn't have it."""
        with pytest.raises(FeedbackQLFieldError):
            parse('conversations | select rating')

    def test_messages_stream_still_works_after_changes(self, user_a):
        """Regression: the existing messages stream behaves exactly as before."""
        conv = WSConversation.objects.create(owner=user_a)
        _msg(conv, 'assistant', 'hi', 1, rating=4)
        results = run(f'messages | where user_id == {user_a.id} | select rating')
        assert any(r.get('rating') == 4 for r in results)


# ---------------------------------------------------------------------------
# View-level query tip detection
# ---------------------------------------------------------------------------
# tools_used == "<tool>" is technically valid but almost always wrong: the
# field is a comma-separated string, so == only matches conversations that
# used exactly that one tool. The view surfaces a non-blocking tip instead
# of erroring so the query still runs.

class TestQueryTipDetection:
    def test_tools_used_equals_triggers_tip(self):
        from apps.platform_admin.views.pages import _detect_query_tips
        parsed = parse('conversations | where tools_used == "vector_search"')
        notice = _detect_query_tips(parsed)
        assert notice is not None
        assert 'tools_used' in notice
        assert 'contains' in notice

    def test_tools_used_contains_no_tip(self):
        from apps.platform_admin.views.pages import _detect_query_tips
        parsed = parse('conversations | where tools_used contains "vector_search"')
        assert _detect_query_tips(parsed) is None

    def test_tools_used_eq_on_messages_stream_no_tip(self):
        """tools_used isn't a messages-stream field; this query won't even parse,
        so the tip detector should never see it. Just check the messages stream
        isn't accidentally flagged by some unrelated comparison."""
        from apps.platform_admin.views.pages import _detect_query_tips
        parsed = parse('messages | where rating == 5')
        assert _detect_query_tips(parsed) is None

    def test_other_conversations_query_no_tip(self):
        from apps.platform_admin.views.pages import _detect_query_tips
        parsed = parse('conversations | where avg_rating < 3')
        assert _detect_query_tips(parsed) is None
