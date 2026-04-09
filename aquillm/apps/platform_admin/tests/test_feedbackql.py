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

    def test_unknown_field_raises(self):
        with pytest.raises(FeedbackQLFieldError):
            parse('messages | where password == "secret"')

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
        assert set(results[0].keys()) == {'rating', 'model'}

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
