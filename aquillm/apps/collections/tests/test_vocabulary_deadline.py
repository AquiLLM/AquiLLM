"""Vocabulary work inherits stricter operator deadlines without replacing them."""
from time import monotonic

import pytest
from django.db import connection, transaction

from apps.collections.services.vocabulary_database import _DEADLINE, vocabulary_database


@pytest.mark.django_db(transaction=True)
def test_existing_database_timeout_bounds_entire_vocabulary_operation():
    with transaction.atomic(), connection.cursor() as cursor:
        cursor.execute("SET LOCAL statement_timeout = '80ms'")
        started = monotonic()
        with vocabulary_database("default"):
            cursor.execute("SELECT setting::bigint FROM pg_settings WHERE name = 'statement_timeout'")
            assert 0 < cursor.fetchone()[0] <= 80
            assert _DEADLINE.get() <= started + .09
        cursor.execute("SELECT setting::bigint FROM pg_settings WHERE name = 'statement_timeout'")
        assert cursor.fetchone()[0] == 80


@pytest.mark.django_db(transaction=True)
def test_expired_nested_allowance_cannot_restart():
    token = _DEADLINE.set(monotonic() - 1)
    try:
        with pytest.raises(ValueError, match="deadline"), vocabulary_database("default"):
            pytest.fail("expired allowance entered")
    finally:
        _DEADLINE.reset(token)
