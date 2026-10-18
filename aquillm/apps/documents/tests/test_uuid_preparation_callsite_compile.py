"""Real owner UNION compilation prepares each fresh input only once."""

from types import SimpleNamespace
from uuid import UUID

import pytest
from django.db.models import UUIDField
from django.db.models.query import QuerySet

from apps.documents.services.representation_validation_rows import (
    validate_document_rows,
)


def test_fresh_owner_union_compiles_without_repeating_rhs_preparation(monkeypatch):
    selections = tuple(
        SimpleNamespace(document=SimpleNamespace(id=UUID(int=i))) for i in range(3)
    )
    prepared_ids, compiled = [], []
    original = UUIDField.get_db_prep_value

    class BeforeDispatch(Exception):
        pass

    def prepare(field, value, connection, prepared=False):
        prepared_values = original(field, value, connection, prepared=prepared)
        prepared_ids.append(value)
        return prepared_values

    def boundary(query):
        compiled.append(query.query.get_compiler(using=query.db).as_sql())
        raise BeforeDispatch

    monkeypatch.setattr(UUIDField, "get_db_prep_value", prepare)
    # Stop precisely at the database boundary after compiling the real UNION;
    # no fabricated owner rows or successful validation result are supplied.
    monkeypatch.setattr(QuerySet, "_fetch_all", boundary)
    for invocation in range(1, 3):
        with pytest.raises(BeforeDispatch):
            validate_document_rows(selections, using="default")
        assert len(compiled) == invocation
        assert compiled[-1][0].count("UNION ALL") == 7
        assert len(prepared_ids) == 3 * invocation
