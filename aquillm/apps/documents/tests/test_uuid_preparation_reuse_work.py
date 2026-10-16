"""Fresh physical owner reads reuse only UUID preparation inside their UNION."""

from collections import Counter

import pytest
from django.db import connection
from django.db.models import UUIDField

from apps.collections.services.django_retrieval_authorization import (
    DjangoCollectionRetrievalPermissionPolicy,
)
from apps.documents.services.representation_validation_rows import (
    validate_document_rows,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture

from .test_uuid_scope_membership_work import owners

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("operation", ["representation", "authorization"])
def test_each_fresh_union_prepares_owners_once_and_still_dispatches(
    operation, monkeypatch
):
    case = owners(3)
    counts = Counter()
    original = UUIDField.get_db_prep_value
    policy = DjangoCollectionRetrievalPermissionPolicy(b"uuid-reuse-test")
    selected = tuple(d.id for d in case.documents)

    def prepare(field, value, database, prepared=False):
        counts["prepared"] += 1
        return original(field, value, database, prepared=prepared)

    def execute(real, sql, params, many, context):
        if "UNION ALL" in sql:
            counts["unions"] += 1
            assert sql.count("UNION ALL") == 7
        return real(sql, params, many, context)

    with monkeypatch.context() as observation, connection.execute_wrapper(execute):
        observation.setattr(UUIDField, "get_db_prep_value", prepare)
        for invocation in range(1, 3):
            if operation == "representation":
                assert (
                    validate_document_rows(case.snapshot.selections, using="default")
                    is None
                )
            else:
                budget = FrontierFixture().budget()
                rows = policy.current_authorized_selected_document_scope(
                    principal=case.user,
                    database_alias="default",
                    selected_collection_ids=(case.collection.pk,),
                    selected_document_ids=selected,
                    read_budget=budget,
                )
                assert set(rows) == {(case.collection.pk, value) for value in selected}
                assert budget.queries == 2 and budget.source_rows == 4
            assert counts["unions"] == invocation
            assert counts["prepared"] == invocation * len(selected)
