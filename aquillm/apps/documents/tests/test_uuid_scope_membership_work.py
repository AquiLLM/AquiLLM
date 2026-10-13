"""Packed UUID membership bounds bindings without caching source or authority."""

from hashlib import sha256
from time import monotonic
from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.db import connection

from apps.collections.models import Collection, CollectionPermission
from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.collections.services.retrieval_authorization import (
    revalidate_retrieval_authorization_context,
)
from apps.documents.models import DESCENDED_FROM_DOCUMENT, PDFDocument, RawTextDocument
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.services.representation_validation_rows import (
    validate_document_rows,
)
from apps.documents.tests.representation_support import reference
from apps.knowledge_graph.retrieval.direct_seed_sql import DirectSeedReadTimeout
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture

pytestmark = pytest.mark.django_db(transaction=True)


def owners(count):
    user = get_user_model().objects.create_user(
        username="uuid-membership-" + uuid4().hex
    )
    collection = Collection.objects.create(name="UUID membership scope")
    permission = CollectionPermission.objects.create(
        user=user, collection=collection, permission="VIEW"
    )
    documents = RawTextDocument.objects.bulk_create(
        [
            RawTextDocument(
                title=f"Selected {index}",
                full_text=f"Canonical source {index}",
                full_text_hash=sha256(f"Canonical source {index}".encode()).hexdigest(),
                collection=collection,
                ingested_by=user,
            )
            for index in range(count)
        ]
    )
    snapshot = freeze_active_representations(
        tuple(reference(document) for document in documents), using="default"
    )
    authority = build_selected_scope_authorization_context(
        principal=user,
        selected_collection_ids=(collection.pk,),
        selected_documents=documents,
    )
    assert authority is not None
    return SimpleNamespace(
        user=user,
        collection=collection,
        permission=permission,
        documents=documents,
        snapshot=snapshot,
        authority=authority,
    )


@pytest.mark.parametrize("count", [1, 65])
@pytest.mark.parametrize("operation", ["representation", "authorization"])
def test_actual_union_parameter_width_is_bounded_by_concrete_models(count, operation):
    case = owners(count)
    statements = []

    def execute(real, sql, params, many, context):
        result = real(sql, params, many, context)
        if "UNION ALL" in sql:
            statements.append((sql, len(params), context["cursor"].rowcount))
        return result

    with connection.execute_wrapper(execute):
        if operation == "representation":
            assert (
                validate_document_rows(case.snapshot.selections, using="default")
                is None
            )
        else:
            budget = FrontierFixture().budget()
            current = revalidate_retrieval_authorization_context(
                context=case.authority, read_budget=budget
            )
            assert set(current.document_ids) == {d.id for d in case.documents}
            assert current.collection_ids == (case.collection.pk,)
            assert budget.queries == 2 and budget.source_rows == count + 1
    assert len(statements) == 1
    sql, bindings, actual_rows = statements[0]
    assert actual_rows == count
    assert sql.rstrip().endswith(f"LIMIT {count + 1}")
    assert sql.count("UNION ALL") == len(DESCENDED_FROM_DOCUMENT) - 1
    assert bindings <= 2 * len(DESCENDED_FROM_DOCUMENT)


@pytest.mark.parametrize("change", ["revoke", "inactive", "move", "duplicate"])
def test_fresh_authority_and_duplicate_sentinel_survive_packed_membership(change):
    case = owners(3)
    document = case.documents[0]
    if change == "revoke":
        case.permission.delete()
    elif change == "inactive":
        get_user_model().objects.filter(pk=case.user.pk).update(is_active=False)
    elif change == "move":
        RawTextDocument.objects.filter(pk=document.pk).update(
            collection=Collection.objects.create(name="Outside original scope")
        )
    else:
        PDFDocument.objects.bulk_create(
            [
                PDFDocument(
                    id=document.id,
                    title="Duplicate concrete owner",
                    full_text=document.full_text,
                    full_text_hash=document.full_text_hash,
                    collection=case.collection,
                    ingested_by=case.user,
                )
            ]
        )
    budget = FrontierFixture().budget()
    if change == "duplicate":
        with pytest.raises(ValueError, match="duplicate scope rows"):
            revalidate_retrieval_authorization_context(
                context=case.authority, read_budget=budget
            )
        assert budget.source_rows == 5 and budget.queries == 2
        with pytest.raises(SourceScopeUnavailable):
            validate_document_rows(case.snapshot.selections, using="default")
    else:
        current = revalidate_retrieval_authorization_context(
            context=case.authority, read_budget=budget
        )
        expected = {d.id for d in case.documents[1:]} if change == "move" else set()
        assert set(current.document_ids) == expected
        assert budget.queries == (2 if change == "move" else 1)


def test_source_validation_hashes_current_body_not_only_stored_hash():
    case = owners(3)
    document = case.documents[0]
    assert (
        RawTextDocument.objects.filter(pk=document.pk).update(
            full_text="Untracked changed canonical body"
        )
        == 1
    )
    with pytest.raises(SourceScopeUnavailable):
        validate_document_rows(case.snapshot.selections, using="default")


def test_expired_graph_read_budget_dispatches_no_authorization_data():
    case = owners(3)
    fixture = FrontierFixture()
    fixture.deadline = monotonic() - 1
    budget = fixture.budget()
    with pytest.raises(DirectSeedReadTimeout):
        revalidate_retrieval_authorization_context(
            context=case.authority, read_budget=budget
        )
    assert budget.source_rows == budget.queries == 0
