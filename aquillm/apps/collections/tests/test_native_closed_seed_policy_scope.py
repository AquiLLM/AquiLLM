"""Focused native closed-seed policy contracts."""

import pytest

from apps.collections.models import CollectionPermission
from apps.documents.models import RawTextDocument, TeXDocument

from .native_closed_seed_policy_fixtures import (
    native_frame,
    policy_data_counts,
    prepare,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


@pytest.mark.parametrize(
    "change",
    [
        "permission",
        "inactive",
        "delete",
        "lost_collection",
        "duplicate_same",
        "duplicate_other",
        "cross_model",
        "cross_model_other",
        "missing_duplicate",
    ],
)
def test_second_native_proof_rejects_full_scope_drift_even_without_a_seed(
    monkeypatch, change
):
    from apps.knowledge_graph.models import CollectionEntityDocumentLink
    from apps.knowledge_graph.retrieval import closed_seed_projection_batch as batch

    with native_frame(monkeypatch) as frame:
        active, owner, _, _, budget = frame
        f, _, source, scope, _, _, _, _ = active
        original, observed = batch.seed_query_rows, []

        def read(query, **kwargs):
            rows = original(query, **kwargs)
            if query.model is CollectionEntityDocumentLink:
                assert rows
                observed.append(True)
                a, b = f.documents["A"], f.documents["B"]
                if change == "permission":
                    CollectionPermission.objects.filter(
                        collection_id=b.collection_id
                    ).delete()
                elif change == "inactive":
                    type(a.ingested_by).objects.filter(pk=a.ingested_by_id).update(
                        is_active=False
                    )
                elif change == "delete":
                    RawTextDocument.objects.filter(pk=b.pk).delete()
                elif change == "lost_collection":
                    RawTextDocument.objects.filter(pk=b.pk).update(
                        collection_id=a.collection_id
                    )
                elif change == "missing_duplicate":
                    RawTextDocument.objects.filter(pk=b.pk).update(id=a.id)
                else:
                    model = (
                        TeXDocument
                        if change.startswith("cross_model")
                        else RawTextDocument
                    )
                    model.objects.bulk_create(
                        [
                            model(
                                id=a.id,
                                title="duplicate",
                                full_text="duplicate",
                                full_text_hash="d" * 64,
                                collection_id=b.collection_id
                                if change in ("duplicate_other", "cross_model_other")
                                else a.collection_id,
                                ingested_by_id=a.ingested_by_id,
                            )
                        ]
                    )
            return rows

        monkeypatch.setattr(batch, "seed_query_rows", read)
        before = (source.budget.sources_used, dict(source.cache), dict(scope.bodies))
        with pytest.raises((ValueError, TimeoutError)):
            prepare(frame)
        assert observed == [True] and owner.mapping is None
        assert before == (source.budget.sources_used, source.cache, scope.bodies)
        assert budget.queries > 0


def test_deleted_original_principal_has_no_fresh_native_permission(monkeypatch):
    from django.contrib.auth.models import User

    from apps.documents.models.document import _get_document_types

    with native_frame(monkeypatch) as frame:
        principal = frame[0][3].authorization.reauthorization_capability._principal
        replacement = User.objects.create_user(username="replacement-ingester")
        for model in _get_document_types():
            model._base_manager.filter(ingested_by_id=principal.pk).update(
                ingested_by=replacement
            )
        User.objects.filter(pk=principal.pk).delete()
        with policy_data_counts() as counts, pytest.raises(ValueError):
            prepare(frame)
        assert counts == [1] and frame[1].mapping is None


def test_unselected_document_collision_stays_irrelevant_to_policy(monkeypatch):
    from uuid import uuid4

    with native_frame(monkeypatch) as frame:
        f = frame[0][0]
        a = f.documents["A"]
        other = uuid4()
        RawTextDocument.objects.bulk_create(
            [
                RawTextDocument(
                    id=other,
                    title=f"unselected-{i}",
                    full_text=f"unselected-{i}",
                    full_text_hash=str(i) * 64,
                    collection_id=a.collection_id,
                    ingested_by_id=a.ingested_by_id,
                )
                for i in (1, 2)
            ]
        )
        prepare(frame)
        assert frame[1].mapping


def test_native_proof_does_not_change_public_partial_scope_revalidation(monkeypatch):
    from apps.collections.services.retrieval_authorization import (
        revalidate_retrieval_authorization_context,
    )

    with native_frame(monkeypatch) as frame:
        f, _, _, scope, _, _, _, _ = frame[0]
        CollectionPermission.objects.filter(
            collection_id=f.documents["B"].collection_id
        ).delete()
        current = revalidate_retrieval_authorization_context(
            context=scope.authorization
        )
        assert current.document_ids == (f.documents["A"].id,)
        assert current.collection_ids == (f.documents["A"].collection_id,)
