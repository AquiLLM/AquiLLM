"""Missing authority is partial; present corruption is never narrowed away."""

import pytest

from apps.knowledge_graph.models import CollectionGraphProjection
from apps.knowledge_graph.retrieval.ready_scope import ReadyScopeError
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

pytestmark = pytest.mark.django_db


@pytest.fixture
def scope_fixture(monkeypatch):
    return ScopeFixture(monkeypatch)


def test_missing_readiness_is_partial_but_corruption_is_shared(scope_fixture):
    result = scope_fixture.select_v2(ready=("A",), pending=("B",))
    assert result.coverage.ready_documents == 1
    assert result.coverage.selected_documents == 2
    assert result.coverage.status == "partial"
    assert result.coverage.reasons == ("not_ready",)
    assert (
        result.groups[0].scope.ready.authorization_context_signature
        == scope_fixture.authorization.authorization_context_signature
    )
    with pytest.raises(ReadyScopeError):
        scope_fixture.select_v2(ready=("A",), pending=("B",), corrupt_checksum=True)


def test_all_valid_incompatible_groups_remain_captured(scope_fixture):
    selected = scope_fixture.select_v2(ready=("A", "B"), pending=(), alternate=True)
    assert len(selected.groups) == 2
    assert selected.coverage.status == "complete"
    assert {g.ontology_checksum for g in selected.groups} == {"f" * 64, "b" * 64}
    assert all(
        g.scope.ready.authorization_context_signature
        == selected.authorization_context_signature
        for g in selected.groups
    )
    CollectionGraphProjection.objects.filter(
        pk=scope_fixture.projections["B"].pk
    ).update(membership_checksum="0" * 64)
    with pytest.raises(ReadyScopeError):
        scope_fixture.select_v2(ready=(), pending=())


def test_unsupported_present_signature_fails_shared(scope_fixture):
    scope_fixture.persist_ready("A")
    CollectionGraphProjection.objects.filter(
        pk=scope_fixture.projections["A"].pk
    ).update(schema_version="unsupported-v99")
    with pytest.raises(ReadyScopeError):
        scope_fixture.select_v2(ready=())


def test_absent_scope_has_no_fabricated_ready_bundle(scope_fixture):
    result = scope_fixture.select_v2(ready=(), pending=("A", "B"))
    assert result.groups == ()
    assert result.coverage.status == "unavailable"
    assert result.coverage.selected_documents == 2


def test_different_documents_with_same_policy_share_a_compatible_group(scope_fixture):
    selected = scope_fixture.select_v2(ready=("A", "B"), pending=())
    assert len(selected.groups) == 1
    assert set(selected.groups[0].scope.selected_document_ids) == {
        document.id for document in scope_fixture.documents.values()
    }


def test_malformed_present_pending_metadata_is_shared_failure(scope_fixture):
    from django.db.models import QuerySet

    from apps.knowledge_graph.models import GraphArtifact

    scope_fixture.persist_ready("A")
    pending = scope_fixture.persist_ready("B")
    CollectionGraphProjection.objects.filter(pk=pending.pk).update(
        state="pending", ready_at=None
    )
    QuerySet.update(
        GraphArtifact.objects.filter(pk=pending.artifact_id), ontology_version=" bad "
    )
    with pytest.raises(ReadyScopeError):
        scope_fixture.select_v2(ready=(), pending=())
