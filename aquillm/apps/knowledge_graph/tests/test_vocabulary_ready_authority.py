"""Actual ready selection and final materialization must enforce vocabulary pins."""
import pytest

from apps.collections.models import CollectionPermission
from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.collections.tests.test_vocabulary_scope import entry, publish, service
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.tests import ready_subset_support as support
from apps.knowledge_graph.tests.test_ready_materialization_v2 import candidate
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def prepared(monkeypatch, *, bound):
    from apps.knowledge_graph.resolution import vocabulary_capture as v
    from apps.knowledge_graph.resolution.vocabulary_policy import bind_digest
    fixture = support.ScopeFixture(monkeypatch)
    actor = fixture.documents["A"].ingested_by
    for document in fixture.documents.values():
        CollectionPermission.objects.create(user=actor, collection=document.collection, permission="MANAGE")
    fixture.authorization = build_selected_scope_authorization_context(principal=actor,
        selected_collection_ids=(7, 9), selected_documents=list(fixture.documents.values()))
    _, version = publish(actor, fixture.documents["A"].collection,
                         [entry("Alpha", "Gamma", kind="never_merge")])
    pin = service().attach_vocabulary(actor=actor, collection=fixture.documents["A"].collection,
                                     version=version, expected_revision=0)
    monkeypatch.setenv("KG_VOCABULARY_VETO_ENABLED", "1")
    capture = v.capture_build_vocabulary((7, 9))
    original = support._artifact
    def artifact(**kwargs):
        result = original(**kwargs)
        cid = int(result.scope_id) if result.scope_type == "collection" else next(
            d.collection_id for d in fixture.documents.values() if str(d.id) == str(result.scope_id))
        capsule = capture.get(cid)
        if bound and capsule:
            from apps.knowledge_graph.models import graph_identity_checksum
            result.resolution_config_checksum = graph_identity_checksum(
                "document-resolver" if result.scope_type == "document" else "resolver-version",
                result.resolver_version)
            result.metadata = v.metadata(capsule, result.resolution_config_checksum)
            result.resolution_config_checksum = bind_digest(result.resolution_config_checksum, capsule.policy)
        return result
    monkeypatch.setattr(support, "_artifact", artifact)
    fixture.persist_ready("A", with_evidence=True)
    runtime = SourceRuntime(TurnBudget(TurnLimits()), fixture.authorization)
    return fixture, runtime, pin


@pytest.mark.django_db(transaction=True)
def test_existing_unbound_ready_graph_requires_rebuild_when_veto_is_added(monkeypatch):
    from apps.knowledge_graph.retrieval.ready_scope import ReadyScopeError
    fixture, runtime, _ = prepared(monkeypatch, bound=False)
    with source_runtime_scope(runtime), pytest.raises(ReadyScopeError):
        fixture.select_v2(ready=())


@pytest.mark.django_db(transaction=True)
def test_bound_ready_graph_materializes_then_detachment_revokes_final_use(monkeypatch):
    from apps.knowledge_graph.models import ProjectionChunkReference
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer
    fixture, runtime, pin = prepared(monkeypatch, bound=True)
    monkeypatch.setattr(materializer, "load_projection_runtime_settings", lambda: fixture.settings)
    monkeypatch.setattr(materializer, "projection_identifier_codec", lambda _: fixture.codec)
    with source_runtime_scope(runtime):
        selected = fixture.select_v2(ready=())
        key = ProjectionChunkReference.objects.get(projection=fixture.projections["A"]).projection_chunk_key
        ref = candidate(selected, chunk_key=key, evidence_provenance_keys=(fixture.provenance_keys["A"],))
        assert materializer.materialize_graph_candidates_v2(selected=selected, refs=(ref,),
            authorization=fixture.authorization) == (fixture.chunks["A"],)
        pin.delete()
        assert materializer.materialize_graph_candidates_v2(selected=selected, refs=(ref,),
            authorization=fixture.authorization) == ()


@pytest.mark.django_db(transaction=True)
def test_v1_rechecks_policy_after_private_body_load(monkeypatch):
    from apps.knowledge_graph.models import ProjectionChunkReference
    from apps.knowledge_graph.projection.identifiers import OpaqueProjectionKey, ProjectionIdentifierDomain
    from apps.knowledge_graph.retrieval.ready_materialization import (
        DjangoPrivateChunkMapRepository, materialize_selected_ready_chunks)
    from apps.knowledge_graph.resolution.vocabulary_capture import VocabularyGraphUnavailable
    fixture, runtime, pin = prepared(monkeypatch, bound=True)
    class RevokingRepository(DjangoPrivateChunkMapRepository):
        def load_chunk_objects(self, **kwargs):
            rows = super().load_chunk_objects(**kwargs)
            pin.delete()
            return rows
    with source_runtime_scope(runtime):
        selected = fixture.select_v2(ready=())
        key = ProjectionChunkReference.objects.get(projection=fixture.projections["A"]).projection_chunk_key
        with pytest.raises(VocabularyGraphUnavailable):
            materialize_selected_ready_chunks(scope=selected.groups[0].scope,
                chunk_keys=(OpaqueProjectionKey(ProjectionIdentifierDomain.CHUNK, key),),
                authorization=fixture.authorization, repository=RevokingRepository("default"))
