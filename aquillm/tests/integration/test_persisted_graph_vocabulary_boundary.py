"""Real nonnull graph cannot outlive captured vocabulary."""
import pytest

from apps.collections.models import Collection, CollectionPermission
from apps.collections.tests.test_vocabulary_scope import entry, publish, service
from apps.chat.services import rag_turn
from apps.documents.services import chunk_search_scope, source_loading
from apps.knowledge_graph.evals.graph_effective_controls import graph_effective_controls
from apps.knowledge_graph.resolution.vocabulary_capture import KEY
from apps.knowledge_graph.tests.test_graph_gate_applicability import reports, validate
from tests.integration.persisted_graph_acquisition_support import acquire_graph
from tests.integration.persisted_graph_gateway_support import owned_gateway
from tests.integration.persisted_graph_projection_support import (
    graph_configuration as graph_configuration, project_active_collection,
)
from tests.integration.persisted_graph_role_support import projection_roles as projection_roles
from tests.integration.persisted_graph_source_support import graph_source as graph_source, upload as upload
from tests.integration.test_persisted_graph_generation_boundary import FIRST, assert_actual_evidence

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"]
)


def vocabulary_for(source, monkeypatch):
    actor, collection = source.document.ingested_by, source.document.collection
    CollectionPermission.objects.update_or_create(user=actor, collection=collection,
        defaults={"permission": "MANAGE"})
    # A separate vocabulary owner isolates revocation from source VIEW authority.
    owner = Collection.objects.create(name="P1 vocabulary owner")
    access = CollectionPermission.objects.create(user=actor, collection=owner, permission="MANAGE")
    vocabulary, first = publish(actor, owner, [entry("Orion", "MMLU", kind="never_merge")])
    s = service()
    pin = s.attach_vocabulary(actor=actor, collection=collection, version=first, expected_revision=0)
    draft = s.save_draft(actor=actor, vocabulary=vocabulary, expected_revision=2,
        entries=[entry("Orion", "Vega", kind="never_merge")])
    second = s.publish_vocabulary(actor=actor, vocabulary=vocabulary, expected_revision=draft.revision)
    monkeypatch.setenv("KG_VOCABULARY_VETO_ENABLED", "1")

    def mutate(kind):
        if kind == "detach":
            s.detach_vocabulary(actor=actor, attachment=pin, expected_revision=pin.revision)
        elif kind == "repin":
            s.attach_vocabulary(actor=actor, collection=collection, version=second,
                expected_revision=pin.revision)
        elif kind == "access":
            access.delete()
            assert CollectionPermission.objects.filter(user=actor, collection=collection,
                permission="MANAGE").exists()
        else:
            raise AssertionError(kind)
    return first, mutate


@pytest.mark.parametrize("mutation,phase", [(None, "before")] + [
    (mutation, phase) for mutation in ("detach", "repin", "access")
    for phase in ("before", "after_body")
])
def test_captured_vocabulary_fences_real_nonnull_graph(
    graph_source, projection_roles, monkeypatch, settings, tmp_path, mutation, phase,
):
    source = graph_source
    version, mutate = vocabulary_for(source, monkeypatch)
    artifact = source.activate_and_build_document(FIRST)
    projection = project_active_collection(source)
    chunk = source.document.chunks.get()
    assert chunk.representation_generation_id is not None
    for built in (artifact, projection.artifact):
        capsule = built.metadata[KEY]
        assert version.version_key in capsule["capsule"]["versions"]
        assert built.resolution_config_checksum == capsule["checksum"]
    captured, reads, mutations = {}, [], []
    captures = []
    original_read = source_loading.source_query_rows
    original_runtime = rag_turn.create_runtime
    original_freeze = chunk_search_scope.freeze_active_representations

    def create_runtime(*args, **kwargs):
        captures.append("authorization")
        return original_runtime(*args, **kwargs)

    def freeze(*args, **kwargs):
        captures.append("representation")
        return original_freeze(*args, **kwargs)

    def mutate_once():
        assert not mutations
        mutate(mutation)
        mutations.append(mutation)
        runtime = captured["runtime"]
        runtime.budget.check_active()
        assert runtime.budget.remaining_ms() >= 1000
        assert runtime._graph_vocabulary is captured["vocabulary"]

    def barrier():
        runtime = source_loading.current_source_runtime()
        assert runtime is not None and runtime._graph_vocabulary.capsules
        captured.update(runtime=runtime, vocabulary=runtime._graph_vocabulary,
            snapshot=runtime._frozen_retrieval_scope.representation,
            authorization=runtime.authorization, budget=runtime.budget)
        if mutation is not None and phase == "before":
            mutate_once()

    def load(*args, **kwargs):
        rows = original_read(*args, **kwargs)
        if captured and rows:
            assert {row.pk for row in rows} == {chunk.pk}
            assert {row.content for row in rows} == {FIRST}
            reads.append(tuple(row.pk for row in rows))
            if mutation is not None and phase == "after_body":
                mutate_once()
        return rows

    with monkeypatch.context() as probe:
        probe.setattr(source_loading, "source_query_rows", load)
        probe.setattr(rag_turn, "create_runtime", create_runtime)
        probe.setattr(chunk_search_scope, "freeze_active_representations", freeze)
        with owned_gateway(tmp_path) as gateway:
            result, observed = acquire_graph(source, gateway, monkeypatch, settings, barrier=barrier)
    assert observed.calls == ["hydration"] and observed.references
    assert captures == ["authorization", "representation"]
    assert observed.scope.runtime is captured["runtime"]
    assert observed.representation is captured["snapshot"]
    assert observed.authorization is captured["authorization"]
    assert observed.budget is captured["budget"]
    assert captured["runtime"]._graph_vocabulary is captured["vocabulary"]
    assert result.selection.representation_snapshot is observed.representation
    assert not {"deadline", "frontier_budget"}.intersection(result.handoff.coverage.reasons)
    if mutation is None:
        assert mutations == [] and reads
        assert_actual_evidence(result, observed, FIRST, chunk.representation_generation_id)
    else:
        assert mutations == [mutation]
        assert bool(reads) is (phase == "after_body")
        assert observed.inputs == () and result.handoff.candidates == ()
    # A tested opt-in capsule is still unrecognized by the strict evaluator.
    controls = graph_effective_controls()
    assert controls["scoped_vocabulary"] == {"state": "unknown"}
    assert not validate(None, quality_reports=reports(controls))["passed"]
