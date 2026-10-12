"""Actual homonyms, canonical bridge and duplicate credit."""
from dataclasses import replace
from hashlib import sha256

import pytest
from django.contrib.auth.models import User

from apps.collections.models import CollectionPermission
from apps.knowledge_graph.models import CanonicalEntityLink, CollectionEntity, ProjectionChunkReference, RelationMention
from tests.integration.persisted_graph_acquisition_support import acquire_graph
from tests.integration.persisted_graph_corpus_support import (
    ANSWER, BRIDGE, SEED, graph_corpus as graph_corpus,
)
from tests.integration.persisted_graph_gateway_support import owned_gateway
from tests.integration.persisted_graph_projection_support import graph_configuration as graph_configuration
from tests.integration.persisted_graph_role_support import projection_roles as projection_roles
from tests.integration.persisted_graph_scope_observation import (
    assert_runtime_bounds, graph_observation, identity_labels, keys_for, normalized_rank,
)
from tests.integration.persisted_graph_source_support import upload as upload

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"]
)
QUERY = f"Which paper does {SEED} cite?"
FIRST = f"{SEED} cites {BRIDGE}."
SECOND = f"{BRIDGE} cites {ANSWER}. Answer sentinel lives here."


def bridge_corpus(corpus):
    source = corpus.add("first", collection="A", text=FIRST)
    corpus.add("second", collection="B", text=SECOND)
    corpus.prepare()
    corpus.project()
    return source


def run_bridge(corpus, source, gateway, monkeypatch, settings, **kwargs):
    documents = kwargs.pop("documents", corpus.documents())
    with graph_observation(monkeypatch) as graph:
        result, observed = acquire_graph(source, gateway, monkeypatch, settings,
            documents=documents, query=QUERY, query_entity=("paper", SEED), **kwargs)
    assert_runtime_bounds(result, observed, query=QUERY, documents=documents)
    assert graph.allocations and graph.ranks and result.handoff.candidates
    return result, observed, graph


def test_actual_homonym_source_entities_never_become_one_identity(
    graph_corpus, projection_roles, monkeypatch, settings, tmp_path,
):
    corpus = graph_corpus
    source = corpus.add("first", collection="A", text="Mercury uses DatasetA.")
    corpus.add("second", collection="B", text="Mercury uses DatasetB.")
    corpus.prepare()
    corpus.project()
    entities = tuple(CollectionEntity.objects.filter(label="Mercury", status="active",
        artifact__status="active", collection_id__in=[c.pk for c in corpus.collections.values()]))
    assert len(entities) == 2
    automatic = tuple(CanonicalEntityLink.objects.filter(collection_entity__in=entities,
        outcome="automatic", status="active").values_list("canonical_entity_id", flat=True))
    assert len(automatic) == len(set(automatic))
    labels = identity_labels(corpus)
    mercury = keys_for(labels, "Mercury")
    assert len(mercury) == 2
    query = "Which dataset does Mercury use?"
    with owned_gateway(tmp_path) as gateway:
        with graph_observation(monkeypatch) as ambiguous:
            result, observed = acquire_graph(source, gateway, monkeypatch, settings,
                documents=corpus.documents(), query=query, query_entity=("model", "Mercury"),
                expect_references=False)
        assert_runtime_bounds(result, observed, query=query, documents=corpus.documents())
        assert ambiguous.allocations and not ambiguous.ranks
        assert not ambiguous.identities and not result.handoff.candidates and not observed.inputs
        for _, (allocations, _) in ambiguous.allocations:
            assert all(not seeds for seeds, _, _ in allocations.values())
        reached = set()
        for name, expected in (("first", "DatasetA"), ("second", "DatasetB")):
            documents = corpus.documents(name)
            with graph_observation(monkeypatch) as graph:
                result, observed = acquire_graph(corpus.sources[name], gateway, monkeypatch, settings,
                    documents=documents, query=query, query_entity=("model", "Mercury"))
            assert_runtime_bounds(result, observed, query=query, documents=documents)
            current = mercury.intersection(graph.identities)
            assert len(current) == 1 and not reached.intersection(current)
            reached.update(current)
            targets = set()
            for row in observed.inputs:
                for association in row.ref.associations:
                    if association.relation_type == "uses_dataset":
                        assert association.source_identity_key in current
                        targets.update(labels[association.target_identity_key])
                        assert association.direction == "directed"
            assert targets == {expected}
            assert result.handoff.candidates
        assert reached == mercury


def test_actual_bridge_is_discovered_but_unselected_private_endpoint_is_absent(
    graph_corpus, projection_roles, monkeypatch, settings, tmp_path,
):
    corpus = graph_corpus
    source = bridge_corpus(corpus)
    labels = identity_labels(corpus)
    assert len(keys_for(labels, BRIDGE)) == 1
    viewer = User.objects.create_user(username="p1-bridge-a-only")
    CollectionPermission.objects.create(user=viewer, collection=corpus.collections["A"], permission="VIEW")
    with owned_gateway(tmp_path) as gateway:
        result, observed, graph = run_bridge(corpus, source, gateway, monkeypatch, settings)
        assert graph.seeds == keys_for(labels, SEED)
        assert keys_for(labels, ANSWER) <= graph.identities
        assert any("Answer sentinel" in row.text for row in observed.inputs)
        answer_keys = {row.ref.chunk_key for row in observed.inputs if "Answer sentinel" in row.text}
        assert answer_keys <= {ref.chunk_key for ref in result.handoff.candidates}
        assert any(a.source_identity_key in keys_for(labels, BRIDGE)
            and a.target_identity_key in keys_for(labels, ANSWER)
            and a.relation_type == "cites" and a.direction == "directed"
            for row in observed.inputs for a in row.ref.associations)
        limited, selected, limited_graph = run_bridge(corpus, source, gateway, monkeypatch,
            settings, documents=corpus.documents("first"), principal=viewer)
    assert not keys_for(labels, ANSWER).intersection(limited_graph.identities)
    assert all("Answer sentinel" not in row.text for row in selected.inputs)
    assert {a.collection_id for g in limited.selection.groups for a in g.scope.projections} == {
        corpus.collections["A"].pk,
    }
    assert selected.inputs and limited.handoff.candidates


@pytest.mark.parametrize("copy_kind", ["duplicate_observation", "physical_copy"])
def test_actual_duplicate_evidence_keeps_semantic_endpoint_credit(
    graph_corpus, projection_roles, monkeypatch, settings, tmp_path, copy_kind,
):
    corpus = graph_corpus
    source = bridge_corpus(corpus)
    with owned_gateway(tmp_path) as gateway:
        _, baseline, graph = run_bridge(corpus, source, gateway, monkeypatch, settings)
        before_weights, before_scores = normalized_rank(graph, identity_labels(corpus))
        text = SECOND + ("\n" + FIRST if copy_kind == "physical_copy" else " Repeated extraction control.")
        artifact = corpus.rebuild("second", text,
            duplicate_observations=copy_kind == "duplicate_observation")
        assert RelationMention.objects.filter(artifact=artifact).count() == (
            2 if copy_kind == "physical_copy" else 1
        )
        corpus.prepare("B")
        # Registry changes can enqueue a replacement for either collection.
        for name in ("A", "B"):
            from apps.knowledge_graph.models import CollectionGraphProjection
            if CollectionGraphProjection.objects.filter(collection=corpus.collections[name], state="pending").exists():
                corpus.project(name)
        _, duplicated, graph = run_bridge(corpus, source, gateway, monkeypatch, settings)
        after_weights, after_scores = normalized_rank(graph, identity_labels(corpus))
    assert after_weights == pytest.approx(before_weights, rel=1e-12, abs=1e-12)
    assert after_scores == pytest.approx(before_scores, rel=1e-12, abs=1e-12)
    assert any("Answer sentinel" in row.text for row in duplicated.inputs)
    assert duplicated.representation is not baseline.representation



def test_losing_physical_copy_qualification_reaches_actual_hydration_and_handoff(
    graph_corpus, projection_roles, monkeypatch, settings, tmp_path,
):
    corpus = graph_corpus
    qualification = "The estimate is valid only at pH 6; above pH 6 it is unsupported."
    high_text = f"{SEED} cites {BRIDGE}. This is a preliminary estimate."
    low_text = f"{SEED} cites {BRIDGE}. {qualification}"
    extract = corpus.backend.extract_batch

    def confidence_per_source(texts, *, ontology):
        # Only the existing external extraction fixture varies; real storage,
        # canonical resolution, projection, gateway and ranking stay untouched.
        results = extract(texts, ontology=ontology)
        return tuple(replace(result, relations=tuple(
            replace(row, confidence=0.60 if qualification in text else 0.95)
            for row in result.relations))
            for text, result in zip(texts, results, strict=True))

    monkeypatch.setattr(corpus.backend, "extract_batch", confidence_per_source)
    high = corpus.add("high", collection="A", text=high_text)
    low = corpus.add("low", collection="B", text=low_text)
    corpus.prepare()
    corpus.project()
    high_chunk, low_chunk = high.document.chunks.get(), low.document.chunks.get()
    assert high_chunk.content == high_text and low_chunk.content == low_text
    mapping = dict(ProjectionChunkReference.objects.filter(
        integer_chunk_pk__in=(high_chunk.pk, low_chunk.pk), projection__state="ready",
    ).values_list("projection_chunk_key", "integer_chunk_pk"))
    assert set(mapping.values()) == {high_chunk.pk, low_chunk.pk}
    labels = identity_labels(corpus)
    seed_keys, bridge_keys = keys_for(labels, SEED), keys_for(labels, BRIDGE)
    assert len(seed_keys) == len(bridge_keys) == 1
    query = f"For {SEED}, what qualification applies to the estimate at pH 6?"
    with owned_gateway(tmp_path) as gateway:
        with graph_observation(monkeypatch) as graph:
            result, observed = acquire_graph(high, gateway, monkeypatch, settings,
                documents=corpus.documents(), query=query, query_entity=("paper", SEED))
    assert_runtime_bounds(result, observed, query=query, documents=corpus.documents())
    assert graph.seeds == seed_keys and graph.allocations and graph.ranks
    transitions = [group for arguments, _ in graph.ranks
        for group in arguments["snapshot"].relation_groups
        if group.source_identity_key in seed_keys and group.target_identity_key in bridge_keys
        and group.relation_type == "cites"]
    assert len(transitions) == 1
    assert {mapping[row.chunk_key] for row in transitions[0].evidence} == {high_chunk.pk}
    # The low copy really lost topology selection, yet its unique source text
    # must remain discoverable and selectable through the graph channel.
    hydrated = {mapping[row.ref.chunk_key]: row for row in observed.inputs}
    assert set(hydrated) == {high_chunk.pk, low_chunk.pk}
    selected = hydrated[low_chunk.pk]
    assert selected.text == low_text
    assert selected.text_sha256 == sha256(low_text.encode()).hexdigest()
    assert any(a.source_identity_key in seed_keys and a.target_identity_key in bridge_keys
        and a.relation_type == "cites" and a.direction == "directed"
        for a in selected.ref.associations)
    candidates = {ref.chunk_key: ref for ref in result.handoff.candidates}
    assert selected.ref.chunk_key in candidates
    assert candidates[selected.ref.chunk_key].evidence_provenance_keys == selected.ref.evidence_provenance_keys
    assert observed.budget.sources_used == 2
    assert observed.budget.text_used["materialized"] == len(high_text) + len(low_text)
