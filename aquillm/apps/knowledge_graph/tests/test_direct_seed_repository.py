# ruff: noqa: E501
from __future__ import annotations

import inspect
from dataclasses import replace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.retrieval import direct_seed_repository
from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectResolutionSpanInputV1,
    DirectResolutionTier,
)
from apps.knowledge_graph.retrieval.direct_seed_repository import (
    DirectSeedCandidateRow,
    DirectSeedRepository,
    DirectSeedScopeV1,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    AuthorizedProjectedDocumentV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
    ready_generation_bundle_checksum,
)
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1

K = tuple(character * 64 for character in "123456789abcdef")
DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")


def _ready() -> ReadyGenerationBundleV1:
    generation = SelectedCollectionGenerationV1(
        K[0],
        K[1],
        K[2],
        K[3],
        K[4],
        "schema-v1",
        "projection-v1",
        "key-v1",
        1,
        K[5],
        "resolver-v1",
        K[6],
        K[7],
        "embed-v1",
    )
    documents = (AuthorizedProjectedDocumentV1(K[8], K[0], K[1]),)
    checksum = ready_generation_bundle_checksum((generation,), documents, K[9])
    return ReadyGenerationBundleV1((generation,), documents, K[9], checksum)


def _scope(ready: ReadyGenerationBundleV1) -> DirectSeedScopeV1:
    return DirectSeedScopeV1(
        ready_bundle_checksum=ready.bundle_checksum,
        selected_collection_ids=(3,),
        selected_artifact_ids=(11,),
        selected_document_ids=(DOCUMENT_ID,),
        selected_document_artifact_ids=(22,),
        generation_keys_by_artifact=((11, K[1]),),
        ontology_checksum=K[7],
        resolver_version="resolver-v1",
    )


def _membership_state(
    ready: ReadyGenerationBundleV1, **changes: object
) -> tuple[dict[str, object], ...]:
    generation = ready.selected_generations[0]
    state = {
        "collection_id": 3,
        "active_artifact_id": 11,
        "registry_epoch": generation.membership_epoch,
        "membership_checksum": generation.membership_checksum,
        "resolver_version": generation.resolver_version,
        "resolution_config_checksum": generation.resolution_config_checksum,
    }
    state.update(changes)
    return (state,)


def test_identifier_name_and_indexed_alias_use_bounded_scoped_predicates() -> None:
    ready = _ready()
    span = QueryEntitySpanV1("paper", 0, 13, 0.8)
    local = DirectResolutionSpanInputV1(span, "doi:10.1234/x")
    calls: list[dict[str, object]] = []

    def rows(**kwargs):
        calls.append(kwargs)
        return (
            DirectSeedCandidateRow(
                entity_id=7,
                artifact_id=11,
                ontology_type="paper",
                automatic_identity_key=None,
                similarity=1.0,
            ),
        )

    repository = DirectSeedRepository(
        scope=_scope(ready),
        codec=HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1"),
        span_inputs=(local,),
        row_loader=rows,
        membership_state_loader=lambda **_options: _membership_state(ready),
    )
    identifier = repository.exact_identifier_matches(span=span, ready=ready, limit=4)
    name = repository.canonical_name_matches(span=span, ready=ready, limit=4)
    alias = repository.indexed_alias_matches(span=span, ready=ready, limit=4)

    assert [call["tier"] for call in calls] == [
        DirectResolutionTier.IDENTIFIER,
        DirectResolutionTier.NAME,
        DirectResolutionTier.ALIAS,
    ]
    assert calls[0]["lookup"] == "doi:10.1234/x"
    assert calls[1]["lookup"] == "doi:10.1234/x"
    assert calls[2]["lookup_field"].endswith("mention__normalized_text")
    assert "metadata" not in calls[2]["lookup_field"]
    assert identifier[0].component_key == identifier[0].entity_key
    assert name[0].tier is DirectResolutionTier.NAME
    assert alias[0].tier is DirectResolutionTier.ALIAS


# fmt: off
def test_alias_query_binds_scope_provenance_and_text_to_one_join(monkeypatch) -> None:
    from django.db import models

    from apps.knowledge_graph.models import CanonicalEntityLink, CollectionEntity

    ready = _ready()
    query, automatic = MagicMock(), MagicMock()
    for chain in (query, automatic):
        chain.using.return_value = chain
        chain.filter.return_value = chain
        chain.annotate.return_value = chain
        chain.distinct.return_value = chain
        chain.order_by.return_value = chain
        chain.values.return_value = chain
    automatic.__getitem__.return_value = automatic
    required = {"document_links__manifest_input__document_id__in", "document_links__document_entity__mention_links__mention__normalized_text"}
    slices = []
    def candidate_rows(result_slice):
        slices.append(result_slice.stop)
        row = {"id": 7, "artifact_id": 11, "entity_type": "paper", "automatic_identity_key": None, "similarity": 1.0}
        return [row] * 5 if any(required <= set(call.kwargs) for call in query.filter.call_args_list) else []
    query.__getitem__.side_effect = candidate_rows
    monkeypatch.setattr(CanonicalEntityLink, "objects", automatic)
    monkeypatch.setattr(CollectionEntity, "objects", query)
    monkeypatch.setattr(models, "Subquery", lambda value: value)

    with pytest.raises(ValueError, match="candidate"):
        direct_seed_repository._load_candidate_rows(tier=DirectResolutionTier.ALIAS, lookup="doi:10.1234/x", lookup_field="document_links__document_entity__mention_links__mention__normalized_text", embedding=None, model_signature="", ontology_type="paper", membership_states=_membership_state(ready), automatic_only=True, scope=_scope(ready), using="default", limit=4)

    assert slices == [5]
    assert any(required <= set(call.kwargs) for call in query.filter.call_args_list)
# fmt: on


def test_matching_current_membership_admits_automatic_link_and_excludes_candidate() -> (
    None
):
    ready = _ready()
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    calls: list[dict[str, object]] = []

    def rows(**options):
        calls.append(options)
        assert options["automatic_only"] is True
        return (
            DirectSeedCandidateRow(9, 11, "model", "automatic-a", 1.0),
            DirectSeedCandidateRow(7, 11, "model", None, 1.0),
            DirectSeedCandidateRow(8, 11, "model", "candidate-a", 1.0, "candidate"),
        )

    repository = DirectSeedRepository(
        scope=_scope(ready),
        codec=HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1"),
        span_inputs=(DirectResolutionSpanInputV1(span, "model"),),
        row_loader=rows,
        membership_state_loader=lambda **_options: _membership_state(ready),
    )

    matches = repository.canonical_name_matches(span=span, ready=ready, limit=4)

    assert calls[0]["membership_states"] == _membership_state(ready)
    assert sum(row.entity_key != row.component_key for row in matches) == 1
    assert sum(row.entity_key == row.component_key for row in matches) == 2
    source = inspect.getsource(direct_seed_repository._load_candidate_rows)
    assert "decision_checksum" not in source


@pytest.mark.parametrize(
    "change",
    (
        {"registry_epoch": 2},
        {"membership_checksum": K[10]},
        {"resolver_version": "resolver-v2"},
        {"resolution_config_checksum": K[10]},
    ),
)
def test_stale_current_membership_rejects_mapping_before_candidate_query(
    change: dict[str, object],
) -> None:
    ready = _ready()
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    repository = DirectSeedRepository(
        scope=_scope(ready),
        codec=HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1"),
        span_inputs=(DirectResolutionSpanInputV1(span, "model"),),
        row_loader=lambda **_options: pytest.fail("stale mapping reached candidates"),
        membership_state_loader=lambda **_options: _membership_state(ready, **change),
    )

    with pytest.raises(ValueError, match="current membership"):
        repository.canonical_name_matches(span=span, ready=ready, limit=4)


def test_repository_deduplicates_entities_before_applying_the_result_cap() -> None:
    source = inspect.getsource(direct_seed_repository._load_candidate_rows)
    assert source.index(".distinct()") < source.index('[: int(options["limit"]) + 1]')


def test_repository_rejects_scope_that_omits_a_selected_ready_generation() -> None:
    ready = _ready()
    second = replace(
        ready.selected_generations[0],
        collection_key=K[10],
        generation_key=K[11],
        active_artifact_key=K[12],
        projection_key=K[13],
        membership_checksum=K[14],
    )
    generations = (*ready.selected_generations, second)
    checksum = ready_generation_bundle_checksum(
        generations, ready.authorized_documents, ready.authorization_context_signature
    )
    expanded = ReadyGenerationBundleV1(
        generations,
        ready.authorized_documents,
        ready.authorization_context_signature,
        checksum,
    )
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    repository = DirectSeedRepository(
        scope=_scope(expanded),
        codec=HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1"),
        span_inputs=(DirectResolutionSpanInputV1(span, "model"),),
        row_loader=lambda **_options: (),
        membership_state_loader=lambda **_options: _membership_state(expanded),
    )

    with pytest.raises(ValueError, match="membership scope"):
        repository.canonical_name_matches(span=span, ready=expanded, limit=4)


def test_automatic_components_cross_generations_and_singletons_do_not() -> None:
    ready = _ready()
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    rows = (
        DirectSeedCandidateRow(9, 11, "model", "canonical-a", 1.0),
        DirectSeedCandidateRow(7, 11, "model", None, 1.0),
    )
    codec = HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1")
    repository = DirectSeedRepository(
        scope=_scope(ready),
        codec=codec,
        span_inputs=(DirectResolutionSpanInputV1(span, "model"),),
        row_loader=lambda **_kwargs: rows,
        membership_state_loader=lambda **_options: _membership_state(ready),
    )

    matches = repository.canonical_name_matches(span=span, ready=ready, limit=4)

    automatic = next(
        match for match in matches if match.entity_key != match.component_key
    )
    singleton = next(
        match for match in matches if match.entity_key == match.component_key
    )
    assert automatic.component_key == str(
        codec.encode(
            ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
            source="canonical-a",
        )
    )
    assert singleton.component_key == singleton.entity_key
    assert tuple(match.entity_key for match in matches) == tuple(
        sorted(match.entity_key for match in matches)
    )
