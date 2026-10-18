"""Membership EXISTS shape and whole-scope reread controls."""

from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from django.db import models

from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.retrieval import direct_seed_queries
from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectResolutionSpanInputV1,
    DirectResolutionTier,
)
from apps.knowledge_graph.retrieval.direct_seed_repository import DirectSeedRepository
from apps.knowledge_graph.retrieval.direct_seed_types import DirectSeedCandidateRow
from apps.knowledge_graph.retrieval.topology.contracts import (
    ReadyGenerationBundleV1,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.tests.test_direct_seed_repository import (
    K,
    _membership_state,
    _ready,
    _scope,
)
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1


def test_membership_exists_correlates_complete_expected_tuples(monkeypatch) -> None:
    from apps.knowledge_graph.models import (
        CanonicalEntityLink,
        CollectionEntity,
        CollectionEntityDocumentLink,
        CollectionGraphMembershipState,
    )

    ready = _ready()
    first_state = _membership_state(ready)[0]
    second_state = {
        **first_state,
        "collection_id": 4,
        "active_artifact_id": 12,
        "registry_epoch": 2,
        "membership_checksum": K[10],
        "resolver_version": "resolver-v2",
        "resolution_config_checksum": K[11],
    }
    query, automatic, witness, membership = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    for chain in (query, automatic, witness, membership):
        chain.using.return_value = chain
        chain.filter.return_value = chain
        chain.annotate.return_value = chain
        chain.distinct.return_value = chain
        chain.order_by.return_value = chain
        chain.values.return_value = chain
    automatic.__getitem__.return_value = automatic
    query.__getitem__.return_value = []
    monkeypatch.setattr(CanonicalEntityLink, "objects", automatic)
    monkeypatch.setattr(CollectionEntity, "objects", query)
    monkeypatch.setattr(CollectionEntityDocumentLink, "objects", witness)
    monkeypatch.setattr(CollectionGraphMembershipState, "objects", membership)
    monkeypatch.setattr(models, "Subquery", lambda value: value)
    monkeypatch.setattr(models, "Exists", lambda value: value)
    monkeypatch.setattr(
        direct_seed_queries, "bounded_seed_read", lambda **_: nullcontext()
    )

    assert (
        direct_seed_queries._load_candidate_rows(
            tier=DirectResolutionTier.NAME,
            lookup="atlas",
            lookup_field="normalized_label",
            embedding=None,
            model_signature="",
            ontology_type="paper",
            membership_states=(first_state, second_state),
            automatic_only=True,
            scope=_scope(ready),
            using="default",
            limit=4,
        )
        == ()
    )

    membership_filter = membership.filter.call_args
    assert set(membership_filter.kwargs) == {"collection_id", "active_artifact_id"}
    assert all(
        value.__class__.__name__ == "OuterRef"
        for value in membership_filter.kwargs.values()
    )
    membership_terms = membership_filter.args[0].children
    assert len(membership_terms) == 2
    assert [set(term.children) for term in membership_terms] == [
        set(state.items()) for state in (first_state, second_state)
    ]
    outer_filter = query.filter.call_args_list[0]
    assert membership in outer_filter.args
    assert witness in outer_filter.args
    assert (
        outer_filter.kwargs["collection_id__in"]
        == _scope(ready).selected_collection_ids
    )
    assert outer_filter.kwargs["artifact_id__in"] == _scope(ready).selected_artifact_ids


@pytest.mark.parametrize(
    "tier,lookup_field",
    [
        (DirectResolutionTier.IDENTIFIER, "identifier"),
        (
            DirectResolutionTier.ALIAS,
            "document_links__document_entity__mention_links__mention__normalized_text",
        ),
        (DirectResolutionTier.EMBEDDING, ""),
    ],
)
def test_non_name_tiers_keep_candidate_a_membership_join(
    monkeypatch, tier, lookup_field
) -> None:
    from apps.knowledge_graph.models import (
        CanonicalEntityLink,
        CollectionEntity,
        CollectionEntityDocumentLink,
        CollectionGraphMembershipState,
    )

    ready = _ready()
    first = _membership_state(ready)[0]
    second = {
        **first,
        "collection_id": 4,
        "active_artifact_id": 12,
        "registry_epoch": 2,
        "membership_checksum": K[10],
        "resolver_version": "resolver-v2",
        "resolution_config_checksum": K[11],
    }
    query, automatic, witness, membership = [MagicMock() for _ in range(4)]
    for chain in (query, automatic, witness, membership):
        chain.using.return_value = chain
        chain.filter.return_value = chain
        chain.annotate.return_value = chain
        chain.distinct.return_value = chain
        chain.order_by.return_value = chain
        chain.values.return_value = chain
    automatic.__getitem__.return_value = automatic
    query.__getitem__.return_value = []
    monkeypatch.setattr(CanonicalEntityLink, "objects", automatic)
    monkeypatch.setattr(CollectionEntity, "objects", query)
    monkeypatch.setattr(CollectionEntityDocumentLink, "objects", witness)
    monkeypatch.setattr(CollectionGraphMembershipState, "objects", membership)
    monkeypatch.setattr(models, "Subquery", lambda value: value)
    monkeypatch.setattr(models, "Exists", lambda value: value)
    monkeypatch.setattr(
        direct_seed_queries, "bounded_seed_read", lambda **_: nullcontext()
    )

    assert (
        direct_seed_queries._load_candidate_rows(
            tier=tier,
            lookup="atlas",
            lookup_field=lookup_field,
            embedding=[1.0] + [0.0] * 1023,
            model_signature="test-model",
            minimum_similarity=0.5,
            ontology_type="paper",
            membership_states=(first, second),
            automatic_only=True,
            scope=_scope(ready),
            using="default",
            limit=4,
        )
        == ()
    )

    membership.filter.assert_not_called()
    outer = query.filter.call_args_list[0]
    assert len(outer.args) == 2
    expected = [
        {
            "collection_id": state["collection_id"],
            "artifact_id": state["active_artifact_id"],
            **{
                f"collection__graph_membership_state__{field}": state[field]
                for field in (
                    "active_artifact_id",
                    "registry_epoch",
                    "membership_checksum",
                    "resolver_version",
                    "resolution_config_checksum",
                )
            },
        }
        for state in (first, second)
    ]
    assert [set(term.children) for term in outer.args[0].children] == [
        set(row.items()) for row in expected
    ]
    assert witness in outer.args


def test_unrelated_selected_state_is_reread_before_next_tier() -> None:
    ready = _ready()
    other = replace(
        ready.selected_generations[0],
        collection_key=K[10],
        generation_key=K[11],
        active_artifact_key=K[12],
        projection_key=K[13],
        membership_checksum=K[14],
    )
    generations = (*ready.selected_generations, other)
    expanded = ReadyGenerationBundleV1(
        generations,
        ready.authorized_documents,
        ready.authorization_context_signature,
        ready_generation_bundle_checksum(
            generations,
            ready.authorized_documents,
            ready.authorization_context_signature,
        ),
    )
    first_state = _membership_state(ready)[0]
    unrelated_state = {
        **first_state,
        "collection_id": 4,
        "active_artifact_id": 12,
        "membership_checksum": other.membership_checksum,
    }
    base_scope = _scope(ready)
    scope = replace(
        base_scope,
        ready_bundle_checksum=expanded.bundle_checksum,
        selected_collection_ids=(3, 4),
        selected_artifact_ids=(11, 12),
        generation_keys_by_artifact=(
            (11, ready.selected_generations[0].generation_key),
            (12, other.generation_key),
        ),
        generation_ids_by_artifact=(
            *base_scope.generation_ids_by_artifact,
            (12, uuid4()),
        ),
    )
    seen = []

    def rows(**options):
        seen.append(options["tier"])
        return (DirectSeedCandidateRow(7, 11, "model", None, 1.0),)

    repository = DirectSeedRepository(
        scope=scope,
        codec=HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1"),
        span_inputs=(
            DirectResolutionSpanInputV1(QueryEntitySpanV1("model", 0, 5, 1.0), "model"),
        ),
        row_loader=rows,
        membership_state_loader=lambda **_options: (
            first_state.copy(),
            unrelated_state.copy(),
        ),
    )
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    assert (
        len(repository.canonical_name_matches(span=span, ready=expanded, limit=4)) == 1
    )
    unrelated_state["registry_epoch"] = 2
    with pytest.raises(ValueError, match="current membership"):
        repository.indexed_alias_matches(span=span, ready=expanded, limit=4)
    assert seen == [DirectResolutionTier.NAME]
