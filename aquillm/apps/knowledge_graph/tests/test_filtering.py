from __future__ import annotations

import ast
import inspect
import socket
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from django.conf import settings

from apps.knowledge_graph.graph.filtering import (
    EntityFilterInput,
    FilterPolicy,
    FilterStatus,
    PositionKind,
    UtilityWeights,
    decide_entity_filter,
    filter_collection_entities,
    filter_policy_checksum,
    score_retrieval_utility,
)
from apps.knowledge_graph.graph.filtering import (
    _filter_inputs_from_artifact as _orm_filter_inputs,
)


def _type_definition(
    name: str,
    *,
    weight: float = 1.0,
    suppression_policy: str = "below_confidence",
    suppression_threshold: float = 0.15,
):
    return SimpleNamespace(
        name=name,
        aliases=(),
        default_retrieval_weight=weight,
        default_suppression_policy=suppression_policy,
        default_suppression_threshold=suppression_threshold,
    )


def _ontology(*extra_types: str):
    types = {
        "model": _type_definition("model"),
        "method": _type_definition("method", weight=0.9),
    }
    types.update({name: _type_definition(name) for name in extra_types})
    return SimpleNamespace(
        checksum="b" * 64,
        entity_types=MappingProxyType(types),
        provenance=MappingProxyType(
            {
                "delta_checksum": "c" * 64,
                "enabled_entity_types": ",".join(extra_types),
            }
            if extra_types
            else {}
        ),
    )


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable(),
    reason="configured PostgreSQL database is not reachable",
)


def _input(**overrides) -> EntityFilterInput:
    values = {
        "entity_id": "entity-1",
        "entity_type": "model",
        "mention_ids": ("mention-1",),
        "document_ids": ("document-1",),
        "extraction_confidence": 0.9,
        "resolution_confidence": 0.95,
        "promotion_confidence": None,
        "relation_participation": 0,
        "positions": (PositionKind.BODY,),
    }
    values.update(overrides)
    return EntityFilterInput(**values)


def test_frequency_contribution_is_logarithmic_and_capped():
    weights = UtilityWeights(
        frequency=1.0,
        document_dispersion=0.0,
        extraction_confidence=0.0,
        resolution_confidence=0.0,
        relation_participation=0.0,
        salient_position=0.0,
    )
    policy = FilterPolicy(frequency_cap=100, weights=weights)
    one = _input(
        mention_ids=("m1",),
        document_ids=("d1",),
    )
    hundred = _input(
        mention_ids=tuple(f"m{i}" for i in range(100)),
        document_ids=("d1",) * 100,
    )
    ten_thousand = _input(
        mention_ids=tuple(f"m{i}" for i in range(10_000)),
        document_ids=("d1",) * 10_000,
    )

    assert score_retrieval_utility(one, policy) < score_retrieval_utility(
        hundred, policy
    )
    assert score_retrieval_utility(hundred, policy) == score_retrieval_utility(
        ten_thousand, policy
    )


def test_document_dispersion_outweighs_repeated_boilerplate_frequency():
    repeated = _input(
        mention_ids=tuple(f"boilerplate-{i}" for i in range(100)),
        document_ids=("one-document",) * 100,
    )
    dispersed = _input(
        mention_ids=tuple(f"useful-{i}" for i in range(5)),
        document_ids=tuple(f"document-{i}" for i in range(5)),
    )

    assert score_retrieval_utility(dispersed, FilterPolicy()) > score_retrieval_utility(
        repeated, FilterPolicy()
    )


def test_relation_participation_and_salient_positions_raise_utility():
    body_only = _input()
    connected = _input(
        relation_participation=3,
        positions=(PositionKind.TITLE,),
    )

    assert score_retrieval_utility(connected, FilterPolicy()) > score_retrieval_utility(
        body_only, FilterPolicy()
    )


@pytest.mark.parametrize(
    "position",
    [PositionKind.TITLE, PositionKind.ABSTRACT, PositionKind.CAPTION],
)
def test_each_title_abstract_and_caption_position_boosts_utility(position):
    body = _input(positions=(PositionKind.BODY,))
    salient = _input(positions=(position,))

    assert score_retrieval_utility(salient, FilterPolicy()) > score_retrieval_utility(
        body, FilterPolicy()
    )


def test_filter_decision_keeps_confidence_namespaces_distinct():
    evidence = _input(
        extraction_confidence=0.81,
        resolution_confidence=0.92,
        promotion_confidence=0.67,
    )
    decision = decide_entity_filter(evidence, _ontology(), FilterPolicy())

    assert decision.extraction_confidence == 0.81
    assert decision.resolution_confidence == 0.92
    assert decision.retrieval_utility == score_retrieval_utility(
        evidence, FilterPolicy()
    )
    assert decision.promotion_confidence == 0.67
    assert decision.policy_checksum == filter_policy_checksum(FilterPolicy())


def test_explicit_rejection_list_sets_status_and_reason_without_deleting_mentions():
    evidence = _input(
        entity_type="legal_boilerplate",
        mention_ids=("m1", "m2"),
        document_ids=("d1", "d1"),
    )
    policy = FilterPolicy(rejected_entity_types=frozenset({"legal_boilerplate"}))

    decision = decide_entity_filter(evidence, _ontology(), policy)

    assert decision.status is FilterStatus.REJECTED
    assert "entity_type_rejected_by_policy" in decision.reason_codes
    assert decision.retained_mention_ids == ("m1", "m2")


def test_unknown_ontology_type_is_rejected_but_publisher_is_suppressed_by_default():
    unknown = decide_entity_filter(
        _input(entity_type="random_header"), _ontology(), FilterPolicy()
    )
    publisher = decide_entity_filter(
        _input(entity_type="publisher"), _ontology(), FilterPolicy()
    )

    assert unknown.status is FilterStatus.REJECTED
    assert "entity_type_not_in_ontology" in unknown.reason_codes
    assert publisher.status is FilterStatus.SUPPRESSED
    assert "publisher_suppressed_by_default" in publisher.reason_codes


def test_activated_ontology_extension_can_enable_publishers():
    decision = decide_entity_filter(
        _input(
            entity_type="publisher",
            mention_ids=("m1", "m2"),
            document_ids=("d1", "d2"),
            relation_participation=1,
            positions=(PositionKind.TITLE,),
        ),
        _ontology("publisher"),
        FilterPolicy(utility_activation_threshold=0.0),
    )

    assert decision.status is FilterStatus.ACTIVE
    assert "publisher_suppressed_by_default" not in decision.reason_codes


def test_base_ontology_publisher_definition_does_not_implicitly_enable_it():
    ontology = _ontology("publisher")
    ontology = SimpleNamespace(
        checksum=ontology.checksum,
        entity_types=ontology.entity_types,
        provenance=MappingProxyType({}),
    )

    decision = decide_entity_filter(
        _input(entity_type="publisher"), ontology, FilterPolicy()
    )

    assert decision.status is FilterStatus.SUPPRESSED
    assert "publisher_suppressed_by_default" in decision.reason_codes


def test_ontology_suppression_threshold_is_status_only():
    ontology = SimpleNamespace(
        checksum="b" * 64,
        entity_types=MappingProxyType(
            {"model": _type_definition("model", suppression_threshold=0.99)}
        ),
    )
    evidence = _input(mention_ids=("m1",), document_ids=("d1",))

    decision = decide_entity_filter(evidence, ontology, FilterPolicy())

    assert decision.status is FilterStatus.SUPPRESSED
    assert "below_ontology_utility_threshold" in decision.reason_codes
    assert decision.retained_mention_ids == evidence.mention_ids


def test_policy_rerun_can_change_status_without_changing_extraction_evidence():
    evidence = _input(
        mention_ids=("m1", "m2"),
        document_ids=("d1", "d2"),
    )
    permissive = filter_collection_entities(
        (evidence,),
        _ontology(),
        FilterPolicy(version="filter-v1", utility_activation_threshold=0.0),
    )
    strict = filter_collection_entities(
        (evidence,),
        _ontology(),
        FilterPolicy(version="filter-v2", utility_activation_threshold=1.0),
    )

    assert permissive[0].status is FilterStatus.ACTIVE
    assert strict[0].status is FilterStatus.SUPPRESSED
    assert permissive[0].retained_mention_ids == strict[0].retained_mention_ids
    assert permissive[0].extraction_confidence == strict[0].extraction_confidence
    assert permissive[0].resolution_confidence == strict[0].resolution_confidence


def test_filter_results_are_deterministic_and_entity_order_invariant():
    first = _input(entity_id="b", mention_ids=("m2",), document_ids=("d2",))
    second = _input(entity_id="a", mention_ids=("m1",), document_ids=("d1",))

    forward = filter_collection_entities((first, second), _ontology(), FilterPolicy())
    reverse = filter_collection_entities((second, first), _ontology(), FilterPolicy())

    assert forward == reverse
    assert tuple(item.entity_id for item in forward) == ("a", "b")


def test_duplicate_entity_or_mention_identity_is_rejected():
    first = _input(entity_id="same", mention_ids=("m1",))
    second = _input(entity_id="same", mention_ids=("m2",))

    with pytest.raises(ValueError, match="duplicate entity"):
        filter_collection_entities((first, second), _ontology(), FilterPolicy())

    with pytest.raises(ValueError, match="duplicate mention"):
        filter_collection_entities(
            (
                _input(entity_id="a", mention_ids=("shared",)),
                _input(entity_id="b", mention_ids=("shared",)),
            ),
            _ontology(),
            FilterPolicy(),
        )


def test_filtering_module_does_not_import_gliner2_embedding_or_chat_llm():
    import apps.knowledge_graph.graph.filtering as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = ("gliner2", "openai", "anthropic", "embedding", "lib.llm")

    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in imported
        for prefix in forbidden
    )


def test_filter_rerun_locks_child_evidence_before_cloning():
    implementation = inspect.getsource(_orm_filter_inputs)

    assert implementation.count("select_for_update") >= 4


def test_filter_policy_is_immutable_and_checksum_covers_semantics():
    first = FilterPolicy(version="filter-v1", utility_activation_threshold=0.2)
    same = FilterPolicy(version="filter-v1", utility_activation_threshold=0.2)
    changed = FilterPolicy(version="filter-v2", utility_activation_threshold=0.2)

    assert first == same
    assert filter_policy_checksum(first) == filter_policy_checksum(same)
    assert filter_policy_checksum(first) != filter_policy_checksum(changed)
    with pytest.raises((AttributeError, TypeError)):
        first.version = "mutated"


def test_position_boost_requires_explicit_position_metadata():
    unspecified = _input(positions=())
    abstract = _input(positions=(PositionKind.ABSTRACT,))

    assert score_retrieval_utility(abstract, FilterPolicy()) > score_retrieval_utility(
        unspecified, FilterPolicy()
    )


def test_collection_entity_status_and_filter_scores_are_immutable():
    from apps.knowledge_graph.models import CollectionEntity

    immutable = set(CollectionEntity._IMMUTABLE_FIELDS)
    queryset_immutable = set(CollectionEntity._QUERYSET_IMMUTABLE_FIELDS)
    assert {
        "status",
        "filter_reason",
        "retrieval_utility",
        "extraction_confidence",
        "resolution_confidence",
        "promotion_confidence",
    } <= immutable
    assert immutable <= queryset_immutable


@pytest.mark.django_db(transaction=True)
@database_required
def test_filter_rerun_creates_new_building_artifact_and_never_mutates_active():
    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.filtering import create_filter_rerun_artifact
    from apps.knowledge_graph.models import (
        GraphArtifact,
        collection_manifest_source_hash,
    )

    collection = Collection.objects.create(name="Filter rerun")
    active = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=collection.pk,
        status=GraphArtifact.Status.ACTIVE,
        source_hash=collection_manifest_source_hash(()),
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="collection-resolution-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature="test-local:model@rev:dims=1024:prep=v1",
        metadata={"ontology_checksum": "b" * 64},
    )

    shadow = create_filter_rerun_artifact(
        active.pk,
        FilterPolicy(version="filter-v2"),
        _ontology(),
    )

    active.refresh_from_db()
    assert active.status == GraphArtifact.Status.ACTIVE
    assert shadow.pk != active.pk
    assert shadow.status == GraphArtifact.Status.BUILDING
    assert shadow.source_hash == active.source_hash
    assert shadow.resolver_version == active.resolver_version
    assert shadow.embedding_model_signature == active.embedding_model_signature
    assert shadow.filter_policy_version == "filter-v2"
