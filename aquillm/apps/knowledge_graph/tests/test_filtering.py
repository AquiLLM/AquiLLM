from __future__ import annotations

import ast
import inspect
import os
import socket
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml
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
from apps.knowledge_graph.graph.filtering import (
    _filter_rerun_entity_matches as _rerun_entity_matches,
)


@cache
def _ontology(*extra_types: str):
    from apps.knowledge_graph.services.ontology import (
        load_ontology,
        load_ontology_yaml,
    )

    path = Path(__file__).resolve().parents[1] / "ontologies" / "research-v1.yaml"
    base = load_ontology(path)
    if not extra_types:
        return base
    document = yaml.safe_load(base.canonical_yaml)
    document["version"] = "1.1.0"
    for name in sorted(extra_types):
        document["entity_types"].append(
            {
                "name": name,
                "description": f"Enabled extension entity type {name}.",
                "aliases": [],
                "default_retrieval_weight": 1.0,
                "default_suppression_policy": "below_confidence",
                "default_suppression_threshold": 0.15,
                "extension_enabled": True,
            }
        )
    return load_ontology_yaml(yaml.safe_dump(document, sort_keys=True))


def _ontology_with_entity_fields(name: str, **changes):
    from apps.knowledge_graph.services.ontology import load_ontology_yaml

    document = yaml.safe_load(_ontology().canonical_yaml)
    document["version"] = "1.1.0"
    matching = [item for item in document["entity_types"] if item["name"] == name]
    if matching:
        matching[0].update(changes)
    else:
        matching.append(
            {
                "name": name,
                "description": f"Entity type {name}.",
                "aliases": [],
                "default_retrieval_weight": 1.0,
                "default_suppression_policy": "below_confidence",
                "default_suppression_threshold": 0.15,
                "extension_enabled": False,
                **changes,
            }
        )
        document["entity_types"].extend(matching)
    return load_ontology_yaml(yaml.safe_dump(document, sort_keys=True))


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
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
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
    ontology = replace(
        _ontology_with_entity_fields("publisher", extension_enabled=False),
        provenance=MappingProxyType({"enabled_entity_types": "publisher"}),
    )

    decision = decide_entity_filter(
        _input(entity_type="publisher"), ontology, FilterPolicy()
    )

    assert decision.status is FilterStatus.SUPPRESSED
    assert "publisher_suppressed_by_default" in decision.reason_codes


def test_publisher_status_cannot_change_from_unchecksummed_provenance():
    definition = _ontology_with_entity_fields("publisher", extension_enabled=False)
    without_provenance = replace(
        definition,
        provenance=MappingProxyType({}),
    )
    forged_provenance = replace(
        definition,
        provenance=MappingProxyType({"enabled_entity_types": "publisher"}),
    )

    first = decide_entity_filter(
        _input(entity_type="publisher"), without_provenance, FilterPolicy()
    )
    second = decide_entity_filter(
        _input(entity_type="publisher"), forged_provenance, FilterPolicy()
    )

    assert first.status is second.status is FilterStatus.SUPPRESSED
    assert first.reason_codes == second.reason_codes


def test_filtering_rejects_changed_ontology_semantics_under_same_checksum():
    ontology = _ontology()
    forged_types = dict(ontology.entity_types)
    forged_types["model"] = replace(
        forged_types["model"], default_suppression_threshold=0.99
    )
    forged = replace(
        ontology,
        entity_types=MappingProxyType(forged_types),
    )

    with pytest.raises(ValueError, match="semantic|checksum"):
        filter_collection_entities((_input(),), forged, FilterPolicy())


def test_ontology_suppression_threshold_is_status_only():
    ontology = _ontology_with_entity_fields("model", default_suppression_threshold=0.99)
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


def test_filter_rerun_rejects_mutation_even_when_row_audit_is_recomputed():
    from apps.knowledge_graph.resolution.collection import (
        _collection_entity_row_audit,
    )

    policy = FilterPolicy(utility_activation_threshold=0.0)
    policy_checksum = filter_policy_checksum(policy)
    projection_checksum = "d" * 64
    decision = decide_entity_filter(
        _input(entity_id="11", promotion_confidence=None),
        _ontology(),
        policy,
    )
    source = SimpleNamespace(
        pk=11,
        collection_id=7,
        cluster_key="a" * 64,
        label="Atlas",
        normalized_label="atlas",
        version_signature="",
        entity_type="model",
        identifier="",
        embedding_model_signature="",
        embedding_input_hash="",
        embedding=None,
        metadata={"aliases": ["Atlas"], "row_audit_checksum": "old"},
    )
    destination = SimpleNamespace(pk=22)
    metadata = {
        "aliases": ["Atlas"],
        "filter_policy_checksum": policy_checksum,
        "filter_result_checksum": projection_checksum,
        "filter_reason_codes": list(decision.reason_codes),
        "filter_source_entity_id": source.pk,
    }
    row_fields = {
        "artifact_id": destination.pk,
        "collection_id": source.collection_id,
        "cluster_key": source.cluster_key,
        "label": source.label,
        "normalized_label": source.normalized_label,
        "version_signature": source.version_signature,
        "entity_type": source.entity_type,
        "identifier": source.identifier,
        "status": decision.status.value,
        "extraction_confidence": decision.extraction_confidence,
        "resolution_confidence": decision.resolution_confidence,
        "retrieval_utility": decision.retrieval_utility,
        "promotion_confidence": decision.promotion_confidence,
        "filter_reason": decision.reason_codes[0],
        "embedding_model_signature": source.embedding_model_signature,
        "embedding_input_hash": source.embedding_input_hash,
        "embedding": source.embedding,
    }
    row = SimpleNamespace(metadata=metadata, **row_fields)
    row.metadata["row_audit_checksum"] = _collection_entity_row_audit(row)

    assert _rerun_entity_matches(
        row,
        source=source,
        destination=destination,
        decision=decision,
        policy_checksum=policy_checksum,
        projection_checksum=projection_checksum,
    )

    forged = SimpleNamespace(metadata=dict(row.metadata), **row_fields)
    forged.label = "Forged Atlas"
    forged.metadata["row_audit_checksum"] = _collection_entity_row_audit(forged)
    assert not _rerun_entity_matches(
        forged,
        source=source,
        destination=destination,
        decision=decision,
        policy_checksum=policy_checksum,
        projection_checksum=projection_checksum,
    )


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
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from django.contrib.auth.models import User
    from django.db import close_old_connections

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.graph.filtering import (
        create_filter_rerun_artifact,
        filter_collection_resolution,
    )
    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionEntityDocumentLink,
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        GraphArtifact,
        GraphBuildRun,
    )
    from apps.knowledge_graph.resolution.collection import (
        CollectionEmbeddingSession,
        CollectionResolutionConfig,
        build_collection_snapshot,
        load_collection_filter_inputs,
        load_collection_resolution_inputs,
        persist_collection_resolution,
        resolve_collection_entities,
    )

    embedding_signature = (
        f"test-local:model@rev:endpoint={'e' * 64}:dims=1024:"
        "prep=kg-entity-v1:max_chars=8192:batch=64"
    )
    user = User.objects.create_user(username="kg-filter-rerun", password="unused")
    collection = Collection.objects.create(name="Filter rerun")
    document = RawTextDocument(
        title="Atlas",
        full_text="Atlas is a model.",
        collection=collection,
        ingested_by=user,
        full_text_hash=RawTextDocument.hash_fn("Atlas is a model."),
    )
    document.save(dont_rechunk=True)
    document_artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document.id,
        status=GraphArtifact.Status.ACTIVE,
        source_hash="d" * 64,
        ontology_version=_ontology().version,
        extractor_version="extractor-v1",
        resolver_version="document-coreference-v1",
        filter_policy_version="document-filter-v1",
        ontology_checksum=_ontology().checksum,
    )
    chunk = TextChunk.objects.create(
        content=document.full_text,
        start_position=0,
        end_position=len(document.full_text),
        chunk_number=0,
        modality=TextChunk.Modality.TEXT,
        doc_id=document.id,
        embedding=[0.0] * 1024,
    )
    mention = EntityMention.objects.create(
        artifact=document_artifact,
        document_id=document.id,
        chunk=chunk,
        start=0,
        end=5,
        position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
        raw_text="Atlas",
        normalized_text="Atlas",
        entity_type="model",
        extraction_confidence=0.9,
    )
    document_entity = DocumentEntity.objects.create(
        artifact=document_artifact,
        document_id=document.id,
        cluster_key="e" * 64,
        label="Atlas",
        normalized_label="atlas",
        entity_type="model",
        resolution_confidence=0.95,
    )
    DocumentEntityMention.objects.create(
        document_entity=document_entity,
        mention=mention,
        method=DocumentEntityMention.Method.ROOT,
        resolver_version=document_artifact.resolver_version,
    )
    source_policy = FilterPolicy(version="filter-v1", utility_activation_threshold=0.0)
    config = CollectionResolutionConfig()
    active, _manifest = build_collection_snapshot(
        collection=collection,
        document_artifacts=(document_artifact,),
        ontology=_ontology(),
        extractor_version="extractor-v1",
        resolver_version="collection-resolution-v1",
        filter_policy=source_policy,
        resolution_config=config,
        embedding_model_signature=embedding_signature,
    )
    resolution_run = GraphBuildRun.objects.create(
        artifact=active,
        stage=GraphBuildRun.Stage.RESOLUTION,
        status=GraphBuildRun.Status.RUNNING,
        attempt=1,
    )
    snapshot, entities, relations = load_collection_resolution_inputs(
        active.pk, resolution_run.pk
    )

    def unexpected_embedding_call(_texts):
        raise AssertionError("singleton resolution must not call embeddings")

    session = CollectionEmbeddingSession(
        expected_model_signature=embedding_signature,
        backend=unexpected_embedding_call,
    )
    resolution = resolve_collection_entities(
        snapshot,
        entities,
        _ontology(),
        relations=relations,
        config=config,
        embedding_session=session,
    )
    evidence = load_collection_filter_inputs(active.pk, resolution_run.pk, resolution)
    source_filter = filter_collection_resolution(
        resolution, evidence, _ontology(), source_policy
    )
    persist_collection_resolution(
        active.pk,
        resolution_run.pk,
        resolution,
        source_filter,
        filter_policy=source_policy,
        ontology=_ontology(),
    )
    active.status = GraphArtifact.Status.ACTIVE
    active.save(update_fields=["status"])

    rerun_policy = FilterPolicy(version="filter-v2", utility_activation_threshold=1.0)
    shadow = create_filter_rerun_artifact(
        active.pk,
        rerun_policy,
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
    source_entity = CollectionEntity.objects.get(artifact=active)
    shadow_entity = CollectionEntity.objects.get(artifact=shadow)
    assert shadow_entity.label == source_entity.label
    assert shadow_entity.cluster_key == source_entity.cluster_key
    assert shadow_entity.status == shadow_entity.Status.SUPPRESSED
    assert shadow_entity.promotion_confidence is None
    source_link = CollectionEntityDocumentLink.objects.get(artifact=active)
    shadow_link = CollectionEntityDocumentLink.objects.get(artifact=shadow)
    assert shadow_link.document_entity_id == source_link.document_entity_id
    assert shadow_link.score == source_link.score
    assert shadow_link.method == source_link.method
    assert shadow_link.outcome == source_link.outcome

    duplicate = create_filter_rerun_artifact(
        active.pk,
        rerun_policy,
        _ontology(),
    )
    assert duplicate.pk == shadow.pk

    race_policy = FilterPolicy(version="filter-v3", utility_activation_threshold=0.5)
    barrier = Barrier(2)

    def concurrent_rerun() -> int:
        close_old_connections()
        try:
            barrier.wait(timeout=10)
            return create_filter_rerun_artifact(active.pk, race_policy, _ontology()).pk
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(concurrent_rerun) for _ in range(2)]
        raced_ids = {future.result(timeout=30) for future in futures}
    assert len(raced_ids) == 1
    raced_id = next(iter(raced_ids))
    assert (
        GraphBuildRun.objects.filter(
            artifact_id=raced_id,
            stage=GraphBuildRun.Stage.FILTERING,
        ).count()
        == 1
    )

    run = GraphBuildRun.objects.get(artifact=shadow)
    run.stats["filter_commit"]["entity_count"] = 99
    run.save(update_fields=["stats"])
    with pytest.raises(ValueError, match="partial|corrupt|marker"):
        create_filter_rerun_artifact(
            active.pk,
            rerun_policy,
            _ontology(),
        )
