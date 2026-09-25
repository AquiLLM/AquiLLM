"""Historical exclusions do not become relation support at collection filtering."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from django.db import transaction

from apps.knowledge_graph.graph import filtering
from apps.knowledge_graph.resolution import collection
from apps.knowledge_graph.tests.test_legacy_resolution_exclusions import (
    _committed_extraction,
    _persist,
)
from apps.knowledge_graph.tests.test_legacy_resolution_exclusions import (
    committed_extraction as committed_extraction,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("invalid_id", [True, 1.0])
def test_exclusion_marker_ids_require_exact_integers(invalid_id):
    from apps.knowledge_graph.resolution.source_exclusions import (
        exclusion_marker_is_valid,
    )

    audit = {
        "excluded_mention_ids": [1],
        "excluded_relation_ids": [],
        "exclusion_reason": "unresolvable_entity_label",
    }
    marker = dict(audit, version=2, excluded_mention_ids=[invalid_id])
    assert not exclusion_marker_is_valid(marker, audit)


def _project(fixture, monkeypatch, loader):
    _context, artifact, _run, _owner, _generation, _mentions, _relations = fixture
    rows, _result = _persist(fixture)
    selected = next(row for row in rows if row.label == "MMLU")
    cluster = SimpleNamespace(
        pk=42,
        cluster_key="c" * 64,
        document_entity_ids=(selected.pk,),
        entity_type=selected.entity_type,
        extraction_confidence=0.9,
        resolution_confidence=selected.resolution_confidence,
        promotion_confidence=None,
    )
    config = collection.CollectionResolutionConfig()
    with transaction.atomic():
        if loader == "resolution":
            result = SimpleNamespace(config=config, clusters=(cluster,))
            return collection._filter_inputs_for_resolution(
                result, (selected,), for_update=True
            )[0]
        link = SimpleNamespace(
            collection_entity_id=cluster.pk,
            document_entity_id=selected.pk,
            document_entity=selected,
        )
        # Only the collection projection is synthetic; both paths query the same
        # real persisted mention memberships, relation evidence and source audit.
        monkeypatch.setattr(
            collection,
            "_bounded_query_rows",
            lambda _query, _maximum, label: (
                (cluster,) if label == "filter collection entity" else (link,)
            ),
        )
        return filtering._filter_inputs_from_artifact(artifact, config)[1][0]


@pytest.mark.parametrize("loader", ["resolution", "artifact"])
def test_noise_relation_support_matches_filtered_extraction(
    committed_extraction, monkeypatch, loader
):
    evidence = _project(committed_extraction, monkeypatch, loader)

    # Orion is a valid, resolved counterpart outside this selected subset.
    # Its edge remains support; the retained punctuation edge is excluded.
    assert evidence.relation_participation == 1
    clean = _committed_extraction("clean")
    clean_evidence = _project(clean, monkeypatch, loader)
    assert clean[2].stats["resolution_commit"]["version"] == 1
    clean_evidence = replace(
        clean_evidence,
        entity_id=evidence.entity_id,
        mention_ids=evidence.mention_ids,
        document_ids=evidence.document_ids,
    )
    assert clean_evidence == evidence
    ontology = committed_extraction[0].ontology
    policy = filtering.FilterPolicy()
    assert filtering.decide_entity_filter(evidence, ontology, policy) == (
        filtering.decide_entity_filter(clean_evidence, ontology, policy)
    )


@pytest.mark.parametrize("loader", ["resolution", "artifact"])
def test_malformed_relation_exclusion_ids_are_not_ignored(
    committed_extraction, monkeypatch, loader
):
    _persist(committed_extraction)
    run = committed_extraction[2]
    relation_id = committed_extraction[6][1].pk
    run.stats["resolution_commit"]["excluded_relation_ids"] = [float(relation_id)]
    run.save(update_fields=["stats"])
    # Avoid repeating persistence, which already independently rejects this audit.
    from apps.knowledge_graph.models import DocumentEntity

    monkeypatch.setattr(
        __import__(__name__, fromlist=["_persist"]),
        "_persist",
        lambda _fixture: (
            tuple(DocumentEntity.objects.filter(artifact=committed_extraction[1])),
            None,
        ),
    )
    with pytest.raises(ValueError, match="exclusion"):
        _project(committed_extraction, monkeypatch, loader)


@pytest.mark.parametrize("committed_extraction", ["all_noise"], indirect=True)
def test_all_excluded_mentions_have_valid_zero_output_commit(committed_extraction):
    from apps.knowledge_graph.resolution.coreference import resolve_document_mentions
    from apps.knowledge_graph.resolution.persistence import persist_document_resolution
    from apps.knowledge_graph.services import builds

    context, artifact, run, owner, generation, mentions, _relations = (
        committed_extraction
    )
    result = resolve_document_mentions((), context.ontology)
    assert not persist_document_resolution(
        artifact.pk, run.pk, result, lease_owner=owner, lease_generation=generation
    )
    run.refresh_from_db()
    assert run.stats["resolution_commit"]["excluded_mention_ids"] == [mentions[0].pk]
    assert run.stats["resolution_commit"]["membership_count"] == 0
    assert builds._document_resolution_commit_state(artifact, run) is (
        builds.CommitMarkerState.VALID
    )
