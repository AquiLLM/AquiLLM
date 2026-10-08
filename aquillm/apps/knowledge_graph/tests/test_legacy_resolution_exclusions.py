"""Resume old committed extraction without mutating its evidence."""

from copy import deepcopy

import pytest

from apps.knowledge_graph.extraction import pipeline
from apps.knowledge_graph.models import (
    DocumentEntityMention,
    EntityMention,
    GraphArtifact,
    GraphBuildRun,
    RelationMention,
)
from apps.knowledge_graph.resolution.coreference import resolve_document_mentions
from apps.knowledge_graph.resolution.persistence import (
    ResolutionPersistenceError,
    persist_document_resolution,
    source_mention_fingerprint,
)
from apps.knowledge_graph.services import builds
from apps.knowledge_graph.tests.test_build_orchestration_postgres_races import (
    _document_context,
    _document_occurrence,
    _patch_document_activation,
    _persist_document,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def committed_extraction(request):
    return _committed_extraction(getattr(request, "param", "mixed"))


def _committed_extraction(kind="mixed"):
    _collection, document, chunk = _persist_document(label=f"legacy-label-{kind}")
    context = _document_context(document, chunk)
    artifact, run, owner, generation = _document_occurrence(
        context,
        generation=1,
        artifact_status="building",
        run_stage="resolving",
        run_status="running",
        claim=True,
    )
    labels = (("Orion", "model"), ("MMLU", "dataset"), (".", "model"))
    if kind == "all_noise":
        labels = ((".", "model"),)
    elif kind == "clean":
        labels = labels[:2]
    mentions = tuple(
        EntityMention.objects.create(
            artifact=artifact,
            document_id=document.id,
            chunk=chunk,
            start=chunk.content.index(text),
            end=chunk.content.index(text) + len(text),
            position_basis="document_global",
            raw_text=text,
            normalized_text=text,
            entity_type=kind,
            extraction_confidence=0.9,
            metadata={"observations": []},
        )
        for text, kind in labels
    )
    heads = (mentions[0], mentions[2]) if len(mentions) == 3 else mentions[:1]
    relations = tuple(
        RelationMention.objects.create(
            artifact=artifact,
            document_id=document.id,
            chunk=chunk,
            head=head,
            tail=mentions[1],
            relation_type="uses_dataset",
            extraction_confidence=0.9,
            metadata={"observations": []},
        )
        for head in (heads if len(mentions) > 1 else ())
    )
    run.stats = {
        "extraction_commit": {
            "version": 1,
            "assembly_version": artifact.assembly_version,
            "assembly_config_checksum": artifact.assembly_config_checksum,
            "entity_mention_count": len(mentions),
            "relation_mention_count": len(relations),
            "max_chunks": pipeline.DOCUMENT_EXTRACTION_V1_MAX_CHUNKS,
            "max_characters": pipeline.DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS,
            "max_entities": pipeline.DOCUMENT_EXTRACTION_V1_MAX_ENTITIES,
            "max_relations": pipeline.DOCUMENT_EXTRACTION_V1_MAX_RELATIONS,
        },
        "extraction_evidence_fingerprint": pipeline.extraction_evidence_fingerprint(
            mentions, relations
        ),
    }
    run.save(update_fields=["stats"])
    return context, artifact, run, owner, generation, mentions, relations


def _persist(fixture):
    context, artifact, run, owner, generation, mentions, _relations = fixture
    result = resolve_document_mentions(mentions[:2], context.ontology)
    rows = persist_document_resolution(
        artifact.pk,
        run.pk,
        result,
        lease_owner=owner,
        lease_generation=generation,
    )
    run.refresh_from_db()
    return rows, result


def test_legacy_exclusions_preserve_raw_evidence_and_valid_relation_provenance(
    committed_extraction,
):
    _context, artifact, run, _owner, _generation, mentions, relations = (
        committed_extraction
    )
    original_stats = deepcopy(run.stats)
    original_source = source_mention_fingerprint(mentions)

    rows, result = _persist(committed_extraction)

    assert {row.label for row in rows} == {"Orion", "MMLU"}
    marker = run.stats["resolution_commit"]
    assert marker["version"] == 2
    assert marker["excluded_mention_ids"] == [mentions[2].pk]
    assert marker["excluded_relation_ids"] == [relations[1].pk]
    assert marker["exclusion_reason"] == "unresolvable_entity_label"
    assert marker["source_mention_count"] == 3
    assert marker["membership_count"] == 2
    assert marker["source_mention_fingerprint"] == original_source
    assert marker["result_checksum"] == result.checksum
    assert run.stats["extraction_commit"] == original_stats["extraction_commit"]
    persisted_mentions = tuple(
        EntityMention.objects.filter(artifact=artifact).order_by("pk")
    )
    persisted_relations = tuple(
        RelationMention.objects.filter(artifact=artifact).order_by("pk")
    )
    assert (
        pipeline.extraction_evidence_fingerprint(
            persisted_mentions, persisted_relations
        )
        == original_stats["extraction_evidence_fingerprint"]
    )
    owners = set(
        DocumentEntityMention.objects.filter(mention__artifact=artifact).values_list(
            "mention_id", flat=True
        )
    )
    assert relations[0].head_id in owners and relations[0].tail_id in owners
    assert relations[1].head_id not in owners
    assert (
        builds._document_resolution_commit_state(artifact, run)
        is builds.CommitMarkerState.VALID
    )
    repeated, _result = _persist(committed_extraction)
    assert [row.pk for row in repeated] == [row.pk for row in rows]


@pytest.mark.parametrize(
    "tamper",
    ["omit_noise", "exclude_valid", "omit_relation", "extra_relation", "float_id"],
)
def test_legacy_exclusion_commit_rejects_tampered_coverage(
    committed_extraction, tamper
):
    _context, artifact, run, _owner, _generation, mentions, relations = (
        committed_extraction
    )
    _persist(committed_extraction)
    marker = run.stats["resolution_commit"]
    if tamper == "omit_noise":
        marker["excluded_mention_ids"] = []
    elif tamper == "exclude_valid":
        marker["excluded_mention_ids"] = [mentions[0].pk]
    elif tamper == "omit_relation":
        marker["excluded_relation_ids"] = []
    elif tamper == "float_id":
        marker["excluded_mention_ids"] = [float(mentions[2].pk)]
    else:
        marker["excluded_relation_ids"] = [row.pk for row in relations]
    run.save(update_fields=["stats"])

    assert (
        builds._document_resolution_commit_state(artifact, run)
        is builds.CommitMarkerState.CORRUPT
    )
    with pytest.raises(ResolutionPersistenceError, match="marker|exclusion"):
        _persist(committed_extraction)


def test_legacy_exclusion_result_cannot_drop_valid_mentions(committed_extraction):
    context, artifact, run, owner, generation, mentions, _relations = (
        committed_extraction
    )
    incomplete = resolve_document_mentions(mentions[:1], context.ontology)
    with pytest.raises(ResolutionPersistenceError, match="partition"):
        persist_document_resolution(
            artifact.pk,
            run.pk,
            incomplete,
            lease_owner=owner,
            lease_generation=generation,
        )


def test_document_build_resumes_legacy_extraction_without_provider_or_evidence_rewrite(
    committed_extraction, monkeypatch
):
    context, artifact, run, owner, generation, mentions, relations = (
        committed_extraction
    )
    original_fingerprint = pipeline.extraction_evidence_fingerprint(mentions, relations)
    commit_counts = builds._document_commit_counts
    _patch_document_activation(monkeypatch, lambda: context)
    monkeypatch.setattr(builds, "_document_commit_counts", commit_counts)
    monkeypatch.setattr(
        builds,
        "_bootstrap_document_build",
        lambda *_args: (artifact, run, owner, generation, False),
    )

    def no_provider(*_args, **_kwargs):
        pytest.fail("committed extraction must not call provider")

    monkeypatch.setattr(pipeline, "extract_into_build", no_provider)

    activated = builds.build_document_graph(
        context.identity.document_id,
        context.identity.source_hash,
        builds.derive_document_build_key(context.identity),
    )

    run.refresh_from_db()
    assert activated.pk == artifact.pk
    assert activated.status == GraphArtifact.Status.ACTIVE
    assert run.status == GraphBuildRun.Status.SUCCEEDED
    assert run.stats["resolution_commit"]["membership_count"] == 2
    assert (
        pipeline.extraction_evidence_fingerprint(
            EntityMention.objects.filter(artifact=artifact),
            RelationMention.objects.filter(artifact=artifact),
        )
        == original_fingerprint
    )


def test_exclusion_audit_materializes_single_pass_inputs():
    from apps.knowledge_graph.resolution.source_exclusions import source_exclusion_audit

    audit = source_exclusion_audit(
        iter(({"id": 1, "raw_text": "."}, {"id": 2, "raw_text": "C++"})),
        iter(({"id": 3, "head_id": 1, "tail_id": 2},)),
    )
    assert audit == {
        "excluded_mention_ids": [1],
        "excluded_relation_ids": [3],
        "exclusion_reason": "unresolvable_entity_label",
    }


@pytest.mark.parametrize("label", ["", "x" * 4097, "Orion\x00", None])
def test_other_label_validation_errors_remain_fatal(label):
    from apps.knowledge_graph.resolution.source_exclusions import resolvable_mentions

    with pytest.raises(ValueError):
        resolvable_mentions(({"id": 1, "raw_text": label},))


def test_source_cap_applies_before_excluding_noise(monkeypatch):
    from apps.knowledge_graph.resolution import coreference
    from apps.knowledge_graph.resolution.source_exclusions import resolvable_mentions

    monkeypatch.setattr(coreference, "MAX_DOCUMENT_MENTIONS", 1)
    with pytest.raises(ValueError, match="mention cap"):
        resolvable_mentions(iter(({"raw_text": "."}, {"raw_text": "Orion"})))
