"""No representation equality optimization may bless incompatible record values."""
from dataclasses import replace
from uuid import uuid4

import pytest

from apps.documents.services.representation_contracts import RepresentationSnapshotV1
from apps.knowledge_graph.projection.representation_inputs import binding_values
from apps.knowledge_graph.retrieval import ready_authority_reads_v2 as reads
from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as repository
from apps.knowledge_graph.tests.test_ready_representation_coalescing_v2 import (
    authorities, configured,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("conflict", ("source", "generation"))
def test_individually_valid_but_conflicting_record_payloads_never_form_authority(monkeypatch, conflict):
    fixture = configured(monkeypatch)
    selection = next(s for s in fixture.snapshot.selections
                     if s.document.id == fixture.documents["A"].id)
    if conflict == "source":
        changed = replace(selection, document=replace(selection.document, source_hash="e" * 64),
                          canonical_sha256="e" * 64)
    else:
        changed = replace(selection, generation_key=uuid4(), source_revision_sha256="e" * 64)
    conflicting = binding_values(RepresentationSnapshotV1((changed,)))
    original = reads.fresh_metadata
    validated, returned = [], []

    def conflicting_artifact(*args, **kwargs):
        metadata = original(*args, **kwargs)
        artifact = metadata[1][fixture.projections["A"].artifact_id]
        for name, value in conflicting.items():
            setattr(artifact, name, value)
        return metadata

    original_validation = repository.validate_representation_snapshot

    def validation(snapshot, *, using):
        validated.append(snapshot)
        return original_validation(snapshot, using=using)

    monkeypatch.setattr(reads, "fresh_metadata", conflicting_artifact)
    monkeypatch.setattr(repository, "validate_representation_snapshot", validation)
    with pytest.raises(ValueError, match="projection representation differs"):
        returned.append(authorities(fixture))
    assert validated == [] and returned == []
