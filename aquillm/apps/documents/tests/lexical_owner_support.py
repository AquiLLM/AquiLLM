"""Initial ranking construction, distinct from populated legacy or retained history."""
import pytest
from django.contrib.auth.models import User

from apps.collections.models import Collection
from apps.documents.models import RawTextDocument


@pytest.fixture
def unchunked():
    user = User.objects.create(username="initial-lexical-owner")
    collection = Collection.objects.create(name="Initial lexical")
    document = RawTextDocument(title="Synthetic", full_text="source",
                               collection=collection, ingested_by=user)
    # Choose the canonical source before the first save. An ordinary edit of a
    # persisted owner correctly captures its predecessor and is not initial setup.
    return [document, None]


def staged_projection(retained):
    """Persist a real retained-input revision and stage its complete exact slices."""
    from apps.documents.models import DocumentRepresentationGeneration
    from apps.documents.services.representation_activation import _begin, stage_representation
    from apps.documents.services.representation_inputs import (
        configured_chunker_key, extraction_input, representation_chunks,
    )
    from apps.ingestion.services.extraction_revisions import persist_extraction_revision
    from apps.ingestion.services.source_artifacts import persist_occurrence
    from lib.parsers.source_contracts import (
        ArchiveMemberStepV1, ExtractionV1, SourceArtifactRefV1, SourceElementV1,
    )
    from .representation_support import reference
    document, output, _ = retained
    artifact = SourceArtifactRefV1(**output.payload["source_artifact"])
    occurrence = output.source_occurrence
    retained_occurrence = persist_occurrence(artifact,
        archive_lineage=tuple(ArchiveMemberStepV1(**row) for row in occurrence.archive_lineage),
        original_filename=occurrence.original_filename, content_type=occurrence.content_type, using="default")
    canonical = "Measured\nATP BRCA1"
    extraction = ExtractionV1("synthetic-structural-v1", artifact, canonical, (
        SourceElementV1("heading", "heading", 0, 9, None, None, None, None),
        SourceElementV1("body", "paragraph", 9, len(canonical), None, None, None, "heading"),
    ), "complete", ())
    revision_id = persist_extraction_revision(retained_occurrence, extraction, using="default")
    ref = reference(document)
    origin, revision, extraction = extraction_input(ref, revision_id, using="default")
    key = configured_chunker_key()
    chunks = representation_chunks(origin, extraction, key)
    lease = _begin(ref, origin, revision, extraction, chunks, key, using="default")
    stage_representation(lease, chunks=chunks, using="default")
    return lease, DocumentRepresentationGeneration.objects.get(pk=lease.generation_key), extraction
