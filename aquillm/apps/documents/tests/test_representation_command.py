"""Bounded operator planning never certifies bytes or mutates a representation."""
import io
import json
from unittest.mock import patch

import pytest
from django.core.management import call_command, CommandError

from apps.documents.models import DocumentRepresentationState
from apps.documents.services.representation_inputs import configured_chunker_key
from .representation_support import retained as retained, upload as upload
from .representation_support import revision

pytestmark = pytest.mark.django_db(transaction=True)


def options(doc, output):
    return dict(model_label=doc._meta.label_lower, document_pkid=doc.pkid,
        document_id=str(doc.id), parser_key=output.extraction_revision.parser_key,
        extraction_revision_id=output.extraction_revision_id, chunker_key=configured_chunker_key())


def test_dry_run_exact_revision_is_metadata_only_then_real_command_activates(retained):
    doc, output, old = retained
    stdout = io.StringIO()
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run performed source I/O")
    with patch("apps.ingestion.services.source_artifacts._verify_blob", forbidden):
        call_command("reindex_document_representations", **options(doc, output), dry_run=True, stdout=stdout)
    report = json.loads(stdout.getvalue())
    assert report["dry_run"] is True and report["integrity_verified"] is False
    assert report["documents"][0]["document_id"] == str(doc.id)
    assert not DocumentRepresentationState.objects.filter(document_id=doc.id).exists()
    stdout = io.StringIO()
    call_command("reindex_document_representations", **options(doc, output), stdout=stdout)
    assert json.loads(stdout.getvalue())["documents"][0]["status"] == "activated"
    assert old.pk not in list(doc.chunks.values_list("pk", flat=True))


@pytest.mark.parametrize("change", [
    {"database": "other"}, {"model_label": "auth.user"}, {"document_id": "wrong"},
    {"chunker_key": "structural-v1"}, {"parser_key": "structure-v2"},
])
def test_invalid_owner_or_policy_cannot_write(retained, change):
    doc, output, _ = retained
    arguments = {**options(doc, output), **change}
    with pytest.raises(CommandError):
        call_command("reindex_document_representations", **arguments, stdout=io.StringIO())
    assert not DocumentRepresentationState.objects.filter(document_id=doc.id).exists()


def test_inventory_requires_explicit_bound_and_reports_keyset_cursor(retained):
    doc, output, _ = retained
    arguments = dict(model_label=doc._meta.label_lower, collection_id=doc.collection_id,
        parser_key=output.extraction_revision.parser_key, chunker_key=configured_chunker_key(), dry_run=True)
    for limit in (None, 0, 1001):
        with pytest.raises(CommandError):
            call_command("reindex_document_representations", **arguments, limit=limit, after_pkid=0, stdout=io.StringIO())
    stdout = io.StringIO()
    call_command("reindex_document_representations", **arguments, limit=1, after_pkid=0, stdout=stdout)
    result = json.loads(stdout.getvalue())
    assert len(result["documents"]) == 1 and result["next_after_pkid"] == doc.pkid


def test_ambiguous_revision_is_reported_without_picking_newest(retained):
    doc, output, _ = retained
    parser = output.extraction_revision.parser_key
    revision(output, "A second valid canonical projection", parser)
    arguments = options(doc, output)
    arguments.pop("extraction_revision_id")
    stdout = io.StringIO()
    with pytest.raises(CommandError):
        call_command("reindex_document_representations", **arguments, dry_run=True, stdout=stdout)
    assert json.loads(stdout.getvalue())["documents"][0]["status"] == "ambiguous_revision"
    assert not DocumentRepresentationState.objects.exists()


def test_inventory_pages_never_expand_to_other_collections(retained):
    from apps.documents.models import RawTextDocument
    from apps.collections.models import Collection
    from apps.documents.services.representation_snapshot_build import snapshot_chunker_key
    doc, _, _ = retained
    second = RawTextDocument(title="Second", full_text="Second canonical source", collection=doc.collection,
                             ingested_by=doc.ingested_by)
    second.save(dont_rechunk=True)
    elsewhere = RawTextDocument(title="Elsewhere", full_text="Private other source",
                               collection=Collection.objects.create(name="Other"), ingested_by=doc.ingested_by)
    elsewhere.save(dont_rechunk=True)
    stdout = io.StringIO()
    call_command("reindex_document_representations", model_label=doc._meta.label_lower,
        collection_id=doc.collection_id, limit=1, after_pkid=doc.pkid, dry_run=True,
        parser_key="canonical-snapshot-v1", chunker_key=snapshot_chunker_key(), stdout=stdout)
    result = json.loads(stdout.getvalue())
    assert [row["document_id"] for row in result["documents"]] == [str(second.id)]
    assert result["next_after_pkid"] == second.pkid
    assert not DocumentRepresentationState.objects.exists()
