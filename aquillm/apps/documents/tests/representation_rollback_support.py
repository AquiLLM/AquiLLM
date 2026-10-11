"""Real CLI and immutable-history helpers for the I6 rollback rehearsal."""
import io
import json
from unittest.mock import patch

from django.core.management import call_command

from apps.documents.models import DocumentRepresentationGeneration, DocumentRepresentationState, TextChunk
from apps.documents.services import embedding_index_query as dense_query, embedding_index_store
from apps.documents.services.representation_inputs import configured_chunker_key
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.tasks.embedding_index import build_embedding_generation, rebuild_document_embeddings
from .representation_support import reference, revision


def command_options(doc, revision_id, parser_key):
    return dict(model_label=doc._meta.label_lower, document_pkid=doc.pkid,
        document_id=str(doc.id), extraction_revision_id=revision_id,
        parser_key=parser_key, chunker_key=configured_chunker_key())


def execute(doc, revision_id, parser_key, *, dry_run=False):
    stdout = io.StringIO()
    call_command("reindex_document_representations",
        **command_options(doc, revision_id, parser_key), dry_run=dry_run, stdout=stdout)
    return json.loads(stdout.getvalue())


def active(doc):
    state = DocumentRepresentationState.objects.get(
        model_label=doc._meta.label_lower, document_pkid=doc.pkid, document_id=doc.id)
    return DocumentRepresentationGeneration.objects.get(pk=state.active_generation_id)


def chunk_records(generation):
    return list(TextChunk.objects.filter(representation_generation=generation)
        .order_by("chunk_number", "pk").values(
            "pk", "doc_id", "content", "start_position", "end_position", "chunk_number",
            "modality", "metadata", "representation_generation_id", "search_vector_binding"))


def two_generations(doc, output):
    original_revision = output.extraction_revision
    assert execute(doc, original_revision.pk, original_revision.parser_key)["documents"][0]["status"] == "activated"
    a = active(doc)
    doc.refresh_from_db()
    b_revision = revision(output, "Corrected reading is 100 units.", "i6-rollback-parser-b")
    assert execute(doc, b_revision, "i6-rollback-parser-b")["documents"][0]["status"] == "activated"
    doc.refresh_from_db()
    return a, active(doc), original_revision


def frozen(doc):
    doc.refresh_from_db()
    return freeze_active_representations((reference(doc),), using="default")


def build_dense(doc, space):
    embedding_index_store.persist_embedding_space(space, using="default")
    snapshot = frozen(doc)
    with patch.object(build_embedding_generation, "delay"):
        generation = rebuild_document_embeddings(snapshot.selections[0], space_key=space.space_key, using="default")
    assert build_embedding_generation.run(generation, using="default", timeout_ms=120000) == "activated"
    dense = dense_query.freeze_dense_generations(snapshot, space_key=space.space_key, using="default")
    assert tuple(map(str, dense.generation_keys)) == (generation,)
    return snapshot, dense
