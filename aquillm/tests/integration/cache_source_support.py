"""STAGED, UNEXECUTED: actual ingestion, representation and dense producers."""
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services import embedding_index_store as dense
from apps.documents.services import representation_activation as representation
from apps.documents.services import representation_inputs as inputs
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.tests.representation_support import reference
from apps.ingestion.services.extraction_revisions import persist_extraction_revision
from apps.ingestion.services.source_artifacts import persist_occurrence
from lib.parsers.source_contracts import (
    ArchiveMemberStepV1, ExtractionV1, SourceArtifactRefV1, SourceElementV1,
)
from apps.ingestion.tests.output_support import upload as upload
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from lib.embeddings.document_contracts import build_budget
from lib.embeddings.document_provider import embed_document_batch
from lib.embeddings.tests.test_document_provider import transport as transport


def revision(output, text, parser="cache-paragraph-v1"):
    occurrence = output.source_occurrence
    artifact = SourceArtifactRefV1(**output.payload["source_artifact"])
    retained = persist_occurrence(artifact,
        archive_lineage=tuple(ArchiveMemberStepV1(**v) for v in occurrence.archive_lineage),
        original_filename=occurrence.original_filename,
        content_type=occurrence.content_type, using="default")
    elements, start = [], 0
    for number, paragraph in enumerate(text.splitlines(keepends=True)):
        end = start + len(paragraph)
        elements.append(SourceElementV1(f"paragraph-{number}", "paragraph",
            start, end, None, None, None, None))
        start = end
    assert start == len(text) and len(elements) > 1
    extraction = ExtractionV1(parser, artifact, text, tuple(elements), "complete", ())
    return persist_extraction_revision(retained, extraction, using="default")


def stage_dense(document, space):
    dense.persist_embedding_space(space, using="default")
    snapshot = freeze_active_representations((reference(document),), using="default")
    selection = snapshot.selections[0]
    assert selection.generation_key is not None
    manifest = dense.capture_embedding_manifest(selection, using="default", space=space)
    rows = list(TextChunk.objects.filter(
        doc_id=document.id, representation_generation_id=selection.generation_key,
    ).order_by("chunk_number"))
    assert tuple(row.pk for row in rows) == tuple(item["chunk_id"] for item in manifest)
    lease = dense.begin_embedding_generation(selection, space_key=space.space_key,
        manifest_sha256=dense.embedding_manifest_digest(selection, manifest,
            space_key=space.space_key), expected_count=len(rows), using="default")
    results = embed_document_batch(tuple(dense.source_chunk(row) for row in rows),
        space=space, budget=build_budget())
    dense.stage_embedding_batch(lease, chunk_ids=tuple(row.pk for row in rows),
        results=results, input_sha256=tuple(item["prepared_input_sha256"] for item in manifest),
        using="default")
    return lease


def stage_same_bytes(document, output):
    ref, key = reference(document), inputs.configured_chunker_key()
    revision_id = revision(output, document.full_text, parser="cache-same-byte-v2")
    origin, rev, extraction = inputs.extraction_input(ref, revision_id, using="default")
    chunks = inputs.representation_chunks(origin, extraction, key)
    manifest = inputs.generation_manifest(ref, origin=origin, revision=rev,
        extraction=extraction, chunks=chunks, chunker_key=key)
    lease = representation.begin_representation(ref, extraction_revision_id=revision_id,
        chunker_key=key, manifest_sha256=inputs.manifest_hash(manifest),
        expected_count=len(chunks), using="default")
    representation.stage_representation(lease, chunks=chunks, using="default")
    old = freeze_active_representations((ref,), using="default").selections[0].generation_key
    return lease, old


@dataclass
class CacheCorpus:
    documents: tuple
    outputs: tuple
    rows: tuple
    space: object
    settings: object
    query: str
    principal: object

    @property
    def consumer(self):
        return SimpleNamespace(user=self.principal,
            col_ref=SimpleNamespace(collections=[self.documents[0].collection_id]))

    def replacement(self, kind, index):
        """All expensive preparation happens before the measured original turn."""
        doc = self.documents[index]
        if kind == "source":
            lease, old = stage_same_bytes(doc, self.outputs[index])
            def apply():
                assert representation.activate_representation(lease,
                    expected_active_key=old, using="default") == "activated"
            return apply, lease.generation_key
        if kind == "dense":
            lease = stage_dense(doc, self.space)
            def apply():
                assert dense.activate_embedding_generation(lease, using="default") == "activated"
            return apply, lease.generation_key
        if kind == "profile":
            from lib.embeddings import document_provider as provider
            from lib.embeddings.tests.test_document_contracts import profile
            other = profile(weights_revision="cache-replacement-weights-v1", compute_precision="bf16")
            provider.register_verified_profile(other, endpoint="http://fake-models:8000/v1",
                deployment_artifact_sha256="c" * 64, engine_manifest=provider.engine_manifest_for(other),
                query_template="{text}", document_template="{text}")
            dense.persist_embedding_space(other, using="default")
            def apply():
                self.settings.DOCUMENT_EMBEDDING_SPACE_KEY = other.space_key
            return apply, other.space_key
        raise AssertionError(kind)


@pytest.fixture
def cache_corpus_factory(upload, transport, settings, monkeypatch):
    from lib.embeddings import config
    monkeypatch.setattr(config, "get_local_embed_config",
        lambda: ("http://fake-models:8000/v1", "synthetic", "synthetic"))
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = transport[0].space_key
    monkeypatch.setenv("KG_BUILD_ENABLED", "false")
    settings.KG_MEMGRAPH_TRAVERSAL_ENABLED = False
    settings.KG_OVERLAY_ENABLED = False
    settings.KG_GRAPH_DIRECT_ENABLED = False
    settings.KG_GRAPH_EXTENDED_ENABLED = False

    def make():
        token = uuid4().hex
        collection = Collection.objects.create(name=f"Cache {token}")
        bodies = tuple("\n".join(
            f"instrument calibration owner {name} observation {token} measurement {number}."
            for number in range(3)) for name in ("A", "B"))
        item = upload(tuple((f"{name}.txt", body.encode())
                            for name, body in zip(("A", "B"), bodies)))
        batch = item.batch
        batch.collection = collection
        batch.save(update_fields=["collection"])
        CollectionPermission.objects.create(user=batch.user, collection=collection, permission="MANAGE")
        run_ingest_uploaded_file(item.pk)
        outputs = tuple(item.outputs.filter(status="success").order_by("document_pkid"))
        assert len(outputs) == 2
        documents, all_rows = [], []
        for output in outputs:
            doc = RawTextDocument.objects.get(pk=output.document_pkid)
            assert representation.build_document_representation(reference(doc),
                extraction_revision_id=revision(output, doc.full_text),
                chunker_key=inputs.configured_chunker_key(), using="default") == "activated"
            doc.refresh_from_db()
            selection = freeze_active_representations((reference(doc),), using="default").selections[0]
            rows = tuple(TextChunk.objects.filter(doc_id=doc.id,
                representation_generation_id=selection.generation_key).order_by("chunk_number"))
            assert len(rows) == 3
            assert all(0 < len(row.content) < 1200 for row in rows)
            assert all(row.content == doc.full_text[row.start_position:row.end_position] for row in rows)
            assert dense.activate_embedding_generation(stage_dense(doc, transport[0]),
                using="default") == "activated"
            documents.append(doc)
            all_rows.append(rows)
        return CacheCorpus(tuple(documents), outputs, tuple(all_rows), transport[0], settings,
                           f"instrument calibration {token}", batch.user)
    return make
