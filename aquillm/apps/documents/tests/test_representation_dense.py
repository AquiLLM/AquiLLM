"""Two verified dense profiles share one actual retained structural generation."""
from unittest.mock import patch

import pytest

from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.models import TextChunk
from apps.documents.services import embedding_index_store as store, embedding_index_query as query
from apps.documents.services.representation_store import freeze_active_representations, active_chunk_scope
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.documents.tasks.embedding_index import rebuild_document_embeddings, build_embedding_generation
from lib.embeddings.document_provider import register_verified_profile, engine_manifest_for, observe_query_embedding
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.embeddings.tests.test_document_contracts import profile
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .representation_support import retained as retained, upload as upload, activate, reference

pytestmark = pytest.mark.django_db(transaction=True)


def test_legacy_null_digest_bytes_remain_stable():
    from uuid import UUID
    from apps.documents.services.representation_contracts import (
        DocumentRefV1, RepresentationSelectionV1, RepresentationSnapshotV1,
    )
    from apps.documents.services.representation_store import representation_snapshot_digest
    ref = DocumentRefV1("apps_documents.rawtextdocument", 1, UUID("11111111-1111-4111-8111-111111111111"), 2, "a" * 64)
    snapshot = RepresentationSnapshotV1((RepresentationSelectionV1(ref, None, "a" * 64, None),))
    assert representation_snapshot_digest(snapshot) == "1fb03a912b9dae3e7783d36a778d7268d5682ebc2fbea00300e654bee5ff8795"
    assert query.dense_generation_snapshot_digest(snapshot, (), space_key="b" * 64) == (
        "3bf871b61cb44d1f80482bf3b4dc7de4b475215e55fc12a41f1a65da45bb6653")


def test_real_nonnull_generation_builds_and_ranks_two_profiles(retained, transport):
    doc, output, old = retained
    assert activate(doc, output) == "activated"
    doc.refresh_from_db()
    frozen = freeze_active_representations((reference(doc),), using="default")
    selection = frozen.selections[0]
    source_rows = list(active_chunk_scope(TextChunk.objects.all(), snapshot=frozen))
    b = profile(compute_precision="bf16")
    register_verified_profile(b, endpoint="http://fake-models:8000/v1", deployment_artifact_sha256="c" * 64,
                              engine_manifest=engine_manifest_for(b), query_template="{text}", document_template="{text}")
    generations = []
    for space in (transport[0], b):
        store.persist_embedding_space(space, using="default")
        with patch.object(build_embedding_generation, "delay"):
            key = rebuild_document_embeddings(selection, space_key=space.space_key, using="default")
        assert build_embedding_generation.run(key, using="default", timeout_ms=120000) == "activated"
        generations.append(key)
        dense = query.freeze_dense_generations(frozen, space_key=space.space_key, using="default")
        assert len(dense.generation_keys) == 1
        auth = build_selected_scope_authorization_context(
            principal=doc.ingested_by, selected_collection_ids=(doc.collection_id,), selected_documents=(doc,),
        )
        assert auth is not None
        runtime = SourceRuntime(TurnBudget(TurnLimits()), auth)
        observation = observe_query_embedding("table", space=space, budget=runtime.budget)
        with source_runtime_scope(runtime):
            ranked = query.rank_compatible_chunks(snapshot=frozen, dense_snapshot=dense, query=observation,
                limit=10, using="default", budget=runtime.budget)
        assert {item.chunk_id for item in ranked} == {row.pk for row in source_rows}
        assert old.pk not in {item.chunk_id for item in ranked}
    assert generations[0] != generations[1]
    assert TextChunk.objects.filter(pk=old.pk, content="old table 10").exists()
