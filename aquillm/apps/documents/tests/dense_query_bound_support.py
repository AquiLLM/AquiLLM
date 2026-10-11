"""Persisted verified dense generations with complete, honest canonical chunks."""
from apps.collections.models import Collection
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services import embedding_index_store as store
from apps.documents.services.representation_store import freeze_active_representations
from django.contrib.auth.models import User
from lib.embeddings.document_contracts import build_budget
from lib.embeddings.document_provider import embed_document_batch, observe_query_embedding

from .representation_support import reference
from .test_embedding_generation_query import runtime_for


def ready_dense(transport, count=45):
    user = User.objects.create_user(username="bounded-dense")
    collection = Collection.objects.create(name="Bounded dense")
    bodies = [f"source {i}" for i in range(count)]
    doc = RawTextDocument(title="Bounded", full_text="\n".join(bodies),
                          collection=collection, ingested_by=user)
    doc.save(dont_rechunk=True)
    position, rows = 0, []
    for number, body in enumerate(bodies):
        rows.append(TextChunk(doc_id=doc.id, content=body, chunk_number=number,
            start_position=position, end_position=position + len(body)))
        position += len(body) + 1
    TextChunk.objects.bulk_create(rows)
    ref = reference(doc)
    space = transport[0]
    store.persist_embedding_space(space, using="default")
    snapshot = freeze_active_representations((ref,), using="default")
    selection = snapshot.selections[0]
    manifest = store.capture_embedding_manifest(selection, using="default", space=space)
    lease = store.begin_embedding_generation(selection, space_key=space.space_key,
        manifest_sha256=store.embedding_manifest_digest(selection, manifest, space_key=space.space_key),
        expected_count=len(manifest), using="default")
    offline = build_budget()
    for start in range(0, count, 32):
        batch = rows[start:start + 32]
        results = embed_document_batch(tuple(store.source_chunk(row) for row in batch),
                                       space=space, budget=offline)
        store.stage_embedding_batch(lease, chunk_ids=tuple(row.pk for row in batch), results=results,
            input_sha256=tuple(item["prepared_input_sha256"] for item in manifest[start:start + 32]),
            using="default")
    assert store.activate_embedding_generation(lease, using="default") == "activated"
    runtime, policy = runtime_for((doc, ref))
    observation = observe_query_embedding("source", space=space, budget=runtime.budget)
    return snapshot, lease, runtime, policy, observation
