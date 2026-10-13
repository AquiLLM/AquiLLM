"""Disposable native paragraphs and real immutable receipt producers."""
from dataclasses import asdict
import json
from types import SimpleNamespace
from time import monotonic

from django.contrib.auth.models import User
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models.embedding_index import EmbeddingGeneration, ChunkEmbedding
from lib.embeddings.document_contracts import declare_space
from lib.embeddings import document_provider as provider
from .ann_experiment_support import install_provider, register, activate
from .ann_native_source import document


def build(upload, monkeypatch, settings, frozen, output, deadline):
    monkeypatch.setenv("KG_BUILD_ENABLED", "false")
    space, calls = install_provider(monkeypatch, settings)
    actor = User.objects.create(username="r5-measurement-reader")
    writer = User.objects.create(username="r5-measurement-ingestion-owner")
    public = Collection.objects.create(name="R5 measurement public")
    private = Collection.objects.create(name="R5 measurement hidden")
    CollectionPermission.objects.create(user=actor, collection=public, permission="VIEW")
    for collection in (public, private):
        CollectionPermission.objects.create(user=writer, collection=collection, permission="EDIT")
    assert public.user_can_view(actor) and not private.user_can_view(actor)
    docs, chunks, observations, leases = [], [], [], []
    offset = 0

    def create(texts, collection, title):
        if monotonic() + 120 >= deadline:
            raise TimeoutError("corpus preparation wall allowance")
        doc, rows, observed = document(upload, writer, collection, texts, title)
        artifact = observed.pop("artifact_bytes")
        (output / (title + ".docx")).write_bytes(artifact)
        observed.update(document_id=str(doc.id), collection_id=doc.collection_id,
                        chunk_ids=[r.pk for r in rows], source_sha256=doc.full_text_hash)
        observations.append(observed)
        (output / "native-progress.json").write_text(json.dumps(observations, default=str, indent=2), encoding="utf-8")
        return doc, rows

    for index, size in enumerate(frozen["groups"]):
        texts = [p["text"] for p in frozen["passages"][offset:offset + size]]
        doc, rows = create(texts, public, f"main-{index:02d}")
        lease = activate(doc, space)
        docs.append(doc)
        chunks.extend(rows)
        leases.append(lease)
        (output / "dense-progress.json").write_text(json.dumps(
            [{"generation_key": str(x.generation_key)} for x in leases]), encoding="utf-8")
        offset += size
    hidden, hidden_rows = create([q["text"] for q in frozen["queries"]], private, "hidden")
    hidden_lease = activate(hidden, space)
    control, control_rows = create([q["text"] for q in frozen["queries"]], public, "control")
    old = activate(control, space)
    # A real second compatible generation retires the first. Both remain history.
    current = activate(control, space)
    engine = provider.engine_manifest_for(space) | {"weights_revision": "r5-foreign-geometry-v1"}
    foreign = declare_space(engine, provider=space.provider, model=space.model,
        query_template="{text}", document_template="{text}", permitted_modalities=("text",),
        max_input_tokens=8192, max_batch_tokens=32768)
    register(foreign)
    foreign_lease = activate(control, foreign)
    assert old.generation_key != current.generation_key != foreign_lease.generation_key
    # A different profile does not retire a compatible owner's current generation.
    # This control retains both valid spaces; query-space filtering must exclude foreign rows.
    keys = [lease.generation_key for lease in leases] + [hidden_lease.generation_key,
        old.generation_key, current.generation_key, foreign_lease.generation_key]
    generations = list(EmbeddingGeneration.objects.filter(key__in=keys).values(
        "key", "owner_id", "state", "manifest_sha256", "expected_count",
        "owner__space_id", "owner__active_generation_id"))
    by_key = {row["key"]: row for row in generations}
    assert by_key[old.generation_key]["owner__active_generation_id"] == current.generation_key
    assert by_key[current.generation_key]["owner_id"] != by_key[foreign_lease.generation_key]["owner_id"]
    assert by_key[foreign_lease.generation_key]["owner__active_generation_id"] == foreign_lease.generation_key
    # Offline corpus attestation is separate from query-ledger materialization.
    receipts = list(ChunkEmbedding.objects.filter(generation_id__in=keys).order_by("pk").values(
        "pk", "chunk_id", "generation_id", "input_sha256", "source_md5", "receipt", "serving_binding"))
    return SimpleNamespace(actor=actor, docs=tuple(docs), chunks=tuple(chunks), space=space,
        control=control, control_rows=tuple(control_rows), control_current=current,
        hidden_rows=tuple(hidden_rows), retired=old, foreign=foreign_lease,
        manifest={"native_sources": observations, "generations": generations,
                  "receipts": receipts, "space": asdict(space),
                  "full_authorized_document_ids": [str(d.id) for d in docs] + [str(control.id)],
                  "hidden_document_id": str(hidden.id), "provider_calls": len(calls)})
