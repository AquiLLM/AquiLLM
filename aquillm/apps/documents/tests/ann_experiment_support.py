"""Public synthetic geometry through real I4 receipts and I3 publication."""

import re
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from django.contrib.auth.models import User

from apps.collections.models import Collection, CollectionPermission
from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.services import embedding_index_store as store
from apps.documents.services import representation_store
from apps.documents.services.chunk_search_scope import freeze_search_scope
from apps.documents.services.representation_contracts import DocumentRefV1
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.embeddings import document_provider as provider
from lib.embeddings.document_contracts import build_budget, declare_space, digest
from lib.embeddings.tests.test_document_contracts import profile
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .ann_native_source import document

MODEL = "r5-deterministic-synthetic-geometry-v1"
ENDPOINT = "http://fake-models:8000/v1"


def synthetic_vector(text):
    match = re.match(r"R5 geometry: (-?\d+),(-?\d+),(-?\d+),(-?\d+)\.", text)
    if match is None:
        raise ValueError("not an R5 public geometry input")
    return [int(value) / 256 for value in match.groups()] + [1.0] + [0.0] * 1019


def install_provider(monkeypatch, settings):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model=MODEL,
            data=[
                SimpleNamespace(index=i, embedding=synthetic_vector(text))
                for i, text in reversed(list(enumerate(kwargs["input"])))
            ],
        )

    monkeypatch.setenv("APP_EMBED_BASE_URL", ENDPOINT)
    monkeypatch.setenv("APP_EMBED_MODEL", MODEL)
    monkeypatch.setattr(
        provider,
        "_client",
        lambda: SimpleNamespace(embeddings=SimpleNamespace(create=create)),
    )
    declaration = asdict(profile())
    engine = provider.engine_manifest_for(profile())
    engine.update(
        weights_revision="r5-public-geometry-v1",
        tokenizer_revision="r5-ascii-integer-geometry-v1",
        compute_precision="fp32",
        pooling_policy="synthetic-coordinate-projection-v1",
        engine_binary_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        engine_config_sha256=digest(
            {"coordinates": 4, "scale": 256, "fixed_nonzero_coordinate": 1.0}
        ),
    )
    space = declare_space(
        engine,
        provider="local-openai",
        model=MODEL,
        query_template="{text}",
        document_template="{text}",
        permitted_modalities=("text",),
        max_input_tokens=declaration["max_input_tokens"],
        max_batch_tokens=declaration["max_batch_tokens"],
    )
    register(space)
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = space.space_key
    return space, calls


def register(space):
    provider.register_verified_profile(
        space,
        endpoint=ENDPOINT,
        deployment_artifact_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        engine_manifest=provider.engine_manifest_for(space),
        query_template="{text}",
        document_template="{text}",
    )
    store.persist_embedding_space(space, using="default")


def activate(doc, space):
    ref = DocumentRefV1(
        doc._meta.label_lower, doc.pkid, doc.id, doc.collection_id, doc.full_text_hash
    )
    selection = representation_store.freeze_active_representations(
        (ref,), using="default"
    ).selections[0]
    manifest = store.capture_embedding_manifest(selection, using="default", space=space)
    lease = store.begin_embedding_generation(
        selection,
        space_key=space.space_key,
        manifest_sha256=store.embedding_manifest_digest(
            selection, manifest, space_key=space.space_key
        ),
        expected_count=len(manifest),
        using="default",
    )
    rows = list(doc.chunks.order_by("chunk_number"))
    assert rows and all(row.representation_generation_id == selection.generation_key for row in rows)
    budget = build_budget()
    for offset in range(0, len(rows), 32):
        batch = rows[offset : offset + 32]
        results = provider.embed_document_batch(
            tuple(store.source_chunk(row) for row in batch), space=space, budget=budget
        )
        store.stage_embedding_batch(
            lease,
            chunk_ids=tuple(row.pk for row in batch),
            results=results,
            input_sha256=tuple(
                item["prepared_input_sha256"] for item in manifest[offset : offset + 32]
            ),
            using="default",
        )
    assert store.activate_embedding_generation(lease, using="default") == "activated"
    return lease


def runtime_for(actor, docs):
    authorization = build_selected_scope_authorization_context(
        principal=actor,
        selected_collection_ids=tuple({doc.collection_id for doc in docs}),
        selected_documents=tuple(docs),
    )
    assert authorization is not None
    return SourceRuntime(TurnBudget(TurnLimits()), authorization)


@pytest.fixture
def ann_case(monkeypatch, settings, upload):
    monkeypatch.setenv('KG_BUILD_ENABLED', 'false')
    space, calls = install_provider(monkeypatch, settings)
    actor = User.objects.create(username="r5-ann-reader")
    writer = User.objects.create(username="r5-ann-ingestion-owner")
    public = Collection.objects.create(name="R5 public synthetic geometry")
    private = Collection.objects.create(name="R5 hidden synthetic geometry")
    CollectionPermission.objects.create(
        user=actor, collection=public, permission="VIEW"
    )
    for collection in (public, private):
        CollectionPermission.objects.create(user=writer, collection=collection, permission="EDIT")
    assert public.user_can_view(actor) and not private.user_can_view(actor)
    doc, chunks, eligible_provenance = document(
        upload,
        writer,
        public,
        [f"R5 geometry: {i},0,0,0. Public synthetic passage.\n" for i in (1, 2, 3)],
        "Eligible",
    )
    hidden, hidden_chunks, hidden_provenance = document(
        upload, writer, private, ["R5 geometry: 0,0,0,0. Hidden nearest control.\n"], "Hidden"
    )
    foreign, foreign_chunks, foreign_provenance = document(
        upload,
        writer,
        public,
        ["R5 geometry: 0,0,0,0. Incompatible nearest control.\n"],
        "Foreign",
    )
    old = activate(doc, space)
    current = activate(doc, space)
    activate(hidden, space)
    engine = provider.engine_manifest_for(space) | {
        "weights_revision": "r5-foreign-geometry-v1"
    }
    other = declare_space(
        engine,
        provider=space.provider,
        model=space.model,
        query_template="{text}",
        document_template="{text}",
        permitted_modalities=("text",),
        max_input_tokens=8192,
        max_batch_tokens=32768,
    )
    register(other)
    activate(foreign, other)
    runtime = runtime_for(actor, (doc, foreign))
    with source_runtime_scope(runtime):
        scope = freeze_search_scope((doc, foreign), using="default")
        observation = provider.observe_query_embedding(
            "R5 geometry: 0,0,0,0. Public query.", space=space, budget=runtime.budget
        )
    return SimpleNamespace(
        actor=actor,
        public=public,
        docs=(doc, foreign),
        chunks=chunks,
        hidden_chunks=hidden_chunks,
        foreign_chunks=foreign_chunks,
        old=old,
        current=current,
        runtime=runtime,
        scope=scope,
        observation=observation,
        space=space,
        calls=calls,
        native_sources=(eligible_provenance, hidden_provenance, foreign_provenance),
    )
