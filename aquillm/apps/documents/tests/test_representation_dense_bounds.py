"""Serving admits only exact matched dense metadata before JSON hydration."""
import json

import pytest
from django.db import connection
from django.db.models import JSONField
from django.test.utils import CaptureQueriesContext

from apps.documents.models.embedding_index import ChunkEmbedding, EmbeddingGeneration
from apps.documents.services import embedding_index_query as query
from apps.documents.services.source_loading import source_runtime_scope, SourcePreparationLimited
from lib.embeddings.tests.test_document_provider import transport as transport

from .dense_query_bound_support import ready_dense

pytestmark = pytest.mark.django_db(transaction=True)


def reject_large_json_decode(monkeypatch):
    original = JSONField.from_db_value
    def guarded(field, value, expression, connection):
        assert not isinstance(value, str) or len(value) <= 250000, "oversized JSON hydrated"
        return original(field, value, expression, connection)
    monkeypatch.setattr(JSONField, "from_db_value", guarded)


def rank(snapshot, dense, observation, runtime, limit=45):
    with source_runtime_scope(runtime):
        return query.rank_compatible_chunks(snapshot=snapshot, dense_snapshot=dense,
            query=observation, limit=limit, using="default", budget=runtime.budget)


def test_large_real_dense_manifest_freeze_is_scalar_only(transport, monkeypatch):
    snapshot, lease, runtime, _, observation = ready_dense(transport, count=640)
    generation = EmbeddingGeneration.objects.get(pk=lease.generation_key)
    assert len(json.dumps(generation.manifest)) > runtime.budget.limits.materialized_codepoints
    original = EmbeddingGeneration.from_db
    def guard(db, names, values):
        assert "manifest" not in names, "serving hydrated the complete generation manifest"
        return original(db, names, values)
    monkeypatch.setattr(EmbeddingGeneration, "from_db", guard)
    dense = query.freeze_dense_generations(snapshot, space_key=observation.compatibility_key, using="default")
    assert len(dense.generation_keys) == 1
    assert len(rank(snapshot, dense, observation, runtime)) == 45
    assert 0 < runtime.budget.text_used["materialized"] <= runtime.budget.limits.materialized_codepoints


def test_45_valid_candidates_fit_original_default_metadata_allowance(transport):
    snapshot, _, runtime, _, observation = ready_dense(transport)
    dense = query.freeze_dense_generations(snapshot, space_key=observation.compatibility_key, using="default")
    with CaptureQueriesContext(connection) as queries:
        rows = rank(snapshot, dense, observation, runtime)
    assert len(rows) == 45 and len({row.chunk_id for row in rows}) == 45
    assert 0 < runtime.budget.text_used["materialized"] <= runtime.budget.limits.materialized_codepoints
    assert not any('"apps_documents_embeddinggeneration"."manifest",' in item["sql"] for item in queries)


@pytest.mark.parametrize("payload", ["receipt", "serving_binding", "manifest"])
def test_oversized_dense_payload_is_rejected_before_materialization(transport, monkeypatch, payload):
    snapshot, lease, runtime, _, observation = ready_dense(transport, count=1)
    dense = query.freeze_dense_generations(snapshot, space_key=observation.compatibility_key, using="default")
    huge = "x" * (runtime.budget.limits.materialized_codepoints + 1)
    if payload == "manifest":
        generation = EmbeddingGeneration.objects.get(pk=lease.generation_key)
        changed = generation.manifest
        changed[0]["unexpected"] = huge
        EmbeddingGeneration.objects.filter(pk=generation.pk).update(manifest=changed)
    else:
        row = ChunkEmbedding.objects.get(generation_id=lease.generation_key)
        changed = {**getattr(row, payload), "unexpected": huge}
        ChunkEmbedding.objects.filter(pk=row.pk).update(**{payload: changed})
    original = ChunkEmbedding.from_db
    def guard(db, names, values):
        assert not {"receipt", "serving_binding"}.intersection(names), "payload read before admission"
        return original(db, names, values)
    monkeypatch.setattr(ChunkEmbedding, "from_db", guard)
    reject_large_json_decode(monkeypatch)
    with pytest.raises(SourcePreparationLimited):
        rank(snapshot, dense, observation, runtime)
    assert runtime.budget.text_used["materialized"] == 0


@pytest.mark.parametrize("change", ["duplicate", "missing"])
def test_unique_matched_manifest_entry_is_required(transport, change):
    snapshot, lease, runtime, _, observation = ready_dense(transport, count=1)
    dense = query.freeze_dense_generations(snapshot, space_key=observation.compatibility_key, using="default")
    generation = EmbeddingGeneration.objects.get(pk=lease.generation_key)
    manifest = generation.manifest * 2 if change == "duplicate" else []
    EmbeddingGeneration.objects.filter(pk=generation.pk).update(manifest=manifest)
    assert rank(snapshot, dense, observation, runtime) == ()


@pytest.mark.parametrize("change", ["receipt", "binding", "entry", "duplicate", "body"])
def test_dense_payload_swap_after_admission_cannot_be_hydrated(transport, monkeypatch, change):
    snapshot, lease, runtime, _, observation = ready_dense(transport, count=1)
    dense = query.freeze_dense_generations(snapshot, space_key=observation.compatibility_key, using="default")
    row = ChunkEmbedding.objects.get(generation_id=lease.generation_key)
    generation = EmbeddingGeneration.objects.get(pk=lease.generation_key)
    original = runtime.budget.reserve_text
    changed = False
    def reserve(count, *, kind):
        nonlocal changed
        accepted = original(count, kind=kind)
        if kind == "materialized" and accepted and not changed:
            changed = True
            if change in {"receipt", "binding"}:
                name = "receipt" if change == "receipt" else "serving_binding"
                ChunkEmbedding.objects.filter(pk=row.pk).update(**{name: {"replacement": "x" * 300000}})
            elif change == "body":
                from apps.documents.models import TextChunk
                TextChunk.objects.filter(pk=row.chunk_id).update(content="changed")
            else:
                manifest = generation.manifest * 2 if change == "duplicate" else [
                    {**generation.manifest[0], "unexpected": "x" * 300000}]
                EmbeddingGeneration.objects.filter(pk=generation.pk).update(manifest=manifest)
        return accepted
    monkeypatch.setattr(runtime.budget, "reserve_text", reserve)
    reject_large_json_decode(monkeypatch)
    assert rank(snapshot, dense, observation, runtime) == ()
    assert changed and runtime.budget.text_used["materialized"] > 0
