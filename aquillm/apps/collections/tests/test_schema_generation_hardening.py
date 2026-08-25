from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def _candidate_with_aliases(aliases: list[str]) -> dict:
    return {
        "entities": [
            {"name": "Researcher", "description": "A person.", "aliases": aliases},
            {"name": "Organization", "description": "A company.", "aliases": []},
        ],
        "relations": [{
            "name": "Works For", "description": "Employment.", "direction": "directed",
            "allowed_head_types": ["Researcher"], "allowed_tail_types": ["Organization"],
        }],
    }


def test_default_backend_forces_a_local_gliner2_cache_only_factory_configuration(monkeypatch):
    """Ambient extractor defaults must not let generated-schema evidence reach Hugging Face."""

    from apps.collections.services import schema_generation
    from apps.collections.services import schema_generation_support as support
    from lib.knowledge_graph import config
    from lib.knowledge_graph.extractors import factory
    from lib.knowledge_graph.config import ExtractionSettings

    ambient = ExtractionSettings(
        build_enabled=False, provider="unsafe_remote", model_id="example/model", model_revision="",
        device="cpu", batch_size=8, max_batch_characters=4_000, cache_dir=Path("C:/cache"),
        local_files_only=False, fail_open=True,
    )
    captured = {}
    sentinel = object()
    monkeypatch.setattr(config, "load_extraction_settings", lambda: ambient)
    monkeypatch.setattr(
        factory,
        "get_extraction_backend",
        lambda *, settings: captured.setdefault("settings", settings) and sentinel,
    )

    assert schema_generation.collect_candidate_evidence is support.collect_candidate_evidence
    assert support._default_backend() is sentinel
    assert captured["settings"].provider == "gliner2_local"
    assert captured["settings"].local_files_only is True
    assert captured["settings"].fail_open is False


def test_normalizer_rejects_aliases_that_exceed_persisted_dto_bounds():
    """Removing alias limits could persist echoed collection text from a local response."""

    from apps.collections.services.schema_generation import InvalidSchemaCandidate, normalize_schema_candidate

    oversized_alias_sets = (
        [f"alias_{index}" for index in range(17)],
        ["a" * 129],
        [f"{index:02d}" + ("a" * 63) for index in range(16)],
    )

    for aliases in oversized_alias_sets:
        with pytest.raises(InvalidSchemaCandidate, match="aliases"):
            normalize_schema_candidate(_candidate_with_aliases(aliases))


def test_normalizer_accepts_aliases_at_each_persisted_dto_boundary():
    """Tightening an alias boundary by one character must not reject a bounded candidate."""

    from apps.collections.services.schema_generation import normalize_schema_candidate

    aliases = [f"{index:02d}" + ("a" * 62) for index in range(16)]

    definitions = normalize_schema_candidate(_candidate_with_aliases(aliases))

    assert definitions["entities"][1]["values"]["aliases"] == sorted(aliases)


def test_normalizer_emits_backend_editor_capabilities_without_atomic_renames():
    """Generated definitions must expose exactly the fields the backend editor accepts."""

    from apps.collections.services.schema_generation import normalize_schema_candidate

    definitions = normalize_schema_candidate(_candidate_with_aliases([]))
    entities = {item["key"]: item["capabilities"] for item in definitions["entities"]}

    assert entities["researcher"] == {
        "editable_fields": [
            "description", "aliases", "default_retrieval_weight", "default_suppression_policy",
            "default_suppression_threshold",
        ],
        "removable": True,
        "renameable": False,
    }
    assert definitions["relations"][0]["capabilities"] == {
        "editable_fields": ["description", "direction", "allowed_head_types", "allowed_tail_types"],
        "removable": True,
        "renameable": False,
    }


def test_public_sampler_does_not_probe_documents_rejected_by_eligibility(monkeypatch):
    """The EXISTS identity filter must remove early empty documents before sampling."""

    from apps.collections.services import schema_generation

    rows = {"document-a": [], "document-b": [], "document-c": [(3, 0, "charlie")], "document-d": [(4, 0, "delta")]}
    calls = []
    monkeypatch.setattr(schema_generation, "_eligible_collection_documents", lambda collection_id: iter({"id": document_id} for document_id in ("document-c", "document-d")))
    monkeypatch.setattr(
        schema_generation,
        "_next_sample_chunk",
        lambda document_id, after_chunk_number, character_limit: calls.append(document_id) or next((schema_generation.SchemaSample(document_id, chunk_id, number, text[:character_limit]) for chunk_id, number, text in rows[document_id] if number > after_chunk_number), None),
    )

    samples = schema_generation.sample_collection_chunks(1, max_chunks=2, max_characters=20)

    assert [sample.document_id for sample in samples] == ["document-c", "document-d"]
    assert calls == ["document-c", "document-d"]


def test_public_sampler_uses_eligible_documents_for_true_round_robin(monkeypatch):
    """Existence filtering must skip empties before the bounded chunk queries start."""

    from apps.collections.services import schema_generation

    rows = {"document-a": [(1, 0, "a1"), (3, 1, "a2")], "document-c": [(2, 0, "c1")]}
    calls = []
    monkeypatch.setattr(
        schema_generation,
        "_eligible_collection_documents",
        lambda collection_id: iter({"id": document_id} for document_id in rows),
    )
    monkeypatch.setattr(
        schema_generation,
        "_next_sample_chunk",
        lambda document_id, after_chunk_number, character_limit: calls.append(document_id) or next(
            (schema_generation.SchemaSample(document_id, chunk_id, number, text) for chunk_id, number, text in rows[document_id] if number > after_chunk_number),
            None,
        ),
    )

    samples = schema_generation.sample_collection_chunks(1, max_chunks=3, max_characters=20)

    assert [sample.text for sample in samples] == ["a1", "c1", "a2"]
    assert calls == ["document-a", "document-c", "document-a"]


def test_public_sampler_all_empty_uses_no_per_document_chunk_queries(monkeypatch):
    """All-empty collections must end after the fixed eligibility queries."""

    from apps.collections.services import schema_generation

    monkeypatch.setattr(schema_generation, "_eligible_collection_documents", lambda collection_id: iter(()))
    monkeypatch.setattr(schema_generation, "_next_sample_chunk", lambda *args: pytest.fail("empty documents were probed individually"))

    assert schema_generation.sample_collection_chunks(1, max_chunks=32, max_characters=48_000) == []


def test_eligible_document_lookup_is_bounded_for_an_all_empty_collection(monkeypatch):
    """All-empty input performs one EXISTS-backed identity query per document type."""

    from django.apps import apps
    from apps.collections.services import schema_generation

    names = ("PDFDocument", "TeXDocument", "RawTextDocument", "VTTDocument", "HandwrittenNotesDocument", "ImageUploadDocument", "MediaUploadDocument", "DocumentFigure")
    filters, annotations, selections = [], [], []

    class QuerySet:
        def __init__(self, name):
            self.name = name

        def filter(self, **kwargs):
            filters.append((self.name, kwargs))
            return self

        def annotate(self, **kwargs):
            annotations.append((self.name, kwargs))
            return self

        def order_by(self, *fields):
            return self

        def values(self, *fields):
            selections.append((self.name, fields))
            return self

        def iterator(self):
            return iter(())

    monkeypatch.setattr(apps, "get_model", lambda app_label, name: SimpleNamespace(objects=QuerySet(name)))

    assert list(schema_generation._eligible_collection_documents(1)) == []
    assert filters == [
        item
        for name in names
        for item in ((name, {"collection_id": 1, "ingestion_complete": True}), (name, {"has_usable_text": True}))
    ]
    assert [name for name, _ in annotations] == list(names)
    assert selections == [(name, ("id",)) for name in names]


def test_locked_source_signature_locks_all_documents_but_hashes_only_completed(monkeypatch):
    """The final fence must block document completion changes and FK inserts."""

    from django.apps import apps
    from apps.collections.services import schema_generation

    names = ("PDFDocument", "TeXDocument", "RawTextDocument", "VTTDocument", "HandwrittenNotesDocument", "ImageUploadDocument", "MediaUploadDocument", "DocumentFigure")
    records = {"PDFDocument": [{"id": "complete", "full_text_hash": "a" * 64, "ingestion_complete": True}], "TeXDocument": [{"id": "pending", "full_text_hash": "b" * 64, "ingestion_complete": False}]}
    filters, locks, projections = [], [], []

    class QuerySet:
        def __init__(self, name):
            self.name = name

        def filter(self, **kwargs):
            filters.append((self.name, kwargs))
            return self

        def select_for_update(self):
            locks.append(self.name)
            return self

        def order_by(self, *fields):
            return self

        def values(self, *fields):
            projections.append((self.name, fields))
            return self

        def iterator(self):
            return iter(records.get(self.name, []))

    monkeypatch.setattr(apps, "get_model", lambda app_label, name: SimpleNamespace(objects=QuerySet(name)))

    signature = schema_generation._locked_collection_source_signature(1)

    digest = hashlib.sha256()
    digest.update(b"complete\0" + (b"a" * 64) + b"\n")
    assert signature == digest.hexdigest()
    assert filters == [(name, {"collection_id": 1}) for name in names]
    assert locks == list(names)
    assert all(fields == ("id", "full_text_hash", "ingestion_complete") for _, fields in projections)


@pytest.mark.parametrize("failure", ("timeout", "status", "malformed", "oversize"))
def test_direct_vllm_transport_rejects_unsafe_response_states(monkeypatch, failure):
    """A local transport must never buffer unbounded or malformed server output."""

    from apps.collections.services import schema_generation_support as support

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            assert limit == support._MAX_VLLM_RESPONSE_BYTES + 1
            if failure == "oversize":
                return b"x" * limit
            if failure == "malformed":
                return b"not-json"
            return b'{"ok":true}'

    if failure == "timeout":
        monkeypatch.setattr(support, "_open_local_request", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))
    elif failure == "status":
        response = Response()
        response.status = 503
        monkeypatch.setattr(support, "_open_local_request", lambda *args, **kwargs: response)
    else:
        monkeypatch.setattr(support, "_open_local_request", lambda *args, **kwargs: Response())

    with pytest.raises(support.LocalVLLMTransportError):
        support._post_local_vllm_json("http://vllm:8000/v1/chat/completions", {}, {}, 1)
