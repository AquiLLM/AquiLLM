"""Offline gold mapping cannot require or weaken a serving source runtime."""

import importlib.util
import json
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model

from apps.chat.refs import CollectionsRef
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import current_turn

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def cli(settings):
    settings.RAG_CACHE_ENABLED = False
    spec = importlib.util.spec_from_file_location(
        "target_preflight_cli",
        Path(__file__).parents[1] / "scripts/replay_retrieval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_for(cli, document, chunk):
    return {
        "source_sha256": cli.fingerprint(document.full_text),
        "text_sha256": cli.fingerprint(chunk.content),
    }


@pytest.mark.parametrize("mode", ["legacy", "source"])
def test_offline_targets_map_without_creating_a_serving_runtime(
    docs, cli, monkeypatch, mode
):
    from apps.chat.services.tool_wiring.source_documents import (
        selected_document_metadata,
    )

    user, document, chunks, _ = docs
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", mode)
    assert current_source_runtime() is None and current_turn() is None
    if mode == "source":
        with pytest.raises(SourceScopeUnavailable):
            selected_document_metadata(user, CollectionsRef([document.collection_id]))
    sources = cli.selected_sources(user, [document.collection_id])
    assert [source["source"] for source in sources] == [document.full_text]
    assert cli.resolve_targets([target_for(cli, document, chunks[1])], sources)[0][
        "chunk_ids"
    ] == [chunks[1].pk]
    assert current_source_runtime() is None and current_turn() is None
    if mode == "source":
        with pytest.raises(SourceScopeUnavailable):
            selected_document_metadata(user, CollectionsRef([document.collection_id]))


def test_offline_preflight_excludes_unselected_and_ungranted_sources(
    docs, cli, monkeypatch
):
    user, document, _, _ = docs
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    hidden_collection = Collection.objects.create(name="hidden oracle")
    hidden = RawTextDocument(
        title="hidden",
        full_text="private gold",
        collection=hidden_collection,
        ingested_by=user,
    )
    hidden.save(dont_rechunk=True)
    source = cli.selected_sources(user, [document.collection_id, hidden_collection.pk])
    assert [row["source"] for row in source] == [document.full_text]
    assert cli.selected_sources(user, [hidden_collection.pk]) == []
    CollectionPermission.objects.create(
        user=user, collection=hidden_collection, permission="VIEW"
    )
    assert [
        row["source"] for row in cli.selected_sources(user, [document.collection_id])
    ] == [document.full_text]


def test_offline_preflight_rejects_stale_deactivated_principal(docs, cli, monkeypatch):
    user, document, _, _ = docs
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "legacy")
    get_user_model().objects.filter(pk=user.pk).update(is_active=False)
    assert user.is_active  # Persisted activity must override the stale object.
    with pytest.raises(ValueError, match="principal_inactive"):
        cli.selected_sources(user, [document.collection_id])


@pytest.mark.parametrize("mutation", ["permission", "activity"])
def test_offline_preflight_rechecks_authority_after_corpus_read(
    docs, cli, monkeypatch, mutation
):
    user, document, _, _ = docs
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "legacy")
    original = TextChunk.objects.filter

    def revoke(*args, **kwargs):
        rows = original(*args, **kwargs)
        if mutation == "permission":
            CollectionPermission.objects.filter(user=user).delete()
        else:
            get_user_model().objects.filter(pk=user.pk).update(is_active=False)
        return rows

    monkeypatch.setattr(TextChunk.objects, "filter", revoke)
    expected = (
        "source_authorization_failed"
        if mutation == "permission"
        else "principal_inactive"
    )
    with pytest.raises(ValueError, match=expected):
        cli.selected_sources(user, [document.collection_id])


@pytest.mark.parametrize("mapping", ["valid", "missing", "ambiguous"])
def test_source_cli_resolves_independent_gold_before_actual_replay(
    docs, cli, monkeypatch, tmp_path, mapping
):
    from apps.chat.tests.test_rag_preservation_routes import fake_search

    user, document, chunks, _ = docs
    fake_search(monkeypatch)
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "32000")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    target = target_for(cli, document, chunks[0])
    collections = [document.collection_id]
    if mapping == "missing":
        target["source_sha256"] = "a" * 64
    elif mapping == "ambiguous":
        other = Collection.objects.create(name="duplicate source")
        CollectionPermission.objects.create(
            user=user, collection=other, permission="VIEW"
        )
        collections.append(other.pk)
        duplicate = RawTextDocument(
            title="duplicate",
            full_text=document.full_text,
            collection=other,
            ingested_by=user,
        )
        duplicate.save(dont_rechunk=True)
    manifest, output = tmp_path / "question.json", tmp_path / "report.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "questions": [
                    {
                        "id": "Q3",
                        "question": "Summarize the selected documents",
                        "targets": [target],
                    }
                ],
            }
        )
    )
    result = cli.main(
        [
            "--manifest",
            str(manifest),
            "--principal-id",
            str(user.pk),
            "--collection-ids",
            *map(str, collections),
            "--output",
            str(output),
            "--revision",
            "b" * 40,
            "--experiment",
            "current",
            "--evidence-text-mode",
            "source",
        ]
    )
    report = json.loads(output.read_text())
    assert current_source_runtime() is None and current_turn() is None
    assert document.full_text not in output.read_text()
    assert chunks[0].content not in output.read_text()
    if mapping == "valid":
        assert result == 0 and report["success"] is True, report
        assert report["effective"]["cache_enabled"] is False
        turn = report["turns"][0]
        assert turn["success"] and turn["targets"][0]["packet_ranks"]
        assert not turn["generation_attempted"] and not turn["sdk_started"]
        assert turn["targets"][0]["source_sha256"] == target["source_sha256"]
    else:
        assert result == 1 and report["success"] is False
        assert report["error_type"] == "MappingError" and report["turns"] == []
