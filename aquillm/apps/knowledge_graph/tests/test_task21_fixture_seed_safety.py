from __future__ import annotations

from uuid import uuid5

import pytest
from django.contrib.auth.models import User
from django.core.management import CommandError, call_command
from django.test import override_settings
from django.utils import timezone

from apps.knowledge_graph.tests.task21_fixture_test_support import (
    FIXTURE_ID,
    FIXTURE_NAMESPACE,
    HIDDEN_USERNAME,
    MODEL_SIGNATURE,
    VISIBLE_USERNAME,
    database_counts,
    fixture_module,
    fixture_row_counts,
    install_deterministic_embeddings,
    manifest_checksum,
    seed,
    strict_eval_environment,
)

_STRICT_EVAL_ENVIRONMENT = strict_eval_environment


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
@pytest.mark.parametrize(
    ("setting_overrides", "environment", "message"),
    [
        ({"DEBUG": False}, {}, "DEBUG"),
        ({}, {"KG_EVAL_BYPASS_ALLOWED": "0"}, "KG_EVAL_BYPASS_ALLOWED"),
        ({}, {"KG_BUILD_ENABLED": "1"}, "KG_BUILD_ENABLED"),
        ({}, {"KG_OVERLAY_ENABLED": "1"}, "KG_OVERLAY_ENABLED"),
        ({}, {"COHERE_KEY": "forbidden"}, "COHERE_KEY"),
        ({}, {"APP_EMBED_BASE_URL": "https://example.invalid/v1"}, "local"),
        ({}, {"APP_EMBED_MODEL_REVISION": "main"}, "checkpoint"),
        ({}, {"APP_EMBED_MODEL_REVISION": "release-v1"}, "checkpoint"),
        ({}, {"APP_EMBED_TOKENIZER_REVISION": "c" * 40}, "tokenizer"),
        ({}, {"APP_EMBED_CODE_REVISION": "c" * 40}, "code"),
        ({}, {"APP_EMBED_MODEL": "/app/local-model"}, "model"),
        ({}, {"APP_EMBED_MODEL": "https://host/model"}, "model"),
        ({}, {"APP_EMBED_MODEL": "owner/name with space"}, "model"),
        ({}, {"APP_EMBED_MODEL": "--owner/name"}, "model"),
        ({}, {"APP_EMBED_MODEL": "../mutable"}, "model"),
        ({}, {"APP_EMBED_MODEL": r"owner\name"}, "model"),
        ({}, {"APP_EMBED_MODEL": "owner/name/extra"}, "model"),
    ],
)
def test_seed_preconditions_fail_before_embedding_or_database_writes(
    tmp_path, monkeypatch, setting_overrides, environment, message
) -> None:
    monkeypatch.setattr(
        fixture_module(),
        "strict_index_embedding_signature",
        lambda: (_ for _ in ()).throw(AssertionError("embedding reached")),
    )
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    manifest_path = tmp_path / "fixture.json"
    with override_settings(**setting_overrides):
        with pytest.raises(CommandError, match=message):
            call_command(
                "seed_knowledge_graph_eval_fixture",
                "--fixture-manifest",
                str(manifest_path),
            )
    assert fixture_row_counts() == (0, 0, 0, 0)
    assert not User.objects.filter(
        username__in=(VISIBLE_USERNAME, HIDDEN_USERNAME)
    ).exists()
    assert not manifest_path.exists()


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_seed_rejects_manifest_path_line_break_before_embedding(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        fixture_module(),
        "strict_index_embedding_signature",
        lambda: (_ for _ in ()).throw(AssertionError("embedding reached")),
    )
    with pytest.raises(CommandError, match="path"):
        call_command(
            "seed_knowledge_graph_eval_fixture",
            "--fixture-manifest",
            str(tmp_path / "fixture\nmanifest.json"),
        )
    assert fixture_row_counts() == (0, 0, 0, 0)


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
@pytest.mark.parametrize("failure", ("provider", "vector"))
def test_seed_embedding_failure_leaves_no_rows_or_manifest(
    tmp_path, monkeypatch, failure
) -> None:
    fixture_seed = fixture_module()
    monkeypatch.setattr(
        fixture_seed, "strict_index_embedding_signature", lambda: MODEL_SIGNATURE
    )
    if failure == "provider":

        def embed(*_args, **_kwargs):
            raise RuntimeError("synthetic endpoint unavailable")

    else:

        def embed(queries, *, expected_model_signature):
            assert expected_model_signature == MODEL_SIGNATURE
            return [(index, [float("nan")] * 1024) for index in range(len(queries))], (
                MODEL_SIGNATURE
            )

    monkeypatch.setattr(fixture_seed, "get_strict_index_embeddings", embed)
    manifest_path = tmp_path / "fixture.json"
    with pytest.raises(CommandError, match="embedding"):
        call_command(
            "seed_knowledge_graph_eval_fixture",
            "--fixture-manifest",
            str(manifest_path),
        )
    assert fixture_row_counts() == (0, 0, 0, 0)
    assert not manifest_path.exists()


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_seed_rejects_operator_manifest_before_embedding(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        fixture_module(),
        "strict_index_embedding_signature",
        lambda: (_ for _ in ()).throw(AssertionError("embedding reached")),
    )
    manifest_path = tmp_path / "fixture.json"
    manifest_path.write_bytes(b"operator-owned\n")
    with pytest.raises(CommandError, match="already exists"):
        call_command(
            "seed_knowledge_graph_eval_fixture",
            "--fixture-manifest",
            str(manifest_path),
        )
    assert manifest_path.read_bytes() == b"operator-owned\n"
    assert fixture_row_counts() == (0, 0, 0, 0)


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
@pytest.mark.parametrize("collision", ("collection", "document", "chunk", "request"))
def test_seed_rejects_deterministic_identity_collision_before_embedding(
    tmp_path, monkeypatch, collision
) -> None:
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.models import GraphRebuildRequest

    monkeypatch.setattr(
        fixture_module(),
        "strict_index_embedding_signature",
        lambda: (_ for _ in ()).throw(AssertionError("embedding reached")),
    )
    document_id = uuid5(FIXTURE_NAMESPACE, "document:doc-atlas-dataset")
    if collision == "collection":
        Collection.objects.create(name=f"{FIXTURE_ID}-authorized-a")
    elif collision == "document":
        owner = User.objects.create_user(username="collision-document-owner")
        collection = Collection.objects.create(name="collision-document-collection")
        RawTextDocument.objects.bulk_create(
            [
                RawTextDocument(
                    id=document_id,
                    title="Synthetic collision",
                    full_text="Synthetic collision text.",
                    full_text_hash=RawTextDocument.hash_fn("Synthetic collision text."),
                    collection=collection,
                    ingested_by=owner,
                    ingestion_complete=True,
                )
            ]
        )
    elif collision == "chunk":
        TextChunk.objects.bulk_create(
            [
                TextChunk(
                    doc_id=document_id,
                    content="S",
                    start_position=0,
                    end_position=1,
                    chunk_number=0,
                    modality=TextChunk.Modality.TEXT,
                    metadata={"foreign": True},
                    embedding=[0.0] * 1024,
                )
            ]
        )
    else:
        GraphRebuildRequest.objects.create(
            id=uuid5(FIXTURE_NAMESPACE, "rebuild:authorized-a"),
            scope_type=GraphRebuildRequest.ScopeType.ALL,
            scope_id="",
            requested_documents=[],
            status=GraphRebuildRequest.Status.RUNNING,
            started_at=timezone.now(),
            enumeration_high_water=0,
            document_publication_state=GraphRebuildRequest.PublicationState.NOT_APPLICABLE,
            collection_publication_state=GraphRebuildRequest.PublicationState.NOT_APPLICABLE,
        )
    before = database_counts()
    with pytest.raises(CommandError, match="collision"):
        call_command(
            "seed_knowledge_graph_eval_fixture",
            "--fixture-manifest",
            str(tmp_path / "fixture.json"),
        )
    assert database_counts() == before


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_manifest_publication_failure_rolls_back_and_is_retryable(
    tmp_path, monkeypatch
) -> None:
    fixture_seed = fixture_module()
    observed = install_deterministic_embeddings(monkeypatch)
    publisher = fixture_seed._atomic_publish_manifest
    monkeypatch.setattr(
        fixture_seed,
        "_atomic_publish_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            fixture_seed.FixtureSeedError("manifest publication failed")
        ),
    )
    manifest_path = tmp_path / "fixture.json"
    with pytest.raises(CommandError, match="publication"):
        call_command(
            "seed_knowledge_graph_eval_fixture",
            "--fixture-manifest",
            str(manifest_path),
        )
    assert len(observed) == 1
    assert fixture_row_counts() == (0, 0, 0, 0)
    assert not manifest_path.exists()
    monkeypatch.setattr(fixture_seed, "_atomic_publish_manifest", publisher)
    payload, _output, retried = seed(manifest_path, monkeypatch)
    assert len(retried) == 1 and manifest_checksum(payload)
    assert fixture_row_counts() == (5, 28, 30, 5)


def test_exact_unique_rows_reject_duplicate_regardless_of_row_order() -> None:
    from apps.knowledge_graph.evals.fixture_seed_contract import FixtureSeedError
    from apps.knowledge_graph.evals.fixture_seed_query import require_exact_unique_rows

    for rows in (((1, "owned"), (1, "foreign")), ((1, "foreign"), (1, "owned"))):
        with pytest.raises(FixtureSeedError, match="topology"):
            require_exact_unique_rows(rows, {1}, key=lambda row: row[0])
