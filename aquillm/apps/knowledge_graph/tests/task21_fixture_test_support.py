from __future__ import annotations

import importlib
from io import StringIO
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from django.contrib.auth.models import Group, User
from django.core.management import call_command
from django.utils import timezone

FIXTURE_ID = "kg-task20-synthetic-v1"
FIXTURE_NAMESPACE = uuid5(NAMESPACE_URL, FIXTURE_ID)
VISIBLE_USERNAME = f"{FIXTURE_ID}-visible"
HIDDEN_USERNAME = f"{FIXTURE_ID}-hidden"
MODEL = "Qwen/Qwen3-VL-Embedding-2B"
REVISION = "b" * 40
MODEL_SIGNATURE = (
    f"local-openai:{MODEL}@{REVISION}:endpoint={'a' * 64}:"
    "dims=1024:prep=kg-entity-v1:max_chars=8192:batch=64"
)
PHYSICAL_BINDINGS = {
    "collection-policy-a": "authorized-a",
    "collection-policy-b": "authorized-b",
    "collection-public": "authorized-a",
    "collection-research-a": "authorized-c",
    "collection-research-b": "authorized-d",
    "collection-security-private": "hidden",
}


@pytest.fixture(autouse=True)
def strict_eval_environment(monkeypatch) -> None:
    monkeypatch.setenv("KG_EVAL_BYPASS_ALLOWED", "1")
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    monkeypatch.setenv("KG_OVERLAY_ENABLED", "0")
    monkeypatch.setenv("COHERE_KEY", "")
    monkeypatch.setenv("APP_EMBED_BASE_URL", "http://vllm_embed:8000/v1")
    monkeypatch.setenv("APP_EMBED_API_KEY", "EMPTY")
    monkeypatch.setenv("APP_EMBED_MODEL", MODEL)
    monkeypatch.setenv("APP_EMBED_MODEL_REVISION", REVISION)
    monkeypatch.setenv("APP_EMBED_TOKENIZER_REVISION", REVISION)
    monkeypatch.setenv("APP_EMBED_CODE_REVISION", REVISION)
    monkeypatch.setenv("APP_EMBED_DIMS", "1024")
    monkeypatch.setenv("APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE", "0")


def fixture_module():
    return importlib.import_module("apps.knowledge_graph.evals.fixture_seed")


def install_deterministic_embeddings(monkeypatch):
    fixture_seed = fixture_module()
    observed: list[list[str]] = []

    def embed(queries, *, expected_model_signature):
        assert expected_model_signature == MODEL_SIGNATURE
        observed.append(list(queries))
        return [
            (index, [float(index + 1) / 100.0] * 1024) for index in range(len(queries))
        ], MODEL_SIGNATURE

    monkeypatch.setattr(
        fixture_seed, "strict_index_embedding_signature", lambda: MODEL_SIGNATURE
    )
    monkeypatch.setattr(fixture_seed, "get_strict_index_embeddings", embed)
    return observed


def seed(manifest_path: Path, monkeypatch):
    observed = install_deterministic_embeddings(monkeypatch)
    output = StringIO()
    call_command(
        "seed_knowledge_graph_eval_fixture",
        "--fixture-manifest",
        str(manifest_path),
        stdout=output,
    )
    from apps.knowledge_graph.evals.fixture_manifest import load_fixture_manifest

    return load_fixture_manifest(manifest_path), output.getvalue(), observed


def manifest_checksum(payload: dict[str, object]) -> str:
    from apps.knowledge_graph.evals.fixture_manifest import fixture_manifest_checksum

    return fixture_manifest_checksum(payload)


def cleanup(manifest_path: Path, payload: dict[str, object]) -> str:
    output = StringIO()
    call_command(
        "seed_knowledge_graph_eval_fixture",
        "--cleanup",
        "--fixture-manifest",
        str(manifest_path),
        "--expected-manifest-checksum",
        manifest_checksum(payload),
        stdout=output,
    )
    return output.getvalue()


def fixture_row_counts() -> tuple[int, int, int, int]:
    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument, TextChunk

    collection_ids = tuple(
        Collection.objects.filter(name__startswith=FIXTURE_ID).values_list(
            "pk", flat=True
        )
    )
    document_ids = RawTextDocument.objects.filter(
        collection_id__in=collection_ids
    ).values_list("id", flat=True)
    return (
        len(collection_ids),
        RawTextDocument.objects.filter(id__in=document_ids).count(),
        TextChunk.objects.filter(doc_id__in=document_ids).count(),
        CollectionPermission.objects.filter(collection_id__in=collection_ids).count(),
    )


def database_counts() -> dict[str, int]:
    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument, TextChunk

    return {
        "collections": Collection.objects.count(),
        "documents": RawTextDocument.objects.count(),
        "chunks": TextChunk.objects.count(),
        "permissions": CollectionPermission.objects.count(),
        "users": User.objects.count(),
        "groups": Group.objects.count(),
    }


def assert_no_fixture_graph_rows() -> None:
    from apps.knowledge_graph.models import (
        CanonicalEntity,
        CanonicalEntityLink,
        CollectionArtifactInput,
        CollectionEntity,
        CollectionEntityDocumentLink,
        CollectionRelation,
        CollectionRelationEvidence,
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
        RelationMention,
    )

    forbidden = (
        GraphRebuildRequest,
        GraphArtifact,
        GraphBuildRun,
        EntityMention,
        DocumentEntity,
        DocumentEntityMention,
        CanonicalEntity,
        CollectionEntity,
        CanonicalEntityLink,
        CollectionEntityDocumentLink,
        CollectionArtifactInput,
        RelationMention,
        CollectionRelation,
        CollectionRelationEvidence,
    )
    assert all(not model.objects.exists() for model in forbidden)


def create_eval_request(payload, *, status, error_code=""):
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import GraphRebuildRequest

    scope = payload["authorized_scope"][0]
    documents = list(
        RawTextDocument.objects.filter(collection_id=scope["collection_id"]).order_by(
            "id"
        )
    )
    snapshots = sorted(
        (
            {
                "document_id": str(document.id),
                "document_pkid": document.pkid,
                "model_label": RawTextDocument._meta.label_lower,
                "collection_id": document.collection_id,
                "source_hash": document.full_text_hash,
            }
            for document in documents
        ),
        key=lambda row: (row["model_label"], row["document_id"], row["document_pkid"]),
    )
    terminal = status not in {
        GraphRebuildRequest.Status.QUEUED,
        GraphRebuildRequest.Status.RUNNING,
    }
    request = GraphRebuildRequest.objects.create(
        id=UUID(scope["rebuild_request_id"]),
        scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
        scope_id=str(scope["collection_id"]),
        requested_documents=snapshots,
        expected_aggregate_signature="d" * 64,
        status=status,
        evaluation_only=True,
        document_count=len(snapshots),
        collection_count=1,
        started_at=timezone.now(),
        completed_at=timezone.now() if terminal else None,
        terminal_failure_count=len(snapshots) if terminal else 0,
        failed_collection_count=1 if terminal else 0,
        document_publication_state=(
            GraphRebuildRequest.PublicationState.FAILED
            if terminal
            else GraphRebuildRequest.PublicationState.PENDING
        ),
        collection_publication_state=(
            GraphRebuildRequest.PublicationState.FAILED
            if terminal
            else GraphRebuildRequest.PublicationState.NOT_APPLICABLE
        ),
        error_code=error_code or ("task21_test_failure" if terminal else ""),
    )
    return request, documents


def create_document_eval_artifact(request, document, *, status):
    from apps.knowledge_graph.models import GraphArtifact

    terminal = status != GraphArtifact.Status.BUILDING
    return GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=str(document.id),
        status=status,
        source_hash=document.full_text_hash,
        ontology_version="task21-test-ontology",
        extractor_version="task21-test-extractor",
        resolver_version="task21-test-resolver",
        filter_policy_version="task21-test-filter",
        rebuild_request=request,
        evaluation_only=True,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        completed_at=timezone.now() if terminal else None,
    )
