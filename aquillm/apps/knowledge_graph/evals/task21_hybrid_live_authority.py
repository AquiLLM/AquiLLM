"""Exact synthetic-fixture authority for the live Task21 cloud run."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .fixture_manifest import (
    ResolvedFixtureManifest,
    canonical_embedding_sha256,
    load_fixture_manifest,
)
from .fixture_seed_cases import logical_fixture
from .fixture_seed_contract import PHYSICAL_BINDINGS, VISIBLE_USERNAME
from .fixture_seed_manifest_io import canonical_manifest_bytes, validate_payload


@dataclass(frozen=True, slots=True)
class PreparedLiveCase:
    case: dict[str, object]
    authorization: object
    selected_documents: tuple[object, ...]
    accessible_chunk_symbols: frozenset[str]
    adversarial_chunk_symbols: tuple[str, ...]
    chunk_symbols_by_pk: dict[int, str]
    snapshot: object | None = None


@dataclass(frozen=True, slots=True)
class LiveFixtureAuthority:
    manifest: ResolvedFixtureManifest
    principal: object
    documents_by_id: dict[object, object]
    chunk_symbols_by_pk: dict[int, str]

    def prepare_case(self, case: dict[str, object]) -> PreparedLiveCase | None:
        from apps.collections.services.django_retrieval_authorization import (
            build_production_retrieval_authorization_context,
        )

        accessible_logical = frozenset(case["accessible_collection_ids"])
        accessible_case_chunks = frozenset(
            chunk["chunk_id"]
            for document in case["documents"]
            if document["collection_id"] in accessible_logical
            for chunk in document["chunks"]
        )
        adversarial = tuple(
            sorted(
                chunk["chunk_id"]
                for document in case["documents"]
                if document["collection_id"] not in accessible_logical
                for chunk in document["chunks"]
            )
        )
        case_documents = tuple(
            document
            for document in case["documents"]
            if document["collection_id"] in accessible_logical
        )
        if not case_documents:
            return None
        selected_collection_ids = tuple(
            sorted(
                {
                    self.manifest.collections[logical].collection_id
                    for logical in accessible_logical
                }
            )
        )
        selected_documents = tuple(
            sorted(
                (
                    self.documents_by_id[binding.document_id]
                    for binding in self.manifest.documents.values()
                    if binding.collection_id in selected_collection_ids
                ),
                key=lambda row: row.id.int,
            )
        )
        authorization = build_production_retrieval_authorization_context(
            principal=self.principal,
            selected_collection_ids=selected_collection_ids,
            selected_documents=list(selected_documents),
        )
        if authorization is None:
            raise RuntimeError("live fixture authorization could not be frozen")
        return PreparedLiveCase(
            case,
            authorization,
            selected_documents,
            accessible_case_chunks,
            adversarial,
            self.chunk_symbols_by_pk,
        )


def _validate_runtime_flags() -> None:
    from django.conf import settings

    from apps.documents.services.hybrid_graph_dependencies import (
        django_hybrid_retrieval_settings,
    )

    selected = django_hybrid_retrieval_settings()
    enabled = (
        getattr(settings, "KG_OVERLAY_ENABLED", False),
        selected.memgraph_traversal_enabled,
        selected.graph_direct_enabled,
        selected.graph_extended_enabled,
    )
    if enabled != (True, True, True, True):
        raise RuntimeError("live hybrid observation flags are not exactly enabled")


def _load_exact_manifest(path: Path) -> ResolvedFixtureManifest:
    payload = load_fixture_manifest(path)
    if path.read_bytes() != canonical_manifest_bytes(payload):
        raise RuntimeError("fixture manifest bytes are not canonical")
    return validate_payload(payload, logical_fixture())


def validate_live_fixture_rows(*, manifest, documents, chunks) -> None:
    """Bind every live document/chunk byte to the canonical signed manifest."""

    documents_by_id = {row.id: row for row in documents}
    if set(documents_by_id) != {
        binding.document_id for binding in manifest.documents.values()
    }:
        raise RuntimeError("fixture documents are incomplete")
    for binding in manifest.documents.values():
        row = documents_by_id[binding.document_id]
        digest = sha256(row.full_text.encode()).hexdigest()
        if (
            row.collection_id != binding.collection_id
            or digest != binding.full_text_sha256
            or row.full_text_hash != binding.full_text_sha256
        ):
            raise RuntimeError("fixture document content drifted")
    chunks_by_pk = {row.pk: row for row in chunks}
    if set(chunks_by_pk) != {
        binding.chunk_id for binding in manifest.chunks.values()
    }:
        raise RuntimeError("fixture chunks are incomplete")
    for symbol, binding in manifest.chunks.items():
        row = chunks_by_pk[binding.chunk_id]
        document_id = manifest.documents[binding.document_symbol].document_id
        if (row.doc_id, row.chunk_number, row.start_position, row.end_position) != (
            document_id,
            binding.chunk_number,
            binding.start,
            binding.end,
        ):
            raise RuntimeError("fixture chunk coordinates drifted")
        if sha256(row.content.encode()).hexdigest() != binding.content_sha256:
            raise RuntimeError("fixture chunk content drifted")
        if (
            row.modality != "text"
            or row.metadata
            != {"chunk_symbol": symbol, "fixture_id": manifest.fixture_id}
        ):
            raise RuntimeError("fixture chunk metadata drifted")
        try:
            embedding_checksum = canonical_embedding_sha256(tuple(row.embedding))
        except (TypeError, ValueError) as error:
            raise RuntimeError("fixture chunk embedding drifted") from error
        if embedding_checksum != binding.embedding_sha256:
            raise RuntimeError("fixture chunk embedding drifted")


def _validate_embedding_identity(manifest: ResolvedFixtureManifest) -> None:
    from aquillm.utils import strict_index_embedding_signature

    signature = strict_index_embedding_signature()
    expected_prefix = (
        f"local-openai:{manifest.embedding.model}@{manifest.embedding.checkpoint}:"
    )
    if (
        not signature.startswith(expected_prefix)
        or f":dims={manifest.embedding.dimensions}:" not in signature
    ):
        raise RuntimeError("fixture embedding runtime identity drifted")


def load_live_fixture_authority(path: Path) -> LiveFixtureAuthority:
    """Validate the checked-in fixture's exact live rows and policy principal."""

    from django.contrib.auth.models import User

    from apps.documents.models.chunks import TextChunk
    from apps.documents.models.document_types.raw_text import RawTextDocument

    _validate_runtime_flags()
    manifest = _load_exact_manifest(path)
    _validate_embedding_identity(manifest)
    principal = User.objects.get(username=VISIBLE_USERNAME)
    if principal.is_active is not False or principal.is_authenticated is not True:
        raise RuntimeError("fixture principal identity drifted")
    document_ids = tuple(row.document_id for row in manifest.documents.values())
    documents = tuple(RawTextDocument.objects.filter(id__in=document_ids))
    documents_by_id = {row.id: row for row in documents}
    chunk_ids = tuple(row.chunk_id for row in manifest.chunks.values())
    chunks = tuple(TextChunk.objects.filter(pk__in=chunk_ids))
    validate_live_fixture_rows(
        manifest=manifest, documents=documents, chunks=chunks
    )
    expected_labels = {
        binding.collection_id
        for logical, binding in manifest.collections.items()
        if PHYSICAL_BINDINGS[logical] != "hidden"
    }
    if expected_labels != {row[0] for row in manifest.authorized_scope}:
        raise RuntimeError("fixture authorized scope drifted")
    return LiveFixtureAuthority(
        manifest,
        principal,
        documents_by_id,
        {binding.chunk_id: symbol for symbol, binding in manifest.chunks.items()},
    )


__all__ = [
    "LiveFixtureAuthority",
    "PreparedLiveCase",
    "canonical_embedding_sha256",
    "load_live_fixture_authority",
    "validate_live_fixture_rows",
]
