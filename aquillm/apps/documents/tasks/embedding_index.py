"""Vector-only workers. Celery carries relative timeouts and durable keys only."""

from uuid import UUID

from celery import shared_task
from django.db import transaction

from apps.documents.models import TextChunk
from apps.documents.models.embedding_index import EmbeddingGeneration
from apps.documents.services.embedding_index_store import (
    activate_embedding_generation,
    begin_embedding_generation,
    capture_embedding_manifest,
    claim_embedding_delivery,
    embedding_manifest_digest,
    fail_embedding_delivery,
    load_embedding_space,
    source_chunk,
    stage_embedding_batch,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.embeddings.document_contracts import build_budget
from lib.embeddings.document_provider import embed_document_batch, prepare_document
from lib.embeddings.errors import EmbeddingUpstreamUnavailableError
from lib.embeddings.utils import EmbeddingContractError
from apps.documents.services.embedding_index_deadline import embedding_build_transaction


def rebuild_document_embeddings(selection, *, space_key, using, timeout_ms=120000):
    budget = build_budget(timeout_ms)
    space = load_embedding_space(space_key, using=using)
    manifest = capture_embedding_manifest(selection, using=using, space=space)
    if budget.remaining_seconds() <= 0:
        raise TimeoutError("embedding_deadline")
    lease = begin_embedding_generation(
        selection,
        space_key=space_key,
        manifest_sha256=embedding_manifest_digest(
            selection, manifest, space_key=space_key
        ),
        expected_count=len(manifest),
        using=using,
    )
    transaction.on_commit(
        lambda: build_embedding_generation.delay(
            str(lease.generation_key), using=using, timeout_ms=timeout_ms
        ),
        using=using,
    )
    return str(lease.generation_key)


@shared_task(bind=True, max_retries=2, name="documents.build_embedding_generation")
def build_embedding_generation(
    self, generation_key, *, using="default", timeout_ms=120000
):
    budget = build_budget(timeout_ms)
    lease = claim_embedding_delivery(UUID(generation_key), using=using)
    if lease is None:
        return "not_claimed"
    try:
        generation = (
            EmbeddingGeneration.objects.using(using)
            .select_related("owner")
            .get(pk=lease.generation_key)
        )
        space = load_embedding_space(generation.owner.space_id, using=using)
        # Exact successful rows survive retry; no cross-document reuse.
        completed = set(generation.embeddings.values_list("chunk_id", flat=True))
        batch, ids, hashes, size = [], [], [], 0

        def flush():
            if not batch:
                return
            results = embed_document_batch(tuple(batch), space=space, budget=budget)
            if budget.remaining_seconds() <= 0:
                raise TimeoutError("embedding_deadline")
            with embedding_build_transaction(budget, using=using):
                stage_embedding_batch(
                    lease,
                    chunk_ids=tuple(ids),
                    results=results,
                    input_sha256=tuple(hashes),
                    using=using,
                )

        for item in generation.manifest:
            if budget.remaining_seconds() <= 0:
                raise TimeoutError("embedding_deadline")
            if item["chunk_id"] in completed:
                continue
            row = TextChunk.objects.using(using).get(pk=item["chunk_id"])
            chunk = source_chunk(row)
            wire, sha = prepare_document(chunk, space=space)
            if sha != item["prepared_input_sha256"]:
                raise EmbeddingContractError("embedding_input_mismatch")
            amount = len(
                (wire if isinstance(wire, str) else wire["text"]).encode("utf-8")
            )
            if batch and (
                len(batch) == budget.max_items
                or size + amount > min(budget.max_input_bytes, space.max_batch_tokens)
            ):
                flush()
                batch, ids, hashes, size = [], [], [], 0
            if amount > min(budget.max_input_bytes, space.max_input_tokens):
                raise EmbeddingContractError("prepared_input_token_limit")
            batch.append(chunk)
            ids.append(row.pk)
            hashes.append(sha)
            size += amount
        flush()
        if budget.remaining_seconds() <= 0:
            raise TimeoutError("embedding_deadline")
        with embedding_build_transaction(budget, using=using):
            return activate_embedding_generation(lease, using=using)
    except EmbeddingUpstreamUnavailableError:
        retry = fail_embedding_delivery(lease, kind="upstream", using=using)
        if retry:
            raise self.retry(countdown=30 * 2**self.request.retries)
        return "failed"
    except Exception as exc:
        kind = (
            "contract"
            if isinstance(exc, EmbeddingContractError)
            else (
                "deadline"
                if isinstance(exc, TimeoutError)
                else (
                    "source_scope_unavailable"
                    if isinstance(exc, SourceScopeUnavailable)
                    else "unexpected"
                )
            )
        )
        fail_embedding_delivery(lease, kind=kind, using=using)
        raise
