"""Vector-only workers. Celery carries relative timeouts and durable keys only."""

from uuid import UUID

from celery import shared_task
from django.db import transaction, DatabaseError

from apps.documents.models import TextChunk
from apps.documents.models.embedding_index import EmbeddingGeneration
from apps.documents.services.embedding_index_store import (
    activate_embedding_generation,
    begin_embedding_generation,
    capture_embedding_manifest,
    claim_embedding_delivery,
    embedding_manifest_digest,
    fail_embedding_delivery,
    handoff_embedding_delivery,
    load_embedding_space,
    source_chunk,
    stage_embedding_batch,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.embeddings.document_contracts import build_budget
from lib.embeddings.document_provider import embed_document_batch, prepare_document
from lib.embeddings.document_preparation import prepared_input_bytes
from lib.embeddings.errors import EmbeddingUpstreamUnavailableError
from lib.embeddings.utils import EmbeddingContractError
from apps.documents.services.embedding_index_deadline import (
    embedding_build_transaction,
    embedding_build_database,
    check_embedding_deadline,
)

# Kept back after the last batch for activation and lease bookkeeping.
HANDOFF_RESERVE_SECONDS = 15


class EmbeddingHandoff(Exception):
    """This delivery staged progress but cannot fit its next step before the deadline."""


def rebuild_document_embeddings(selection, *, space_key, using, timeout_ms=120000):
    budget = build_budget(timeout_ms)
    with embedding_build_database(budget, using=using):
        space = load_embedding_space(space_key, using=using)
        manifest = capture_embedding_manifest(selection, using=using, space=space)
        lease = begin_embedding_generation(
            selection,
            space_key=space_key,
            manifest_sha256=embedding_manifest_digest(
                selection, manifest, space_key=space_key
            ),
            expected_count=len(manifest),
            using=using,
        )

    def enqueue():
        check_embedding_deadline(budget)
        build_embedding_generation.delay(
            str(lease.generation_key), using=using, timeout_ms=timeout_ms
        )

    check_embedding_deadline(budget)
    transaction.on_commit(enqueue, using=using)
    return str(lease.generation_key)


@shared_task(bind=True, max_retries=2, name="documents.build_embedding_generation")
def build_embedding_generation(
    self, generation_key, *, using="default", timeout_ms=120000
):
    budget = build_budget(timeout_ms)
    lease = None
    try:
        with embedding_build_database(budget, using=using):
            lease = claim_embedding_delivery(UUID(generation_key), using=using)
            if lease is None:
                return "not_claimed"
            return _build(lease, budget=budget, using=using)
    except EmbeddingHandoff:
        # Staged rows survive; continue in a fresh delivery instead of lapsing the lease.
        with embedding_build_transaction(budget, using=using):
            handed_off = handoff_embedding_delivery(lease, using=using)
        if handed_off:
            build_embedding_generation.delay(generation_key, using=using, timeout_ms=timeout_ms)
            return "handed_off"
        return "lease_lost"
    except EmbeddingUpstreamUnavailableError:
        check_embedding_deadline(budget)
        with embedding_build_transaction(budget, using=using):
            retry = fail_embedding_delivery(lease, kind="upstream", using=using)
            if retry:
                deliveries = (
                    EmbeddingGeneration.objects.using(using)
                    .values_list("deliveries", flat=True)
                    .get(pk=lease.generation_key)
                )
                # Fresh explicit-resume messages reset Celery's retry counter,
                # but must retain the persisted generation's 30/60-second fence.
                retry_delay = 30 * 2 ** max(0, deliveries - 1)
        if retry:
            check_embedding_deadline(budget)
            raise self.retry(countdown=retry_delay)
        return "failed"
    except Exception as exc:
        # Expiry never acquires a new bookkeeping allowance or enqueues recovery.
        # A later explicit rebuild can reclaim the durable lease after its due time.
        if lease is not None and budget.remaining_seconds() >= 0.001:
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
            try:
                with embedding_build_transaction(budget, using=using):
                    fail_embedding_delivery(lease, kind=kind, using=using)
            except (TimeoutError, DatabaseError):
                pass  # Preserve the original failure; the durable lease remains.
        raise


def _build(lease, *, budget, using):
    generation = (
        EmbeddingGeneration.objects.using(using)
        .select_related("owner")
        .get(pk=lease.generation_key)
    )
    space = load_embedding_space(generation.owner.space_id, using=using)
    check_embedding_deadline(budget)
    # Exact successful rows survive retry; no cross-document reuse.
    completed = set(generation.embeddings.values_list("chunk_id", flat=True))
    batch, ids, hashes, wire_size, text_size = [], [], [], 0, 0
    staged, slowest = 0, 0.0

    def flush():
        nonlocal staged, slowest
        if not batch:
            return
        # Hand off only after progress, so a delivery that stages nothing still counts.
        if staged and budget.remaining_seconds() < slowest * 1.5 + HANDOFF_RESERVE_SECONDS:
            raise EmbeddingHandoff()
        started = budget.clock()
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
        staged += len(ids)
        slowest = max(slowest, budget.clock() - started)

    pending = [item["chunk_id"] for item in generation.manifest if item["chunk_id"] not in completed]
    rows, verified = TextChunk.objects.using(using).in_bulk(pending), {}
    for item in generation.manifest:
        if budget.remaining_seconds() <= 0:
            raise TimeoutError("embedding_deadline")
        if item["chunk_id"] in completed:
            continue
        row = rows.get(item["chunk_id"])
        if row is None:
            raise TextChunk.DoesNotExist(item["chunk_id"])
        check_embedding_deadline(budget)
        chunk = source_chunk(row, verified)
        wire, sha = prepare_document(chunk, space=space)
        check_embedding_deadline(budget)
        if sha != item["prepared_input_sha256"]:
            raise EmbeddingContractError("embedding_input_mismatch")
        wire_amount = prepared_input_bytes(wire)
        text_amount = len((wire if isinstance(wire, str) else wire["text"]).encode("utf-8"))
        if wire_amount > budget.max_input_bytes:
            raise EmbeddingContractError("document_batch_bytes")
        if text_amount > space.max_input_tokens:
            raise EmbeddingContractError("prepared_input_token_limit")
        if batch and (
            len(batch) == budget.max_items
            or wire_size + wire_amount > budget.max_input_bytes
            or text_size + text_amount > space.max_batch_tokens
        ):
            flush()
            batch, ids, hashes, wire_size, text_size = [], [], [], 0, 0
        batch.append(chunk)
        ids.append(row.pk)
        hashes.append(sha)
        wire_size += wire_amount
        text_size += text_amount
    flush()
    if staged and budget.remaining_seconds() < HANDOFF_RESERVE_SECONDS:
        raise EmbeddingHandoff()
    if budget.remaining_seconds() <= 0:
        raise TimeoutError("embedding_deadline")
    with embedding_build_transaction(budget, using=using):
        return activate_embedding_generation(lease, using=using)
