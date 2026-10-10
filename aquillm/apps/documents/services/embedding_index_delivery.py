"""Durable at-most-three-delivery leases and failure fences."""

from datetime import timedelta
from uuid import uuid4
from django.db import transaction
from django.utils import timezone
from lib.embeddings.document_contracts import EmbeddingLeaseV1
from apps.documents.models.embedding_index import EmbeddingGeneration


def claim_embedding_delivery(generation_key, *, using):
    with transaction.atomic(using=using):
        generation = (
            EmbeddingGeneration.objects.using(using)
            .select_for_update()
            .get(pk=generation_key)
        )
        now = timezone.now()
        if (
            generation.state != "building"
            or generation.failure_kind
            or generation.deliveries >= 3
            or (generation.next_attempt_at and generation.next_attempt_at > now)
        ):
            return None
        generation.deliveries += 1
        generation.lease_token = uuid4()
        generation.next_attempt_at = now + timedelta(seconds=120)
        generation.save(
            using=using, update_fields=["deliveries", "lease_token", "next_attempt_at"]
        )
        return EmbeddingLeaseV1(generation.key, generation.lease_token)


def fail_embedding_delivery(lease, *, kind, using):
    if kind not in (
        "upstream",
        "contract",
        "unexpected",
        "deadline",
        "source_scope_unavailable",
    ):
        raise ValueError("invalid_failure_kind")
    with transaction.atomic(using=using):
        generation = (
            EmbeddingGeneration.objects.using(using)
            .select_for_update()
            .filter(
                pk=lease.generation_key, lease_token=lease.lease_token, state="building"
            )
            .first()
        )
        if generation is None:
            return False
        terminal = kind != "upstream" or generation.deliveries >= 3
        generation.failure_kind = kind if terminal else ""
        generation.state = "failed" if terminal else "building"
        generation.next_attempt_at = timezone.now() + timedelta(
            seconds=30 * 2 ** max(0, generation.deliveries - 1)
        )
        generation.save(
            using=using, update_fields=["failure_kind", "state", "next_attempt_at"]
        )
        return not terminal
