"""Durable at-most-three-delivery leases and failure fences."""

from datetime import timedelta
from uuid import uuid4
from django.db import transaction
from django.db.models import Q
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
            or (generation.next_attempt_at and generation.next_attempt_at > now)
        ):
            return None
        if generation.deliveries >= 3:
            # A later bounded explicit recovery can close an abandoned last lease.
            generation.state = "failed"
            generation.failure_kind = "delivery_exhausted"
            generation.save(using=using, update_fields=["state", "failure_kind"])
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


def handoff_embedding_delivery(lease, *, using):
    """Release a lease that staged progress; the continuation does not spend a delivery."""
    with transaction.atomic(using=using):
        generation = (
            EmbeddingGeneration.objects.using(using)
            .select_for_update()
            .filter(
                pk=lease.generation_key,
                lease_token=lease.lease_token,
                state="building",
                failure_kind="",
            )
            .first()
        )
        if generation is None:
            return False
        generation.deliveries = max(0, generation.deliveries - 1)
        generation.next_attempt_at = timezone.now()
        generation.save(using=using, update_fields=["deliveries", "next_attempt_at"])
        return True


RECOVERY_QUIET_PERIOD = timedelta(minutes=5)


def recover_due_embedding_generations(*, limit=3, using="default"):
    """Redeliver building generations whose lease lapsed or whose first dispatch was lost."""
    from apps.documents.tasks.embedding_index import build_embedding_generation

    limit = min(25, max(1, int(limit)))
    now = timezone.now()
    quiet = now - RECOVERY_QUIET_PERIOD
    lapsed = Q(state="building", failure_kind="") & (
        Q(next_attempt_at__lte=quiet) | Q(next_attempt_at__isnull=True, created_at__lte=quiet)
    )
    generations = EmbeddingGeneration.objects.using(using)
    keys = list(generations.filter(lapsed).order_by("created_at", "pk").values_list("pk", flat=True)[:limit])
    dispatched = 0
    for key in keys:
        # Stamping the dispatch keeps a queued redelivery from being sent again for the
        # quiet period; the claim still accepts it because the stamp is not in the future.
        if generations.filter(lapsed, pk=key).update(next_attempt_at=now):
            # Claiming re-checks the fence; exhausted generations close as delivery_exhausted.
            build_embedding_generation.delay(str(key), using=using)
            dispatched += 1
    return dispatched
