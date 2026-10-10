"""Capture/compute/compare-and-publish with no in-call retry or deadline reset."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from django.db import DatabaseError, connections

from . import canonical as c
from .canonical_deadline_v2 import (
    ReconciliationDeadlineExceeded,
    bounded_transaction,
    check_deadline,
)
from .canonical_snapshot_v2 import capture_source, lock_source


def configured_reconciliation(*, using):
    import os
    from time import monotonic

    from lib.knowledge_graph.retrieval_config import load_hybrid_retrieval_settings

    settings = load_hybrid_retrieval_settings(
        {
            key: os.environ[key]
            for key in ("KG_GRAPH_REMEDIATION_MODE", "KG_GRAPH_OVERALL_TIMEOUT_MS")
            if key in os.environ
        }
    )
    if settings.graph_remediation_mode != "v2":
        return None
    outcome = reconcile_canonical_v2(
        using=using,
        clock=monotonic,
        deadline=monotonic() + settings.graph_overall_timeout_ms / 1000,
    )
    if outcome.status != "published":
        raise RuntimeError(f"bounded canonical reconciliation {outcome.status}")
    return outcome.result


@dataclass(frozen=True, slots=True)
class CanonicalReconciliationOutcomeV2:
    status: Literal["published", "stale", "timed_out"]
    result: c.CanonicalRebuildResult | None = None

    def __post_init__(self):
        if self.status not in ("published", "stale", "timed_out"):
            raise ValueError("invalid reconciliation status")
        if (self.status == "published") != (
            type(self.result) is c.CanonicalRebuildResult
        ):
            raise ValueError("only successful publication carries a registry result")


def reconcile_canonical_v2(
    *, using: str, deadline: float, clock: Callable[[], float]
) -> CanonicalReconciliationOutcomeV2:
    from lib.knowledge_graph.config import load_identity_policy

    if connections[using].in_atomic_block:
        raise RuntimeError("V2 reconciliation must compute outside caller transactions")
    try:
        with bounded_transaction(using=using, deadline=deadline, clock=clock):
            snapshot = capture_source(
                using=using, identity_policy=load_identity_policy()
            )
        # CPU-only resolution and vector candidate derivation never hold SQL locks.
        check_deadline(deadline, clock)
        resolution = c.resolve_canonical_entities(
            snapshot.inputs,
            identity_policy=snapshot.identity_policy,
            embedding_candidates=c._derive_locked_embedding_candidates(
                snapshot.entities,
                artifact_embedding_signatures=dict(snapshot.artifact_signatures),
            ),
        )
        projected_links = c.project_canonical_link_decisions(resolution)
        check_deadline(deadline, clock)
        with bounded_transaction(using=using, deadline=deadline, clock=clock):
            lock_source(snapshot, using=using)
            current = capture_source(
                using=using, identity_policy=snapshot.identity_policy
            )
            if current.revision_digest != snapshot.revision_digest:
                return CanonicalReconciliationOutcomeV2("stale")
            c._lock_canonical_registry(using=using)
            from apps.knowledge_graph.models import CollectionEntity

            source_rows = CollectionEntity.objects.using(using).in_bulk(
                e.pk for e in current.entities
            )
            result = c._reconcile_locked_registry(
                resolution=resolution,
                active_artifact_ids=tuple(a for a, _ in snapshot.artifacts),
                source_entities_by_id=source_rows,
                using=using,
                projected_links=projected_links,
                deadline_check=lambda: check_deadline(deadline, clock),
            )
            check_deadline(deadline, clock)
        return CanonicalReconciliationOutcomeV2("published", result)
    except c._CanonicalSnapshotChanged:
        return CanonicalReconciliationOutcomeV2("stale")
    except ReconciliationDeadlineExceeded:
        return CanonicalReconciliationOutcomeV2("timed_out")
    except DatabaseError as exc:
        code = getattr(exc.__cause__, "sqlstate", None) or getattr(
            exc.__cause__, "pgcode", None
        )
        if code not in ("55P03", "57014"):
            raise
        return CanonicalReconciliationOutcomeV2("timed_out")
