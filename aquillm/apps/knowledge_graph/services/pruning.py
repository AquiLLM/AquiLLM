"""Conservative, bounded retention for terminal knowledge-graph state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from django.conf import settings
from django.db import transaction
from django.db.models import (
    Case,
    Count,
    DateTimeField,
    Exists,
    F,
    IntegerField,
    OuterRef,
    Q,
    Subquery,
    Value,
    When,
)
from django.db.models.functions import Coalesce, Greatest
from django.utils import timezone

from lib.knowledge_graph.config import (
    DEFAULT_ARTIFACT_KEEP_SUPERSEDED,
    DEFAULT_ARTIFACT_RETENTION_DAYS,
)

_MAX_PRUNE_REPORT_IDS = 10_000
_MAX_QUERY_PREDICATE_IDS = 5_000
_DEFAULT_BATCH_SIZE = 100
_MAX_BATCH_SIZE = 1_000


def _bounded_ids(values: object, field: str) -> tuple[int, ...]:
    if type(values) is not tuple:
        raise ValueError(f"{field} must be an exact tuple")
    if len(values) > _MAX_PRUNE_REPORT_IDS:
        raise ValueError("pruning report must remain bounded")
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{field} must contain positive integer ids")
    result = tuple(sorted(set(values)))
    if len(result) != len(values):
        raise ValueError(f"{field} must not contain duplicate ids")
    return result


@dataclass(frozen=True, slots=True)
class PruneReport:
    """Privacy-safe pruning result containing only scalar counts and row IDs."""

    dry_run: bool
    artifact_ids: tuple[int, ...]
    run_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.dry_run) is not bool:
            raise ValueError("dry_run must be a boolean")
        object.__setattr__(
            self, "artifact_ids", _bounded_ids(self.artifact_ids, "artifact_ids")
        )
        object.__setattr__(self, "run_ids", _bounded_ids(self.run_ids, "run_ids"))

    @property
    def artifact_count(self) -> int:
        return len(self.artifact_ids)

    @property
    def run_count(self) -> int:
        return len(self.run_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "dry_run": self.dry_run,
            "artifact_count": self.artifact_count,
            "run_count": self.run_count,
            "artifact_ids": self.artifact_ids,
            "run_ids": self.run_ids,
        }


def _candidate_row_id(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _candidate_time(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or timezone.is_naive(value):
        raise ValueError(f"{field} must be an aware datetime")
    return value


@dataclass(frozen=True, slots=True)
class _ArtifactCandidate:
    row_id: int
    prune_before: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "row_id", _candidate_row_id(self.row_id, "artifact candidate id")
        )
        object.__setattr__(
            self,
            "prune_before",
            _candidate_time(self.prune_before, "artifact terminal time"),
        )


@dataclass(frozen=True, slots=True)
class _RunCandidate:
    row_id: int
    prune_before: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "row_id", _candidate_row_id(self.row_id, "run candidate id")
        )
        object.__setattr__(
            self,
            "prune_before",
            _candidate_time(self.prune_before, "run terminal time"),
        )


@dataclass(frozen=True, slots=True)
class _PrunePlan:
    artifact_ids: tuple[int, ...]
    run_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_ids", _bounded_ids(self.artifact_ids, "artifact_ids")
        )
        object.__setattr__(self, "run_ids", _bounded_ids(self.run_ids, "run_ids"))
        if len(self.artifact_ids) + len(self.run_ids) > _MAX_BATCH_SIZE:
            raise ValueError("pruning plan exceeds the total row budget")


def _ordered_candidates(values, expected_type, field: str):
    rows = tuple(values)
    if len(rows) > _MAX_BATCH_SIZE + 1:
        raise ValueError(f"{field} candidate input exceeds its cap")
    if any(type(row) is not expected_type for row in rows):
        raise ValueError(f"{field} candidates must use exact audited rows")
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError(f"{field} candidates must not contain duplicate ids")
    return tuple(sorted(rows, key=lambda row: (row.prune_before, row.row_id)))


def _plan_pruning_candidates(
    *,
    artifact_candidates: tuple[_ArtifactCandidate, ...],
    run_candidates: tuple[_RunCandidate, ...],
    batch_size: int,
) -> _PrunePlan:
    """Create one deterministic plan under one total database-row budget."""

    if type(batch_size) is not int or not 1 <= batch_size <= _MAX_BATCH_SIZE:
        raise ValueError("batch_size must be an integer in [1, 1000]")
    artifacts = _ordered_candidates(artifact_candidates, _ArtifactCandidate, "artifact")
    runs = _ordered_candidates(run_candidates, _RunCandidate, "run")
    selected_artifacts = artifacts[:batch_size]
    remaining = batch_size - len(selected_artifacts)
    selected_runs = runs[:remaining]
    return _PrunePlan(
        artifact_ids=tuple(row.row_id for row in selected_artifacts),
        run_ids=tuple(row.row_id for row in selected_runs),
    )


def _positive_setting(name: str, default: int) -> int:
    value = getattr(settings, name, default)
    return value if type(value) is int and value > 0 else default


def _nonnegative_setting(name: str, default: int) -> int:
    value = getattr(settings, name, default)
    return value if type(value) is int and value >= 0 else default


def pruning_boundary(*, now: datetime | None = None) -> datetime:
    resolved_now = timezone.now() if now is None else now
    if not isinstance(resolved_now, datetime) or timezone.is_naive(resolved_now):
        raise ValueError("pruning time must be an aware datetime")
    retention_days = _positive_setting(
        "KG_ARTIFACT_RETENTION_DAYS", DEFAULT_ARTIFACT_RETENTION_DAYS
    )
    return resolved_now - timedelta(days=retention_days)


def _nonterminal_request_filter(prefix: str = "") -> Q:
    from apps.knowledge_graph.models import GraphRebuildRequest

    statuses = (
        GraphRebuildRequest.Status.QUEUED,
        GraphRebuildRequest.Status.RUNNING,
    )
    return Q(**{f"{prefix}status__in": statuses}) | Q(
        **{f"{prefix}parent_request__status__in": statuses}
    )


def _terminal_run_identity_filter() -> Q:
    """Require a run to be the exact durable occurrence of its artifact."""

    identity = Q(
        build_kind=F("artifact__scope_type"),
        scope_type=F("artifact__scope_type"),
        scope_id=F("artifact__scope_id"),
        evaluation_only=F("artifact__evaluation_only"),
        build_key=F("artifact__build_key"),
        build_generation=F("artifact__build_generation"),
        orchestration_version=F("artifact__orchestration_version"),
        source_hash=F("artifact__source_hash"),
        ontology_version=F("artifact__ontology_version"),
        extractor_version=F("artifact__extractor_version"),
        resolver_version=F("artifact__resolver_version"),
        filter_policy_version=F("artifact__filter_policy_version"),
        embedding_model_signature=F("artifact__embedding_model_signature"),
        ontology_checksum=F("artifact__ontology_checksum"),
        filter_policy_checksum=F("artifact__filter_policy_checksum"),
        resolution_config_checksum=F("artifact__resolution_config_checksum"),
        assembly_version=F("artifact__assembly_version"),
        assembly_config_checksum=F("artifact__assembly_config_checksum"),
    )
    request = Q(
        artifact__rebuild_request_id__isnull=True,
        rebuild_request_id__isnull=True,
    ) | Q(rebuild_request_id=F("artifact__rebuild_request_id"))
    return identity & request


def _terminal_run_stage_filter() -> Q:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    return (
        Q(
            artifact__status=GraphArtifact.Status.FAILED,
            stage=GraphBuildRun.Stage.FAILED,
            status=GraphBuildRun.Status.FAILED,
        )
        | Q(
            artifact__status=GraphArtifact.Status.STALE,
            stage=GraphBuildRun.Stage.STALE,
            status=GraphBuildRun.Status.CANCELLED,
        )
        | Q(
            artifact__status=GraphArtifact.Status.SUPERSEDED,
            stage=GraphBuildRun.Stage.SUPERSEDED,
            status=GraphBuildRun.Status.CANCELLED,
        )
    )


def _artifact_terminal_run_query():
    from apps.knowledge_graph.models import GraphBuildRun

    return (
        GraphBuildRun.objects.filter(
            artifact_id=OuterRef("pk"),
            finished_at__isnull=False,
            lease_owner="",
            lease_expires_at__isnull=True,
        )
        .filter(_terminal_run_identity_filter())
        .filter(_terminal_run_stage_filter())
        .order_by("-finished_at", "-pk")
    )


def _newer_superseded_count_query():
    from apps.knowledge_graph.models import GraphArtifact

    return (
        GraphArtifact.objects.filter(
            scope_type=OuterRef("scope_type"),
            scope_id=OuterRef("scope_id"),
            status=GraphArtifact.Status.SUPERSEDED,
            evaluation_only=OuterRef("evaluation_only"),
        )
        .filter(
            Q(build_generation__gt=OuterRef("build_generation"))
            | Q(
                build_generation=OuterRef("build_generation"),
                pk__gt=OuterRef("pk"),
            )
        )
        .order_by()
        .values("scope_type")
        .annotate(total=Count("pk"))
        .values("total")
    )


def _candidate_artifact_queryset(*, boundary: datetime):
    from apps.knowledge_graph.models import (
        CanonicalEntityLink,
        CollectionArtifactInput,
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )

    nonterminal = (
        GraphRebuildRequest.Status.QUEUED,
        GraphRebuildRequest.Status.RUNNING,
    )
    blocking_runs = GraphBuildRun.objects.filter(artifact_id=OuterRef("pk")).filter(
        ~_terminal_run_identity_filter()
        | ~_terminal_run_stage_filter()
        | Q(finished_at__isnull=True)
        | Q(finished_at__gte=boundary)
        | ~Q(lease_owner="")
        | Q(lease_expires_at__isnull=False)
        | Q(rebuild_request__status__in=nonterminal)
        | Q(rebuild_request__parent_request__status__in=nonterminal)
    )
    live_requests = GraphRebuildRequest.objects.filter(
        pk=OuterRef("rebuild_request_id")
    ).filter(_nonterminal_request_filter())
    queryset = (
        GraphArtifact.objects.filter(
            status__in=(
                GraphArtifact.Status.SUPERSEDED,
                GraphArtifact.Status.STALE,
                GraphArtifact.Status.FAILED,
            )
        )
        .annotate(
            _terminal_run_at=Subquery(
                _artifact_terminal_run_query().values("finished_at")[:1],
                output_field=DateTimeField(),
            ),
            _has_terminal_run=Exists(_artifact_terminal_run_query()),
            _has_collection_use=Exists(
                CollectionArtifactInput.objects.filter(
                    document_artifact_id=OuterRef("pk")
                )
            ),
            _has_blocking_run=Exists(blocking_runs),
            _has_live_request=Exists(live_requests),
            _has_live_canonical_link=Exists(
                CanonicalEntityLink.objects.filter(
                    collection_entity__artifact_id=OuterRef("pk"),
                    status="active",
                )
            ),
        )
        .annotate(
            _artifact_status_at=Case(
                When(
                    status=GraphArtifact.Status.SUPERSEDED,
                    then=F("superseded_at"),
                ),
                default=F("completed_at"),
                output_field=DateTimeField(),
            )
        )
        .annotate(
            _terminal_at=Greatest(
                F("_artifact_status_at"),
                F("_terminal_run_at"),
                output_field=DateTimeField(),
            )
        )
        .filter(
            _artifact_status_at__isnull=False,
            _terminal_run_at__isnull=False,
            _has_terminal_run=True,
        )
        .filter(_terminal_at__lt=boundary)
        .filter(
            _has_collection_use=False,
            _has_blocking_run=False,
            _has_live_request=False,
            _has_live_canonical_link=False,
        )
    )
    keep = _nonnegative_setting(
        "KG_ARTIFACT_KEEP_SUPERSEDED", DEFAULT_ARTIFACT_KEEP_SUPERSEDED
    )
    if keep:
        queryset = queryset.annotate(
            _newer_superseded=Coalesce(
                Subquery(_newer_superseded_count_query(), output_field=IntegerField()),
                Value(0),
            )
        ).filter(
            ~Q(status=GraphArtifact.Status.SUPERSEDED) | Q(_newer_superseded__gte=keep)
        )
    return queryset


def _candidate_artifacts(
    *, boundary: datetime, batch_size: int
) -> tuple[_ArtifactCandidate, ...]:
    rows = tuple(
        _candidate_artifact_queryset(boundary=boundary)
        .order_by("_terminal_at", "scope_type", "scope_id", "build_generation", "pk")
        .values_list("pk", "_terminal_at")[: batch_size + 1]
    )
    return tuple(
        _ArtifactCandidate(row_id=row_id, prune_before=terminal_at)
        for row_id, terminal_at in rows[:batch_size]
    )


def _newer_scope_run_query():
    from apps.knowledge_graph.models import GraphBuildRun

    return GraphBuildRun.objects.filter(
        build_kind=OuterRef("build_kind"),
        scope_type=OuterRef("scope_type"),
        scope_id=OuterRef("scope_id"),
        orchestration_version=OuterRef("orchestration_version"),
    ).filter(
        Q(build_generation__gt=OuterRef("build_generation"))
        | Q(
            build_generation=OuterRef("build_generation"),
            pk__gt=OuterRef("pk"),
        )
    )


def _candidate_run_queryset(
    artifact_ids: tuple[int, ...],
    *,
    boundary: datetime,
):
    from apps.knowledge_graph.models import GraphBuildRun, GraphRebuildRequest

    if len(artifact_ids) > _MAX_QUERY_PREDICATE_IDS:
        raise ValueError("artifact predicate exceeds its cap")
    nonterminal = (
        GraphRebuildRequest.Status.QUEUED,
        GraphRebuildRequest.Status.RUNNING,
    )
    queryset = (
        GraphBuildRun.objects.filter(
            Q(
                stage=GraphBuildRun.Stage.FAILED,
                status=GraphBuildRun.Status.FAILED,
            )
            | Q(
                stage__in=(
                    GraphBuildRun.Stage.STALE,
                    GraphBuildRun.Stage.SUPERSEDED,
                ),
                status=GraphBuildRun.Status.CANCELLED,
            ),
            finished_at__lt=boundary,
            lease_owner="",
            lease_expires_at__isnull=True,
        )
        .filter(Q(artifact_id__isnull=True) | Q(artifact_id__in=artifact_ids))
        .exclude(rebuild_request__status__in=nonterminal)
        .exclude(rebuild_request__parent_request__status__in=nonterminal)
        .annotate(_newer_scope_run_exists=Exists(_newer_scope_run_query()))
        .filter(_newer_scope_run_exists=True)
    )
    return queryset


def _candidate_runs(
    artifact_ids: tuple[int, ...],
    *,
    boundary: datetime,
    batch_size: int,
) -> tuple[_RunCandidate, ...]:
    if not batch_size:
        return ()
    rows = tuple(
        _candidate_run_queryset(artifact_ids, boundary=boundary)
        .order_by("finished_at", "build_kind", "scope_id", "build_generation", "pk")
        .values_list("pk", "finished_at")[: batch_size + 1]
    )
    return tuple(
        _RunCandidate(row_id=row_id, prune_before=finished_at)
        for row_id, finished_at in rows[:batch_size]
    )


def _build_prune_plan(*, boundary: datetime, batch_size: int) -> _PrunePlan:
    artifact_candidates = _candidate_artifacts(
        boundary=boundary,
        batch_size=batch_size,
    )
    artifact_plan = _plan_pruning_candidates(
        artifact_candidates=artifact_candidates,
        run_candidates=(),
        batch_size=batch_size,
    )
    remaining = batch_size - len(artifact_plan.artifact_ids)
    run_candidates = _candidate_runs(
        artifact_plan.artifact_ids,
        boundary=boundary,
        batch_size=remaining,
    )
    return _plan_pruning_candidates(
        artifact_candidates=artifact_candidates,
        run_candidates=run_candidates,
        batch_size=batch_size,
    )


def _artifact_is_referenced(artifact) -> bool:
    from apps.knowledge_graph.models import GraphBuildRun, GraphRebuildRequest

    nonterminal = (
        GraphRebuildRequest.Status.QUEUED,
        GraphRebuildRequest.Status.RUNNING,
    )
    request_query = GraphRebuildRequest.objects.filter(
        pk=artifact.rebuild_request_id
    ).filter(_nonterminal_request_filter())
    return (
        artifact.collection_build_uses.exists()
        or artifact.build_runs.filter(
            Q(status__in=(GraphBuildRun.Status.PENDING, GraphBuildRun.Status.RUNNING))
            | Q(rebuild_request__status__in=nonterminal)
            | Q(rebuild_request__parent_request__status__in=nonterminal)
        ).exists()
        or artifact.collection_entities.filter(
            canonical_links__status="active"
        ).exists()
        or request_query.exists()
    )


def _newer_superseded_count(artifact) -> int:
    from apps.knowledge_graph.models import GraphArtifact

    return (
        GraphArtifact.objects.filter(
            scope_type=artifact.scope_type,
            scope_id=artifact.scope_id,
            status=GraphArtifact.Status.SUPERSEDED,
            evaluation_only=artifact.evaluation_only,
        )
        .filter(
            Q(build_generation__gt=artifact.build_generation)
            | Q(build_generation=artifact.build_generation, pk__gt=artifact.pk)
        )
        .count()
    )


_ARTIFACT_RUN_IDENTITY_FIELDS = (
    "scope_type",
    "scope_id",
    "rebuild_request_id",
    "evaluation_only",
    "build_key",
    "build_generation",
    "orchestration_version",
    "source_hash",
    "ontology_version",
    "extractor_version",
    "resolver_version",
    "filter_policy_version",
    "embedding_model_signature",
    "ontology_checksum",
    "filter_policy_checksum",
    "resolution_config_checksum",
    "assembly_version",
    "assembly_config_checksum",
)


def _run_has_clear_lease(run) -> bool:
    return (
        getattr(run, "lease_owner", None) == ""
        and getattr(run, "lease_expires_at", object()) is None
    )


def _artifact_retention_age(artifact, locked_runs: tuple[object, ...]):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    if type(locked_runs) is not tuple or not locked_runs:
        return None
    terminal_contract = {
        GraphArtifact.Status.FAILED: (
            "completed_at",
            GraphBuildRun.Stage.FAILED,
            GraphBuildRun.Status.FAILED,
        ),
        GraphArtifact.Status.STALE: (
            "completed_at",
            GraphBuildRun.Stage.STALE,
            GraphBuildRun.Status.CANCELLED,
        ),
        GraphArtifact.Status.SUPERSEDED: (
            "superseded_at",
            GraphBuildRun.Stage.SUPERSEDED,
            GraphBuildRun.Status.CANCELLED,
        ),
    }.get(getattr(artifact, "status", None))
    if terminal_contract is None:
        return None
    artifact_time = getattr(artifact, terminal_contract[0], None)
    if not isinstance(artifact_time, datetime) or timezone.is_naive(artifact_time):
        return None
    terminal_times = [artifact_time]
    for run in locked_runs:
        if (
            not _run_has_clear_lease(run)
            or getattr(run, "build_kind", None) != getattr(artifact, "scope_type", None)
            or any(
                getattr(run, field, object()) != getattr(artifact, field, object())
                for field in _ARTIFACT_RUN_IDENTITY_FIELDS
            )
            or getattr(run, "stage", None) != terminal_contract[1]
            or getattr(run, "status", None) != terminal_contract[2]
            or not isinstance(getattr(run, "finished_at", None), datetime)
            or timezone.is_naive(run.finished_at)
        ):
            return None
        terminal_times.append(run.finished_at)
    return max(terminal_times)


def _lock_pruning_scope(scope_type: str, scope_id: str) -> bool:
    from apps.knowledge_graph.models import GraphArtifact

    if scope_type == GraphArtifact.ScopeType.COLLECTION:
        from apps.collections.models import Collection
        from apps.knowledge_graph.graph.assembly import (
            lock_collection_graph_advisory_scope,
        )

        try:
            collection_id = int(scope_id)
        except (TypeError, ValueError):
            return False
        if collection_id <= 0 or str(collection_id) != scope_id:
            return False
        lock_collection_graph_advisory_scope(collection_id)
        # A deleted source scope is valid retention input. The logical advisory
        # fence remains stable even when there is no Collection row left to lock.
        Collection.objects.select_for_update().filter(pk=collection_id).first()
        return True
    if scope_type == GraphArtifact.ScopeType.DOCUMENT:
        from apps.knowledge_graph.services.builds import (
            lock_document_graph_advisory_scope,
        )

        try:
            document_id = uuid.UUID(scope_id)
        except (AttributeError, TypeError, ValueError):
            return False
        if str(document_id) != scope_id:
            return False
        lock_document_graph_advisory_scope(document_id)
        return True
    return False


def _bounded_lock_artifact_runs(artifact_id: int):
    from apps.knowledge_graph.models import GraphBuildRun

    run_ids = tuple(
        GraphBuildRun.objects.filter(artifact_id=artifact_id)
        .order_by("pk")
        .values_list("pk", flat=True)[: _MAX_QUERY_PREDICATE_IDS + 1]
    )
    if len(run_ids) > _MAX_QUERY_PREDICATE_IDS:
        return None
    return tuple(
        GraphBuildRun.objects.select_for_update().filter(pk__in=run_ids).order_by("pk")
    )


def _delete_one_artifact(artifact_id: int, *, boundary: datetime) -> bool:
    from apps.knowledge_graph.graph.invalidation import _delete_artifacts
    from apps.knowledge_graph.models import GraphArtifact

    with transaction.atomic(using="default"):
        probe = (
            GraphArtifact.objects.filter(pk=artifact_id)
            .values("scope_type", "scope_id")
            .first()
        )
        if probe is None or not _lock_pruning_scope(
            probe["scope_type"], probe["scope_id"]
        ):
            return False
        artifact = (
            GraphArtifact.objects.select_for_update().filter(pk=artifact_id).first()
        )
        if artifact is None:
            return False
        locked_runs = _bounded_lock_artifact_runs(artifact.pk)
        if locked_runs is None:
            return False
        terminal_at = _artifact_retention_age(artifact, locked_runs)
        if (
            artifact.status
            not in (
                GraphArtifact.Status.SUPERSEDED,
                GraphArtifact.Status.STALE,
                GraphArtifact.Status.FAILED,
            )
            or terminal_at is None
            or terminal_at >= boundary
            or _artifact_is_referenced(artifact)
        ):
            return False
        keep = _nonnegative_setting(
            "KG_ARTIFACT_KEEP_SUPERSEDED", DEFAULT_ARTIFACT_KEEP_SUPERSEDED
        )
        if (
            artifact.status == GraphArtifact.Status.SUPERSEDED
            and _newer_superseded_count(artifact) < keep
        ):
            return False
        _delete_artifacts((artifact,), using="default")
        return True


def _newer_scope_run_exists(run) -> bool:
    from apps.knowledge_graph.models import GraphBuildRun

    return (
        GraphBuildRun.objects.filter(
            build_kind=run.build_kind,
            scope_type=run.scope_type,
            scope_id=run.scope_id,
            orchestration_version=run.orchestration_version,
        )
        .filter(
            Q(build_generation__gt=run.build_generation)
            | Q(build_generation=run.build_generation, pk__gt=run.pk)
        )
        .exists()
    )


def _run_is_generation_high_water(*, newer_scope_run_exists: bool) -> bool:
    if type(newer_scope_run_exists) is not bool:
        raise ValueError("newer run marker must be a boolean")
    return not newer_scope_run_exists


def _run_is_deletable(run, *, boundary: datetime) -> bool:
    from apps.knowledge_graph.models import GraphBuildRun, GraphRebuildRequest

    coherent_terminal = (
        run.stage == GraphBuildRun.Stage.FAILED
        and run.status == GraphBuildRun.Status.FAILED
    ) or (
        run.stage in (GraphBuildRun.Stage.STALE, GraphBuildRun.Stage.SUPERSEDED)
        and run.status == GraphBuildRun.Status.CANCELLED
    )
    if (
        run.artifact_id is not None
        or not coherent_terminal
        or not _run_has_clear_lease(run)
        or run.finished_at is None
        or run.finished_at >= boundary
        or _run_is_generation_high_water(
            newer_scope_run_exists=_newer_scope_run_exists(run)
        )
    ):
        return False
    if run.rebuild_request_id is None:
        return True
    return (
        not GraphRebuildRequest.objects.filter(pk=run.rebuild_request_id)
        .filter(_nonterminal_request_filter())
        .exists()
    )


def _delete_one_run(run_id: int, *, boundary: datetime) -> bool:
    from apps.knowledge_graph.models import GraphBuildRun

    with transaction.atomic(using="default"):
        probe = (
            GraphBuildRun.objects.filter(pk=run_id)
            .values("scope_type", "scope_id", "artifact_id")
            .first()
        )
        if (
            probe is None
            or probe["artifact_id"] is not None
            or not _lock_pruning_scope(probe["scope_type"], probe["scope_id"])
        ):
            return False
        run = GraphBuildRun.objects.select_for_update().filter(pk=run_id).first()
        if run is None or not _run_is_deletable(run, boundary=boundary):
            return False
        GraphBuildRun.objects.filter(pk=run.pk).delete()
        return True


def prune_graph_artifacts(
    *, execute: bool = False, batch_size: int = _DEFAULT_BATCH_SIZE
) -> PruneReport:
    """Preview or delete one bounded, revalidated batch of terminal state."""

    if type(execute) is not bool:
        raise ValueError("execute must be a boolean")
    if type(batch_size) is not int or not 1 <= batch_size <= _MAX_BATCH_SIZE:
        raise ValueError("batch_size must be an integer in [1, 1000]")
    boundary = pruning_boundary()
    plan = _build_prune_plan(boundary=boundary, batch_size=batch_size)
    if not execute:
        return PruneReport(
            dry_run=True,
            artifact_ids=plan.artifact_ids,
            run_ids=plan.run_ids,
        )

    deleted_artifact_ids = tuple(
        artifact_id
        for artifact_id in plan.artifact_ids
        if _delete_one_artifact(artifact_id, boundary=boundary)
    )
    deleted_run_ids = tuple(
        run_id for run_id in plan.run_ids if _delete_one_run(run_id, boundary=boundary)
    )
    return PruneReport(
        dry_run=False,
        artifact_ids=deleted_artifact_ids,
        run_ids=deleted_run_ids,
    )


__all__ = ["PruneReport", "prune_graph_artifacts", "pruning_boundary"]
