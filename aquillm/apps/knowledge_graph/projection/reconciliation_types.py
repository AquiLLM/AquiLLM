from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReconcileSummaryV1:
    examined_count: int
    enqueued_count: int
    dry_run: bool
    drift_count: int = 0
    orphan_count: int = 0
    replayed_count: int = 0


@dataclass(frozen=True, slots=True)
class PruneSummaryV1:
    candidate_count: int
    deleted_count: int
    dry_run: bool
    orphan_count: int = 0
