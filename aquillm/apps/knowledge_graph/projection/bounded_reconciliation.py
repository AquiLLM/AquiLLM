"""Scheduled reconciliation examines a bounded slice; explicit audits stay full."""

from dataclasses import dataclass
from time import monotonic

from .maintenance import MAX_ARTIFACTS, DeadlineDriver
from .memgraph_repository import MemgraphProjectionRepository
from .state_repository import StaleProjectionAuthority


@dataclass(frozen=True)
class BatchSummary:
    examined_count: int
    enqueued_count: int
    failure_count: int
    next_cursor: int


def reconcile_projection_batch(
    *,
    after_id,
    page_size,
    deadline,
    collection_id=None,
    clock=monotonic,
    save_cursor=None,
):
    from . import reconciler as control

    size = min(control._size(page_size, "page_size"), MAX_ARTIFACTS)
    selected_collection = control._collection(collection_id)
    settings = control._projection_settings()
    postgres = control._postgres_repository()
    repository = control._memgraph_repository()
    graph = MemgraphProjectionRepository(
        DeadlineDriver(
            repository._driver,
            deadline=deadline,
            timeout_seconds=settings.projection_timeout_ms / 1000.0,
            clock=clock,
        )
    )
    codec = control.projection_identifier_codec(settings)
    page = control._active_artifact_page(
        after_id=after_id,
        page_size=size,
        collection_id=selected_collection,
    )
    examined = enqueued = failed = 0
    cursor = after_id
    for active_collection_id, artifact_id in page:
        if clock() >= deadline:
            break
        examined += 1
        cursor = artifact_id
        try:
            row = control._projection_for_active(
                collection_id=active_collection_id,
                artifact_id=artifact_id,
            )
            audit = control._generation_audit(
                row=row,
                postgres=postgres,
                graph=graph,
                settings=settings,
            )
            if audit.replay_reason is not None and clock() < deadline:
                control._replay_projection(
                    row=row,
                    collection_id=active_collection_id,
                    artifact_id=artifact_id,
                    codec=codec,
                )
                enqueued += 1
        except StaleProjectionAuthority:
            pass
        except Exception:
            # Backend failures are not evidence of drift. Preserve authority,
            # record only a fixed count, and give the next collection its turn.
            failed += 1
        if save_cursor is not None and not save_cursor(cursor):
            raise RuntimeError("maintenance_coordination_unavailable")
    if examined == len(page) and len(page) < size:
        cursor = 0
    if save_cursor is not None and not save_cursor(cursor):
        raise RuntimeError("maintenance_coordination_unavailable")
    return BatchSummary(examined, enqueued, failed, cursor)
