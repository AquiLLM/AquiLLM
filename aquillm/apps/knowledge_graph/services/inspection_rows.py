"""Privacy-safe derived lifecycle fields for bounded graph inspection."""

from __future__ import annotations

_BUILD_ROW_FIELDS = (
    "pk",
    "artifact_id",
    "scope_type",
    "scope_id",
    "stage",
    "status",
    "error_code",
    "rebuild_request_id",
    "evaluation_only",
)


def bounded_build_inspection_rows(query: object, *, maximum: int) -> tuple[dict, ...]:
    """Materialize bounded run rows with one canonical eval-completion bit."""

    from apps.knowledge_graph.services.builds import _evaluation_occurrence_completed

    runs = tuple(query.select_related("artifact")[: maximum + 1])
    return tuple(
        {
            **{field: getattr(run, field) for field in _BUILD_ROW_FIELDS},
            "evaluation_completed": bool(
                run.evaluation_only is True
                and run.artifact_id is not None
                and _evaluation_occurrence_completed(
                    run.artifact,
                    run,
                    build_kind=run.build_kind,
                )
            ),
        }
        for run in runs
    )


__all__ = ["bounded_build_inspection_rows"]
