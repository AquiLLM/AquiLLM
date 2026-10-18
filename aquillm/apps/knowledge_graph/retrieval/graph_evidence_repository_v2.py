"""Bounded private hydration using the original representation and body ledger."""

from time import monotonic

from apps.collections.services.retrieval_authorization import (
    RetrievalAuthorizationContext,
)
from apps.documents.services.chunk_search_scope import current_scope
from apps.documents.services.source_loading import (
    bounded_source_database,
)
from apps.knowledge_graph.projection.runtime import (
    load_projection_runtime_settings,
    projection_identifier_codec,
)
from lib.retrieval.turn_budget import TurnBudget

from .graph_evidence_selection_v2 import GraphEvidenceSelectionInputV2
from .graph_handoff import GraphCandidateRefV2
from .graph_hydration_operation_v2 import hydrate_selection_operation
from .graph_query_contracts import GraphEvidenceRefV2
from .ready_scope_v2 import SelectedGraphScopeV2


def check_deadline(budget, deadline):
    budget.check_active()
    if monotonic() >= deadline or not budget.can_publish():
        raise TimeoutError("graph evidence deadline")


def load_selection_inputs_v2(
    *,
    selected: SelectedGraphScopeV2,
    references: tuple[GraphEvidenceRefV2, ...],
    authorization: RetrievalAuthorizationContext,
    budget: TurnBudget,
    max_rows: int,
    deadline: float,
) -> tuple[GraphEvidenceSelectionInputV2, ...]:
    return _load_selection_inputs_v2(
        selected=selected,
        references=references,
        authorization=authorization,
        budget=budget,
        max_rows=max_rows,
        deadline=deadline,
        hydrate=hydrate_selection_operation,
    )


def _load_selection_inputs_v2(
    *,
    selected,
    references,
    authorization,
    budget,
    max_rows,
    deadline,
    hydrate,
):
    """Private pairing/SQL core; public and owned callers retain their boundaries."""
    if type(selected) is not SelectedGraphScopeV2 or type(budget) is not TurnBudget:
        raise TypeError("exact captured selection and original budget required")
    if (
        type(references) is not tuple
        or len(references) > 5000
        or any(type(r) is not GraphEvidenceRefV2 for r in references)
    ):
        raise TypeError("bounded exact evidence references required")
    if type(max_rows) is not int or not 1 <= max_rows <= 45:
        raise ValueError("hydration row allowance must be within 45")
    from apps.documents.services.graph_source_runtime import graph_source_runtime

    runtime = graph_source_runtime()
    if (
        runtime is None
        or runtime.budget is not budget
        or runtime.authorization is not authorization
    ):
        raise ValueError("graph hydration requires original source runtime")
    scope = current_scope()
    if scope is not None and (
        scope.runtime is not runtime
        or scope.representation is not selected.representation_snapshot
    ):
        raise ValueError("graph hydration replaced original source scope")
    groups = {g.group_checksum: g for g in selected.groups}
    refs = []
    for ref in references:
        group = groups.get(ref.authority_group_checksum)
        if group is None:
            raise ValueError("reference is outside captured groups")
        group.scope.projection_for_generation(ref.graph_generation_key)
        refs.append(
            GraphCandidateRefV2(
                ref.chunk_key,
                ref.graph_generation_key,
                ref.representation_generation_key,
                ref.authority_group_checksum,
                group.scope.ready.bundle_checksum,
                ref.evidence_provenance_keys,
                ref.relation_types,
                ref.confidence,
            )
        )
    if len({r.chunk_key for r in refs}) != len(refs):
        raise ValueError("duplicate evidence references")
    check_deadline(budget, deadline)
    codec = projection_identifier_codec(load_projection_runtime_settings())
    # Resolve only bounded private metadata before reading any bodies. A narrower
    # retrieval request must never hydrate other documents in the original scope.
    from apps.knowledge_graph.models import ProjectionChunkReference

    with bounded_source_database(runtime, authorization.database_alias):
        requested = (
            scope.requested_document_ids
            if scope
            else authorization.selected_document_ids
        )
        keys = {r.chunk_key for r in refs[:max_rows]}
        allowed = set(
            ProjectionChunkReference.objects.using(authorization.database_alias)
            .filter(
                projection_chunk_key__in=keys,
                document_uuid__in=requested,
                projection_id__in=[
                    p.projection_id
                    for g in selected.groups
                    for p in g.scope.projections
                ],
            )
            .values_list("projection_chunk_key", flat=True)[: max_rows + 1]
        )
        if len(allowed) > max_rows:
            raise ValueError("private evidence mapping overflow")
        pairs = [
            (ref, source)
            for ref, source in zip(refs[:max_rows], references[:max_rows], strict=True)
            if ref.chunk_key in allowed
        ]
    return hydrate(
        selected=selected,
        pairs=tuple(pairs),
        runtime=runtime,
        document_codec=codec,
        deadline=deadline,
    )
