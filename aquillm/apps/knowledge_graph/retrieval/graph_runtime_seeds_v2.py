"""Existing branch seed resolvers under captured group and request boundaries."""

from copy import copy
from math import fsum

from apps.documents.services.chunk_search_candidates import HybridCandidateSnapshot
from apps.documents.services.chunk_search_scope import current_scope
from apps.documents.services.chunk_search_scoped_rows import validate_rows

from .production_extended import prepare_extended_branch
from .production_runtime_support import ProductionSharedScopeV1
from .query_ontology import load_query_ontology
from .relation_request import parse_relation_request_v2
from .scheduler_support import LocalBranchSchedulerFailure
from .topology.contracts import ProjectedSeedV1


def group_request(*, query, group, authorization):
    outcome = load_query_ontology(
        selected_artifact_ids=tuple(
            sorted({p.artifact_id for p in group.scope.projections})
        ),
        using=authorization.database_alias,
    )
    ontology = outcome.ontology
    if ontology is None or ontology.checksum != group.ontology_checksum:
        raise ValueError("captured group has no exact approved ontology")
    return parse_relation_request_v2(
        query,
        ontology_checksum=ontology.checksum,
        allowed_relations=frozenset(ontology.relations),
    )


def group_seeds(
    *, runtime, group, query, baseline, authorization, settings, caps, deadline
):
    values, reasons = [], set()
    if settings.graph_direct_enabled:
        try:
            direct = runtime._direct_seeds(
                query=query, scope=group.scope, deadline=deadline
            )
        except LocalBranchSchedulerFailure:
            direct = None
        if type(direct) is tuple:
            values.extend(direct)
        elif direct is not None and "timeout" in str(direct).lower():
            reasons.add("deadline")
        else:
            reasons.add("not_ready")
    if settings.graph_extended_enabled:
        scope = current_scope()
        valid = (
            type(baseline) is HybridCandidateSnapshot
            and scope is not None
            and scope.authorization == authorization
            and {d.id for d in baseline.documents} == set(scope.requested_document_ids)
        )
        if valid:
            rows = validate_rows(baseline.baseline_candidates, scope=scope)
            valid = rows == baseline.baseline_candidates and all(
                r.doc_id in scope.requested_document_ids for r in rows
            )
        if valid:
            local = copy(runtime)
            local._shared = ProductionSharedScopeV1(group.scope)
            try:
                extended = prepare_extended_branch(
                    local,
                    baseline=baseline,
                    shared=local._shared,
                    authorization=authorization,
                    settings=settings,
                    deadline=deadline,
                )
            except LocalBranchSchedulerFailure:
                extended = None
            if type(extended) is tuple:
                values.extend(extended)
            elif extended is not None and "timeout" in str(extended).lower():
                reasons.add("deadline")
        elif baseline is not None:
            reasons.add("not_ready")
    masses = {}
    for seed in values:
        if type(seed) is not ProjectedSeedV1:
            raise TypeError("seed resolver returned an invalid seed")
        masses[seed.identity_key] = max(seed.mass, masses.get(seed.identity_key, 0.0))
    ordered = sorted(masses, key=lambda k: (-masses[k], k))
    if len(ordered) > caps.max_seeds:
        reasons.add("frontier_budget")
    keys = ordered[: caps.max_seeds]
    total = fsum(masses[k] for k in keys)
    return tuple(ProjectedSeedV1(k, masses[k] / total) for k in sorted(keys)), reasons
