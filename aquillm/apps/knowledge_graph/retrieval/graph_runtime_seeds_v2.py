"""Existing branch seed resolvers under captured group and request boundaries."""

from copy import copy
from math import fsum

from .production_extended import prepare_extended_branch
from .production_runtime_support import ProductionSharedScopeV1
from .query_ontology import load_query_ontology
from .relation_request import parse_relation_request_v2
from .scheduler_support import LocalBranchSchedulerFailure
from .topology.contracts import ProjectedSeedV1
from .topology.frontier_contracts_v2 import FrontierBudgetExceeded


def failure_reason(error):
    while error is not None:
        if isinstance(error, FrontierBudgetExceeded):
            return "frontier_budget"
        if isinstance(error, TimeoutError):
            return "deadline"
        error = error.__cause__
    return "not_ready"


def group_request(*, query, group, authorization, read_budget=None):
    outcome = load_query_ontology(
        selected_artifact_ids=tuple(
            sorted({p.artifact_id for p in group.scope.projections})
        ),
        using=authorization.database_alias,
        **({"read_budget": read_budget} if read_budget is not None else {}),
    )
    ontology = outcome.ontology
    if ontology is None or ontology.checksum != group.ontology_checksum:
        raise ValueError("captured group has no exact approved ontology")
    return parse_relation_request_v2(
        query,
        ontology_checksum=ontology.checksum,
        allowed_relations=frozenset(ontology.relations),
    )


def _group_request_native(*, owner, query, group, authorization, read_budget):
    from .native_query_ontology import load_native_ontology

    metadata = owner._native_metadata.bind(group=group, budget=read_budget, authorization=authorization)
    outcome = load_native_ontology(metadata, group=group)
    ontology = outcome.ontology
    if ontology is None or ontology.checksum != group.ontology_checksum:
        raise ValueError("captured group has no exact approved ontology")
    result = parse_relation_request_v2(query, ontology_checksum=ontology.checksum,
        allowed_relations=frozenset(ontology.relations))
    metadata.check()
    return result


def group_seeds(
    *,
    runtime,
    group,
    query,
    baseline,
    authorization,
    settings,
    caps,
    deadline,
    frontier,
    seed_validation=None,
):
    values, reasons = [], set()
    if settings.graph_direct_enabled:
        try:
            from .native_direct_seed_repository import _prepare_direct_native
            native = seed_validation is not None and seed_validation._native_metadata is not None
            direct = (_prepare_direct_native if native else runtime._direct_seeds)(
                **({"runtime": runtime, "owner": seed_validation, "group": group} if native else {}),
                query=query,
                **({} if native else {"scope": group.scope}),
                deadline=deadline,
                read_budget=frontier,
            )
        except LocalBranchSchedulerFailure as error:
            reasons.add(failure_reason(error))
            direct = None
        if type(direct) is tuple:
            values.extend(direct)
        elif direct is not None and "timeout" in str(direct).lower():
            reasons.add("deadline")
        else:
            reasons.add("not_ready")
    if settings.graph_extended_enabled:
        if baseline is not None:
            local = copy(runtime)
            local._shared = ProductionSharedScopeV1(group.scope)
            try:
                from .production_extended import _prepare_extended_branch

                prepare = prepare_extended_branch if seed_validation is None else _prepare_extended_branch
                extended = prepare(
                    local,
                    baseline=baseline,
                    shared=local._shared,
                    authorization=authorization,
                    settings=settings,
                    deadline=deadline,
                    read_budget=frontier,
                    **({"seed_validation": seed_validation, "query": "", "with_policy": False}
                       if seed_validation is not None else {}),
                )
            except LocalBranchSchedulerFailure as error:
                reasons.add(failure_reason(error))
                extended = None
            if type(extended) is tuple:
                values.extend(extended)
            elif extended is not None and "timeout" in str(extended).lower():
                reasons.add("deadline")
            else:
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

_NATIVE_GROUP_REQUEST = group_request
