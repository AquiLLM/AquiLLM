"""Whole opt-in graph runtime: one capture, frontier, hydration and selector."""

from apps.knowledge_graph.projection.frontier_snapshot_v2 import frontier_config

from .frontier_ranking_validation_v2 import CONSUMED_POLICY, POLICY, frontier_tags
from .graph_evidence_repository_v2 import load_selection_inputs_v2
from .graph_evidence_selection_v2 import (
    GraphEvidenceSelectionEnvelopeV2,
    select_graph_evidence_v2,
)
from .graph_handoff import GraphCoverageV2, GraphHandoffV2
from .graph_query_contracts import GraphGroupRelationRequestV2
from .graph_runtime_contracts_v2 import (
    graph_algorithm_signature_v2,
)
from .graph_runtime_seeds_v2 import group_request
from .graph_seed_allocation_v2 import allocate_group_seeds_v2
from .projected_relation_ranking_v2 import rank_relations_v2
from .ready_scope_repository_v2 import select_ready_scope_v2
from .topology.factory import create_canonical_frontier_loader_v2
from .topology.frontier_contracts_v2 import FrontierBudgetExceeded
from .topology.frontier_loader_v2 import FrontierUnavailableV2


def execute_graph_v2(
    *,
    query,
    runtime,
    authorization,
    representation_snapshot,
    budget,
    settings,
    deadline,
    baseline,
    requested_document_ids,
    caps,
    frontier,
):
    frontier.check_active()
    selected = select_ready_scope_v2(
        authorization=authorization,
        settings=settings,
        representation_snapshot=representation_snapshot,
        requested_document_ids=requested_document_ids,
    )
    return _execute_captured_graph_v2(
        query=query,
        runtime=runtime,
        authorization=authorization,
        representation_snapshot=representation_snapshot,
        budget=budget,
        settings=settings,
        deadline=deadline,
        baseline=baseline,
        requested_document_ids=requested_document_ids,
        caps=caps,
        frontier=frontier,
        selected=selected,
        hydrate=load_selection_inputs_v2,
    )


def _execute_captured_graph_v2(
    *,
    query,
    runtime,
    authorization,
    representation_snapshot,
    budget,
    settings,
    deadline,
    baseline,
    requested_document_ids,
    caps,
    frontier,
    selected,
    hydrate,
    seed_validation=None,
    acquisition_policy=None,
):
    """Private eager core; native owner or public caller captures before entry."""
    frontier.check_active()
    if (
        selected.representation_snapshot != representation_snapshot
        or selected.authorization_context_signature
        != authorization.authorization_context_signature
    ):
        raise ValueError("ready selector replaced original authority")
    requests = []
    from .graph_runtime_seeds_v2 import _group_request_native
    native = seed_validation is not None and seed_validation._native_metadata is not None
    if native and seed_validation._native_metadata.selected is not selected:
        raise ValueError("native metadata original selection changed")
    expected_policy = CONSUMED_POLICY if native else POLICY
    acquisition_policy = expected_policy if acquisition_policy is None else acquisition_policy
    frontier_tags(acquisition_policy)
    if acquisition_policy != expected_policy:
        raise ValueError("frontier acquisition policy differs from native owner")
    for g in selected.groups:
        frontier.check_active()
        requests.append(
            GraphGroupRelationRequestV2(
                g.group_checksum,
                (_group_request_native if native else group_request)(
                    **({"owner": seed_validation} if native else {}),
                    query=query,
                    group=g,
                    authorization=authorization,
                    read_budget=frontier,
                ),
            )
        )
        frontier.check_active()
    requests = tuple(requests)
    configs = {
        g.group_checksum: frontier_config(
            caps, g.scope.ready.selected_generations[0].resolver_version
        )
        for g in selected.groups
    }
    signature = graph_algorithm_signature_v2(
        selected=selected,
        requests=requests,
        caps=caps,
        budget=frontier,
        configs=configs,
        acquisition_policy=acquisition_policy,
    )
    loader = (
        create_canonical_frontier_loader_v2(
            selection=selected,
            codec=runtime.codec,
            gateway_client=runtime.topology_loader.driver,
            acquisition_policy=acquisition_policy,
        )
        if selected.groups
        else None
    )
    references, scores, reasons = [], {}, set(selected.coverage.reasons)
    try:
        from .graph_seed_allocation_v2 import _allocate_closed_group_seeds_v2

        allocate = allocate_group_seeds_v2 if seed_validation is None else _allocate_closed_group_seeds_v2
        allocations, seed_reasons = allocate(
            groups=selected.groups,
            runtime=runtime,
            query=query,
            baseline=baseline,
            authorization=authorization,
            settings=settings,
            caps=caps,
            deadline=deadline,
            frontier=frontier,
            **({"seed_validation": seed_validation} if seed_validation is not None else {}),
        )
    except (FrontierBudgetExceeded, TimeoutError) as exc:
        reason = (
            "frontier_budget" if isinstance(exc, FrontierBudgetExceeded) else "deadline"
        )
        seed_reasons = {reason}
        allocations = {g.group_checksum: ((), 0.0, {reason}) for g in selected.groups}
    reasons.update(seed_reasons)
    available = 0
    for group, bound in zip(selected.groups, requests, strict=True):
        try:
            frontier.check_active()
            seeds, group_mass, seed_reasons = allocations[group.group_checksum]
            reasons.update(seed_reasons)
            if not seeds:
                if not seed_reasons:
                    available += len(group.scope.selected_document_ids)
                continue
            result = loader.load(
                group=group,
                seeds=seeds,
                request=bound.request,
                caps=caps,
                budget=frontier,
                deadline=deadline,
            )
            if result.authority_group_checksum != group.group_checksum:
                raise ValueError("frontier replaced captured group")
            context = loader.ranking_context(result)
            available += result.coverage.ready_documents
            reasons.update(result.coverage.reasons)
            if any(s.identity_key not in result.snapshot.identity_keys for s in seeds):
                reasons.add(
                    "frontier_budget"
                    if "frontier_budget" in result.coverage.reasons
                    else "not_ready"
                )
                continue
            if result.snapshot.identity_keys and seeds:
                ranked = rank_relations_v2(
                    snapshot=result.snapshot,
                    seeds=seeds,
                    request=bound.request,
                    config=configs[group.group_checksum],
                    frontier_context=context,
                    deadline_check=frontier.check_active,
                )
                for key, mass in ranked.scores:
                    scores[key] = max(mass * group_mass, scores.get(key, 0.0))
                references.extend(result.evidence_candidates)
        except FrontierUnavailableV2 as exc:
            reasons.update(exc.coverage.reasons)
        except FrontierBudgetExceeded:
            reasons.add("frontier_budget")
        except TimeoutError:
            reasons.add("deadline")
            break
    total = selected.coverage.selected_documents
    coverage = GraphCoverageV2(
        total,
        available,
        "complete"
        if available == total and not reasons
        else "partial"
        if available
        else "unavailable",
        tuple(sorted(reasons)),
        False,
    )
    try:
        frontier.check_active()
        from .graph_evidence_associations_v2 import association_prefix

        admitted = (
            association_prefix(
                references, maximum=min(c.max_evidence_rows for c in configs.values())
            )
            if configs
            else ()
        )
        if len(admitted) != len(references):
            from .graph_evidence_selection_v2 import partial_coverage

            coverage = partial_coverage(coverage)
        inputs = hydrate(
            selected=selected,
            references=admitted,
            authorization=authorization,
            budget=budget,
            max_rows=45,
            deadline=deadline,
        )
        if len(inputs) != len(references):
            from .graph_evidence_selection_v2 import partial_coverage

            coverage = partial_coverage(coverage)
        envelope = GraphEvidenceSelectionEnvelopeV2(
            inputs,
            coverage,
            signature,
            tuple(
                (g.group_checksum, g.scope.ready.bundle_checksum)
                for g in selected.groups
            ),
        )
        frontier.check_active()
        handoff = select_graph_evidence_v2(
            query=query,
            requests=requests,
            envelope=envelope,
            identity_scores=tuple(sorted(scores.items())),
            max_candidates=caps.max_results,
            max_mentions_per_identity=2,
            max_evidence_per_transition=3,
        )
        frontier.check_active()
    except (TimeoutError, FrontierBudgetExceeded) as exc:
        from .graph_evidence_selection_v2 import partial_coverage

        handoff = GraphHandoffV2(
            (),
            partial_coverage(
                coverage,
                "frontier_budget"
                if isinstance(exc, FrontierBudgetExceeded)
                else "deadline",
            ),
            signature,
        )
    return selected, handoff
