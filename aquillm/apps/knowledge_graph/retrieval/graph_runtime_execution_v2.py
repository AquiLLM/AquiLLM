"""Whole opt-in graph runtime: one capture, frontier, hydration and selector."""

from apps.knowledge_graph.projection.frontier_snapshot_v2 import frontier_config

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
from .graph_runtime_seeds_v2 import group_request, group_seeds
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
    if (
        selected.representation_snapshot != representation_snapshot
        or selected.authorization_context_signature
        != authorization.authorization_context_signature
    ):
        raise ValueError("ready selector replaced original authority")
    requests = []
    for g in selected.groups:
        frontier.check_active()
        requests.append(
            GraphGroupRelationRequestV2(
                g.group_checksum,
                group_request(query=query, group=g, authorization=authorization),
            )
        )
        frontier.check_active()
    requests = tuple(requests)
    signature = graph_algorithm_signature_v2(
        selected=selected, requests=requests, caps=caps, budget=frontier
    )
    loader = (
        create_canonical_frontier_loader_v2(
            selection=selected,
            codec=runtime.codec,
            gateway_client=runtime.topology_loader.driver,
        )
        if selected.groups
        else None
    )
    references, scores, reasons = [], {}, set(selected.coverage.reasons)
    available = 0
    for group, bound in zip(selected.groups, requests, strict=True):
        try:
            frontier.check_active()
            seeds, seed_reasons = group_seeds(
                runtime=runtime,
                group=group,
                query=query,
                baseline=baseline,
                authorization=authorization,
                settings=settings,
                caps=caps,
                deadline=deadline,
            )
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
                reasons.add("frontier_budget" if "frontier_budget" in
                    result.coverage.reasons else "not_ready")
                continue
            if result.snapshot.identity_keys and seeds:
                config = frontier_config(
                    caps, group.scope.ready.selected_generations[0].resolver_version
                )
                ranked = rank_relations_v2(
                    snapshot=result.snapshot,
                    seeds=seeds,
                    request=bound.request,
                    config=config,
                    frontier_context=context,
                    deadline_check=frontier.check_active,
                )
                for key, mass in ranked.scores:
                    scores[key] = max(mass, scores.get(key, 0.0))
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
        inputs = load_selection_inputs_v2(
            selected=selected,
            references=tuple(references),
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
                "deadline" if isinstance(exc, TimeoutError) else "frontier_budget",
            ),
            signature,
        )
    return selected, handoff
