"""Read-only final observation of the original request-global acquisition ledger."""

from dataclasses import asdict, dataclass
from uuid import uuid4

from lib.evidence_observation import publish

from .frontier_ranking_validation_v2 import POLICY
from .ready_scope_v2 import _digest
from .topology.frontier_contracts_v2 import FrontierBudgetV2


@dataclass(frozen=True, slots=True)
class FrontierBudgetObservationV2:
    policy_signature: str
    max_queries: int
    max_rows: int
    max_bytes: int
    max_identities: int
    max_edges: int
    max_depth: int
    queries_used: int
    rows_used: int
    bytes_used: int
    identities_used: int
    edges_used: int
    accounting_complete: bool
    uncertain: bool


@dataclass(frozen=True, slots=True)
class GraphFrontierInvocationV2:
    invocation_id: str
    query_sha256: str
    requested_subset_signature: str
    frontier_budget: FrontierBudgetObservationV2
    request_algorithm_signature: str | None


def capture_frontier_budget_v2(budget, *, max_depth):
    if type(budget) is not FrontierBudgetV2:
        raise TypeError("observation requires the original frontier ledger")
    with budget._lock:
        limits = {
            "max_queries": budget.max_queries,
            "max_rows": budget.max_source_rows,
            "max_bytes": budget.max_wire_bytes,
            "max_identities": budget.max_identities,
            "max_edges": budget.max_edges,
            "max_depth": max_depth,
        }
        result = FrontierBudgetObservationV2(
            policy_signature=_digest(
                {
                    "version": "frontier-budget-v2/physical-reservation/raw-http-v2",
                    "acquisition_policy": POLICY,
                    "source_query_policy": (
                        "seed-data/seed-reauthorization/frontier-data-v2;"
                        "returned-records/not-plan-scans;"
                        "excludes-ready-capture/body/transaction-setup-control-sql"
                    ),
                    "limits": limits,
                }
            ),
            **limits,
            queries_used=budget.queries,
            rows_used=budget.source_rows,
            bytes_used=budget.wire_bytes,
            identities_used=budget.identities,
            edges_used=budget.edges,
            accounting_complete=not budget._uncertain and budget._reserved_rows == 0,
            uncertain=budget._uncertain,
        )
    return result


def publish_frontier_budget_v2(
    observation, *, query, requested_document_ids, request_algorithm_signature=None
):
    invocation = GraphFrontierInvocationV2(
        uuid4().hex,
        _digest(query),
        _digest(sorted(str(i) for i in requested_document_ids)),
        observation,
        request_algorithm_signature,
    )
    publish("graph_frontier_budget", asdict(invocation))
    return invocation
