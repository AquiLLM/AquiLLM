"""Sequential group admission under a single frontier and caller TurnBudget."""

from dataclasses import dataclass
from math import isfinite

from lib.retrieval.turn_budget import TurnBudget

from .graph_handoff import GraphCandidateRefV2, GraphCoverageV2
from .ready_scope_v2 import SelectedGraphScopeV2


@dataclass(slots=True)
class GraphGroupFrontierV2:
    max_identities: int = 200
    max_edges: int = 1000
    identities: int = 0
    edges: int = 0

    def __post_init__(self):
        for value, maximum in ((self.max_identities, 200), (self.max_edges, 1000)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("frontier must preserve existing hard caps")
        if self.identities != 0 or self.edges != 0:
            raise ValueError("new frontier counters must start empty")

    def admit(self, *, identities, edges):
        if any(type(v) is not int or v < 0 for v in (identities, edges)):
            raise ValueError("frontier usage must be exact nonnegative integers")
        if (
            self.identities + identities > self.max_identities
            or self.edges + edges > self.max_edges
        ):
            return False
        self.identities += identities
        self.edges += edges
        return True

    @property
    def exhausted(self):
        return self.identities >= self.max_identities or self.edges >= self.max_edges


def schedule_ready_groups_v2(
    *,
    selected,
    execute_group,
    turn_budget,
    deadline,
    clock,
    max_identities=200,
    max_edges=1000,
    maximum=20,
):
    if (
        type(selected) is not SelectedGraphScopeV2
        or type(turn_budget) is not TurnBudget
    ):
        raise TypeError("selected scope and original TurnBudget must be exact")
    if type(maximum) is not int or not 1 <= maximum <= 20:
        raise ValueError(
            "final candidate cap must preserve the existing branch maximum"
        )
    if type(deadline) not in (int, float) or not isfinite(deadline):
        raise ValueError("deadline must be finite")
    deadline = min(deadline, clock() + turn_budget.remaining_ms() / 1000)
    frontier = GraphGroupFrontierV2(max_identities, max_edges)
    candidates, reasons = {}, set(selected.coverage.reasons)
    for group in selected.groups:
        reason = (
            "deadline"
            if clock() >= deadline or turn_budget.remaining_ms() <= 0
            else ("frontier_budget" if frontier.exhausted else None)
        )
        if reason is not None:
            reasons.update((reason, "incompatible_generation"))
            break
        # G3 supplies the ontology-specific seed/relation request and charges the
        # same frontier/source ledger. Nothing is reset between callbacks.
        refs = execute_group(
            group, frontier=frontier, turn_budget=turn_budget, deadline=deadline
        )
        if type(refs) is not tuple or any(
            type(r) is not GraphCandidateRefV2 for r in refs
        ):
            raise TypeError("group executor must return exact private candidate refs")
        if clock() >= deadline or turn_budget.remaining_ms() <= 0:
            reasons.add("deadline")
            continue
        for ref in refs:
            if (
                ref.authority_group_checksum != group.group_checksum
                or ref.ready_bundle_checksum != group.scope.ready.bundle_checksum
            ):
                raise ValueError("executor replaced captured authority")
            group.scope.projection_for_generation(ref.graph_generation_key)
            previous = candidates.get(ref.chunk_key)
            if previous is None or ref.score > previous.score:
                candidates[ref.chunk_key] = ref
    coverage = selected.coverage
    if reasons and coverage.ready_documents:
        coverage = GraphCoverageV2(
            coverage.selected_documents,
            coverage.ready_documents,
            "partial",
            tuple(sorted(reasons)),
            False,
        )
    return tuple(
        sorted(candidates.values(), key=lambda c: (-c.score, c.chunk_key))[:maximum]
    ), coverage
