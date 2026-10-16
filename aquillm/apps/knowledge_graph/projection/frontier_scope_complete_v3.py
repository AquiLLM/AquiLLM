"""Scope-only fixed complete reads; copy and legacy raw planning stay separate."""

from dataclasses import dataclass

from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)

from .frontier_scope_preparation_v2 import ScopePreparation


def complete_scope_enabled(driver):
    from apps.knowledge_graph.retrieval.topology.frontier_gateway_client_v2 import (
        FrontierGatewayClientV2,
    )

    return (
        type(driver) is FrontierGatewayClientV2
        and driver.complete_scope_windows is True
    )


@dataclass(frozen=True)
class ScopeCompleteRead:
    pairs: tuple
    kind: str = "scope_raw_complete_batch"
    maximum: int = 320


class ScopeCompletePreparation(ScopePreparation):
    def _effect(self, kind, pending, allowance):
        if kind != "raw_batch_multi":
            return super()._effect(kind, pending, allowance)
        available = (
            self.reads.budget.available_read_rows() if allowance is None else allowance
        )
        if available < 320:
            raise FrontierBudgetExceeded("frontier physical work exhausted")
        self.effect = ScopeCompleteRead(tuple(pending[:128]))
        return self.effect
