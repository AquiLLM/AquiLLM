"""One private finite reservation on the original nonrefundable frontier ledger."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowGrant:
    queries: int
    rows: int
    wire: int
    planning_rows: int


def preview_window(budget):
    from .frontier_contracts_v2 import FrontierBudgetExceeded

    with budget._lock:
        budget.check_active()
        budget._require_no_window()
        queries = min(
            8,
            budget.max_queries
            - budget.queries
            - sum(q for _, q, _ in budget._authority_reservations.values()),
        )
        available = budget.available_read_rows()
        wire = budget.available_wire_bytes()
        if queries < 1 or available < 1 or wire < 1 or budget._reserved_rows:
            raise FrontierBudgetExceeded("frontier window cannot fit")
        grant = WindowGrant(queries, min(available, queries * 320), wire, available)
        return grant


def reserve_window(budget):
    with budget._lock:
        grant = preview_window(budget)
        budget._window = grant
        return grant


def settle_window(budget, grant, *, queries, scanned_rows):
    with budget._lock:
        if budget._window is not grant:
            raise ValueError("frontier window reservation missing")
        if (
            type(queries) is not int
            or not 1 <= queries <= grant.queries
            or type(scanned_rows) is not int
            or not 0 <= scanned_rows <= min(grant.rows, queries * 320)
        ):
            budget.stop_uncertain()
            raise ValueError("frontier window receipt exceeds reservation")
        budget.queries += queries
        budget.source_rows += scanned_rows
        budget._window = None


class WindowReservationMixin:
    """Private reservation operations over the inheriting original ledger only."""

    reserve_window = reserve_window
    settle_window = settle_window

    def stop_uncertain(self):
        with self._lock:
            self._uncertain = True

    def _require_no_window(self):
        from .frontier_contracts_v2 import FrontierBudgetExceeded

        if self._window is not None:
            raise FrontierBudgetExceeded("frontier window already outstanding")
