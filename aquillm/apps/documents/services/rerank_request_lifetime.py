"""One outage latch per ledger, with leases held through actual worker cleanup."""

from concurrent.futures import Future, TimeoutError
from contextvars import copy_context
from threading import Thread

from lib.retrieval.turn_budget import TurnBudget, TurnLimits


class TurnRerankState:
    def __new__(cls, budget: TurnBudget, *, worker_cleanup=None):
        if not isinstance(budget, TurnBudget):
            raise TypeError("reranking requires the shared turn budget")
        with budget._lock:
            existing = getattr(budget, "_rerank_state", None)
            if existing is not None:
                if worker_cleanup is not None and existing.worker_cleanup is None:
                    existing.worker_cleanup = worker_cleanup
                return existing
            state = super().__new__(cls)
            state.budget, state._failed = budget, False
            state.worker_cleanup = worker_cleanup
            budget._rerank_state = state
            return state

    def can_admit(self):
        with self.budget._lock:
            return not self._failed and self.budget.can_publish()

    def fail(self):
        """Invalidate scoring admission only; source authority remains available."""
        with self.budget._lock:
            self._failed = True

    def publish(self, callback):
        with self.budget._lock:
            return not self._failed and self.budget.publish(callback)

    def submit(self, count, callback, *, phase="acquisition", timeout=None):
        with self.budget._lock:
            remaining = self.budget.scoring_remaining_ms(phase) / 1000
            if timeout is not None:
                remaining = min(remaining, timeout)
            if (
                remaining <= 0
                or not self.can_admit()
                or not self.budget.start_pairs(count, phase=phase)
            ):
                return None
            future = Future()
            # Admitted work cannot be cancelled into releasing a live I/O lease.
            future.set_running_or_notify_cancel()
            context = copy_context()

        def work():
            result = None
            try:
                result = callback(remaining)
                if result is None:
                    self.fail()
                if not self.can_admit() or self.budget.scoring_remaining_ms(phase) <= 0:
                    result = None
            except Exception:
                self.fail()
            finally:
                try:
                    if self.worker_cleanup:
                        self.worker_cleanup()
                except Exception:
                    self.fail()
                    result = None
                finally:
                    self.budget.finish_pairs(count)
                    future.set_result(result)

        try:
            Thread(
                target=context.run, args=(work,), daemon=True, name="rerank-io"
            ).start()
        except Exception:
            self.budget.finish_pairs(count)
            self.fail()
            return None
        return future

    def run(self, count, callback, *, phase="acquisition", timeout=None):
        future = self.submit(count, callback, phase=phase, timeout=timeout)
        if future is None:
            return None
        remaining = self.budget.scoring_remaining_ms(phase) / 1000
        if timeout is not None:
            remaining = min(remaining, timeout)
        try:
            result = future.result(max(0, remaining))
            return (
                result
                if self.can_admit() and self.budget.scoring_remaining_ms(phase) > 0
                else None
            )
        except TimeoutError:
            self.fail()
            return None


def rerank_state_for(budget=None, state=None):
    """Application adapter: inherit the current ledger instead of resetting work."""
    from lib.llm.turn_context import current_turn

    turn = current_turn()
    budget = budget or (turn.budget if turn else None)
    if state is not None:
        if not isinstance(state, TurnRerankState) or (
            budget is not None and state.budget is not budget
        ):
            raise ValueError("rerank state must share its parent ledger")
        if turn and state.worker_cleanup is None:
            state.worker_cleanup = turn.worker_cleanup
        return state
    return TurnRerankState(
        budget or TurnBudget(TurnLimits()),
        worker_cleanup=turn.worker_cleanup if turn else None,
    )
