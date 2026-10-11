"""Protocol startup and renewal using the original lifespan module hooks."""

class RerankProtocolController:
    """One startup/renewal slot; three attempts/15s, no serving-driven discovery."""

    def __init__(self):
        self.thread = self.task = self.budget = self.binding = None
        self.stopping = False
        self.base = self.model = None
        from threading import RLock

        self._binding_lock = RLock()
        self._binding_base = None
        self._attempt_active = False

    def _record_binding(self, base, binding):
        from .rerank_transport_policy import revoke_rerank_protocol

        with self._binding_lock:
            previous = self._binding_base, self.binding
            self._binding_base, self.binding = base, binding
        if previous[1] is not None and previous != (base, binding):
            revoke_rerank_protocol(*previous)

    def _release_binding(self):
        from .rerank_transport_policy import revoke_rerank_protocol

        with self._binding_lock:
            base, binding = self._binding_base, self.binding
            self._binding_base = self.binding = None
        if binding is not None:
            revoke_rerank_protocol(base, binding)

    async def attempt(self):
        if self.stopping or self._attempt_active or (self.thread and self.thread.is_alive()):
            return None
        self._attempt_active = True
        try:
            return await self._attempt()
        finally:
            self._attempt_active = False

    async def _attempt(self):
        from . import pair_worker_lifecycle as worker
        from .rerank_transport_policy import (
            discover_rerank_protocol,
            resolve_rerank_protocol,
        )

        try:
            timeout = min(
                15,
                max(
                    0.001,
                    float(worker.os.getenv("RAG_RERANK_PROTOCOL_WARM_TIMEOUT_SECONDS", "15")),
                ),
            )
            attempts = min(
                3, max(1, int(worker.os.getenv("RAG_RERANK_PROTOCOL_WARM_ATTEMPTS", "3")))
            )
        except ValueError:
            timeout, attempts = 15, 3
        base, model = worker.config.rerank_base_url(), worker.config.rerank_model()
        if (self.base, self.model) != (base, model):
            self._release_binding()
        self.base, self.model = base, model
        cached = resolve_rerank_protocol(base, model)
        if cached is not None and cached.expires_at - worker.time() > 60:
            self._record_binding(base, cached)
            return cached
        self.budget = budget = worker.TurnBudget(
            worker.TurnLimits(
                acquisition_pairs=attempts,
                final_pairs=0,
                in_flight_pairs=1,
                retrieval_ms=max(1, int(timeout * 1000)),
                final_scoring_ms=0,
            )
        )
        deadline = worker.monotonic() + timeout
        results, done = [], worker.threading.Event()

        def work():
            try:
                binding = discover_rerank_protocol(
                    base,
                    model,
                    deadline=deadline,
                    max_attempts=attempts,
                    headers=worker.config.rerank_headers(),
                    budget=budget,
                    on_registered=lambda binding: self._record_binding(base, binding),
                )
                if binding and budget.can_publish():
                    results.append(binding)
            finally:
                done.set()

        self.thread = worker.threading.Thread(
            target=work, daemon=True, name="rerank-protocol-warm"
        )
        self.thread.start()
        try:
            while not done.is_set() and worker.monotonic() < deadline:
                await worker.asyncio.sleep(0.01)
            if not done.is_set() or self.stopping:
                budget.close("warm_deadline")
                self._release_binding()
                return None
            return results[0] if results else None
        except worker.asyncio.CancelledError:
            budget.close("warm_cancelled")
            self._release_binding()
            raise

    async def run(self):
        from . import pair_worker_lifecycle as worker

        # Per-worker jitter avoids simultaneous fleet renewal bursts; all actual
        # probes still share the process-wide weighted admission mechanism.
        import random

        try:
            while not self.stopping and worker.protocol_warm_enabled():
                binding = await self.attempt()
                delay = max(30, binding.expires_at - worker.time() - 60) if binding else 30
                await worker.asyncio.sleep(delay + random.uniform(0, 5))
        finally:
            self.revoke()

    def revoke(self):
        if self.budget:
            self.budget.close("worker_stopped")
        self._release_binding()

    async def stop(self):
        from . import pair_worker_lifecycle as worker

        self.stopping = True
        self.revoke()
        if self.task:
            self.task.cancel()
            await worker.asyncio.gather(self.task, return_exceptions=True)
