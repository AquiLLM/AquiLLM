"""Protocol startup and renewal using the original lifespan module hooks."""

class RerankProtocolController:
    """One startup/renewal slot; three attempts/15s, no serving-driven discovery."""

    def __init__(self):
        self.thread = self.task = self.budget = self.binding = None
        self.stopping = False
        self.base = self.model = None

    async def attempt(self):
        from . import pair_worker_lifecycle as worker
        from .rerank_transport_policy import (
            discover_rerank_protocol,
            resolve_rerank_protocol,
        )

        if self.stopping or (self.thread and self.thread.is_alive()):
            return None
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
        self.base, self.model = worker.config.rerank_base_url(), worker.config.rerank_model()
        cached = resolve_rerank_protocol(self.base, self.model)
        if cached is not None and cached.expires_at - worker.time() > 60:
            self.binding = cached
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
                    self.base,
                    self.model,
                    deadline=deadline,
                    max_attempts=attempts,
                    headers=worker.config.rerank_headers(),
                    budget=budget,
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
                return None
            self.binding = results[0] if results else None
            return self.binding
        except worker.asyncio.CancelledError:
            budget.close("warm_cancelled")
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
        from .rerank_transport_policy import revoke_rerank_protocol

        if self.budget:
            self.budget.close("worker_stopped")
        if self.binding:
            revoke_rerank_protocol(self.base, self.binding)
        self.binding = None

    async def stop(self):
        from . import pair_worker_lifecycle as worker

        self.stopping = True
        self.revoke()
        if self.task:
            self.task.cancel()
            await worker.asyncio.gather(self.task, return_exceptions=True)
