"""Default-off ASGI worker capability renewal; no request-time warm inference."""

import asyncio
import json  # noqa: F401 - attestation module uses the historical patch hooks
import os
import threading
from pathlib import Path  # noqa: F401
from time import monotonic, time  # noqa: F401

from lib.retrieval.turn_budget import TurnBudget, TurnLimits

from . import chunk_rerank_config as config
from .chunk_rerank_pair_capability import (
    _prepare_worker_pair_capability,
    _valid_identity,  # noqa: F401
    invalidate_pair_capability,
    registered_pair_counter,
)


def enabled():
    return os.getenv("RAG_PAIR_CAPABILITY_WARM_ENABLED", "0") == "1" and (
        config.rerank_text_mode() == "windowed" or config.rerank_shadow_scoring()
    )


def canonical_provider(*, deadline):
    from .chunk_rerank_selection_provider import LocalSelectionScorer

    if config.rerank_provider() not in ("auto", "local", "vllm"):
        return None
    base = config.rerank_base_url()
    return LocalSelectionScorer(
        endpoint=base.removesuffix("/v1") + "/score",
        shape="score_single_text_pair",
        model_name=config.rerank_model(),
        revision=config.rerank_model_revision(),
        char_limit=config.rerank_doc_char_limit(),
        pair_limit=config.rerank_pair_token_limit(),
        reserve=config.rerank_template_reserve_tokens(),
        timeout=config.rerank_timeout_seconds(),
        deadline=deadline,
        headers=config.rerank_headers(),
    )


def load_attested_attempt():
    from .pair_worker_attestation import load_attested_attempt as load

    return load()


class PairCapabilityController:
    """One daemon slot, no queue, at most three pairs/15 seconds per attempt.

    Tokenizer/filesystem/HTTP work is entirely inside that slot. A stuck native
    operation may outlive the caller, but never creates more threads or publishes
    through a closed ledger. Shutdown does not join an unbounded native call.
    """

    def __init__(
        self,
        *,
        load=load_attested_attempt,
        initialize=_prepare_worker_pair_capability,
        timeout=15,
    ):
        self.load, self.initialize = load, initialize
        self.timeout = min(15, max(0.001, timeout))
        self.thread = self.budget = self.task = None
        self.generation = 0
        self.stopping = False
        self.protocol_binding = None
        self._lock = threading.RLock()
        self._counter_revocation = None

    def invalidate(self):
        with self._lock:
            self.generation += 1
            if self.budget:
                self.budget.close("worker_invalidated")
            self._revoke()

    def _revoke(self):
        if self._counter_revocation is not None:
            self._counter_revocation.set()
            self._counter_revocation = None
        invalidate_pair_capability()
        if self.protocol_binding is not None:
            from .pair_worker_refresh import revoke_local_protocol

            base, binding = self.protocol_binding
            revoke_local_protocol(base, binding)
            self.protocol_binding = None

    async def attempt(self):
        if self.stopping or (self.thread and self.thread.is_alive()):
            self.invalidate()
            return None
        with self._lock:
            self.generation += 1
            generation = self.generation
            if self.budget:
                self.budget.close("worker_refresh")
        self.budget = budget = TurnBudget(
            TurnLimits(
                acquisition_pairs=3,
                final_pairs=0,
                in_flight_pairs=1,
                retrieval_ms=max(1, int(self.timeout * 1000)),
                final_scoring_ms=0,
            )
        )
        result, done = [], threading.Event()

        def work():
            try:
                from .pair_worker_refresh import prepare_attempt

                counter = prepare_attempt(self, budget, generation)
                if counter is not None:
                    result.append(counter)
            except Exception:
                # Unknown capability is the specified outcome of warm failure.
                pass
            finally:
                done.set()

        self.thread = threading.Thread(target=work, daemon=True, name="pair-capability")
        self.thread.start()
        deadline = monotonic() + self.timeout
        try:
            while not done.is_set() and monotonic() < deadline:
                await asyncio.sleep(min(0.01, self.timeout))
            if not done.is_set() or not result or generation != self.generation:
                self.invalidate()
                return None
            return result[0]
        except asyncio.CancelledError:
            self.invalidate()
            raise

    async def run(self, initial=None, *, attempted=False):
        while not self.stopping and enabled():
            counter = initial if attempted else await self.attempt()
            attempted = False
            initial = None
            renewal = (
                max(monotonic() + 30, counter.expires - 60)
                if counter
                else monotonic() + 30
            )
            while not self.stopping and enabled() and monotonic() < renewal:
                await asyncio.sleep(1)
                if counter is None:
                    continue
                # Expired/withdrawn/replaced attestation invalidates immediately
                # on the next bounded background check, never on request inference.
                current = await self.check_attestation()
                if not current or (counter and current["identity"] != self.identity):
                    self.invalidate()
                    break
        self.invalidate()

    async def check_attestation(self):
        if self.thread and self.thread.is_alive():
            return None
        result, done = [], threading.Event()

        def read():
            try:
                result.append(self.load())
            except Exception:
                pass
            finally:
                done.set()

        self.thread = threading.Thread(
            target=read, daemon=True, name="pair-attestation"
        )
        self.thread.start()
        deadline = monotonic() + min(0.2, self.timeout)
        while not done.is_set() and monotonic() < deadline:
            await asyncio.sleep(0.01)
        return result[0] if done.is_set() and result else None

    async def stop(self):
        self.stopping = True
        self.invalidate()
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


class PairCapabilityLifespan:
    def __init__(self, app):
        self.app, self.controller = app, None
        self.protocol_controller = None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "lifespan":
            return await self.app(scope, receive, send)
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                if protocol_warm_enabled() and self.protocol_controller is None:
                    self.protocol_controller = RerankProtocolController()
                    self.protocol_controller.task = asyncio.create_task(
                        self.protocol_controller.run()
                    )
                if enabled() and self.controller is None:
                    self.controller = PairCapabilityController()
                    self.controller.task = asyncio.create_task(self.controller.run())
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self.protocol_controller:
                    await self.protocol_controller.stop()
                if self.controller:
                    await self.controller.stop()
                await send({"type": "lifespan.shutdown.complete"})
                return


def verified_canonical_provider(deadline):
    provider = canonical_provider(deadline=deadline)
    return provider if registered_pair_counter(provider) is not None else None


def protocol_warm_enabled():
    return os.getenv(
        "RAG_RERANK_PROTOCOL_WARM_ENABLED", "0"
    ) == "1" and config.rerank_provider() in ("auto", "local", "vllm")


# Keep one controller class and the existing public import/patch path.
from .pair_worker_protocol import RerankProtocolController  # noqa: E402

RerankProtocolController.__module__ = __name__
