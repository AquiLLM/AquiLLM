"""Default-off ASGI worker capability renewal; no request-time warm inference."""

import asyncio
import json
import os
import threading
from pathlib import Path
from time import monotonic, time

from lib.retrieval.turn_budget import TurnBudget, TurnLimits

from . import chunk_rerank_config as config
from .chunk_rerank_pair_capability import (
    _valid_identity,
    initialize_worker_pair_capability,
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
    """Consume an operator-verified, short-lived, locally mounted attestation.

    The worker has no Docker privilege. This file is deployment evidence, not
    inferred from /models, environment names or the successful warm probes.
    """
    path = os.getenv("RAG_PAIR_CAPABILITY_ATTESTATION")
    if not path:
        return None
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            return None
        data = json.loads(raw)
        provider = canonical_provider(deadline=monotonic() + 15)
        template = data["template"]
        identity = data["identity"]
        if not (
            provider
            and data["verified_by"]
            and data["verification_record"]
            and data["verified_at"] <= time() < data["expires_at"]
            and 0 < data["expires_at"] - data["verified_at"] <= 300
            and data["endpoint"] == provider.endpoint
            and data["served_model"] == provider.model_name
            and data["pair_context"] == provider.pair_limit == 1024
            and data["tokenizer_name"] == config.rerank_tokenizer()
            and identity["model_revision"] == config.rerank_model_revision()
            and identity["tokenizer_revision"] == config.rerank_tokenizer_revision()
            and identity["code_revision"] == config.rerank_code_revision()
            and _valid_identity(identity, template)
        ):
            return None
        return dict(
            provider=provider,
            tokenizer_name=data["tokenizer_name"],
            template=template,
            identity=identity,
            attestation_expires=data["expires_at"],
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


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
        initialize=initialize_worker_pair_capability,
        timeout=15,
    ):
        self.load, self.initialize = load, initialize
        self.timeout = min(15, max(0.001, timeout))
        self.thread = self.budget = self.task = None
        self.generation = 0
        self.stopping = False

    def invalidate(self):
        self.generation += 1
        if self.budget:
            self.budget.close("worker_invalidated")
        invalidate_pair_capability()

    async def attempt(self):
        self.invalidate()
        if self.stopping or (self.thread and self.thread.is_alive()):
            return None
        generation = self.generation
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
                args = self.load()
                if args and budget.can_publish():
                    expiry = args.pop("attestation_expires", time() + 300)
                    with budget._lock:
                        budget._deadline = min(
                            budget._deadline, monotonic() + max(0, expiry - time())
                        )
                    self.identity = args.get("identity")
                    counter = self.initialize(**args, budget=budget)
                    if (
                        counter
                        and generation == self.generation
                        and budget.can_publish()
                    ):
                        # Never outlive the independent deployment proof.
                        object.__setattr__(
                            counter,
                            "expires",
                            min(counter.expires, monotonic() + max(0, expiry - time())),
                        )
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

    async def __call__(self, scope, receive, send):
        if scope["type"] != "lifespan":
            return await self.app(scope, receive, send)
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                if enabled() and self.controller is None:
                    self.controller = PairCapabilityController()
                    self.controller.task = asyncio.create_task(self.controller.run())
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self.controller:
                    await self.controller.stop()
                await send({"type": "lifespan.shutdown.complete"})
                return


def verified_canonical_provider(deadline):
    provider = canonical_provider(deadline=deadline)
    return provider if registered_pair_counter(provider) is not None else None
