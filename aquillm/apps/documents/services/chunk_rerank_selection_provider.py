"""Known-capability local scorer for bounded final evidence comparison."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from time import monotonic

import requests

from apps.documents.services import rag_cache as rag_cache
from apps.documents.services.chunk_rerank_budget import trim_rerank_pair
from apps.documents.services.chunk_rerank_config import (
    rerank_api_key,
    rerank_base_url,
    rerank_doc_char_limit,
    rerank_model,
    rerank_model_revision,
    rerank_pair_token_limit,
    rerank_provider,
    rerank_template_reserve_tokens,
    rerank_timeout_seconds,
    rerank_tokenizer,
    rerank_tokenizer_revision,
)
from apps.documents.services.chunk_rerank_parse import (
    parse_score_results,
    parse_single_score,
)
from apps.documents.services.chunk_rerank_results import fingerprint_text

type Pair = tuple[str, str]


def local_scorer_fingerprint(
    *,
    endpoint: str,
    shape: str,
    model_name: str,
    revision: str,
    char_limit: int,
    pair_limit: int,
    reserve: int,
) -> str:
    """Match the scored retrieval adapter's exact input/model signature."""
    policy_fp = fingerprint_text(
        f"scored-v2:cl100k:{char_limit}:{pair_limit}:{reserve}"
    )
    return fingerprint_text(
        f"local:{endpoint}:{shape}:{model_name}:{revision}:"
        f"{rerank_tokenizer()}:{rerank_tokenizer_revision()}:{policy_fp}"
    )


class LocalSelectionScorer:
    def __init__(
        self,
        *,
        endpoint: str,
        shape: str,
        model_name: str,
        revision: str,
        char_limit: int,
        pair_limit: int,
        reserve: int,
        timeout: float,
        deadline: float,
        clock: Callable[[], float] = monotonic,
        headers: dict[str, str] | None = None,
        turn_budget=None,
        protocol_binding=None,
    ) -> None:
        if shape not in (
            "score_single_text_pair",
            "score_batch_text_pairs",
            "rerank_documents",
        ):
            raise ValueError("unsupported scorer shape")
        self.endpoint = endpoint
        self.shape = shape
        self.model_name = model_name
        self.char_limit = char_limit
        self.pair_limit = pair_limit
        self.reserve = reserve
        self.timeout = timeout
        self.deadline = deadline
        self.clock = clock
        self.headers = headers or {"Content-Type": "application/json"}
        # Serving factories retain the immutable proof through actual I/O.
        # Unbound construction is also used by independently attested warm probes.
        self.protocol_binding = protocol_binding
        from lib.llm.turn_context import current_turn
        from lib.retrieval.turn_budget import TurnBudget, TurnLimits

        from .rerank_request_lifetime import rerank_state_for

        turn = current_turn()
        budget = (
            turn_budget
            or (turn.budget if turn else None)
            or TurnBudget(TurnLimits(), clock=clock)
        )
        self.rerank_state = rerank_state_for(budget)
        self.scoring_kind = "listwise" if shape == "rerank_documents" else "pointwise"
        self.scorer_fingerprint = local_scorer_fingerprint(
            endpoint=endpoint,
            shape=shape,
            model_name=model_name,
            revision=revision,
            char_limit=char_limit,
            pair_limit=pair_limit,
            reserve=reserve,
        )

    def prepare_pair(self, query: str, chunk) -> Pair:
        return trim_rerank_pair(
            query, chunk.content[: self.char_limit], self.pair_limit, self.reserve
        )

    def prepare_emitted_pair(self, query: str, emitted_text: str) -> Pair | None:
        """Refuse final evidence that this provider would silently truncate."""
        pair = trim_rerank_pair(
            query, emitted_text[: self.char_limit], self.pair_limit, self.reserve
        )
        return pair if pair == (query, emitted_text) else None

    def score_budgeted_pair(self, pair, timeout_seconds, *, budget, phase):
        from .chunk_rerank_window_http import score_budgeted_pair

        return score_budgeted_pair(
            self, pair, timeout_seconds, budget=budget, phase=phase, post=requests.post
        )

    def can_score_protocol(self):
        binding = self.protocol_binding
        return binding is None or (
            binding.endpoint == self.endpoint
            and binding.payload_kind == self.shape
            and binding.live(self.model_name)
        )

    def _post(self, payload: dict, budget: float):
        remaining = min(self.timeout, budget, self.deadline - self.clock())
        if remaining <= 0 or not self.can_score_protocol():
            return None
        from .rerank_transport_policy import finite_protocol_scores

        count = len(payload["documents"]) if "documents" in payload else 1

        def send(timeout):
            if not self.can_score_protocol():
                return None
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=min(timeout, remaining),
                allow_redirects=False,
            )
            if (
                response.status_code != 200
                or self.clock() >= self.deadline
                or getattr(response, "url", self.endpoint) != self.endpoint
                or not self.can_score_protocol()
            ):
                return None
            if (
                finite_protocol_scores(
                    response.json(),
                    payload_kind=self.shape,
                    count=count,
                    model_key=self.model_name,
                )
                is None
            ):
                return None
            return response

        return self.rerank_state.run(count, send, phase="final", timeout=remaining)

    def score_pair(self, pair: Pair, timeout_seconds: float):
        from .chunk_rerank_selection_scores import score_pair

        return score_pair(self, pair, timeout_seconds)

    def score_pool(self, pairs: tuple[Pair, ...], timeout_seconds: float):
        from .chunk_rerank_selection_scores import score_pool

        return score_pool(self, pairs, timeout_seconds)


def _legacy_selection_scorer(
    *, deadline: float, clock: Callable[[], float] = monotonic, turn_budget=None
) -> LocalSelectionScorer | None:
    """Use only a known local capability; no endpoint discovery in this path."""
    if rerank_provider() not in ("auto", "local", "vllm"):
        return None
    base_v1, model_name = rerank_base_url(), rerank_model()
    from .rerank_transport_policy import resolve_rerank_protocol

    binding = resolve_rerank_protocol(base_v1, model_name, budget=turn_budget)
    if not binding:
        return None
    root = base_v1[:-3] if base_v1.endswith("/v1") else base_v1
    shape, endpoint = binding.payload_kind, binding.endpoint
    allowed = (
        {f"{root}/rerank", f"{root}/v2/rerank", f"{base_v1}/rerank"}
        if shape == "rerank_documents"
        else {f"{root}/score", f"{base_v1}/score"}
    )
    if endpoint not in allowed or clock() >= deadline:
        return None
    headers = {"Content-Type": "application/json"}
    api_key = rerank_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return LocalSelectionScorer(
        endpoint=endpoint,
        shape=shape,
        model_name=model_name,
        revision=rerank_model_revision(),
        char_limit=rerank_doc_char_limit(),
        pair_limit=rerank_pair_token_limit(),
        reserve=rerank_template_reserve_tokens(),
        timeout=rerank_timeout_seconds(),
        deadline=deadline,
        clock=clock,
        headers=headers,
        turn_budget=turn_budget,
        protocol_binding=binding,
    )


def current_selection_scorer(
    *, deadline, clock=monotonic, turn_budget=None, windowed=None, pair_counter=None
):
    from .chunk_rerank_config import rerank_text_mode
    from .chunk_rerank_pair_capability import registered_pair_counter
    from .chunk_rerank_window_adapter import WindowSelectionScorer, unknown_pair_count

    active = rerank_text_mode() == "windowed" if windowed is None else windowed
    provider = None
    if active:
        from .pair_worker_lifecycle import verified_canonical_provider

        provider = verified_canonical_provider(deadline)
    if provider is None:
        provider = _legacy_selection_scorer(
            deadline=deadline, clock=clock, turn_budget=turn_budget
        )
    if not active:
        return provider
    verified = registered_pair_counter(provider)
    if verified is not None:
        provider.verified_pair_counter = verified
    return WindowSelectionScorer(
        provider,
        budget=turn_budget,
        deadline=deadline,
        clock=clock,
        pair_counter=pair_counter or verified or unknown_pair_count,
        cache_enabled=verified is not None,
    )


__all__ = [
    "LocalSelectionScorer",
    "current_selection_scorer",
    "local_scorer_fingerprint",
]
