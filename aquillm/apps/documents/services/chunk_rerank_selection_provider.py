"""Known-capability local scorer for bounded final evidence comparison."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from time import monotonic

import requests

from apps.documents.services import rag_cache
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

    def score_budgeted_pair(self, pair, timeout_seconds, *, budget, phase):
        from .chunk_rerank_window_http import score_budgeted_pair

        return score_budgeted_pair(
            self, pair, timeout_seconds, budget=budget, phase=phase, post=requests.post
        )

    def _post(self, payload: dict, budget: float):
        remaining = min(self.timeout, budget, self.deadline - self.clock())
        if remaining <= 0:
            return None
        try:
            return requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=remaining,
            )
        except requests.RequestException:
            return None

    def score_pair(self, pair: Pair, timeout_seconds: float):
        if self.scoring_kind != "pointwise":
            return None

        def payload(prepared: Pair) -> dict:
            return {
                "model": self.model_name,
                "text_1": prepared[0],
                "text_2": (
                    [prepared[1]]
                    if self.shape == "score_batch_text_pairs"
                    else prepared[1]
                ),
                "truncate_prompt_tokens": self.pair_limit,
                "truncation_side": "right",
            }

        response = self._post(payload(pair), timeout_seconds)
        successful = pair
        if response is not None and response.status_code == 400 and pair[1]:
            adaptive = min(
                max(0, self.pair_limit - 2),
                max(self.reserve + 256, self.pair_limit // 2),
            )
            stricter = min(
                max(0, self.pair_limit - 2),
                max(adaptive + 256, (self.pair_limit * 3) // 4),
            )
            retry_pair = trim_rerank_pair(pair[0], pair[1], self.pair_limit, stricter)
            if retry_pair != pair and self.clock() < self.deadline:
                response = self._post(payload(retry_pair), timeout_seconds)
                successful = retry_pair
        if response is None or response.status_code >= 400:
            return None
        try:
            body = response.json()
            if self.shape == "score_batch_text_pairs":
                items = (
                    body.get("data", body.get("results"))
                    if isinstance(body, dict)
                    else None
                )
                indexed = parse_score_results(body)
                if not isinstance(items, list) or len(items) != 1 or len(indexed) != 1:
                    return None
                index, value = indexed[0]
                if type(index) is not int or index != 0:
                    return None
            else:
                value = parse_single_score(body)
        except (TypeError, ValueError, requests.RequestException):
            return None
        return (float(value), successful) if isfinite(float(value)) else None

    def score_pool(self, pairs: tuple[Pair, ...], timeout_seconds: float):
        if (
            self.scoring_kind != "listwise"
            or not pairs
            or len({query for query, _ in pairs}) != 1
        ):
            return None
        documents = [document for _, document in pairs]
        payload = {
            "model": self.model_name,
            "query": pairs[0][0],
            "documents": documents,
            "top_n": len(pairs),
        }
        response = self._post(payload, timeout_seconds)
        if response is not None and response.status_code == 400:
            payload["documents"] = [{"text": document} for document in documents]
            response = self._post(payload, timeout_seconds)
        if response is None or response.status_code >= 400:
            return None
        try:
            indexed = parse_score_results(response.json())
        except (TypeError, ValueError, requests.RequestException):
            return None
        if len(indexed) != len(pairs) or {index for index, _ in indexed} != set(
            range(len(pairs))
        ):
            return None
        by_index = dict(indexed)
        return tuple((by_index[index], pair) for index, pair in enumerate(pairs))


def _legacy_selection_scorer(
    *, deadline: float, clock: Callable[[], float] = monotonic, turn_budget=None
) -> LocalSelectionScorer | None:
    """Use only a known local capability; no endpoint discovery in this path."""
    if rerank_provider() not in ("auto", "local", "vllm"):
        return None
    base_v1, model_name = rerank_base_url(), rerank_model()
    capability = rag_cache.get_cached_rerank_capability(
        base_v1, model_name, **({"budget": turn_budget} if turn_budget else {})
    )
    if not capability:
        return None
    root = base_v1[:-3] if base_v1.endswith("/v1") else base_v1
    shape, endpoint = capability["shape"], capability["endpoint"]
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
