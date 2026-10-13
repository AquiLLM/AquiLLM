"""Only the verified protocol is dispatched; failed work closes admission."""

from concurrent.futures import FIRST_COMPLETED, wait
from time import time

import requests

from .chunk_rerank_config import rerank_score_concurrency
from .chunk_rerank_results import order_scored_pairs
from .rerank_request_lifetime import rerank_state_for
from .rerank_transport_policy import (
    RerankProtocolV1,
    finite_protocol_scores,
    protocol_payload,
)


def score_bound_pairs(
    binding,
    pairs,
    *,
    state,
    headers,
    timeout,
    pair_limit=None,
    phase="acquisition",
    max_workers=6,
):
    """Sequential weighted batches or at most six admitted single-pair workers."""
    pairs = tuple(pairs)
    if not pairs or not state.can_admit() or not binding.live(binding.model_key):
        return None

    def send(batch, remaining):
        if not binding.live(binding.model_key):
            return None
        response = requests.post(
            binding.endpoint,
            headers=headers,
            json=protocol_payload(
                binding.payload_kind, binding.model_key, batch, pair_limit=pair_limit
            ),
            timeout=min(timeout, remaining),
            allow_redirects=False,
        )
        if (
            response.status_code != 200
            or getattr(response, "url", binding.endpoint) != binding.endpoint
            or not binding.live(binding.model_key)
        ):
            return None
        return finite_protocol_scores(
            response.json(),
            payload_kind=binding.payload_kind,
            count=len(batch),
            model_key=binding.model_key,
        )

    width = min(6, state.budget.limits.in_flight_pairs, max(1, max_workers))
    if width <= 0:
        return None
    if binding.payload_kind != "score_single_text_pair":
        values = []
        for start in range(0, len(pairs), width):
            batch = pairs[start : start + width]
            result = state.run(
                len(batch),
                lambda remaining: send(batch, remaining),
                phase=phase,
                timeout=timeout,
            )
            if result is None:
                return None
            values.extend(result)
        return tuple(values) if state.can_admit() else None

    pending, completed, next_index = {}, {}, 0
    while next_index < len(pairs) or pending:
        while next_index < len(pairs) and len(pending) < width and state.can_admit():
            index = next_index
            pair = pairs[index]
            future = state.submit(
                1,
                lambda remaining, pair=pair: send((pair,), remaining),
                phase=phase,
                timeout=timeout,
            )
            if future is None:
                break
            pending[future] = index
            next_index += 1
        if not pending or not state.can_admit():
            return None
        done, _ = wait(
            pending,
            timeout=state.budget.scoring_remaining_ms(phase) / 1000,
            return_when=FIRST_COMPLETED,
        )
        if not done:
            state.fail()
            return None
        for future in done:
            result = future.result()
            index = pending.pop(future)
            if result is None:
                state.fail()
                return None
            completed[index] = result[0]
        if next_index < len(pairs) and not pending and not state.can_admit():
            return None
    if not state.can_admit() or state.budget.scoring_remaining_ms(phase) <= 0:
        return None
    return tuple(completed[i] for i in range(len(pairs)))


def _score_one_document(
    *,
    endpoint,
    index,
    query,
    document,
    headers,
    timeout,
    model_name,
    pair_token_limit,
    reserve_tokens,
    rerank_state=None,
    turn_budget=None,
):
    state = rerank_state_for(turn_budget, rerank_state)
    binding = RerankProtocolV1(
        endpoint, "score_single_text_pair", model_name, time() - 1, time() + 300
    )
    values = score_bound_pairs(
        binding,
        ((query, document),),
        state=state,
        headers=headers,
        timeout=timeout,
        pair_limit=pair_token_limit,
    )
    return (index, values[0], (query, document)) if values is not None else None


def _score_single_pool(
    endpoint,
    pairs,
    headers,
    timeout,
    model_name,
    pair_limit,
    reserve,
    *,
    turn_budget=None,
    rerank_state=None,
    max_workers=None,
):
    state = rerank_state_for(turn_budget, rerank_state)
    binding = RerankProtocolV1(
        endpoint, "score_single_text_pair", model_name, time() - 1, time() + 300
    )
    values = score_bound_pairs(
        binding,
        pairs,
        state=state,
        headers=headers,
        timeout=timeout,
        pair_limit=pair_limit,
        max_workers=(
            rerank_score_concurrency() if max_workers is None else max_workers
        ),
    )
    return (
        [(i, value, pairs[i]) for i, value in enumerate(values)]
        if values is not None
        else []
    )


def _complete_pairs(indexed, candidate_ids):
    try:
        return tuple(pk for pk, _ in order_scored_pairs(tuple(indexed), candidate_ids))
    except (TypeError, ValueError):
        return None
