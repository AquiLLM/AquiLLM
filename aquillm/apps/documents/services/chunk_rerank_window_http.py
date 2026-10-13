"""Verified window scoring shares the same weighted permits and failure latch."""

from .chunk_rerank_window_scores import _AttemptBudget
from .rerank_request_lifetime import rerank_state_for
from .rerank_transport_policy import finite_protocol_scores


def score_budgeted_pair(scorer, pair, timeout_seconds, *, budget, phase, post):
    if (
        scorer.scoring_kind != "pointwise"
        or budget is None
        or not scorer.can_score_protocol()
    ):
        return None
    ledger = budget.budget if isinstance(budget, _AttemptBudget) else budget
    state = rerank_state_for(ledger)
    on_admitted = (
        budget.record_admission if isinstance(budget, _AttemptBudget) else None
    )
    remaining = min(scorer.timeout, timeout_seconds, scorer.deadline - scorer.clock())
    if remaining <= 0:
        return None

    def send(timeout):
        if not scorer.can_score_protocol():
            return None
        response = post(
            scorer.endpoint,
            headers=scorer.headers,
            json={
                "model": scorer.model_name,
                "text_1": pair[0],
                "text_2": (
                    [pair[1]] if scorer.shape == "score_batch_text_pairs" else pair[1]
                ),
                "max_tokens_per_query": 0,
                "max_tokens_per_doc": 0,
            },
            timeout=min(timeout, remaining),
            allow_redirects=False,
        )
        if (
            response.status_code != 200
            or scorer.clock() >= scorer.deadline
            or getattr(response, "url", scorer.endpoint) != scorer.endpoint
            or not scorer.can_score_protocol()
        ):
            return None
        body = response.json()
        counter = getattr(scorer, "verified_pair_counter", None)
        if counter is not None:
            if not isinstance(body, dict) or not isinstance(body.get("usage"), dict):
                return None
            work = (
                counter.input_codepoints(*pair)
                if hasattr(counter, "input_codepoints")
                else len(pair[0]) + len(pair[1])
            )
            if work and not budget.reserve_text(work, kind="tokenized"):
                return None
            expected = counter(*pair)
            if expected is None or body["usage"].get("prompt_tokens") != expected:
                return None
        values = finite_protocol_scores(
            body, payload_kind=scorer.shape, count=1, model_key=scorer.model_name
        )
        return (values[0], pair) if values is not None else None

    return state.run(1, send, phase=phase, timeout=remaining, on_admitted=on_admitted)
