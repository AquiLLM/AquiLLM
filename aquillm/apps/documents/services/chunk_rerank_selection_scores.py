"""Payload and response handling for the existing local selection scorer."""

from __future__ import annotations

def score_pair(self, pair, timeout_seconds: float):
    from . import chunk_rerank_selection_provider as provider

    if self.scoring_kind != "pointwise":
        return None

    def payload(prepared) -> dict:
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
            indexed = provider.parse_score_results(body)
            if not isinstance(items, list) or len(items) != 1 or len(indexed) != 1:
                return None
            index, value = indexed[0]
            if type(index) is not int or index != 0:
                return None
        else:
            value = provider.parse_single_score(body)
    except (TypeError, ValueError, provider.requests.RequestException):
        return None
    return (float(value), successful) if provider.isfinite(float(value)) else None


def score_pool(self, pairs, timeout_seconds: float):
    from . import chunk_rerank_selection_provider as provider

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
    width = min(6, self.rerank_state.budget.limits.in_flight_pairs)
    if width <= 0:
        return None
    all_indexed = []
    for start in range(0, len(pairs), width):
        payload["documents"] = documents[start : start + width]
        payload["top_n"] = len(payload["documents"])
        response = self._post(dict(payload), timeout_seconds)
        if response is None:
            return None
        all_indexed.extend(
            (start + i, value) for i, value in provider.parse_score_results(response.json())
        )
    if response is None or response.status_code >= 400:
        return None
    try:
        indexed = all_indexed
    except (TypeError, ValueError, provider.requests.RequestException):
        return None
    if len(indexed) != len(pairs) or {index for index, _ in indexed} != set(
        range(len(pairs))
    ):
        return None
    by_index = dict(indexed)
    return tuple((by_index[index], pair) for index, pair in enumerate(pairs))
