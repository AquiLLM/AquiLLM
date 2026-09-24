"""Pointwise window HTTP: each attempt leases the shared turn allowance."""

from math import isfinite

import requests

from .chunk_rerank_parse import parse_score_results, parse_single_score


def score_budgeted_pair(scorer, pair, timeout_seconds, *, budget, phase, post):
    if scorer.scoring_kind != "pointwise" or budget is None:
        return None
    deadline = min(scorer.deadline, scorer.clock() + timeout_seconds)
    successful = pair
    for attempt in range(2):
        remaining = min(
            scorer.timeout,
            deadline - scorer.clock(),
            budget.scoring_remaining_ms(phase) / 1000,
        )
        if remaining <= 0 or not budget.start_pair(phase=phase):
            return None
        try:
            response = post(
                scorer.endpoint,
                headers=scorer.headers,
                json={
                    "model": scorer.model_name,
                    "text_1": successful[0],
                    "text_2": [successful[1]]
                    if scorer.shape == "score_batch_text_pairs"
                    else successful[1],
                    "max_tokens_per_query": 0,
                    "max_tokens_per_doc": 0,
                },
                timeout=remaining,
            )
        except requests.RequestException:
            return None
        finally:
            budget.finish_pair()
        if not budget.can_publish() or scorer.clock() >= deadline:
            return None
        if response.status_code == 400 and attempt == 0 and len(pair[1]) > 1:
            # Exact source prefix on retry; never shorten the primary question.
            # This successful input cannot stand in for the required full window.
            successful = (pair[0], pair[1][: len(pair[1]) // 2])
            continue
        if response.status_code >= 400:
            return None
        try:
            body = response.json()
            counter = getattr(scorer, "verified_pair_counter", None)
            if counter is not None:
                work = len(successful[0]) + len(successful[1])
                if hasattr(counter, "input_codepoints"):
                    work = counter.input_codepoints(*successful)
                if work and not budget.reserve_text(work, kind="tokenized"):
                    return None
                expected = counter(*successful)
                if (
                    expected is None
                    or body.get("usage", {}).get("prompt_tokens") != expected
                ):
                    return None
            if scorer.shape == "score_batch_text_pairs":
                indexed = parse_score_results(body)
                if len(indexed) != 1 or indexed[0][0] != 0:
                    return None
                value = indexed[0][1]
            else:
                value = parse_single_score(body)
            return (float(value), successful) if isfinite(float(value)) else None
        except (TypeError, ValueError, requests.RequestException):
            return None
    return None
