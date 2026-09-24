"""Bounded, exact source slices; estimates never establish model coverage."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import monotonic

from lib.retrieval import SourceEvidence, SourceSpan, TurnBudget

from .chunk_rerank_results import fingerprint_pair, fingerprint_text


@dataclass(frozen=True)
class WindowPlan:
    query: str
    source: SourceEvidence
    windows: tuple[SourceSpan, ...]
    tokenizer_identity: str
    template_identity: str
    preparation_coverage: str
    reason: str | None = None
    source_revision: str = ""

    @property
    def required_window_ids(self) -> tuple[str, ...]:
        return tuple(
            fingerprint_text(
                json.dumps(
                    (
                        self.source.source_fingerprint,
                        self.source.chunk_id,
                        w.start,
                        w.end,
                        fingerprint_pair(self.query, w.text),
                    )
                )
            )
            for w in self.windows
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint_text(
            json.dumps(
                (
                    "window-max-v1",
                    fingerprint_text(self.query),
                    self.source.source_fingerprint,
                    self.source.document_id,
                    self.source.chunk_number,
                    fingerprint_text(self.source.text),
                    self.tokenizer_identity,
                    self.template_identity,
                    self.required_window_ids,
                    self.preparation_coverage,
                    self.source_revision,
                )
            )
        )


class PreparationStopped(ValueError):
    pass


def prepare_source_windows(
    query,
    source,
    *,
    pair_counter,
    pair_limit=1024,
    budget=None,
    deadline=None,
    clock=monotonic,
):
    """Count complete pairs first; retain every source code point in exact slices.

    ``pair_counter`` must count the *complete model template* or return None.
    The byte-based fallback is deliberately an estimate, never a verification.
    Every actual counter invocation is charged; memoized repeated slices are free.
    """
    if not isinstance(source, SourceEvidence) or not isinstance(query, str):
        raise ValueError("exact query and SourceEvidence required")
    if type(pair_limit) is not int or not 1 <= pair_limit <= 1024:
        raise ValueError("pair limit exceeds pilot context")
    if budget is not None and not isinstance(budget, TurnBudget):
        raise ValueError("shared TurnBudget required")
    windows = []
    unknown = False
    memo = {}

    def check_preparation_open():
        if deadline is not None and clock() >= deadline:
            raise PreparationStopped("preparation_deadline")
        if budget is not None and not budget.can_publish():
            raise PreparationStopped("preparation_budget")
        if deadline is not None and clock() >= deadline:
            raise PreparationStopped("preparation_deadline")

    def counted(text):
        nonlocal unknown
        check_preparation_open()
        if text in memo:
            return memo[text]
        work_counter = getattr(pair_counter, "input_codepoints", None)
        work = work_counter(query, text) if work_counter else len(query) + len(text)
        check_preparation_open()
        if (
            budget is not None
            and work
            and not budget.reserve_text(work, kind="tokenized")
        ):
            raise PreparationStopped("preparation_budget")
        check_preparation_open()
        value = pair_counter(query, text)
        check_preparation_open()
        if value is None:
            unknown = True
            value = len(query.encode("utf-8")) + len(text.encode("utf-8")) + 256
        if type(value) is not int or value < 0:
            raise ValueError("invalid complete pair count")
        memo[text] = value
        return value

    def finish(coverage, reason=None):
        from lib.evidence_observation import active, publish

        if active():
            publish(
                "window_preparation",
                {
                    "chunk_id": source.chunk_id,
                    "fingerprint": source.source_fingerprint,
                    "full_pair_tokens": memo.get(source.text) if not unknown else None,
                    "query_template_tokens": memo.get("") if not unknown else None,
                    "pair_limit": pair_limit,
                    "windows": [(s.start, s.end) for s in windows],
                    "tokenizer": getattr(pair_counter, "tokenizer_identity", "unknown"),
                    "template": getattr(pair_counter, "template_identity", "unknown"),
                    "coverage": coverage,
                    "reason": reason,
                },
            )
        return WindowPlan(
            query,
            source,
            tuple(windows),
            getattr(pair_counter, "tokenizer_identity", "unknown"),
            getattr(pair_counter, "template_identity", "unknown"),
            coverage,
            reason,
        )

    def add(start, end):
        check_preparation_open()
        windows.append(
            SourceSpan(
                source.chunk_id,
                source.source_fingerprint,
                start,
                end,
                source.text[start:end],
            )
        )

    try:
        check_preparation_open()
        if len(query) + len(source.text) > 250_000:
            return finish("partial", "source_limit")
        full_count = counted(source.text)
        if full_count <= pair_limit:
            if source.text:
                add(0, len(source.text))
            return finish("unknown" if unknown else "complete")
        base = counted("")
        if base >= pair_limit or not source.text:
            return finish("partial", "query_capacity")
        # Compute boundaries once; never decode token prefixes back into source.
        check_preparation_open()
        boundaries = [m.end() for m in re.finditer(r"\n\s*\n|[.!?]\s+", source.text)]
        start = 0
        while start < len(source.text):
            check_preparation_open()
            if len(windows) >= 135:
                return finish("partial", "window_limit")
            low, high = start, len(source.text)
            while low < high:
                mid = (low + high + 1) // 2
                if counted(source.text[start:mid]) <= pair_limit:
                    low = mid
                else:
                    high = mid - 1
            end = low
            if end <= start:
                return finish("partial", "query_capacity")
            if end < len(source.text):
                preferred = [
                    b for b in boundaries if start + (end - start) // 2 <= b <= end
                ]
                if preferred:
                    end = preferred[-1]
            # Recheck because tokenization need not be strictly monotone.
            if counted(source.text[start:end]) > pair_limit:
                return finish("partial", "boundary_capacity")
            add(start, end)
            if end == len(source.text):
                break
            overlap = min(64, max(0, (counted(source.text[start:end]) - base) // 4))
            low, high = 0, end - start - 1
            while low < high:
                mid = (low + high + 1) // 2
                if counted(source.text[end - mid : end]) - base <= overlap:
                    low = mid
                else:
                    high = mid - 1
            start = end - low
        return finish("unknown" if unknown else "complete")
    except PreparationStopped as stopped:
        return finish("partial", str(stopped))
