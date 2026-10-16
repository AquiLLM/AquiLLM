"""Window scores adapted to the existing private selection score contract."""

from dataclasses import replace
from time import monotonic

from lib.retrieval import SourceEvidence

from .chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    WindowCoverage,
    fingerprint_pair,
    fingerprint_pool,
    fingerprint_text,
    score_identity_for_chunk,
    validate_score_set,
)
from .chunk_rerank_score_cache import (
    get_window_result,
    set_window_result,
    window_cache_key,
)
from .chunk_rerank_window_dispatch import score_source_plans
from .chunk_rerank_window_scores import aggregate_window_scores
from .chunk_rerank_windows import PreparationStopped, prepare_source_windows
from .rerank_request_lifetime import rerank_state_for


class _ScoringCacheBudget:
    """Cache I/O stays bounded by the ledger; score pointers use its scoring latch."""

    def __init__(self, state):
        self.state, self.budget = state, state.budget

    def __getattr__(self, name):
        return getattr(self.budget, name)

    def can_publish(self):
        return self.state.can_admit()

    def can_start_optional(self, *args, **kwargs):
        return self.state.can_admit() and self.budget.can_start_optional(
            *args, **kwargs
        )

    def publish(self, callback):
        return self.state.publish(callback)


def unknown_pair_count(query, document):
    return None


def expected_score_fingerprint(scorer, query, chunk):
    if isinstance(scorer, WindowSelectionScorer):
        return scorer.plan(query, chunk).fingerprint
    return fingerprint_pair(*scorer.prepare_pair(query, chunk))


def compatible_window_score(scorer, score, query, chunk):
    if not isinstance(scorer, WindowSelectionScorer):
        return score.window_coverage is None
    plan = scorer.plan(query, chunk)
    pairs = tuple(fingerprint_pair(query, window.text) for window in plan.windows)
    expected = WindowCoverage(plan.required_window_ids, pairs, pairs, "complete")
    return plan.preparation_coverage == "complete" and score.window_coverage == expected


class WindowSelectionScorer:
    """Score stored text, including captions; coverage never includes pixels."""

    scoring_kind = "pointwise"

    def __init__(
        self,
        provider,
        *,
        budget,
        pair_counter=unknown_pair_count,
        scorer_identity=None,
        deadline=None,
        clock=monotonic,
        cache_enabled=False,
    ):
        self.provider, self.budget, self.pair_counter = provider, budget, pair_counter
        self.rerank_state = rerank_state_for(budget) if budget is not None else None
        self._cache_budget = (
            _ScoringCacheBudget(self.rerank_state) if self.rerank_state else None
        )
        self.clock = clock
        self.cache_enabled = cache_enabled
        self.deadline = (
            deadline
            if deadline is not None
            else clock() + (budget.remaining_ms() / 1000 if budget is not None else 0)
        )
        self.scorer_fingerprint = fingerprint_text(
            "window-max-v1:text-content-v1:"
            + (scorer_identity or getattr(provider, "scorer_fingerprint", "unknown"))
            + ":"
            + getattr(pair_counter, "identity", "unknown")
        )
        self._plans = {}
        self._scores = {}

    def plan(self, query, chunk):
        if not self._preparation_open():
            raise PreparationStopped("window preparation allowance exhausted")
        source = SourceEvidence(
            chunk.pk,
            str(chunk.doc_id),
            chunk.chunk_number,
            fingerprint_text(chunk.content),
            chunk.content,
        )
        # Bind opaque storage revision as well as content, when supplied by hydration.
        revision = getattr(chunk, "source_revision", "")
        key = (query, source, revision)
        if key not in self._plans:
            self._plans[key] = replace(
                prepare_source_windows(
                    query,
                    source,
                    pair_counter=self.pair_counter,
                    budget=self.budget,
                    deadline=self.deadline,
                    clock=self.clock,
                ),
                source_revision=revision,
            )
        return self._plans[key]

    def _preparation_open(self):
        if self.clock() >= self.deadline:
            return False
        ledger_open = self.rerank_state is None or self.rerank_state.can_admit()
        return ledger_open and self.clock() < self.deadline

    def score_windows(self, query, chunks, *, phase="final", on_submit=None):
        if self.budget is not None:
            self.deadline = min(
                self.deadline,
                self.clock() + self.budget.scoring_remaining_ms(phase) / 1000,
            )
        rows = tuple(chunks)
        order = tuple(row.pk for row in rows)
        pool = fingerprint_pool(())

        def envelope(scores=()):
            return RerankScoreSet(
                "v3-window",
                fingerprint_text(query),
                self.scorer_fingerprint,
                pool,
                "pointwise" if scores else "rank_only",
                "complete" if scores else "unavailable",
                order,
                tuple(scores),
            )

        if (
            self.budget is None
            or self.provider is None
            or self.pair_counter is unknown_pair_count
            or len(rows) > 45
            or len(set(order)) != len(order)
            or self.clock() >= self.deadline
            or not self.rerank_state.can_admit()
        ):
            return envelope()
        plans = []
        for row in rows:
            if not self._preparation_open():
                return envelope()
            try:
                plans.append(self.plan(query, row))
            except PreparationStopped:
                return envelope()
        if not self._preparation_open():
            return envelope()
        pool = fingerprint_pool(
            tuple(
                (row.pk, plan.source.source_fingerprint, plan.fingerprint)
                for row, plan in zip(rows, plans)
            )
        )
        slots = []
        for row, plan in zip(rows, plans):
            if self.clock() >= self.deadline or not self.rerank_state.can_admit():
                return envelope()
            key = (plan.fingerprint, getattr(row, "source_revision", ""))
            cached = self._scores.get(key)
            cache_key = window_cache_key(self.scorer_fingerprint, plan.fingerprint)
            if cached is None and self.cache_enabled:
                envelope_hit = get_window_result(cache_key, budget=self._cache_budget)
                if envelope_hit is not None:
                    try:
                        validate_score_set(
                            envelope_hit,
                            authorized_identities=(
                                score_identity_for_chunk(
                                    row, effective_pair_fingerprint=plan.fingerprint
                                ),
                            ),
                            expected_query_fingerprint=fingerprint_text(query),
                            expected_scorer_fingerprint=self.scorer_fingerprint,
                        )
                        if (
                            envelope_hit.status == "complete"
                            and envelope_hit.candidate_order == (row.pk,)
                            and compatible_window_score(
                                self, envelope_hit.scores[0], query, row
                            )
                        ):
                            cached = envelope_hit.scores[0]
                    except (ValueError, TypeError):
                        pass
            slots.append((row, plan, key, cache_key, cached))
        results = score_source_plans(
            {i: slot[1] for i, slot in enumerate(slots) if slot[4] is None},
            provider=self.provider,
            budget=self.budget,
            state=self.rerank_state,
            phase=phase,
            deadline=self.deadline,
            clock=self.clock,
            on_submit=on_submit,
        )
        if results is None:
            return envelope()
        scores = []
        for i, (row, plan, key, cache_key, cached) in enumerate(slots):
            if self.clock() >= self.deadline or not self.rerank_state.can_admit():
                return envelope()
            if cached is not None:
                if not self.rerank_state.publish(lambda: scores.append(cached)):
                    return envelope()
                continue
            result = results[i]
            aggregate = aggregate_window_scores(result)
            if aggregate.prepared_input_scoring_coverage != "complete":
                return envelope()
            coverage = WindowCoverage(
                plan.required_window_ids,
                tuple(fingerprint_pair(query, w.text) for w in plan.windows),
                tuple(s.pair_fingerprint for s in result.successful),
                "complete",
            )
            score = PassageScore(
                row.pk,
                row.doc_id,
                row.chunk_number,
                plan.source.source_fingerprint,
                plan.fingerprint,
                aggregate.value,
                coverage,
            )
            if not self.rerank_state.publish(
                lambda: self._scores.__setitem__(key, score)
            ):
                return envelope()
            if self.cache_enabled:
                single = RerankScoreSet(
                    "v3-window",
                    fingerprint_text(query),
                    self.scorer_fingerprint,
                    fingerprint_pool(
                        ((row.pk, plan.source.source_fingerprint, plan.fingerprint),)
                    ),
                    "pointwise",
                    "complete",
                    (row.pk,),
                    (score,),
                )
                set_window_result(cache_key, single, budget=self._cache_budget)
            scores.append(score)
        published = []
        self.rerank_state.publish(lambda: published.append(envelope(scores)))
        return published[0] if published else envelope()
