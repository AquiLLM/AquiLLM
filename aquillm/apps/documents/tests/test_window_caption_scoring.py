"""Caption relevance covers exact text pairs without inspecting image payloads."""

from threading import Event
from time import monotonic
from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.documents.services.chunk_rerank_pair_capability import VerifiedPairCounter
from apps.documents.services.chunk_rerank_results import fingerprint_text
from apps.documents.services.chunk_rerank_score_cache import window_cache_key
from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from lib.retrieval import TurnBudget, TurnLimits


class TextTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "|" + messages[1]["content"]

    def encode(self, rendered, **kwargs):
        return list(range(len(rendered)))


def verified_counter():
    return VerifiedPairCounter(
        TextTokenizer(),
        "text-pair",
        "caption-proof",
        "tokenizer",
        "template",
        monotonic() + 60,
        Event(),
    )


class TextProvider:
    scorer_fingerprint = "one-text-model"

    def __init__(self):
        self.seen = []

    def score_pair(self, pair, timeout_seconds):
        assert all(type(value) is str for value in pair)
        self.seen.append(pair)
        return float(len(pair[1])), pair


class Caption(SimpleNamespace):
    @property
    def pixels(self):
        pytest.fail("text scoring inspected pixels")

    image = pixel_bytes = image_url = media_artifact = pixels


def caption(text="Stored figure caption", revision="one"):
    return Caption(
        pk=1,
        doc_id=UUID(int=1),
        chunk_number=0,
        content=text,
        modality="image",
        source_revision=revision,
    )


def test_caption_counts_and_scores_exact_text_with_verified_pair_counter():
    budget, provider, counter = (
        TurnBudget(TurnLimits()),
        TextProvider(),
        verified_counter(),
    )
    scorer = WindowSelectionScorer(provider, budget=budget, pair_counter=counter)
    row = caption()
    try:
        plan = scorer.plan("q", row)
        assert plan.preparation_coverage == "complete"
        assert tuple(window.text for window in plan.windows) == (row.content,)
        result = scorer.score_windows("q", (row,))
        assert result.status == "complete" and result.scoring_kind == "pointwise"
        assert provider.seen == [("q", row.content)]
        assert result.scores[0].window_coverage.is_complete()
        assert result.scores[0].effective_pair_fingerprint == plan.fingerprint
        assert budget.pairs_used == {"acquisition": 0, "final": 1}
        assert row.modality == "image"
    finally:
        budget.close("caption_test")


@pytest.mark.parametrize("loss", ["unknown", "revoked"])
def test_caption_genuine_counter_loss_keeps_whole_pool_rank_fallback(loss):
    budget, provider = TurnBudget(TurnLimits()), TextProvider()
    counter = verified_counter()
    if loss == "revoked":
        counter.revoked.set()
    else:

        def counter(query, document):
            return None

    rows = (
        caption(),
        SimpleNamespace(
            pk=2,
            doc_id=UUID(int=2),
            chunk_number=0,
            content="ordinary text",
            modality="text",
        ),
    )
    scorer = WindowSelectionScorer(provider, budget=budget, pair_counter=counter)
    try:
        result = scorer.score_windows("q", rows)
        assert result.status == "unavailable" and result.scoring_kind == "rank_only"
        assert result.candidate_order == (1, 2) and not result.scores
        assert provider.seen == [] and budget.pairs_used["final"] == 0
    finally:
        budget.close("caption_unknown_test")


def test_caption_policy_and_revision_have_exact_score_and_cache_identity():
    budget, provider, counter = (
        TurnBudget(TurnLimits()),
        TextProvider(),
        verified_counter(),
    )
    scorer = WindowSelectionScorer(provider, budget=budget, pair_counter=counter)
    row = caption()
    try:
        first = scorer.score_windows("q", (row,))
        assert first.status == "complete"
        old_policy = fingerprint_text(
            f"window-max-v1:{provider.scorer_fingerprint}:{counter.identity}"
        )
        assert scorer.scorer_fingerprint != old_policy
        assert scorer.scorer_fingerprint == fingerprint_text(
            f"window-max-v1:text-content-v1:{provider.scorer_fingerprint}:{counter.identity}"
        )
        plan = scorer.plan("q", row)
        assert window_cache_key(old_policy, plan.fingerprint) != window_cache_key(
            scorer.scorer_fingerprint, plan.fingerprint
        )
        changed = caption(revision="two")
        assert scorer.plan("q", changed).fingerprint != plan.fingerprint
        changed.content += " edited caption"
        assert scorer.plan("q", changed).fingerprint != plan.fingerprint
        second = scorer.score_windows("q", (changed,))
        assert second.status == "complete" and len(provider.seen) == 2
        assert (
            second.scores[0].effective_pair_fingerprint
            != first.scores[0].effective_pair_fingerprint
        )
    finally:
        budget.close("caption_identity_test")


@pytest.mark.parametrize(
    "text", ["", "caption " * 10000], ids=["empty", "over_allowance"]
)
def test_empty_or_over_allowance_caption_never_claims_complete_score(text):
    budget, provider = TurnBudget(TurnLimits()), TextProvider()
    scorer = WindowSelectionScorer(
        provider, budget=budget, pair_counter=verified_counter()
    )
    try:
        result = scorer.score_windows("q", (caption(text),))
        assert result.status == "unavailable" and not result.scores
        assert budget.pairs_used["final"] <= 45
        assert budget.limits.final_scoring_ms == 3000
    finally:
        budget.close("caption_bounds_test")


def test_long_caption_scores_all_exact_windows_without_pixel_coverage():
    budget, provider = TurnBudget(TurnLimits()), TextProvider()
    row = caption("caption " * 220)
    scorer = WindowSelectionScorer(
        provider, budget=budget, pair_counter=verified_counter()
    )
    try:
        plan = scorer.plan("q", row)
        assert plan.preparation_coverage == "complete" and len(plan.windows) > 1
        assert plan.windows[0].start == 0 and plan.windows[-1].end == len(row.content)
        assert all(a.end >= b.start for a, b in zip(plan.windows, plan.windows[1:]))
        result = scorer.score_windows("q", (row,))
        assert result.status == "complete"
        assert provider.seen == [("q", window.text) for window in plan.windows]
        assert budget.pairs_used["final"] == len(plan.windows) <= 45
        assert result.scores[0].window_coverage.is_complete()
    finally:
        budget.close("caption_long_test")
