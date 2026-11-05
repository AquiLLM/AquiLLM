"""Optional preparation has a child slice; the ordinary final lease is unchanged."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from apps.documents.services.source_loading import SourcePreparationLimited
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.tests import test_scoped_recovery_final_preparation as flow

pytestmark = pytest.mark.django_db(transaction=True)


def _capture_case(monkeypatch):
    cases = []
    original = flow.support.native_case

    @contextmanager
    def captured(*args, **kwargs):
        with original(*args, **kwargs) as case:
            cases.append(case)
            yield case

    monkeypatch.setattr(flow.support, "native_case", captured)
    return cases


@pytest.mark.parametrize("boundary", ["optional_child", "parent_final"])
def test_recovery_preparation_child_and_original_parent_deadlines(monkeypatch, boundary):
    cases = _capture_case(monkeypatch)
    initialize = WindowSelectionScorer.__init__
    crossed = []

    def scorer(self, *args, **kwargs):
        original = kwargs["pair_counter"]
        parent_deadline = kwargs["deadline"]

        def counter(query, text):
            if len(text) >= 5000 and not crossed:
                crossed.append(True)
                cases[-1].now[0] = parent_deadline + .001 - (1.25 if boundary == "optional_child" else 0)
            return original(query, text)

        initialize(self, *args, **{**kwargs, "pair_counter": counter})

    monkeypatch.setattr(WindowSelectionScorer, "__init__", scorer)
    if boundary == "parent_final":
        with pytest.raises(SourcePreparationLimited, match="final source preparation deadline"):
            flow.test_optional_scoped_recovery_preserves_ordinary_final_preparation(monkeypatch, True, False)
    else:
        flow.test_optional_scoped_recovery_preserves_ordinary_final_preparation(monkeypatch, True, False)
    assert crossed


def test_optional_provider_outage_does_not_bypass_shared_scorer_latch(monkeypatch):
    initialize = WindowSelectionScorer.__init__
    failed = []

    def scorer(self, provider, **kwargs):
        original = provider.score_pair

        def score(pair, timeout_seconds):
            result = original(pair, timeout_seconds)
            if result[0] == 2.0:  # The cheap optional source, after ordinary scoring.
                failed.append(True)
                return None
            return result

        provider.score_pair = score
        initialize(self, provider, **kwargs)

    monkeypatch.setattr(WindowSelectionScorer, "__init__", scorer)
    with pytest.raises(SourcePreparationLimited, match="source scorer unavailable"):
        flow.test_optional_scoped_recovery_preserves_ordinary_final_preparation(monkeypatch, True, True)
    assert failed


@pytest.mark.parametrize("query,expected_final", [("Atlas", 0), ("different question", 25)])
def test_same_scorer_reuses_only_exact_query_prepared_windows(monkeypatch, query, expected_final):
    cases = _capture_case(monkeypatch)
    initialize = WindowSelectionScorer.__init__

    def scorer(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        case = cases[-1]
        scored = self.score_windows(query, case.snapshot.baseline_candidates, phase="acquisition")
        assert scored.status == "complete" and len(scored.scores) == 25

    monkeypatch.setattr(WindowSelectionScorer, "__init__", scorer)
    flow.test_optional_scoped_recovery_preserves_ordinary_final_preparation(monkeypatch, True, False)
    assert cases[-1].source.budget.pairs_used == {"acquisition": 25, "final": expected_final}


@pytest.mark.parametrize("mutation", ["provider", "scorer_fingerprint", "pair_counter"])
def test_optional_preparation_cannot_replace_original_scoring_capability(monkeypatch, mutation):
    from apps.chat.services import rag_source_hydration

    scorers, changed = [], []
    initialize = WindowSelectionScorer.__init__
    prepare = rag_source_hydration.prepare_evidence

    def scorer(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        scorers.append(self)

    def prepare_evidence(source, **kwargs):
        result = prepare(source, **kwargs)
        if len(source.text) >= 5000 and not changed:
            changed.append(True)
            current = scorers[-1]
            if mutation == "provider":
                current.provider = SimpleNamespace(score_pair=current.provider.score_pair)
            elif mutation == "scorer_fingerprint":
                current.scorer_fingerprint = "replacement-model"
            else:
                original = current.pair_counter
                current.pair_counter = lambda query, text: original(query, text)
        return result

    monkeypatch.setattr(WindowSelectionScorer, "__init__", scorer)
    monkeypatch.setattr(rag_source_hydration, "prepare_evidence", prepare_evidence)
    with pytest.raises(SourceScopeUnavailable, match="recovery_scoring_capability_changed"):
        flow.test_optional_scoped_recovery_preserves_ordinary_final_preparation(monkeypatch, True, False)
    assert changed
