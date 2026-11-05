"""Already charged native sources survive wider metadata unions without new slots."""

from dataclasses import replace

import pytest

from apps.documents.models import TextChunk
from apps.documents.services import graph_candidate_admission as admission
from apps.documents.services.graph_candidate_metadata import rank_refs
from apps.documents.tests.test_graph_source_admission import capture
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame
from lib.retrieval.operation import operation_scope

pytestmark = pytest.mark.django_db(transaction=True)


def pressure_rows(fixture):
    document = fixture.documents["A"]
    text = document.full_text
    rows = []
    for number in range(90):
        start = 2 + number // 49
        end = start + 1 + number % 49
        rows.append(TextChunk(doc_id=document.id, content=text[start:end],
                              chunk_number=100 + number,
                              start_position=start, end_position=end))
    TextChunk.objects.bulk_create(rows)


def controlled_channels(monkeypatch, count):
    """Control lane overlap only; real refs, body readers and authorities remain."""
    original = admission.capture_metadata
    observed = []

    def metadata(model, query, scope, *args, **kwargs):
        result = original(model, query, scope, *args, **kwargs)
        rows = model.objects.using(scope.using).filter(
            doc_id__in=scope.requested_document_ids, chunk_number__gte=100,
        ).order_by("chunk_number")
        split = (count + 1) // 2
        exact, trigram = rank_refs(rows[:split]), rank_refs(rows[split:count])
        channels = {"dense": (), "exact": exact, "trigram": trigram, "full_text": ()}
        observed.append(channels)
        return (channels, {}, *result[2:])

    monkeypatch.setattr(admission, "capture_metadata", metadata)
    return observed


def test_initial_publication_retains_paid_tails_beyond_45_metadata_refs(monkeypatch):
    observed = controlled_channels(monkeypatch, 49)
    paid = []

    def read_paid(model, query, scope, limits, observation, channels, dense, *, stage, check):
        refs = channels["exact"][:5] + channels["exact"][-2:] + channels["trigram"][-2:]
        paid.extend(refs)
        stage.read_captured(refs)
        check()
        return channels, dense, None

    monkeypatch.setattr(admission, "prepare_comparison_coverage", read_paid)
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        source, scope = active[2], active[3]
        with operation_scope(source.budget, source_limit=30):
            snapshot = capture(active)
            plan = snapshot.source_admission
            expected = {ref.pk for ref in paid}
            assert sum(len(lane) for lane in observed[0].values()) == 49
            assert len(paid) == 9
            assert expected <= set(plan.initial_ids)
            assert expected <= {row.pk for row in snapshot.baseline_candidates}
            assert expected <= set(scope.bodies)
            assert expected <= {row.pk for row in source.cache.values()}
            assert source.budget.sources_used == len(plan.initial_ids) == 11
            assert plan.graph_ceiling == 19
            # The two novel slots retain independent lexical lane fairness.
            novel = set(plan.initial_ids) - expected
            assert novel == {observed[0]["exact"][5].pk, observed[0]["trigram"][0].pk}


def test_zero_capacity_top_up_retains_paid_tails_from_wide_deferred_union(monkeypatch):
    observed = controlled_channels(monkeypatch, 90)
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        source, scope = active[2], active[3]
        snapshot = capture(active)
        plan = snapshot.source_admission
        assert len(plan.initial_ids) == source.budget.sources_used == 25
        tails = tuple(ref.pk for name in ("exact", "trigram") for ref in observed[0][name][-2:])
        remaining = tuple(pk for pk in plan.baseline_order if pk not in plan.initial_ids and pk not in tails)
        paid = (*remaining[:16], *tails)
        assert len(plan._closed_read(paid)) == 20
        assert source.budget.sources_used == 45 and source.budget.remaining_source_capacity() == 0
        before = source.budget.text_used.copy()
        expected = tuple(pk for pk in plan.baseline_order if pk in paid)
        assert plan.top_up_ids() == expected
        rows = plan.top_up()
        assert tuple(row.pk for row in rows) == expected
        assert source.budget.sources_used == 45 and source.budget.text_used == before
        assert set(expected) <= set(scope.bodies)
        # A paid PK under a different content revision is not reusable.
        changed = replace(plan, channels=tuple((name, tuple(
            replace(ref, revision="f" * 32) if ref.pk == tails[-1] else ref
            for ref in refs)) for name, refs in plan.channels))
        assert tails[-1] not in changed.top_up_ids()
