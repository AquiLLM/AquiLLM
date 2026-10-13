"""Private graph provenance is observed only after successful closed publication."""

from contextlib import contextmanager

import pytest

from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame, run_graph

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("observer_fails", [False, True])
def test_actual_selected_graph_event_follows_successful_publication(monkeypatch, observer_fails):
    from lib import replay_observation

    with graph_frame(monkeypatch) as active:
        f, _, source, scope, _, _, _, _ = active
        observed = []

        def observe(pks):
            assert type(pks) is tuple
            assert set(pks) <= set(scope.bodies)
            assert set(pks) <= {row.pk for row in source.cache.values()}
            assert source.lock._is_owned()
            observed.append(pks)
            if observer_fails:
                raise RuntimeError("synthetic observer failure")

        monkeypatch.setattr(replay_observation, "graph_selected_sources_v2", observe)
        rows, diagnostics = run_graph(active)
        assert diagnostics["graph_status"] == "hit"
        assert len(observed) == 1 and set(observed[0]) <= {r.pk for r in rows}
        assert len(observed[0]) == len(set(observed[0]))
        assert set(observed[0]) == {f.chunks["A"].pk, f.chunks["B"].pk}


def test_cleanup_revocation_never_observes_unpublished_selected_graph(monkeypatch):
    from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as owner
    from lib import replay_observation

    with graph_frame(monkeypatch) as active:
        _, _, source, scope, _, _, operation, _ = active
        database, events = owner.bounded_source_database, []

        @contextmanager
        def revoke(*args, **kwargs):
            with database(*args, **kwargs):
                yield
            operation.live = False

        monkeypatch.setattr(owner, "bounded_source_database", revoke)
        monkeypatch.setattr(replay_observation, "graph_selected_sources_v2", events.append)
        rows, diagnostics = run_graph(active)
        assert rows == () and diagnostics["graph_status"] == "error"
        assert events == [] and source.cache == {} and scope.bodies == {}
