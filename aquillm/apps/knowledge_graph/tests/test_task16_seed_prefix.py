"""Closed native composition retains fresh SQL at entry and consumption."""

from types import SimpleNamespace

import pytest

from apps.collections.tests.native_closed_seed_policy_fixtures import native_frame
from apps.documents.models import TextChunk
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.knowledge_graph.models import CollectionEntity, CollectionEntityDocumentLink
from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain
from apps.knowledge_graph.retrieval import closed_graph_seed_validation as validation
from apps.knowledge_graph.retrieval import closed_seed_projection_batch as batch
from apps.knowledge_graph.retrieval.graph_runtime_seeds_v2 import (
    _group_request_native,
    group_request,
)
from apps.knowledge_graph.retrieval.graph_seed_allocation_v2 import (
    _allocate_closed_group_seeds_v2,
)
from apps.knowledge_graph.tests.test_native_ontology_proof_postgres import (
    extractor_runtime,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def allocate(frame, *, native):
    active, owner, _, groups, ledger = frame
    f, graph, _, scope, _, dependencies, _, _ = active
    group = next(
        g for g in groups if f.documents["A"].id in g.scope.selected_document_ids
    )
    kwargs = dict(
        query="Atlas",
        group=group,
        authorization=scope.authorization,
        read_budget=ledger,
    )
    (_group_request_native if native else group_request)(
        **kwargs, **({"owner": owner} if native else {})
    )
    return _allocate_closed_group_seeds_v2(
        seed_validation=owner,
        groups=(group,),
        runtime=graph,
        query="Atlas",
        baseline=owner.snapshot,
        authorization=scope.authorization,
        settings=dependencies.settings,
        caps=SimpleNamespace(max_seeds=64),
        deadline=ledger.deadline,
        frontier=ledger,
    )


@pytest.mark.parametrize("native", [True, False])
def test_one_group_allocation_only_sweeps_at_native_entry_and_consumption(
    monkeypatch, native
):
    calls = {"live": 0, "dense": 0}
    before_direct = []

    def runtime(graph):
        if not native:
            del graph._direct_seeds
        graph.settings.graph_direct_enabled = True
        extractor_runtime(graph)
        original = graph.extractor_factory

        def factory(**kwargs):
            extractor = original(**kwargs)

            def extract(**kwargs):
                before_direct.append(dict(calls))
                return extractor.extract(**kwargs)

            return SimpleNamespace(extract=extract)

        graph.extractor_factory = factory

    with native_frame(monkeypatch, metadata=native, prepare_runtime=runtime) as frame:
        active, owner, view, _, ledger = frame
        f, graph, source, scope, _, _, _, _ = active
        projection = f.projections["A"]
        entity = CollectionEntity.objects.get(
            artifact_id=projection.artifact_id, normalized_label="a"
        )
        expected_key = graph.codec.encode(
            ProjectionIdentifierDomain.ENTITY,
            generation=projection.generation_key,
            source=entity.pk,
        ).value
        original_seeds = owner.snapshot.graph_seeds
        original_refs = view.seed_references
        before = (source.budget.sources_used, source.cache.copy(), scope.bodies.copy())
        for name, key in (
            ("_live_seed_references", "live"),
            ("_dense_seed_keys", "dense"),
        ):
            original = getattr(validation, name)

            def read(*args, _original=original, _key=key, **kwargs):
                calls[_key] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(validation, name, read)
        result, reasons = allocate(frame, native=native)
        assert not reasons and len(result) == 1
        seeds, mass, local = next(iter(result.values()))
        assert tuple((s.identity_key, s.mass) for s in seeds) == ((expected_key, 1.0),)
        assert mass == 1.0 and local == {"not_ready"}
        assert tuple(s.identity_key for s in seeds) == tuple(
            sorted(s.identity_key for s in seeds)
        )
        assert sum(s.mass for s in seeds) == pytest.approx(1.0)
        assert owner.snapshot.graph_seeds is original_seeds
        assert owner.views[-1][0].graph_seeds == view.graph_seeds
        assert all(
            a is b
            for a, b in zip(
                owner.views[-1][0].seed_references, original_refs, strict=True
            )
        )
        assert owner.mapping == {
            projection.pk: {
                r.pk: (expected_key,) if r.chunk_number in (0, 1, 2) else ()
                for r in original_refs
            }
        }
        assert before == (source.budget.sources_used, source.cache, scope.bodies)
        assert ledger.turn_budget is owner.stage.budget is source.budget
        assert ledger.max_queries == 64 and ledger.max_source_rows == 5000
        assert ledger.max_wire_bytes == 1048576 and not ledger._uncertain
        assert (
            ledger._reserved_rows == 0 and ledger.queries > 0 and ledger.source_rows > 0
        )
        print({"native": native, "before_direct": before_direct, "sweeps": calls})
        assert before_direct == [
            {"live": 1 if native else 3, "dense": 1 if native else 3}
        ]
        assert calls == {"live": 2 if native else 4, "dense": 2 if native else 4}


@pytest.mark.parametrize("boundary", ["batch", "direct"])
def test_changed_source_is_rejected_before_prepared_mapping_consumption(
    monkeypatch, boundary
):
    mutations = []

    def mutate():
        assert TextChunk.objects.filter(pk=chunk_id[0]).update(content="changed") == 1
        mutations.append(boundary)

    def runtime(graph):
        graph.settings.graph_direct_enabled = True
        extractor_runtime(graph)
        original = graph.extractor_factory

        def factory(**kwargs):
            extractor = original(**kwargs)

            def extract(**kwargs):
                result = extractor.extract(**kwargs)
                if boundary == "direct":
                    mutate()
                return result

            return SimpleNamespace(extract=extract)

        graph.extractor_factory = factory

    chunk_id = []
    with native_frame(monkeypatch, metadata=True, prepare_runtime=runtime) as frame:
        active, owner, view, _, ledger = frame
        source, scope = active[2:4]
        chunk_id.append(view.seed_references[0].pk)
        before = (source.budget.sources_used, source.cache.copy(), scope.bodies.copy())
        original = batch.seed_query_rows

        def read(query, **kwargs):
            rows = original(query, **kwargs)
            if boundary == "batch" and query.model is CollectionEntityDocumentLink:
                assert rows
                mutate()
            return rows

        monkeypatch.setattr(batch, "seed_query_rows", read)
        with pytest.raises(SourceScopeUnavailable):
            allocate(frame, native=True)
        assert mutations == [boundary]
        assert before == (source.budget.sources_used, source.cache, scope.bodies)
        assert ledger.turn_budget is owner.stage.budget and ledger.queries > 0


@pytest.mark.parametrize("target", ["instance", "class"])
def test_structural_helper_replacement_latches_selected_native_owner(
    monkeypatch, target
):
    with native_frame(monkeypatch, metadata=True) as frame:
        _, owner, _, groups, ledger = frame
        kwargs = dict(
            owner=owner,
            query="Atlas",
            group=groups[0],
            authorization=owner.stage.scope.authorization,
            read_budget=ledger,
        )
        _group_request_native(**kwargs)
        before = (ledger.queries, ledger.source_rows)
        monkeypatch.setattr(
            owner if target == "instance" else type(owner),
            "_checked_view",
            lambda *a, **kw: (),
            raising=False,
        )
        with pytest.raises(ValueError):
            _group_request_native(**kwargs)
        assert owner._native_metadata.failed
        assert (ledger.queries, ledger.source_rows) == before


@pytest.mark.parametrize("entrypoint", ["group", "prepare"])
def test_composition_rejects_replaced_issued_views(monkeypatch, entrypoint):
    from dataclasses import replace

    with native_frame(monkeypatch, metadata=True) as frame:
        active, owner, view, groups, ledger = frame
        _group_request_native(
            owner=owner,
            query="Atlas",
            group=groups[0],
            authorization=active[3].authorization,
            read_budget=ledger,
        )
        view = replace(view)
        before = (ledger.queries, ledger.source_rows)
        with pytest.raises(SourceScopeUnavailable):
            if entrypoint == "group":
                owner.group(view, groups[0])
            else:
                owner.prepare_projections(
                    view,
                    groups,
                    runtime=active[1],
                    authorization=active[3].authorization,
                    settings=active[5].settings,
                    frontier=ledger,
                )
        assert (ledger.queries, ledger.source_rows) == before
