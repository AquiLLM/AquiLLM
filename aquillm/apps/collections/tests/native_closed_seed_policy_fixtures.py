"""Shared real PostgreSQL native-owner setup and statement observations."""

from contextlib import contextmanager
from dataclasses import replace

from django.db import connection

from apps.collections.models import CollectionPermission
from apps.collections.services.bounded_retrieval_authorization import (
    bounded_reauthorization_reads,
)
from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_source_operation import _closed_header
from apps.documents.tests.test_closed_graph_seed_work import large_selected_scope
from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage
from apps.knowledge_graph.retrieval.ready_scope_repository_v2 import (
    select_ready_scope_v2,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetV2,
)
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame


def small_native_scope(f):
    pressure_rows(f)
    user = f.documents["A"].ingested_by
    CollectionPermission.objects.bulk_create(
        [
            CollectionPermission(user=user, collection_id=key, permission="VIEW")
            for key in (7, 9)
        ]
    )
    f.authorization = build_selected_scope_authorization_context(
        principal=user,
        selected_collection_ids=(7, 9),
        selected_documents=tuple(f.documents.values()),
    )


@contextmanager
def native_frame(
    monkeypatch,
    *,
    large=False,
    frontier_timeout=5,
    custom_slots=False,
    max_source_rows=5000,
):
    def sources(f):
        (large_selected_scope if large else small_native_scope)(f)
        if custom_slots:
            from apps.collections.services.retrieval_authorization import (
                bind_retrieval_reauthorization_capability,
            )

            original = f.authorization.reauthorization_capability

            class SlotsPolicy:
                __slots__ = ("native",)
                policy_version = original._policy.policy_version
                policy_checksum = original._policy.policy_checksum

                def __init__(self):
                    self.native = original._policy

                def opaque_principal_reference(self, **kw):
                    return self.native.opaque_principal_reference(**kw)

                def current_authorized_document_scope(self, **kw):
                    return self.native.current_authorized_document_scope(**kw)

                def current_authorized_selected_document_scope(self, **kw):
                    return self.native.current_authorized_selected_document_scope(**kw)

            f.authorization = replace(
                f.authorization,
                reauthorization_capability=bind_retrieval_reauthorization_capability(
                    principal=original._principal, policy=SlotsPolicy()
                ),
            )

    with graph_frame(monkeypatch, prepare_sources=sources, dataset=True) as active:
        f, graph, source, scope, _, dependencies, _, _ = active
        graph.authorization = scope.authorization
        dependencies.settings.graph_direct_enabled = False
        dependencies.settings.graph_extended_enabled = True
        snapshot = capture(active)
        selected = select_ready_scope_v2(
            authorization=scope.authorization,
            settings=dependencies.settings,
            representation_snapshot=scope.representation,
            requested_document_ids=scope.requested_document_ids,
        )
        base = TextChunk.objects.all()
        stage = _GraphSourceStage(scope, base, _closed_header(base, scope))
        stage.initial_keys = set(scope.validate())
        owner = stage.seed_validation(snapshot, stage.check)
        view, _ = owner.baseline(
            snapshot, authorization=scope.authorization, settings=dependencies.settings
        )
        budget = FrontierBudgetV2(
            turn_budget=source.budget,
            deadline=graph.clock() + frontier_timeout,
            clock=graph.clock,
            max_identities=200,
            max_edges=1000,
            max_queries=64,
            max_source_rows=max_source_rows,
            max_wire_bytes=1048576,
        )
        try:
            yield active, owner, view, selected.groups, budget
        finally:
            stage.withdraw()


def prepare(frame):
    active, owner, view, groups, budget = frame
    _, graph, _, scope, _, dependencies, _, _ = active
    with bounded_reauthorization_reads(scope.authorization, budget):
        owner.prepare_projections(
            view,
            groups,
            runtime=graph,
            authorization=scope.authorization,
            settings=dependencies.settings,
            frontier=budget,
        )


@contextmanager
def policy_data_counts():
    counts = []
    table = CollectionPermission._meta.db_table

    def observe(execute, sql, params, many, context):
        result = execute(sql, params, many, context)
        if table in sql or sql.count("UNION ALL") == 7:
            counts.append(context["cursor"].rowcount)
        return result

    with connection.execute_wrapper(observe):
        yield counts
