"""Fresh native closed-seed policy proofs retain the full original selected scope."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
from django.db import connection

from apps.collections.models import CollectionPermission
from apps.collections.services.bounded_retrieval_authorization import (
    bounded_reauthorization_reads,
)
from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.models import RawTextDocument, TeXDocument, TextChunk
from apps.documents.services.chunk_search_source_operation import _closed_header
from apps.documents.tests.test_closed_graph_seed_work import large_selected_scope
from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.knowledge_graph.retrieval.extended_seed_repository import (
    ExtendedSeedRepository,
)
from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage
from apps.knowledge_graph.retrieval.ready_scope_repository_v2 import (
    select_ready_scope_v2,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetV2,
)
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


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


def test_real_31_530_ordinary_batch_policy_baseline_is_four_statements_1122_rows(
    monkeypatch,
):
    from apps.knowledge_graph.retrieval.extended_seed_repository import (
        ExtendedSeedRepository,
    )

    with native_frame(monkeypatch, large=True) as frame:
        active, owner, view, groups, budget = frame
        _, graph, _, scope, _, _, _, _ = active
        requests = tuple(
            (
                p,
                tuple(
                    (r.pk, r.doc_id)
                    for r in view.seed_references
                    if r.doc_id in dict(p.documents)
                ),
            )
            for group in groups
            for p in group.scope.projections
        )
        requests = tuple((p, chunks) for p, chunks in requests if chunks)
        with (
            bounded_reauthorization_reads(scope.authorization, budget),
            policy_data_counts() as counts,
        ):
            mapping = ExtendedSeedRepository()._load_seed_identities_batch(
                requests=requests,
                authorization=scope.authorization,
                codec=graph.codec,
                max_rows=4999,
                read_budget=budget,
            )
        assert mapping and all(any(value.values()) for value in mapping.values())
        assert len(scope.authorization.selected_collection_ids) == 31
        assert len(scope.authorization.selected_document_ids) == 530
        assert (len(counts), sum(counts)) == (4, 1122)
        print(
            f"ordinary native policy boundaries: statements={len(counts)}, "
            f"returned_rows={sum(counts)}"
        )


def test_real_31_530_native_owner_uses_two_fresh_single_row_policy_proofs(monkeypatch):
    with native_frame(monkeypatch, large=True) as frame:
        active, owner, _, _, budget = frame
        _, _, source, scope, _, _, _, _ = active
        original = (
            source.budget,
            scope.authorization,
            owner.plan,
            budget.deadline,
            budget.clock,
        )
        before = (source.budget.sources_used, dict(source.cache), dict(scope.bodies))
        with policy_data_counts() as counts:
            prepare(frame)
        assert owner.mapping and all(
            any(values.values()) for values in owner.mapping.values()
        )
        assert budget.turn_budget is original[0] and scope.authorization is original[1]
        assert (
            owner.plan is original[2]
            and budget.deadline == original[3]
            and budget.clock is original[4]
        )
        assert before == (source.budget.sources_used, source.cache, scope.bodies)
        assert (budget.max_queries, budget.max_source_rows, budget.max_wire_bytes) == (
            64,
            5000,
            1048576,
        )
        print(
            f"closed native policy boundaries: statements={len(counts)}, "
            f"returned_rows={sum(counts)}; "
            f"original frontier statements={budget.queries}, "
            f"returned_rows={budget.source_rows}"
        )
        assert (len(counts), sum(counts)) == (2, 2), (
            "native full-scope proof still materializes selected policy rows"
        )


@pytest.mark.parametrize(
    "change",
    [
        "permission",
        "inactive",
        "delete",
        "lost_collection",
        "duplicate_same",
        "duplicate_other",
        "cross_model",
        "cross_model_other",
        "missing_duplicate",
    ],
)
def test_second_native_proof_rejects_full_scope_drift_even_without_a_seed(
    monkeypatch, change
):
    from apps.knowledge_graph.models import CollectionEntityDocumentLink
    from apps.knowledge_graph.retrieval import closed_seed_projection_batch as batch

    with native_frame(monkeypatch) as frame:
        active, owner, _, _, budget = frame
        f, _, source, scope, _, _, _, _ = active
        original, observed = batch.seed_query_rows, []

        def read(query, **kwargs):
            rows = original(query, **kwargs)
            if query.model is CollectionEntityDocumentLink:
                assert rows
                observed.append(True)
                a, b = f.documents["A"], f.documents["B"]
                if change == "permission":
                    CollectionPermission.objects.filter(
                        collection_id=b.collection_id
                    ).delete()
                elif change == "inactive":
                    type(a.ingested_by).objects.filter(pk=a.ingested_by_id).update(
                        is_active=False
                    )
                elif change == "delete":
                    RawTextDocument.objects.filter(pk=b.pk).delete()
                elif change == "lost_collection":
                    RawTextDocument.objects.filter(pk=b.pk).update(
                        collection_id=a.collection_id
                    )
                elif change == "missing_duplicate":
                    RawTextDocument.objects.filter(pk=b.pk).update(id=a.id)
                else:
                    model = (
                        TeXDocument
                        if change.startswith("cross_model")
                        else RawTextDocument
                    )
                    model.objects.bulk_create(
                        [
                            model(
                                id=a.id,
                                title="duplicate",
                                full_text="duplicate",
                                full_text_hash="d" * 64,
                                collection_id=b.collection_id
                                if change in ("duplicate_other", "cross_model_other")
                                else a.collection_id,
                                ingested_by_id=a.ingested_by_id,
                            )
                        ]
                    )
            return rows

        monkeypatch.setattr(batch, "seed_query_rows", read)
        before = (source.budget.sources_used, dict(source.cache), dict(scope.bodies))
        with pytest.raises((ValueError, TimeoutError)):
            prepare(frame)
        assert observed == [True] and owner.mapping is None
        assert before == (source.budget.sources_used, source.cache, scope.bodies)
        assert budget.queries > 0


def test_deleted_original_principal_has_no_fresh_native_permission(monkeypatch):
    from django.contrib.auth.models import User

    from apps.documents.models.document import _get_document_types

    with native_frame(monkeypatch) as frame:
        principal = frame[0][3].authorization.reauthorization_capability._principal
        replacement = User.objects.create_user(username="replacement-ingester")
        for model in _get_document_types():
            model._base_manager.filter(ingested_by_id=principal.pk).update(
                ingested_by=replacement
            )
        User.objects.filter(pk=principal.pk).delete()
        with policy_data_counts() as counts, pytest.raises(ValueError):
            prepare(frame)
        assert counts == [1] and frame[1].mapping is None


def test_document_input_is_globally_capped_at_n_plus_one_before_counts(monkeypatch):
    with native_frame(monkeypatch) as frame:
        a = frame[0][0].documents["A"]
        RawTextDocument.objects.bulk_create(
            [
                RawTextDocument(
                    id=a.id,
                    title=f"overflow-{i}",
                    full_text=f"overflow-{i}",
                    full_text_hash=str(i) * 64,
                    collection_id=a.collection_id,
                    ingested_by_id=a.ingested_by_id,
                )
                for i in range(1, 7)
            ]
        )
        authorization = frame[0][3].authorization
        with bounded_reauthorization_reads(authorization, frame[4]):
            proof = frame[1]._native_policy.select(
                authorization=authorization,
                repository=ExtendedSeedRepository(),
                read_budget=frame[4],
            )
            sql, params = proof._statement()
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
            assert len(rows) == 1 and rows[0][2] == 3
            with pytest.raises(ValueError):
                proof.require_current()
        assert (frame[4].queries, frame[4].source_rows) == (1, 1)


def test_unselected_document_collision_stays_irrelevant_to_policy(monkeypatch):
    from uuid import uuid4

    with native_frame(monkeypatch) as frame:
        f = frame[0][0]
        a = f.documents["A"]
        other = uuid4()
        RawTextDocument.objects.bulk_create(
            [
                RawTextDocument(
                    id=other,
                    title=f"unselected-{i}",
                    full_text=f"unselected-{i}",
                    full_text_hash=str(i) * 64,
                    collection_id=a.collection_id,
                    ingested_by_id=a.ingested_by_id,
                )
                for i in (1, 2)
            ]
        )
        prepare(frame)
        assert frame[1].mapping


def test_native_proof_does_not_change_public_partial_scope_revalidation(monkeypatch):
    from apps.collections.services.retrieval_authorization import (
        revalidate_retrieval_authorization_context,
    )

    with native_frame(monkeypatch) as frame:
        f, _, _, scope, _, _, _, _ = frame[0]
        CollectionPermission.objects.filter(
            collection_id=f.documents["B"].collection_id
        ).delete()
        current = revalidate_retrieval_authorization_context(
            context=scope.authorization
        )
        assert current.document_ids == (f.documents["A"].id,)
        assert current.collection_ids == (f.documents["A"].collection_id,)


def test_custom_slots_policy_keeps_ordinary_batch_before_native_construction(
    monkeypatch,
):
    with native_frame(monkeypatch, custom_slots=True) as frame:
        with policy_data_counts() as counts:
            prepare(frame)
        assert frame[1].mapping and (len(counts), sum(counts)) == (4, 8)


@pytest.mark.parametrize("metadata", ["column", "table"])
def test_preexisting_native_metadata_changes_decline_before_dispatch(
    monkeypatch, metadata
):
    with native_frame(monkeypatch) as frame:
        authorization = frame[0][3].authorization
        if metadata == "column":
            monkeypatch.setattr(RawTextDocument._meta.get_field("id"), "column", "pkid")
        else:
            monkeypatch.setattr(
                RawTextDocument._meta, "db_table", "altered_document_table"
            )
        with (
            bounded_reauthorization_reads(authorization, frame[4]),
            policy_data_counts() as counts,
        ):
            proof = frame[1]._native_policy.select(
                authorization=authorization,
                repository=ExtendedSeedRepository(),
                read_budget=frame[4],
            )
        assert proof is None and not counts


@pytest.mark.parametrize(
    "override",
    [
        "instance",
        "class",
        "supplier",
        "uuid_converter",
        "backend",
        "manager",
        "queryset",
        "user_field",
        "uuid_column",
        "repository",
    ],
)
def test_initial_customization_keeps_ordinary_fresh_policy_route(monkeypatch, override):
    from apps.collections.services import (
        native_closed_seed_policy_proof as proof_module,
    )
    from apps.collections.services.django_retrieval_authorization import (
        DjangoCollectionRetrievalPermissionPolicy,
    )
    from apps.documents.models import document as document_module

    with native_frame(monkeypatch) as frame:
        policy = frame[0][3].authorization.reauthorization_capability._policy
        if override == "instance":
            original = policy.current_authorized_selected_document_scope
            monkeypatch.setattr(
                policy,
                "current_authorized_selected_document_scope",
                lambda **kw: original(**kw),
            )
        elif override == "class":
            original = getattr(
                DjangoCollectionRetrievalPermissionPolicy,
                "current_authorized_selected_document_scope",
            )
            monkeypatch.setattr(
                DjangoCollectionRetrievalPermissionPolicy,
                "current_authorized_selected_document_scope",
                lambda self, **kw: original(self, **kw),
            )
        elif override == "supplier":
            original = document_module._get_document_types
            monkeypatch.setattr(
                document_module, "_get_document_types", lambda: original()
            )
        elif override == "uuid_converter":
            field = RawTextDocument._meta.get_field("id")
            monkeypatch.setitem(
                field.__dict__,
                "get_db_prep_value",
                lambda *args, **kw: field.__class__.get_db_prep_value(
                    field, *args, **kw
                ),
            )
        elif override == "backend":
            monkeypatch.setattr(proof_module, "_native_postgres", lambda _: False)
        elif override == "manager":
            manager = RawTextDocument._base_manager
            original = manager.get_queryset
            monkeypatch.setitem(manager.__dict__, "get_queryset", lambda: original())
        elif override == "queryset":
            from django.db.models import QuerySet

            original = QuerySet.filter
            monkeypatch.setattr(
                QuerySet,
                "filter",
                lambda self, *args, **kw: original(self, *args, **kw),
            )
        elif override == "user_field":
            from django.contrib.auth.models import User

            field = User._meta.get_field("is_active")
            original = field.get_db_prep_value
            monkeypatch.setitem(
                field.__dict__,
                "get_db_prep_value",
                lambda *args, **kw: original(*args, **kw),
            )
        elif override == "repository":
            original = ExtendedSeedRepository._load_seed_identities_batch
            monkeypatch.setattr(
                ExtendedSeedRepository,
                "_load_seed_identities_batch",
                lambda self, **kw: original(self, **kw),
            )
        else:
            # A native field type whose identity differs from the captured layout.
            field = RawTextDocument._meta.get_field("id")
            monkeypatch.setattr(field, "column", "id")
            replacement = field.clone()
            replacement.model = RawTextDocument
            replacement.set_attributes_from_name("id")
            monkeypatch.setitem(
                RawTextDocument._meta._forward_fields_map, "id", replacement
            )
        with policy_data_counts() as counts:
            prepare(frame)
        assert frame[1].mapping and (len(counts), sum(counts)) == (4, 8)


@pytest.mark.parametrize(
    "change",
    [
        "principal_object",
        "principal_key",
        "signature",
        "budget",
        "alias",
        "resigned_scope",
    ],
)
def test_original_authority_replacement_rejected_before_policy_dispatch(
    monkeypatch, change
):
    with native_frame(monkeypatch) as frame:
        active, owner, _, _, budget = frame
        authorization = active[3].authorization
        capability = authorization.reauthorization_capability
        if change == "principal_object":
            object.__setattr__(
                capability,
                "_principal",
                type(capability._principal).objects.get(pk=capability._principal.pk),
            )
        elif change == "principal_key":
            capability._principal.pk += 10000
        elif change == "signature":
            object.__setattr__(
                authorization, "authorization_context_signature", "0" * 64
            )
        elif change == "budget":
            from lib.retrieval.turn_budget import TurnBudget, TurnLimits

            object.__setattr__(budget, "turn_budget", TurnBudget(TurnLimits()))
        elif change == "alias":
            object.__setattr__(authorization, "database_alias", "projection_state")
        else:
            from apps.collections.services.retrieval_authorization import (
                _context_signature,
            )

            a = active[0].documents["A"]
            object.__setattr__(
                authorization, "selected_collection_ids", frozenset((a.collection_id,))
            )
            object.__setattr__(
                authorization, "selected_document_ids", frozenset((a.id,))
            )
            object.__setattr__(
                authorization,
                "authorization_context_signature",
                _context_signature(
                    authorization.principal_reference,
                    authorization.database_alias,
                    authorization.policy_version,
                    authorization.policy_checksum,
                    authorization.selected_collection_ids,
                    authorization.selected_document_ids,
                ),
            )
        with policy_data_counts() as counts, pytest.raises((ValueError, TypeError)):
            prepare(frame)
        assert not counts and owner.mapping is None


def test_native_final_policy_reservation_uses_one_unspent_row_and_keeps_sentinel(
    monkeypatch,
):
    with native_frame(monkeypatch, max_source_rows=52) as frame:
        prepare(frame)
        budget = frame[4]
        assert frame[1].mapping
        assert (budget.queries, budget.source_rows, budget._reserved_rows) == (6, 51, 0)
        assert budget.max_source_rows == 52


@pytest.mark.parametrize(
    "bad_rows", ["missing", "extra", "boolean", "negative", "short", "wrong"]
)
def test_malformed_actual_proof_results_never_authorize_or_retry(monkeypatch, bad_rows):
    statements = {
        "missing": "SELECT 2,2,2,2,2 WHERE FALSE",
        "extra": "SELECT 2,2,2,2,2 UNION ALL SELECT 2,2,2,2,2",
        "boolean": "SELECT TRUE,2,2,2,2",
        "negative": "SELECT -1,2,2,2,2",
        "short": "SELECT 2,2,2,2",
        "wrong": "SELECT 2,2,2,1,2",
    }
    with native_frame(monkeypatch) as frame:
        calls = []

        def corrupt(execute, sql, params, many, context):
            if sql.startswith("WITH native_seed_policy_permissions"):
                calls.append(True)
                return execute(statements[bad_rows], None, many, context)
            return execute(sql, params, many, context)

        with connection.execute_wrapper(corrupt), pytest.raises(ValueError):
            prepare(frame)
        assert calls == [True] and frame[1].mapping is None
        spent = (frame[4].queries, frame[4].source_rows)
        with (
            connection.execute_wrapper(corrupt),
            pytest.raises((ValueError, TimeoutError)),
        ):
            prepare(frame)
        assert calls == [True] and spent == (frame[4].queries, frame[4].source_rows)


@pytest.mark.parametrize(
    "change",
    [
        "principal",
        "method",
        "field",
        "settings",
        "late",
        "unknown",
        "manager",
        "user_field",
        "repository",
    ],
)
def test_mid_statement_changes_abort_after_actual_work_without_fallback(
    monkeypatch, change
):
    from asyncio import CancelledError

    with native_frame(monkeypatch) as frame:
        active, owner, _, _, budget = frame
        authorization = active[3].authorization
        calls = []

        def mutate(execute, sql, params, many, context):
            native = sql.startswith("WITH native_seed_policy_permissions")
            if native:
                calls.append(True)
            if native and change == "unknown":

                class Unknown:
                    rowcount = -1

                context["cursor"] = Unknown()
            result = execute(sql, params, many, context)
            if native:
                capability = authorization.reauthorization_capability
                if change == "principal":
                    from copy import copy

                    object.__setattr__(
                        capability, "_principal", copy(capability._principal)
                    )
                elif change == "method":
                    monkeypatch.setattr(
                        capability._policy,
                        "_principal_id",
                        lambda _: capability._principal.pk,
                    )
                elif change == "field":
                    monkeypatch.setattr(
                        RawTextDocument._meta.get_field("id"), "column", "replaced_id"
                    )
                elif change == "settings":
                    active[5].settings.graph_extended_max_seeds += 1
                elif change == "late":
                    active[2].budget.close("test-late-proof")
                elif change == "unknown":
                    pass
                elif change == "manager":
                    manager = RawTextDocument._base_manager
                    original = manager.get_queryset
                    monkeypatch.setitem(
                        manager.__dict__, "get_queryset", lambda: original()
                    )
                elif change == "repository":
                    monkeypatch.setattr(
                        ExtendedSeedRepository,
                        "_load_seed_identities_batch",
                        lambda self, **kw: {},
                    )
                else:
                    from django.contrib.auth.models import User

                    monkeypatch.setattr(
                        User._meta.get_field("is_active"), "column", "replaced_active"
                    )
            return result

        with (
            connection.execute_wrapper(mutate),
            pytest.raises((ValueError, TimeoutError, CancelledError)),
        ):
            prepare(frame)
        assert calls == [True] and owner.mapping is None and budget.queries == 1
        assert budget.source_rows == (0 if change == "unknown" else 1)
        assert budget._uncertain is (change == "unknown")


def test_actual_postgres_policy_timeout_poisons_original_ledger_without_retry(
    monkeypatch,
):
    from apps.knowledge_graph.retrieval.direct_seed_sql import DirectSeedReadTimeout

    with native_frame(monkeypatch, frontier_timeout=0.2) as frame:
        calls = []

        def delayed(execute, sql, params, many, context):
            if sql.startswith("WITH native_seed_policy_permissions"):
                calls.append(True)
                return execute("SELECT pg_sleep(1),2,2,2,2", None, many, context)
            return execute(sql, params, many, context)

        with connection.execute_wrapper(delayed), pytest.raises(DirectSeedReadTimeout):
            prepare(frame)
        budget = frame[4]
        assert calls == [True] and budget.queries == 1 and budget.source_rows == 0
        assert (
            budget._uncertain
            and budget._reserved_rows == 1
            and frame[1].mapping is None
        )
        with (
            connection.execute_wrapper(delayed),
            pytest.raises((ValueError, TimeoutError)),
        ):
            prepare(frame)
        assert calls == [True] and budget.queries == 1


def test_mutation_during_transaction_cleanup_is_rejected_after_charging(monkeypatch):
    from apps.knowledge_graph.retrieval import graph_seed_reads_v2 as reads

    with native_frame(monkeypatch) as frame:
        original = reads.bounded_seed_read

        @contextmanager
        def cleanup(**kwargs):
            with original(**kwargs):
                yield
            monkeypatch.setattr(
                RawTextDocument._meta.get_field("id"), "column", "cleanup_replacement"
            )

        monkeypatch.setattr(reads, "bounded_seed_read", cleanup)
        with pytest.raises(ValueError):
            prepare(frame)
        assert frame[4].queries == frame[4].source_rows == 1
        assert frame[1].mapping is None


def test_actual_postgres_plan_bounds_undeduplicated_inputs_before_count(monkeypatch):
    with native_frame(monkeypatch, large=True) as frame:
        active, owner, _, _, budget = frame
        with bounded_reauthorization_reads(active[3].authorization, budget):
            proof = owner._native_policy.select(
                authorization=active[3].authorization,
                repository=ExtendedSeedRepository(),
                read_budget=budget,
            )
            sql, params = proof._statement()
            with connection.cursor() as cursor:
                cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params)
                plan = cursor.fetchone()[0][0]["Plan"]

        def walk(node):
            yield node
            for child in node.get("Plans", ()):
                yield from walk(child)

        nodes = tuple(walk(plan))
        permissions = next(
            n
            for n in nodes
            if n.get("Subplan Name") == "CTE native_seed_policy_permissions"
        )
        documents = next(
            n
            for n in nodes
            if n.get("Subplan Name") == "CTE native_seed_policy_documents"
        )
        assert permissions["Node Type"] == documents["Node Type"] == "Limit"
        assert permissions["Actual Rows"] == 31 and documents["Actual Rows"] == 530
        assert documents["Plans"][0]["Node Type"] == "Append"
        assert len(documents["Plans"][0]["Plans"]) == 8
        # A permission semijoin may hash its permission input. The complete
        # document relation itself remains an undeduplicated Append below Limit.
        # Keep statement parameters, source bodies and filter expressions private.
        print(
            "native policy bounded plan:",
            [
                (n["Node Type"], n.get("Actual Rows"), n.get("Index Name"))
                for n in nodes
            ],
        )


@pytest.mark.parametrize(
    "change", ["body", "source_hash", "representation", "generation"]
)
def test_native_aggregate_keeps_late_source_and_projection_publication_fences(
    monkeypatch, change
):
    from uuid import uuid4

    from apps.documents.models import DocumentRepresentationState
    from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
    from apps.knowledge_graph.models import CollectionGraphProjection
    from apps.knowledge_graph.tests.test_graph_candidate_operation_races import (
        after_hydration,
    )

    with graph_frame(
        monkeypatch, prepare_sources=small_native_scope, dataset=True
    ) as active:
        f, graph, source, scope, _, dependencies, _, _ = active
        graph.authorization = scope.authorization
        dependencies.settings.graph_direct_enabled = False
        dependencies.settings.graph_extended_enabled = True
        dependencies.settings.graph_extended_timeout_ms = 5000
        snapshot = capture(active)
        proofs, observed = [], []

        def track(execute, sql, params, many, context):
            result = execute(sql, params, many, context)
            if sql.startswith("WITH native_seed_policy_permissions"):
                proofs.append(context["cursor"].rowcount)
            return result

        def mutate():
            assert proofs == [1, 1]
            observed.append(True)
            b = f.documents["B"]
            if change == "body":
                assert (
                    TextChunk.objects.filter(pk=f.chunks["B"].pk).update(
                        content="changed"
                    )
                    == 1
                )
            elif change == "source_hash":
                assert (
                    RawTextDocument.objects.filter(pk=b.pk).update(
                        full_text_hash="f" * 64
                    )
                    == 1
                )
            elif change == "representation":
                assert (
                    DocumentRepresentationState.objects.filter(document_id=b.id).update(
                        legacy_closed=True
                    )
                    == 1
                )
            else:
                assert (
                    CollectionGraphProjection.objects.filter(
                        pk=f.projections["B"].pk
                    ).update(generation_key=uuid4())
                    == 1
                )

        after_hydration(monkeypatch, mutate)
        with connection.execute_wrapper(track):
            rows, diagnostics = hybrid_graph_candidate_pool_v2(
                snapshot, "Atlas", f.authorization, dependencies
            )
        assert observed == [True]
        assert diagnostics["graph_status"] == ("miss" if change == "body" else "error")
        assert f.chunks["B"].pk not in {row.pk for row in rows}
        assert f.chunks["B"].pk not in {row.pk for row in source.cache.values()}
        assert f.chunks["B"].pk not in scope.bodies
