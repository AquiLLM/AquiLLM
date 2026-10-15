"""Real PostgreSQL results, fresh authority reads and dispatch deadline fences."""

from dataclasses import replace
from time import monotonic
from uuid import uuid4

import pytest
from django.db import connection
from django.db.models import Case, F, UUIDField, Value, When
from django.test.utils import CaptureQueriesContext

from apps.collections.models import Collection
from apps.collections.services.django_retrieval_authorization import (
    DjangoCollectionRetrievalPermissionPolicy,
)
from apps.documents.models import PDFDocument, RawTextDocument
from apps.documents.services import _uuid_membership, representation_validation_rows
from apps.documents.services.representation_store import (
    validate_representation_snapshot,
)
from apps.documents.services.representation_tuple_predicate import ExactTupleMembership
from apps.knowledge_graph.retrieval.direct_seed_sql import DirectSeedReadTimeout
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetV2,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture

from .test_uuid_scope_membership_work import owners
from .uuid_preparation_support import fields

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("negated", [False, True])
@pytest.mark.parametrize("nullable", [False, True])
def test_uuid_positive_negative_and_current_null_results_match_generic(
    negated, nullable
):
    case = owners(3)
    chosen = case.documents[0]
    rows = ((chosen.id,), (chosen.id,), (uuid4(),))
    column = "current_uuid" if nullable else "id"
    actual = _uuid_membership.uuid_membership(
        (column,),
        rows,
        using="default",
        fields=fields(),
    )
    expected = ExactTupleMembership((column,), rows)
    base = RawTextDocument.objects.order_by("pk")
    if nullable:
        base = base.annotate(
            current_uuid=Case(
                When(
                    pk=case.documents[1].pk, then=Value(None, output_field=UUIDField())
                ),
                default=F("id"),
                output_field=UUIDField(),
            )
        )
    actual_query = base.filter(~actual if negated else actual).values_list(
        "id", flat=True
    )
    generic_query = base.filter(~expected if negated else expected).values_list(
        "id", flat=True
    )
    assert actual_query.query.sql_with_params() == generic_query.query.sql_with_params()
    wanted = (
        [d.id for d in case.documents[(2 if nullable else 1) :]]
        if negated
        else [chosen.id]
    )
    assert list(actual_query) == list(generic_query) == wanted


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "missing",
        "move",
        "body",
        "revoke",
        "inactive",
        "duplicate",
        "stale_generation",
    ],
)
def test_fresh_result_and_physical_read_counts_equal_generic(change, monkeypatch):
    case = owners(3)
    victim = case.documents[0]
    snapshot = case.snapshot
    if change == "missing":
        RawTextDocument.objects.filter(pk=victim.pk).delete()
    elif change == "move":
        RawTextDocument.objects.filter(pk=victim.pk).update(
            collection=Collection.objects.create(name="Moved current UUID owner"),
        )
    elif change == "body":
        RawTextDocument.objects.filter(pk=victim.pk).update(
            full_text="Untracked body edit"
        )
    elif change == "revoke":
        case.permission.delete()
    elif change == "inactive":
        type(case.user).objects.filter(pk=case.user.pk).update(is_active=False)
    elif change == "duplicate":
        PDFDocument.objects.bulk_create(
            [
                PDFDocument(
                    id=victim.id,
                    title="Second concrete owner",
                    full_text=victim.full_text,
                    full_text_hash=victim.full_text_hash,
                    collection=case.collection,
                    ingested_by=case.user,
                )
            ]
        )
    elif change == "stale_generation":
        first, *remaining = snapshot.selections
        snapshot = replace(
            snapshot,
            selections=(
                replace(
                    first,
                    generation_key=uuid4(),
                    source_revision_sha256="f" * 64,
                ),
                *remaining,
            ),
        )
    policy = DjangoCollectionRetrievalPermissionPolicy(b"fresh-uuid-differential")

    def measure():
        budget = FrontierFixture().budget()
        with CaptureQueriesContext(connection) as reads:
            try:
                source = validate_representation_snapshot(snapshot, using="default")
                source = source is snapshot
            except ValueError as error:
                source = type(error), error.args
            try:
                authority = policy.current_authorized_selected_document_scope(
                    principal=case.user,
                    database_alias="default",
                    selected_collection_ids=(case.collection.pk,),
                    selected_document_ids=tuple(d.id for d in case.documents),
                    read_budget=budget,
                )
            except ValueError as error:
                authority = type(error), error.args
        return source, authority, len(reads), budget.queries, budget.source_rows

    optimized = measure()

    def generic(columns, rows, **kwargs):
        return ExactTupleMembership(columns, rows)

    monkeypatch.setattr(_uuid_membership, "uuid_membership", generic)
    monkeypatch.setattr(representation_validation_rows, "uuid_membership", generic)
    assert measure() == optimized
    assert optimized[3] == (1 if change in ("revoke", "inactive") else 2)


def test_native_deadline_expiring_after_eager_preparation_prevents_owner_dispatch(
    monkeypatch,
):
    case = owners(3)
    initial = FrontierFixture().budget()
    now = [monotonic()]
    budget = FrontierBudgetV2(
        turn_budget=initial.turn_budget,
        deadline=initial.deadline,
        clock=lambda: now[0],
        **{
            name: getattr(initial, name)
            for name in (
                "max_identities",
                "max_edges",
                "max_source_rows",
                "max_queries",
                "max_wire_bytes",
            )
        },
    )
    original = _uuid_membership.uuid_membership
    payloads, dispatched = [], []

    def prepare_then_expire(*args, **kwargs):
        predicate = original(*args, **kwargs)
        payloads.append(predicate._payload)
        now[0] = budget.deadline
        return predicate

    def execute(real, sql, params, many, context):
        if "UNION ALL" in sql:
            dispatched.append(True)
        return real(sql, params, many, context)

    monkeypatch.setattr(_uuid_membership, "uuid_membership", prepare_then_expire)
    policy = DjangoCollectionRetrievalPermissionPolicy(b"deadline-uuid-test")
    with connection.execute_wrapper(execute), pytest.raises(DirectSeedReadTimeout):
        policy.current_authorized_selected_document_scope(
            principal=case.user,
            database_alias="default",
            selected_collection_ids=(case.collection.pk,),
            selected_document_ids=tuple(d.id for d in case.documents),
            read_budget=budget,
        )
    assert len(payloads) == 1 and not dispatched
    assert budget.queries == 1 and budget.source_rows == 1
