"""Exact original policy with bounded selected-document PostgreSQL revalidation."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

import pytest
from django.contrib.auth.models import User

from apps.collections.models import Collection, CollectionPermission
from apps.collections.services.bounded_retrieval_authorization import (
    BoundedAuthorizationUnavailable,
    bounded_reauthorization_reads,
    resolve_read_budget,
)
from apps.collections.services.django_retrieval_authorization import (
    DjangoCollectionRetrievalPermissionPolicy,
)
from apps.collections.services.retrieval_authorization import (
    bind_retrieval_reauthorization_capability,
    freeze_retrieval_authorization_context,
    revalidate_retrieval_authorization_context,
)
from apps.documents.models import RawTextDocument
from apps.documents.tests.hybrid_graph_test_support import Policy, authorization
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture

pytestmark = pytest.mark.django_db


def fixture():
    user = User.objects.create_user(username="bounded-graph")
    collection = Collection.objects.create(name="bounded")
    permission = CollectionPermission.objects.create(
        user=user, collection=collection, permission="VIEW"
    )
    docs = []
    for index in range(40):
        text = f"synthetic-{index}"
        row = RawTextDocument(
            collection=collection,
            ingested_by=user,
            title=text,
            full_text=text,
            full_text_hash=sha256(text.encode()).hexdigest(),
        )
        row.save(dont_rechunk=True)
        docs.append(row)
    policy = DjangoCollectionRetrievalPermissionPolicy(b"synthetic-only")
    context = freeze_retrieval_authorization_context(
        principal=user,
        database_alias="default",
        policy=policy,
        selected_collection_ids=(collection.pk,),
        selected_document_ids=(docs[0].id,),
        reauthorization_capability=bind_retrieval_reauthorization_capability(
            principal=user, policy=policy
        ),
    )
    return context, permission, docs


@pytest.mark.parametrize("race", [None, "revoke", "move", "duplicate", "cross_model"])
def test_large_collection_tiny_original_subset_and_current_races(race):
    context, permission, docs = fixture()
    if race == "revoke":
        permission.delete()
    elif race == "move":
        RawTextDocument.objects.filter(pk=docs[0].pk).update(
            collection=Collection.objects.create(name="moved")
        )
    elif race == "duplicate":
        RawTextDocument.objects.filter(pk=docs[1].pk).update(id=docs[0].id)
    elif race == "cross_model":
        from apps.documents.models import TeXDocument

        d = docs[0]
        TeXDocument.objects.bulk_create(
            [
                TeXDocument(
                    id=d.id,
                    collection_id=d.collection_id,
                    ingested_by_id=d.ingested_by_id,
                    full_text=d.full_text,
                    full_text_hash=d.full_text_hash,
                    title=d.title,
                )
            ]
        )
    budget = FrontierFixture().budget()
    if race in ("duplicate", "cross_model"):
        with pytest.raises(ValueError):
            revalidate_retrieval_authorization_context(
                context=context, read_budget=budget
            )
    else:
        current = revalidate_retrieval_authorization_context(
            context=context, read_budget=budget
        )
        assert current.document_ids == ((docs[0].id,) if race is None else ())
    assert budget.source_rows <= 3
    assert 1 <= budget.queries <= 2


def test_too_small_budget_refuses_before_document_dispatch():
    context, _, _ = fixture()
    budget = FrontierFixture().budget(max_source_rows=1)
    with pytest.raises(FrontierBudgetExceeded):
        revalidate_retrieval_authorization_context(context=context, read_budget=budget)
    assert budget.queries == budget.source_rows == 1


def test_expired_original_deadline_prevents_authorization_data_dispatch():
    from time import monotonic

    from apps.knowledge_graph.retrieval.direct_seed_sql import DirectSeedReadTimeout

    context, _, _ = fixture()
    frontier = FrontierFixture()
    frontier.deadline = monotonic() - 1
    budget = frontier.budget()
    with pytest.raises(DirectSeedReadTimeout):
        revalidate_retrieval_authorization_context(context=context, read_budget=budget)
    assert budget.queries == budget.source_rows == 0


def test_unsupported_custom_policy_never_receives_a_callback():
    policy = Policy()
    context = authorization(policy)
    policy.current_authorized_document_scope = lambda **_: pytest.fail(
        "unbounded fallback"
    )
    with pytest.raises(BoundedAuthorizationUnavailable):
        revalidate_retrieval_authorization_context(
            context=context, read_budget=FrontierFixture().budget()
        )


def test_context_isolation_exception_reset_and_no_allowance_override():
    contexts = (authorization(Policy()), authorization(Policy()))
    budgets = (FrontierFixture().budget(), FrontierFixture().budget())

    def work(index):
        with pytest.raises(RuntimeError):
            with bounded_reauthorization_reads(contexts[index], budgets[index]):
                assert resolve_read_budget(contexts[index]) is budgets[index]
                with pytest.raises(ValueError):
                    resolve_read_budget(contexts[index], budgets[1 - index])
                with pytest.raises(ValueError):
                    with bounded_reauthorization_reads(
                        contexts[1 - index], budgets[index]
                    ):
                        pass
                raise RuntimeError("synthetic")
        assert resolve_read_budget(contexts[index]) is None

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(work, (0, 1)))
    assert resolve_read_budget(contexts[0]) is None
