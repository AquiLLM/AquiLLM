"""Returned retrieval order survives exact-source verification and planner packing."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from apps.chat.services.rag_acquisition import acquire_evidence
from apps.chat.services.rag_action_tools import source_views
from apps.chat.services.rag_coverage import AcquisitionAction, CoverageAssessment
from apps.chat.services.rag_planner_context import attempt_context, planner_views
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.retrieval.evidence import fingerprint_source
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def row(pk, document, text, number=0):
    return SimpleNamespace(pk=pk, doc_id=document, content=text, chunk_number=number)


def cache_rows(runtime, *rows):
    for cached in rows:
        runtime.cache[cached.pk, fingerprint_source(cached.content)] = cached


def result(*rows):
    return {
        "result": [],
        "_source_provenance": [
            {
                "chunk_id": cached.pk,
                "document_id": str(cached.doc_id),
                "chunk_number": cached.chunk_number,
                "source_fingerprint": fingerprint_source(cached.content),
                "source_codepoints": len(cached.content),
            }
            for cached in rows
        ],
    }


@pytest.mark.parametrize("cache_order", [(0, 1, 2), (2, 1, 0)])
def test_views_follow_returned_provenance_order_not_cache_insertion(cache_order):
    rows = (
        row(11, "first", "first exact body"),
        row(22, "second", "second exact body"),
        row(33, "third", "third exact body"),
    )
    runtime = SourceRuntime(TurnBudget(TurnLimits()), None)
    cache_rows(runtime, *(rows[index] for index in cache_order))

    with source_runtime_scope(runtime):
        views = source_views([result(rows[1], rows[2], rows[0])])

    assert [source.chunk_id for source in views] == [22, 33, 11]
    assert [source.text for source in views] == [
        "second exact body",
        "third exact body",
        "first exact body",
    ]
    assert all(
        source.source_fingerprint == fingerprint_source(source.text) for source in views
    )


def test_only_matching_materialized_provenance_survives_in_returned_order():
    first = row(11, "first", "first current body")
    second = row(22, "second", "second current body")
    stale = row(33, "stale", "old body")
    unreturned = row(44, "unreturned", "cached but not returned")
    missing = row(55, "missing", "returned but not materialized")
    mismatched = row(66, "mismatched", "materialized body")
    runtime = SourceRuntime(TurnBudget(TurnLimits()), None)
    cache_rows(runtime, first, stale, unreturned, mismatched, second)
    returned = result(missing, second, row(33, "stale", "new body"), mismatched, first)
    returned["_source_provenance"][3]["source_fingerprint"] = "not-the-body-revision"

    with source_runtime_scope(runtime):
        views = source_views([returned])

    assert [(source.chunk_id, source.text) for source in views] == [
        (22, "second current body"),
        (11, "first current body"),
    ]


def test_duplicate_accepted_identity_keeps_first_returned_position():
    first = row(11, "first", "first body")
    second = row(22, "second", "second body")
    third = row(33, "third", "third body")
    runtime = SourceRuntime(TurnBudget(TurnLimits()), None)
    cache_rows(runtime, third, first, second)

    with source_runtime_scope(runtime):
        views = source_views([result(second, first, second), result(third, first)])

    assert [source.chunk_id for source in views] == [22, 11, 33]
    assert len({source.identity for source in views}) == 3


@pytest.mark.parametrize("new_revision_cached_first", [False, True])
def test_superseded_revision_is_excluded_before_matching_identity_is_ordered(
    new_revision_cached_first,
):
    old = row(11, "first", "old body")
    current = row(11, "first", "current body")
    other = row(22, "second", "other body")
    runtime = SourceRuntime(TurnBudget(TurnLimits()), None)
    revisions = (current, old) if new_revision_cached_first else (old, current)
    cache_rows(runtime, *revisions, other)

    with source_runtime_scope(runtime):
        views = source_views([result(old, other), result(current, current)])

    assert [(source.chunk_id, source.text) for source in views] == [
        (22, "other body"),
        (11, "current body"),
    ]


def test_no_runtime_cannot_materialize_provenance_only_sources():
    assert source_views([result(row(11, "first", "unmaterialized"))]) == ()


@pytest.mark.asyncio
async def test_successive_actions_keep_latest_priority_and_unknown_external_query():
    budget = TurnBudget(TurnLimits())
    runtime = SourceRuntime(budget, None)
    first = row(11, "first", "first mechanism excerpt")
    second = row(22, "second", "second mechanism excerpt")
    latest_same_document = row(33, "second", "new second-document excerpt", 1)
    latest_document = row(44, "third", "new third-document excerpt")
    cache_rows(runtime, first, second)
    assert budget.reserve_action("external-initial-action-with-unknown-query")
    planned = []

    async def execute(action):
        assert budget.reserve_action(action.signature)
        cache_rows(runtime, latest_document, latest_same_document)
        return result(latest_same_document, latest_document)

    async def assess(question, views, anchors, **kwargs):
        visible = planner_views(views, latest_source_ids=kwargs["latest_source_ids"])
        planned.append(
            {
                "sources": [asdict(source) for source in visible],
                "latest": [
                    identity.chunk_id for identity in kwargs["latest_source_ids"]
                ],
                "attempts": attempt_context(kwargs["attempts"], 3),
            }
        )
        return CoverageAssessment(
            ("mechanism",),
            (),
            ("mechanism",),
            AcquisitionAction(
                "document", "mechanism details", "second", aspect="mechanism"
            )
            if len(planned) == 1
            else None,
            "assessed",
        )

    with source_runtime_scope(runtime):
        acquired = await acquire_evidence(
            "Explain the mechanism.",
            budget=budget,
            llm=None,
            execute=execute,
            evidence_views=source_views,
            initial_results=(result(second, first),),
            initial_acquired=True,
            allowed_documents={"first", "second", "third"},
            iterative=True,
            assess=assess,
        )

    assert acquired.stop_reason == "assessed"
    assert [source["chunk_id"] for source in planned[0]["sources"]] == [22, 11]
    assert [source["chunk_id"] for source in planned[1]["sources"]] == [33, 44, 11, 22]
    assert planned[1]["latest"] == [33, 44]
    assert planned[0]["attempts"] == [
        {
            "kind": "external",
            "query": None,
            "document_id": None,
            "chunk_id": None,
            "outcome": "external",
            "available_source_count": 2,
            "new_source_count": 2,
            "query_truncated": False,
        }
    ]
    assert planned[1]["attempts"][0] == planned[0]["attempts"][0]
    assert planned[1]["attempts"][1]["query"] == "mechanism details"
    assert all(
        set(source)
        == {"chunk_id", "document_id", "chunk_number", "source_fingerprint", "text"}
        for step in planned
        for source in step["sources"]
    )
