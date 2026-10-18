"""Planner context keeps source coordinates exact within a diverse bounded view."""

import json
from types import SimpleNamespace

import pytest

from apps.chat.services.rag_coverage import assess_coverage
from lib.retrieval.evidence import SourceEvidence
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def source(pk, document, length):
    return SourceEvidence(pk, document, pk, f"revision-{pk}", str(pk) * length)


async def planner_prompt(sources, **kwargs):
    requests = []

    async def message(**arguments):
        requests.append(json.loads(arguments["messages"][0]["content"]))
        return SimpleNamespace(
            text=json.dumps(
                {
                    "requested_aspects": ["mechanism"],
                    "support": [],
                    "unresolved_aspects": ["mechanism"],
                    "next_action": None,
                }
            ),
            stop_reason="stop",
        )

    await assess_coverage(
        "Explain the mechanism.",
        sources,
        None,
        llm=SimpleNamespace(base_args={}, get_message=message),
        budget=TurnBudget(TurnLimits()),
        allowed_documents={s.document_id for s in sources},
        **kwargs,
    )
    return requests[0]


@pytest.mark.asyncio
async def test_first_document_cannot_hide_fitting_second_document():
    prompt = await planner_prompt(
        (source(1, "a", 6000), source(2, "a", 6000), source(3, "b", 6000))
    )
    assert [s["chunk_id"] for s in prompt["evidence"]] == [1, 3]
    assert sum(len(s["text"]) for s in prompt["evidence"]) <= 16000


@pytest.mark.asyncio
async def test_later_planner_prioritizes_new_group_and_new_view_within_group():
    sources = (
        source(1, "a", 4000),
        source(2, "b", 4000),
        source(3, "b", 4000),
        source(4, "c", 4000),
        source(5, "a", 4000),
    )
    prompt = await planner_prompt(sources, latest_source_ids=(sources[2].identity,))
    assert [s["chunk_id"] for s in prompt["evidence"]] == [3, 1, 4, 2]


@pytest.mark.asyncio
async def test_duplicate_source_identity_uses_context_only_once():
    first, second = source(1, "a", 4000), source(2, "b", 6000)
    prompt = await planner_prompt((first, first, second))
    assert [s["chunk_id"] for s in prompt["evidence"]] == [1, 2]
    assert prompt["omitted_sources"] == 0


@pytest.mark.asyncio
async def test_oversized_view_is_omitted_without_changing_remaining_offsets():
    sources = (source(1, "a", 16001), source(2, "a", 6000), source(3, "b", 6000))
    prompt = await planner_prompt(sources)
    assert [s["chunk_id"] for s in prompt["evidence"]] == [2, 3]
    assert prompt["evidence"][0]["text"] == sources[1].text
    assert prompt["evidence"][1]["source_fingerprint"] == sources[2].source_fingerprint
    assert prompt["omitted_sources"] == 1
    assert prompt["action_coordinates"]["adjacent_sources"] == [
        {"document_id": "a", "chunk_id": 2},
        {"document_id": "b", "chunk_id": 3},
    ]


def test_whole_view_helper_preserves_original_source_objects():
    from apps.chat.services.rag_planner_context import planner_views

    first, another, second = source(1, "a", 4), source(2, "a", 4), source(3, "b", 4)
    views = planner_views((first, another, second), max_chars=8)
    assert views == (first, second)
    assert views[0] is first and views[1] is second


@pytest.mark.asyncio
async def test_prompt_history_is_bounded_and_keeps_original_question():
    from apps.chat.services.rag_planner_context import AcquisitionAttempt

    attempt = AcquisitionAttempt(
        "vector", "method " * 400, None, None, "completed", 2, 1
    )
    prompt = await planner_prompt((source(1, "a", 8),), attempts=(attempt,) * 5)
    assert prompt["question"] == "Explain the mechanism."
    assert len(prompt["attempts"]) == 3
    assert prompt["attempts"][0] == {
        "kind": "vector",
        "query": ("method " * 400)[:2000],
        "document_id": None,
        "chunk_id": None,
        "outcome": "completed",
        "available_source_count": 2,
        "new_source_count": 1,
        "query_truncated": True,
    }


@pytest.mark.asyncio
async def test_denied_action_does_not_become_an_admitted_attempt():
    from apps.chat.services.rag_acquisition import acquire_evidence

    async def denied(action):
        return {"sources": []}

    result = await acquire_evidence(
        "Explain the mechanism.",
        budget=TurnBudget(TurnLimits()),
        llm=None,
        execute=denied,
        evidence_views=lambda _: (),
    )
    assert result.attempts == ()
    assert result.rounds == 0


def acquisition_source(pk, text):
    from lib.retrieval.evidence import fingerprint_source

    return SourceEvidence(pk, "doc", pk, fingerprint_source(text), text)


@pytest.mark.asyncio
async def test_next_planner_gets_attempts_and_revision_novelty_from_cached_union():
    from apps.chat.services.rag_acquisition import acquire_evidence
    from apps.chat.services.rag_coverage import AcquisitionAction, CoverageAssessment

    budget = TurnBudget(TurnLimits())
    cache, planned = [], []
    first, revised = (
        acquisition_source(1, "method first revision"),
        acquisition_source(1, "method next revision"),
    )

    async def execute(action):
        assert budget.reserve_action(action.signature)
        cache.extend([first, first] if not cache else [revised, revised])
        return {"result": []}  # Evidence comes from the runtime cache, not this body.

    async def assess(question, views, anchors, **kwargs):
        planned.append(kwargs)
        return CoverageAssessment(
            ("mechanism",),
            (),
            ("mechanism",),
            AcquisitionAction(
                "document", "observed mechanism", "doc", aspect="mechanism"
            )
            if len(planned) == 1
            else None,
            "assessed",
        )

    result = await acquire_evidence(
        "Explain the mechanism.",
        budget=budget,
        llm=None,
        execute=execute,
        evidence_views=lambda _: tuple(cache),
        assess=assess,
        allowed_documents={"doc"},
        iterative=True,
    )

    assert result.stop_reason == "assessed"
    assert len(planned) == 2
    attempts = planned[1].get("attempts", ())
    assert len(attempts) == 2
    assert [(a.kind, a.query, a.document_id) for a in attempts] == [
        ("vector", "Explain the mechanism.", None),
        ("document", "observed mechanism", "doc"),
    ]
    assert [(a.available_source_count, a.new_source_count) for a in attempts] == [
        (1, 1),
        (2, 1),
    ]
    assert planned[1]["latest_source_ids"] == (revised.identity,)
    assert len(result.attempts) == budget.actions_used == 2
    assert "method first revision" not in repr(result.attempts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,outcome",
    [
        (RuntimeError("private provider body"), "failure"),
        (TimeoutError("private timeout body"), "timeout"),
    ],
)
async def test_failed_admitted_action_records_safe_history_and_retains_sources(
    error, outcome
):
    from apps.chat.services.rag_acquisition import acquire_evidence
    from apps.chat.services.rag_coverage import AcquisitionAction, CoverageAssessment

    budget = TurnBudget(TurnLimits())
    calls = []
    failed = False

    async def execute(action):
        nonlocal failed
        assert budget.reserve_action(action.signature)
        calls.append(action)
        if len(calls) == 2:
            failed = True
            raise error
        return {"sources": [acquisition_source(1, "method only")]}

    def views(results):
        assert not failed, "failed worker may still own source runtime locks"
        return tuple(s for r in results for s in r["sources"])

    async def assess(*args, **kwargs):
        return CoverageAssessment(
            ("mechanism",),
            (),
            ("mechanism",),
            AcquisitionAction("vector", "method details", aspect="mechanism"),
        )

    result = await acquire_evidence(
        "Explain the mechanism.",
        budget=budget,
        llm=None,
        execute=execute,
        evidence_views=views,
        assess=assess,
        iterative=True,
    )

    assert result.stop_reason == "acquisition_" + outcome
    assert len(result.results) == 1
    attempts = getattr(result, "attempts", ())
    assert len(attempts) == budget.actions_used == 2
    assert (attempts[-1].outcome, attempts[-1].new_source_count) == (outcome, 0)
    assert "private" not in repr(attempts)


@pytest.mark.asyncio
async def test_injected_planner_cannot_use_document_omitted_from_visible_context():
    from apps.chat.services.rag_acquisition import acquire_evidence
    from apps.chat.services.rag_coverage import AcquisitionAction, CoverageAssessment

    budget = TurnBudget(TurnLimits())
    calls = []

    async def execute(action):
        calls.append(action)
        assert budget.reserve_action(action.signature)
        return {"sources": [SourceEvidence(1, "omitted", 0, "revision", "x" * 16001)]}

    async def assess(*args, **kwargs):
        return CoverageAssessment(
            ("mechanism",),
            (),
            ("mechanism",),
            AcquisitionAction(
                "document", "method details", "omitted", aspect="mechanism"
            ),
        )

    result = await acquire_evidence(
        "Explain the mechanism.",
        budget=budget,
        llm=None,
        execute=execute,
        evidence_views=lambda results: tuple(s for r in results for s in r["sources"]),
        assess=assess,
        allowed_documents={"omitted"},
        iterative=True,
    )
    assert len(calls) == 1
    assert result.stop_reason == "planner_failure"
