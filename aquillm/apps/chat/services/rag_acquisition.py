"""Gather first within one ledger; selection and synthesis belong to the caller."""

import asyncio
from dataclasses import dataclass, replace

from lib.evidence_observation import active, publish

from .rag_coverage import (
    AcquisitionAction,
    CoverageAssessment,
    assess_coverage,
    needs_coverage_assessment,
    validate_action,
)
from .rag_lookup_coverage import lookup_coverage

COMPLETION_RESERVE_MS = 1000
ACTION_TIMEOUT_MS = 3000


@dataclass(frozen=True)
class AcquiredEvidence:
    results: tuple
    assessment: CoverageAssessment
    stop_reason: str
    rounds: int


async def acquire_evidence(
    question,
    *,
    budget,
    llm,
    execute,
    evidence_views,
    anchors=None,
    allowed_documents=(),
    iterative=False,
    initial_results=(),
    initial_acquired=False,
    assess=assess_coverage,
):
    results = list(initial_results)
    assessment = CoverageAssessment()
    reason = "initial"

    async def run(action):
        from lib.llm.turn_context import tool_timeout
        from lib.retrieval.operation import operation_scope

        timeout = min(
            ACTION_TIMEOUT_MS,
            tool_timeout(retrieval=True) * 1000,
            max(
                0,
                budget.remaining_ms()
                - budget.limits.final_scoring_ms
                - COMPLETION_RESERVE_MS,
            ),
        )
        if timeout <= 0:
            raise TimeoutError("acquisition reserve")
        with operation_scope(budget):
            publish(
                "action_start",
                {
                    "kind": action.kind,
                    "query": action.query,
                    "signature": action.signature,
                    "ledger_id": str(id(budget)),
                },
            )
            result = await asyncio.wait_for(execute(action), timeout=timeout / 1000)
            if active():
                publish(
                    "action_end",
                    {
                        "signature": action.signature,
                        "ledger_id": str(id(budget)),
                        "sources": [str(s.identity) for s in evidence_views([result])],
                    },
                )
            return result

    try:
        if not initial_acquired and budget.actions_used < budget.limits.actions:
            results.append(await run(AcquisitionAction("vector", question)))
        for _ in range(budget.limits.planner_calls if iterative else 0):
            views = evidence_views(results)
            if not needs_coverage_assessment(question, views, anchors):
                assessment = lookup_coverage(question, views, anchors)
                reason = "adequate_lookup"
                break
            if budget.actions_used >= budget.limits.actions:
                reason = "action_limit"
                break
            if not budget.can_start_optional(
                budget.limits.planner_call_ms + ACTION_TIMEOUT_MS, COMPLETION_RESERVE_MS
            ):
                reason = "completion_reserve"
                break
            if not budget.reserve_planner():
                reason = "planner_limit"
                break
            assessment = await asyncio.wait_for(
                assess(
                    question,
                    views,
                    anchors,
                    llm=llm,
                    budget=budget,
                    allowed_documents=allowed_documents,
                ),
                timeout=budget.limits.planner_call_ms / 1000,
            )
            if active():
                from dataclasses import asdict

                publish("coverage_assessment", asdict(assessment))
            action = assessment.next_action
            if action is None:
                reason = "assessed"
                break
            validate_action(
                action, assessment.unresolved_aspects, views, allowed_documents
            )
            if budget.has_action(action.signature):
                reason = "duplicate_action"
                break
            if not budget.can_start_optional(ACTION_TIMEOUT_MS, COMPLETION_RESERVE_MS):
                reason = "completion_reserve"
                break
            before = {s.identity for s in views}
            results.append(await run(action))
            after = {s.identity for s in evidence_views(results)}
            if not after - before:
                reason = "no_progress"
                break
            assessment = replace(assessment, certainty="unassessed", next_action=None)
            reason = "action_limit"
    except asyncio.CancelledError:
        budget.close("cancelled")
        raise
    except (Exception,):
        reason = "partial_unknown"
    return AcquiredEvidence(tuple(results), assessment, reason, budget.actions_used)
