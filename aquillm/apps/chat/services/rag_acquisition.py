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
from .rag_planner_context import acquisition_attempt, planner_views

COMPLETION_RESERVE_MS = 1000
# Admission floor, not a deadline: a follow-up may use the remaining turn time.
ACTION_TIMEOUT_MS = 3000


@dataclass(frozen=True)
class AcquiredEvidence:
    results: tuple
    assessment: CoverageAssessment
    stop_reason: str
    rounds: int
    attempts: tuple = ()


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
    query_proposals=(),
):
    results = list(initial_results)
    assessment = CoverageAssessment()
    reason = "initial"
    phase = "acquisition"
    attempts, latest_source_ids = [], ()

    def source_ids(current_results):
        return tuple(dict.fromkeys(s.identity for s in evidence_views(current_results)))

    async def run(action, proposal=None, *, initial=False):
        nonlocal latest_source_ids
        from lib.llm.turn_context import tool_timeout
        from lib.retrieval.operation import operation_scope

        timeout = min(
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
        source_limit = None
        if (
            initial
            and iterative
            and budget.limits.actions - budget.actions_used > 1
            and budget.planner_calls < budget.limits.planner_calls
        ):
            # Reserve one third of the remaining novel-source allowance. Native
            # graph admission and body materialization both honor this ceiling.
            free = budget.remaining_source_capacity()
            source_limit = budget.sources_used + free - free // 3
        before_ids, actions_before = source_ids(results), budget.actions_used
        completed_results, outcome = [], "failure"
        latest_source_ids = ()
        try:
            with operation_scope(budget, timeout_ms=timeout, source_limit=source_limit):
                from .retrieval_query_runtime import action_observation

                observation = action_observation(action, proposal, budget)
                publish(
                    "action_start",
                    observation,
                )
                result = await asyncio.wait_for(execute(action), timeout=timeout / 1000)
                completed_results, outcome = [result], "completed"
                if active():
                    publish(
                        "action_end",
                        {
                            "signature": observation["signature"],
                            "ledger_id": str(id(budget)),
                            "sources": [
                                str(s.identity) for s in evidence_views([result])
                            ],
                        },
                    )
                return result
        except TimeoutError:
            outcome = "timeout"
            raise
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            if (
                budget.actions_used > actions_before
                and len(attempts) < budget.limits.actions
            ):
                # A failed or cancelled worker can still own runtime locks.
                # Without an accepted result, retain the pre-action snapshot.
                after_ids = (
                    source_ids([*results, *completed_results])
                    if outcome == "completed"
                    else before_ids
                )
                previous_ids = set(before_ids)
                latest_source_ids = tuple(
                    identity for identity in after_ids if identity not in previous_ids
                )
                attempts.append(
                    acquisition_attempt(action, before_ids, after_ids, outcome=outcome)
                )

    try:
        if initial_acquired and budget.actions_used:
            latest_source_ids = source_ids(results)
            attempts.append(
                acquisition_attempt(None, (), latest_source_ids, outcome="external")
            )
        if not initial_acquired and budget.actions_used < budget.limits.actions:
            from apps.documents.services.source_loading import current_source_runtime

            from .retrieval_query_runtime import run_optional_proposal

            runtime = current_source_runtime()
            prepared = getattr(runtime, "_query_proposals", ())
            if prepared and prepared == query_proposals:
                if prepared[0].text != question or prepared[0].kind != "original":
                    raise ValueError("original query changed")
                results.append(
                    await run_optional_proposal(
                        prepared[0],
                        anchors=anchors,
                        run=lambda action: run(action, initial=True),
                    )
                )
            else:
                results.append(
                    await run(AcquisitionAction("vector", question), initial=True)
                )
        for proposal in query_proposals[1:3]:
            if budget.actions_used >= budget.limits.actions:
                break
            if not budget.can_start_optional(ACTION_TIMEOUT_MS, COMPLETION_RESERVE_MS):
                break
            from .retrieval_query_runtime import run_optional_proposal

            result = await run_optional_proposal(
                proposal, anchors=anchors, run=lambda action: run(action, proposal)
            )
            results.append(result)
        for _ in range(budget.limits.planner_calls if iterative else 0):
            views = evidence_views(results)
            if not needs_coverage_assessment(question, views, anchors):
                assessment = lookup_coverage(question, views, anchors)
                reason = "adequate_lookup"
                break
            if budget.actions_used >= budget.limits.actions:
                reason = "action_limit"
                break
            if budget.remaining_source_capacity() <= 0:
                reason = "source_limit"
                break
            if not budget.can_start_optional(
                budget.limits.planner_call_ms + ACTION_TIMEOUT_MS, COMPLETION_RESERVE_MS
            ):
                reason = "completion_reserve"
                break
            if not budget.reserve_planner():
                reason = "planner_limit"
                break
            phase = "planner"
            assessment = await asyncio.wait_for(
                assess(
                    question,
                    views,
                    anchors,
                    llm=llm,
                    budget=budget,
                    allowed_documents=allowed_documents,
                    attempts=tuple(attempts),
                    latest_source_ids=latest_source_ids,
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
                action,
                assessment.unresolved_aspects,
                planner_views(views, latest_source_ids=latest_source_ids),
                allowed_documents,
            )
            if budget.has_action(action.signature):
                reason = "duplicate_action"
                break
            if not budget.can_start_optional(ACTION_TIMEOUT_MS, COMPLETION_RESERVE_MS):
                reason = "completion_reserve"
                break
            before = {s.identity for s in views}
            phase = "acquisition"
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
    except Exception as exc:
        from .rag_coverage import PlannerFailure

        if isinstance(exc, PlannerFailure):
            reason = "planner_" + exc.reason
        elif isinstance(exc, TimeoutError):
            reason = phase + "_timeout"
        else:
            reason = phase + "_failure"
        # Never log the exception message: provider errors can contain evidence.
        publish("acquisition_failure", {"phase": phase, "reason": reason})
    return AcquiredEvidence(
        tuple(results), assessment, reason, budget.actions_used, tuple(attempts)
    )
