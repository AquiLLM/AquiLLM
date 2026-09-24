"""Common acquisition/selection/synthesis path for active preservation modes."""

from time import perf_counter

from apps.chat.services.rag_acquisition import acquire_evidence
from apps.chat.services.rag_action_tools import (
    execute_action,
    source_views,
)
from apps.chat.services.rag_config import (
    direct_rag_candidate_top_k,
    direct_rag_top_k,
    rag_preservation_config,
)
from apps.chat.services.rag_coverage import recheck_support
from apps.chat.services.rag_pipeline_messages import (
    _append_retrieval_messages,
    _latest_user_message,
)
from apps.chat.services.rag_selection_coordinator import (
    prepare_selection_turn,
    revalidate_selection_turn,
    selection_question,
)
from apps.chat.services.rag_source_continuity import (
    resolve_source_anchors,
)
from apps.chat.services.rag_source_continuity_turn import continuity_candidates
from apps.chat.services.rag_source_synthesis import LIMITED_MESSAGE
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.providers.request_observability import (
    new_correlation_id,
    observability_scope,
)
from lib.llm.turn_context import (
    bounded_retrieval,
    check_turn_active,
    suppress_evidence_handoff,
)
from lib.llm.types.messages import AssistantMessage

from .rag_normal_handoff import normal_source_handoff as normal_source_handoff


async def finish_preservation(
    consumer,
    llm,
    convo,
    *,
    question,
    initial_results=(),
    initial_acquired=False,
    vector_runner=None,
    stream_func=None,
    notice=None,
    iterative=None,
    auxiliary_handoff=None,
    prepare_fn=None,
    revalidate_fn=None,
    synthesis_fn=None,
):
    runtime = current_source_runtime()
    if runtime is None:
        return convo + [
            AssistantMessage(content=LIMITED_MESSAGE, stop_reason="end_turn")
        ]
    config = rag_preservation_config()
    started = perf_counter()
    budget = runtime.budget
    if not runtime.observation.get("question_charged"):
        if question and not budget.reserve_text(len(question), kind="materialized"):
            return convo + [
                AssistantMessage(content=LIMITED_MESSAGE, stop_reason="end_turn")
            ]
        runtime.observation["question_charged"] = True
    allowed = {
        str(v) for v in getattr(runtime.authorization, "selected_document_ids", ())
    }
    anchors = (
        resolve_source_anchors(question, convo)
        if config.followup_evidence_enabled
        else None
    )

    async def execute(action):
        return await execute_action(
            consumer,
            action,
            top_k=direct_rag_candidate_top_k(),
            vector_runner=vector_runner,
        )

    acquired = await acquire_evidence(
        question,
        budget=budget,
        llm=llm,
        execute=execute,
        evidence_views=source_views,
        anchors=anchors,
        allowed_documents=allowed,
        initial_results=initial_results,
        initial_acquired=initial_acquired,
        iterative=config.iterative_retrieval_enabled
        if iterative is None
        else iterative,
    )
    runtime.observation["acquisition"] = acquired
    retrieved = perf_counter()
    if not acquired.results:
        return convo + [
            AssistantMessage(content=notice or LIMITED_MESSAGE, stop_reason="end_turn")
        ]
    from apps.chat.services.tool_wiring.source_documents import (
        prepare_source_tool_handoff,
    )

    _, packet, raw = await bounded_retrieval(
        prepare_source_tool_handoff(
            consumer,
            llm,
            convo,
            acquired.results,
            question=question,
            top_k=direct_rag_top_k(),
            prepare_fn=prepare_fn or prepare_selection_turn,
            revalidate_fn=revalidate_fn or revalidate_selection_turn,
        ),
        budget,
    )
    assessment = recheck_support(acquired.assessment, packet.source_evidence)
    packet.coverage_assessment = assessment
    packet.auxiliary_handoff = auxiliary_handoff
    selected = perf_counter()
    if (
        notice
        or assessment.unresolved_aspects
        or acquired.stop_reason
        in {"partial_unknown", "completion_reserve", "action_limit", "no_progress"}
    ):
        packet.retrieval_status = "partial" if packet.chunks else "context_limited"
        packet.diagnostic_message = notice or (
            "Coverage is partial or unassessed. Answer supported "
            "portions at the requested depth and state any missing "
            "requested aspects; missing evidence is not proof of "
            "absence."
        )
    working = (
        convo
        if initial_acquired and convo.messages and convo[-1].role == "tool"
        else _append_retrieval_messages(convo, question, raw, direct_rag_top_k())
    )
    with (
        observability_scope(new_correlation_id(), "direct_synthesis"),
        suppress_evidence_handoff(),
    ):
        try:
            result = await (synthesis_fn or synthesize_from_evidence)(
                llm, working, packet, stream_func=stream_func
            )
        except Exception:
            from apps.chat.services.rag_synthesis import _extractive_summary

            frozen = runtime.observation.get("delivered_packet")
            if frozen is None or budget._synthesis is None:
                raise
            check_turn_active()
            result = working + [
                AssistantMessage(
                    content="Answer synthesis was unavailable. "
                    "Available authorized excerpts follow:\n\n"
                    + _extractive_summary(frozen),
                    stop_reason="end_turn",
                )
            ]
    check_turn_active()
    from apps.chat.services.rag_metrics import log_preservation_turn

    timings = {
        "retrieval_ms": (retrieved - started) * 1000,
        "evidence_ms": (selected - retrieved) * 1000,
        "synthesis_ms": (perf_counter() - selected) * 1000,
    }
    runtime.observation["timings"] = timings
    log_preservation_turn(
        runtime, acquired, runtime.observation.get("delivered_packet", packet), timings
    )
    return result


async def run_preservation_rag(
    consumer,
    llm,
    convo,
    *,
    stream_func=None,
    vector_runner=None,
    prepare_fn=None,
    revalidate_fn=None,
    synthesis_fn=None,
):
    config = rag_preservation_config()
    question = selection_question(convo, _latest_user_message(convo).content or "")
    prior, notice = await continuity_candidates(
        convo,
        question,
        user=consumer.user,
        selected_scope=consumer.col_ref.collections,
        enabled=config.followup_evidence_enabled,
    )
    if notice and not prior:
        consumer.convo = convo + [
            AssistantMessage(content=notice, stop_reason="end_turn")
        ]
        return "handled"
    try:
        result = await finish_preservation(
            consumer,
            llm,
            convo,
            question=question,
            initial_results=(prior,) if prior else (),
            initial_acquired=bool(notice),
            iterative=False if notice else None,
            notice=notice,
            vector_runner=vector_runner,
            stream_func=stream_func,
            prepare_fn=prepare_fn,
            revalidate_fn=revalidate_fn,
            synthesis_fn=synthesis_fn,
        )
    except Exception:
        result = convo + [
            AssistantMessage(content=LIMITED_MESSAGE, stop_reason="end_turn")
        ]
    check_turn_active()
    consumer.convo = result
    return "handled"
