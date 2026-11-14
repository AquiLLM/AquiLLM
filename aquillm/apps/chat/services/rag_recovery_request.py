"""Exact ordinary answer work before optional source expansion, never authority."""

from dataclasses import dataclass
from hashlib import sha256

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.evidence_guard import ContextLimited
from lib.llm.turn_context import current_turn


class UnsupportedRecoveryRequest(ValueError):
    pass


def _conversation_key(conversation):
    return sha256(conversation.model_dump_json().encode()).hexdigest()


def decorate_preservation_packet(packet, *, assessment, stop_reason, notice=None, auxiliary=None):
    from .rag_coverage import recheck_support

    packet.coverage_assessment = recheck_support(assessment, packet.source_evidence)
    packet.auxiliary_handoff = auxiliary
    if (notice or packet.coverage_assessment.unresolved_aspects
            or stop_reason not in {"initial", "adequate_lookup", "assessed"}):
        packet.retrieval_status = "partial" if packet.chunks else "context_limited"
        packet.diagnostic_message = str(notice) if notice else (
            "Coverage is partial or unassessed. Answer supported "
            "portions at the requested depth and state any missing "
            "requested aspects; missing evidence is not proof of "
            "absence.")
    return packet


def preservation_conversation(conversation, question, raw, top_k, *, initial_acquired):
    from .rag_pipeline_messages import _append_retrieval_messages

    return (conversation if initial_acquired and conversation.messages and conversation[-1].role == "tool"
            else _append_retrieval_messages(conversation, question, raw, top_k))


@dataclass(frozen=True)
class RecoveryRequest:
    runtime: object
    budget: object
    turn: object
    conversation: object
    wrapper: object
    question: str
    llm: object
    max_tokens: int
    stage: str | None
    top_k: int
    conversation_key: str
    assessment: object = None
    stop_reason: str = "initial"
    notice: str | None = None
    auxiliary: object = None
    initial_acquired: bool = False

    def check(self):
        if (current_source_runtime() is not self.runtime or self.runtime.budget is not self.budget
                or current_turn() is not self.turn
                or (self.turn is not None and self.turn.budget is not self.budget)
                or _conversation_key(self.conversation) != self.conversation_key):
            raise SourceScopeUnavailable("recovery_original_scope_unavailable")

    def decorate(self, packet):
        if self.assessment is not None:
            decorate_preservation_packet(packet, assessment=self.assessment,
                stop_reason=self.stop_reason, notice=self.notice, auxiliary=self.auxiliary)
        return packet

    def conversation_for(self, raw, top_k):
        from lib.llm.providers.image_context import serialize_tool_result_for_llm

        if top_k != self.top_k:
            raise SourceScopeUnavailable("recovery_request_changed")
        last = self.wrapper[-1].model_copy(update={
            "content": serialize_tool_result_for_llm(raw), "result_dict": raw})
        return self.wrapper.model_copy(update={"messages": [*self.wrapper.messages[:-1], last]})

    def pay(self, selection, authorization, *, pin):
        from lib.llm.evidence_guard import prepay_request_tokens
        from lib.llm.providers.complete_turn_inputs import protected_post_tool_inputs
        from lib.llm.providers.openai import OpenAIInterface
        from lib.llm.providers.openai_request import protected_text_request_arguments
        from .rag_context_budget import provider_context_capacity
        from .rag_evidence import build_selected_evidence_packet
        from .rag_evidence_handoff import prepare_evidence_handoff
        from .rag_selection_coordinator import selected_tool_result
        from .rag_support_delivery import refresh_answer_support

        self.check()
        if (type(self.llm) is not OpenAIInterface or self.auxiliary is not None
                or type(self.llm.base_args) is not dict or set(self.llm.base_args) != {"model"}):
            raise UnsupportedRecoveryRequest("unsupported_recovery_request")
        packet = build_selected_evidence_packet(selection, query=self.question,
                                                search_scope="selected documents")
        packet.source_mode, packet.source_authorization, packet.selection = True, authorization, selection
        self.decorate(packet)
        if packet.image_urls or packet.auxiliary_figures or not packet.chunks:
            raise UnsupportedRecoveryRequest("unsupported_recovery_request")
        refresh_answer_support(packet)
        working = self.conversation_for(selected_tool_result(packet), self.top_k)
        _, request = prepare_evidence_handoff(working, packet)
        inputs = protected_post_tool_inputs(request, self.max_tokens, stage=self.stage)
        arguments = protected_text_request_arguments(self.llm, **inputs) if inputs is not None else None
        if arguments is None:
            raise UnsupportedRecoveryRequest("unsupported_recovery_request")
        tokens = prepay_request_tokens(arguments, turn_budget=self.budget, pin=pin)
        context, margin = provider_context_capacity(self.llm)
        self.check()
        if context <= 0 or tokens + max(self.max_tokens, arguments["max_tokens"]) + margin > context:
            raise ContextLimited("request_context_exhausted")
        return selection


def recovery_request(conversation, question, llm, *, max_tokens, stage,
                     assessment=None, stop_reason="initial", notice=None, auxiliary=None,
                     initial_acquired=False, top_k=None):
    from apps.documents.services.native_search_schedule import acquisition_schedule, scoped_baseline_recovery_enabled

    runtime = current_source_runtime()
    if runtime is None or not (scoped_baseline_recovery_enabled() or acquisition_schedule() == "native_closure"):
        return None
    from .rag_config import direct_rag_top_k
    top_k = direct_rag_top_k() if top_k is None else top_k
    wrapper = preservation_conversation(conversation, question, {}, top_k, initial_acquired=initial_acquired)
    return RecoveryRequest(runtime, runtime.budget, current_turn(), conversation, wrapper,
        question, llm, max_tokens, stage, top_k,
        _conversation_key(conversation), assessment, stop_reason, notice, auxiliary, initial_acquired)
