"""Real request preflight shared by native resource and answer-boundary tests."""

from types import SimpleNamespace


def answer_context(case, monkeypatch):
    from apps.chat.services.rag_recovery_request import recovery_request
    from apps.chat.services.rag_selection import select_evidence
    from apps.chat.services.rag_selection_types import SelectionLimits, SelectionProfile
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import ToolMessage, UserMessage

    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "131072")
    monkeypatch.setenv("PROMPT_BUDGET_CONTEXT_LIMIT", "131072")
    monkeypatch.setenv("OPENAI_STREAM_RESPONSES", "0")
    llm = OpenAIInterface(SimpleNamespace(base_url="http://vllm:8000/v1",
        chat=SimpleNamespace(completions=SimpleNamespace())), "local-test-model")
    conversation = Conversation(system="Answer using the cited source evidence.", messages=[
        UserMessage(content="Atlas"), ToolMessage(tool_name="vector_search", for_whom="assistant",
            content="Acquired source evidence", arguments={"search_string":"Atlas"}, result_dict={})])
    request = recovery_request(conversation, "Atlas", llm, max_tokens=512, stage=None,
                               initial_acquired=True, top_k=12)

    def preflight(prepared, *, pin):
        selection = select_evidence(prepared.candidates, profile=SelectionProfile("test",1,0,"v1"),
            limits=SelectionLimits(12,12,7000), score_status=prepared.score_status)
        return request.pay(selection, case.source.authorization, pin=pin)

    return SimpleNamespace(llm=llm, conversation=conversation, request=request,
                           preflight=preflight if request is not None else None)
