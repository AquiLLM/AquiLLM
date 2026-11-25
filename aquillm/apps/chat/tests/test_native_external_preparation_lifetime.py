"""Exact outer result identity is required to resume native work once."""

from contextlib import contextmanager
from types import SimpleNamespace

from asgiref.sync import async_to_sync
import pytest

from apps.chat.services import rag_external_preparation as external
from apps.chat.services.rag_preparation_order import take_preparation_order
from apps.chat.services.rag_recovery_origins import _ACTION, start_recovery_origins
from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
from apps.chat.tests.test_native_preparation_order import native_frame
from apps.chat.tests.test_recovery_origins import payload, source
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.llm.execution_context import bind_execution, run_tool_once
from lib.llm.turn_context import call_tool_async
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


@contextmanager
def outer_case(monkeypatch, *, replay=False):
    with native_frame(monkeypatch) as case:
        case.runtime._recovery_origins = None  # This fixture has not accepted an acquisition.
        a, b = source(1), source(2)
        raw = payload((a, b))
        raw["retrieval_diagnostics"] = {"source_policy": "native-closure-v4", "initial_ids": [1]}
        consumer = SimpleNamespace(convo=Conversation(system="Sources", messages=[UserMessage(content="question")]))
        case.turn.retrieval_preparation = lambda message: external.native_tool_preparation(consumer, message)
        message = AssistantMessage(content="", stop_reason="tool_use", tool_call_id="native-lifetime",
            tool_call_name="vector_search", tool_call_input={"search_string": "rewritten query"})
        # Like the existing origin unit fixture, model only the native pending
        # identities and evidence views. PostgreSQL integration owns real capture.
        monkeypatch.setattr(external, "source_views", lambda results: (a, b))
        actions, invocations = [], []

        def invoke():
            invocations.append(True)
            action = _ACTION.get()
            identity = lambda s: (s.chunk_id, s.document_id, s.chunk_number, s.source_fingerprint)
            action.preparation.append(((identity(a), identity(b)), (identity(a),)))
            return raw

        def call(current):
            actions.append(_ACTION.get())
            result = run_tool_once(current, invoke)
            return ToolMessage(tool_name="vector_search", for_whom="assistant", content="sources", result_dict=result)

        execution = SimpleNamespace(check_active=lambda: None, run_tool=lambda message, function: raw)
        if replay:
            with bind_execution(execution):
                result = async_to_sync(call_tool_async)(SimpleNamespace(call_tool=call), message)
        else:
            result = async_to_sync(call_tool_async)(SimpleNamespace(call_tool=call), message)
        case.result, case.action, case.sources, case.invocations = result, actions[0], (a, b), invocations
        assert case.action is not None and not case.action.operation.live
        yield case


def test_external_record_resumes_exact_accepted_result_once(monkeypatch):
    with outer_case(monkeypatch) as case:
        raw = case.result.result_dict
        record = external.resume_external_preparation("question", case.budget, (raw, raw), lambda _: case.sources)
        assert record is case.action.origins and case.action.accepted
        assert len(record.occurrences) == 1 and record.occurrences[0].result is raw
        assert external.resume_external_preparation("question", case.budget, (raw,), lambda _: case.sources) is None
        order = take_preparation_order(fuse_ranked_tool_results([raw]), "question", case.runtime.authorization)
        assert order.initial_ids == (1,) and order.deferred_ids == (2,)


@pytest.mark.parametrize("mutation", ["copy", "payload", "question", "authorization", "record"])
def test_external_resume_rejects_changed_acceptance_binding(monkeypatch, mutation):
    with outer_case(monkeypatch) as case:
        raw, question = case.result.result_dict, "question"
        if mutation == "copy":
            raw = dict(raw)  # Equal public rows cannot adopt a private obligation.
        elif mutation == "payload":
            raw["result"][0]["text"] = "changed accepted text"
        elif mutation == "question":
            question = "different task"
        elif mutation == "authorization":
            case.runtime.authorization = object()
        else:
            case.runtime._recovery_origins = None
        with pytest.raises(SourceScopeUnavailable):
            external.resume_external_preparation(question, case.budget, (raw,), lambda _: case.sources)
        assert case.runtime._external_preparation_origins is None
        assert len(case.action.origins.occurrences) == 1


def test_replayed_result_without_native_invocation_has_no_preparation_priority(monkeypatch):
    with outer_case(monkeypatch, replay=True) as case:
        raw = case.result.result_dict
        assert case.invocations == [] and not case.action.accepted
        assert case.action.native == case.action.preparation == case.action.origins.occurrences == []
        assert external.resume_external_preparation("question", case.budget, (raw,), lambda _: case.sources) is None
        ordinary = start_recovery_origins("question", case.budget, (raw,), lambda _: case.sources)
        assert ordinary.occurrences[0].preparation is None
        assert len(ordinary.occurrences[0].ordinary) == 2
        assert take_preparation_order(fuse_ranked_tool_results([raw]), "question", case.runtime.authorization) is None
