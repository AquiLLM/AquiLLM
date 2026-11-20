"""Real normal-tool entry and source preparation; only inference is deterministic."""

from contextlib import contextmanager
import json
from time import monotonic
from types import SimpleNamespace

from apps.documents.models import TextChunk
from apps.documents.tests import _native_graph_search_closure_support as support
from apps.documents.tests.test_scoped_recovery_final_preparation import _all_short_sources, _short_then_long_sources
from apps.knowledge_graph.tests import graph_dataset_fixture_v2 as dataset
from apps.knowledge_graph.tests.test_closed_graph_limits_v2 import returned_envelope
from apps.knowledge_graph.tests.test_ready_scope import _DOC_A


@contextmanager
def external_case(monkeypatch, *, cheap=False, retry=False):
    from aquillm.llm import llm_tool
    from apps.chat.refs import CollectionsRef
    from apps.chat.services import rag_selection_coordinator
    from apps.chat.services.rag_action_tools import admit_tool_action
    from apps.chat.services.rag_recovery_origins import _ACTION
    from apps.documents.services import chunk_rerank_selection_provider
    from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
    from lib.llm import evidence_guard
    from lib.llm.providers import openai_request
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.types.messages import AssistantMessage, UserMessage
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.tools import ToolChoice
    from lib.retrieval.evidence import SourceEvidence, fingerprint_source
    from lib.tools.search.vector_search import pack_chunk_search_results

    for key, value in {
        "RAG_SCOPED_BASELINE_RECOVERY_ENABLED": "0", "RAG_EVIDENCE_TEXT_MODE": "source",
        "RAG_EVIDENCE_SELECTION_MODE": "adaptive", "RAG_RERANK_TEXT_MODE": "windowed",
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0", "RAG_SELECTION_SUPPORT_HINTS_ENABLED": "0",
        "RAG_QUERY_REWRITE_ENABLED": "0", "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0",
        "RAG_VOCABULARY_QUERY_ENABLED": "0", "RAG_SYNTHESIS_MAX_TOKENS": "4096",
        "OPENAI_CONTEXT_LIMIT": "131072", "PROMPT_BUDGET_CONTEXT_LIMIT": "131072",
        "OPENAI_STREAM_RESPONSES": "0",
    }.items():
        monkeypatch.setenv(key, value)
    if not cheap:
        monkeypatch.setattr(dataset, "TEXTS", ("Atlas is a model", "Atlas is a model",
            "Atlas uses the Cedar dataset. " + "procedure analysis " * 450))
    monkeypatch.setattr(support, "pressure_rows", _all_short_sources if cheap else _short_then_long_sources)
    epoch = monotonic()
    with support.native_case(monkeypatch, fail=False, controlled_graph_clock=True,
            graph_epoch=epoch, requested=frozenset({_DOC_A}), timeout_ms=6500,
            initial_capture=False) as case:
        case.native_initial, case.native_final, case.native_actions = (), (), []
        case.prepared, case.dispatched, case.validations, case.failures, case.score_calls = [], [], [], [], []
        case.winner = None
        case.question = "Describe the Atlas model"

        def expire(kwargs, envelope):
            case.now[0] = kwargs["deadline"] - epoch + .001

        returned_envelope(monkeypatch, empty=True, reasons=("deadline",), after=expire)

        @llm_tool(for_whom="assistant", required=["search_string", "top_k"],
                  param_descs={"search_string": "Retrieval query", "top_k": "Requested source count"})
        def vector_search(search_string: str, top_k: int):
            """Run native acquisition inside the actual two tool-worker hops."""
            assert search_string == "Atlas" and top_k == 15 and admit_tool_action("vector", search_string)
            case.native_actions.append(_ACTION.get())
            incoming = case.authorize.authorization
            case.snapshot = support.capture(case.active, native=True, authorization=incoming)
            case.authorize = support._bind_search_authorizer(
                support.chunk_search.authorized_search_rows, incoming, case.graph_on)
            case.native_initial = tuple(case.snapshot.baseline_candidates)
            closure = support.run_current_or_native(case)
            case.native_final = tuple(closure.final_rows[2])
            initial = {row.pk for row in case.native_initial}
            initial_text = {row.content for row in case.native_initial}
            case.winner = next((row for row in case.native_final
                if row.pk not in initial and row.content not in initial_text), None) if cheap else None
            sources = tuple(SourceEvidence(row.pk, str(row.doc_id), row.chunk_number,
                fingerprint_source(row.content), row.content) for row in case.native_final)
            return pack_chunk_search_results(case.native_final,
                titles_by_doc_id={doc.id: doc.title for doc in case.f.documents.values()},
                docs_by_doc_id={doc.id: doc for doc in case.f.documents.values()},
                truncate=lambda text: text, image_modality=TextChunk.Modality.IMAGE,
                compact_items=False, source_evidence=sources)

        class Provider:
            scorer_fingerprint, scoring_kind = "normal-native-work-control", "pointwise"

            def score_pair(self, pair, timeout_seconds):
                case.score_calls.append(pair)
                initial_text = {row.content for row in case.native_initial}
                return (2.0 if case.winner is not None and pair[1] == case.winner.content
                        else 1.0 if pair[1] in initial_text else 0.0), pair

        def scorer_factory(*, deadline, clock, turn_budget):
            assert turn_budget is case.source.budget
            return WindowSelectionScorer(Provider(), budget=turn_budget, deadline=deadline,
                clock=clock, pair_counter=lambda query, text: len(query) + len(text) + 8)

        monkeypatch.setattr(chunk_rerank_selection_provider, "current_selection_scorer", scorer_factory)
        prepare = rag_selection_coordinator.prepare_selection_candidates

        def observed_prepare(**kwargs):
            try:
                result = prepare(**kwargs)
                case.prepared.append(result)
                return result
            except Exception as exc:
                case.failures.append(("prepare", type(exc).__name__, str(exc)))
                raise

        monkeypatch.setattr(rag_selection_coordinator, "prepare_selection_candidates", observed_prepare)
        validate = evidence_guard.validate_request

        def observed_validation(stage):
            def checked(payload, *, output_reserve):
                before = case.source.budget.text_used["tokenized"]
                result = validate(payload, output_reserve=output_reserve)
                case.validations.append((stage, before, case.source.budget.text_used["tokenized"]))
                return result
            return checked

        monkeypatch.setattr(openai_request, "validate_request", observed_validation("prepare_request"))
        monkeypatch.setattr(evidence_guard, "validate_request", observed_validation("provider_dispatch"))

        async def create(**kwargs):
            assert evidence_guard.contains_selected_evidence(kwargs["messages"])
            packet = case.source.observation["delivered_packet"]
            delivered = json.loads(evidence_guard.current_protection().payload)["result"]
            assert [(r["chunk_id"], r["text"]) for r in delivered] == [(r["chunk_id"], r["text"]) for r in packet.chunks]
            row = max(delivered, key=lambda item: len(item["text"]))
            answer = f'The source states: "{row["text"]}" {row["citation"]}'
            case.dispatched.append((answer, packet))
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                message=SimpleNamespace(content=answer, tool_calls=None))], usage=None)

        client = SimpleNamespace(base_url="http://vllm:8000/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        case.llm = OpenAIInterface(client, "local-test-model")
        messages = [UserMessage(content=case.question)]
        if retry:
            messages.extend([AssistantMessage(content="Earlier answer", stop_reason="end_turn"), UserMessage(content="try again")])
        case.conversation = Conversation(system="Answer using cited source evidence.", messages=messages)
        case.consumer = SimpleNamespace(
            user=case.source.authorization.reauthorization_capability._principal,
            col_ref=CollectionsRef(list(case.source.authorization.selected_collection_ids)), convo=case.conversation)
        case.message = AssistantMessage(content="", stop_reason="tool_use", tools=[vector_search],
            tool_choice=ToolChoice(type="auto"), tool_call_id="native-first", tool_call_name="vector_search",
            tool_call_input={"search_string": "Atlas", "top_k": 15})
        yield case
