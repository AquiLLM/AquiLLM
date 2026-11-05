"""Normal turn wiring preserves paid answer work, including partial notices.

The external-initial case admits 20 optional sources after 25 ordinary sources.
The direct case preserves the real 30-source first-action ceiling; it is a
normal-entry/allowance control, not the separate full 45-source stress proof.
"""

from functools import partial
import json
from time import monotonic
from types import SimpleNamespace

from asgiref.sync import async_to_sync
import pytest

from apps.documents.models import TextChunk
from apps.documents.tests import _native_graph_search_closure_support as support
from apps.documents.tests.test_scoped_recovery_final_preparation import _short_then_long_sources
from apps.knowledge_graph.tests import graph_dataset_fixture_v2 as dataset
from apps.knowledge_graph.tests.test_closed_graph_limits_v2 import returned_envelope
from apps.knowledge_graph.tests.test_ready_scope import _DOC_A

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("initial_acquired", [True, False],
    ids=["external25-then-native-followup", "direct-first-action30"])
def test_normal_preservation_turn_reuses_exact_paid_answer_request(monkeypatch, initial_acquired):
    from apps.chat.refs import CollectionsRef
    from apps.chat.services import rag_acquisition, rag_preservation_turn
    from apps.chat.services.rag_action_tools import admit_tool_action
    from apps.chat.services.rag_config import synthesis_max_tokens
    from apps.chat.services.rag_coverage import AcquisitionAction, CoverageAssessment
    from apps.chat.services.rag_selection_coordinator import prepare_selection_turn
    from apps.documents.services import chunk_rerank_selection_provider
    from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
    from lib.llm import evidence_guard
    from lib.llm.providers import openai_request
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.providers.request_observability import current_stage
    from lib.llm.turn_context import TurnContext, bind_turn
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import ToolMessage, UserMessage
    from lib.retrieval.evidence import SourceEvidence, fingerprint_source
    from lib.retrieval.operation import _OPERATION
    from lib.tools.search.vector_search import pack_chunk_search_results

    for key, value in {
        "RAG_SCOPED_BASELINE_RECOVERY_ENABLED": "1", "RAG_EVIDENCE_TEXT_MODE": "source",
        "RAG_EVIDENCE_SELECTION_MODE": "adaptive", "RAG_RERANK_TEXT_MODE": "windowed",
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": "1", "RAG_SELECTION_SUPPORT_HINTS_ENABLED": "0",
        "RAG_QUERY_REWRITE_ENABLED": "0", "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0",
        "RAG_VOCABULARY_QUERY_ENABLED": "0", "RAG_SYNTHESIS_MAX_TOKENS": "4096",
        "OPENAI_CONTEXT_LIMIT": "131072", "PROMPT_BUDGET_CONTEXT_LIMIT": "131072",
        "OPENAI_STREAM_RESPONSES": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(dataset, "TEXTS", (
        "Atlas is a model", "Atlas is a model",
        "Atlas uses the Cedar dataset. " + "procedure analysis " * 450,
    ))
    monkeypatch.setattr(support, "pressure_rows", _short_then_long_sources)
    epoch = monotonic()
    with support.native_case(monkeypatch, fail=False, controlled_graph_clock=True,
            graph_epoch=epoch, requested=frozenset({_DOC_A}), timeout_ms=6500,
            initial_capture=initial_acquired) as case:
        budget = case.source.budget
        assert budget.sources_used == (25 if initial_acquired else 0)
        acquisitions, assessments, prepared, validations, dispatched = [], [], [], [], []
        failures = []
        notice = "The requested measurement is not fully covered by the acquired evidence."

        def expire_child(kwargs, envelope):
            case.now[0] = kwargs["deadline"] - epoch + .001

        returned_envelope(monkeypatch, empty=True, reasons=("deadline",), after=expire_child)

        def pack(rows):
            sources = tuple(SourceEvidence(row.pk, str(row.doc_id), row.chunk_number,
                fingerprint_source(row.content), row.content) for row in rows)
            return pack_chunk_search_results(rows,
                titles_by_doc_id={doc.id: doc.title for doc in case.f.documents.values()},
                docs_by_doc_id={doc.id: doc for doc in case.f.documents.values()},
                truncate=lambda text: text, image_modality=TextChunk.Modality.IMAGE,
                compact_items=False, source_evidence=sources)

        def vector_runner(consumer, query, top_k):
            assert query == "Atlas"
            assert admit_tool_action("vector", query)
            initial_limit = _OPERATION.get().source_limit
            before = budget.sources_used
            try:
                incoming = case.authorize.authorization
                case.snapshot = support.capture(case.active, native=True,
                    authorization=incoming)
                # Normal search binds its authorizer after the current query
                # capture; the direct case intentionally has no earlier one.
                case.authorize = support._bind_search_authorizer(
                    support.chunk_search.authorized_search_rows, incoming, case.graph_on)
                closure = support.run_current_or_native(case)
                assert closure is not None, "current native capture declined before graph execution"
                rows = closure.final_rows[2]
                acquisitions.append((before, budget.sources_used, initial_limit, len(rows)))
                assert closure.graph_diagnostics["source_admission"]["fallback_top_up_count"] > 0
                return pack(rows)
            except Exception as exc:
                failures.append(("acquisition", type(exc).__name__, str(exc)))
                raise

        async def assess(question, views, anchors, **kwargs):
            assessments.append(tuple(source.chunk_id for source in views))
            action = AcquisitionAction("vector", "Atlas", aspect="measurement") if (
                initial_acquired and len(assessments) == 1) else None
            return CoverageAssessment(("measurement",), (), ("measurement",), action, "assessed")

        # Only the coverage inference is fixed; actual acquire_evidence still
        # owns action/source/planner limits and accepted native-origin records.
        monkeypatch.setattr(rag_preservation_turn, "acquire_evidence",
            partial(rag_acquisition.acquire_evidence, assess=assess))

        class ScorerProvider:
            scorer_fingerprint = "normal-turn-complete-source-control"
            scoring_kind = "pointwise"

            def score_pair(self, pair, timeout_seconds):
                return (1.0 if len(pair[1]) < 100 else 0.0), pair

        def scorer_factory(*, deadline, clock, turn_budget):
            assert turn_budget is budget
            return WindowSelectionScorer(ScorerProvider(), budget=budget, deadline=deadline,
                clock=clock, pair_counter=lambda query, text: len(query) + len(text) + 8)

        monkeypatch.setattr(chunk_rerank_selection_provider, "current_selection_scorer", scorer_factory)

        def observed_prepare(*args, **kwargs):
            try:
                result = prepare_selection_turn(*args, **kwargs)
                prepared.append(result)
                return result
            except Exception as exc:
                failures.append(("preparation", type(exc).__name__, str(exc)))
                raise

        original_validate = evidence_guard.validate_request

        def observed_validation(stage):
            def validate(payload, *, output_reserve):
                before = budget.text_used["tokenized"]
                try:
                    return original_validate(payload, output_reserve=output_reserve)
                except Exception as exc:
                    failures.append((stage, type(exc).__name__, str(exc)))
                    raise
                finally:
                    validations.append((stage, current_stage(), before,
                        budget.text_used["tokenized"], output_reserve))
            return validate

        monkeypatch.setattr(openai_request, "validate_request", observed_validation("prepare_request"))
        monkeypatch.setattr(evidence_guard, "validate_request", observed_validation("provider_dispatch"))

        async def create(**kwargs):
            payload = json.loads(evidence_guard.current_protection().payload)
            packet = case.source.observation["delivered_packet"]
            row = max(payload["result"], key=lambda item: len(item["text"]))
            answer = f'The source states: "{row["text"]}" {row["citation"]}'
            dispatched.append(dict(answer=answer, stage=current_stage(), payload=payload,
                packet=packet, max_tokens=kwargs["max_tokens"],
                contains_evidence=evidence_guard.contains_selected_evidence(kwargs["messages"])))
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                message=SimpleNamespace(content=answer, tool_calls=None))], usage=None)

        client = SimpleNamespace(base_url="http://vllm:8000/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        llm = OpenAIInterface(client, "local-test-model")
        conversation = Conversation(system="Answer using the cited source evidence.",
            messages=[UserMessage(content="Atlas")])
        initial_results = ()
        if initial_acquired:
            initial_results = (pack(case.snapshot.baseline_candidates),)
            assert budget.reserve_action("external-authorized-evidence")
            conversation += [ToolMessage(tool_name="vector_search", for_whom="assistant",
                content=json.dumps(initial_results[0]["result"]), result_dict=initial_results[0],
                arguments={"search_string": "Atlas"})]
        consumer = SimpleNamespace(
            user=case.source.authorization.reauthorization_capability._principal,
            col_ref=CollectionsRef(list(case.source.authorization.selected_collection_ids)), convo=conversation)
        with bind_turn(TurnContext(budget)):
            result = async_to_sync(rag_preservation_turn.finish_preservation)(consumer, llm,
                conversation, question="Atlas", initial_results=initial_results,
                initial_acquired=initial_acquired, vector_runner=vector_runner, iterative=True,
                notice=notice, prepare_fn=observed_prepare)

        acquired = case.source.observation.get("acquisition")
        diagnostic = dict(failures=failures, acquisitions=acquisitions,
            assessment_count=len(assessments), prepare_count=len(prepared), validations=validations,
            stop_reason=getattr(acquired, "stop_reason", None), result=result[-1].content,
            text_used=budget.text_used, sources=budget.sources_used, now=case.now[0])
        assert len(dispatched) == 1, diagnostic
        dispatch = dispatched[0]
        assert dispatch["answer"] in result[-1].content, diagnostic
        assert dispatch["stage"] == "direct_synthesis" and dispatch["contains_evidence"]
        assert dispatch["max_tokens"] == synthesis_max_tokens() == 4096
        payload, packet = dispatch["payload"], dispatch["packet"]
        assert payload["retrieval_status"] == "partial" and payload["retrieval_message"] == notice
        assert [(item["chunk_id"], item["text"], item["citation"]) for item in payload["result"]] == [
            (item["chunk_id"], item["text"], item["citation"]) for item in packet.chunks]
        assert len(prepared) == 1 and prepared[0].prepared.prepaid_selection is prepared[0].selection
        assert [row[0] for row in validations] == ["prepare_request", "provider_dispatch"]
        assert all(stage == "direct_synthesis" and before == after and reserve == 4096
            for _, stage, before, after, reserve in validations)
        assert len(acquisitions) == 1
        before, after, source_limit, count = acquisitions[0]
        if initial_acquired:
            assert (before, after, source_limit, count) == (25, 45, None, 45)
            assert budget.actions_used == budget.planner_calls == 2
            assert len(result.messages) == len(conversation.messages) + 1
        else:
            assert before == 0 and source_limit == 30 and after <= 30 and count <= 30
            assert budget.actions_used == budget.planner_calls == 1
            assert [message.role for message in result.messages] == ["user", "assistant", "tool", "assistant"]
        assert case.source.observation["acquisition"].assessment.unresolved_aspects == ("measurement",)
        assert budget.limits.actions == 3 and budget.limits.planner_calls == 2
        assert budget.sources_used <= budget.limits.unique_sources == 45
        assert budget.pairs_used["final"] <= budget.limits.final_pairs == 45
        assert budget.text_used["tokenized"] <= budget.limits.tokenized_codepoints == 1_000_000
        assert budget.text_used["materialized"] <= budget.limits.materialized_codepoints == 250_000
        assert budget.limits.final_scoring_ms == 3000 and budget.limits.retrieval_ms == 15000
