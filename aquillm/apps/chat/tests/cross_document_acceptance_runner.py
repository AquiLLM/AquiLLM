"""Run synthetic cross-document cases through the real synthesis path."""
from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from .cross_document_acceptance_support import AcceptanceCase


async def run_acceptance_case(
    case: AcceptanceCase,
    llm_if: Any,
    *,
    omitted_doc_id: str | None = None,
    retrieval_limit: int = 2,
    prior_tool_rows: list[dict] | None = None,
    preserve_citation_settings: bool = False,
    stream_output: bool = False,
) -> dict[str, Any]:
    """Run one case against an actual provider instance or a request probe.

    Real-provider use intentionally costs a small number of synthesis calls.
    No database, uploaded document, network retrieval, or private text is needed.
    Reports record only synthetic evidence and responses, never API arguments.
    """
    from apps.chat.refs import CollectionsRef
    from apps.chat.services import rag_pipeline
    from apps.chat.services.rag_query import build_retrieval_queries
    from lib.llm.providers.image_context import serialize_tool_result_for_llm
    from lib.llm.providers.rag_citations import extract_citations
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
    from .cross_document_acceptance_support import evaluate_answer, retrieval_payload

    history = []
    if prior_tool_rows:
        prior_payload = {"result": prior_tool_rows, "retrieval_status": "results_found"}
        history = [
            UserMessage(content="Search the previously selected documents."),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="synthetic-prior-call",
                tool_call_name="vector_search",
                tool_call_input={"search_string": "prior"},
            ),
            ToolMessage(
                tool_name="vector_search",
                for_whom="assistant",
                arguments={"search_string": "prior"},
                result_dict=prior_payload,
                content=serialize_tool_result_for_llm(prior_payload),
            ),
            AssistantMessage(
                content="Previous search finished.", stop_reason="end_turn"
            ),
        ]
    convo = Conversation(
        system="Answer from the supplied synthetic papers, citing supporting chunks.",
        messages=[*history, UserMessage(content=case.question)],
    )
    queries = build_retrieval_queries(convo, case.question, max_queries=3)
    consumer = SimpleNamespace(user=object(), col_ref=CollectionsRef([1]), convo=convo)
    captured_requests: list[dict] = []
    stream_payloads: list[dict] = []
    original_get_message = llm_if.get_message

    async def capture_stream(payload: dict):
        stream_payloads.append(copy.deepcopy(payload))

    async def capture_request(*args, **kwargs):
        tool_messages = [
            msg
            for msg in kwargs.get("messages_pydantic", [])
            if isinstance(msg, ToolMessage) and msg.for_whom == "assistant"
        ]
        last_tool = tool_messages[-1] if tool_messages else None
        captured_requests.append(
            {
                "system_citations": sorted(
                    set(extract_citations(kwargs.get("system", "")))
                ),
                "tool_content": last_tool.content if last_tool else "",
                "tool_payload": copy.deepcopy(last_tool.result_dict)
                if last_tool
                else {},
            }
        )
        return await original_get_message(*args, **kwargs)

    def fake_retrieval(_consumer, query, top_k):
        return retrieval_payload(
            case,
            query_index=queries.index(query),
            top_k=top_k,
            omitted_doc_id=omitted_doc_id,
        )

    environment = {
        "RAG_DIRECT_ENABLED": "1",
        "RAG_DIRECT_MAX_QUERIES": "3",
        "RAG_DIRECT_TOP_K": str(retrieval_limit),
        "RAG_MAX_SNIPPETS_PER_DOC": "1",
        "RAG_EVIDENCE_TOKEN_BUDGET": "1000",
        "RAG_QUERY_REWRITE_ENABLED": "0",
    }
    if not preserve_citation_settings:
        environment.update(
            {
                "RAG_ENFORCE_CHUNK_CITATIONS": "1",
                "RAG_APPEND_CITATION_SOURCES": "0",
            }
        )
    with (
        patch.dict(os.environ, environment),
        patch.object(rag_pipeline, "_run_vector_search", side_effect=fake_retrieval),
        patch.object(llm_if, "get_message", side_effect=capture_request),
    ):
        outcome = await rag_pipeline.run_direct_rag_turn(
            consumer,
            llm_if,
            convo,
            stream_func=capture_stream if stream_output else None,
        )

    first_request = captured_requests[0] if captured_requests else {}
    payload = first_request.get("tool_payload") or {}
    visible_rows = payload.get("result") or []
    content = first_request.get("tool_content") or ""
    # Check actual serialized text separately from provider-neutral object metadata.
    try:
        serialized_rows = json.loads(content).get("result", [])
    except (ValueError, AttributeError):
        serialized_rows = []
    evidence_citations = {
        row.get("citation") or row.get("ref")
        for row in visible_rows
        if row.get("citation") or row.get("ref")
    }
    expected_citations = {
        token
        for token in case.required_citations
        if not omitted_doc_id or not token.startswith(f"[doc:{omitted_doc_id} ")
    }
    required_texts = [
        paper["text"] for paper in case.papers if paper["doc_id"] != omitted_doc_id
    ]
    retained_texts = [row.get("text") or row.get("x") or "" for row in visible_rows]
    answer = consumer.convo[-1].content if outcome == "handled" else ""
    final_stream_payloads = [
        payload
        for payload in stream_payloads
        if payload.get("done") is True and payload.get("role") == "assistant"
    ]
    return {
        "case_id": case.case_id,
        "omitted_doc_id": omitted_doc_id,
        "outcome": outcome,
        "queries": queries,
        "provider_call_count": len(captured_requests),
        "requests": captured_requests,
        "answer": answer,
        "stream_event_count": len(stream_payloads),
        "final_stream_received": bool(final_stream_payloads) if stream_output else None,
        "final_stream_content_matches_answer": (
            bool(final_stream_payloads)
            and final_stream_payloads[-1].get("content") == answer
            if stream_output
            else None
        ),
        "evidence_citations": sorted(evidence_citations),
        "evidence_coverage_passed": (
            expected_citations <= evidence_citations
            and all(text in retained_texts for text in required_texts)
        ),
        "serialized_rows_match": serialized_rows == visible_rows,
        "exact_allowlist_passed": bool(captured_requests)
        and all(
            set(request["system_citations"]) == evidence_citations
            for request in captured_requests
        ),
        "answer_checks": evaluate_answer(
            case,
            answer,
            allowed_citations=evidence_citations,
            omitted_doc_id=omitted_doc_id,
        ),
    }
