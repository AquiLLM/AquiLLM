"""Public synthetic fixtures and a real-provider-compatible acceptance harness.

Only retrieval is substituted by ``run_acceptance_case``. The real intent,
query, fusion, evidence, synthesis, provider request, and citation paths run.
Call cases sequentially: environment and retrieval patches are process-global.
The lexical answer checks are conservative review aids, not semantic judges.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


@dataclass(frozen=True)
class ExpectedClaim:
    name: str
    patterns: tuple[str, ...]
    citations: tuple[str, ...]


@dataclass(frozen=True)
class AcceptanceCase:
    case_id: str
    question: str
    papers: tuple[dict[str, Any], ...]
    claims: tuple[ExpectedClaim, ...]
    synthesis: ExpectedClaim
    reference_answer: str

    @property
    def required_citations(self) -> set[str]:
        return {paper["citation"] for paper in self.papers}


def _paper(doc_id: str, chunk_id: int, title: str, text: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "title": title,
        "text": text,
        "citation": f"[doc:{doc_id} chunk:{chunk_id}]",
    }


CAPACITY_A = "[doc:synthetic-capacity-a chunk:101]"
CAPACITY_B = "[doc:synthetic-capacity-b chunk:201]"
CONFLICT_A = "[doc:synthetic-conflict-a chunk:301]"
CONFLICT_B = "[doc:synthetic-conflict-b chunk:401]"

CASES = (
    AcceptanceCase(
        case_id="complementary-capacity",
        question=(
            "Search the selected documents: what is the Lumen unit's dry-mode "
            "capacity per cycle? What is its humid-mode capacity per cycle, and "
            "how much lower is it than dry-mode capacity? Cite each source; "
            "if a necessary paper is absent, state what cannot be determined."
        ),
        papers=(
            _paper(
                "synthetic-capacity-a",
                101,
                "Synthetic Paper A: Dry trial",
                (
                    "In the dry-mode trial, the Lumen unit treated "
                    "18 litres per cycle. "
                    "This paper did not measure humid-mode capacity."
                ),
            ),
            _paper(
                "synthetic-capacity-b",
                201,
                "Synthetic Paper B: Humid trial",
                (
                    "In the humid-mode trial, the same Lumen unit treated "
                    "7 litres per cycle. "
                    "This paper did not measure dry-mode capacity."
                ),
            ),
        ),
        claims=(
            ExpectedClaim(
                "dry-capacity",
                (r"\bdry", r"\b18\s*(?:litres?|liters?|l\b)"),
                (CAPACITY_A,),
            ),
            ExpectedClaim(
                "humid-capacity",
                (r"\bhumid", r"\b7\s*(?:litres?|liters?|l\b)"),
                (CAPACITY_B,),
            ),
        ),
        synthesis=ExpectedClaim(
            "cross-paper-difference",
            (r"\b11\b", r"lower|less|difference|reduc"),
            (CAPACITY_A, CAPACITY_B),
        ),
        reference_answer=(
            f"Dry-mode capacity is 18 litres per cycle {CAPACITY_A}.\n"
            f"Humid-mode capacity is 7 litres per cycle {CAPACITY_B}.\n"
            "Humid capacity is 11 litres per cycle lower (18 - 7) "
            f"{CAPACITY_A} {CAPACITY_B}."
        ),
    ),
    AcceptanceCase(
        case_id="conditioned-disagreement",
        question=(
            "Search the selected documents: how did activating the Vela coating "
            "change sensor response in the controlled trial? How did it change "
            "response in the field trial, and do the reported directions agree? "
            "Preserve the different conditions and cite both sources; if a paper "
            "is absent, state what cannot be determined."
        ),
        papers=(
            _paper(
                "synthetic-conflict-a",
                301,
                "Synthetic Paper A: Controlled coating",
                (
                    "In a controlled trial at 20 degrees Celsius, activating the Vela "
                    "coating increased sensor response by 12 percent versus the "
                    "inactive coating. No field trial was performed."
                ),
            ),
            _paper(
                "synthetic-conflict-b",
                401,
                "Synthetic Paper B: Field coating",
                (
                    "In a field trial at 35 degrees Celsius, activating the same Vela "
                    "coating decreased sensor response by 9 percent versus the "
                    "inactive coating. The trial did not establish why this "
                    "differs from controlled conditions."
                ),
            ),
        ),
        claims=(
            ExpectedClaim(
                "controlled-increase",
                (r"\bcontrolled\b", r"\b12\b", r"increas|improv"),
                (CONFLICT_A,),
            ),
            ExpectedClaim(
                "field-decrease",
                (r"\bfield\b", r"\b9\b", r"decreas|reduc"),
                (CONFLICT_B,),
            ),
        ),
        synthesis=ExpectedClaim(
            "preserved-disagreement",
            (r"disagree|oppos|conflict|contrast|differ|do not agree",),
            (CONFLICT_A, CONFLICT_B),
        ),
        reference_answer=(
            f"The controlled trial reports a 12 percent increase {CONFLICT_A}.\n"
            f"The field trial reports a 9 percent decrease {CONFLICT_B}.\n"
            f"The reported directions disagree under different conditions; "
            f"these trials do not establish the reason {CONFLICT_A} {CONFLICT_B}."
        ),
    ),
)


def retrieval_payload(
    case: AcceptanceCase,
    *,
    query_index: int = 0,
    top_k: int = 10,
    omitted_doc_id: str | None = None,
) -> dict[str, Any]:
    """A ranks repeatedly; B appears in the final aspect query's retrieved pool.

    Even at top_k=2, the union contains both necessary facts. Plain RRF at limit=2
    retains A's redundant row ahead of B; diversity must occur before that cut.
    """
    first, second = copy.deepcopy(case.papers)
    redundant = dict(first, chunk_id=first["chunk_id"] + 1)
    redundant["citation"] = f"[doc:{first['doc_id']} chunk:{redundant['chunk_id']}]"
    redundant["text"] = (
        "This synthetic methods appendix repeats the apparatus description. "
        "It provides no additional measured outcome."
    )
    rows = [second, first] if query_index >= 2 else [first, redundant]
    rows = [row for row in rows if row["doc_id"] != omitted_doc_id][:top_k]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "result": rows,
        "retrieval_status": "results_found" if rows else "no_results",
        "retrieved_count": len(rows),
        "retrieved_documents": sorted({row["title"] for row in rows}),
    }


def _claim_units(answer: str) -> list[str]:
    # A trailing source inventory must never satisfy a claim's citation check.
    body = re.split(r"(?im)^\s*(?:#{1,6}\s*)?sources\s*:", answer)[0]
    # Treat citations placed after a full stop as belonging to that sentence.
    body = re.sub(
        r"([.!?])\s*((?:\[doc:[^\]\s]+\s+chunk:\d+\][ \t]*)+)",
        r" \2\1",
        body,
    )
    return [
        unit.strip() for unit in re.split(r"\n+|(?<=[.!?])\s+", body) if unit.strip()
    ]


def evaluate_answer(
    case: AcceptanceCase,
    answer: str,
    *,
    allowed_citations: set[str],
    omitted_doc_id: str | None = None,
) -> dict[str, Any]:
    """Check explicit expected claims and their local citations, conservatively.

    A successful lexical result still requires human review for semantic errors,
    invented causal explanations, and citation entailment beyond these claims.
    """
    from lib.llm.providers.rag_citations import (
        extract_citations,
        find_invalid_citations,
    )

    units = _claim_units(answer)
    errors: list[str] = []
    present_claims: list[str] = []
    expected = (*case.claims, case.synthesis)
    for claim in expected:
        matching = [
            unit
            for unit in units
            if all(
                re.search(
                    pattern,
                    re.sub(r"\[doc:[^\]\s]+\s+chunk:\d+\]", "", unit).replace("*", ""),
                    flags=re.I,
                )
                for pattern in claim.patterns
            )
        ]
        unavailable = omitted_doc_id is not None and any(
            token.startswith(f"[doc:{omitted_doc_id} ") for token in claim.citations
        )
        if unavailable:
            if matching:
                errors.append(f"unsupported-claim:{claim.name}")
            continue
        if not matching:
            errors.append(f"missing-claim:{claim.name}")
        elif not any(
            set(claim.citations) <= set(extract_citations(unit)) for unit in matching
        ):
            errors.append(f"wrong-or-missing-citation:{claim.name}")
        else:
            present_claims.append(claim.name)
    invalid = find_invalid_citations(answer, allowed_citations)
    if invalid:
        errors.append("citations-outside-evidence")
    if omitted_doc_id and not re.search(
        r"cannot (?:be )?(?:determine|establish|compare|conclude|calculate)|"
        r"can't determine|insufficient|missing|not (?:available|provided|reported)|"
        r"no (?:evidence|data)|not enough",
        answer,
        flags=re.I,
    ):
        errors.append("missing-insufficiency-statement")
    return {
        "passed": not errors,
        "errors": errors,
        "matched_claims": present_claims,
        "invalid_citations": invalid,
        "requires_semantic_review": True,
    }


async def run_acceptance_case(
    case: AcceptanceCase,
    llm_if: Any,
    *,
    omitted_doc_id: str | None = None,
    retrieval_limit: int = 2,
    prior_tool_rows: list[dict] | None = None,
    preserve_citation_settings: bool = False,
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
    original_get_message = llm_if.get_message

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
        outcome = await rag_pipeline.run_direct_rag_turn(consumer, llm_if, convo)

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
    return {
        "case_id": case.case_id,
        "omitted_doc_id": omitted_doc_id,
        "outcome": outcome,
        "queries": queries,
        "provider_call_count": len(captured_requests),
        "requests": captured_requests,
        "answer": answer,
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
