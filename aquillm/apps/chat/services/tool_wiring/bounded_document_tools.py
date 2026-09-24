"""Whole-document and adjacent source tools retain exact bounded chunks."""

from apps.chat.services.rag_config import evidence_token_budget
from apps.chat.services.rag_context_budget import synthesis_evidence_budget
from apps.chat.services.tool_wiring.document_tool_support import (
    format_whole_document_citations,
)
from apps.chat.services.tool_wiring.source_documents import (
    document_source_evidence,
    required_source_runtime,
)
from apps.documents.models import TextChunk
from apps.documents.services.source_loading import (
    bounded_source_database,
    source_query_rows,
)
from lib.llm.evidence_guard import estimate_request_tokens
from lib.retrieval.evidence import source_provenance
from lib.tools.documents.whole_document import (
    image_document_instruction,
    image_document_tool_payload,
)

from .source_figure_payloads import bounded_figure_payloads, revalidate_figure_payloads


def limited_document_result():
    return {
        "result": "",
        "retrieval_status": "context_limited",
        "retrieval_message": (
            "The requested document context exceeds this turn's limits; "
            "complete document coverage is unavailable."
        ),
    }


def bounded_document_result(doc, query, *, chat_ref=None, adjacent=False):
    runtime = required_source_runtime()
    with bounded_source_database(runtime, query.db):
        expected = query.count()
        chunks = source_query_rows(query)
    if not chunks or len(chunks) != expected:
        return limited_document_result()
    text, citations = format_whole_document_citations(
        doc.id, chunks, preserve_text=True
    )
    rows = [
        dict(citation, text=chunk.content) for citation, chunk in zip(citations, chunks)
    ]
    ceiling = (
        synthesis_evidence_budget(chat_ref.chat.convo, chat_ref.chat.llm_if, rows)
        if chat_ref is not None
        else evidence_token_budget()
    )
    cost = estimate_request_tokens(
        {"result": text, "citation_chunks": citations}, turn_budget=runtime.budget
    )
    if cost > ceiling:
        return limited_document_result()
    if adjacent:
        text = (
            f"chunk_numbers:{chunks[0].chunk_number} -> "
            f"{chunks[-1].chunk_number} \n\n {text}"
        )
    return {
        "result": text,
        "citation_chunks": citations,
        "_source_provenance": [
            source_provenance(source) for source in document_source_evidence(chunks)
        ],
    }


def bounded_whole_document(doc, chat_ref, *, user):
    from .source_tool_revalidation import revalidate_source_tool_result

    return revalidate_source_tool_result(
        _bounded_whole_document(doc, chat_ref, user=user),
        user=user,
    )


def _bounded_whole_document(doc, chat_ref, *, user):
    query = (
        TextChunk.objects.using(required_source_runtime().authorization.database_alias)
        .filter(doc_id=doc.id)
        .order_by("chunk_number", "pk")
    )
    result = bounded_document_result(doc, query, chat_ref=chat_ref)
    if result.get("retrieval_status") == "context_limited":
        return result
    text = result["result"]
    ceiling = synthesis_evidence_budget(chat_ref.chat.convo, chat_ref.chat.llm_if, [])
    runtime = required_source_runtime()
    if getattr(doc, "image_file", None):
        url = f"/aquillm/document_image/{doc.id}/"
        result["result"] = image_document_tool_payload(
            full_text=text,
            title=doc.title,
            display_url=url,
        )
        result["_image_instruction"] = image_document_instruction(
            title=doc.title,
            display_url=url,
        )
        if estimate_request_tokens(result, turn_budget=runtime.budget) > ceiling:
            result["result"] = text
            result.pop("_image_instruction", None)
            _mark_figure_omission(result)
        return result
    figures, provenance, omitted = bounded_figure_payloads(doc, user=user)
    fresh, identities = revalidate_figure_payloads(figures, provenance, user=user)
    omitted |= len(fresh) != len(figures)
    accepted, retained = [], []
    for figure, record in zip(fresh, identities):
        proposed = {
            **result,
            "result": {
                "type": "document_with_figures",
                "text": text,
                "figures": [*accepted, figure],
            },
        }
        if estimate_request_tokens(proposed, turn_budget=runtime.budget) > ceiling:
            omitted = True
            continue
        accepted.append(figure)
        retained.append(record)
    if accepted:
        result["result"] = {
            "type": "document_with_figures",
            "text": text,
            "figures": accepted,
        }
        result["_figure_provenance"] = retained
        result["_image_instruction"] = (
            "Related figures include image_url fields. Include relevant figures "
            "in markdown with ![description](image_url)."
        )
    if omitted:
        _mark_figure_omission(result)
    return result


def _mark_figure_omission(result):
    result["retrieval_status"] = "partial"
    result["retrieval_message"] = (
        "Main-document text is available; some figures were omitted because "
        "their content, authorization, or image context exceeded this turn's limits."
    )
