"""Bind direct synthesis to selected evidence without rewriting stored history."""

from __future__ import annotations

from typing import Any

from apps.chat.services.rag_evidence import EvidencePacket
from lib.llm.providers.complete_turn import DIRECT_SYNTHESIS_GROUNDING
from lib.llm.providers.image_context import serialize_tool_result_for_llm
from lib.llm.providers.rag_citations import _chunk_citation_from_row
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage

_ROW_FIELDS = frozenset(
    {
        "rank",
        "r",
        "doc_id",
        "d",
        "chunk_id",
        "i",
        "chunk",
        "c",
        "title",
        "n",
        "text",
        "x",
        "citation",
        "ref",
        "type",
        "ty",
        "image_url",
        "u",
    }
)
_OMITTED_EVIDENCE = "Earlier tool evidence omitted for this synthesis request."


def _selected_payload(packet: EvidencePacket) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if packet.retrieval_status != "no_results":
        for chunk in packet.chunks:
            row = {
                key: value
                for key, value in chunk.items()
                if key in _ROW_FIELDS and isinstance(value, (str, int, float, bool))
            }
            citation = _chunk_citation_from_row(row)
            if not citation:
                raise ValueError("Selected evidence row has no valid chunk citation")
            # Both field formats are accepted by the provider citation collector.
            citation_field = "ref" if "d" in row and "doc_id" not in row else "citation"
            row.pop("ref" if citation_field == "citation" else "citation", None)
            row[citation_field] = citation
            for key in ("image_url", "u"):
                if key in row and row[key] not in packet.image_urls:
                    row.pop(key)
            rows.append(row)

    titles = sorted(
        {
            str(row.get("title") or row.get("n"))
            for row in rows
            if row.get("title") or row.get("n")
        }
    )
    payload: dict[str, Any] = {
        "result": rows,
        "retrieval_status": packet.retrieval_status if rows else "no_results",
        "retrieved_count": len(rows),
        "retrieved_documents": titles,
    }
    if packet.diagnostic_message:
        payload["retrieval_message"] = packet.diagnostic_message
    return payload


def prepare_evidence_handoff(
    conversation: Conversation,
    packet: EvidencePacket,
) -> tuple[Conversation, Conversation]:
    """Return (persisted history, isolated request) with the same message count.

    The current tool result is rebuilt from the packet in both conversations.
    Earlier tool payloads remain in persisted history, but their request copies
    carry no evidence, citation metadata, attached files, or inline images.
    """
    if not conversation.messages:
        raise ValueError("Evidence synthesis requires a current assistant tool result")
    current = conversation[-1]
    if not isinstance(current, ToolMessage) or current.for_whom != "assistant":
        raise ValueError("Evidence synthesis requires a current assistant tool result")

    payload = _selected_payload(packet)
    selected_tool = current.model_copy(
        update={
            "content": serialize_tool_result_for_llm(payload),
            "result_dict": payload,
            "files": None,
        }
    )
    persisted = Conversation(
        system=conversation.system,
        messages=[*conversation.messages[:-1], selected_tool],
    )
    # Providers may mutate message content while applying their context budget.
    request = persisted.model_copy(deep=True)
    request.system = f"{persisted.system}\n\n{DIRECT_SYNTHESIS_GROUNDING}"
    for message in request.messages[:-1]:
        if isinstance(message, ToolMessage):
            message.content = _OMITTED_EVIDENCE
            message.result_dict = {}
            message.files = None
    return persisted, request


__all__ = ["prepare_evidence_handoff"]
