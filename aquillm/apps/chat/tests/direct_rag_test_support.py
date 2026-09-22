"""Tests for the direct RAG pipeline orchestrator (Tasks 4 and 5)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from apps.chat.refs import CollectionsRef
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage


def _user_convo(text: str) -> Conversation:
    return Conversation(system="sys", messages=[UserMessage(content=text)])


def _consumer(convo: Conversation, collections: list) -> SimpleNamespace:
    return SimpleNamespace(
        user=object(),
        col_ref=CollectionsRef(list(collections)),
        convo=convo,
        _send_stream_payload=AsyncMock(),
    )


def _results_payload() -> dict:
    return {
        "result": [
            {
                "rank": 1,
                "chunk_id": 1,
                "doc_id": "doc-a",
                "title": "Paper A",
                "text": "Calibration uses flat fields and dark frames.",
                "citation": "[doc:doc-a chunk:1]",
            }
        ],
        "retrieval_status": "results_found",
        "retrieved_count": 1,
        "retrieved_documents": ["Paper A"],
    }
