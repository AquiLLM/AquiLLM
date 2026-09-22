"""Tests for direct RAG configuration helpers."""
from __future__ import annotations

from apps.chat.services import rag_config


def test_direct_rag_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_DIRECT_ENABLED", raising=False)
    assert rag_config.is_direct_rag_enabled() is False


def test_direct_rag_enabled_when_flag_set(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    assert rag_config.is_direct_rag_enabled() is True


def test_attach_tools_when_collections_selected_default(monkeypatch):
    monkeypatch.delenv("RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED", raising=False)
    assert rag_config.attach_tools_when_collections_selected() is True


def test_synthesis_budget_keeps_thinking_but_bounds_completion(monkeypatch):
    monkeypatch.delenv("RAG_SYNTHESIS_MAX_TOKENS", raising=False)

    assert rag_config.synthesis_max_tokens() == 4096


def test_direct_limit_respects_retrieval_tool_ceiling(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_TOP_K", "999")
    assert rag_config.direct_rag_top_k() == 15


def test_candidate_pool_has_room_for_multiple_documents(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_TOP_K", "2")
    assert rag_config.direct_rag_candidate_top_k() == 6
    monkeypatch.setenv("RAG_DIRECT_TOP_K", "10")
    assert rag_config.direct_rag_candidate_top_k() == 15
    monkeypatch.setenv("RAG_DIRECT_TOP_K", "1")
    assert rag_config.direct_rag_candidate_top_k() == 1
