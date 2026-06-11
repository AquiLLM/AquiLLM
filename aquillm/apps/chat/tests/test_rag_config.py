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
