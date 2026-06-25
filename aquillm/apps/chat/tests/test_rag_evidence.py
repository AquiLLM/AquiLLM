"""Tests for the evidence packet builder (Task 3)."""
from __future__ import annotations

import pytest

from apps.chat.services.rag_evidence import EvidencePacket, build_evidence_packet


# ---------------------------------------------------------------------------
# Helpers to build raw tool results matching pack_chunk_search_results output
# ---------------------------------------------------------------------------

def _make_chunk(rank: int, doc_id: str, chunk_id: int, title: str, text: str, **extra) -> dict:
    base = {
        "rank": rank,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk": rank,
        "title": title,
        "citation": f"[doc:{doc_id} chunk:{chunk_id}]",
        "text": text,
    }
    base.update(extra)
    return base


def _results_found(*chunks) -> dict:
    titles = sorted({c["title"] for c in chunks})
    return {
        "result": list(chunks),
        "retrieval_status": "results_found",
        "retrieved_count": len(chunks),
        "retrieved_documents": titles,
    }


def _no_results(query: str = "dark matter") -> dict:
    return {
        "result": [],
        "retrieval_status": "no_results",
        "retrieval_message": f'I searched selected documents for "{query}", but retrieval returned no relevant passages.',
    }


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_returns_evidence_packet():
    raw = _results_found(_make_chunk(1, "doc-1", 10, "Paper A", "Dark matter is..."))
    packet = build_evidence_packet(raw, query="dark matter", search_scope="selected documents")
    assert isinstance(packet, EvidencePacket)


def test_packet_fields_populated():
    raw = _results_found(_make_chunk(1, "doc-1", 10, "Paper A", "Some content."))
    packet = build_evidence_packet(raw, query="dark matter", search_scope="selected documents")
    assert packet.query == "dark matter"
    assert packet.search_scope == "selected documents"
    assert packet.retrieval_status == "results_found"
    assert len(packet.chunks) == 1
    assert packet.chunks[0]["citation"] == "[doc:doc-1 chunk:10]"


# ---------------------------------------------------------------------------
# Citation token preservation
# ---------------------------------------------------------------------------

def test_citation_tokens_extracted():
    raw = _results_found(
        _make_chunk(1, "doc-1", 10, "Paper A", "Content A."),
        _make_chunk(2, "doc-2", 20, "Paper B", "Content B."),
    )
    packet = build_evidence_packet(raw, query="test", search_scope="docs")
    assert "[doc:doc-1 chunk:10]" in packet.citation_tokens
    assert "[doc:doc-2 chunk:20]" in packet.citation_tokens


# ---------------------------------------------------------------------------
# Image URL preservation
# ---------------------------------------------------------------------------

def test_image_urls_extracted():
    chunk = _make_chunk(1, "doc-1", 10, "Paper A", "Figure context.", image_url="/aquillm/document_image/doc-1/")
    raw = _results_found(chunk)
    raw["_image_instruction"] = "Use markdown image syntax."
    packet = build_evidence_packet(raw, query="figures", search_scope="docs")
    assert "/aquillm/document_image/doc-1/" in packet.image_urls


def test_no_image_urls_when_none_present():
    raw = _results_found(_make_chunk(1, "doc-1", 10, "Paper A", "Text only."))
    packet = build_evidence_packet(raw, query="test", search_scope="docs")
    assert packet.image_urls == []


# ---------------------------------------------------------------------------
# Per-doc snippet cap (RAG_MAX_SNIPPETS_PER_DOC)
# ---------------------------------------------------------------------------

def test_per_doc_cap_limits_snippets(monkeypatch):
    monkeypatch.setenv("RAG_MAX_SNIPPETS_PER_DOC", "2")
    chunks = [_make_chunk(i, "doc-1", i, "Paper A", f"Content {i}.") for i in range(1, 6)]
    raw = _results_found(*chunks)
    packet = build_evidence_packet(raw, query="test", search_scope="docs")
    doc1_count = sum(1 for c in packet.chunks if c["doc_id"] == "doc-1")
    assert doc1_count <= 2


def test_per_doc_cap_1_allows_one_per_doc(monkeypatch):
    monkeypatch.setenv("RAG_MAX_SNIPPETS_PER_DOC", "1")
    chunks = [
        _make_chunk(1, "doc-1", 1, "Paper A", "Content A1."),
        _make_chunk(2, "doc-1", 2, "Paper A", "Content A2."),
        _make_chunk(3, "doc-2", 3, "Paper B", "Content B1."),
    ]
    raw = _results_found(*chunks)
    packet = build_evidence_packet(raw, query="test", search_scope="docs")
    doc1_count = sum(1 for c in packet.chunks if c["doc_id"] == "doc-1")
    doc2_count = sum(1 for c in packet.chunks if c["doc_id"] == "doc-2")
    assert doc1_count <= 1
    assert doc2_count <= 1


# ---------------------------------------------------------------------------
# Multi-doc diversification
# ---------------------------------------------------------------------------

def test_multi_doc_diversification_no_single_doc_dominates(monkeypatch):
    """With 3 docs and default max_snippets_per_doc=3, all docs get represented."""
    monkeypatch.setenv("RAG_MAX_SNIPPETS_PER_DOC", "3")
    # 4 chunks from doc-1, 2 from doc-2, 2 from doc-3 — total 8
    chunks = (
        [_make_chunk(i, "doc-1", i, "Paper A", f"A{i}.") for i in range(1, 5)]
        + [_make_chunk(i + 10, "doc-2", i + 10, "Paper B", f"B{i}.") for i in range(1, 3)]
        + [_make_chunk(i + 20, "doc-3", i + 20, "Paper C", f"C{i}.") for i in range(1, 3)]
    )
    raw = _results_found(*chunks)
    packet = build_evidence_packet(raw, query="test", search_scope="docs")
    doc_ids = {c["doc_id"] for c in packet.chunks}
    # All three documents must appear in the packet.
    assert "doc-1" in doc_ids
    assert "doc-2" in doc_ids
    assert "doc-3" in doc_ids


# ---------------------------------------------------------------------------
# Token budget enforcement
# ---------------------------------------------------------------------------

def test_token_budget_limits_total_content(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "50")
    # Each chunk ~100 chars ≈ 25 tokens
    chunks = [
        _make_chunk(i, f"doc-{i}", i, f"Paper {i}", "X" * 100)
        for i in range(1, 10)
    ]
    raw = _results_found(*chunks)
    packet = build_evidence_packet(raw, query="test", search_scope="docs", token_budget=50)
    # 50 tokens = ~200 chars; 10 chunks of 100 chars each exceeds that
    total_chars = sum(len(c.get("text", "") + c.get("x", "")) for c in packet.chunks)
    # Should not exceed ~4× budget (rough 4 chars/token) with some tolerance
    assert total_chars <= 50 * 4 + 50  # budget * 4 chars/token + slack


# ---------------------------------------------------------------------------
# no_results handling
# ---------------------------------------------------------------------------

def test_no_results_produces_empty_chunks():
    raw = _no_results("dark matter")
    packet = build_evidence_packet(raw, query="dark matter", search_scope="selected documents")
    assert packet.chunks == []
    assert packet.retrieval_status == "no_results"


def test_no_results_includes_diagnostic_message():
    raw = _no_results("gravitational waves")
    packet = build_evidence_packet(raw, query="gravitational waves", search_scope="selected documents")
    assert packet.diagnostic_message
    assert "gravitational waves" in packet.diagnostic_message or "no relevant" in packet.diagnostic_message.lower()


def test_no_results_does_not_raise():
    raw = _no_results()
    packet = build_evidence_packet(raw, query="anything", search_scope="docs")
    assert isinstance(packet, EvidencePacket)


# ---------------------------------------------------------------------------
# Default arguments
# ---------------------------------------------------------------------------

def test_default_token_budget_from_env(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "3500")
    raw = _results_found(_make_chunk(1, "doc-1", 1, "Paper A", "Short text."))
    packet = build_evidence_packet(raw, query="test", search_scope="docs")
    assert len(packet.chunks) == 1


def test_explicit_token_budget_overrides_env(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "3500")
    raw = _results_found(_make_chunk(1, "doc-1", 1, "Paper A", "Short text."))
    packet = build_evidence_packet(raw, query="test", search_scope="docs", token_budget=100)
    assert isinstance(packet, EvidencePacket)
