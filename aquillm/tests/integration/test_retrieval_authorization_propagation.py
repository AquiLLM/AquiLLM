"""Task 14 inventory; Task 19 upgrades this to argument propagation."""

import ast
import inspect
from pathlib import Path

INVENTORY = {
    "apps/documents/services/chunk_search_candidates.py": {
        "collect_hybrid_candidate_snapshot",
        "freeze_authorized_document_scope",
    },
    "apps/documents/services/chunk_search.py": {
        "materialize_and_rerank_candidates",
        "text_chunk_search",
    },
    "apps/documents/models/chunks.py": {"text_chunk_search"},
    "apps/chat/services/tool_wiring/documents.py": {
        "search_single_document_tool",
        "vector_search_tool",
    },
    "apps/core/views/pages.py": {"search"},
    "apps/knowledge_graph/evals/run_kg_eval.py": {"run_one_snapshot_comparison"},
}


def test_current_retrieval_authorization_call_site_inventory_is_complete() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative, expected in INVENTORY.items():
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        observed = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert expected <= observed, relative


def test_retrieval_authorization_context_is_explicit_at_every_python_boundary() -> None:
    from apps.chat.services.tool_wiring.documents import (
        search_single_document_tool,
        vector_search_tool,
    )
    from apps.documents.models.chunks import TextChunk
    from apps.documents.services.chunk_search import text_chunk_search
    from apps.documents.services.chunk_search_candidates import (
        collect_hybrid_candidate_snapshot,
    )
    from apps.knowledge_graph.evals.run_kg_eval import run_one_snapshot_comparison

    for function in (
        collect_hybrid_candidate_snapshot,
        text_chunk_search,
        TextChunk.text_chunk_search,
        vector_search_tool,
        search_single_document_tool,
        run_one_snapshot_comparison,
    ):
        assert "authorization_context" in inspect.signature(function).parameters


def test_every_retrieval_forwarding_call_passes_authorization_context_by_keyword(  # noqa: E501
) -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {
        "apps/documents/services/chunk_search.py": {
            ("text_chunk_search", "collect_hybrid_candidate_snapshot"),
        },
        "apps/documents/models/chunks.py": {
            ("text_chunk_search", "hybrid_search"),
        },
        "apps/chat/services/tool_wiring/documents.py": {
            ("vector_search", "text_chunk_search"),
            ("search_single_document", "text_chunk_search"),
        },
        "apps/core/views/pages.py": {("search", "text_chunk_search")},
        "apps/knowledge_graph/evals/run_kg_eval.py": {
            ("run_one_snapshot_comparison", "collect_candidates"),
        },
    }
    for relative, required in expected.items():
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        observed: set[tuple[str, str]] = set()
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            for call in (
                node for node in ast.walk(function) if isinstance(node, ast.Call)
            ):
                callee = call.func
                name = (
                    callee.id
                    if isinstance(callee, ast.Name)
                    else callee.attr
                    if isinstance(callee, ast.Attribute)
                    else ""
                )
                if any(
                    keyword.arg == "authorization_context" for keyword in call.keywords
                ):
                    observed.add((function.name, name))
        assert required <= observed, (relative, required - observed)
