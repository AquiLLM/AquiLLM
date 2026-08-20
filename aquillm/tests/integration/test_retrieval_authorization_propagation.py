"""Task 14 inventory; Task 19 upgrades this to argument propagation."""

import ast
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
