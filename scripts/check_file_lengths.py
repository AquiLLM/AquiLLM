#!/usr/bin/env python3
"""Fail when source files exceed or weaken the reviewed line-count budget."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_LINES = 300

# Exact reviewed maxima for already-tracked hotspots. New files never enter this
# map. A path must match its count: shrinking ratchets down; growth always fails.
_BASELINE_MAX_LINES: dict[str, int] = {
    "aquillm/apps/chat/services/tool_wiring/documents.py": 361,
    "aquillm/apps/chat/tests/test_direct_rag_pipeline.py": 390,
    "aquillm/apps/chat/tests/test_llm_complete_retry.py": 663,
    "aquillm/apps/chat/tests/test_multimodal_messages.py": 348,
    "aquillm/apps/collections/views/api.py": 308,
    "aquillm/apps/documents/models/document.py": 343,
    "aquillm/apps/documents/models/document_types/figure.py": 386,
    "aquillm/apps/documents/services/chunk_rerank_local_vllm.py": 351,
    "aquillm/apps/documents/services/chunk_search.py": 568,
    "aquillm/apps/documents/services/chunk_search_candidates.py": 393,
    "aquillm/apps/documents/services/rag_cache.py": 365,
    "aquillm/apps/documents/tasks/chunking.py": 430,
    "aquillm/apps/documents/tests/test_chunk_search_candidates.py": 337,
    "aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py": 1109,
    "aquillm/apps/documents/tests/test_document_figure_parent_schema.py": 447,
    "aquillm/apps/documents/tests/test_pdf_response.py": 361,
    "aquillm/apps/knowledge_graph/evals/run_kg_eval.py": 5330,
    "aquillm/apps/knowledge_graph/extraction/pipeline.py": 1386,
    "aquillm/apps/knowledge_graph/graph/assembly.py": 3230,
    "aquillm/apps/knowledge_graph/graph/filtering.py": 1711,
    "aquillm/apps/knowledge_graph/graph/invalidation.py": 2543,
    "aquillm/apps/knowledge_graph/models/artifacts.py": 2370,
    "aquillm/apps/knowledge_graph/models/associations.py": 673,
    "aquillm/apps/knowledge_graph/models/entities.py": 908,
    "aquillm/apps/knowledge_graph/models/inputs.py": 325,
    "aquillm/apps/knowledge_graph/models/relations.py": 643,
    "aquillm/apps/knowledge_graph/resolution/canonical.py": 2543,
    "aquillm/apps/knowledge_graph/resolution/collection.py": 4483,
    "aquillm/apps/knowledge_graph/resolution/coreference.py": 1634,
    "aquillm/apps/knowledge_graph/resolution/normalization.py": 369,
    "aquillm/apps/knowledge_graph/resolution/persistence.py": 816,
    "aquillm/apps/knowledge_graph/retrieval/expansion.py": 4099,
    "aquillm/apps/knowledge_graph/retrieval/ppr.py": 555,
    "aquillm/apps/knowledge_graph/services/builds.py": 5236,
    "aquillm/apps/knowledge_graph/services/inspection.py": 421,
    "aquillm/apps/knowledge_graph/services/ontology.py": 825,
    "aquillm/apps/knowledge_graph/services/pruning.py": 834,
    "aquillm/apps/knowledge_graph/tasks.py": 615,
    "aquillm/apps/knowledge_graph/tests/test_build_generation_uniqueness.py": 340,
    "aquillm/apps/knowledge_graph/tests/test_build_idempotency.py": 1737,
    (
        "aquillm/apps/knowledge_graph/tests/test_build_orchestration_postgres_races.py"
    ): 1952,
    "aquillm/apps/knowledge_graph/tests/test_canonical_permissions.py": 367,
    "aquillm/apps/knowledge_graph/tests/test_canonical_resolution.py": 1290,
    "aquillm/apps/knowledge_graph/tests/test_collection_resolution.py": 2115,
    "aquillm/apps/knowledge_graph/tests/test_coreference.py": 2465,
    "aquillm/apps/knowledge_graph/tests/test_document_lifecycle.py": 1150,
    "aquillm/apps/knowledge_graph/tests/test_eval_runner.py": 3141,
    "aquillm/apps/knowledge_graph/tests/test_filtering.py": 739,
    "aquillm/apps/knowledge_graph/tests/test_graph_activation_lifecycle.py": 582,
    "aquillm/apps/knowledge_graph/tests/test_graph_assembly.py": 874,
    "aquillm/apps/knowledge_graph/tests/test_graph_assembly_limits.py": 395,
    "aquillm/apps/knowledge_graph/tests/test_graph_query_envelopes.py": 517,
    "aquillm/apps/knowledge_graph/tests/test_lifecycle_collector_fences.py": 302,
    "aquillm/apps/knowledge_graph/tests/test_lifecycle_origin_prelock.py": 309,
    "aquillm/apps/knowledge_graph/tests/test_management_commands.py": 1412,
    "aquillm/apps/knowledge_graph/tests/test_mention_extraction.py": 1593,
    "aquillm/apps/knowledge_graph/tests/test_models.py": 2231,
    "aquillm/apps/knowledge_graph/tests/test_ontology.py": 540,
    "aquillm/apps/knowledge_graph/tests/test_pruning.py": 540,
    "aquillm/apps/knowledge_graph/tests/test_retrieval_contracts.py": 396,
    "aquillm/apps/knowledge_graph/tests/test_retrieval_expansion.py": 888,
    "aquillm/apps/knowledge_graph/tests/test_retrieval_ppr.py": 506,
    "aquillm/apps/knowledge_graph/tests/test_retrieval_snapshot.py": 717,
    "aquillm/apps/knowledge_graph/tests/test_tasks.py": 1350,
    "aquillm/apps/memory/tests/test_profile_fact_bridge.py": 320,
    "aquillm/aquillm/crawler_tasks.py": 425,
    "aquillm/aquillm/memory.py": 437,
    "aquillm/aquillm/settings.py": 496,
    "aquillm/lib/knowledge_graph/extractors/gliner2_local.py": 653,
    "aquillm/lib/knowledge_graph/tests/test_contracts.py": 388,
    "aquillm/lib/knowledge_graph/tests/test_gliner2_local.py": 722,
    "aquillm/lib/llm/providers/complete_turn.py": 1153,
    "aquillm/lib/llm/providers/openai.py": 324,
    "aquillm/lib/llm/providers/rag_citations.py": 339,
    "aquillm/lib/llm/tests/test_rag_citations.py": 1292,
    "aquillm/lib/llm/tests/test_spin_tool_budget.py": 339,
    "aquillm/lib/llm/utils/context_packer.py": 427,
    "aquillm/lib/memory/mem0/memgraph_compat.py": 651,
    "aquillm/lib/memory/mem0/operations.py": 704,
    "aquillm/lib/memory/tests/test_mem0_graph_mode.py": 390,
    "aquillm/lib/memory/tests/test_mem0_graph_mode_async.py": 436,
    "aquillm/tests/integration/test_knowledge_graph_compose.py": 496,
    "aquillm/tests/integration/test_vllm_extra_args_parser.py": 635,
    "react/src/features/chat/components/Chat.tsx": 371,
    "react/src/features/chat/components/CitationModalProvider.test.tsx": 517,
    "react/src/features/chat/components/CitationModalProvider.tsx": 322,
    "react/src/features/chat/components/PDFCitationModal.test.tsx": 833,
    "react/src/features/chat/components/PDFCitationModal.tsx": 801,
    "react/src/features/chat/components/TextCitationModal.tsx": 334,
    "react/src/features/chat/components/VirtualizedPdfPages.test.tsx": 665,
    "react/src/features/chat/components/VirtualizedPdfPages.tsx": 303,
    "react/src/features/chat/components/pdfPageExtraction.test.ts": 567,
    "react/src/utils/pdfTextMatch.ts": 451,
}


def _line_count(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))


def _find_violations(
    repo: Path,
    baseline: dict[str, int],
) -> list[tuple[str, int, str]]:
    roots = [
        (repo / "aquillm", {".py"}),
        (repo / "react" / "src", {".ts", ".tsx"}),
    ]
    skip_parts = frozenset(
        {"migrations", "__pycache__", "node_modules", "dist", "build"}
    )
    seen: set[str] = set()
    violations: list[tuple[str, int, str]] = []

    for base, suffixes in roots:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in suffixes:
                continue
            if any(p in skip_parts for p in path.parts):
                continue
            rel = path.relative_to(repo).as_posix()
            n = _line_count(path)
            reviewed = baseline.get(rel)
            if reviewed is not None:
                seen.add(rel)
            if n > MAX_LINES:
                if reviewed is None:
                    violations.append((rel, n, f"new file exceeds {MAX_LINES}"))
                elif n > reviewed:
                    violations.append(
                        (rel, n, f"grew beyond reviewed maximum {reviewed}")
                    )
                elif n < reviewed:
                    violations.append(
                        (rel, n, f"ratchet reviewed maximum down from {reviewed}")
                    )
            elif reviewed is not None:
                violations.append(
                    (rel, n, f"remove reviewed path now within {MAX_LINES}")
                )

    for rel in baseline.keys() - seen:
        violations.append((rel, 0, "remove missing reviewed path"))

    return sorted(violations)


def main() -> int:
    violations = _find_violations(REPO, _BASELINE_MAX_LINES)

    if violations:
        print(f"Line-count ratchet violations (default {MAX_LINES}):", file=sys.stderr)
        for rel, n, reason in violations:
            print(f"  {n:5d}  {rel}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
