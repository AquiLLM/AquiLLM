#!/usr/bin/env python3
"""Fail when source files exceed or weaken the reviewed line-count budget."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_LINES = 300

# Exact reviewed maxima, including imported development hotspots in the
# 2026-09-22 nongraph backport (see docs/backports/2026-09-22-non-graph-parity.md).
# New work must stay below the default. Each exception must match exactly;
# shrinking ratchets down and growth fails.
_BASELINE_MAX_LINES: dict[str, int] = {
    "aquillm/apps/chat/services/tool_wiring/documents.py": 397,
    "aquillm/apps/chat/tests/cross_document_acceptance_support.py": 462,
    "aquillm/apps/chat/tests/test_direct_rag_pipeline.py": 595,
    "aquillm/apps/chat/tests/test_llm_complete_retry.py": 849,
    "aquillm/apps/chat/tests/test_multimodal_messages.py": 348,
    "aquillm/apps/collections/views/api.py": 308,
    "aquillm/apps/documents/models/document_types/figure.py": 386,
    "aquillm/apps/documents/services/rag_cache.py": 348,
    "aquillm/apps/documents/tasks/chunking.py": 377,
    "aquillm/apps/documents/tests/test_chunk_search_diagnostics.py": 304,
    "aquillm/apps/documents/tests/test_document_figure_parent_schema.py": 447,
    "aquillm/apps/documents/tests/test_pdf_response.py": 453,
    "aquillm/apps/documents/tests/test_rerank_http_cache.py": 318,
    "aquillm/apps/memory/tests/test_profile_fact_bridge.py": 320,
    "aquillm/aquillm/crawler_tasks.py": 425,
    "aquillm/aquillm/memory.py": 450,
    "aquillm/aquillm/settings.py": 401,
    "aquillm/lib/llm/providers/complete_turn.py": 1405,
    "aquillm/lib/llm/providers/openai.py": 445,
    "aquillm/lib/llm/providers/rag_citations.py": 411,
    "aquillm/lib/llm/tests/test_rag_citations.py": 1400,
    "aquillm/lib/llm/tests/test_spin_tool_budget.py": 396,
    "aquillm/lib/llm/utils/context_packer.py": 427,
    "aquillm/lib/memory/mem0/memgraph_compat.py": 651,
    "aquillm/lib/memory/mem0/operations.py": 704,
    "aquillm/lib/memory/tests/test_mem0_graph_mode.py": 390,
    "aquillm/lib/memory/tests/test_mem0_graph_mode_async.py": 436,
    "aquillm/tests/integration/test_vllm_extra_args_parser.py": 682,
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
