"""Direct RAG case checking with an explicitly supplied persisted fixture adapter.

Retrieval/provider transport may be synthetic; original principal, collection,
source coordinates and publication guards remain real. The owned test harness
supplies that adapter. Fixture-free CLI execution fails with actionable usage
instead of manufacturing source authority or claiming publication coverage.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import patch
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent

_DEFAULT_CASES_PATH = _HERE / "rag_cases.yaml"

async def _run_case(case: dict, *, case_factory=None, verbose: bool = False) -> dict:
    """Run a single eval case and return a result dict."""
    if case_factory is None:
        raise ValueError("A persisted, authorized case_factory is required; use the owned test harness.")
    from lib.llm.types.messages import AssistantMessage
    from apps.chat.services import rag_pipeline

    consumer, llm, fake_search = await case_factory(case)
    convo = consumer.convo
    retrieval_called: list[bool] = []

    def _tracked_search(c: Any, q: str, top_k: int) -> dict:
        retrieval_called.append(True)
        return fake_search(c, q, top_k)

    monkeypatch_env = {"RAG_DIRECT_ENABLED": "1"}

    # Replace only retrieval transport; original runtime and publication use real DB authority.
    with patch.object(rag_pipeline, "_run_vector_search", _tracked_search):
        # Temporarily set env vars.
        old_env = {k: os.environ.get(k) for k in monkeypatch_env}
        for k, v in monkeypatch_env.items():
            os.environ[k] = v
        try:
            outcome = await rag_pipeline.run_direct_rag_turn(
                consumer, llm, convo, stream_func=None
            )
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    actual_retrieval_called = bool(retrieval_called)
    final_content = ""
    if hasattr(consumer.convo, "messages") and consumer.convo.messages:
        last = consumer.convo.messages[-1]
        if isinstance(last, AssistantMessage):
            final_content = last.content or ""

    expect_outcome = case.get("expect_outcome", "handled")
    expect_retrieval = bool(case.get("expect_retrieval_called", True))
    expect_contains: list[str] = case.get("expect_content_contains") or []

    failures: list[str] = []
    if outcome != expect_outcome:
        failures.append(f"outcome: expected={expect_outcome!r} actual={outcome!r}")
    if actual_retrieval_called != expect_retrieval:
        failures.append(
            f"retrieval_called: expected={expect_retrieval} actual={actual_retrieval_called}"
        )
    for substring in expect_contains:
        if substring.lower() not in final_content.lower():
            failures.append(f"content missing {substring!r}: content={final_content[:120]!r}")

    passed = not failures
    if verbose:
        status_icon = "✓" if passed else "✗"
        print(f"  {status_icon} [{case['id']}] {case.get('description', '')}")
        if not passed:
            for f in failures:
                print(f"      FAIL: {f}")

    return {"id": case["id"], "passed": passed, "failures": failures}


def load_cases(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return list(data.get("cases") or [])


async def _run_all(cases: list[dict], *, case_factory, verbose: bool = True) -> list[dict]:
    results = []
    for case in cases:
        result = await _run_case(case, case_factory=case_factory, verbose=verbose)
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline RAG eval runner")
    parser.add_argument(
        "--cases",
        default=str(_DEFAULT_CASES_PATH),
        help="Path to YAML cases file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures",
    )
    args = parser.parse_args(argv)

    parser.error(
        "Standalone fixture-free execution cannot establish source authority. "
        "Use scripts/run_rag_remediation_tests.py with the owned project and "
        "aquillm/apps/chat/tests/test_rag_eval_runner.py, or call _run_case "
        "with an explicit persisted case_factory inside that harness."
    )


if __name__ == "__main__":
    sys.exit(main())
