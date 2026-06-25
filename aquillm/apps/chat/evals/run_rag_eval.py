"""Offline eval runner for the direct RAG pipeline.

Loads cases from ``rag_cases.yaml``, runs each through
:func:`~apps.chat.services.rag_pipeline.run_direct_rag_turn` using monkeypatched
retrieval and a stub LLM, then prints a pass/fail report.

No live database, LLM API, or Django application state is required.

Usage::

    cd aquillm
    python -m apps.chat.evals.run_rag_eval
    python -m apps.chat.evals.run_rag_eval --cases path/to/other_cases.yaml

Exit code 0 = all cases passed; 1 = one or more failures.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import yaml

# Allow running from the aquillm package root.
_HERE = Path(__file__).resolve().parent
_AQUILLM_ROOT = _HERE.parent.parent.parent
if str(_AQUILLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_AQUILLM_ROOT))

# Django settings must be configured before importing models / services.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aquillm.settings")

try:
    import django
    django.setup()
except Exception:
    pass  # Allow import without full Django in pure-unit scenarios.

from lib.llm.types.conversation import Conversation  # noqa: E402
from lib.llm.types.messages import AssistantMessage, UserMessage  # noqa: E402
from lib.llm.types.response import LLMResponse  # noqa: E402

from apps.chat.refs import CollectionsRef  # noqa: E402
from apps.chat.services import rag_pipeline  # noqa: E402


_DEFAULT_CASES_PATH = _HERE / "rag_cases.yaml"

_SAMPLE_CHUNK = {
    "rank": 1,
    "chunk_id": 1,
    "doc_id": "doc-eval",
    "title": "Eval Paper",
    "text": "This passage contains relevant information about calibration and spectral analysis.",
    "citation": "[doc:doc-eval chunk:1]",
}

_SAMPLE_ANSWER = (
    "Based on the retrieved documents, the calibration and spectral analysis methods "
    "involve flat fields and dark frames [doc:doc-eval chunk:1]."
)


def _build_mock_retrieval(case: dict) -> Any:
    """Return a fake _run_vector_search function matching the case spec."""
    status = case.get("mock_retrieval_status", "results_found")
    count = int(case.get("mock_retrieved_count", 1))

    if status == "no_results":
        result = {
            "result": [],
            "retrieval_status": "no_results",
            "retrieved_count": 0,
            "retrieval_message": (
                f'I searched the selected documents for "{case["message"]}", '
                "but retrieval returned no relevant passages."
            ),
        }
    else:
        chunks = [_SAMPLE_CHUNK] * max(1, count)
        result = {
            "result": chunks,
            "retrieval_status": "results_found",
            "retrieved_count": count,
            "retrieved_documents": ["Eval Paper"],
        }

    def _fake_search(consumer: Any, query: str, top_k: int) -> dict:
        return result

    return _fake_search


class _StubLLM:
    """Minimal LLM stub that returns a canned answer for synthesis."""

    base_args: dict = {}

    async def get_message(self, *args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            text=_SAMPLE_ANSWER,
            tool_call=None,
            stop_reason="end_turn",
            input_usage=1,
            output_usage=1,
        )

    async def complete(
        self, convo: Conversation, max_tokens: int, stream_func: Any = None
    ) -> tuple[Conversation, str]:
        return (
            convo + [AssistantMessage(content=_SAMPLE_ANSWER, stop_reason="end_turn")],
            "changed",
        )

    async def token_count(self, conversation: Conversation, new_message: str | None = None) -> int:
        return 0


def _make_consumer(case: dict) -> SimpleNamespace:
    collection_ids = list(case.get("collection_ids") or [])
    convo = Conversation(
        system="You are a helpful assistant.",
        messages=[UserMessage(content=case["message"])],
    )
    return SimpleNamespace(
        user=object(),
        col_ref=CollectionsRef(collection_ids),
        convo=convo,
    )


async def _run_case(case: dict, *, verbose: bool = False) -> dict:
    """Run a single eval case and return a result dict."""
    consumer = _make_consumer(case)
    convo = consumer.convo
    llm = _StubLLM()
    fake_search = _build_mock_retrieval(case)
    retrieval_called: list[bool] = []

    def _tracked_search(c: Any, q: str, top_k: int) -> dict:
        retrieval_called.append(True)
        return fake_search(c, q, top_k)

    monkeypatch_env = {"RAG_DIRECT_ENABLED": "1"}

    # Patch _run_vector_search so we avoid DB access.
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


async def _run_all(cases: list[dict], *, verbose: bool = True) -> list[dict]:
    results = []
    for case in cases:
        result = await _run_case(case, verbose=verbose)
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

    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    print(f"Running {len(cases)} RAG eval cases from {cases_path.name} ...")

    results = asyncio.run(_run_all(cases, verbose=not args.quiet))

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    print(f"\nResults: {passed}/{len(results)} passed, {failed} failed")

    if failed:
        print("\nFailed cases:")
        for r in results:
            if not r["passed"]:
                print(f"  [{r['id']}] {r['failures']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
