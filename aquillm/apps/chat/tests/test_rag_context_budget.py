"""Full request capacity and shared legacy/adaptive selection."""

from apps.chat.services.rag_selection_types import (
    SelectionCandidate,
    SelectionLimits,
    SelectionProfile,
)


def test_capacity_contract():
    from apps.chat.services.rag_context_budget import (
        available_evidence_tokens,
        resolve_document_cap,
    )

    assert (
        resolve_document_cap(
            mode="budgeted", legacy_cap=3, explicit_hard_cap=0, final_passage_limit=10
        )
        == 10
    )
    assert (
        resolve_document_cap(
            mode="budgeted", legacy_cap=3, explicit_hard_cap=4, final_passage_limit=10
        )
        == 4
    )
    assert (
        resolve_document_cap(
            mode="legacy", legacy_cap=3, explicit_hard_cap=4, final_passage_limit=10
        )
        == 3
    )
    assert (
        available_evidence_tokens(
            model_context=8192,
            prompt_tokens=3000,
            output_reserve=4096,
            safety_margin=256,
            configured_budget=3500,
        )
        == 840
    )
    assert (
        available_evidence_tokens(
            model_context=0,
            prompt_tokens=0,
            output_reserve=0,
            safety_margin=0,
            configured_budget=3500,
        )
        == 0
    )


def test_shared_legacy_selector_does_not_charge_skipped_passage_slots():
    from apps.chat.services.rag_selection import select_evidence

    candidates = tuple(
        SelectionCandidate(i, "doc", i, text, 1, i, "fp", {})
        for i, text in enumerate(["x" * 1000, "one", "two", "three", "four", "five"], 1)
    )
    profile = SelectionProfile("test", 1, 0, "v1")
    selected = select_evidence(
        candidates,
        profile=profile,
        limits=SelectionLimits(10, 5, 100),
        score_status="rank_fallback",
        mode="legacy",
    )
    assert [c.chunk_id for c in selected.candidates] == [2, 3, 4, 5, 6]
    capped = select_evidence(
        candidates,
        profile=profile,
        limits=SelectionLimits(10, 4, 100),
        score_status="rank_fallback",
        mode="legacy",
    )
    assert len(capped.candidates) == 4


def test_legacy_repeated_document_rotation():
    from apps.chat.services.rag_selection import select_evidence

    candidates = tuple(
        SelectionCandidate(i, doc, i, "ok", 1, i, "fp", {})
        for i, doc in enumerate(["a", "a", "a", "b", "b", "b"], 1)
    )
    result = select_evidence(
        candidates,
        profile=SelectionProfile("test", 1, 0, "v1"),
        limits=SelectionLimits(10, 10, 100),
        score_status="rank_fallback",
        mode="legacy",
    )
    assert [c.chunk_id for c in result.candidates] == [1, 4, 2, 5, 3, 6]


def test_full_synthesis_skeleton_counts_history_tools_images(monkeypatch):
    from types import SimpleNamespace

    from apps.chat.services.rag_context_budget import synthesis_evidence_budget
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import UserMessage

    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "8192")
    llm = SimpleNamespace(base_args={"model": "test"})
    short = Conversation(system="grounding", messages=[UserMessage(content="question")])
    long = Conversation(
        system="grounding",
        messages=[
            UserMessage(content="history " * 3000),
            UserMessage(content="question"),
        ],
    )
    assert synthesis_evidence_budget(
        short, llm, (), output_reserve=4096
    ) > synthesis_evidence_budget(long, llm, (), output_reserve=4096)
