import json

from apps.chat.services.rag_comparison_synthesis import comparison_plan, edition_candidates, reference_alignment
from apps.chat.services.rag_evidence_handoff import prepare_evidence_handoff
from lib.llm.providers.complete_turn_inputs import protected_post_tool_inputs
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage

from .test_rag_comparison_references import packet


def test_alignment_pairs_references_despite_reordering_without_asserting_identity():
    sections = (
        "Instruments: Wren (Cole et al. 1998), Kestrel (Rudd et al. 2001), tern v1 (Dean et al. 1999).",
        "Instruments: Kestrel (Rudd et al. 2001), tern v2 (Li et al. 2008), Wren (Cole et al. 1998).",
    )
    index = reference_alignment(sections)
    assert index["same_printed_references"] == [["Wren", "Wren"], ["Kestrel", "Kestrel"]]
    assert index["left_distinct_references"] == ["tern v1"]
    assert index["right_distinct_references"] == ["tern v2"]
    duplicate = sections[1] + ", another item (Cole et al. 1998)."
    assert reference_alignment((sections[0], duplicate)) is None


def test_edition_hint_requires_a_unique_shared_stem_and_numbering_scheme():
    rows = {
        "left_distinct_references": ["Orchid vx3", "Archive P1", "Archive P2", "Loam X7"],
        "right_distinct_references": ["Orchid vx4", "Archive P6", "Loam Y8", "Gravel X7"],
    }
    assert edition_candidates(rows) == [("Orchid vx3", "Orchid vx4")]


def test_brief_is_opt_in_and_only_contains_current_source_quotations(monkeypatch):
    selected = packet()
    monkeypatch.delenv("RAG_COMPARISON_SYNTHESIS_ENABLED", raising=False)
    assert comparison_plan(selected) is None
    monkeypatch.setenv("RAG_COMPARISON_SYNTHESIS_ENABLED", "1")
    plan = comparison_plan(selected)
    assert plan and len(plan.brief) <= 6500
    assert {ref for _, refs in plan.groups for ref in refs} == set(selected.citation_tokens)
    selected.chunks[-1] = dict(selected.chunks[-1], x="unauthorized replacement")
    assert comparison_plan(selected) is None


def test_handoff_preflights_same_transient_instruction_without_persisting_it(monkeypatch):
    monkeypatch.setenv("RAG_COMPARISON_SYNTHESIS_ENABLED", "1")
    selected = packet()
    current = ToolMessage(content="{}", result_dict={}, tool_name="vector_search",
                          tool_call_id="comparison-test", for_whom="assistant")
    stored, request = prepare_evidence_handoff(Conversation(system="", messages=[current]), selected)
    brief = comparison_plan(selected).brief
    assert brief
    assert brief not in str(stored.model_dump()) and brief not in str(request.model_dump())
    preflight = protected_post_tool_inputs(request, 1024, stage=None, supports_json_schema=True)
    assert preflight["message_list"][-1] == {"role": "user", "content": brief}
    assert preflight["thinking_budget"] == 0
    assert "response_format" in preflight
    assert "response_format" not in protected_post_tool_inputs(request, 1024, stage=None)
    assert stored[-1].result_dict["result"] == request[-1].result_dict["result"]
    assert json.loads(request[-1].content)["result"] == request[-1].result_dict["result"]
    assert protected_post_tool_inputs(stored, 1024, stage=None)["message_list"][-1]["content"] != brief
