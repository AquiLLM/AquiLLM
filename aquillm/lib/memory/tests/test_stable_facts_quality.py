"""Quality fixtures for balanced stable-fact extraction."""

from lib.memory.extraction.stable_facts import heuristic_facts_from_turn


def test_explicit_remember_directive_normalizes_to_durable_fact():
    facts = heuristic_facts_from_turn(
        "Please remember going forward that AquiLLM uses Qdrant and Memgraph for memory.",
        "I'll keep that in mind for future graph-memory work.",
    )

    assert "AquiLLM uses Qdrant and Memgraph for memory." in facts
    assert all(not fact.startswith("User asked to remember:") for fact in facts)
    assert all(not fact.startswith("Remembered context:") for fact in facts)


def test_durable_project_tooling_statement_is_retained():
    facts = heuristic_facts_from_turn(
        "We use Qdrant and Memgraph in AquiLLM for long-term memory.",
        "",
    )

    assert "We use Qdrant and Memgraph in AquiLLM for long-term memory." in facts


def test_transient_tactical_turn_is_not_promoted_to_memory():
    facts = heuristic_facts_from_turn(
        "Can you retry the deploy after you summarize the last two failures?",
        "Yes, I'll retry the deploy once I've summarized them.",
    )

    assert facts == []


def test_vague_self_referential_remember_text_is_filtered():
    facts = heuristic_facts_from_turn(
        "Please remember that you should remember this going forward.",
        "Okay, I'll remember that.",
    )

    assert facts == []
