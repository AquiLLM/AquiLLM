"""Repair feedback preserves independently supported premises and localizes missing citations."""

from lib.llm.providers import rag_citations as citations


def test_repair_points_to_uncited_bullets_and_preserves_cited_numeric_premise():
    ref = "[doc:source chunk:4]"
    lead = f"The two instruments ran at 6 volts {ref}:"
    bullets = [
        "* The first instrument heated up.",
        "* The second instrument cooled down.",
    ]
    draft = lead + "\n" + "\n".join(bullets)
    assert citations.find_uncited_numeric_claims(draft) == []
    prompt = citations.build_citation_retry_prompt(
        prior_answer=draft, allowed_citations={ref}
    )
    instructions, prior = prompt.split("Previous answer:\n")
    assert prior == draft
    assert "copy the sentence and its allowed citation together" in instructions
    assert "including a lead-in ending in a colon" in instructions
    faults = instructions.split("Revise these uncited factual list items.\n")[1].split(
        "Allow-list:"
    )[0]
    assert all(line in faults for line in bullets)
    assert lead not in faults


def test_numeric_diagnostics_and_factual_fragments_share_one_copy_budget():
    numeric = [f"The reading was {i} units." for i in range(3)]
    bullets = [f"* Separate factual finding number {c}." for c in "abcdef"]
    draft = "\n\n".join(numeric + bullets)
    prompt = citations.build_citation_retry_prompt(
        prior_answer=draft, allowed_citations={"[doc:a chunk:1]"}
    )
    diagnostic = prompt.split("Uncited numeric sentences to revise:\n")[1].split(
        "Allow-list:"
    )[0]
    assert all(sentence in diagnostic for sentence in numeric)
    assert bullets[0] in diagnostic
    assert all(line not in diagnostic for line in bullets[1:])


def test_repaired_bullets_cannot_supply_a_separate_numeric_premise():
    ref = "[doc:source chunk:4]"
    lead = f"The two instruments ran at 6 volts {ref}:"
    bullets = (
        f"\n* The first instrument heated up {ref}.\n* The second cooled down {ref}."
    )
    assert citations.response_has_required_citations(
        lead + bullets, {ref}, require_cited_numeric_claims=True
    )
    assert not citations.response_has_required_citations(
        "The instruments operated at 6 volts:" + bullets,
        {ref},
        require_cited_numeric_claims=True,
    )
