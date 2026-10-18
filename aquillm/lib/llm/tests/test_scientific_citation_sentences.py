"""Scientific author-year abbreviations must not split a cited numeric claim."""

import pytest

from lib.llm.providers import rag_citations as citations

_CITATION = "[doc:a chunk:1]"


@pytest.mark.parametrize(
    "author_year",
    [
        "Bosch et al. 2018a",
        "Bosch et al.2018a",
        "Bosch et al.\n2018a",
        "Bosch ET AL. 2018a",
    ],
)
def test_cited_section_reference_keeps_author_year_in_one_claim(author_year):
    # Shape observed in a real draft: the section number precedes the author
    # abbreviation and the allowed citation follows the publication year.
    answer = f"The methods are described in Section 4.9.9 of {author_year} {_CITATION}."

    assert citations.find_uncited_numeric_claims(answer) == []
    assert citations.response_has_required_citations(
        answer, {_CITATION}, require_cited_numeric_claims=True
    )


@pytest.mark.parametrize(
    "uncited",
    [
        "The estimate is 18 according to Bosch et al.",
        "The estimate is 18 according to Bosch et al. 2018a.",
        "The estimate is 18 according to Bosch et al.2018a.",
        "The estimate is 18.",
    ],
)
def test_following_cited_sentence_cannot_supply_an_earlier_claim(uncited):
    answer = f"{uncited} Another estimate is 7 {_CITATION}."

    assert citations.find_uncited_numeric_claims(answer) == [uncited]
    assert not citations.response_has_required_citations(
        answer, {_CITATION}, require_cited_numeric_claims=True
    )


def test_author_year_claim_still_requires_its_own_citation():
    claim = "The methods are described in Section 4.9.9 of Bosch et al. 2018a."
    answer = f"{claim}\n\nSources:\n{_CITATION}"

    assert citations.find_uncited_numeric_claims(answer) == [claim]
    assert not citations.response_has_required_citations(
        answer, {_CITATION}, require_cited_numeric_claims=True
    )


@pytest.mark.parametrize("quantity", ["2020", "2020.5"])
def test_sentence_starting_with_four_digit_quantity_is_not_an_author_year(quantity):
    claim = "The estimate is 18 according to Bosch et al."
    answer = f"{claim} {quantity} observations were collected {_CITATION}."

    assert citations.find_uncited_numeric_claims(answer) == [claim]
    assert not citations.response_has_required_citations(
        answer, {_CITATION}, require_cited_numeric_claims=True
    )


@pytest.mark.parametrize("continuation_indent", ["", "  "])
def test_wrapped_markdown_bullet_uses_its_continuation_citation(continuation_indent):
    answer = (
        "* **Pipeline Updates:** The algorithm subtracts sky on small\n"
        f"{continuation_indent}scales before measurement {_CITATION}."
    )

    assert citations.find_uncited_factual_lines(answer) == []
    assert citations.response_has_required_citations(
        answer, {_CITATION}, require_cited_numeric_claims=True
    )


@pytest.mark.parametrize(
    "separator",
    [
        f"\n\nOther findings {_CITATION}.",
        f"\n* Another finding {_CITATION}.",
        f"\n+ Another finding {_CITATION}.",
        f"\n1) Another finding {_CITATION}.",
        f"\n## Other findings {_CITATION}",
        f"\n```\n{_CITATION}\n```",
    ],
)
def test_bullet_cannot_borrow_citation_from_a_later_markdown_block(separator):
    claim = "* The algorithm subtracts sky on small scales."

    assert citations.find_uncited_factual_lines(claim + separator) == [claim]
    assert not citations.response_has_required_citations(
        claim + separator, {_CITATION}, require_cited_numeric_claims=True
    )


def test_wrapped_bullet_keeps_separate_uncited_numeric_sentence_rejected():
    claim = "The survey collected 30 observations."
    answer = f"* {claim}\n  Another survey collected 7 observations {_CITATION}."

    assert citations.find_uncited_numeric_claims(answer) == [claim]
    assert not citations.response_has_required_citations(
        answer, {_CITATION}, require_cited_numeric_claims=True
    )


@pytest.mark.parametrize("marker", ["+", "1)"])
def test_sibling_bullet_cannot_supply_an_unpunctuated_numeric_claim(marker):
    claim = "There are 30 observations"
    answer = f"* {claim}\n{marker} Another survey collected 7 observations {_CITATION}."

    assert citations.find_uncited_numeric_claims(answer) == [claim]
    assert citations.find_uncited_factual_lines(answer) == [f"* {claim}"]
    assert not citations.response_has_required_citations(
        answer, {_CITATION}, require_cited_numeric_claims=True
    )


def test_release_identifiers_alone_are_not_quantitative_claims():
    answer = (
        "The actual changes from PDR2 to PDR3 include:\n\n"
        f"* A new sky subtraction algorithm {_CITATION}."
    )

    assert citations.find_uncited_numeric_claims(answer) == []
    assert citations.response_has_required_citations(
        answer, {_CITATION}, require_cited_numeric_claims=True
    )


@pytest.mark.parametrize(
    "claim", ["There are 30 observations:", "PDR3 collected 30 observations."]
)
def test_uncited_quantity_is_not_exempted_by_colon_or_release_identifier(claim):
    answer = f"{claim}\n\n* Another survey is described {_CITATION}."

    assert citations.find_uncited_numeric_claims(answer) == [claim]
    assert not citations.response_has_required_citations(
        answer, {_CITATION}, require_cited_numeric_claims=True
    )
