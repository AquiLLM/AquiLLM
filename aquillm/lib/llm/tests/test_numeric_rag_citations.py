"""Numeric prose requires a local source only when explicitly requested."""

import pytest

from lib.llm.providers import rag_citations as citations

_ALLOWED = {"[doc:a chunk:1]", "[doc:b chunk:2]"}


@pytest.mark.parametrize(
    ("answer", "valid"),
    [
        (
            "Paper A reports 18 litres [doc:a chunk:1]. "
            "Paper B reports 7 litres [doc:b chunk:2]. "
            "The humid-mode capacity is 11 litres lower (18 - 7 = 11).\n\n"
            "Sources:\n- [doc:a chunk:1]\n- [doc:b chunk:2]",
            False,
        ),
        ("Paper A reports 18 litres. [doc:a chunk:1]", True),
        ("Paper A reports 18 litres.[doc:a chunk:1]", True),
        ("Paper A reports 18 litres.\n[doc:a chunk:1]", True),
        (
            "Paper A reports 18 litres. [doc:a chunk:1] "
            "Paper B reports 7 litres. [doc:b chunk:2]",
            True,
        ),
        (
            "Paper A reports 18 litres. [doc:a chunk:1] The difference is 11 litres.",
            False,
        ),
        (
            "Paper A reports 1.5 litres [doc:a chunk:1]. "
            "Paper B reports 1.0 litres [doc:b chunk:2]. "
            "The computed difference is 0.5 litres [doc:a chunk:1] [doc:b chunk:2].",
            True,
        ),
        (
            "Paper A reports 1.5 litres [doc:a chunk:1]. "
            "The computed difference is 0.5 litres.",
            False,
        ),
        ("Paper A reports 18 litres [doc:a chunk:1].", True),
        (
            "The studies describe different outcomes [doc:a chunk:1]. "
            "Further evidence would help explain the difference.",
            True,
        ),
        (
            "The study reports a finding [doc:a chunk:1].\n\n"
            "## 2026 results\n\n```python\ncapacity = 18\n```\n"
            "![Figure 2](/aquillm/document_image/2026/)\n\n"
            "18 - 7 = 11\n\nSources:\n1. [doc:a chunk:1]\n2. [doc:b chunk:2]",
            True,
        ),
        (
            "The study reports a finding [doc:a chunk:1].\n\n"
            "The capacity is 18 litres.\nSources:\n[doc:a chunk:1]",
            False,
        ),
        (
            "The capacity is 18 litres\n**Sources**:\n[doc:a chunk:1]",
            False,
        ),
    ],
)
def test_numeric_prose_requires_local_citations_without_counting_reference_ids(
    answer,
    valid,
):
    assert (
        citations.response_has_required_citations(
            answer,
            _ALLOWED,
            require_cited_numeric_claims=True,
        )
        is valid
    )


def test_numeric_prose_check_is_opt_in_for_existing_tool_loop_callers():
    answer = "Paper A reports 18 litres [doc:a chunk:1]. The difference is 11 litres."
    assert citations.response_has_required_citations(answer, _ALLOWED)
