from __future__ import annotations

from lib.llm.providers.tool_evidence import extract_recent_tool_evidence
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage, UserMessage


def test_extract_recent_tool_evidence_reads_verbose_vector_search_rows():
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="What is the book's thesis?"),
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {
                            "chunk_id": 7,
                            "doc_id": "doc-a",
                            "title": "Disputed Inheritance",
                            "text": (
                                "Radick argues that Mendelian genetics did not simply prevail because it was uniquely true. "
                                "He emphasizes that alternative biological programs were historically viable."
                            ),
                        }
                    ]
                },
            ),
        ],
    )

    query, evidence = extract_recent_tool_evidence(convo)

    assert query == "What is the book's thesis?"
    assert evidence
    assert evidence[0][0] == "Disputed Inheritance"
    assert "Mendelian genetics did not simply prevail" in evidence[0][1]


def test_extract_recent_tool_evidence_reads_compact_vector_search_rows():
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Summarize the key point."),
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {
                            "i": 7,
                            "d": "doc-a",
                            "n": "Disputed Inheritance",
                            "x": (
                                "The book frames Mendelian dominance as contingent and argues that Weldonian biology "
                                "could plausibly have become the mainstream alternative."
                            ),
                        }
                    ]
                },
            ),
        ],
    )

    query, evidence = extract_recent_tool_evidence(convo)

    assert query == "Summarize the key point."
    assert evidence
    assert evidence[0][0] == "Disputed Inheritance"
    assert "Weldonian biology" in evidence[0][1]
