"""Blank human support questions and strict offline response validation."""

from copy import deepcopy

from .evidence_support_metrics import LABELS


DESCRIPTIONS = {
    "quote_accuracy": "Any quotations accurately reproduce the supplied evidence; true if there are no quotations.",
    "source_entailment": "The supplied evidence supports the answer's factual assertions and their cited attribution.",
    "inference": "Explanations or inferences follow from the evidence and are distinguished from directly reported facts.",
    "numerical_conditions": "Numbers, units, comparisons and material experimental conditions are preserved; true if none apply.",
    "contradiction_free": "The answer avoids unsupported consensus and handles any conflicts in the supplied evidence.",
    "coverage": "All material requested aspects are actually answered with supporting evidence; acknowledging a missing aspect does not count as covering it.",
    "false_complete": "The visible answer presents full coverage despite an unsupported material requested aspect; do not infer a hidden system assessment.",
    "unsupported_answer": "The answer makes at least one factual assertion unsupported by the supplied evidence.",
    "abstained": "The answer withholds or limits at least one requested claim because supporting evidence is insufficient.",
    "abstention_correct": "That withholding or limitation is justified by the supplied evidence; false when abstained is false.",
}


def support_template(row):
    return {"support_labels": dict.fromkeys(LABELS)} if "support_decision" in row else {}


def support_question_matches(row, question):
    if "support_decision" not in row:
        return True
    return (type(question) is str and bool(question.strip())
            and row.get("primary_question") == question == row.get("support_question"))


def review_instructions(item):
    if not item.get("support_review"):
        return ["", "### Required claims", "", *[
            f"- {claim['claim_id']}: {claim['statement']}"
            for claim in item["required_claims"]
        ]]
    return [
        "", "### Support judgments", "",
        "Record these judgments from the question, actual answer and supplied evidence first. "
        "Do not copy a model assessment or earlier ratings. Every field requires your own true/false judgment.",
        "The packet and response schemas remain v1. Support-bearing items require all ten support fields; "
        "older v1 forms without them must be re-exported. Legacy items without support decisions retain their existing format.",
        "The existing claim rubric and full audit are in the private packet and are unblinded. "
        "Consult them afterward for claim judgments and the required audit attestation. "
        "A completed form is review input, not calibration or activation approval.",
        "Example: an answer that supports one requested result and honestly withholds a second "
        "can have source_entailment=true, abstained=true, abstention_correct=true, coverage=false "
        "and false_complete=false. Honesty does not supply the missing result.",
        *[f"- {label}: {DESCRIPTIONS[label]}" for label in LABELS],
    ]


def reviewed_support_labels(row, template, response, packet, opaque_id):
    """V1 extension: support-bearing observations cannot import incomplete forms."""
    if "support_decision" not in row:
        if "support_labels" in template or "support_labels" in response:
            raise ValueError("unexpected support judgments for legacy observation")
        return {}
    from .evidence_review_packet import _sheet
    items = [item for item in packet["items"] if item["id"] == opaque_id]
    if (len(items) != 1 or not support_question_matches(row, items[0].get("question"))
            or packet.get("sheet") != _sheet(packet["items"])):
        raise ValueError("support review question or displayed sheet mismatch")
    if any(items[0].get(field) != row.get(field, "" if field == "answer" else [])
           for field in ("answer", "delivered", "citations", "source_bindings", "sdk_payloads", "events")):
        raise ValueError("support review original evidence mismatch")
    expected = {label: None for label in LABELS}
    labels = response.get("support_labels")
    if (
        template.get("support_labels") != expected
        or not isinstance(labels, dict)
        or set(labels) != set(LABELS)
        or any(type(value) is not bool for value in labels.values())
    ):
        raise ValueError("support judgments incomplete or invalid; re-export older support forms")
    return {"support_labels": deepcopy(labels)}
