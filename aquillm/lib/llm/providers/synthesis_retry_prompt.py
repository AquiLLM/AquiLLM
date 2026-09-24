"""Completion recovery instructions kept outside the turn orchestrator."""

from .complete_turn_policy import DIRECT_SYNTHESIS_GROUNDING


def build_synthesis_retry_prompt(
    query, attempt, stage, wants_figures, allow_evidence_retry
):
    if stage == "direct_synthesis":
        return "\n\n".join(
            [
                f"User request: {query or 'Answer using the selected evidence.'}",
                DIRECT_SYNTHESIS_GROUNDING,
                "Your previous reply was empty or incomplete. Give the "
                "supported answer "
                "now, using only the selected evidence. Do not call "
                "tools, describe "
                "future retrieval, or emit tool markup. Include a "
                "selected figure only "
                "when it helps answer the user's request.",
            ]
        )
    lines = [
        f"User request: {query or 'Answer using the retrieved documents above.'}",
        "",
        (
            "Write a complete, thorough final answer using evidence "
            "already in this conversation."
        ),
        (
            "Use multiple sections when helpful; aim for depth "
            "(typically 400-900 words when sources support it)."
        ),
        (
            "Answer directly in plain text; do not describe future "
            "retrieval steps or emit tool markup."
        ),
    ]
    if wants_figures:
        lines.append(
            "Include relevant figures using markdown image syntax "
            "from tool results "
            "(![caption](url))."
        )
    lines.append("Explain equations and technical terms in readable language.")
    if attempt >= 1:
        lines.append(
            "Your previous reply was empty or incomplete. Synthesize"
            " a thorough answer "
            "from the document text and tool results above."
        )
    if attempt >= 2:
        lines.append(
            "Cover the main thesis, methods or math (with "
            "intuition), and key figures or findings."
        )
    if attempt >= 3 and allow_evidence_retry:
        lines.append(
            "If existing excerpts are too thin for a specific claim,"
            " you may call "
            "vector_search or search_single_document once with a "
            "focused query, then stop."
        )
    return "\n".join(lines)
