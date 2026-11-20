"""One bounded repair call against the original selected passages and draft."""

from ..types.messages import AssistantMessage, UserMessage
from . import rag_citations as citations
from . import visibility
from .rag_citation_evidence import exact_draft_claims, misplaced_identifiers


async def retry_citation_draft(
    llm,
    draft,
    allowed,
    sources,
    misplaced,
    *,
    prior_max_chars,
    system_prompt,
    messages,
    pydantic_messages,
    max_tokens,
    stream_callback,
    stream_message_uuid,
):
    """Return a complete repair or a candidate with more recoverable statements."""
    prior = draft
    if len(prior) > prior_max_chars:
        prior = prior[:prior_max_chars].rstrip() + "\n[Truncated for citation retry.]"
    prompt = citations.build_citation_retry_prompt(
        prior_answer=prior,
        allowed_citations=allowed,
        invalid_citations=citations.find_invalid_citations(draft, allowed),
    )
    if misplaced:
        prompt += (
            "\n\nThese identifiers appear in other selected passages "
            "and are missing from the passages cited at "
            "the claim. Check these passages for the full assertion and "
            "qualifications. A matching name alone does not establish "
            "a relation between documents. Cite passages that establish "
            "the assertion or remove the unsupported detail.\n"
            + "\n".join(
                f"{identifier}: {' '.join(refs)}"
                for identifier, refs in misplaced.items()
            )
        )
    prompt += (
        "\n\nNumeric factual sentences need citations at the claim. "
        "For a computed comparison, cite every source supplying an input "
        "at that comparison; a trailing Sources list is not sufficient."
    )
    retry_args = llm.base_args | {
        "_synthesis_phase": "citation_repair",
        "system": system_prompt,
        "messages": messages
        + [
            {"role": "assistant", "content": draft},
            {"role": "user", "content": prompt},
        ],
        "messages_pydantic": pydantic_messages
        + [
            AssistantMessage(content=draft, stop_reason="stop"),
            UserMessage(content=prompt),
        ],
        "max_tokens": max_tokens,
        "stream_callback": stream_callback,
        "stream_message_uuid": stream_message_uuid,
    }
    try:
        response = await llm.get_message(**retry_args)
    except Exception:
        return None
    if not response or response.tool_call:
        return None
    revised = visibility.sanitize_completed_response(response.text or "", response)
    revised_misplaced = misplaced_identifiers(revised, sources)
    cutoff = response.stop_reason.strip().lower() in {"length", "max_tokens"}
    valid = (
        not cutoff
        and revised
        and not revised_misplaced
        and citations.response_has_required_citations(
            revised,
            allowed,
            require_cited_numeric_claims=True,
        )
    )
    # Publication still checks presentation, authorization and the final text.
    if valid or len(exact_draft_claims(revised, sources)) > len(
        exact_draft_claims(draft, sources)
    ):
        return response, revised, revised_misplaced, cutoff
    return None
