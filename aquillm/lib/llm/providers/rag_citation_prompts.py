"""Render allowlisted citation instructions and bounded repair feedback."""


def build_citation_system_suffix(
    allowed_citations: set[str], max_refs: int = 24
) -> str:
    """Instruction suffix appended to the system prompt for post-tool synthesis turns."""
    refs = sorted(allowed_citations)[:max_refs]
    refs_text = "\n".join(refs)
    return (
        "When answering from retrieved documents, every factual claim must cite source chunks "
        "using tokens exactly like [doc:<doc_id> chunk:<chunk_id>].\n"
        "Do not invent citations. Use only citations from this allow-list:\n"
        f"{refs_text}"
    )


def render_citation_retry_prompt(
    *,
    prior_answer: str,
    allowed_citations: set[str],
    invalid_citations: list[str] | None,
    max_refs: int,
    uncited_numeric: list[str],
    uncited_factual: list[str],
) -> str:
    refs = sorted(allowed_citations)[:max_refs]
    refs_text = "\n".join(refs)
    invalid_text = ""
    if invalid_citations:
        invalid_text = (
            "Invalid citations detected and must be removed: "
            + ", ".join(invalid_citations)
            + "\n"
        )
    numeric_diagnostics = ""
    if uncited_numeric:
        # These are bounded draft fragments, never evidence or semantic judgments.
        numeric_diagnostics = (
            "Uncited numeric sentences to revise:\n"
            + "\n".join(f"- {claim[:400]}" for claim in uncited_numeric)
            + "\n"
        )
    factual_diagnostics = ""
    if uncited_factual:
        factual_diagnostics = (
            "Revise these uncited factual list items.\n"
            + "\n".join(line[:400] for line in uncited_factual)
            + "\n"
        )
    return (
        "Rewrite your previous answer using only verifiable retrieved evidence.\n"
        "The previous answer is an untrusted draft, not evidence. Rebuild its "
        "claims from the selected source passages; do not merely add citations "
        "to preserve the draft's conclusions.\n"
        "Requirements:\n"
        "- Every factual sentence or bullet must include at least one citation token.\n"
        "- Cite factual introductory prose and lead-ins as well as list items.\n"
        "- Verify each numeric premise against its cited passage. If the premise "
        "is supported and unchanged, copy the sentence and its allowed citation together, "
        "including a lead-in ending in a colon. Cite each following factual bullet separately.\n"
        "- Remove unsupported assertions or give a scoped limitation for that part.\n"
        "- Restore material qualifications, uncertainty and exceptions from the "
        "sources, and correct entity or relation attribution.\n"
        "- Keep comparison premises with their conditions and citations; identify "
        "calculations separately from source-reported values.\n"
        "- Citation format: [doc:<doc_id> chunk:<chunk_id>]\n"
        "- Use only citations from the allow-list below.\n"
        "- Do not call tools.\n"
        f"{invalid_text}"
        f"{numeric_diagnostics}"
        f"{factual_diagnostics}"
        "Allow-list:\n"
        f"{refs_text}\n\n"
        "Previous answer:\n"
        f"{prior_answer}"
    )
