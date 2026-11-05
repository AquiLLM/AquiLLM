"""Small prose-boundary helpers for mechanical citation-presence checks."""

import re


def numeric_claim_sentences(marked):
    # Closed parenthesized years can continue into prose. Bare years still need
    # a boundary, so "et al. 2020 observations..." remains separate sentences.
    protected = re.sub(
        r"(?i)(\bet[ \t]+al)\."
        r"(?=[ \t]*(?:\(\d{4}[a-z]?\)"
        r"|\d{4}[a-z]?(?:[ \t]*(?:[\x00),;:!?]|\.(?!\d))|[ \t]*$)))",
        lambda match: match[1] + "\x01",
        marked,
    )
    return [
        sentence.replace("\x01", ".")
        for sentence in re.split(
            r"(?<=[.!?])\s+|(?<=[.!?][\"')\u201d\u2019])\s+", protected
        )
    ]


def logical_factual_lines(text, bullet_pattern):
    """Join a list item's wrapped paragraph, without crossing another block."""
    pending = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        bullet = bool(bullet_pattern.match(line))
        boundary = (
            not line
            or bullet
            or line.startswith(("#", ">", "```", "~~~"))
            or re.fullmatch(r"(?:[-*_]\s*){3,}", line)
        )
        if pending is not None and boundary:
            yield pending
            pending = None
        if bullet:
            pending = raw_line
        elif pending is not None:
            pending += " " + line
        else:
            yield raw_line
    if pending is not None:
        yield pending
