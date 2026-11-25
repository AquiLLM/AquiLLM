"""Map a short planner quotation to its original source for ranking only."""

import re
from dataclasses import dataclass
from os import getenv

from django.core.exceptions import ImproperlyConfigured

LEAD_POLICY = "quoted-evidence-leads-v1"


@dataclass(frozen=True, slots=True)
class EvidenceLead:
    chunk_id: int
    source_fingerprint: str
    start: int
    end: int


def evidence_leads_enabled():
    name = "RAG_PLANNER_EVIDENCE_LEADS_ENABLED"
    value = getenv(name, "0").strip().lower()
    if value in {"0", "false", "off", "no"}:
        return False
    if value in {"1", "true", "on", "yes"}:
        return True
    raise ImproperlyConfigured(f"invalid {name}")


def resolve_leads(records, views, budget):
    """Keep uniquely located quotations. PDF line wraps may differ in the reply."""
    if not isinstance(records, list) or len(records) > 2:
        return ()
    leads = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"quote", "chunk_id"}:
            continue
        chunk = record["chunk_id"]
        quote = record["quote"]
        if (
            type(chunk) is not int
            or not isinstance(quote, str)
            or not 1 <= len(quote) <= 350
        ):
            continue
        sources = [view for view in views if view.chunk_id == chunk]
        words = quote.split()
        if len(sources) != 1 or len(words) < 2:
            continue
        source = sources[0]
        if not budget.reserve_lexical_scan(len(quote) + 2 * len(source.text)):
            continue
        left = r"(?<!\w)" if re.match(r"\w", words[0][0]) else ""
        right = r"(?!\w)" if re.match(r"\w", words[-1][-1]) else ""
        pattern = re.compile(left + r"\s+".join(map(re.escape, words)) + right)
        match = pattern.search(source.text)
        if match is not None and pattern.search(source.text, match.start() + 1) is None:
            leads.append(EvidenceLead(chunk, source.source_fingerprint, *match.span()))
    return tuple(dict.fromkeys(leads)) if budget.can_publish() else ()
