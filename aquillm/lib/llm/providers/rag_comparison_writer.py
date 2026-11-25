"""Render the comparison writer's own claims and selected source groups."""

from dataclasses import dataclass
from html import unescape
import json
import re
import unicodedata

from lib.llm.answer_publication import FinalCitationPolicy, validate_final_answer
from .rag_citation_extracts import _CITATION_RE
from .rag_citation_prose import numeric_claim_sentences


_RELATIONS = ("added", "removed", "updated", "retained", "designation", "qualification", "other")
_KEY = re.compile(r"A|B|C|S[1-9][0-9]?")
_ACTIVE_OR_CITATION = re.compile(
    r"\[\s*doc\s*:|!?\[[^\]\n]*\]\s*\(|^\s*\[[^\]\n]+\]\s*:"
    r"|<(?:[!?]|/?[a-z][a-z0-9:-]*(?=[\s/>])[^<>]*>)"
    r"|</?(?:script|style|iframe|object|embed|img|svg|math|link|meta|form|input|template)\b"
    r"|[\x00-\x1f\x7f]",
    re.I,
)


def _bounded_text(value, limit):
    return isinstance(value, str) and 0 < len(value) <= limit and bool(value.strip())


def _clean(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True, slots=True)
class ComparisonAnswerPlan:
    brief: str
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    labels: tuple[str, str]
    shared_names: tuple[str, ...] = ()
    references: tuple[tuple[str, int, int, str], ...] = ()

    def __post_init__(self):
        if (not _bounded_text(self.brief, 8000) or not isinstance(self.groups, tuple)
                or not 3 <= len(self.groups) <= 15 or not isinstance(self.labels, tuple)
                or len(self.labels) != 2 or not all(_bounded_text(v, 180) for v in self.labels)
                or not isinstance(self.shared_names, tuple) or len(self.shared_names) > 36
                or not all(_bounded_text(v, 80) for v in self.shared_names)):
            raise ValueError("invalid comparison plan")
        mapping = {}
        for entry in self.groups:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("invalid comparison group")
            key, refs = entry
            if (not isinstance(key, str) or not _KEY.fullmatch(key) or key in mapping
                    or not isinstance(refs, tuple) or not 1 <= len(refs) <= (3 if key in ("A", "B") else 1)
                    or not all(_bounded_text(ref, 180) and _CITATION_RE.fullmatch(ref) for ref in refs)
                    or len(set(refs)) != len(refs)):
                raise ValueError("invalid comparison group")
            mapping[key] = refs
        if not {"A", "B", "C"} <= mapping.keys():
            raise ValueError("missing comparison group")
        owners = [{r.rsplit(" chunk:", 1)[0] for r in mapping[k]} for k in ("A", "B", "C")]
        if any(len(side) != 1 for side in owners) or len(set.union(*owners)) != 3:
            raise ValueError("comparison needs three distinct source owners")
        if not isinstance(self.references, tuple) or len(self.references) not in (0, 3):
            raise ValueError("invalid comparison anchors")
        for key, anchor in zip(("A", "B", "C"), self.references):
            if not isinstance(anchor, tuple) or len(anchor) != 4:
                raise ValueError("invalid comparison anchor")
            ref, start, end, excerpt = anchor
            if (ref not in mapping[key] or type(start) is not int or type(end) is not int
                    or not 0 <= start < end or not _bounded_text(excerpt, 320) or end-start != len(excerpt)):
                raise ValueError("invalid comparison anchor")


def response_format(plan: ComparisonAnswerPlan):
    """The model selects every source pointer. The transport carries the JSON schema."""
    claim = {
        "type": "object", "additionalProperties": False, "required": ["relation", "statement", "sources"],
        "properties": {
            "relation": {"type": "string", "enum": list(_RELATIONS)},
            "statement": {"type": "string"},
            "sources": {"type": "array", "minItems": 1, "maxItems": 6,
                        "items": {"type": "string", "enum": [key for key, _ in plan.groups]}},
        },
    }
    schema = {
        "type": "object", "additionalProperties": False, "required": ["claims"],
        "properties": {"claims": {"type": "array", "minItems": 1, "maxItems": 5, "items": claim}},
    }
    return {"type": "json_schema", "json_schema": {"name": "comparison_claims", "strict": True, "schema": schema}}


def _unique_keys(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate JSON field")
    return obj


def plan_bound(plan: ComparisonAnswerPlan, policy: FinalCitationPolicy) -> bool:
    """Check that every group and exact source anchor still matches the current policy."""
    texts = {source.citation: source.text for source in policy.sources}
    refs = {ref for _, group in plan.groups for ref in group}
    return (
        len(texts) == len(policy.sources)
        and refs <= policy.allowed_citations
        and refs <= texts.keys()
        and all(texts[ref][start:end] == excerpt for ref, start, end, excerpt in plan.references)
    )


def _render_claim(claim, groups, shared_names):
    if not isinstance(claim, dict) or set(claim) != {"relation", "statement", "sources"}:
        return None
    kind, statement, keys = claim["relation"], claim["statement"], claim["sources"]
    if (kind not in _RELATIONS or not _bounded_text(statement, 1100) or len(statement.strip()) < 4
            or _ACTIVE_OR_CITATION.search(unescape(statement)) or not isinstance(keys, list)
            or not 1 <= len(keys) <= 6 or not all(isinstance(k, str) and k in groups for k in keys)
            or len(keys) != len(set(keys))):
        return None
    statement = statement.strip()
    if len(numeric_claim_sentences(statement)) != 1:
        return None
    refs = tuple(dict.fromkeys(ref for key in keys for ref in groups[key]))
    if kind == "designation" and not set(groups["A"]+groups["B"]+groups["C"]) <= set(refs):
        return None
    if kind in ("added", "removed"):
        words = _clean(statement)
        if any(re.search(r"(?<!\w)" + re.escape(_clean(name)) + r"(?!\w)", words) for name in shared_names):
            return None
    punctuation = statement[-1] if statement[-1] in ".!?" else "."
    return statement.removesuffix(punctuation) + " " + "".join(refs) + punctuation


def decode(text, plan: ComparisonAnswerPlan, policy: FinalCitationPolicy) -> str | None:
    """Return grounded-shape prose subject to ordinary presentation and current source authorization."""
    if not isinstance(text, str) or len(text) > 12000:
        return None
    groups = dict(plan.groups)
    if not plan_bound(plan, policy):
        return None
    try:
        obj = json.loads(text, object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"claims"}:
        return None
    claims = obj["claims"]
    if not isinstance(claims, list) or not 1 <= len(claims) <= 5:
        return None
    lines = []
    for claim in claims:
        line = _render_claim(claim, groups, plan.shared_names)
        if line is None:
            return None
        lines.append(line)
    result = "\n\n".join(lines)
    return result if validate_final_answer(result, policy).valid else None
