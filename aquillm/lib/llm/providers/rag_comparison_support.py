"""Let the model choose paired inventory evidence or quote a source reporting a change."""

import json
import re

from .rag_comparison_writer import ComparisonAnswerPlan, _unique_keys, plan_bound
from ..answer_publication import FinalCitationPolicy


PAIRED_SUPPORT_INSTRUCTION = (
    "\nAB is a model-selectable citation group comprising both complete inventories A and B. "
    "For any addition, removal, version update, or retention inferred by comparing their member lists, "
    "select AB for that claim and set reporting_quote to the empty string. Merely appearing in a later "
    "list cannot establish addition. Merely appearing in both does not establish when it was introduced. "
    "A single primary source suffices only when it explicitly states that specific change itself: "
    "select the relevant source and copy its exact, complete, short change clause in reporting_quote. "
    "A passage saying the list was updated and then listing members does not say those members were added. "
    "Keep other claims' reporting_quote empty. AB is your choice; no citation is supplied automatically."
)

_CHANGE = {
    "added": re.compile(r"\b(?:adds?|added|introduced|introduces|newly (?:installed|included|listed))\b", re.I),
    "removed": re.compile(r"\b(?:removed|removes|dropped|drops|excluded|excludes|discontinued)\b", re.I),
    "updated": re.compile(r"\b(?:updated|updates|upgraded|upgrades|replaced|replaces|superseded)\b", re.I),
    "retained": re.compile(r"\b(?:retained|retains|kept|remained|remains|continued|continues)\b", re.I),
}
_UNCERTAIN = re.compile(r"\b(?:not|never|if|might|could|would|perhaps|allegedly)\b", re.I)
_CLAUSE_BOUNDARIES = ".!?;:\n•"
_RECORD = r"(?:inventor(?:y|ies)|catalog(?:ue)?s?|registers?|lists?|reports?|editions?|guides?|releases?)"
_RECORD_MODIFIER = r"(?:the|this|both|each|new|old|earlier|later|original|previous|revised|updated)"
_RECORD_PHRASE = rf"(?:{_RECORD_MODIFIER}\s+){{0,4}}{_RECORD}\b"
_ACTIVE_MEMBERSHIP = re.compile(rf"^{_RECORD_PHRASE}\s+(\w+)\s+(.+)", re.I)
_PASSIVE_MEMBERSHIP = re.compile(r"^(.{1,160}?)\s+(?:was|were|is|are)\s+(\w+)\b", re.I)
_MEMBERSHIP_TARGET = re.compile(rf"\s+(?:in|from|to|by)\s+{_RECORD_PHRASE}", re.I)


def paired_schema(form: dict) -> dict:
    """Add the pair alias and reporting quote to this request's own schema."""
    claim = form["json_schema"]["schema"]["properties"]["claims"]["items"]
    keys = claim["properties"]["sources"]["items"]["enum"]
    if "AB" not in keys:
        keys.append("AB")
    if "reporting_quote" not in claim["required"]:
        claim["required"].append("reporting_quote")
    claim["properties"]["reporting_quote"] = {"type": "string"}
    return form


def _keys(chosen, groups):
    if not isinstance(chosen, list) or not 1 <= len(chosen) <= 6:
        return None
    if any(not isinstance(key, str) or (key != "AB" and key not in groups) for key in chosen):
        return None
    if len(chosen) != len(set(chosen)):
        return None
    return list(dict.fromkeys(key for item in chosen for key in (("A", "B") if item == "AB" else (item,))))


def _whole_clause(quote, text):
    start = text.find(quote)
    if start == -1:
        return False
    before = text[:start].rstrip(" \t")
    after = text[start + len(quote):].lstrip(" \t")
    return ((not before or before[-1] in _CLAUSE_BOUNDARIES)
            and (not after or quote[-1] in _CLAUSE_BOUNDARIES or after[0] in _CLAUSE_BOUNDARIES))


def _explicit_report(claim, quote, texts):
    # Keep reporting authority only for an independently checkable complete assertion.
    # Narrower snippets may hide the original sentence's negation or change the subject.
    if (not 10 <= len(quote) <= 340 or not _CHANGE[claim["relation"]].search(quote)
            or _UNCERTAIN.search(quote) or re.search(r"[;!?\n]|\.\s+[A-Z]", quote)):
        return False
    canonical = lambda value: " ".join(value.casefold().strip().rstrip(".!?").split())
    return (canonical(claim["statement"]) == canonical(quote)
            and any(_whole_clause(quote, text) for text in texts))


def _asserted_change(statement, texts, shared_names=()):
    """Recognize a direct inventory predicate only for a member named in the selected evidence."""
    text = statement.strip().rstrip(".!?")
    if _UNCERTAIN.search(text):
        return None
    active = _ACTIVE_MEMBERSHIP.match(text)
    if active:
        verb, objects = active.groups()
        member = re.split(r",|\s+(?:and|with)\s+", objects, maxsplit=1, flags=re.I)[0]
    else:
        passive = _PASSIVE_MEMBERSHIP.match(text)
        if not passive:
            return None
        member, verb = passive.groups()
        if not _MEMBERSHIP_TARGET.match(text, passive.end()):
            # A bare change is unambiguous only when the entire subject names a proven shared entry.
            if passive.end() == len(text) and verb.casefold() in ("added", "removed"):
                subject = " ".join(member.casefold().split())
                if any(subject in (name.casefold(), "the " + name.casefold()) for name in shared_names):
                    return verb.casefold()
            return None
    relation = next((kind for kind, verbs in _CHANGE.items() if verbs.fullmatch(verb)), None)
    if relation is None:
        return None
    member = re.sub(r"^(?:the|a|an)\s+", "", member.strip(), flags=re.I)
    if not 2 <= len(member) <= 160:
        return None
    words = r"\s+".join(re.escape(word) for word in member.split())
    mention = re.compile(r"(?<!\w)" + words + r"(?!\w)", re.I)
    return relation if any(mention.search(source) for source in texts) else None


def prepare_paired_claims(raw: str, plan: ComparisonAnswerPlan, policy: FinalCitationPolicy):
    """Return decoded claims and indices needing an explicit support decision, or None."""
    if not isinstance(raw, str) or len(raw) > 12_000 or not plan_bound(plan, policy):
        return None
    try:
        obj = json.loads(raw, object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError):
        return None
    if not isinstance(obj, dict) or obj.keys() - {"claims", "naming_decision"}:
        return None
    claims = obj.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= 4:
        return None
    groups = dict(plan.groups)
    both = set(groups["A"] + groups["B"])
    texts = {source.citation: source.text for source in policy.sources}
    needs = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict) or set(claim) != {"relation", "statement", "sources", "reporting_quote"}:
            return None
        relation, statement = claim["relation"], claim["statement"]
        if not isinstance(relation, str) or not isinstance(statement, str) or not 4 <= len(statement) <= 1100:
            return None
        if relation not in _CHANGE and relation not in ("other", "qualification", "designation"):
            return None
        inferred = _asserted_change(statement, texts.values(), plan.shared_names)
        if relation == "designation" and inferred in ("added", "removed"):
            return None
        if inferred and relation != "designation":
            relation = claim["relation"] = inferred
        chosen = _keys(claim["sources"], groups)
        quote = claim.pop("reporting_quote")
        if chosen is None or not isinstance(quote, str) or len(quote) > 340:
            return None
        claim["sources"] = chosen
        references = {ref for key in chosen for ref in groups[key]}
        if (relation in _CHANGE and not both <= references
                and not _explicit_report(claim, quote, (texts[ref] for ref in references))):
            needs.append(index)
    naming = obj.get("naming_decision")
    if isinstance(naming, dict) and naming.get("sources"):
        chosen = _keys(naming["sources"], groups)
        if chosen is not None:
            naming["sources"] = chosen
    return obj, tuple(needs)


def _choices(reply, indices):
    if not isinstance(reply, str) or len(reply) > 1600:
        return {}
    try:
        obj = json.loads(reply, object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError):
        return {}
    if not isinstance(obj, dict) or set(obj) != {"choices"} or not isinstance(obj["choices"], list):
        return {}
    decisions = {}
    for row in obj["choices"]:
        if not isinstance(row, dict) or set(row) != {"i", "source"}:
            return {}
        index = row["i"]
        if type(index) is not int or index not in indices or index in decisions or row["source"] not in ("AB", "omit"):
            return {}
        decisions[index] = row["source"]
    return decisions


def apply_paired_choices(obj: dict, indices: tuple[int, ...], reply: str | None) -> dict:
    """Keep all unaffected claims. Only an explicit model choice can attach the A+B citations."""
    choices = _choices(reply, indices)
    claims = []
    for index, claim in enumerate(obj["claims"]):
        if index not in indices:
            claims.append(claim)
        elif choices.get(index) == "AB":
            claims.append({**claim, "sources": ["A", "B"]})
    return {**obj, "claims": claims}
