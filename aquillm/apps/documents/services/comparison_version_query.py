"""Conservative literal version comparisons and retrieved self-declarations."""

from hashlib import sha256
import re

ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split()
ORDINAL = r"(?:" + "|".join(ORDINALS) + r"|[1-9][0-9]?(?:st|nd|rd|th))"
VERSION = r"(?:release|revision|version|edition|generation)"
AXIS = r"(?:[A-Za-z]+ ){0,2}" + VERSION
FROM_ORDINAL = re.compile(
    r"\bfrom (?:the )?(?P<left>" + ORDINAL + r") (?P<axis>" + AXIS +
    r") to (?:the )?(?P<right>" + ORDINAL + r") (?P<other_axis>" + AXIS + r")\b", re.I)
FROM_NUMBER = re.compile(
    r"\bfrom (?:the )?(?P<axis>" + VERSION + r") (?P<left>[0-9]{1,3}) "
    r"to (?:the )?(?P<other_axis>" + VERSION + r") (?P<right>[0-9]{1,3})(?![\w.])", re.I)
UNSAFE_QUERY = re.compile(
    r"\b(?:not|nor|neither|no|without|unless|except|excluding|respectively|or|instead)\b|[;(){}\[\]\"“”]", re.I)
SELF_DECLARATION = re.compile(
    r"\b(?:(?:this|the current)\s+(?:paper|report|work|study|document|article)\s+"
    r"(?:presents?|introduces?|describes?|reports?)|we\s+(?:have\s+)?"
    r"(?:present(?:ed)?|introduce(?:d)?|describe(?:d)?))\b", re.I)
UNSAFE_DECLARATION = re.compile(
    r"\b(?:not|never|uses?|using|comparison|earlier\s+(?:work|study)|as\s+reported|"
    r"upcoming|expect(?:ed)?|plan(?:ned)?|will|anticipat\w*|predict\w*)\b", re.I)


def _number(value):
    lower = value.casefold()
    return ORDINALS.index(lower) + 1 if lower in ORDINALS else int(re.match(r"[0-9]+", value)[0])


def parse_version_comparison(question):
    """Accept a single explicit two-version question with a named shared project."""
    if (not isinstance(question, str) or len(question) > 600 or question.count("?") != 1
            or not question.endswith("?") or UNSAFE_QUERY.search(question)):
        return None
    matches = [(kind, match) for kind, pattern in (("ordinal", FROM_ORDINAL), ("number", FROM_NUMBER))
               for match in pattern.finditer(question)]
    if len(matches) != 1:
        return None
    kind, match = matches[0]
    if (match["axis"].casefold() != match["other_axis"].casefold()
            or _number(match["left"]) == _number(match["right"])):
        return None
    for value in (match["left"], match["right"]):
        if value[0].isdigit() and not value.isdigit():
            number = _number(value)
            suffix = "th" if 10 < number % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
            if not value.casefold().endswith(suffix):
                return None
    axis = re.escape(match["axis"])
    labels = re.compile(r"\b" + ORDINAL + " " + axis + r"\b", re.I) if kind == "ordinal" else re.compile(
        r"\b" + axis + r" [0-9]{1,3}(?![\w.])", re.I)
    if len(list(labels.finditer(question))) != 2:
        return None
    if not re.search(r"\b(?:chang\w*|differ\w*|compar\w*|evolv\w*|transition\w*|increas\w*|decreas\w*)\b",
                     question[:match.start()], re.I):
        return None
    if re.search(r"\b(?:and|versus|from|to|compared|both|either|this|that|these|those|their|its)\b",
                 question[match.end():], re.I):
        return None
    project = re.search(r"\b(?:of|for|in)\s+([A-Z][\w-]{1,32})\?$", question)
    if project is None:
        return None
    return dict(kind=kind, axis=match["axis"], labels=(match["left"], match["right"]), project=project[1])


def resolve_primary_documents(question, views):
    """Use only the first five charged source views and unambiguous self-attribution."""
    comparison = parse_version_comparison(question)
    if comparison is None:
        return dict(documents=(), witnesses=())
    words = comparison["axis"].casefold().split()
    descriptors = "".join(r"(?:" + re.escape(word) + r"\s+)?" for word in words[:-1]) + re.escape(words[-1])
    labels = [re.compile(r"\b" + (re.escape(label) + r"\s+" + descriptors
              if comparison["kind"] == "ordinal" else descriptors + r"\s+" + re.escape(label))
              + r"(?![\w.-])", re.I) for label in comparison["labels"]]
    project = re.compile(r"(?<!\w)" + re.escape(comparison["project"]) + r"(?!\w)", re.I)
    observed, witnesses = [set(), set()], []
    for view in views[:5]:
        body = view.text
        for declaration in SELF_DECLARATION.finditer(body):
            start = declaration.start()
            tail = body[declaration.end():declaration.end() + 320]
            stop = re.search(r"[!?;]|\.(?!\d)", tail)
            end = declaration.end() + (stop.start() if stop else len(tail))
            sentence = body[start:end]
            before = body[max(0, start - 40):start]
            attributed = re.search(r"(?:says?|wrote|reports?|quotes?|writ(?:es|ing))\s*[:,]?\s*$|[“‘\"]\s*$", before, re.I)
            if attributed or UNSAFE_DECLARATION.search(sentence) or not project.search(sentence):
                continue
            sides = [index for index, pattern in enumerate(labels) if pattern.search(sentence)]
            if len(sides) != 1:
                continue
            side = sides[0]
            observed[side].add(view.document_id)
            witnesses.append((side, view.chunk_id, view.document_id, start, end,
                              sha256(sentence.encode()).hexdigest()))
    if not all(len(documents) == 1 for documents in observed) or observed[0] == observed[1]:
        return dict(documents=(), witnesses=())
    return dict(documents=tuple(next(iter(documents)) for documents in observed), witnesses=tuple(witnesses))
