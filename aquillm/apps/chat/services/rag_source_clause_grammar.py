"""Local question syntax separates fresh search intent from source antecedents."""
import re

_SOURCE = frozenset({"paper", "report", "document", "source"})
_QUOTES = re.compile(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]+\x27')
_POSSESSOR = re.compile(r"\b([^\W\d_][\w-]*)['’]s\b", re.I)
_QUANTIFIED = re.compile(
    r"\b(?:explain|summarize|describe|review|evaluate)\s+(?:what\s+)?"
    r"each\s+(?:paper|report|document|source)\b", re.I,
)
_AUXILIARY = r"does|do|did|is|are|was|were|can|could|will|would"
_AUXILIARY_START = re.compile(r"(?:^|[.!?;\n])\s*(" + _AUXILIARY + r")\s+$", re.I)
_INTERROGATIVE = re.compile(
    r"(?:^|[.!?;\n])\s*(?:(?:what|how|why|where|when)\s+)?"
    r"(?:" + _AUXILIARY + r")\s+$", re.I,
)
_NAMED_QUOTE = re.compile(r'\s*(?:(?:named|titled|called)\s+)?["“]', re.I)
_SEARCH = re.compile(r"\b(?:search|find|retrieve)\s+(?:[\w-]+\s+){2,}[\w-]+", re.I)
_QUESTION = re.compile(
    r"(?:^|[.!?;\n])\s*(?:what|how|why|where|when|which)\s+[^?;\n]{3,}\?", re.I,
)
_WORD = r"[^\W\d_][\w-]*"
_PROCESS_NOUN_LIMIT = 6


def auxiliary_source_pronoun(question, start, end):
    """Admit referential auxiliary subjects, excluding dummy 'it is true that'."""
    auxiliary = _AUXILIARY_START.search(question[:start])
    if auxiliary is None or _overlap(
        start, end, (match.span() for match in _QUOTES.finditer(question))):
        return False
    return not (auxiliary.group(1).casefold() in {"is", "was"}
                and re.match(r"\s+true\s+that\b", question[end:], re.I))


def _local_auxiliary_subject(question, start, end, references, quotes):
    """Match a completed local question with the same auxiliary and predicate."""
    auxiliary = _AUXILIARY_START.search(question[:start])
    predicate = re.match(r"\s+(" + _WORD + r")\b", question[end:])
    if auxiliary is None or predicate is None:
        return False
    earlier = re.compile(
        r"(?:^|[.!?;\n])\s*" + re.escape(auxiliary.group(1))
        + r"\s+(?P<subject>(?:" + _WORD + r"ing\s+)?(?:a|an|the)\s+(?:"
        + _WORD + r"\s+){0," + str(_PROCESS_NOUN_LIMIT - 1) + r"}"
        + _WORD + r")\s+" + re.escape(predicate.group(1))
        + r"(?![\w-])[^.!?;\n]*\?", re.I,
    )
    for introduction in earlier.finditer(question, 0, start):
        words = {word.casefold() for word in re.findall(_WORD, introduction.group("subject"))}
        if (not words.intersection(_SOURCE | {word + "s" for word in _SOURCE})
                and not _overlap(introduction.start(), start, quotes)
                and not any(introduction.start() <= left < start
                            for left, _, _, _ in references)):
            return True
    return False


def _local_process(question, start, end, references, quotes):
    """A repeated predicate may refer to an explicit local gerund subject."""
    if not re.search(r"(?:^|[.!?;\n])\s*how\s+did\s+$", question[:start], re.I):
        return False
    predicate = re.match(r"\s+(" + _WORD + r")\b", question[end:])
    if predicate is None:
        return False
    earlier = re.compile(
        r"(?:^|[.!?;:\n])\s*how\s+did\s+(?P<subject>"
        + _WORD + r"ing\s+(?:a|an|the)\s+(?:" + _WORD + r"\s+){0,"
        + str(_PROCESS_NOUN_LIMIT - 1) + r"}" + _WORD + r")\s+"
        + re.escape(predicate.group(1)) + r"(?![\w-])[^.!?;\n]*\?", re.I,
    )
    for introduction in earlier.finditer(question, 0, start):
        words = {word.casefold() for word in re.findall(_WORD, introduction.group("subject"))}
        if (not words.intersection(_SOURCE | {word + "s" for word in _SOURCE})
                and not _overlap(introduction.start(), start, quotes)
                and not any(introduction.start() <= left < start
                            for left, _, _, _ in references)):
            return True
    return False


def _overlap(start, end, spans):
    return any(start < right and end > left for left, right in spans)


def _barrier(start, end, references):
    return any(
        start <= left < end
        and (kind in {"citation", "document", "ordinal", "title", "named_source", "plural"}
             or (kind == "singular" and " " in value))
        for left, _, kind, value in references
    )


def local_reference_spans(question, references):
    """Return only local pronoun spans; never identities or replacement text."""
    quotes = tuple(match.span() for match in _QUOTES.finditer(question))
    local = []
    for start, end, kind, value in references:
        if _overlap(start, end, quotes):
            continue
        cue = value.casefold()
        if kind == "singular" and cue == "it":
            if (_local_process(question, start, end, references, quotes)
                    or _local_auxiliary_subject(question, start, end, references, quotes)):
                local.append((start, end))
        elif kind == "singular" and cue == "its":
            possessors = tuple(_POSSESSOR.finditer(question, 0, start))
            if possessors:
                possessor = possessors[-1]
                if (possessor.group(1).casefold() not in _SOURCE
                        and not _overlap(*possessor.span(), quotes)
                        and not _barrier(possessor.end(), start, references)):
                    local.append((start, end))
        elif kind == "plural" and cue == "them":
            introductions = tuple(_QUANTIFIED.finditer(question, 0, start))
            if introductions:
                introduction = introductions[-1]
                if (not _overlap(*introduction.span(), quotes)
                        and not _barrier(introduction.end(), start, references)):
                    local.append((start, end))
        elif kind == "plural" and cue == "both":
            cite = re.search(r"\bcite\s+$", question[:start], re.I)
            if cite and re.match(r"\s+sources\b", question[end:], re.I):
                preceding = question[:cite.start()]
                # An output directive cannot supply its own missing antecedent.
                unquoted = list(preceding)
                for left, right in quotes:
                    if left < len(unquoted):
                        stop = min(right, len(unquoted))
                        unquoted[left:stop] = " " * (stop - left)
                preceding = "".join(unquoted)
                if _SEARCH.search(preceding) or _QUESTION.search(preceding):
                    local.append((start, end))
    return frozenset(local)


def source_predicate_spans(question, references):
    """Demonstrative interrogative subjects cannot be consumed as title labels."""
    quotes = tuple(match.span() for match in _QUOTES.finditer(question))
    predicates = []
    for start, end, kind, value in references:
        if (kind != "singular"
                or not re.fullmatch(r"(?:this|that)\s+(?:paper|report|document|source)", value, re.I)
                or _overlap(start, end, quotes)
                or not _INTERROGATIVE.search(question[:start])
                or _NAMED_QUOTE.match(question, end)):
            continue
        boundary = re.search(r"[.!?;\n]", question[end:])
        stop = end + boundary.end() if boundary else len(question)
        predicates.append((start, stop))
    return tuple(predicates)


def overlaps_source_predicate(start, end, predicates):
    return _overlap(start, end, predicates)
