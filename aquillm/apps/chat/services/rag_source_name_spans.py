"""Complete local source names; a historical substring is never identity."""

import re

_REQUEST = re.compile(r"\b(?:summarize|explain|describe|review|check|evaluate|compare|contrast)\s+", re.I)
_JOIN = re.compile(r"\b(?:and|with|versus|vs)\s+", re.I)
_TRAILING_JOIN = re.compile(r"\s+(?:and|with|versus|vs)(?:\s+the)?\s*$", re.I)
_LABEL = re.compile(r'\s*(?:(?:titled|named|called)\s+(?=["“]))?', re.I)
_QUOTED = re.compile(r'"[^"\n]+"|“[^”\n]+”')
_FOLLOWUP = re.compile(r"(?:with|about|from|in|of|again|please|which|that)\b")
_GENERIC = frozenset({"paper", "report", "document", "source"})


def _span(question, start, matches, references):
    quoted = _QUOTED.match(question, start)
    if quoted:
        return quoted.end(), quoted.group(0)[1:-1], True
    # Recognized full historical names can contain source-reference words.
    minimum = max((end for begin, end in matches if begin == start), default=start)
    following = [begin for begin, _, _, _ in references if begin >= minimum and begin > start]
    end = min(following, default=len(question))
    literal = question[start:end].rstrip()
    complete = end == len(question)
    if end < len(question):
        join = _TRAILING_JOIN.search(literal)
        if join:
            literal = literal[:join.start()].rstrip()
            complete = True
    return start + len(literal), literal, complete


def source_name_events(question, titles, references):
    """Return events plus occupied spans, without granting substring authority.

    Name starts require a source noun, a request/comparison introduction, explicit
    quotes, or the whole question. Unquoted ends consume all remaining text up to
    an independent reference. Only a clear joining phrase makes that boundary
    complete; other uncertain syntax stays search-only. Punctuation is name text.
    Labels introduce quoted names only after a source noun. A joining word needs
    an earlier recognized reference; it cannot invent a name's left boundary.
    """
    titles = set(titles) - _GENERIC
    matches = [match.span() for title in titles for match in re.finditer(
        r"(?<!\w)" + re.escape(title) + r"(?!\w)", question, re.I)]
    candidates = [(end, begin, True) for begin, end, kind, value in references
                  if kind == "singular" and " " in value]
    candidates.extend((m.end(), m.end(), False) for m in _REQUEST.finditer(question))
    candidates.extend((m.end(), m.end(), False) for m in _JOIN.finditer(question)
                      if any(end <= m.start() for _, end, _, _ in references))
    candidates.extend((m.start(), m.start(), False) for m in _QUOTED.finditer(question))
    if any(begin == 0 for begin, _ in matches):
        candidates.append((0, 0, False))
    names = []
    for offset, occupied_start, source_noun in candidates:
        start = _LABEL.match(question, offset).end() if source_noun else offset
        if start >= len(question):
            continue
        known_start = any(begin == start for begin, _ in matches)
        # Do not reinterpret a genuine pronoun/ordinal or comparison as a name.
        if not known_start and any(begin == start for begin, _, _, _ in references):
            continue
        if source_noun and not known_start and _FOLLOWUP.match(question, start):
            continue
        end, literal, complete = _span(question, start, matches, references)
        if not literal:
            continue
        has_history = any(start <= begin and finish <= end for begin, finish in matches)
        if not source_noun and not has_history:
            continue
        key = literal.casefold()
        kind = "title" if complete and key in titles else "named_source"
        value = key if kind == "title" else question[start:end]
        names.append(((start, end, kind, value), (occupied_start, end)))
    return sorted(names, key=lambda item: (item[0][0], -item[0][1], item[1][0]))
