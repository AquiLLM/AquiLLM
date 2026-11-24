"""Locate corroborating, printed references in an already-selected comparison."""

from dataclasses import dataclass
import re
import unicodedata

from apps.documents.services.named_item_projection import BOUNDARY, MARKER

from .rag_comparison_selection import _COMPARISON, _next_start, _paired_sections


_REFERENCE = re.compile(r"\(([\w^´ˆˇ\-\s\u0300-\u036f]{1,45}?)\s+et\s+al[.]?,?\s+((?:19|20)\d{2}[a-z]?)\)")
_NAME = re.compile(r"(?:the\s+)?([A-Z][A-Z0-9-]{1,15}(?:\s+[A-Za-z0-9-]{1,16}){0,5})\s*")
_FAMILY = re.compile(r"[A-Z][A-Z0-9-]{2,15}")
_VERSION = re.compile(r"\b[A-Z]{1,8}[-.]?\d+[A-Za-z0-9.-]*\b")
_NAMING = re.compile(r"\b(?:name[d]?|designat(?:ed|ion)|catalog(?:u?ed|ing)|known|register(?:ed)?|identif(?:ied|ies)|alias)\b", re.I)
_CONTRAST = re.compile(r"\b(?:not|never|distinct|different|separate|unrelated|unclear|unknown|ambiguous|cannot|multiple)\b", re.I)
_AUTHOR_YEAR_PERIOD = re.compile(r"(?<=\bet al)\.(?=\s+(?:\(\d{4}[a-z]?\)|\d{4}[a-z]?\b))", re.I)
_MAX_ROWS = 12
_MAX_INPUT = 96_000
_MAX_MENTIONS = 160


@dataclass(frozen=True)
class ComparisonReference:
    document_id: str
    chunk_id: int
    start: int
    end: int
    label: str
    text: str
    inventory_citations: tuple[str, ...]

    @property
    def citation(self):
        return f"[doc:{self.document_id} chunk:{self.chunk_id}]"


@dataclass(frozen=True)
class ComparisonReferenceContext:
    anchors: tuple[ComparisonReference, ComparisonReference, ComparisonReference]
    sections: tuple[str, str]


@dataclass(frozen=True)
class _Mention:
    reference: tuple[str, str]
    label: str
    start: int
    end: int

    @property
    def name(self):
        return " ".join(self.label.split())


def _matches(row, keys, expected, *, optional=False):
    seen = [row[key] for key in keys if key in row]
    return (optional or bool(seen)) and all(type(x) is type(expected) and x == expected for x in seen)


def _current_selection(packet):
    selection = packet.selection
    if not packet.source_mode or packet.source_authorization is None or selection is None:
        return ()
    rows = selection.candidates
    if not 3 <= len(rows) <= _MAX_ROWS or len(rows) != len(packet.chunks) or len(rows) != len(packet.source_evidence):
        return ()
    if sum(len(row.text) for row in rows) > _MAX_INPUT or len({row.chunk_id for row in rows}) != len(rows):
        return ()
    if len({(row.doc_id, row.chunk_number) for row in rows}) != len(rows):
        return ()
    permitted = set(packet.citation_tokens)
    for row, emitted, proof in zip(rows, packet.chunks, packet.source_evidence, strict=True):
        if row.prepared_evidence is not proof or proof.source_coverage != "complete":
            return ()
        source = proof.source
        if (source.document_id, source.chunk_id, source.chunk_number, source.source_fingerprint, source.text) != (
            row.doc_id, row.chunk_id, row.chunk_number, row.source_fingerprint, row.text
        ):
            return ()
        ref = f"[doc:{row.doc_id} chunk:{row.chunk_id}]"
        if ref not in permitted:
            return ()
        for data in (emitted, row.row):
            if not (
                _matches(data, ("d", "doc_id"), row.doc_id)
                and _matches(data, ("i", "chunk_id"), row.chunk_id)
                and _matches(data, ("c", "chunk"), row.chunk_number)
                and _matches(data, ("x", "text"), row.text)
                and _matches(data, ("ref", "citation"), ref, optional=True)
            ):
                return ()
    return rows


def _mentions(text):
    found = []
    for match in _REFERENCE.finditer(text):
        if len(found) == _MAX_MENTIONS:
            return None
        author = "".join(c for c in unicodedata.normalize("NFKD", match[1]).casefold() if c.isalpha())
        prior = text[max(0, match.start()-120):match.start()]
        clause = re.split(r"[,;():•]", prior)[-1]
        cleaned = clause.lstrip()
        named = _NAME.fullmatch(cleaned)
        label = named[1] if named else ""
        start = match.start() - len(cleaned) + named.start(1) if named else match.start()
        found.append(_Mention((author, match[2]), label, start, match.end()))
    return found


def _section_coordinates(text, members):
    joined = members[0].text
    positions = [(0, members[0])]
    for row in members[1:]:
        at, previous = positions[-1]
        overlap = _next_start(previous, row)
        if overlap is None:
            return ()
        at += overlap
        joined = joined[:at] + row.text
        positions.append((at, row))
    start = joined.find(text)
    if start < 0 or joined.find(text, start+1) >= 0:
        return ()
    return tuple((at-start, row) for at, row in positions)


def _anchor(mention, positions, inventory_citations):
    for offset, row in positions:
        a, b = mention.start-offset, mention.end-offset
        if 0 <= a < b <= len(row.text):
            return ComparisonReference(row.doc_id, row.chunk_id, a, b, mention.label, row.text[a:b], inventory_citations)
    return None


def _by_reference(mentions):
    groups = {}
    for mention in mentions:
        groups.setdefault(mention.reference, {})[mention.name] = mention
    return groups


def _local_scope(text, occurrence):
    # Another printed section or paragraph cannot corroborate this entry.
    left = max(0, occurrence.start-1800)
    right = min(len(text), occurrence.end+1800)
    for boundary in (BOUNDARY, MARKER):
        for match in boundary.finditer(text, left, occurrence.start):
            left = match.end()
        following = boundary.search(text, occurrence.end, right)
        if following:
            right = following.start()
    return left, right


def _corroborated(text, mention, independent, neighbors, primary_label, versions):
    left, right = _local_scope(text, mention)
    local = " ".join(text[max(left, mention.start-450):min(right, mention.end+450)].split())
    explicit = False
    primary = re.compile(r"(?<!\w)" + re.escape(primary_label) + r"(?!\w)", re.I)
    version = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, versions)) + r")(?!\w)")
    # Ignore the abbreviation's period only when an author citation continues into a year.
    # This is a working copy for sentence matching; returned offsets use the original row.
    for sentence in re.split(r"(?<=[.!?])\s+", _AUTHOR_YEAR_PERIOD.sub("", local)):
        if primary.search(sentence):
            if _CONTRAST.search(sentence):
                return False
            if version.search(sentence) and _NAMING.search(sentence):
                explicit = True
    if explicit:
        return True
    matches = {(m.reference, m.name) for m in independent if left <= m.start and m.end <= right
               and m.reference != mention.reference and (m.reference, m.name) in neighbors}
    if len({name for _, name in matches}) >= 2 and len({ref for ref, _ in matches}) >= 2:
        return True
    return None  # No positive or contradictory evidence in this occurrence's context.


def _independent_anchor(alternatives, neighbors, primary_label):
    identities = {(row.doc_id, mention.name) for mention, row, _ in alternatives}
    if len(identities) != 1:
        return None
    _, label = identities.pop()
    family = primary_label.split()[0]
    if not _FAMILY.fullmatch(family) or not label or label.split()[0] != family:
        return None
    versions = set(_VERSION.findall(label)) - set(_VERSION.findall(primary_label))
    if not versions:
        return None
    supported = []
    for mention, row, evidence in alternatives:
        verdict = _corroborated(row.text, mention, evidence, neighbors, primary_label, versions)
        if verdict is False:
            return None
        if verdict is True:
            supported.append((mention, row))
    if not supported:
        return None
    # Overlapping chunks can repeat the same printed name. Use source order,
    # independent of retrieval rank, after checking every occurrence for a contradiction.
    mention, row = min(supported, key=lambda item: (item[1].chunk_number, item[1].chunk_id, item[0].start))
    return _anchor(mention, ((0, row),), ())


def comparison_reference_context(packet) -> ComparisonReferenceContext | None:
    """Locate one corroborated reference across the selected primary sections and an independent source."""
    if not isinstance(packet.query, str) or len(packet.query) > 2048 or not _COMPARISON.search(packet.query):
        return None
    selected = _current_selection(packet)
    pair = _paired_sections(selected, selected, packet.query)
    if pair is None:
        return None
    owners = {members[0].doc_id for _, members in pair}
    sides = []
    for text, members in pair:
        mentions = _mentions(text)
        positions = _section_coordinates(text, members)
        if mentions is None or not positions:
            return None
        refs = tuple(f"[doc:{r.doc_id} chunk:{r.chunk_id}]" for r in members)
        sides.append((_by_reference(mentions), positions, refs))
    first, second = (side[0] for side in sides)
    independent = {}
    for row in selected:
        if row.doc_id not in owners:
            mentions = _mentions(row.text)
            if mentions is None:
                return None
            for mention in mentions:
                independent.setdefault(mention.reference, []).append((mention, row, mentions))
    neighbors = {(key, name) for key, names in first.items() if len(names) == 1 and key in second
                 and names.keys() == second[key].keys() for name in names if name}
    result = None
    for key, name in neighbors:
        independent_anchor = _independent_anchor(independent.get(key, ()), neighbors, name)
        if independent_anchor is None:
            continue
        anchors = tuple(_anchor(groups[key][name], coords, refs) for groups, coords, refs in sides)
        if all(anchors):
            if result:
                return None
            result = ComparisonReferenceContext(
                (*anchors, independent_anchor),
                (pair[0][0], pair[1][0]),
            )
    return result
