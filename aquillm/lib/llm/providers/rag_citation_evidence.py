"""Bounded literal evidence checks. An identifier match does not establish a claim."""

import re
from dataclasses import dataclass
from itertools import zip_longest

from .rag_citation_extracts import _CITATION_RE, _chunk_citation_from_row
from .rag_citation_prose import numeric_claim_sentences

_REF_GROUP = re.compile(rf"{_CITATION_RE.pattern}(?:[ \t]*{_CITATION_RE.pattern})*")
_WORD = re.compile(r"(?<![\w.-])[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*(?![\w-])")
_LIST = re.compile(r"^(?:[-+*]|\d+[.)])\s+")
_NAME_VERSION = re.compile(
    r"\b([A-Z][A-Za-z0-9-]{1,31})\s+"
    r"([A-Z]{1,6}\d+[a-zA-Z0-9]*|[vV]?\d+(?:\.\d+){1,3})\b"
)
_QUOTED = re.compile(r"""["'“‘`]([^"'“”‘’`\n]{1,64})["'”’`]""")


def _identifiers(text, *, source=False):
    # Case and punctuation matter. Plain quantities, units, and arithmetic
    # results need not appear literally in a source describing the inputs.
    quoted = set(_QUOTED.findall(text)) if not source else set()
    words = {
        word
        for word in _WORD.findall(text)
        if (source or "-" in word or word in quoted)
        and len(word) <= 64
        and any(c.isdigit() for c in word)
        and (any(c.isupper() for c in word) or re.match(r"v\d", word))
        and not re.fullmatch(r"\d+(?:\.\d+)?(?:[A-Z]{1,3}|E[+-]?\d+)", word)
    }
    for match in _NAME_VERSION.finditer(text):
        if source or match[1].isupper() or match[0] in quoted:
            words.add(" ".join(match.groups()))
    return frozenset(words)


@dataclass(frozen=True)
class CitedSource:
    citation: str
    document: str
    text: str
    identifiers: frozenset[str]


def bind_citation_sources(rows, allowed):
    # A prefix of a passage cannot establish that an identifier is absent.
    if len(rows) > 64:
        return ()
    sources, remaining = [], 256_000
    for row in rows:
        citation = _chunk_citation_from_row(row)
        text = row.get("text", row.get("x"))
        if citation not in allowed or not isinstance(text, str) or not text:
            continue
        remaining -= len(text)
        if remaining < 0:
            return ()
        sources.append(
            CitedSource(
                citation,
                citation.rsplit(" chunk:", 1)[0],
                text,
                _identifiers(text, source=True),
            )
        )
    return tuple(sources)


def _prose_lines(text, *, join_wrapped=False):
    fence = None
    pending = []
    for line in text.splitlines():
        stripped = line.strip()
        marker = stripped.startswith(("```", "~~~"))
        excluded = not stripped or stripped.startswith(("#", "![")) or marker
        if pending and (excluded or _LIST.match(stripped)):
            yield " ".join(pending)
            pending.clear()
        if marker:
            if fence is None:
                fence = stripped[:3]
            elif stripped.startswith(fence):
                fence = None
            continue
        if fence is None and not excluded:
            if join_wrapped:
                pending.append(stripped)
            else:
                yield stripped
    if pending:
        yield " ".join(pending)


def _unique_named_source(identifier, sources):
    # Across papers, standalone labels and ambiguous names cannot establish
    # even the location of the referent. A unique exact name/version can point
    # the repair at its passage; it never proves a relation to another paper.
    if not _NAME_VERSION.fullmatch(identifier):
        return None
    matches = [s.citation for s in sources if identifier in s.identifiers]
    return matches[0] if len(matches) == 1 else None


def misplaced_identifiers(text, sources):
    """Locate missing literal IDs in siblings or a unique named source."""
    if not sources or len(text) > 64_000:
        return {}
    by_ref = {source.citation for source in sources}
    findings = {}
    for line in _prose_lines(text, join_wrapped=True):
        start = 0
        for group in _REF_GROUP.finditer(line):
            refs = set(_CITATION_RE.findall(group[0]))
            prose = line[start : group.start()].strip()
            start = group.end()
            if not prose or not refs <= by_ref:
                continue
            claim = numeric_claim_sentences(prose)[-1]
            ids = _identifiers(claim)
            if not ids:
                continue
            cited = [source for source in sources if source.citation in refs]
            docs = {source.document for source in cited}
            missing = ids.difference(*(source.identifiers for source in cited))
            for identifier in sorted(missing):
                candidates = tuple(
                    dict.fromkeys(
                        source.citation
                        for source in sources
                        if source.document in docs and identifier in source.identifiers
                    )
                )
                if not candidates and (
                    other := _unique_named_source(identifier, sources)
                ):
                    candidates = (other,)
                if candidates:
                    findings[identifier] = candidates
                    if len(findings) == 4:
                        return findings
    return findings


def exact_draft_claims(text, sources):
    """Keep independent, cited draft lines equal to a complete delivered passage."""
    if not sources or len(text) > 64_000:
        return ()
    kept = []
    for line in _prose_lines(text):
        refs = _CITATION_RE.findall(line)
        if len(refs) != 1:
            continue
        before, after = line.split(refs[0])
        if after.strip() not in {"", ".", "!", "?"}:
            continue
        claim = _LIST.sub("", before.rstrip(), count=1) + after.strip()
        # Plain, independently stated sentences only. Inline markup, quotes,
        # headings and nested context require their original presentation.
        if not claim[:1].isupper() or not claim.endswith((".", "!", "?")):
            continue
        if (
            any(
                source.citation == refs[0] and claim == source.text.strip()
                for source in sources
            )
            and line not in kept
        ):
            kept.append(line)
        if len(kept) == 5:
            break
    return tuple(kept)


def draft_excerpt_order(text, sources):
    """Focus literal excerpts on cited documents, including their checked siblings."""
    if not sources or len(text) > 64_000:
        return ()
    # Ignore bare reference inventories. They say nothing about the original
    # claim's source. A literal on an uncited line can order attributed
    # excerpts, but can never attach a reference to that line.
    cited, uncited_ids = [], set()
    for line in _prose_lines(text):
        refs = _CITATION_RE.findall(line)
        prose = _LIST.sub("", _CITATION_RE.sub("", line)).strip()
        if not any(c.isalpha() for c in prose):
            continue
        if refs:
            cited.extend(refs)
        else:
            uncited_ids.update(_identifiers(prose))
    misplaced = misplaced_identifiers(text, sources)
    refs = dict.fromkeys(
        [*(ref for found in misplaced.values() for ref in found), *cited]
    )
    by_ref = {source.citation: source for source in sources}
    if not any(ref in by_ref for ref in refs):
        for identifier in sorted(uncited_ids):
            if ref := _unique_named_source(identifier, sources):
                refs[ref] = None
    groups = {}
    for ref in refs:
        if ref in by_ref:
            source = by_ref[ref]
            groups.setdefault(source.document, []).append(source)
    for group in groups.values():
        group.sort(key=lambda item: len(item.identifiers & uncited_ids), reverse=True)
    return tuple(
        source.citation
        for layer in zip_longest(*groups.values())
        for source in layer
        if source is not None
    )
