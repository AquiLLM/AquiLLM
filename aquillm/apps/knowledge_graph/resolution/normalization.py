"""Conservative entity-label and stable-identifier normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

_MAX_LABEL_CHARACTERS = 4_096
_MAX_IDENTIFIER_CHARACTERS = 2_048
_MAX_NORMALIZED_LABEL_CHARACTERS = 512
_QUOTES = frozenset("'\"`\u00b4\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f")
_SEMANTIC_SYMBOLS = frozenset("+/#:")
_VERSION_TOKEN = re.compile(
    r"(?:v[0-9]+(?:\.[0-9]+)*(?:[a-z][a-z0-9]*)?|"
    r"[0-9]+(?:\.[0-9]+)*(?:[bm])?|"
    r"rc[0-9]+|alpha[0-9]*|beta[0-9]*)"
)
_RELEASE_QUALIFIERS = frozenset(("base", "chat", "instruct"))
_DOI = re.compile(r"10\.[0-9]{4,9}/[A-Za-z0-9][A-Za-z0-9._;()/:+\-]*")
_ARXIV_NEW = re.compile(
    r"[0-9]{2}(?:0[1-9]|1[0-2])\.[0-9]{4,5}(?:v[1-9][0-9]*)?",
    re.IGNORECASE,
)
_ARXIV_OLD = re.compile(
    r"[a-z][a-z0-9.\-]*(?:\.[a-z]{2})?/[0-9]{7}(?:v[1-9][0-9]*)?",
    re.IGNORECASE,
)
_ORCID = re.compile(r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]", re.I)
_REPOSITORY_COMPONENT = re.compile(r"[A-Za-z0-9_.\-]+")
_REPOSITORY_HOSTS = frozenset(("github.com", "gitlab.com", "bitbucket.org"))


def _require_nonempty_bounded_text(
    value: object,
    *,
    label: str,
    maximum: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds the {maximum}-character limit")
    return value.strip()


@dataclass(frozen=True, slots=True)
class NormalizedEntityLabel:
    """A case-preserving label and its conservative comparison key."""

    display_label: str
    key: str
    base_key: str
    version_signature: str | None

    def __post_init__(self) -> None:
        if not self.display_label or not self.key or not self.base_key:
            raise ValueError("normalized labels must be nonempty")
        if len(self.key) > _MAX_NORMALIZED_LABEL_CHARACTERS:
            raise ValueError("normalized label exceeds persistence limit")

    @property
    def version_suffix(self) -> str:
        if self.version_signature is None:
            return ""
        explicit = re.search(r"\bv[0-9][a-z0-9.]*$", self.version_signature)
        return explicit.group(0) if explicit else self.version_signature


@dataclass(frozen=True, slots=True)
class StableIdentifier:
    """A typed identifier whose canonical value is safe for exact matching."""

    scheme: str
    value: str

    def __post_init__(self) -> None:
        if self.scheme not in {"doi", "arxiv", "orcid", "repository"}:
            raise ValueError("unsupported stable identifier scheme")
        if not self.value or self.value != self.value.strip():
            raise ValueError("stable identifier value must be nonempty and trimmed")
        if len(self.canonical) > 255:
            raise ValueError("stable identifier exceeds persistence limit")

    @property
    def canonical(self) -> str:
        return f"{self.scheme}:{self.value}"


def _comparison_key(display_label: str) -> str:
    characters: list[str] = []
    for index, character in enumerate(display_label):
        if character in _QUOTES:
            continue
        category = unicodedata.category(character)
        if character == ".":
            previous_is_alnum = index > 0 and display_label[index - 1].isalnum()
            next_is_alnum = (
                index + 1 < len(display_label) and display_label[index + 1].isalnum()
            )
            characters.append("." if previous_is_alnum and next_is_alnum else " ")
        elif character in _SEMANTIC_SYMBOLS:
            characters.append(character)
        elif category.startswith("P"):
            characters.append(" ")
        else:
            characters.append(character)
    return " ".join("".join(characters).casefold().split())


def normalize_entity_label(value: object) -> NormalizedEntityLabel:
    """Return a deterministic label without erasing version distinctions."""

    raw = _require_nonempty_bounded_text(
        value,
        label="entity label",
        maximum=_MAX_LABEL_CHARACTERS,
    )
    display_label = " ".join(unicodedata.normalize("NFC", raw).split())
    key = _comparison_key(unicodedata.normalize("NFKC", display_label))
    if not key:
        raise ValueError("entity label must contain meaningful characters")
    base_key, version_signature = _split_version_signature(key)
    return NormalizedEntityLabel(
        display_label=display_label,
        key=key,
        base_key=base_key,
        version_signature=version_signature,
    )


def _split_version_signature(key: str) -> tuple[str, str | None]:
    tokens = key.split()
    if len(tokens) < 2:
        return key, None
    for index in range(1, len(tokens)):
        suffix = tokens[index:]
        first_is_version = bool(
            _VERSION_TOKEN.fullmatch(suffix[0])
            or suffix[0] in _RELEASE_QUALIFIERS
            or (
                suffix[0] == "version"
                and len(suffix) > 1
                and _VERSION_TOKEN.fullmatch(suffix[1])
            )
        )
        if not first_is_version:
            continue
        if all(
            token == "version"
            or _VERSION_TOKEN.fullmatch(token)
            or token in _RELEASE_QUALIFIERS
            for token in suffix
        ):
            return " ".join(tokens[:index]), " ".join(suffix)
    return key, None


def _url_parts(raw: str):
    try:
        parts = urlsplit(raw)
        _ = parts.port
    except ValueError:
        return None
    if parts.username or parts.password or parts.query or parts.fragment:
        return None
    return parts


def _doi_identifier(raw: str) -> StableIdentifier | None:
    candidate = raw
    parts = _url_parts(raw) if "://" in raw else None
    if parts is not None:
        if parts.scheme.casefold() not in {"http", "https"}:
            return None
        if (parts.hostname or "").casefold() not in {"doi.org", "dx.doi.org"}:
            return None
        candidate = unquote(parts.path).lstrip("/")
    elif re.match(r"(?i)^doi\s*:", candidate):
        candidate = re.sub(r"(?i)^doi\s*:\s*", "", candidate, count=1)
    if not _DOI.fullmatch(candidate):
        return None
    return StableIdentifier("doi", candidate.casefold())


def _arxiv_identifier(raw: str) -> StableIdentifier | None:
    candidate = raw
    parts = _url_parts(raw) if "://" in raw else None
    if parts is not None:
        if parts.scheme.casefold() not in {"http", "https"}:
            return None
        if (parts.hostname or "").casefold() not in {"arxiv.org", "www.arxiv.org"}:
            return None
        path = unquote(parts.path).strip("/")
        if path.startswith("abs/"):
            candidate = path.removeprefix("abs/")
        elif path.startswith("pdf/") and path.endswith(".pdf"):
            candidate = path.removeprefix("pdf/").removesuffix(".pdf")
        else:
            return None
    elif re.match(r"(?i)^arxiv\s*:", candidate):
        candidate = re.sub(r"(?i)^arxiv\s*:\s*", "", candidate, count=1)
    if not (_ARXIV_NEW.fullmatch(candidate) or _ARXIV_OLD.fullmatch(candidate)):
        return None
    return StableIdentifier("arxiv", candidate.casefold())


def _valid_orcid(candidate: str) -> bool:
    if not _ORCID.fullmatch(candidate):
        return False
    compact = candidate.replace("-", "").upper()
    total = 0
    for character in compact[:15]:
        total = (total + int(character)) * 2
    remainder = (12 - total % 11) % 11
    expected = "X" if remainder == 10 else str(remainder)
    return compact[-1] == expected


def _orcid_identifier(raw: str) -> StableIdentifier | None:
    candidate = raw
    parts = _url_parts(raw) if "://" in raw else None
    if parts is not None:
        if parts.scheme.casefold() not in {"http", "https"}:
            return None
        if (parts.hostname or "").casefold() not in {"orcid.org", "www.orcid.org"}:
            return None
        candidate = unquote(parts.path).strip("/")
    elif re.match(r"(?i)^orcid\s*:", candidate):
        candidate = re.sub(r"(?i)^orcid\s*:\s*", "", candidate, count=1)
    candidate = candidate.upper()
    if not _valid_orcid(candidate):
        return None
    return StableIdentifier("orcid", candidate)


def _repository_identifier(raw: str) -> StableIdentifier | None:
    candidate = raw
    scp = re.fullmatch(
        r"git@(?P<host>github\.com|gitlab\.com|bitbucket\.org):(?P<path>[^\s]+)",
        candidate,
        re.IGNORECASE,
    )
    if scp is not None:
        host = scp.group("host").casefold()
        path = scp.group("path")
    elif candidate.casefold().startswith("ssh://"):
        try:
            parts = urlsplit(candidate)
            port = parts.port
        except ValueError:
            return None
        if (
            parts.scheme.casefold() != "ssh"
            or parts.username != "git"
            or parts.password
            or port is not None
            or parts.query
            or parts.fragment
        ):
            return None
        host = (parts.hostname or "").casefold()
        path = unquote(parts.path).strip("/")
    elif "://" not in raw:
        shorthand = re.fullmatch(
            r"(?i)(github|gitlab|bitbucket)\s*:\s*([^\s]+)", candidate
        )
        if shorthand is None:
            return None
        host = f"{shorthand.group(1).casefold()}.com"
        if host == "bitbucket.com":
            host = "bitbucket.org"
        path = shorthand.group(2)
    else:
        parts = _url_parts(raw)
        if parts is None or parts.scheme.casefold() != "https":
            return None
        host = (parts.hostname or "").casefold()
        path = unquote(parts.path).strip("/")
    if host not in _REPOSITORY_HOSTS:
        return None
    if path.endswith(".git"):
        path = path[:-4]
    components = path.split("/")
    expected_length = 2 if host != "gitlab.com" else len(components)
    if (
        len(components) < 2
        or len(components) != expected_length
        or "-" in components
        or any(not _REPOSITORY_COMPONENT.fullmatch(item) for item in components)
    ):
        return None
    value = f"{host}/{'/'.join(components)}".casefold()
    return StableIdentifier("repository", value)


def parse_stable_identifier(value: object) -> StableIdentifier | None:
    """Parse a complete DOI, arXiv, ORCID, or repository identifier."""

    if not isinstance(value, str) or not value.strip():
        return None
    if len(value) > _MAX_IDENTIFIER_CHARACTERS:
        return None
    raw = unicodedata.normalize("NFKC", value).strip()
    for parser in (
        _doi_identifier,
        _arxiv_identifier,
        _orcid_identifier,
        _repository_identifier,
    ):
        try:
            identifier = parser(raw)
        except ValueError:
            return None
        if identifier is not None:
            return identifier
    return None


__all__ = [
    "NormalizedEntityLabel",
    "StableIdentifier",
    "normalize_entity_label",
    "parse_stable_identifier",
]
