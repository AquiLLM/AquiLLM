"""Bounded vocabulary values; checksums identify content, never grant access."""
import json
import unicodedata
from dataclasses import asdict, dataclass, field
from hashlib import sha256

MAX_ENTRIES = 256
MAX_BYTES = 131072


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def text(value, limit):
    if type(value) is not str or not value or value != value.strip() or len(value) > limit or "\x00" in value:
        raise ValueError("invalid vocabulary text")
    return value


def validate_entries(entries):
    if type(entries) not in (list, tuple) or len(entries) > MAX_ENTRIES:
        raise ValueError("vocabulary entry limit")
    if len(json.dumps(entries, ensure_ascii=True).encode()) > MAX_BYTES:
        raise ValueError("vocabulary byte limit")
    rows, keys, terms, vetoes = [], set(), {}, set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"entry_key", "kind", "term", "expansions", "contributor"}:
            raise ValueError("invalid vocabulary entry")
        row = dict(entry)
        for name, cap in (("entry_key", 128), ("term", 256), ("contributor", 256)):
            text(row[name], cap)
        if row["entry_key"] in keys or row["kind"] not in {"alias", "gazetteer", "never_merge"}:
            raise ValueError("conflicting vocabulary entry")
        keys.add(row["entry_key"])
        expansions = row["expansions"]
        if type(expansions) not in (list, tuple) or not 1 <= len(expansions) <= 6:
            raise ValueError("invalid expansions")
        row["expansions"] = [text(v, 512) for v in expansions]
        if len(set(map(normalized, expansions))) != len(expansions):
            raise ValueError("duplicate expansions")
        term = normalized(row["term"])
        if row["kind"] == "never_merge":
            if len(expansions) != 1 or term == normalized(expansions[0]):
                raise ValueError("invalid veto")
            vetoes.add(tuple(sorted((term, normalized(expansions[0])))))
        else:
            values = tuple(map(normalized, expansions))
            if term in terms and terms[term] != values:
                raise ValueError("conflicting aliases")
            terms[term] = values
        rows.append(row)
    if any(tuple(sorted((term, phrase))) in vetoes for term, phrases in terms.items() for phrase in phrases):
        raise ValueError("alias conflicts with never_merge")
    return tuple(sorted(rows, key=lambda r: r["entry_key"]))


@dataclass(frozen=True, slots=True)
class VocabularyAliasV1:
    term: str
    expansions: tuple[str, ...]
    contributor: str
    entry_key: str


@dataclass(frozen=True, slots=True)
class VocabularyBindingV1:
    authorization_signature: str
    selected_collection_ids: tuple[int, ...]
    # Exact attachment rows and root-to-selected paths, serialized as tuples.
    pins: tuple
    paths: tuple


@dataclass(frozen=True, slots=True)
class VocabularySnapshotV1:
    version_keys: tuple[str, ...]
    aliases: tuple[VocabularyAliasV1, ...]
    never_merge: tuple[tuple[str, str], ...]
    checksum: str
    binding: VocabularyBindingV1 | None = field(default=None, repr=False)


def snapshot_digest(snapshot):
    return digest({k: v for k, v in asdict(snapshot).items() if k != "checksum"})


def version_payload(vocabulary_id, owner_id, revision, actor_ref, entries):
    return dict(schema="vocabulary-version-v1", vocabulary_id=vocabulary_id,
                owner_collection_id=owner_id, draft_revision=revision,
                published_by_ref=actor_ref, entries=list(entries))
