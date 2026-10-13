"""Pure, explicit negative-only policy; signatures never confer service authority."""
from dataclasses import dataclass
import re

from apps.collections.services.vocabulary_contracts import MAX_BYTES, MAX_ENTRIES, digest, normalized


@dataclass(frozen=True, slots=True)
class ScopedVetoV1:
    scopes: tuple
    checksum: str

    def __post_init__(self):
        if type(self.scopes) is not tuple or len(self.scopes) > 128:
            raise ValueError("scoped vocabulary limit")
        previous, total = 0, 0
        for cid, pairs in self.scopes:
            if type(cid) is not int or cid <= previous or type(pairs) is not tuple:
                raise ValueError("invalid scoped vocabulary")
            previous = cid
            if tuple(sorted(set(pairs))) != pairs:
                raise ValueError("noncanonical scoped vocabulary")
            for pair in pairs:
                if (type(pair) is not tuple or len(pair) != 2 or pair[0] >= pair[1]
                    or any(type(t) is not str or t != normalized(t) or not t or len(t) > 512 for t in pair)):
                    raise ValueError("invalid scoped vocabulary pair")
            total += len(pairs)
        if total > MAX_ENTRIES * 128 or len(repr(self.scopes).encode()) > MAX_BYTES:
            raise ValueError("scoped vocabulary work limit")
        if type(self.checksum) is not str or not re.fullmatch("[0-9a-f]{64}", self.checksum):
            raise ValueError("invalid scoped vocabulary identity")


def bind_digest(base, vocabulary):
    if vocabulary is None:
        return base
    if type(base) is not str or not re.fullmatch("[0-9a-f]{64}", base):
        raise ValueError("invalid vocabulary base identity")
    if type(vocabulary) is not ScopedVetoV1:
        raise ValueError("exact scoped vocabulary required")
    vocabulary.__post_init__()
    return digest(dict(policy="scoped-vocabulary-v1", base=base, capsule=vocabulary.checksum))


def component_veto(vocabulary, endpoints):
    if vocabulary is None:
        return False
    by_label = {}
    for cid, label in endpoints:
        by_label.setdefault(normalized(label), set()).add(cid)
    return any(left in by_label and right in by_label
               and cid in (by_label[left] | by_label[right])
               for cid, pairs in vocabulary.scopes for left, right in pairs)


def endpoint_terms(vocabulary, endpoints):
    if vocabulary is None:
        return frozenset()
    terms = {term for _, pairs in vocabulary.scopes for pair in pairs for term in pair}
    return frozenset((cid, normalized(label)) for cid, label in endpoints if normalized(label) in terms)
