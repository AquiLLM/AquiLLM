"""Pure component identity and compatibility for collection resolution."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from .normalization import normalize_entity_label
from .vocabulary_policy import component_veto, endpoint_terms

if TYPE_CHECKING:
    from .collection import DocumentEntityInput


class _DisjointSet:
    def __init__(
        self,
        values: Iterable[int],
        *,
        identifiers: Mapping[int, frozenset[str]] | None = None,
        versions: Mapping[int, frozenset[str]] | None = None,
        entity_types: Mapping[int, frozenset[str]] | None = None,
        vocabulary=None, vocabulary_endpoints=None,
    ) -> None:
        self.parent = {value: value for value in values}
        self.vocabulary = vocabulary
        self.vocabulary_members = {
            value: endpoint_terms(vocabulary, vocabulary_endpoints[value])
            if vocabulary_endpoints is not None else frozenset()
            for value in self.parent
        }
        self.identifiers = {
            value: frozenset() if identifiers is None else identifiers[value]
            for value in self.parent
        }
        self.versions = {
            value: frozenset() if versions is None else versions[value]
            for value in self.parent
        }
        self.entity_types = {
            value: frozenset() if entity_types is None else entity_types[value]
            for value in self.parent
        }

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, left: int, right: int) -> int:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return left_root
        root, child = sorted((left_root, right_root))
        self.parent[child] = root
        self.identifiers[root] = self.identifiers[root] | self.identifiers[child]
        self.versions[root] = self.versions[root] | self.versions[child]
        self.entity_types[root] = self.entity_types[root] | self.entity_types[child]
        self.vocabulary_members[root] |= self.vocabulary_members[child]
        return root

    def try_union(self, left: int, right: int) -> tuple[int | None, str | None]:
        """Merge only when every component-level cannot-link remains true."""

        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return left_root, None
        identifiers = self.identifiers[left_root] | self.identifiers[right_root]
        if len(identifiers) > 1:
            return None, "component_stable_identifier_conflict"
        versions = self.versions[left_root] | self.versions[right_root]
        if len(versions) > 1:
            return None, "component_version_signature_conflict"
        entity_types = self.entity_types[left_root] | self.entity_types[right_root]
        if len(entity_types) > 1:
            return None, "component_ontology_type_conflict"
        if component_veto(self.vocabulary,
            self.vocabulary_members[left_root] | self.vocabulary_members[right_root]):
            return None, "component_vocabulary_veto"
        return self.union(left_root, right_root), None

    def groups(self) -> dict[int, tuple[int, ...]]:
        result: dict[int, list[int]] = defaultdict(list)
        for value in sorted(self.parent):
            result[self.find(value)].append(value)
        return {root: tuple(values) for root, values in sorted(result.items())}


def _conflict_reason(
    left: DocumentEntityInput, right: DocumentEntityInput
) -> str | None:
    if left.version_signature != right.version_signature and (
        left.version_signature or right.version_signature
    ):
        return "version_signature_conflict"
    if left.identifier and right.identifier and left.identifier != right.identifier:
        return "conflicting_stable_identifiers"
    return None


def _identity_signature(
    entity: DocumentEntityInput, canonical_type: str | None
) -> tuple[str, str, str, str]:
    return (
        canonical_type or f"unknown:{entity.entity_type}",
        entity.identifier,
        normalize_entity_label(entity.normalized_label).key,
        entity.version_signature,
    )


def _component_anchors(
    members: Sequence[int], entities: Mapping[int, DocumentEntityInput]
) -> tuple[frozenset[str], frozenset[str]]:
    identifiers = frozenset(
        entities[item].identifier for item in members if entities[item].identifier
    )
    versions = frozenset(
        entities[item].version_signature
        for item in members
        if entities[item].version_signature
    )
    return identifiers, versions


def _component_pair_stable_key(
    left: int,
    right: int,
    component_keys: Mapping[int, str],
) -> tuple[str, str]:
    return tuple(sorted((component_keys[left], component_keys[right])))
