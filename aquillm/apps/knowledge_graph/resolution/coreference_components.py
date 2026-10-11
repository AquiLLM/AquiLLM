"""Pure component compatibility for document mention clustering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .coreference_types import PairDecision, _MentionView
from .vocabulary_policy import component_veto, endpoint_terms


_METHOD_PRECEDENCE = {
    "stable_identifier": 0,
    "defined_acronym": 1,
    "ontology_alias": 2,
    "normalized_name": 3,
    "singleton": 4,
}


_HARD_CANNOT_LINK_METHODS = frozenset(
    (
        "ambiguous_acronym",
        "component_conflict",
        "conflicting_stable_identifiers",
        "incompatible_entity_types",
        "lowercase_acronym",
        "pre_definition_acronym",
        "pronoun_only",
        "source_mismatch",
        "undefined_acronym",
        "version_mismatch",
        "vocabulary_veto",
    )
)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self._parents = list(range(size))
        self._sizes = [1] * size

    def find(self, item: int) -> int:
        parent = self._parents[item]
        while parent != self._parents[parent]:
            self._parents[parent] = self._parents[self._parents[parent]]
            parent = self._parents[parent]
        self._parents[item] = parent
        return parent

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self._sizes[left_root] < self._sizes[right_root] or (
            self._sizes[left_root] == self._sizes[right_root]
            and left_root > right_root
        ):
            left_root, right_root = right_root, left_root
        self._parents[right_root] = left_root
        self._sizes[left_root] += self._sizes[right_root]
        return left_root


def _constrain_component_merges(
    mentions: tuple[_MentionView, ...],
    decisions: tuple[PairDecision, ...],
    vocabulary=None, collection_id=None,
) -> tuple[tuple[PairDecision, ...], tuple[PairDecision, ...]]:
    """Accept candidate edges only when their complete components are compatible."""

    indexes = {mention.mention_id: index for index, mention in enumerate(mentions)}
    decision_indexes = {
        frozenset((decision.left_mention_id, decision.right_mention_id)): index
        for index, decision in enumerate(decisions)
    }
    hard_conflicts: dict[int, dict[int, PairDecision]] = defaultdict(dict)
    for decision in decisions:
        if decision.accepted or decision.method not in _HARD_CANNOT_LINK_METHODS:
            continue
        left_index = indexes[decision.left_mention_id]
        right_index = indexes[decision.right_mention_id]
        hard_conflicts[left_index][right_index] = decision
        hard_conflicts[right_index][left_index] = decision
    component_forbidden: dict[int, set[int]] = {
        index: set(hard_conflicts[index]) for index in range(len(mentions))
    }

    candidates = sorted(
        (decision for decision in decisions if decision.accepted),
        key=lambda decision: (
            _METHOD_PRECEDENCE[decision.method],
            min(
                indexes[decision.left_mention_id],
                indexes[decision.right_mention_id],
            ),
            max(
                indexes[decision.left_mention_id],
                indexes[decision.right_mention_id],
            ),
        ),
    )
    constrained = list(decisions)
    disjoint_set = _DisjointSet(len(mentions))
    component_members: dict[int, set[int]] = {
        index: {index} for index in range(len(mentions))
    }
    vocabulary_members = {
        i: endpoint_terms(vocabulary, ((collection_id, m.raw_text),))
        for i, m in enumerate(mentions)
    }
    merge_edges: list[PairDecision] = []
    for candidate in candidates:
        left_root = disjoint_set.find(indexes[candidate.left_mention_id])
        right_root = disjoint_set.find(indexes[candidate.right_mention_id])
        if left_root == right_root:
            continue
        left_members = component_members[left_root]
        right_members = component_members[right_root]
        blocked_right = component_forbidden[left_root].intersection(right_members)
        blockers = [
            decision
            for right_index in blocked_right
            for left_index, decision in hard_conflicts[right_index].items()
            if left_index in left_members
        ]
        combined_vocabulary = vocabulary_members[left_root] | vocabulary_members[right_root]
        if component_veto(vocabulary, combined_vocabulary):
            blockers.append(replace(candidate, method="vocabulary_veto"))
        if blockers:
            blocker = min(
                blockers,
                key=lambda decision: (
                    decision.method,
                    min(
                        indexes[decision.left_mention_id],
                        indexes[decision.right_mention_id],
                    ),
                    max(
                        indexes[decision.left_mention_id],
                        indexes[decision.right_mention_id],
                    ),
                ),
            )
            suppressed = PairDecision(
                left_mention_id=candidate.left_mention_id,
                right_mention_id=candidate.right_mention_id,
                accepted=False,
                method="component_conflict",
                confidence=0.0,
                explanation=(
                    "Rejected: component merge would violate "
                    f"{blocker.method} between {blocker.left_mention_id} and "
                    f"{blocker.right_mention_id}."
                ),
                outcome="rejected" if candidate.outcome is not None else None,
            )
            pair_key = frozenset(
                (candidate.left_mention_id, candidate.right_mention_id)
            )
            constrained[decision_indexes[pair_key]] = suppressed
            continue
        if len(left_members) < len(right_members):
            left_members, right_members = right_members, left_members
        left_members.update(right_members)
        left_forbidden = component_forbidden[left_root]
        right_forbidden = component_forbidden[right_root]
        if len(left_forbidden) < len(right_forbidden):
            left_forbidden, right_forbidden = right_forbidden, left_forbidden
        left_forbidden.update(right_forbidden)
        new_root = disjoint_set.union(left_root, right_root)
        component_members.pop(left_root)
        component_members.pop(right_root)
        component_forbidden.pop(left_root)
        component_forbidden.pop(right_root)
        component_members[new_root] = left_members
        component_forbidden[new_root] = left_forbidden
        vocabulary_members.pop(left_root)
        vocabulary_members.pop(right_root)
        vocabulary_members[new_root] = combined_vocabulary
        merge_edges.append(candidate)
    return tuple(constrained), tuple(merge_edges)
