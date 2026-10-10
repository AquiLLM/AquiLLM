"""Local question-aware selection over bounded, privately authorized evidence."""

import re
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256

from ._graph_v2_validation import checksum, count, score
from .graph_handoff import GraphCandidateRefV2, GraphCoverageV2, GraphHandoffV2
from .graph_query_contracts import GraphEvidenceRefV2, GraphGroupRelationRequestV2
from .relation_request import _normalized_query


@dataclass(frozen=True, slots=True)
class GraphEvidenceSelectionInputV2:
    ref: GraphEvidenceRefV2
    text: str
    text_sha256: str
    document_key: str

    def __post_init__(self):
        if type(self.ref) is not GraphEvidenceRefV2 or type(self.text) is not str:
            raise TypeError("selection input requires exact private evidence and text")
        checksum(self.text_sha256, "text_sha256")
        checksum(self.document_key, "document_key")
        if (
            len(self.text) > 250000
            or sha256(self.text.encode()).hexdigest() != self.text_sha256
        ):
            raise ValueError("selection text exceeds allowance or checksum differs")


@dataclass(frozen=True, slots=True)
class GraphEvidenceSelectionEnvelopeV2:
    inputs: tuple[GraphEvidenceSelectionInputV2, ...]
    coverage: GraphCoverageV2
    algorithm_signature: str
    ready_bundle_checksums: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if type(self.inputs) is not tuple or any(
            type(r) is not GraphEvidenceSelectionInputV2 for r in self.inputs
        ):
            raise TypeError("selection inputs must be exact")
        if len(self.inputs) > 5000 or sum(len(r.text) for r in self.inputs) > 250000:
            raise ValueError("selection inputs exceed bounded work")
        if sum(len(r.ref.associations) for r in self.inputs) > 3000:
            raise ValueError("selection associations exceed global evidence allowance")
        if type(self.coverage) is not GraphCoverageV2:
            raise TypeError("coverage must be exact")
        checksum(self.algorithm_signature, "algorithm_signature")
        rows = self.ready_bundle_checksums
        if type(rows) is not tuple or len(rows) > 10000:
            raise TypeError("ready bindings must be an exact bounded tuple")
        for row in rows:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("ready binding must be an exact pair")
            for value in row:
                checksum(value, "ready binding")
        if rows != tuple(sorted(rows)) or len(dict(rows)) != len(rows):
            raise ValueError("ready bindings must be unique and canonical")
        if any(r.ref.authority_group_checksum not in dict(rows) for r in self.inputs):
            raise ValueError("evidence lacks captured ready binding")


def partial_coverage(coverage, reason="evidence_budget"):
    return replace(
        coverage,
        status="partial" if coverage.ready_documents else "unavailable",
        reasons=tuple(sorted(set(coverage.reasons) | {reason})),
    )


_STOP = frozenset(
    (
        "a an the is are was were does do which what who how show list find evidence "
        "and or of for to in use uses"
    ).split()
)


def _tokens(text):
    return {
        t.removesuffix("s")
        for t in re.findall(r"\w+", text.casefold())
        if t not in _STOP
    }


def select_graph_evidence_v2(
    *,
    query: str,
    requests: tuple[GraphGroupRelationRequestV2, ...],
    envelope: GraphEvidenceSelectionEnvelopeV2,
    identity_scores: tuple[tuple[str, float], ...],
    max_candidates: int,
    max_mentions_per_identity: int,
    max_evidence_per_transition: int,
) -> GraphHandoffV2:
    if type(envelope) is not GraphEvidenceSelectionEnvelopeV2:
        raise TypeError("selection envelope must be exact")
    for value, name, limit in (
        (max_candidates, "candidates", 20),
        (max_mentions_per_identity, "mentions", 2),
        (max_evidence_per_transition, "transition evidence", 3),
    ):
        count(value, name, limit)
        if value < 1:
            raise ValueError("selection allowances must be positive")
    if type(requests) is not tuple or any(
        type(r) is not GraphGroupRelationRequestV2 for r in requests
    ):
        raise TypeError("requests must be exact group-bound values")
    bound = {r.authority_group_checksum: r.request for r in requests}
    ready = dict(envelope.ready_bundle_checksums)
    if len(bound) != len(requests) or set(bound) != set(ready):
        raise ValueError("exactly one request per captured group is required")
    query_digest = sha256(_normalized_query(query).encode()).hexdigest()
    if any(r.query_digest != query_digest for r in bound.values()):
        raise ValueError("relation request belongs to another question")
    if type(identity_scores) is not tuple:
        raise TypeError("identity scores must be exact")
    masses = {}
    for identity, mass in identity_scores:
        checksum(identity, "identity")
        score(mass, "identity mass")
        masses[identity] = max(mass, masses.get(identity, 0.0))
    tokens, unique = _tokens(query), {}
    for row in envelope.inputs:
        ref = row.ref
        weights = dict(bound[ref.authority_group_checksum].predicate_weights)
        if not set(ref.relation_types) <= set(weights):
            raise ValueError("evidence predicate is outside its group's ontology")
        previous = unique.get(ref.chunk_key)
        if previous is not None and previous != row:
            raise ValueError("same chunk carries conflicting source evidence")
        unique[ref.chunk_key] = row
    ranked = []
    for row in unique.values():
        ref = row.ref
        mass = max((masses.get(i, 0.0) for i in ref.identity_keys), default=0.0)
        if not mass:
            continue
        lexical = len(tokens & _tokens(row.text)) / max(1, len(tokens))
        weights = dict(bound[ref.authority_group_checksum].predicate_weights)
        predicate = max((weights[t] for t in ref.relation_types), default=0.0)
        relevance = (2 * lexical + predicate) * (0.75 + 0.25 * ref.confidence)
        value = float(relevance + 0.01 * mass + 0.001 * ref.confidence)
        ranked.append((value, row))
    ranked.sort(key=lambda pair: (-pair[0], pair[1].ref.chunk_key))
    # Query cues request comparison; only same-scope extracted classes establish
    # disagreement. Neither lexical source text nor unrelated claims do so.
    preferred = []
    normalized = _normalized_query(query)
    conflict = bool(
        re.search(r"\b(?:conflict\w*|oppos\w*|disagree\w*)\b", normalized)
    ) or (
        bool(re.search(r"\bsupport\w*\b", normalized))
        and bool(re.search(r"\bcontradict\w*\b", normalized))
    )
    if conflict:
        claims = {}
        for pair in ranked:
            ref = pair[1].ref
            for association in ref.associations:
                if association.relation_type in ("supports", "contradicts"):
                    claim = (
                        ref.authority_group_checksum,
                        association.source_identity_key,
                        association.target_identity_key,
                        association.direction,
                    )
                    claims.setdefault(claim, {}).setdefault(
                        association.relation_type, pair
                    )
        for classes in claims.values():
            if {"supports", "contradicts"} <= set(classes):
                preferred.extend(classes[r] for r in ("supports", "contradicts"))
    mentions, transitions, documents = Counter(), Counter(), Counter()
    retained, clipped = {}, False
    for value, row in preferred + ranked:
        ref = row.ref
        if ref.chunk_key in retained:
            continue
        identities = {
            (ref.authority_group_checksum, a.source_identity_key)
            for a in ref.associations
            if a.kind == "mention"
        }
        transition_keys = {
            (
                ref.authority_group_checksum,
                a.source_identity_key,
                a.relation_type,
                a.target_identity_key,
                a.direction,
            )
            for a in ref.associations
            if a.kind == "relation"
        }
        exceeds = (
            len(retained) >= max_candidates
            or documents[row.document_key] >= 3
            or (any(mentions[i] >= max_mentions_per_identity for i in identities))
            or (
                any(
                    transitions[t] >= max_evidence_per_transition
                    for t in transition_keys
                )
            )
        )
        if exceeds:
            clipped = True
            continue
        retained[ref.chunk_key] = GraphCandidateRefV2(
            ref.chunk_key,
            ref.graph_generation_key,
            ref.representation_generation_key,
            ref.authority_group_checksum,
            ready[ref.authority_group_checksum],
            ref.evidence_provenance_keys,
            ref.relation_types,
            value,
        )
        documents[row.document_key] += 1
        transitions.update(transition_keys)
        mentions.update(identities)
    coverage = partial_coverage(envelope.coverage) if clipped else envelope.coverage
    return GraphHandoffV2(
        tuple(sorted(retained.values(), key=lambda r: (-r.score, r.chunk_key))),
        coverage,
        envelope.algorithm_signature,
    )
