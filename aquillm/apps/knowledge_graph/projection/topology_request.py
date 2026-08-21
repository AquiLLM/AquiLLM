"""Strict decoding for the closed projected-topology transport."""

from __future__ import annotations

import json
from collections.abc import Mapping

from apps.knowledge_graph.retrieval.topology import contracts as c

_PARAMETER_KEYS = frozenset(
    {
        "bundle_checksum",
        "generation_keys_json",
        "document_keys_json",
        "membership_checksums_json",
        "seed_keys_json",
        "selected_generations_json",
        "authorized_documents_json",
        "authorization_context_signature",
        "seeds_json",
        "seed_checksum",
        "caps_json",
    }
)


def _json(parameters: Mapping[str, c.TopologyScalar], name: str):
    raw = parameters[name]
    if type(raw) is not str or len(raw.encode("utf-8")) > 2_000_000:
        raise ValueError(f"{name} must be bounded exact JSON")
    value = json.loads(raw)
    if (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        != raw
    ):
        raise ValueError(f"{name} must be canonical JSON")
    return value


def decode_topology_request(parameters: Mapping[str, c.TopologyScalar]):
    if not isinstance(parameters, Mapping) or set(parameters) != _PARAMETER_KEYS:
        raise ValueError("topology parameters do not match the closed transport")
    selected = tuple(
        c.SelectedCollectionGenerationV1(**row)
        for row in _json(parameters, "selected_generations_json")
    )
    documents = tuple(
        c.AuthorizedProjectedDocumentV1(**row)
        for row in _json(parameters, "authorized_documents_json")
    )
    signature = parameters["authorization_context_signature"]
    checksum = parameters["bundle_checksum"]
    if type(signature) is not str or type(checksum) is not str:
        raise TypeError("topology signatures must be exact strings")
    ready = c.ReadyGenerationBundleV1(selected, documents, signature, checksum)
    seeds = tuple(
        c.ProjectedSeedV1(row["identity_key"], float.fromhex(row["mass"]))
        for row in _json(parameters, "seeds_json")
    )
    seed_checksum = parameters["seed_checksum"]
    if type(seed_checksum) is not str:
        raise TypeError("seed_checksum must be an exact string")
    caps_value = _json(parameters, "caps_json")
    caps = c.TopologyCapsV1(
        c.HybridBranchKind(caps_value.pop("branch_kind")), **caps_value
    )
    c.validate_projected_seed_sequence(
        seeds, maximum=caps.max_seeds, expected_checksum=seed_checksum
    )
    if _json(parameters, "generation_keys_json") != [
        row.generation_key for row in selected
    ]:
        raise ValueError("generation key transport disagrees with ready bundle")
    if _json(parameters, "document_keys_json") != [
        row.document_key for row in documents
    ]:
        raise ValueError("document key transport disagrees with ready bundle")
    if _json(parameters, "membership_checksums_json") != [
        row.membership_checksum for row in selected
    ]:
        raise ValueError("membership transport disagrees with ready bundle")
    if _json(parameters, "seed_keys_json") != [row.identity_key for row in seeds]:
        raise ValueError("seed key transport disagrees with seed bundle")
    return ready, seeds, caps


__all__ = ["decode_topology_request"]
