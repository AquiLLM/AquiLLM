"""Closed private data witnesses; never retain a successful database proof."""

from copy import deepcopy
from dataclasses import fields
from uuid import UUID

from apps.documents.services.representation_contracts import (
    DocumentRefV1,
    RepresentationSelectionV1,
    RepresentationSnapshotV1,
)

from .direct_seed_types import DirectSeedScopeV1
from .graph_handoff import GraphCoverageV2
from .ready_scope import ReadyProjectionAuthorityV1, SelectedReadyScopeV1
from .ready_scope_v2 import (
    _NATIVE_GRAPH_GROUP_VALIDATOR,
    _NATIVE_GRAPH_SELECTION_VALIDATOR,
    GraphAuthorityGroupV2,
    SelectedGraphScopeV2,
)
from .topology.contracts import (
    _NATIVE_READY_BUNDLE_VALIDATOR,
    AuthorizedProjectedDocumentV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
)

# Fixed descriptors: homogeneous tuple, fixed pair, optional cell, UUID set.
_SCHEMA = {
    SelectedGraphScopeV2: (
        ("groups", (tuple, GraphAuthorityGroupV2)),
        ("authorization_context_signature", str),
        ("representation_snapshot", RepresentationSnapshotV1),
        ("coverage", GraphCoverageV2),
        ("requested_document_ids", (None, (frozenset, UUID))),
    ),
    GraphAuthorityGroupV2: (("scope", SelectedReadyScopeV1),
        ("ontology_checksum group_checksum", str)),
    SelectedReadyScopeV1: (("ready", ReadyGenerationBundleV1),
        ("projections", (tuple, ReadyProjectionAuthorityV1)),
        ("selected_document_ids", (tuple, UUID)),
        ("generation_keys_by_projection", (tuple, (UUID, str)))),
    ReadyProjectionAuthorityV1: (
        ("projection_id generation_id", UUID),
        ("collection_id artifact_id membership_epoch", int),
        ("schema_version projection_version identifier_key_version "
         "membership_checksum graph_checksum private_mapping_checksum "
         "resolver_version resolution_config_checksum ontology_version "
         "ontology_checksum embedding_model_signature", str),
        ("documents", (tuple, (UUID, int))),
    ),
    ReadyGenerationBundleV1: (
        ("selected_generations", (tuple, SelectedCollectionGenerationV1)),
        ("authorized_documents", (tuple, AuthorizedProjectedDocumentV1)),
        ("authorization_context_signature bundle_checksum", str)),
    SelectedCollectionGenerationV1: (
        ("collection_key generation_key active_artifact_key projection_key "
         "graph_checksum schema_version projection_version identifier_key_version "
         "membership_checksum resolver_version resolution_config_checksum "
         "ontology_checksum embedding_model_signature", str),
        ("membership_epoch", int)),
    AuthorizedProjectedDocumentV1: (
        ("document_key collection_key generation_key", str),),
    RepresentationSnapshotV1: (
        ("selections", (tuple, RepresentationSelectionV1)),),
    RepresentationSelectionV1: (("document", DocumentRefV1),
        ("generation_key", (None, UUID)), ("canonical_sha256", str),
        ("source_revision_sha256", (None, str))),
    DocumentRefV1: (("model_label source_hash", str),
        ("pkid collection_id", int), ("id", UUID)),
    GraphCoverageV2: (("selected_documents ready_documents", int),
        ("status", str), ("reasons", (tuple, str)), ("exhaustive", bool)),
    DirectSeedScopeV1: (
        ("ready_bundle_checksum ontology_checksum resolver_version "
         "expected_embedding_signature", str),
        ("selected_collection_ids selected_artifact_ids "
         "selected_document_artifact_ids", (tuple, int)),
        ("selected_document_ids", (tuple, UUID)),
        ("generation_keys_by_artifact", (tuple, (int, str))),
        ("generation_ids_by_artifact", (tuple, (int, UUID)))),
}
_SCHEMA = {cls: tuple((n, t) for names, t in rows for n in names.split())
           for cls, rows in _SCHEMA.items()}


def _walk(live, detached, spec, edges, cells):
    if type(spec) is tuple and spec[0] is None:
        spec = type(None) if detached is None else spec[1]
    if spec in (str, int, bool, type(None)):
        if type(live) is not spec or type(detached) is not spec:
            raise TypeError("unsupported native scalar")
        return
    kind = spec[0] if type(spec) is tuple and spec[0] in (tuple, frozenset) else spec
    if type(spec) is tuple and kind is spec:
        kind = tuple  # Explicit heterogeneous pair descriptor.
    if type(live) is not kind or type(detached) is not kind:
        raise TypeError("unsupported native node or container")
    if kind is UUID:
        if type(detached.int) is not int or not 0 <= detached.int < 2**128:
            raise ValueError("unsupported native UUID")
        cells.append((live, "int", int, detached.int))
        edges.append((live, None, UUID, live))
    elif kind in (tuple, frozenset):
        if len(live) != len(detached):
            raise ValueError("native container changed during capture")
        left, right = live, detached
        if kind is frozenset:
            if any(type(v) is not UUID or type(v.int) is not int
                   for values in (live, detached) for v in values):
                raise TypeError("unsupported native UUID set")
            left = sorted(live, key=lambda v: v.int)
            right = sorted(detached, key=lambda v: v.int)
        pair = spec[0] not in (tuple, frozenset)
        if pair and len(live) != len(spec):
            raise TypeError("unsupported native pair")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            child = spec[index] if pair else spec[1]
            _walk(a, b, child, edges, cells)
            if kind is tuple:
                _record(live, index, a, b, edges, cells)
    else:
        rows = _SCHEMA[kind]
        if {f.name for f in fields(kind)} != {n for n, _ in rows}:
            raise TypeError("native schema fields changed")
        for name, child in rows:
            a, b = getattr(live, name), getattr(detached, name)
            _walk(a, b, child, edges, cells)
            _record(live, name, a, b, edges, cells)


def _record(parent, key, live, detached, edges, cells):
    if type(detached) in (str, int, bool, type(None)):
        cells.append((parent, key, type(detached), detached))
    else:
        edges.append((parent, key, type(live), live))


def _capture(value, spec):
    # Reject unknown objects before deepcopy can dispatch their custom behavior.
    _walk(value, value, spec, [], [])
    detached = deepcopy(value)
    edges, cells = [], []
    _walk(value, detached, spec, edges, cells)
    return detached, (value, spec, tuple(edges), tuple(cells))


def capture_selected(selected):
    detached, witness = _capture(selected, SelectedGraphScopeV2)
    for group in detached.groups:
        _NATIVE_GRAPH_GROUP_VALIDATOR(group)
        _NATIVE_READY_BUNDLE_VALIDATOR(group.scope.ready)
    _NATIVE_GRAPH_SELECTION_VALIDATOR(detached)
    compare(witness, selected)
    return witness


def capture_direct(scope, spans):
    _, witness = _capture(scope, DirectSeedScopeV1)
    if type(spans) is not dict:
        raise TypeError("unsupported native span mapping")
    expected = []
    for key, value in spans.items():
        _walk(key, key, (int, int, str), [], [])
        _walk(value, value, (int, str), [], [])
        expected.append((tuple(key), tuple(value)))
    result = (witness, spans, tuple(expected))
    compare(result, scope, spans)
    return result


def compare(witness, value, spans=None):
    if spans is not None:
        witness, original, expected = witness
        if (type(spans) is not dict or spans is not original
            or len(spans) != len(expected)):
            raise ValueError("native span mapping changed")
        # Exact types before equality prevent bool/int keys and hostile __eq__.
        for (key, row), (wanted_key, wanted_row) in zip(
            spans.items(), expected, strict=True
        ):
            if (type(key) is not tuple or type(row) is not tuple
                or len(key) != 3 or len(row) != 2
                or any(type(v) is not t for v, t in zip(
                    key, (int, int, str), strict=True))
                or any(type(v) is not t for v, t in zip(row, (int, str), strict=True))
                or key != wanted_key or row != wanted_row):
                raise ValueError("native span values changed")
    root, kind, edges, cells = witness
    if value is not root or type(value) is not kind:
        raise ValueError("native data root changed")
    for parent, key, kind, expected in edges:
        actual = (parent if key is None else parent[key]
                  if type(key) is int else getattr(parent, key))
        if type(actual) is not kind or actual is not expected:
            raise ValueError("native data identity changed")
    for parent, key, kind, expected in cells:
        actual = parent[key] if type(key) is int else getattr(parent, key)
        if type(actual) is not kind or actual != expected:
            raise ValueError("native data value changed")


_NATIVE_VALUE_FUNCTIONS = tuple((n, globals()[n]) for n in (
    "_walk", "_record", "_capture", "capture_selected", "capture_direct", "compare"))
