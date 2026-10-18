"""Strict stage-specific bounds before a finite checkpoint reaches a worker."""

from math import isfinite

from apps.knowledge_graph.projection.frontier_index_contracts_v2 import (
    DIRECTORY_VERSIONS,
)
from apps.knowledge_graph.projection.frontier_reads_v2 import KINDS

from .frontier_contracts_v2 import _count, _key
from .frontier_window_mask_v2 import decode_mask

STAGES = {
    "scope_directory": {"paths", "heads"},
    "scope_entries": {"heads"},
    "scope_raw": {"expected", "raw_pending"},
    "copy_directory": {"paths", "heads"},
    "copy_entries": {"paths", "entries"},
    "copy_raw": {"entries", "expected", "raw_pending", "earlier_raw"},
    "copy_first_heads": {"first"},
}


def exact(value, fields):
    if type(value) is not dict or set(value) != fields:
        raise ValueError("frontier checkpoint shape differs")


def array(value, maximum, width=None):
    if type(value) is not list or len(value) > maximum:
        raise ValueError("frontier checkpoint array exceeds bound")
    if width is not None and any(
        type(row) is not list or len(row) != width for row in value
    ):
        raise ValueError("frontier checkpoint row shape differs")
    return value


def lexical(value, maximum=256):
    if type(value) is not str or len(value) > maximum:
        raise ValueError("frontier checkpoint text exceeds bound")


def entry(value, predicates):
    exact(value, {"stream", "order", "refs", "links", "next", "family"})
    lexical(value["stream"])
    array(value["order"], 8)
    if not value["order"] or value["family"] != "copy":
        raise ValueError("frontier copy descriptor invalid")
    for part in value["order"]:
        lexical(part)
    array(value["refs"], 32, 3)
    if not value["refs"]:
        raise ValueError("frontier copy descriptor refs missing")
    for label, key, checksum in value["refs"]:
        if type(label) is not str or label not in KINDS:
            raise ValueError("frontier source label invalid")
        _key(key)
        _key(checksum)
    exact(value["links"], {"relations", "mentions"})
    relations = value["links"]["relations"]
    if (
        type(relations) is not dict
        or len(relations) > 128
        or not set(relations) <= predicates
    ):
        raise ValueError("frontier copy predicates invalid")
    for head in [*relations.values(), value["links"]["mentions"], value["next"]]:
        if head is not None:
            _key(head)


def validate_checkpoint(ready, operation, checkpoint):
    if type(operation) is not dict or operation.get("kind") not in {"scope", "copy"}:
        raise ValueError("frontier finite operation invalid")
    copying = operation["kind"] == "copy"
    exact(
        operation,
        {"kind", "roots", "request_digest"}
        | (
            {"identities", "pair_mask", "weights", "relation_heads"}
            if copying
            else set()
        ),
    )
    _key(operation["request_digest"])
    generations = ready.selected_generations

    def slot(value):
        if type(value) is not int or not 0 <= value < len(generations):
            raise ValueError("frontier generation slot invalid")

    roots = array(operation["roots"], 128, 2)
    for i, key in roots:
        slot(i)
        _key(key)
    if len({i for i, _ in roots}) != len(roots):
        raise ValueError("frontier duplicate root")
    predicates, selected_pairs = set(), set()
    if copying:
        identities = array(operation["identities"], 200)
        for identity in identities:
            _key(identity)
        if (
            identities != sorted(set(identities))
            or type(operation["relation_heads"]) is not bool
        ):
            raise ValueError("frontier copy context invalid")
        selected_pairs = set(
            decode_mask(operation["pair_mask"], len(generations), len(identities))
        )
        for name, weight in array(operation["weights"], 128, 2):
            lexical(name)
            if (
                not name
                or name in predicates
                or type(weight) not in {float, int}
                or not isfinite(weight)
                or not 0 <= weight <= 1
            ):
                raise ValueError("frontier predicate weight invalid")
            predicates.add(name)
    else:
        indices = {g.generation_key: i for i, g in enumerate(generations)}
        selected_pairs = {(i, g.collection_key) for i, g in enumerate(generations)} | {
            (indices[d.generation_key], d.document_key)
            for d in ready.authorized_documents
        }

    def target(i, name):
        slot(i)
        if (
            type(name) is not (int if copying else str)
            or (i, name) not in selected_pairs
        ):
            raise ValueError("frontier target outside captured scope")

    if type(checkpoint) is not dict or checkpoint.get("stage") not in STAGES:
        raise ValueError("frontier finite stage invalid")
    stage = checkpoint["stage"]
    if not stage.startswith(operation["kind"] + "_"):
        raise ValueError("frontier operation stage mismatch")
    version_fields = (
        {"directory_versions"} if "directory_versions" in checkpoint else set()
    )
    exact(checkpoint, {"stage", "guard", "wave"} | STAGES[stage] | version_fields)
    version_slots = set()
    for i, version in array(checkpoint.get("directory_versions", []), 128, 2):
        slot(i)
        if (
            i in version_slots
            or i not in {r[0] for r in roots}
            or type(version) is not str
            or version not in DIRECTORY_VERSIONS
        ):
            raise ValueError("frontier captured directory recipe invalid")
        version_slots.add(i)
    if type(checkpoint["wave"]) is not bool or (
        stage.endswith("raw") and checkpoint["wave"]
    ):
        raise ValueError("frontier wave discriminator invalid")
    guard = array(checkpoint["guard"], 5000, 2)
    for i, key in guard:
        slot(i)
        _key(key)
    if len({tuple(p) for p in guard}) != len(guard):
        raise ValueError("frontier duplicate cached key")
    for field in ("paths", "heads"):
        if field not in checkpoint:
            continue
        width = 3 if field == "heads" else 4 if stage == "copy_entries" else 6
        rows = array(checkpoint[field], len(selected_pairs), width)
        seen = set()
        for i, name, key, *rest in rows:
            target(i, name)
            if (i, name) in seen:
                raise ValueError("frontier duplicate target")
            seen.add((i, name))
            if key is not None or field != "heads" or not copying:
                _key(key)
            if field == "paths" and stage.endswith("directory"):
                low, high, depth = rest
                lexical(low)
                lexical(high)
                _count(depth, 33)
                if not low < high or (
                    not copying and checkpoint["wave"] and depth == 0
                ):
                    raise ValueError("frontier path bounds invalid")
            elif rest:
                previous = rest[0]
                if previous is not None:
                    array(previous, 8)
                    for part in previous:
                        lexical(part)
    if "entries" in checkpoint:
        for i, name, value in array(checkpoint["entries"], 5000, 3):
            target(i, name)
            entry(value, predicates)
            if value["stream"] != "copies/" + operation["identities"][name]:
                raise ValueError("frontier copy descriptor stream invalid")
    if "expected" in checkpoint:
        expected = {}
        for i, label, key, checksum in array(checkpoint["expected"], 5000, 4):
            slot(i)
            if type(label) is not str or label not in KINDS:
                raise ValueError("frontier source label invalid")
            _key(key)
            _key(checksum)
            if (i, label, key) in expected:
                raise ValueError("frontier duplicate expected ref")
            expected[i, label, key] = checksum
        raw = array(checkpoint["raw_pending"], 5000, 2)
        for i, key in raw:
            slot(i)
            _key(key)
        if raw != sorted(
            [list(p) for p in set(map(tuple, raw))],
            key=lambda p: (generations[p[0]].generation_key, p[1]),
        ) or not set(map(tuple, raw)) <= {(i, key) for i, _, key in expected}:
            raise ValueError("frontier remaining raw pairs invalid")
    if "earlier_raw" in checkpoint and type(checkpoint["earlier_raw"]) is not bool:
        raise ValueError("frontier raw boundary invalid")
    if "first" in checkpoint:
        if not operation["relation_heads"] and checkpoint["first"]:
            raise ValueError("frontier horizon forbids relation heads")
        for i, key, stream in array(checkpoint["first"], 5000, 3):
            slot(i)
            _key(key)
            lexical(stream)
            parts = stream.split("/")
            if len(parts) != 3 or parts[0] != "relations" or parts[2] not in predicates:
                raise ValueError("frontier relation stream invalid")
            _key(parts[1])
