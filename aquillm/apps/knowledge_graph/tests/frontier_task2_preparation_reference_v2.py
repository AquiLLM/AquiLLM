"""Reviewed Task2 finite-preparation reference, retained only for parity tests."""

from apps.knowledge_graph.projection.frontier_copy_batch_v2 import (
    copy_entries,
    copy_heads,
)
from apps.knowledge_graph.projection.frontier_generation_batch_v2 import (
    hydrate_all,
    indices_all,
)
from apps.knowledge_graph.projection.frontier_reads_v2 import KINDS, schema_error
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import _key


def _scope_rows_all_reference(reads, sources):
    from apps.knowledge_graph.projection.frontier_generation_batch_v2 import (
        hydrate_all,
        indices_all,
    )

    pending = {
        (g, "scope/" + source): (
            reads.manifests[g]["frontier_root_key"],
            set(),
            "",
            "\uffff",
        )
        for g, keys in sources.items()
        for source in keys
    }
    heads = {}
    while pending:
        for key, seen, _, _ in pending.values():
            if key in seen or len(seen) > 32:
                raise schema_error()
            seen.add(key)
        records = indices_all(
            reads, ((g, state[0]) for (g, _), state in pending.items())
        )
        following = {}
        for (generation, name), (key, seen, low, high) in pending.items():
            kind, node = records[generation, key]
            if (
                kind != "directory"
                or type(node) is not dict
                or set(node) != {"name", "head", "left", "right"}
            ):
                raise schema_error()
            current = node["name"]
            if type(current) is not str or not low < current < high:
                raise schema_error()
            if current == name:
                if node["head"] is None:
                    raise schema_error()
                heads[generation, name] = node["head"]
                continue
            child = node["left"] if name < current else node["right"]
            if child is None:
                raise schema_error()
            following[generation, name] = (
                child,
                seen,
                low if name < current else current,
                current if name < current else high,
            )
        pending = following
    indices_all(reads, ((g, head) for (g, _), head in heads.items()))
    expected = {}
    for (generation, stream), head in heads.items():
        entry = reads.entry(generation, head, stream=stream)
        if entry["family"] != "scope" or entry["next"] is not None:
            raise schema_error()
        for ref in entry["refs"]:
            if type(ref) is not list or len(ref) != 3 or ref[0] not in KINDS:
                raise schema_error()
            label, key, checksum = ref
            _key(key)
            _key(checksum)
            identity = generation, label, key
            if identity in expected and expected[identity] != checksum:
                raise schema_error()
            expected[identity] = checksum
    return hydrate_all(reads, expected)


def _prepare_copies_reference(session, pairs, *, relation_heads):
    reads = session.reads
    entries = copy_entries(session, copy_heads(session, pairs))
    expected = {}
    for (generation, _), copies in entries.items():
        for entry in copies:
            reads._check_index_deadline()
            for ref in entry["refs"]:
                if type(ref) is not list or len(ref) != 3 or ref[0] not in KINDS:
                    raise schema_error()
                label, key, checksum = ref
                _key(key)
                _key(checksum)
                target = generation, label, key
                if target in expected and expected[target] != checksum:
                    raise schema_error()
                expected[target] = checksum
    fresh = hydrate_all(reads, expected)
    prepared, first = {pair: [] for pair in pairs}, []
    for (generation, identity), copies in entries.items():
        for entry in copies:
            reads._check_index_deadline()
            rows = {
                (label, key): fresh[generation][label, key]
                for label, key, _ in entry["refs"]
            }
            session.validate_copy_rows(generation, identity, entry, rows)
            prepared[generation, identity].append((entry, rows))
            if relation_heads:
                for predicate, head in sorted(entry["links"]["relations"].items()):
                    if head is not None:
                        first.append(
                            (
                                generation,
                                head,
                                "relations/" + entry["order"][0] + "/" + predicate,
                            )
                        )
    indices_all(reads, ((generation, head) for generation, head, _ in first))
    for generation, head, stream in first:
        reads.entry(generation, head, stream=stream)
    return prepared
