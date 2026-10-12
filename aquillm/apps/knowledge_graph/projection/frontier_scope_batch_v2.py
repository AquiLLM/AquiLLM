"""Bounded authenticated directory seeks; raw source records always stay fresh."""

from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import _key

from .frontier_reads_v2 import KINDS, schema_error


def _heads(reads, generation, names):
    if not 1 <= len(names) <= 32 or len(set(names)) != len(names):
        raise schema_error()
    root = reads.manifests[generation]["frontier_root_key"]
    pending = {name: (root, set(), "", "\uffff") for name in names}
    heads = {}
    while pending:
        for key, seen, _, _ in pending.values():
            if key in seen or len(seen) > 32:
                raise schema_error()
            seen.add(key)
        keys = tuple(sorted({state[0] for state in pending.values()}))
        records = reads.indices(generation, keys)
        following = {}
        for name, (key, seen, low, high) in pending.items():
            kind, node = records[key]
            if (kind != "directory" or type(node) is not dict
                    or set(node) != {"name", "head", "left", "right"}):
                raise schema_error()
            current = node["name"]
            if type(current) is not str or not low < current < high:
                raise schema_error()
            if current == name:
                if node["head"] is None:
                    raise schema_error()
                heads[name] = node["head"]
                continue
            child = node["left"] if name < current else node["right"]
            if child is None:
                raise schema_error()
            following[name] = (
                child, seen, low if name < current else current,
                current if name < current else high,
            )
        pending = following
    return heads


def scope_rows(reads, generation, sources):
    expected = {}
    for offset in range(0, len(sources), 32):
        names = tuple("scope/" + source for source in sources[offset:offset + 32])
        heads = _heads(reads, generation, names)
        reads.indices(generation, tuple(sorted(set(heads.values()))))
        for stream in names:
            entry = reads.entry(generation, heads[stream], stream=stream)
            if entry["family"] != "scope" or entry["next"] is not None:
                raise schema_error()
            for ref in entry["refs"]:
                if type(ref) is not list or len(ref) != 3 or ref[0] not in KINDS:
                    raise schema_error()
                label, key, checksum = ref
                _key(key)
                _key(checksum)
                identity = label, key
                if identity in expected and expected[identity] != checksum:
                    raise schema_error()
                expected[identity] = checksum
    # Check conflicts across every source before grouping shared raw lookup keys.
    keys = tuple(sorted({key for _, key in expected}))
    rows = {}
    for offset in range(0, len(keys), 32):
        batch = set(keys[offset:offset + 32])
        refs = [[label, key, checksum] for (label, key), checksum in expected.items()
                if key in batch]
        rows.update(reads.hydrate(generation, refs))
    return rows
