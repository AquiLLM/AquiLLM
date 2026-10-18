"""Offline lexical packing and authenticated transitions; entry encoding is v2."""

from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import _key

_LAYOUT = "lexical-pack-v1"
_PIVOTS = 32


def packed_directory(items, put):
    """Deterministic balanced partitions of already ordered stream heads."""
    previous = ""
    for name, head in items:
        if type(name) is not str or not previous < name < "\uffff" or len(name) > 256:
            raise ValueError("invalid packed directory order")
        _key(head)
        previous = name

    def build(part):
        if not part:
            return None
        count = min(_PIVOTS, len(part))
        offsets = [((i + 1) * len(part)) // (count + 1) for i in range(count)]
        entries, children, start = [], [], 0
        for offset in offsets:
            children.append(build(part[start:offset]))
            entries.append(list(part[offset]))
            start = offset + 1
        children.append(build(part[start:]))
        return put(
            "directory", {"layout": _LAYOUT, "entries": entries, "children": children}
        )

    return build(items)


def packed_step(node, name, low, high):
    if type(name) is not str or not low < name < high:
        raise ValueError("invalid packed lookup range")
    if set(node) != {"layout", "entries", "children"} or node["layout"] != _LAYOUT:
        raise ValueError("unsupported packed directory")
    entries, children = node["entries"], node["children"]
    if (
        type(entries) is not list
        or not 1 <= len(entries) <= _PIVOTS
        or type(children) is not list
        or len(children) != len(entries) + 1
    ):
        raise ValueError("invalid packed directory shape")
    previous = low
    for entry in entries:
        if type(entry) is not list or len(entry) != 2:
            raise ValueError("invalid packed directory pivot")
        current, head = entry
        if (
            type(current) is not str
            or len(current) > 256
            or not previous < current < high
        ):
            raise ValueError("invalid packed directory bounds")
        _key(head)
        previous = current
    for child in children:
        if child is not None:
            _key(child)
    for index, (current, head) in enumerate(entries):
        if name == current:
            return True, head, low, high
        if name < current:
            return False, children[index], low, current
        low = current
    return False, children[-1], low, high
