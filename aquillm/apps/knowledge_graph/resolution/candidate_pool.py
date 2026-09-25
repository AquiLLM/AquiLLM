"""Exact deterministic lexical blocking with one temporary candidate set."""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence


def iter_candidate_pools(
    roots: Sequence[int],
    *,
    component_keys: Mapping[int, str],
    lexical_keys: Callable[[int], Iterable[str]],
    exact_scan_limit: int,
    pool_limit: int,
) -> Iterator[tuple[int, set[int]]]:
    """Preserve block insertion priority and stable cyclic neighbor selection.

    The caller applies its stable hash ranking to each returned pool. For large
    types, block indexes are linear in admitted lexical keys (at most 128 per
    root), and only the current root's bounded candidate set is materialized.
    The existing small-type exact scan still exposes all neighbors for ranking.
    """
    if len(roots) <= exact_scan_limit:
        for root in roots:
            yield root, {other for other in roots if other != root}
        return

    blocks: dict[str, list[int]] = defaultdict(list)
    for root in roots:
        for key in sorted(lexical_keys(root))[:128]:
            blocks[key].append(root)
    positions: dict[int, list[tuple[list[int], int]]] = defaultdict(list)
    for members in blocks.values():
        members.sort(key=component_keys.__getitem__)
        if len(members) > 1:
            for index, root in enumerate(members):
                positions[root].append((members, index))

    for root in roots:
        candidates: set[int] = set()
        for members, index in positions.get(root, ()):
            limit = min(pool_limit, len(members) - 1)
            for offset in range(1, limit + 1):
                if len(candidates) >= pool_limit:
                    break
                candidates.add(members[(index + offset) % len(members)])
            if len(candidates) >= pool_limit:
                break
        if candidates:
            yield root, candidates
