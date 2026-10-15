"""Global stream-head merge and relation-hop admission under one ledger."""

from heapq import heappop, heappush

from apps.knowledge_graph.projection.frontier_reads_v2 import schema_error

from .frontier_contracts_v2 import FrontierBudgetExceeded


def acquire_frontier(session, assembly, seeds, caps, budget):
    frontier = tuple(s.identity_key for s in seeds)
    visited, mention_streams = set(), []
    for depth in range(caps.max_depth + 1):
        heap, serial = [], 0
        pairs = tuple(
            (generation, identity)
            for identity in sorted(set(frontier))
            for generation in sorted(session.selected)
            if (generation, identity) not in visited
        )
        prepared = session.prepare_copies(pairs, relation_heads=depth < caps.max_depth)
        # Read every competing stream head before admitting this round.
        for identity in sorted(frontier):
            for generation in sorted(session.selected):
                pair = generation, identity
                if pair in visited:
                    continue
                visited.add(pair)
                for copy, rows in prepared[pair]:
                    if not budget.admit_identity(identity) or not assembly.add_copy(
                        identity, depth, rows
                    ):
                        raise FrontierBudgetExceeded(
                            "frontier identity/physical copy limit"
                        )
                    entity = copy["order"][0]
                    mention_streams.append(
                        (
                            generation,
                            copy["links"]["mentions"],
                            entity,
                            rows,
                            identity,
                        )
                    )
                    # Copy joins and mentions cost zero relation hops, including
                    # at the horizon; no terminal relation head may be fetched.
                    if depth == caps.max_depth:
                        continue
                    for predicate, head in sorted(copy["links"]["relations"].items()):
                        stream = "relations/" + entity + "/" + predicate
                        iterator = session.entries(generation, head, stream)
                        entry = next(iterator, None)
                        if entry is not None:
                            priority = -dict(session.request.predicate_weights)[
                                predicate
                            ]
                            heappush(
                                heap,
                                (
                                    (priority, *entry["order"], generation),
                                    serial,
                                    generation,
                                    identity,
                                    stream,
                                    entry,
                                    iterator,
                                ),
                            )
                            serial += 1
        discovered = set()
        while heap:
            order, ordinal, generation, origin, stream, entry, iterator = heappop(heap)
            for page in session.relation(generation, entry, stream=stream, depth=depth):
                if page.next_cursor is not None:
                    session.require_cursor(
                        page.next_cursor, generation=generation, depth=depth
                    )
                if len(page.relations) != 1 or not page.relation_evidence:
                    raise schema_error()
                members = {
                    m.entity_key: m.automatic_membership_key or m.entity_key
                    for m in page.memberships
                }
                target = members[entry["order"][3]]
                edge = (origin, page.relations[0].relation_type, target)
                if (
                    len(assembly.candidates)
                    + len(
                        {r.chunk_key for r in page.relation_evidence}
                        - set(assembly.candidates)
                    )
                    > 3000
                ):
                    raise FrontierBudgetExceeded("frontier evidence limit")
                if (
                    len(assembly.hops) >= caps.max_nodes and target not in assembly.hops
                ) or (
                    len(assembly.groups) >= min(caps.max_edges, budget.max_edges)
                    and (*edge, entry["order"][4]) not in assembly.groups
                ):
                    raise FrontierBudgetExceeded("frontier topology limit")
                if not budget.admit_identity(target) or not budget.admit_edge(edge):
                    raise FrontierBudgetExceeded("frontier cumulative admission limit")
                if not assembly.add_relation(page, entry, generation, origin, depth):
                    raise FrontierBudgetExceeded("frontier physical membership limit")
                if assembly.hops[target] == depth + 1:
                    discovered.add(target)
            successor = next(iterator, None)
            if successor is not None:
                heappush(
                    heap,
                    (
                        (order[0], *successor["order"], generation),
                        ordinal,
                        generation,
                        origin,
                        stream,
                        successor,
                        iterator,
                    ),
                )
        frontier = tuple(sorted(discovered))
        if not frontier:
            break
    for generation, head, entity, rows, identity in mention_streams:
        for page in session.mention(generation, head, entity, rows):
            if (
                len(assembly.candidates)
                + len(
                    {r.chunk_key for r in page.mention_evidence}
                    - set(assembly.candidates)
                )
                > 3000
            ):
                raise FrontierBudgetExceeded("frontier evidence limit")
            assembly.add_mention(page, generation, identity)
