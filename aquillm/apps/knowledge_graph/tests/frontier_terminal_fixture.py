"""Terminal identities with distinct own-generation or canonical-copy mentions."""

from dataclasses import replace

from .frontier_fixture_support import bridge_bundles, key


def terminal_bundles(depth, other):
    bundles = list(bridge_bundles())
    identity = (
        bundles[0].automatic_memberships[1].automatic_membership_key
        if depth == 1
        else key("answer-c")
    )
    owner = depth - 1
    index = 1 - owner if other else owner
    bundle = bundles[index]
    members, entities = list(bundle.automatic_memberships), list(bundle.entities)
    member = next(
        (
            m
            for m in members
            if (m.automatic_membership_key or m.entity_key) == identity
        ),
        None,
    )
    if member is None:
        entity_key = key("terminal-copy-entity")
        entities.append(replace(entities[0], entity_key=entity_key))
        member = replace(
            members[0], entity_key=entity_key, automatic_membership_key=identity
        )
        members.append(member)
    chunk_key = key("terminal-mention-chunk")
    chunk = replace(bundle.chunks[0], chunk_key=chunk_key, chunk_number=3)
    mention = replace(
        bundle.entity_mentions[0],
        mention_key=key("terminal-mention"),
        entity_key=member.entity_key,
        chunk_key=chunk_key,
        chunk_number=3,
        provenance_key=key("terminal-provenance"),
    )
    bundles[index] = replace(
        bundle,
        entities=tuple(sorted(entities, key=lambda r: r.entity_key)),
        automatic_memberships=tuple(sorted(members, key=lambda r: r.entity_key)),
        chunks=(*bundle.chunks, chunk),
        entity_mentions=tuple(
            sorted(
                (*bundle.entity_mentions, mention),
                key=lambda r: (r.entity_key, r.provenance_key, r.mention_key),
            )
        ),
        counts=replace(
            bundle.counts,
            entity_count=len(entities),
            automatic_membership_count=len(members),
            chunk_count=2,
            entity_mention_count=2,
        ),
    )
    return tuple(bundles), identity, chunk_key, index
