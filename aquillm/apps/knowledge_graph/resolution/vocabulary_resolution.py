"""Thin adapters apply scoped negative policy to the existing resolver records."""
from dataclasses import replace

from .vocabulary_policy import component_veto


def document_veto_decisions(mentions, decisions, vocabulary, collection_id):
    if vocabulary is None:
        return decisions
    if type(collection_id) is not int or collection_id <= 0:
        raise ValueError("vocabulary document collection required")
    by_id = {m.mention_id: m for m in mentions}
    return tuple(replace(d, accepted=False, method="vocabulary_veto", confidence=0.0,
                         explanation="Rejected by scoped vocabulary.",
                         outcome="rejected" if d.outcome is not None else None)
                 if component_veto(vocabulary, (
                     (collection_id, by_id[d.left_mention_id].raw_text),
                     (collection_id, by_id[d.right_mention_id].raw_text))) else d
                 for d in decisions)
