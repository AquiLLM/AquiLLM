"""Public excerpt and current source must describe the same authorized chunk."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from apps.chat.consumers.utils import TOOL_CHUNK_CHAR_LIMIT
from apps.collections.services.retrieval_authorization import (
    OpaquePrincipalReference,
    bind_retrieval_reauthorization_capability,
    freeze_retrieval_authorization_context,
)

DOC = UUID("11111111-1111-4111-8111-111111111111")


class Policy:
    policy_version = "v1"
    policy_checksum = "a" * 64

    def opaque_principal_reference(self, *, principal, database_alias):
        return OpaquePrincipalReference("b" * 64)

    def current_authorized_document_scope(
        self, *, principal, database_alias, selected_collection_ids
    ):
        return ((1, DOC),)


def _authorization():
    principal = object()
    policy = Policy()
    return freeze_retrieval_authorization_context(
        principal=principal,
        database_alias="default",
        policy=policy,
        selected_collection_ids=(1,),
        selected_document_ids=(DOC,),
        reauthorization_capability=bind_retrieval_reauthorization_capability(
            principal=principal, policy=policy
        ),
    )


def test_exact_truncated_excerpt_passes_but_altered_suffix_fails():
    from apps.chat.services.rag_selection_hydration import hydrate_pool_rows

    source = "a" * (TOOL_CHUNK_CHAR_LIMIT + 10)
    chunk = SimpleNamespace(pk=1, doc_id=DOC, chunk_number=0, content=source)
    row = {
        "chunk_id": 1,
        "doc_id": str(DOC),
        "chunk": 0,
        "citation": f"[doc:{DOC} chunk:1]",
        "text": source[:TOOL_CHUNK_CHAR_LIMIT]
        + "\n...[truncated for context window]...",
    }

    def loader(_auth, _ids):
        return (chunk,)

    assert len(hydrate_pool_rows((row,), _authorization(), chunk_loader=loader)) == 1
    forged = {**row, "text": row["text"] + " attacker"}
    assert hydrate_pool_rows((forged,), _authorization(), chunk_loader=loader) == ()
