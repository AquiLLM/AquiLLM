"""Nullable receipt migrations do not reinterpret historical vectors."""

from uuid import uuid4

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.utils import timezone


@pytest.mark.django_db(transaction=True)
def test_receipt_migrations_leave_old_vectors_unknown():
    before = [
        ("apps_documents", "0006_reconcile_document_model_state"),
        ("apps_chat", "0010_reconcile_chat_model_state"),
    ]
    after = [
        ("apps_documents", "0007_textchunk_embedding_provenance"),
        ("apps_chat", "0011_conversationchunk_embedding_provenance"),
    ]
    executor = MigrationExecutor(connection)
    leaves = executor.loader.graph.leaf_nodes()
    try:
        executor.migrate(before)
        historical = executor.loader.project_state(before).apps
        old_text = historical.get_model("apps_documents", "TextChunk").objects.create(
            doc_id=uuid4(),
            content="historical text",
            chunk_number=0,
            start_position=0,
            end_position=15,
            embedding=[0.25] * 1024,
        )
        user = historical.get_model("auth", "User").objects.create(
            username="historical-owner"
        )
        conversation = historical.get_model(
            "apps_chat", "WSConversation"
        ).objects.create(
            owner_id=user.pk,
            name="historical",
            created_at=timezone.now(),
            updated_at=timezone.now(),
        )
        old_chat = historical.get_model(
            "apps_chat", "ConversationChunk"
        ).objects.create(
            conversation_id=conversation.pk,
            content="historical chat",
            chunk_number=0,
            start_sequence=0,
            end_sequence=0,
            embedding=[0.5] * 1024,
        )
        executor = MigrationExecutor(connection)
        executor.migrate(after)
        current = executor.loader.project_state(after).apps
        text = current.get_model("apps_documents", "TextChunk").objects.get(
            pk=old_text.pk
        )
        chat = current.get_model("apps_chat", "ConversationChunk").objects.get(
            pk=old_chat.pk
        )
        assert text.embedding_provenance is None and chat.embedding_provenance is None
        assert list(text.embedding) == [0.25] * 1024
        assert list(chat.embedding) == [0.5] * 1024
        assert text.content == "historical text" and chat.content == "historical chat"
    finally:
        MigrationExecutor(connection).migrate(leaves)
