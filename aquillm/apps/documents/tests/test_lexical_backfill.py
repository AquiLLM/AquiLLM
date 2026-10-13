"""Provider-free refresh is a finite fenced page, not a corpus sweep."""

import json
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.documents.models import TextChunk
from apps.documents.tests.test_embedding_generation import legacy as legacy
from .lexical_owner_support import unchunked as unchunked
from .representation_support import retained as retained, upload as upload, activate
from apps.documents.tests.test_lexical_relevance import replace_chunks

pytestmark = pytest.mark.django_db(transaction=True)


def test_backfill_cursor_bounds_work_and_preserves_source_rows(unchunked):
    legacy = unchunked
    rows = replace_chunks(legacy, ["source one", "source two", "source three"])
    out = StringIO()
    call_command(
        "backfill_chunk_search_vectors", batch_size=2, max_batches=1, stdout=out
    )
    receipt = json.loads(out.getvalue())
    assert receipt["processed"] == 2
    assert receipt["after_pk"] == rows[1].pk
    assert TextChunk.objects.filter(search_vector_binding__isnull=False).count() == 2
    call_command(
        "backfill_chunk_search_vectors",
        batch_size=2,
        max_batches=1,
        after_pk=receipt["after_pk"],
        stdout=StringIO(),
    )
    assert TextChunk.objects.filter(search_vector_binding__isnull=False).count() == 3
    assert list(
        TextChunk.objects.order_by("chunk_number").values_list("content", flat=True)
    ) == ["source one", "source two", "source three"]


@pytest.mark.parametrize(
    "option,value",
    [
        ("batch_size", 0),
        ("batch_size", 1001),
        ("max_batches", 0),
        ("max_batches", 101),
        ("after_pk", -1),
    ],
)
def test_backfill_rejects_unbounded_controls(option, value):
    with pytest.raises(CommandError, match="invalid bounded backfill"):
        call_command(
            "backfill_chunk_search_vectors", **{option: value}, stdout=StringIO()
        )


def test_retired_rows_are_not_refreshed(retained):
    from apps.documents.models import DocumentRepresentationState
    document, output, old = retained
    assert activate(document, output) == "activated"
    document.refresh_from_db()
    state = DocumentRepresentationState.objects.get(document_id=document.id)
    retired_key = state.active_generation_id
    assert activate(document, output, parser="parser-v3") == "activated"
    retired = TextChunk.objects.filter(representation_generation_id=retired_key)
    assert retired.exists()
    retired.update(search_vector=None, search_vector_binding=None)
    call_command("backfill_chunk_search_vectors", stdout=StringIO())
    assert not retired.filter(search_vector_binding__isnull=False).exists()
    assert TextChunk.objects.get(pk=old.pk).search_vector_binding is None
    document.refresh_from_db()
    assert document.chunks.exists() and not document.chunks.filter(search_vector_binding__isnull=True).exists()


def test_backfill_keeps_stricter_operator_statement_timeout(legacy, monkeypatch):
    from django.db import connection, transaction

    from apps.documents.management.commands import (
        backfill_chunk_search_vectors as command,
    )

    observed = []
    original = command.publish_chunk_search_vectors

    def capture(**kwargs):
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT setting FROM pg_settings WHERE name='statement_timeout'"
            )
            observed.append(int(cursor.fetchone()[0]))
        return original(**kwargs)

    monkeypatch.setattr(command, "publish_chunk_search_vectors", capture)
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '1000ms'")
        call_command("backfill_chunk_search_vectors", stdout=StringIO())
    assert observed == [1000]
