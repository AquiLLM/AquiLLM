"""Frozen synthetic plan experiment: 20x500 authorized + 500 unauthorized rows.

Queries, sizes and LIMIT were fixed before execution. No planner knobs are set.
The retained pytest output contains each actual EXPLAIN ANALYZE BUFFERS tree.
"""

import json
from hashlib import sha256
from importlib import import_module

import pytest
from django.apps import apps
from django.contrib.auth.models import User
from django.contrib.postgres.search import SearchVector
from django.db import connection
from django.db.models import Value

from apps.collections.models import Collection
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.chunk_search_scope import freeze_search_scope
from apps.documents.services.lexical_projection import (
    LexicalProjectionV1,
    binding_expression,
)
from apps.documents.services.lexical_readiness import (
    current_projection_predicate,
    full_text_queryset,
)
from apps.documents.services.representation_contracts import DocumentRefV1

pytestmark = pytest.mark.django_db(transaction=True)


def corpus():
    user = User.objects.create(username="fts-plan-synthetic-owner")
    authorized = Collection.objects.create(name="FTS authorized synthetic")
    private = Collection.objects.create(name="FTS unauthorized synthetic")
    docs = []
    for document_number in range(21):
        bodies = [
            "measurement sample segment "
            + str(i)
            + " document "
            + str(document_number)
            + " controlled detector signal " * 3
            + ("spectrometer calibration" if i == 499 else "ordinary")
            for i in range(500)
        ]
        text = "\n".join(bodies)
        doc = RawTextDocument(
            title=f"Synthetic {document_number}",
            full_text=text,
            full_text_hash=sha256(text.encode()).hexdigest(),
            ingested_by=user,
            collection=authorized if document_number < 20 else private,
        )
        RawTextDocument.objects.bulk_create([doc])
        start, rows = 0, []
        for number, body in enumerate(bodies):
            rows.append(
                TextChunk(
                    doc_id=doc.id,
                    content=body,
                    chunk_number=number,
                    start_position=start,
                    end_position=start + len(body),
                )
            )
            start += len(body) + 1
        TextChunk.objects.bulk_create(rows)
        ref = DocumentRefV1(
            doc._meta.label_lower,
            doc.pkid,
            doc.id,
            doc.collection_id,
            doc.full_text_hash,
        )
        projection = LexicalProjectionV1(ref, doc.full_text_hash, None, None, doc.title)
        query = TextChunk.objects.filter(doc_id=doc.id)
        query.update(
            search_vector=SearchVector(Value(doc.title), config="english", weight="A")
            + SearchVector(Value(""), config="english", weight="B")
            + SearchVector("content", config="english", weight="D")
        )
        query.update(search_vector_binding=binding_expression(projection))
        docs.append(doc)
    return docs[:20], docs[20]


def test_real_small_large_authorized_plans_with_and_without_gin():
    docs, unauthorized = corpus()
    migration = import_module(
        "apps.documents.migrations.0012_chunk_search_projection_gin_v1"
    )
    receipts = []
    try:
        for with_index in (False, True):
            with connection.schema_editor(atomic=False) as editor:
                (migration.create_index if with_index else migration.drop_index)(
                    apps, editor
                )
            with connection.cursor() as cursor:
                cursor.execute("ANALYZE aquillm_textchunk")
            for size in (1, 20):
                scope = freeze_search_scope(tuple(docs[:size]), using="default")
                eligible = (
                    scope.queryset(TextChunk.objects.all())
                    .filter(modality="text")
                    .filter(current_projection_predicate(scope))
                )
                for query in (
                    "spectrometer calibration",
                    "measurement",
                    "nonexistentmarker",
                ):
                    ranked = full_text_queryset(eligible, query, limit=12)
                    plan = json.loads(
                        ranked.explain(format="JSON", analyze=True, buffers=True)
                    )
                    returned = tuple(ranked.values_list("doc_id", flat=True))
                    assert unauthorized.id not in returned
                    assert set(returned) <= {doc.id for doc in docs[:size]}
                    assert len(returned) == (
                        0
                        if query == "nonexistentmarker"
                        else min(
                            12,
                            size if query == "spectrometer calibration" else size * 500,
                        )
                    )
                    receipts.append(
                        {
                            "with_index": with_index,
                            "authorized_documents": size,
                            "authorized_chunks": size * 500,
                            "query": query,
                            "limit": 12,
                            "plan": plan,
                        }
                    )
    finally:
        with connection.schema_editor(atomic=False) as editor:
            migration.create_index(apps, editor)
    print("R3_FROZEN_POSTGRES_PLANS=" + json.dumps(receipts, sort_keys=True))
