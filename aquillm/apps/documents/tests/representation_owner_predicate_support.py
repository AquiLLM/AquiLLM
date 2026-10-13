"""Small retained legacy fixtures and actual lazy-query shape observations."""
from types import SimpleNamespace
from uuid import uuid4

from django.contrib.auth import get_user_model
from django.db import connection
from django.db.models import Max
from django.test.utils import CaptureQueriesContext

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import DESCENDED_FROM_DOCUMENT, PDFDocument, RawTextDocument, TextChunk
from apps.documents.services.representation_store import _active_chunk_predicates, freeze_active_representations
from .representation_support import reference


def legacy_case(*, mixed=False, middle_has_chunk=True):
    user = get_user_model().objects.create_user(username="owner-predicate-" + uuid4().hex)
    collection = Collection.objects.create(name="Retained owner predicates")
    other = Collection.objects.create(name="Other retained owner collection")
    for selected in (collection, other):
        CollectionPermission.objects.create(user=user, collection=selected, permission="VIEW")
    docs, chunks = [], []
    for number in range(3):
        model = PDFDocument if mixed and number == 1 else RawTextDocument
        text = f"Retained canonical source {number}."
        values = dict(title=f"Retained {number}", full_text=text,
                      collection=collection, ingested_by=user)
        if model is PDFDocument:
            # Concrete tables may legitimately have the same integer primary key.
            doc = model.objects.bulk_create([model(pkid=docs[0].pkid,
                full_text_hash=model.hash_fn(text), **values)])[0]
        else:
            doc = model(**values)
            doc.save(dont_rechunk=True)
        docs.append(doc)
        if number != 1 or middle_has_chunk:
            chunks.extend(TextChunk.objects.bulk_create([TextChunk(doc_id=doc.id,
                content=text, start_position=0, end_position=len(text), chunk_number=0)]))
    snapshot = freeze_active_representations(tuple(reference(doc) for doc in docs), using="default")
    assert all(selection.generation_key is None for selection in snapshot.selections)
    return SimpleNamespace(user=user, collection=collection, other=other,
                           docs=tuple(docs), chunks=tuple(chunks), snapshot=snapshot)


def lazy_query(snapshot, *, document_id=None):
    # Only the private SQL builder is measured here; real capture was completed
    # first, and public entry-boundary validation is tested separately.
    with CaptureQueriesContext(connection) as reads:
        query = _active_chunk_predicates(TextChunk.objects.order_by("pk"),
                                        snapshot=snapshot, document_id=document_id)
        sql, params = query.query.sql_with_params()
    assert len(reads) == 0
    return query, sql, params


def owner_shape(sql):
    return {model._meta.label_lower: sql.count("FROM " + connection.ops.quote_name(model._meta.db_table))
            for model in DESCENDED_FROM_DOCUMENT}


def duplicate_owner(doc, *, model, collection):
    text = "A distinct source with a deliberately colliding UUID."
    # Mixed-table fixtures deliberately set equal PKs; do not depend on their sequences.
    next_pk = (model.objects.aggregate(value=Max("pkid"))["value"] or 0) + 1
    return model.objects.bulk_create([model(pkid=next_pk, id=doc.id, title="Collision",
        full_text=text, full_text_hash=model.hash_fn(text), collection=collection,
        ingested_by=doc.ingested_by)])[0]
