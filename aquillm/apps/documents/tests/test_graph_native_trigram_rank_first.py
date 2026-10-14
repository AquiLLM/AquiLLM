"""Rank-first native trigram capture against the frozen rank-before-limit reader."""

import re
from hashlib import md5
from uuid import uuid4

import pytest
from django.contrib.postgres.search import TrigramSimilarity
from django.db import connection, connections
from django.db.models import Case, FloatField, Value, When
from django.test.utils import CaptureQueriesContext

from apps.documents.models import (
    ChunkEmbedding,
    DocumentEmbeddingState,
    EmbeddingGeneration,
    EmbeddingSpace,
    PDFDocument,
    RawTextDocument,
    TextChunk,
)
from apps.documents.services import graph_candidate_metadata as metadata
from apps.documents.services.representation_store import (
    _active_chunk_predicates,
    freeze_active_representations,
)

from .representation_owner_predicate_support import (
    duplicate_owner,
    lazy_query,
    legacy_case,
)
from .representation_support import (
    activate,
    reference,
)
from .representation_support import (
    retained as retained,
)
from .representation_support import (
    upload as upload,
)

pytestmark = pytest.mark.django_db(transaction=True)


def frozen_262(text, query, limit, similarity):
    """Verbatim 2621eeeb old reader, independent of the candidate builder."""
    return metadata.rank_refs(
        text.annotate(similarity=TrigramSimilarity("content", query))
        .filter(similarity__gt=similarity).order_by("-similarity", "pk")[:limit]
    )


def compare(text, query, limit, threshold):
    expected = frozen_262(text, query, limit, threshold)
    actual = metadata._trigram_refs(text, query, limit, threshold)
    assert actual == expected
    return actual


def insert_chunks(doc, contents):
    return TextChunk.objects.bulk_create([
        TextChunk(doc_id=doc.id, content=content, chunk_number=1000 + i,
                  start_position=10000 + 1000 * i, end_position=10001 + 1000 * i)
        for i, content in enumerate(contents)
    ])


def test_native_rank_query_keeps_scoped_topn_inside_one_statement(monkeypatch):
    case = legacy_case()
    text, _, _ = lazy_query(case.snapshot)
    text = text.filter(doc_id__in=[case.docs[0].id], modality=TextChunk.Modality.TEXT)
    original = metadata.rank_refs
    ranked = []

    def observe(queryset):
        ranked.append(queryset)
        return original(queryset)

    monkeypatch.setattr(metadata, "rank_refs", observe)
    with CaptureQueriesContext(connection) as reads:
        metadata._trigram_refs(text, "Retained canonical", 1, -1)
    assert len(reads) == 1
    assert len(ranked) == 1 and ranked[0].db == text.db
    sql, _ = ranked[0].query.sql_with_params()
    assert bool(re.search(r'"aquillm_textchunk"\."id" IN \(SELECT', sql))
    inner = sql.split('"aquillm_textchunk"."id" IN (SELECT', 1)[1]
    assert "sha256(" in inner
    assert "representation_generation_id" in inner
    assert "NULLS LAST" in inner
    assert all(token in inner for token in (
        "modality", "doc_id", "collection_id", "full_text_hash", "NOT",
    ))
    assert bool(re.search(r'ORDER BY .* DESC NULLS LAST, .*"id" ASC LIMIT 1', inner))
    pk_only = bool(re.match(r' [A-Z]\d+\."id" AS "pk"$',
                            inner.split(" FROM ", 1)[0]))
    assert pk_only
    executed = reads[0]["sql"]
    projection = executed.split(" FROM ", 1)[0]
    assert '"aquillm_textchunk"."content",' not in projection
    assert "MD5(" in projection and "LENGTH(" in projection
    assert executed.count("SELECT") >= 2


def test_rank_first_matches_frozen_reader_across_thresholds_limits_and_ties():
    case = legacy_case()
    rows = insert_chunks(case.docs[0], ["alpha exact"] * 48 + [
        "beta near alpha", "unrelated", "München 東京 alpha", "", "a", "alpha exact",
    ])
    text, _, _ = lazy_query(case.snapshot)
    text = text.filter(pk__in=[row.pk for row in rows])
    equality = text.filter(pk=rows[0].pk).annotate(
        score=TrigramSimilarity("content", "alpha exact")
    ).values_list("score", flat=True).get()
    assert equality > 0
    for threshold in (-1, 0, equality, 0.9, 1):
        for limit in (0, 1, 17, 45, 70, None):
            compare(text, "alpha exact", limit, threshold)
    assert compare(text, "alpha exact", 17, 0)[-1].pk == rows[16].pk
    assert compare(text, "alpha exact", 45, 0)[-1].pk == rows[44].pk
    assert compare(text, "alpha exact", 70, 0) == compare(text, "alpha exact", None, 0)
    assert compare(text, "alpha exact", 17, equality) == ()
    assert compare(text, "alpha exact", 17, 1) == ()
    assert len(compare(text, "alpha exact", None, -1)) == 45
    assert tuple(ref.pk for ref in compare(
        text.filter(pk__in=[row.pk for row in rows[:3]]),
        "alpha exact", 17, 0,
    )) == tuple(row.pk for row in rows[:3])


def test_empty_body_remains_post_limit_and_unicode_content_is_ranked():
    case = legacy_case()
    rows = insert_chunks(case.docs[0], [
        "", "not related", "München 東京", "München 東京",
    ])
    text, _, _ = lazy_query(case.snapshot)
    text = text.filter(pk__in=[row.pk for row in rows])
    assert compare(text.filter(pk__in=[rows[0].pk, rows[1].pk]),
                   "target", 1, -1) == ()
    refs = compare(text, "München 東京", 2, 0)
    assert tuple(ref.pk for ref in refs) == (rows[2].pk, rows[3].pk)
    assert all(ref.revision == md5("München 東京".encode()).hexdigest()
               and ref.length == len("München 東京") for ref in refs)


def test_nullable_score_cannot_displace_qualifying_row(monkeypatch):
    case = legacy_case()
    rows = insert_chunks(case.docs[0], ["alpha exact", "alpha exact"])
    text, _, _ = lazy_query(case.snapshot)
    text = text.filter(pk__in=[row.pk for row in rows])
    original = TrigramSimilarity

    def nullable(field, query):
        return Case(When(pk=rows[0].pk, then=Value(None)),
                    default=original(field, query), output_field=FloatField())

    monkeypatch.setattr(metadata, "TrigramSimilarity", nullable)
    monkeypatch.setitem(frozen_262.__globals__, "TrigramSimilarity", nullable)
    result = compare(text, "alpha exact", 1, 0)
    assert tuple(ref.pk for ref in result) == (rows[1].pk,)


def test_negative_limit_preserves_old_error():
    case = legacy_case()
    text, _, _ = lazy_query(case.snapshot)
    with pytest.raises(ValueError, match="Negative indexing is not supported"):
        frozen_262(text, "source", -1, 0)
    with pytest.raises(ValueError, match="Negative indexing is not supported"):
        metadata._trigram_refs(text, "source", -1, 0)


def test_distinct_and_sliced_custom_shapes_keep_old_behavior():
    case = legacy_case()
    text, _, _ = lazy_query(case.snapshot)
    compare(text.distinct(), "Retained canonical", 2, -1)
    sliced = text[:2]
    with pytest.raises(TypeError):
        frozen_262(sliced, "Retained canonical", 1, -1)
    with pytest.raises(TypeError):
        metadata._trigram_refs(sliced, "Retained canonical", 1, -1)


@pytest.mark.django_db(transaction=True, databases=["default", "projection_source"])
def test_outer_and_inner_execute_on_the_supplied_database_alias(monkeypatch):
    original = metadata.rank_refs
    ranked = []

    def observe(queryset):
        ranked.append(queryset)
        return original(queryset)

    monkeypatch.setattr(metadata, "rank_refs", observe)
    text = TextChunk.objects.using("projection_source").filter(doc_id=uuid4())
    with (CaptureQueriesContext(connections["default"]) as default_reads,
          CaptureQueriesContext(connections["projection_source"]) as projection_reads):
        assert metadata._trigram_refs(text, "alpha", 1, 0) == ()
    assert ranked[0].db == text.db == "projection_source"
    assert len(default_reads) == 0 and len(projection_reads) == 1


def test_duplicate_producing_join_preserves_repeated_pk_rows():
    case = legacy_case()
    row = case.chunks[0]
    space = EmbeddingSpace.objects.create(space_key="trigram-" + uuid4().hex,
        declaration={}, verification_evidence={})
    state = DocumentEmbeddingState.objects.create(
        model_label=case.docs[0]._meta.label_lower, document_pkid=case.docs[0].pkid,
        document_id=case.docs[0].id, collection_id=case.collection.pk,
        representation_key=None, space=space)
    for _ in range(2):
        generation = EmbeddingGeneration.objects.create(owner=state,
            source_hash="0" * 64, canonical_sha256="0" * 64,
            manifest_sha256="0" * 64, manifest={}, expected_count=1)
        ChunkEmbedding.objects.create(generation=generation, chunk=row,
            vector=[0] * 1024, receipt={}, serving_binding={},
            input_sha256="0" * 64, modality="text", source_md5="0" * 32)
    text, _, _ = lazy_query(case.snapshot)
    text = text.filter(pk=row.pk, chunkembedding__isnull=False)
    result = compare(text, "Retained canonical", 2, -1)
    assert tuple(ref.pk for ref in result) == (row.pk, row.pk)


@pytest.mark.parametrize("exclusion", [
    "selected_scope", "wrong_owner", "stale_body", "duplicate_owner",
])
def test_rank_first_retains_original_source_eligibility(exclusion):
    case = legacy_case()
    text, _, _ = lazy_query(case.snapshot)
    if exclusion == "selected_scope":
        text = text.filter(doc_id=case.docs[0].id)
    elif exclusion == "wrong_owner":
        other = RawTextDocument(title="Outside", full_text="Retained canonical",
            full_text_hash=RawTextDocument.hash_fn("Retained canonical"),
            collection=case.other, ingested_by=case.user)
        other.save(dont_rechunk=True)
        insert_chunks(other, ["Retained canonical"])
    elif exclusion == "stale_body":
        case.docs[0].full_text = "changed"
        case.docs[0].save(dont_rechunk=True)
    else:
        duplicate_owner(case.docs[0], model=PDFDocument, collection=case.collection)
    result = compare(text, "Retained canonical", 17, -1)
    excluded = ({case.docs[0].id}
                if exclusion in ("stale_body", "duplicate_owner") else set())
    if exclusion == "selected_scope":
        assert {ref.doc_id for ref in result} == {case.docs[0].id}
    else:
        assert all(ref.doc_id not in excluded for ref in result)
        assert all(ref.doc_id in {doc.id for doc in case.docs} for ref in result)


def test_active_generation_and_same_body_replacement_stay_fenced(retained):
    doc, output, old = retained
    activate(doc, output, text="corrected table 100", parser="parser-v2")
    doc.refresh_from_db()
    original = freeze_active_representations((reference(doc),), using="default")
    current = _active_chunk_predicates(
        TextChunk.objects.filter(doc_id=doc.id), snapshot=original,
    )
    first = compare(current, "corrected table", 17, -1)
    assert first and old.pk not in {ref.pk for ref in first}
    assert all(ref.generation == original.selections[0].generation_key for ref in first)
    activate(doc, output, text="corrected table 100", parser="parser-v3")
    doc.refresh_from_db()
    replacement = freeze_active_representations((reference(doc),), using="default")
    assert (replacement.selections[0].generation_key
            != original.selections[0].generation_key)
    assert compare(current, "corrected table", 17, -1) == ()
    fresh = _active_chunk_predicates(
        TextChunk.objects.filter(doc_id=doc.id), snapshot=replacement,
    )
    assert compare(fresh, "corrected table", 17, -1)
