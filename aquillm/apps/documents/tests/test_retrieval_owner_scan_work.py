"""Bound real source hashing to selected owners, preserving lexical identity/order."""
import json

import pytest
from django.contrib.postgres.search import TrigramSimilarity

from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.lexical_candidates import extract_exact_terms, ranked_exact_queryset
from .representation_owner_predicate_support import lazy_query, legacy_case

pytestmark = pytest.mark.django_db(transaction=True)


def nodes(plan):
    yield plan
    for child in plan.get("Plans", ()):
        yield from nodes(child)


@pytest.mark.parametrize("channel", ["trigram", "exact"])
def test_lexical_canonical_source_hash_runs_once_per_owner_scan(channel):
    case = legacy_case()
    # Many retained chunks share each selected canonical body. The fixed query
    # matches every chunk, so LIMIT/ranking cannot hide repeated owner work.
    docs = RawTextDocument.objects.bulk_create([
        RawTextDocument(title=f"Canonical work {number}", full_text=text,
                        full_text_hash=RawTextDocument.hash_fn(text),
                        ingested_by=case.user, collection=case.collection)
        for number in range(3)
        for text in ["HSC-PDR2 calibration controlled synthetic source. " * 400 + str(number)]
    ])
    from apps.documents.services.representation_store import freeze_active_representations
    from .representation_support import reference
    snapshot = freeze_active_representations(tuple(reference(doc) for doc in docs), using="default")
    TextChunk.objects.bulk_create([
        TextChunk(doc_id=doc.id, content="HSC-PDR2 calibration controlled synthetic source.",
                  chunk_number=number, start_position=number * 48, end_position=number * 48 + 47)
        for doc in docs for number in range(40)
    ])
    eligible, _, _ = lazy_query(snapshot)
    query = "HSC-PDR2 calibration"
    if channel == "trigram":
        ranked = eligible.annotate(similarity=TrigramSimilarity("content", query)).filter(
            similarity__gt=0.02).order_by("-similarity", "pk")[:12]
        expected = TextChunk.objects.filter(doc_id__in=[doc.id for doc in docs]).annotate(similarity=TrigramSimilarity("content", query)).filter(
            similarity__gt=0.02).order_by("-similarity", "pk")[:12]
    else:
        terms = extract_exact_terms(query)
        ranked = ranked_exact_queryset(eligible, terms, limit=12)
        expected = ranked_exact_queryset(TextChunk.objects.filter(doc_id__in=[doc.id for doc in docs]), terms, limit=12)
    assert list(ranked.values_list("pk", flat=True)) == list(expected.values_list("pk", flat=True))
    plan = json.loads(ranked.explain(format="JSON", analyze=True, buffers=True))[0]
    hashing = [node for node in nodes(plan["Plan"]) if "sha256(" in node.get("Filter", "")]
    observed = {"channel": channel, "canonical_scan_loops": [node["Actual Loops"] for node in hashing],
                "execution_ms": plan["Execution Time"]}
    print("LEXICAL_OWNER_SCAN_WORK=" + json.dumps(observed, sort_keys=True))
    assert hashing, observed
    assert all(node["Actual Loops"] <= 1 for node in hashing), observed
