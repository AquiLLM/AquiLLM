from apps.documents.services.graph_source_admission_policy import item_admission_prefix


def test_index_candidates_share_the_original_allowance_with_ordinary_channels():
    assert item_admission_prefix(
        (1, 2, 3, 4, 5), (10, 11, 12, 13), allowance=5, active_channels=2,
    ) == (1, 2, 10, 11, 12)


def test_overlaps_do_not_spend_a_second_slot_or_displace_a_protected_exact_hit():
    assert item_admission_prefix(
        (1, 2, 3, 4, 5), (1, 1, 9, 10), allowance=4, active_channels=2,
    ) == (1, 2, 9, 10)
    assert item_admission_prefix((1, 2), (9,), allowance=1, active_channels=2) == (1,)


def test_missing_index_and_exhausted_budget_preserve_ordinary_behavior():
    assert item_admission_prefix((1, 2, 3), (), allowance=2, active_channels=1) == (1, 2)
    assert item_admission_prefix((1, 2), (7,), allowance=0, active_channels=2) == ()


def test_abstract_parent_order_comes_only_from_the_first_two_distinct_ranked_pdfs(monkeypatch):
    from types import SimpleNamespace
    from uuid import uuid4
    from apps.documents.services import named_item_lookup
    from apps.documents.services.graph_candidate_admission import _named_item_refs
    from apps.documents.services.named_item_reads import PDF_MODEL

    other, a, b, c = (uuid4() for _ in range(4))
    docs = [SimpleNamespace(id=owner, model_label=PDF_MODEL if owner != other else "rawtext")
            for owner in (a, b, c, other)]
    scope = SimpleNamespace(representation=SimpleNamespace(selections=[
        SimpleNamespace(document=doc) for doc in docs]))
    refs = {pk: SimpleNamespace(doc_id=doc) for pk, doc in enumerate((c, other, a, a, b, c), 1)}
    captured = []

    def record(scope, ordinary, question, *, abstract_owners):
        captured.append((ordinary, abstract_owners))
        return ()

    monkeypatch.setattr(named_item_lookup, "lookup_named_item_candidates", record)
    assert _named_item_refs(scope, "unrelated topic", tuple(refs), refs,
                           named=False, dense_parents=tuple(refs[k] for k in (2, 3, 4, 5, 6))) == ((), ())
    assert captured == [((), (a, b))]


def test_abstracts_take_only_slots_left_after_lane_heads_and_named_witnesses():
    assert item_admission_prefix(
        (1, 2, 3, 4, 5), (7, 8, 9, 10), allowance=5, active_channels=2,
    ) == (1, 2, 7, 8, 9)
