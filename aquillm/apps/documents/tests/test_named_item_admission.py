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
