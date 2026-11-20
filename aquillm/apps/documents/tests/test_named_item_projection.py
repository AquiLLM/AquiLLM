from hashlib import sha256

from apps.documents.services.named_item_projection import build_items, label_terms


def member(pk, text, start, end):
    return (pk, pk, start, end, sha256(text[start:end].encode()).hexdigest())


def test_literal_source_items_bind_exact_intersections_across_chunks():
    text = "Intro.\n• Alpha setting: " + "x" * 160 + "\n- Beta setting: " + "y" * 150
    rows = [member(1, text, 0, 135), member(2, text, 105, len(text))]
    items = build_items(text, rows)
    first, second = items
    assert text[first["start"] : first["label_end"]] == "Alpha setting"
    assert first["end"] == text.index("- Beta")
    assert first["refs"] == [1]
    assert second["refs"] == [2]
    assert first["terms"] == ["alpha", "setting"]
    assert label_terms("The ALPHA and beta?") == {"alpha", "beta"}


def test_bad_member_digest_or_bounds_disables_whole_owner():
    body = "• Label: " + "x" * 150
    valid = member(1, body, 0, len(body))
    assert build_items(body, [valid, (2, 2, 0, len(body), "0" * 64)]) is None
    assert build_items(body, [(1, 1, 0, len(body) + 1, valid[-1])]) is None
    assert (
        build_items("Body without labels.", [member(1, "Body without labels.", 0, 20)])
        == []
    )


def test_omitted_nonblank_canonical_range_disables_whole_owner():
    text = "• Printed label: " + "x" * 150 + "\nSecond paragraph."
    assert build_items(text, [member(1, text, 0, text.index("\n"))]) is None
    text += "  \n"
    assert build_items(text, [member(1, text, 0, len(text) - 3)])


def test_unprinted_labels_and_short_intersections_do_not_nominate():
    text = "Plain prose: " + "a" * 120 + "\n• A source label: " + "b" * 95
    items = build_items(
        text, [member(1, text, 0, len(text) - 90), member(2, text, 0, len(text))]
    )
    assert len(items) == 1
    assert items[0]["refs"] == [2]


def test_offline_postings_link_all_literal_labels_to_bound_members():
    from apps.documents.services.named_item_projection import index_payload
    import json

    text = "• Moss traits: " + "a" * 120 + "\n• Moss height: " + "x" * 120
    row = member(9, text, 0, len(text))
    index = json.loads(index_payload(build_items(text, [row]), [row]))
    assert index["postings"] == {"moss": [0, 1], "traits": [0], "height": [1]}
    assert index["members"] == [list(row)]
    assert [i["term_count"] for i in index["items"]] == [2, 2]
