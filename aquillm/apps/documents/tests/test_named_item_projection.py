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


def test_explicit_abstract_chooses_the_member_covering_most_of_its_text():
    import json
    from apps.documents.services.named_item_projection import index_payload

    prefix = "A. Author, Institute.\n" * 20
    body = "Soil properties respond to irrigation. " * 28
    canonical = prefix + "  aBsTrAcT  \r\n" + body + "\n\n1 Introduction\n" + "Science. " * 230
    start = canonical.index("  aBsTrAcT")
    rows = [
        member(1, canonical, 0, start + 190),
        member(2, canonical, start - 40, start + 870),
        member(3, canonical, start + 850, len(canonical)),
    ]
    projected = json.loads(index_payload(build_items(canonical, rows), rows, canonical=canonical))
    abstract = projected["abstract"]
    assert projected["version"] == 2 and abstract["refs"] == [2]
    assert canonical[abstract["start"]:abstract["label_end"]].strip().lower() == "abstract"
    assert projected["postings"] == {} and projected["items"] == []
    assert projected["members"] == [list(rows[1])]


def test_abstract_index_ignores_unmarked_prose_and_late_headings():
    import json
    from apps.documents.services.named_item_projection import index_payload

    for text in ("Our abstract summarizes soil.\n" * 26, "Background. " * 50 + "\nAbstract\n" + "B" * 150):
        rows = [member(1, text, 0, len(text))]
        projected = json.loads(index_payload(build_items(text, rows), rows, canonical=text))
        assert not projected.get("abstract") and projected["members"] == []


def test_abstract_metadata_never_overflows_a_previously_eligible_list_index(monkeypatch):
    from apps.documents.services import named_item_projection as projection

    text = "Abstract\n" + "Microscopy resolved the structure. " * 7 + "\n\n• Crystal facets: " + "x" * 150
    rows = [member(1, text, 0, len(text))]
    items = build_items(text, rows)
    old = projection.index_payload(items, rows)
    assert len(projection.index_payload(items, rows, canonical=text)) > len(old)
    monkeypatch.setattr(projection, "MAX_PAYLOAD", len(old))
    assert projection.index_payload(items, rows, canonical=text) == old
