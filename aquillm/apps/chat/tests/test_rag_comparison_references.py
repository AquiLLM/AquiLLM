from dataclasses import replace

import pytest

from apps.chat.services.rag_comparison_references import comparison_reference_context
from apps.chat.services.rag_evidence import build_selected_evidence_packet
from apps.chat.services.rag_selection_types import EvidenceSelection, SelectionProfile
from .test_rag_comparison_selection import passage


QUERY = "What changed between the field inventory in the two editions?"
COMMON = "ARVO sensing array (Rivera et al. 2017)"
VERSION = "ARVO V4 (Rivera et al. 2017)"
NEIGHBORS = "CEDAR R2 (Cho et al. 2012), VELD M5 (Pine et al. 2014)"


def section(other, extra=""):
    return f"• Field inventory: The occupied stations used: {COMMON}, {other}, {extra}.\n• Care: Clean equipment."


def candidate(pk, owner, number, text):
    raw = passage(pk, owner, number, text, 0.8)
    return replace(raw, row={"i": pk, "d": owner, "c": number, "x": text,
                              "ref": f"[doc:{owner} chunk:{pk}]"})


def packet(*, independent=None, neighbors=NEIGHBORS, question=QUERY):
    independent = independent or f"Other field register: {VERSION}, {neighbors}."
    rows = [candidate(2, "manual-a", 4, section("reed cups", neighbors)),
            candidate(4, "manual-b", 8, section("brass cups", neighbors)),
            candidate(9, "map-register", 1, independent)]
    return build(rows, question)


def build(rows, question=QUERY):
    selection = EvidenceSelection(tuple(rows), sum(v.token_cost for v in rows),
                                  SelectionProfile("adaptive", 0.8, 0.2, "test"), "complete")
    value = build_selected_evidence_packet(selection, query=question, search_scope="field records")
    value.selection = selection
    value.source_authorization = object()
    return value


def anchors(packet):
    context = comparison_reference_context(packet)
    return context.anchors if context else ()


def test_two_other_items_corroborate_source_context_and_every_returned_offset_is_literal():
    p = packet()
    context = comparison_reference_context(p)
    result = context.anchors
    assert context.sections == tuple(row.text.split("• ",1)[1].split("• ",1)[0] for row in p.selection.candidates[:2])
    assert [v.text for v in result] == [COMMON, COMMON, VERSION]
    assert [v.label for v in result] == ["ARVO sensing array", "ARVO sensing array", "ARVO V4"]
    for anchor in result:
        prepared = next(x for x in p.source_evidence if x.source.chunk_id == anchor.chunk_id)
        assert prepared.source.text[anchor.start:anchor.end] == anchor.text
        assert anchor.document_id == prepared.source.document_id
        assert anchor.citation in p.citation_tokens


@pytest.mark.parametrize("attribution", ["Quinn et al. (2012)", "Quinn et al. 2012"])
@pytest.mark.parametrize(("predicate", "neighbors", "expected"), [
    ("is designated V4", "moss cups", True),
    ("is unrelated to ARVO V4", NEIGHBORS, False),
])
def test_corroboration_reads_the_entire_claim_across_author_year_abbreviations(
    attribution, predicate, neighbors, expected,
):
    independent = (f"Station entry: {VERSION}, {neighbors}. "
                   f"The ARVO sensing array described by {attribution} {predicate}.")
    assert bool(anchors(packet(independent=independent, neighbors=neighbors))) is expected


def test_corroboration_keeps_true_sentence_boundary_after_terminal_author_abbreviation():
    independent = (f"Station entry: {VERSION}. The ARVO sensing array was described by Quinn et al. "
                   "A different device is designated V4.")
    assert not anchors(packet(independent=independent, neighbors="moss cups"))


def test_independent_naming_of_retained_item_permits_an_unrelated_transfer_without_colisted_citations():
    independent = (f"Station entry: {VERSION}. The maintenance register lists the "
                   '“ARVO sensing array” under the V4 designation. A neighboring system was separate.')
    assert len(anchors(packet(independent=independent, neighbors="moss cups"))) == 3


@pytest.mark.parametrize("independent", [
    f"Station entry: {VERSION}. The author describes multiple collection methods elsewhere.",
    f"Station entry: {VERSION}. The ARVO sensing array is a different instrument from ARVO V4.",
    f"Station entry: {VERSION}, {NEIGHBORS}. The ARVO sensing array is not the same device as ARVO V4.",
    f"Station entry: {VERSION}, CEDAR R2 (Cho et al. 2012).",
    f"Station entry: {VERSION}, CEDAR R2 (Cho et al. 2012), CEDAR R2 (Cho et al. 2012).",
    f"Station entry: {VERSION}.\n• Unrelated reference collections: {NEIGHBORS}.",
    f"Station entry: {VERSION}.\n \nUnrelated reference collections: {NEIGHBORS}.",
    f"Station entry: {VERSION}, {NEIGHBORS}, ARVO V5 (Rivera et al. 2017).",
    "Station entry: OTHER V4 (Rivera et al. 2017), "+NEIGHBORS+".",
])
def test_ambiguous_or_unsupported_reference_never_proposes_a_positive_link(independent):
    assert not anchors(packet(independent=independent))


def test_single_source_lookup_and_two_documents_only_do_not_target_reference_confirmation():
    p = packet(question="Summarize the field inventory.")
    assert not anchors(p)
    assert not anchors(build(p.selection.candidates[:2]))


def test_duplicate_independent_owners_or_oversized_input_cannot_supply_a_unique_hint():
    p = packet()
    a, b, third = p.selection.candidates
    assert not anchors(build((a, b, third, candidate(16, "other-register", 0, third.text))))
    assert not anchors(build((a, b, third, candidate(13, "noise", 0, "x"*96_000))))


@pytest.mark.parametrize("mutation", ["text", "owner", "ordinal", "citation", "aliases", "fingerprint", "proof", "partial", "duplicate"])
def test_stale_partial_or_disagreeing_selected_identities_do_not_supply_anchors(mutation):
    p = packet()
    c = p.selection.candidates[0]
    if mutation == "text":
        p.chunks[0]["x"] += " changed"
    elif mutation == "owner":
        p.chunks[0]["d"] = "foreign"
    elif mutation == "ordinal":
        p.chunks[0]["c"] += 1
    elif mutation == "citation":
        p.chunks[0]["ref"] = p.chunks[1]["ref"]
    elif mutation == "aliases":
        p.chunks[0]["doc_id"] = "foreign"
    elif mutation == "fingerprint":
        p.selection = replace(p.selection, candidates=(replace(c, source_fingerprint="new"), *p.selection.candidates[1:]))
    elif mutation == "partial":
        partial = replace(c.prepared_evidence, source_coverage="partial")
        p = build((replace(c, prepared_evidence=partial), *p.selection.candidates[1:]))
    elif mutation == "duplicate":
        p = build((*p.selection.candidates, c))
    else:
        p.source_evidence = tuple(reversed(p.source_evidence))
    assert not anchors(p)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(("other", "expected_chunks"), [
    ("The ARVO sensing array is a different instrument from ARVO V4.", ()),
    ("This page lists acquisition dates.", (9,)),
    ("The ARVO sensing array is cataloged as V4.", (9,)),
])
def test_multiple_selected_mentions_in_one_document_do_not_depend_on_retrieval_rank(
    reverse, other, expected_chunks,
):
    prefix = f"Station entry: {VERSION}. "
    first = packet(independent=prefix + "The ARVO sensing array is cataloged as V4.", neighbors="moss cups")
    primary = first.selection.candidates[:2]
    independent = (first.selection.candidates[2], candidate(10, "map-register", 2, prefix + other))
    if reverse:
        independent = independent[::-1]
    selected = anchors(build((*primary, *independent)))
    assert tuple(anchor.chunk_id for anchor in selected[2:]) == expected_chunks


def test_exact_verified_section_overlaps_keep_the_reference_at_its_actual_owner_coordinate():
    p = packet()
    tail = section("reed cups", NEIGHBORS)
    pad = " ".join(f"station{i}" for i in range(30))
    whole = tail.replace("used:", f"used after {pad}:")
    end = whole.index(COMMON) + 12
    a = candidate(1, "manual-a", 3, whole[:end])
    b = candidate(2, "manual-a", 4, whole[end-115:])
    result = anchors(build((a, b, *p.selection.candidates[1:])))
    assert result and result[0].chunk_id == 2 and result[0].text == COMMON
    assert not anchors(build((a, replace(b, chunk_number=6), *p.selection.candidates[1:])))
