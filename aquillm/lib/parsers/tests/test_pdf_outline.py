"""PDF outline anchors must be proven against existing canonical source text."""

from io import BytesIO

import fitz
import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

from lib.parsers.canonical_projection import validate_extraction
from lib.parsers.source_chunking import chunk_extraction, chunker_key_for
from lib.parsers.source_contracts import SourceContractError

from .structural_extraction_support import extract


def pdf(pages, outline):
    with fitz.open() as doc:
        for lines in pages:
            page = doc.new_page()
            for i, line in enumerate(lines):
                page.insert_text((50, 70 + 24 * i), line)
        if outline:
            doc.set_toc(outline)
        return doc.tobytes()


PAGES = [
    [
        "Preamble",
        "1. Methods",
        "First measurement.",
        "1.1 Details",
        "Second measurement.",
    ],
    ["Continuation.", "2. Results", "Final measurement."],
]
OUTLINE = [[1, "Methods", 1], [2, "1.1 Details", 1], [1, "Results", 2]]


class Codepoints:
    revision = "test-codepoint"

    def encode(self, text):
        return text


def test_valid_outline_partitions_the_existing_canonical_text_and_reaches_chunks():
    original = extract("baseline.pdf", pdf(PAGES, []))
    result = extract("outlined.pdf", pdf(PAGES, OUTLINE))
    assert result.parser_key == "native-pdf-ordered-v3"
    assert result.canonical_text == original.canonical_text
    assert (
        result.quality == original.quality
        and result.issue_codes == original.issue_codes
    )
    validate_extraction(result)
    text = result.canonical_text
    headings = [
        (e.page, text[e.start : e.end].rstrip("\n"))
        for e in result.elements
        if e.kind == "heading"
    ]
    assert headings == [(1, "1. Methods"), (1, "1.1 Details"), (2, "2. Results")]
    options = dict(tokenizer=Codepoints(), max_tokens=128, overlap_tokens=0)
    chunks = chunk_extraction(result, chunker_key=chunker_key_for(**options), **options)
    assert all(c.source_text == text[c.start : c.end] for c in chunks)
    assert any(
        c.embedding_text.startswith("heading=1.1 Details\nSecond measurement.")
        for c in chunks
    )
    assert any(
        c.embedding_text.startswith("heading=2. Results\nFinal measurement.")
        for c in chunks
    )


@pytest.mark.parametrize(
    "pages, outline",
    [
        (
            PAGES,
            [[1, "Methods", 1], [1, "Final", 2]],
        ),  # a body substring is not a title
        (PAGES, [[1, "Methods", 1], [1, "Results", 1]]),  # text is on the wrong page
        (PAGES, [[1, "Results", 2], [1, "Methods", 1]]),  # contradictory order
        ([["1. Methods", "2. Methods", "Text"]], [[1, "Methods", 1]]),
        (PAGES, [[1, "Methods", 1], [1, "1. Methods", 1]]),  # two anchors at one offset
    ],
)
def test_unresolved_outline_keeps_all_historical_page_elements(pages, outline):
    original = extract("baseline.pdf", pdf(pages, []))
    result = extract("ambiguous.pdf", pdf(pages, outline))
    assert result.parser_key == original.parser_key == "native-pdf-ordered-v2"
    assert (result.canonical_text, result.elements) == (
        original.canonical_text,
        original.elements,
    )


def test_cyclic_outline_does_not_break_normal_extraction():
    writer = PdfWriter(clone_from=PdfReader(BytesIO(pdf(PAGES, OUTLINE))))
    first = writer.root_object["/Outlines"].raw_get("/First")
    first.get_object()[NameObject("/Next")] = first
    stream = BytesIO()
    writer.write(stream)
    result = extract("cyclic.pdf", stream.getvalue())
    assert result.parser_key == "native-pdf-ordered-v2"
    assert "Final measurement." in result.canonical_text


@pytest.mark.parametrize("damage", [
    "missing_next", "early_last", "missing_child_first", "wrong_prev", "wrong_parent",
])
def test_inconsistent_outline_links_keep_the_historical_page_boundaries(damage):
    writer = PdfWriter(clone_from=PdfReader(BytesIO(pdf(PAGES, OUTLINE))))
    outline = writer.root_object["/Outlines"]
    first, last = outline["/First"], outline["/Last"]
    if damage == "missing_next":
        del first["/Next"]
    elif damage == "early_last":
        outline[NameObject("/Last")] = outline.raw_get("/First")
    elif damage == "missing_child_first":
        del first["/First"]
    elif damage == "wrong_prev":
        last[NameObject("/Prev")] = first.raw_get("/Parent")
    else:
        last[NameObject("/Parent")] = outline.raw_get("/First")
    stream = BytesIO()
    writer.write(stream)
    original = extract("baseline.pdf", pdf(PAGES, []))
    result = extract("damaged.pdf", stream.getvalue())
    assert result.parser_key == "native-pdf-ordered-v2"
    assert (result.canonical_text, result.elements) == (
        original.canonical_text, original.elements,
    )


@pytest.mark.parametrize("target, stores, expected_pages", [
    (NameObject, "legacy", [1]), (TextStringObject, "tree", [2]),
    (NameObject, "tree", []), (TextStringObject, "legacy", []),
    (NameObject, "both", [1]), (TextStringObject, "both", [2]),
    (NameObject, "malformed_tree", []), (TextStringObject, "malformed_tree", []),
])
def test_local_destinations_respect_pdf_namespaces(target, stores, expected_pages):
    data = pdf([["Results", "One."], ["Results", "Two."]], [])
    writer = PdfWriter(clone_from=PdfReader(BytesIO(data)))
    if stores in ("legacy", "both"):
        dest = ArrayObject([writer.pages[0].indirect_reference, NameObject("/Fit")])
        writer.root_object[NameObject("/Dests")] = DictionaryObject({
            NameObject("/mist"): DictionaryObject({NameObject("/D"): dest}),
        })
    if stores in ("tree", "both", "malformed_tree"):
        writer.add_named_destination("/mist", 1)
        if stores == "malformed_tree":
            writer.root_object["/Names"]["/Dests"]["/Names"][0] = NameObject("/mist")
    item = writer.add_outline_item("Results", 0)
    item.get_object()["/A"][NameObject("/D")] = target("/mist")
    stream = BytesIO()
    writer.write(stream)
    result = extract("named.pdf", stream.getvalue())
    assert [e.page for e in result.elements if e.kind == "heading"] == expected_pages
    assert result.parser_key == (
        "native-pdf-ordered-v3" if expected_pages else "native-pdf-ordered-v2"
    )


@pytest.mark.parametrize("state, expected", [
    ("collapsed", 3), ("expanded", 3), ("unspecified", 3),
    ("lost_children", 0), ("contradictory_root", 0),
])
def test_outline_counts_respect_collapsed_and_optional_subtrees(state, expected):
    writer = PdfWriter(clone_from=PdfReader(BytesIO(pdf(PAGES, OUTLINE))))
    root = writer.root_object["/Outlines"]
    parent = root["/First"]
    parent[NameObject("/Count")] = NumberObject(1 if state == "expanded" else -1)
    root[NameObject("/Count")] = NumberObject(3 if state == "expanded" else 2)
    if state == "lost_children":
        del parent["/First"]
        del parent["/Last"]
    elif state == "contradictory_root":
        root[NameObject("/Count")] = NumberObject(3)
    elif state == "unspecified":
        del parent["/Count"]
    stream = BytesIO()
    writer.write(stream)
    result = extract("counts.pdf", stream.getvalue())
    assert sum(e.kind == "heading" for e in result.elements) == expected
    assert result.parser_key == (
        "native-pdf-ordered-v3" if expected else "native-pdf-ordered-v2"
    )


@pytest.mark.parametrize(
    "damage",
    ["bad_fit", "missing_fit_operands", "chained_action", "remote", "unknown_name"],
)
def test_unusable_destination_cannot_promote_a_matching_string(damage):
    writer = PdfWriter(clone_from=PdfReader(BytesIO(pdf(PAGES, []))))
    ref = writer.add_outline_item("Methods", 0)
    action = ref.get_object()["/A"]
    if damage in ("bad_fit", "missing_fit_operands"):
        fit = "/Unsupported" if damage == "bad_fit" else "/FitR"
        action[NameObject("/D")] = ArrayObject(
            [writer.pages[0].indirect_reference, NameObject(fit)]
        )
    elif damage == "chained_action":
        action[NameObject("/Next")] = DictionaryObject(
            {NameObject("/S"): NameObject("/GoToR")}
        )
    elif damage == "remote":
        action[NameObject("/S")] = NameObject("/GoToR")
    else:
        action[NameObject("/D")] = TextStringObject("unavailable")
    stream = BytesIO()
    writer.write(stream)
    result = extract("broken-outline.pdf", stream.getvalue())
    assert result.parser_key == "native-pdf-ordered-v2"
    assert not any(e.kind == "heading" for e in result.elements)


def test_over_budget_outline_leaves_original_page_partition(monkeypatch):
    from lib.parsers import pdf_outline

    monkeypatch.setattr(pdf_outline, "MAX_ENTRIES", 2)
    result = extract("capped.pdf", pdf(PAGES, OUTLINE))
    assert result.parser_key == "native-pdf-ordered-v2"
    assert [e.kind for e in result.elements] == ["paragraph", "paragraph"]


def test_long_outline_title_keeps_complete_source_under_a_smaller_embedding_budget():
    title = "i" * 81
    result = extract("long.pdf", pdf([[title, "Evidence."]], [[1, title, 1]]))
    assert result.parser_key == "native-pdf-ordered-v3"
    options = dict(tokenizer=Codepoints(), max_tokens=64, overlap_tokens=8)
    chunks = chunk_extraction(result, chunker_key=chunker_key_for(**options), **options)
    assert all(c.source_text == result.canonical_text[c.start:c.end] for c in chunks)
    assert all(len(c.embedding_text) <= 64 for c in chunks)
    assert set(range(len(result.canonical_text))) == {
        i for c in chunks for i in range(c.start, c.end)
    }
    assert chunks[-1].source_text == chunks[-1].embedding_text == "Evidence."


def test_nonfitting_pdf_prefix_spends_no_fallback_work(monkeypatch):
    from lib.parsers import source_chunking

    class Tokenizer(Codepoints):
        def encode(self, text):
            if text.startswith("heading=Methods\n") and text != "heading=Methods\n":
                return range(65)
            return super().encode(text)

    monkeypatch.setattr(source_chunking, "MAX_FALLBACK_TOKENIZER_CALLS", 0)
    result = extract("nonfit.pdf", pdf([["Methods", "Evidence."]], [[1, "Methods", 1]]))
    options = dict(tokenizer=Tokenizer(), max_tokens=64, overlap_tokens=0)
    chunks = chunk_extraction(result, chunker_key=chunker_key_for(**options), **options)
    assert "".join(c.source_text for c in chunks) == result.canonical_text
    assert [c.embedding_text for c in chunks] == ["Methods\n", "Evidence."]


@pytest.mark.parametrize("mode, reason", [
    ("error", "local_tokenizer_failed"), ("oversize", "unrepresentable_source_element")
])
def test_optional_pdf_heading_preserves_real_source_errors(mode, reason):
    class Tokenizer(Codepoints):
        def encode(self, text):
            if "@" in text:
                if mode == "error":
                    raise ValueError("tokenizer unavailable")
                return range(65)
            return super().encode(text)

    result = extract("invalid.pdf", pdf([["Methods", "@"]], [[1, "Methods", 1]]))
    options = dict(tokenizer=Tokenizer(), max_tokens=64, overlap_tokens=0)
    with pytest.raises(SourceContractError, match=reason):
        chunk_extraction(result, chunker_key=chunker_key_for(**options), **options)
