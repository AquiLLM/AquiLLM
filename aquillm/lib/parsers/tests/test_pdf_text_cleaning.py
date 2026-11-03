"""test pdf text cleaning: ligatures and line-break hyphenation from pypdf."""

from lib.parsers.text_utils import clean_pdf_text, word_vocabulary


def test_ligatures_become_plain_letters():
    assert clean_pdf_text("ﬁve ﬁlters, ﬂux, eﬀect, eﬃcient, baﬄe") == (
        "five filters, flux, effect, efficient, baffle"
    )


def test_superscripts_are_left_alone():
    assert clean_pdf_text("10⁻³ erg cm⁻²") == "10⁻³ erg cm⁻²"


def test_line_break_split_rejoined_when_word_seen_elsewhere():
    text = "We use photometric data.\nThe pho-\ntometric redshifts are good."
    assert clean_pdf_text(text) == (
        "We use photometric data.\nThe photometric redshifts are good."
    )


def test_line_break_compound_keeps_hyphen():
    assert clean_pdf_text("in the near-\ninfrared band") == "in the near-infrared band"


def test_ligature_inside_split_word_is_rejoined():
    text = "the ﬁlter set; each ﬁl-\nter is broad"
    assert clean_pdf_text(text) == "the filter set; each filter is broad"


def test_short_tail_is_not_joined():
    assert clean_pdf_text("see astro-\nph for details") == "see astro-\nph for details"


def test_hyphen_without_line_break_is_untouched():
    assert clean_pdf_text("a well-known photo-z method") == "a well-known photo-z method"


def test_vocabulary_from_other_pages_is_used():
    vocabulary = word_vocabulary("Spectroscopic redshifts on page one.")
    assert clean_pdf_text("spectro-\nscopic", vocabulary) == "spectroscopic"


def test_pdf_extraction_cleans_pages_with_document_vocabulary(monkeypatch):
    import pypdf
    from .structural_extraction_support import extract

    class _Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class _Reader:
        def __init__(self, stream):
            self.pages = [_Page("ﬁve ﬁlters; spectroscopic data"), _Page("the spectro-\nscopic ﬂux")]

    monkeypatch.setattr(pypdf, "PdfReader", _Reader)
    result = extract('ligatures.pdf', b'%PDF-synthetic')
    assert result.canonical_text == "five filters; spectroscopic data\nthe spectroscopic flux"
    assert [(e.page, e.start, e.end) for e in result.elements] == [(1, 0, 33), (2, 33, 55)]
