"""
Text processing utilities for parsers.
"""

import re
from collections import Counter


def read_text_bytes(data: bytes) -> str:
    """Decode bytes to text with encoding detection."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


# Typographic ligatures (U+FB00-FB06) that pypdf returns as single characters,
# e.g. "ﬁlter". Mapped explicitly rather than with NFKC, which would also
# flatten superscripts and other symbols (10⁻³ -> 10-3).
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_LIGATURE_RE = re.compile("[ﬀ-ﬆ]")
_WORD_RE = re.compile(r"[A-Za-z]+")
# A word hyphenated across a line break: "pho-\ntometric".
_LINE_BREAK_HYPHEN_RE = re.compile(r"([A-Za-z]+)-[ \t]*\n[ \t]*([a-z]{3,})")


def expand_ligatures(text: str) -> str:
    """Replace ligature characters with plain letters ("ﬁlter" -> "filter")."""
    return _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group()], text)


def word_vocabulary(*texts: str) -> Counter:
    """Lower-cased word counts across texts, for rejoining split words."""
    return Counter(word.lower() for text in texts for word in _WORD_RE.findall(text))


def clean_pdf_text(text: str, vocabulary: Counter | None = None) -> str:
    """Repair common pypdf extraction artifacts so search matches the words.

    - Ligature characters become plain letters ("ﬁlter" -> "filter").
    - Words hyphenated across a line break are rejoined when the joined word
      appears unbroken in the vocabulary ("pho-\\ntometric" -> "photometric");
      otherwise only the line break is removed, keeping real compounds
      ("near-\\ninfrared" -> "near-infrared"). The vocabulary defaults to the
      text itself; pass one built from the whole document when cleaning it a
      page at a time.
    """
    text = expand_ligatures(text)
    if vocabulary is None:
        vocabulary = word_vocabulary(text)

    def _join(match: re.Match) -> str:
        head, tail = match.group(1), match.group(2)
        if vocabulary[(head + tail).lower()]:
            return head + tail
        return f"{head}-{tail}"

    return _LINE_BREAK_HYPHEN_RE.sub(_join, text)


__all__ = ['clean_pdf_text', 'expand_ligatures', 'read_text_bytes', 'word_vocabulary']
