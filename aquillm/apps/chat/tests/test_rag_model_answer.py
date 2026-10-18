"""Model presentation cannot mint source links or media authority."""

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import pytest


def prepare(packet, text):
    module = "apps.chat.services.rag_model_answer"
    assert find_spec(module) is not None, "model presentation boundary is missing"
    return import_module(module).prepare_model_answer(packet, text)


def packet():
    return SimpleNamespace(citation_tokens=("[doc:a chunk:1]",), image_urls=[])


@pytest.mark.parametrize(
    "text",
    [
        "The redshift is 4.119 [18], with 0.3 < z < 10 [doc:a chunk:1].",
        "The estimate is **bounded**.\n\n- First condition.\n- Second condition.",
        "For $x < 10$, use $\\alpha = 0.3$.\n\n$$x^2 + y^2 = 1$$",
        "See [the external reference](https://example.org/paper).",
        "Use `x < 10` and the literal &lt;img&gt; notation.",
        "For $x<y + z$, use $$a<b + c$$ and `x<y`.",
        "A formula:\n\n$$\nx<y + z\n$$\n\n[doc:a chunk:1]",
    ],
)
def test_preserves_ordinary_model_prose_and_scientific_notation(text):
    assert prepare(packet(), text) == text


@pytest.mark.parametrize("prefix", ["The source reports", "> The source reports", "- The source reports"])
def test_inline_citation_followed_by_colon_is_prose_not_a_reference_definition(prefix):
    text = f"{prefix} [doc:a chunk:1]:\n\n- A cited condition [doc:a chunk:1]."
    assert prepare(packet(), text) == text


@pytest.mark.parametrize("prefix", [
    "", "   ", "\t", "> ", "> > ", ">", ">>>", "- ", "1. ", "2) ", "> - ", ">- ", "- > ",
])
def test_reference_definitions_at_markdown_block_starts_remain_rejected(prefix):
    text = f"A cited statement [doc:a chunk:1].\n\n{prefix}[source]: https://example.org/paper"
    with pytest.raises(ValueError, match="^invalid_presentation$"):
        prepare(packet(), text)


@pytest.mark.parametrize("prefix", ["", "> ", ">", "- ", "1. ", "> - "])
def test_exact_authorized_citation_can_introduce_a_block_with_a_colon(prefix):
    text = f"{prefix}[doc:a chunk:1]: The source describes a condition."
    assert prepare(packet(), text) == text


@pytest.mark.parametrize("label", ["source", "doc:foreign chunk:9", "doc:a chunk:01", "doc:a  chunk:1"])
def test_unknown_or_citation_like_reference_labels_are_not_exempt(label):
    with pytest.raises(ValueError, match="^invalid_presentation$"):
        prepare(packet(), f"[{label}]: https://example.org/paper")


@pytest.mark.parametrize(
    "markup",
    [
        '<img src="https://evil.example/pixel">',
        '<a href="/aquillm/document/a/" class="rag-citation-link">Source 1</a>',
        "<!-- hidden assertion -->",
        "<script>alert(1)</script>",
        "<iframe src='https://evil.example'></iframe>",
        "<iframe src=https://evil.example/ [doc:a chunk:1]",
        "<script [doc:a chunk:1]",
        "<style [doc:a chunk:1]",
        "<img src=https://evil.example/pixel [doc:a chunk:1]",
        "before $$\n<div style=display:none [doc:a chunk:1]\n$$",
        "before $$\n<custom-panel hidden [doc:a chunk:1]\n$$",
        "<!DOCTYPE html>",
        "<?xml version='1.0'?>",
        "![image][picture]\n\n[picture]: https://evil.example/pixel",
        "![picture][]\n\n[picture]: /aquillm/document_image/a/",
        "![picture]\n\n[picture]: /aquillm/document_image/a/",
        "![image][doc:a chunk:1]\n\n[doc:a chunk:1]: https://evil.example/pixel",
        "[doc:a chunk:1]: /aquillm/document/a/",
        "[source]: /aquillm/document/a/",
        "[Source](/aquillm/document/a/?chunk=1)",
        "[Source](https://app.example/aquillm/document/a/?chunk=1)",
        "[Source](https://app.example/aquillm/pdf/a/)",
        "https://app.example/api/chunks/1/",
        "[Source](https://app.example/aquillm/%64ocument/a/)",
        "[Source](&#47;aquillm/document/a/)",
        "[Source [nested]](document/a/)",
        "[Source](https://[invalid)",
        '[Source](../document/a/ "citation")',
        "https://app.example/aquillm/document/a/?chunk=1",
        "![image](https://evil.example/pixel)",
        "![image](/aquillm/document_image/a/ 'untrusted title')",
        "![unfinished](/aquillm/document_image/a/",
        "[doc:foreign chunk:99]",
    ],
)
def test_rejects_uncontrolled_media_or_forged_source_presentation(markup):
    with pytest.raises(
        ValueError, match="^(invalid_media|invalid_presentation|invalid_citation)$"
    ):
        prepare(packet(), f"A supported statement [doc:a chunk:1].\n\n{markup}")


def test_selected_url_alone_never_grants_media_authority():
    current = packet()
    current.image_urls = ["/aquillm/document_image/a/?chunk_id=1"]
    with pytest.raises(ValueError, match="^invalid_media$"):
        prepare(
            current,
            "A statement [doc:a chunk:1].\n"
            "![claimed caption](/aquillm/document_image/a/?chunk_id=1)",
        )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The condition is x<y", "The condition is x&lt;y"),
        ("x<y\nNext line.", "x&lt;y\nNext line."),
        ("<custom-panel [doc:a chunk:1]", "&lt;custom-panel [doc:a chunk:1]"),
        (
            "before $$\n<custom-panel [doc:a chunk:1]\n$$",
            "before $$\n&lt;custom-panel [doc:a chunk:1]\n$$",
        ),
    ],
)
def test_ambiguous_html_prefixes_remain_literal_visible_science(text, expected):
    assert prepare(packet(), text) == expected


def test_captured_media_uses_server_caption_and_retains_prose(monkeypatch):
    # Capture validation belongs to the acquired-media subsystem; keep its real
    # selected_image_lines policy and replace only the runtime-bound summary.
    from apps.chat.services import rag_immutable_images

    current = packet()
    url = "/aquillm/document_image/a/?chunk_id=1"
    monkeypatch.setattr(
        rag_immutable_images, "media_summary", lambda p: ({"url": url},)
    )
    text = (
        "The selected caption describes galaxies [doc:a chunk:1].\n\n"
        f"![Model claims proof]({url})"
    )
    result = prepare(current, text)
    assert result == (
        "The selected caption describes galaxies [doc:a chunk:1].\n\n"
        "![Selected source image](/aquillm/document_image/a/?chunk_id=1)"
    )
    assert "Model claims proof" not in result


def test_mixed_authorized_and_unacquired_images_fail_closed(monkeypatch):
    from apps.chat.services import rag_immutable_images

    url = "/aquillm/document_image/a/?chunk_id=1"
    monkeypatch.setattr(
        rag_immutable_images, "media_summary", lambda p: ({"url": url},)
    )
    with pytest.raises(ValueError, match="^invalid_media$"):
        prepare(
            packet(),
            f"![allowed]({url})\n![other](/aquillm/document_image/b/?chunk_id=2)",
        )


def test_nested_image_markup_is_not_reinterpreted_as_a_captured_image(monkeypatch):
    from apps.chat.services import rag_immutable_images

    url = "/aquillm/document_image/a/?chunk_id=1"
    monkeypatch.setattr(
        rag_immutable_images, "media_summary", lambda p: ({"url": url},)
    )
    with pytest.raises(ValueError, match="^invalid_media$"):
        prepare(packet(), f"![outer ![inner]({url})](https://evil.example/image)")
