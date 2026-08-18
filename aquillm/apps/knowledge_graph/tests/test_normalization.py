from __future__ import annotations

import pytest

from apps.knowledge_graph.resolution.normalization import (
    normalize_entity_label,
    parse_stable_identifier,
)


@pytest.mark.parametrize(
    ("left", "right", "expected_key"),
    [
        ("Ｒｅｔｒｉｅｖａｌ", "Retrieval", "retrieval"),
        ("Cafe\u0301", "Caf\u00e9", "café"),
        (
            "  Retrieval—Augmented\tGeneration  ",
            "retrieval augmented generation",
            "retrieval augmented generation",
        ),
        ("O’Reilly", "O'Reilly", "oreilly"),
    ],
)
def test_entity_label_normalization_folds_unicode_whitespace_and_safe_punctuation(
    left, right, expected_key
):
    normalized_left = normalize_entity_label(left)
    normalized_right = normalize_entity_label(right)

    assert normalized_left.key == normalized_right.key == expected_key


def test_display_label_preserves_case_and_version_suffix():
    normalized = normalize_entity_label("  LLaMA—3.1-v2  ")

    assert normalized.display_label == "LLaMA—3.1-v2"
    assert normalized.key == "llama 3.1 v2"
    assert normalized.version_suffix == "v2"


def test_display_uses_nfc_while_match_key_uses_nfkc_and_preserves_colons():
    fullwidth = normalize_entity_label("Ｒｅｔｒｉｅｖａｌ")
    colon = normalize_entity_label("T5:base")

    assert fullwidth.display_label == "Ｒｅｔｒｉｅｖａｌ"
    assert fullwidth.key == "retrieval"
    assert colon.key == "t5:base"
    assert colon.key != normalize_entity_label("T5 base").key


def test_version_signature_preserves_compound_release_qualifiers():
    normalized = normalize_entity_label("Llama 3.1 8B Instruct")
    versionless = normalize_entity_label("Llama")

    assert normalized.base_key == "llama"
    assert normalized.version_signature == "3.1+8b+instruct"
    assert versionless.base_key == "llama"
    assert versionless.version_signature is None


@pytest.mark.parametrize(
    ("raw", "expected_base", "expected_signature"),
    [
        ("Orion/v1", "orion", "v1"),
        ("T5:base", "t5", "base"),
        ("DeepSeek-R2", "deepseek", "r2"),
        ("Orion/rc2", "orion", "rc2"),
        ("Llama:8B", "llama", "8b"),
        ("Llama-3.1-8B-Instruct", "llama", "3.1+8b+instruct"),
    ],
)
def test_version_signatures_are_detected_across_preserved_separators(
    raw, expected_base, expected_signature
):
    normalized = normalize_entity_label(raw)

    assert normalized.base_key == expected_base
    assert normalized.version_signature == expected_signature


@pytest.mark.parametrize("raw", ["R2D2", "Basecamp", "Instructor", "Model:release"])
def test_version_detection_does_not_treat_ordinary_words_as_release_signatures(raw):
    normalized = normalize_entity_label(raw)

    assert normalized.base_key == normalized.key
    assert normalized.version_signature is None


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Orion v1", "Orion v2"),
        ("Llama 2", "Llama 3"),
        ("GPT-4", "GPT-4o"),
        ("C", "C++"),
        ("dataset/a", "dataset a"),
    ],
)
def test_normalization_does_not_collapse_distinct_names_or_versions(left, right):
    assert normalize_entity_label(left).key != normalize_entity_label(right).key


@pytest.mark.parametrize(
    ("raw", "scheme", "value", "canonical"),
    [
        (
            "https://doi.org/10.48550/arXiv.1706.03762",
            "doi",
            "10.48550/arxiv.1706.03762",
            "doi:10.48550/arxiv.1706.03762",
        ),
        (
            "DOI: 10.1000/ABC_def.2",
            "doi",
            "10.1000/abc_def.2",
            "doi:10.1000/abc_def.2",
        ),
        (
            "arXiv:1706.03762v7",
            "arxiv",
            "1706.03762v7",
            "arxiv:1706.03762v7",
        ),
        (
            "https://arxiv.org/abs/hep-th/9901001v2",
            "arxiv",
            "hep-th/9901001v2",
            "arxiv:hep-th/9901001v2",
        ),
        (
            "https://orcid.org/0000-0002-1825-0097",
            "orcid",
            "0000-0002-1825-0097",
            "orcid:0000-0002-1825-0097",
        ),
        (
            "https://github.com/Fastino-AI/GLiNER2.git",
            "repository",
            "github.com/fastino-ai/gliner2",
            "repository:github.com/fastino-ai/gliner2",
        ),
        (
            "github:Fastino-AI/GLiNER2",
            "repository",
            "github.com/fastino-ai/gliner2",
            "repository:github.com/fastino-ai/gliner2",
        ),
        (
            "git@github.com:Fastino-AI/GLiNER2.git",
            "repository",
            "github.com/fastino-ai/gliner2",
            "repository:github.com/fastino-ai/gliner2",
        ),
        (
            "ssh://git@gitlab.com/Fastino-AI/GLiNER2.git",
            "repository",
            "gitlab.com/fastino-ai/gliner2",
            "repository:gitlab.com/fastino-ai/gliner2",
        ),
        (
            "https://github.com/Fastino-AI/GLiNER2.GIT",
            "repository",
            "github.com/fastino-ai/gliner2",
            "repository:github.com/fastino-ai/gliner2",
        ),
        (
            "repository:github.com/fastino-ai/gliner2",
            "repository",
            "github.com/fastino-ai/gliner2",
            "repository:github.com/fastino-ai/gliner2",
        ),
    ],
)
def test_stable_identifiers_are_parsed_to_typed_canonical_values(
    raw, scheme, value, canonical
):
    identifier = parse_stable_identifier(raw)

    assert identifier is not None
    assert identifier.scheme == scheme
    assert identifier.value == value
    assert identifier.canonical == canonical


@pytest.mark.parametrize(
    "raw",
    [
        "10.123/not-a-doi",
        "doi:11.1000/not-a-doi",
        "arXiv:2024.99999",
        "https://arxiv.org/abs/1706.03762/extra",
        "0000-0002-1825-0098",
        "https://orcid.org/0000-0002-1825-0098",
        "https://github.com/Fastino-AI/GLiNER2/issues/1",
        "https://example.com/Fastino-AI/GLiNER2",
        "https://gitlab.com/group/repo/-/tree/main",
        "https://github.com/../etc",
        "https://github.com/%2e%2e/etc",
        "https://gitlab.com/group/%2e%2e/repo",
        "https://github.com/owner%2frepository",
        "https://github.com:443/owner/repository",
        "https://gitlab.com:8443/group/repository",
        "https://doi.org:443/10.48550/arXiv.1706.03762",
        "https://arxiv.org:443/abs/1706.03762",
        "https://orcid.org:443/0000-0002-1825-0097",
        "https://github.com:invalid/owner/repository",
        "https://github.com:/owner/repository",
        "ssh://git@github.com:/owner/repository",
        "github:owner-only",
        "owner/repository",
        "A paper with DOI 10.1000/xyz in prose",
    ],
)
def test_invalid_or_embedded_identifier_like_text_is_rejected(raw):
    assert parse_stable_identifier(raw) is None


@pytest.mark.parametrize("raw", ["", "   ", None, 42])
def test_normalization_rejects_non_text_or_empty_labels(raw):
    with pytest.raises(ValueError, match="nonempty string"):
        normalize_entity_label(raw)


@pytest.mark.parametrize("raw", ["\x00Orion", "\x01", "\u200b", "Orion\u202e"])
def test_normalization_rejects_unpersistable_or_control_only_labels(raw):
    with pytest.raises(ValueError, match="control|meaningful"):
        normalize_entity_label(raw)
