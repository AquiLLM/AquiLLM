"""Primary text metadata stays independent of overlapping figure locations."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.documents.services.lexical_projection import structural_projection
from lib.parsers.source_contracts import ExtractionV1, SourceElementV1, source_artifact_ref


def extraction():
    text = "Paragraph with figure overlay"
    artifact = source_artifact_ref(text.encode(), member_path=None, container_sha256=None)
    primary = SourceElementV1("body", "paragraph", 0, len(text), None, None, None, None)
    overlay = SourceElementV1("figure", "figure", 0, len(text), None, None, None, "body")
    return ExtractionV1("synthetic-v1", artifact, text, (primary, overlay), "complete", ())


def test_text_primary_partition_ignores_figure_overlay():
    source = extraction()
    chunk = SimpleNamespace(start_position=0, end_position=len(source.canonical_text))
    assert structural_projection(chunk, source.canonical_text, source) == ("body", [])


def test_ambiguous_primary_partition_still_rejected():
    source = extraction()
    duplicate = replace(source.elements[0], element_key="duplicate")
    source = replace(source, elements=(*source.elements, duplicate))
    with pytest.raises(ValueError):
        structural_projection(SimpleNamespace(start_position=0, end_position=len(source.canonical_text)),
                              source.canonical_text, source)
