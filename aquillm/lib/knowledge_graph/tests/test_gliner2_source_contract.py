from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.knowledge_graph.config import load_extraction_settings
from lib.knowledge_graph.extractors import gliner2_local


def _pinned_schema():
    # Exact captured GLiNER2 1.3.2 relation builder, without model execution.
    namespace = {}
    path = Path(__file__).with_name("fixtures") / "gliner2_1_3_2_relations.py.txt"
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["Schema"]()


def _ontology():
    return SimpleNamespace(
        entity_types={
            "model": {"description": "A named computational model."},
            "dataset": {"description": "A named collection of observations."},
            "finding": {"description": "A complete observed proposition."},
            "claim": {"description": "A complete asserted proposition."},
        },
        relations={
            "uses_dataset": {
                "description": "The model uses the dataset.",
                "direction": "directed",
                "allowed_head_types": ("model",),
                "allowed_tail_types": ("dataset",),
            },
            "supports": {
                "description": "The finding provides evidence for the claim.",
                "direction": "directed",
                "allowed_head_types": ("finding",),
                "allowed_tail_types": ("claim",),
            },
        },
    )


def test_pinned_relations_api_really_discards_description_strings():
    first = _pinned_schema().relations({"uses_dataset": "uses explicitly"})
    second = _pinned_schema().relations({"uses_dataset": "does not use"})
    assert first.build() == second.build()
    assert first.schema["json_descriptions"] == {}


def test_backend_has_no_experimental_compiler_override():
    with pytest.raises(TypeError, match="schema_compiler_version"):
        gliner2_local.GLiNER2LocalBackend(
            settings=load_extraction_settings({}),
            schema_compiler_version="gliner2-structures-v1",
        )


def _endpoint(text, value):
    start = text.index(value)
    return {"text": value, "confidence": 0.9, "start": start, "end": start + len(value)}


def _normalize(raw, text):
    ontology = _ontology()
    entities, diagnostics = gliner2_local._normalize_entities(
        raw,
        text=text,
        input_index=0,
        known_types=frozenset(ontology.entity_types),
    )
    assert diagnostics == []
    return gliner2_local._normalize_relations(
        raw,
        text=text,
        input_index=0,
        ontology_relations=ontology.relations,
        entities=entities,
    )


@pytest.mark.parametrize(
    "failure",
    ["missing_tail", "subspan_tail", "absent_entity", "missing_section", "role_word"],
)
def test_strict_endpoint_and_section_validation(failure):
    text = (
        "Atlas uses Garden Dataset. The finding leaves grew supports growth improves."
    )
    model_entity = _endpoint(text, "Atlas")
    dataset_entity = _endpoint(text, "Garden Dataset")
    finding_entity = _endpoint(text, "leaves grew")
    claim_entity = _endpoint(text, "growth improves")
    raw = {
        "entities": [
            {
                "model": [model_entity],
                "dataset": [dataset_entity],
                "finding": [finding_entity],
                "claim": [claim_entity],
            }
        ],
        "uses_dataset": [{"head": model_entity, "tail": dataset_entity}],
        "supports": [],
    }
    expected_code = "unresolved_relation_endpoint"
    if failure == "missing_tail":
        raw["uses_dataset"][0].pop("tail")
        expected_code = "malformed_relation_endpoint"
    elif failure == "subspan_tail":
        raw["uses_dataset"][0]["tail"] = _endpoint(text, "Garden")
    elif failure == "absent_entity":
        raw["entities"][0]["dataset"] = []
    elif failure == "missing_section":
        raw.pop("uses_dataset")
        expected_code = "missing_relation_output"
    else:
        raw["uses_dataset"] = []
        raw["supports"] = [{"head": _endpoint(text, "finding"), "tail": claim_entity}]
    relations, diagnostics = _normalize(raw, text)
    assert relations == []
    assert expected_code in {item.code for item in diagnostics}


@pytest.mark.parametrize("independently_emitted", [False, True])
def test_subspan_needs_its_own_typed_entity(independently_emitted):
    text = "Atlas uses Garden Dataset."
    head = _endpoint(text, "Atlas")
    tail = _endpoint(text, "Garden")
    dataset_entities = [_endpoint(text, "Garden Dataset")]
    if independently_emitted:
        dataset_entities.append(tail)
    raw = {
        "entities": [{"model": [head], "dataset": dataset_entities}],
        "uses_dataset": [{"head": head, "tail": tail}],
        "supports": [],
    }
    relations, diagnostics = _normalize(raw, text)
    assert len(relations) == int(independently_emitted)
    if independently_emitted:
        assert relations[0].tail_text == "Garden"
    else:
        assert diagnostics[0].code == "unresolved_relation_endpoint"
