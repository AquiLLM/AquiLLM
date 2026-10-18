from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.knowledge_graph.config import load_extraction_settings
from lib.knowledge_graph.extractors import gliner2_local


def _pinned_schema():
    # Exact captured GLiNER2 1.3.2 builder methods, not a permissive API fake.
    namespace = {}
    path = Path(__file__).with_name("fixtures") / "gliner2_1_3_2_schema.py.txt"
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


def test_compiler_preserves_semantics_types_order_and_str_metadata():
    from lib.knowledge_graph.extractors.gliner2_schema import compile_schema

    schema = _pinned_schema()
    result = compile_schema(
        schema, _ontology(), schema_compiler_version="gliner2-structures-v1"
    )
    assert result is schema
    assert schema.schema["relations"] == []
    assert [next(iter(item)) for item in schema.schema["json_structures"]] == [
        "supports",
        "uses_dataset",
    ]
    assert list(schema.schema["entities"]) == ["claim", "dataset", "finding", "model"]
    assert schema._active_builder is None
    assert schema._field_orders == {
        "supports": ["head", "tail"],
        "uses_dataset": ["head", "tail"],
    }
    assert all(value["dtype"] == "str" for value in schema._field_metadata.values())
    descriptions = schema.schema["json_descriptions"]["uses_dataset"]
    for role in ("head", "tail"):
        assert "The model uses the dataset." in descriptions[role]
        assert "head to tail" in descriptions[role]
        assert "affirmed" in descriptions[role]
        assert "complete explicit" in descriptions[role]
    assert "model: A named computational model." in descriptions["head"]
    assert "dataset: A named collection of observations." in descriptions["tail"]


@pytest.mark.parametrize("change", ["description", "allowed_head_types", "direction"])
def test_compiler_output_changes_with_ontology_semantics(change):
    from lib.knowledge_graph.extractors.gliner2_schema import compile_schema

    ontology = _ontology()
    original = compile_schema(
        _pinned_schema(), ontology, schema_compiler_version="gliner2-structures-v1"
    ).build()
    changed = deepcopy(ontology)
    changed.relations["uses_dataset"][change] = {
        "description": "A changed relation meaning.",
        "allowed_head_types": ("finding",),
        "direction": "undirected",
    }[change]
    assert (
        original
        != compile_schema(
            _pinned_schema(), changed, schema_compiler_version="gliner2-structures-v1"
        ).build()
    )


def test_compiler_is_deterministic_for_equivalent_mapping_order():
    from lib.knowledge_graph.extractors.gliner2_schema import compile_schema

    ontology = _ontology()
    changed = SimpleNamespace(
        entity_types=dict(reversed(list(ontology.entity_types.items()))),
        relations=dict(reversed(list(ontology.relations.items()))),
    )
    assert (
        compile_schema(
            _pinned_schema(), ontology, schema_compiler_version="gliner2-structures-v1"
        ).build()
        == compile_schema(
            _pinned_schema(), changed, schema_compiler_version="gliner2-structures-v1"
        ).build()
    )


def _endpoint(text, value):
    start = text.index(value)
    return {"text": value, "confidence": 0.9, "start": start, "end": start + len(value)}


@pytest.mark.parametrize(
    "failure",
    ["missing_tail", "subspan_tail", "absent_entity", "missing_section", "role_word"],
)
def test_offline_candidate_keeps_strict_endpoint_and_section_validation(
    monkeypatch, failure
):
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
    schemas = []
    model = SimpleNamespace(create_schema=_pinned_schema)

    def infer(texts, schema, **options):
        schemas.append(schema)
        return [raw]

    model.batch_extract = infer
    monkeypatch.setattr(gliner2_local, "_load_model", lambda _: model)
    backend = gliner2_local.GLiNER2LocalBackend(
        settings=load_extraction_settings({}),
        schema_compiler_version="gliner2-structures-v1",
    )
    result = backend.extract_batch((text,), ontology=_ontology())[0]
    assert result.relations == ()
    assert expected_code in {item.code for item in result.diagnostics}
    assert schemas[0].schema["json_descriptions"]


def test_backend_default_preserves_legacy_relations_task(monkeypatch):
    schema = _pinned_schema()
    model = SimpleNamespace(
        create_schema=lambda: schema,
        batch_extract=lambda *args, **kwargs: [
            {"entities": [{}], "supports": [], "uses_dataset": []}
        ],
    )
    monkeypatch.setattr(gliner2_local, "_load_model", lambda _: model)
    gliner2_local.GLiNER2LocalBackend(
        settings=load_extraction_settings({})
    ).extract_batch(("text",), ontology=_ontology())
    assert schema.schema["relations"]
    assert schema.schema["json_structures"] == []


@pytest.mark.parametrize("independently_emitted", [False, True])
def test_candidate_subspan_needs_its_own_typed_entity(
    monkeypatch, independently_emitted
):
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
    model = SimpleNamespace(
        create_schema=_pinned_schema,
        batch_extract=lambda *args, **kwargs: [raw],
    )
    monkeypatch.setattr(gliner2_local, "_load_model", lambda _: model)
    result = gliner2_local.GLiNER2LocalBackend(
        settings=load_extraction_settings({}),
        schema_compiler_version="gliner2-structures-v1",
    ).extract_batch((text,), ontology=_ontology())[0]
    assert len(result.relations) == int(independently_emitted)
    if independently_emitted:
        assert result.relations[0].tail_text == "Garden"
    else:
        assert result.diagnostics[0].code == "unresolved_relation_endpoint"
