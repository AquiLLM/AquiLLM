"""Contract tests for provider-neutral knowledge graph value objects."""

import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import UUID

import pytest

from apps.knowledge_graph.retrieval.types import (
    GraphExpansionDiagnostics,
    GraphExpansionRequest,
    GraphExpansionResult,
)
from lib.knowledge_graph.types import (
    EntityCandidate,
    ExtractionBatchResult,
    ExtractionDiagnostic,
    RelationCandidate,
)


def _entity() -> EntityCandidate:
    return EntityCandidate(
        entity_type="model",
        text="Qwen3",
        start=14,
        end=19,
        confidence=0.94,
    )


def _relation() -> RelationCandidate:
    return RelationCandidate(
        relation_type="evaluates_on",
        head_text="Qwen3",
        tail_text="MMLU",
        head_start=14,
        head_end=19,
        tail_start=33,
        tail_end=37,
        confidence=0.88,
    )


def _diagnostic() -> ExtractionDiagnostic:
    return ExtractionDiagnostic(
        code="ambiguous_relation_endpoint",
        candidate_kind="relation",
        input_index=0,
        details=(("relation_type", "evaluates_on"), ("head_text", "Qwen3")),
    )


def _batch() -> ExtractionBatchResult:
    return ExtractionBatchResult(
        entities=(_entity(),), relations=(_relation(),), diagnostics=(_diagnostic(),)
    )


def _request() -> GraphExpansionRequest:
    return GraphExpansionRequest(
        query="Which model uses MMLU?",
        seed_chunk_ids=(1,),
        allowed_doc_ids=(UUID("11111111-1111-4111-8111-111111111111"),),
        allowed_collection_ids=(UUID("22222222-2222-4222-8222-222222222222"),),
    )


def _result() -> GraphExpansionResult:
    return GraphExpansionResult(
        chunk_ids=(2,), diagnostics=GraphExpansionDiagnostics(status="hit")
    )


def test_extraction_contracts_are_equal_value_objects() -> None:
    entity = _entity()
    relation = _relation()
    diagnostic = _diagnostic()
    batch = _batch()

    assert entity == _entity()
    assert relation == _relation()
    assert diagnostic == _diagnostic()
    assert batch == ExtractionBatchResult(
        entities=(_entity(),), relations=(_relation(),), diagnostics=(_diagnostic(),)
    )


@pytest.mark.parametrize(
    ("instance", "field_name", "replacement"),
    [
        (_entity(), "text", "other"),
        (_relation(), "head_text", "other"),
        (_diagnostic(), "code", "other"),
        (_batch(), "entities", ()),
        (_request(), "query", "other"),
        (GraphExpansionDiagnostics(status="hit"), "status", "miss"),
        (_result(), "chunk_ids", (3,)),
    ],
)
def test_contract_dataclasses_are_frozen_and_slotted(
    instance: object, field_name: str, replacement: object
) -> None:
    assert not hasattr(instance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field_name, replacement)


@pytest.mark.parametrize(
    "module_name",
    ["lib.knowledge_graph.types", "apps.knowledge_graph.retrieval.types"],
)
def test_contract_type_modules_import_without_framework_or_provider_dependencies(
    module_name: str,
) -> None:
    source_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not existing_python_path
        else f"{source_root}{os.pathsep}{existing_python_path}"
    )
    import_script = """
import importlib
import importlib.abc
import sys

blocked_prefixes = (
    "django", "sqlalchemy", "tortoise", "peewee", "orm", "apps.documents",
    "aquillm.models", "gliner", "gliner2", "openai", "anthropic", "cohere",
    "google", "providers",
)

class BlockedImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked_prefixes or fullname.startswith(
            tuple(prefix + "." for prefix in blocked_prefixes)
        ):
            raise ImportError("blocked import: " + fullname)
        return None

sys.meta_path.insert(0, BlockedImportFinder())
importlib.import_module(sys.argv[1])
"""

    completed = subprocess.run(
        [sys.executable, "-c", import_script, module_name],
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"start": -1}, "start"),
        ({"start": 19, "end": 19}, "end"),
        ({"confidence": -0.01}, "confidence"),
        ({"confidence": 1.01}, "confidence"),
        ({"confidence": math.nan}, "confidence"),
        ({"entity_type": " "}, "entity_type"),
        ({"text": ""}, "text"),
    ],
)
def test_entity_candidate_rejects_invalid_fields(kwargs: dict[str, object], field: str) -> None:
    values: dict[str, object] = {
        "entity_type": "model",
        "text": "Qwen3",
        "start": 14,
        "end": 19,
        "confidence": 0.94,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=field):
        EntityCandidate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"head_start": -1}, "head_start"),
        ({"head_start": 19, "head_end": 19}, "head_end"),
        ({"tail_start": -1}, "tail_start"),
        ({"tail_start": 37, "tail_end": 37}, "tail_end"),
        ({"confidence": math.inf}, "confidence"),
        ({"relation_type": ""}, "relation_type"),
        ({"head_text": " "}, "head_text"),
        ({"tail_text": ""}, "tail_text"),
    ],
)
def test_relation_candidate_rejects_invalid_fields(kwargs: dict[str, object], field: str) -> None:
    values: dict[str, object] = {
        "relation_type": "evaluates_on",
        "head_text": "Qwen3",
        "tail_text": "MMLU",
        "head_start": 14,
        "head_end": 19,
        "tail_start": 33,
        "tail_end": 37,
        "confidence": 0.88,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=field):
        RelationCandidate(**values)  # type: ignore[arg-type]


def test_extraction_diagnostic_is_provider_neutral_and_validates_details() -> None:
    diagnostic = _diagnostic()

    assert diagnostic.details == (("relation_type", "evaluates_on"), ("head_text", "Qwen3"))
    assert isinstance(diagnostic.details, tuple)
    with pytest.raises(ValueError, match="details"):
        ExtractionDiagnostic("invalid", "relation", 0, [("key", "value")])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="details"):
        ExtractionDiagnostic("invalid", "relation", 0, (["key", "value"],))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="details"):
        ExtractionDiagnostic("invalid", "relation", 0, (("key", {"value": "mutable"}),))
    with pytest.raises(ValueError, match="input_index"):
        ExtractionDiagnostic("invalid", "relation", -1, ())


def test_extraction_batch_requires_tuples_of_contract_candidates() -> None:
    with pytest.raises(ValueError, match="entities"):
        ExtractionBatchResult(entities=[_entity()], relations=(), diagnostics=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="relations"):
        ExtractionBatchResult(entities=(), relations=[_relation()], diagnostics=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="diagnostics"):
        ExtractionBatchResult(entities=(), relations=(), diagnostics=[_diagnostic()])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entities"):
        ExtractionBatchResult(entities=("not an entity",), relations=(), diagnostics=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="relations"):
        ExtractionBatchResult(entities=(), relations=("not a relation",), diagnostics=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="diagnostics"):
        ExtractionBatchResult(entities=(), relations=(), diagnostics=("not a diagnostic",))  # type: ignore[arg-type]


def test_graph_expansion_contracts_are_equal_immutable_value_objects() -> None:
    document_id = UUID("11111111-1111-4111-8111-111111111111")
    collection_id = UUID("22222222-2222-4222-8222-222222222222")
    request = GraphExpansionRequest(
        query="Which model uses MMLU?",
        seed_chunk_ids=(1,),
        allowed_doc_ids=(document_id,),
        allowed_collection_ids=(collection_id,),
    )
    result = GraphExpansionResult(
        chunk_ids=(2,), diagnostics=GraphExpansionDiagnostics(status="hit")
    )

    assert request == GraphExpansionRequest(
        query="Which model uses MMLU?",
        seed_chunk_ids=(1,),
        allowed_doc_ids=(document_id,),
        allowed_collection_ids=(collection_id,),
    )
    assert result == GraphExpansionResult(
        chunk_ids=(2,), diagnostics=GraphExpansionDiagnostics(status="hit")
    )
    with pytest.raises(FrozenInstanceError):
        request.query = "other"  # type: ignore[misc]


@pytest.mark.parametrize("status", ["disabled", "miss", "hit", "timeout", "error"])
def test_graph_expansion_diagnostics_supports_safe_statuses(status: str) -> None:
    diagnostics = GraphExpansionDiagnostics(
        status=status,
        candidate_count=2,
        elapsed_ms=12.5,
        version_signature="graph-v1",
    )

    assert diagnostics.status == status
    assert diagnostics.candidate_count == 2


def test_graph_expansion_contracts_reject_invalid_or_mutable_boundaries() -> None:
    document_id = UUID("11111111-1111-4111-8111-111111111111")
    collection_id = UUID("22222222-2222-4222-8222-222222222222")
    valid = {
        "query": "Which model uses MMLU?",
        "seed_chunk_ids": (1,),
        "allowed_doc_ids": (document_id,),
        "allowed_collection_ids": (collection_id,),
    }

    for field, value in (
        ("seed_chunk_ids", []),
        ("allowed_doc_ids", []),
        ("allowed_collection_ids", []),
        ("seed_chunk_ids", ()),
        ("allowed_doc_ids", ()),
        ("allowed_collection_ids", ()),
        ("seed_chunk_ids", (0,)),
        ("seed_chunk_ids", (True,)),
        ("allowed_doc_ids", (str(document_id),)),
        ("allowed_collection_ids", (str(collection_id),)),
    ):
        values = dict(valid)
        values[field] = value
        with pytest.raises(ValueError, match=field):
            GraphExpansionRequest(**values)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="chunk_ids"):
        GraphExpansionResult(chunk_ids=[2], diagnostics=GraphExpansionDiagnostics(status="hit"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chunk_ids"):
        GraphExpansionResult(chunk_ids=(0,), diagnostics=GraphExpansionDiagnostics(status="hit"))
    with pytest.raises(ValueError, match="chunk_ids"):
        GraphExpansionResult(chunk_ids=(True,), diagnostics=GraphExpansionDiagnostics(status="hit"))
    with pytest.raises(ValueError, match="chunk_ids"):
        GraphExpansionResult(chunk_ids=("2",), diagnostics=GraphExpansionDiagnostics(status="hit"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="diagnostics"):
        GraphExpansionResult(chunk_ids=(2,), diagnostics="not diagnostics")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="status"):
        GraphExpansionDiagnostics(status="unknown")
    with pytest.raises(ValueError, match="candidate_count"):
        GraphExpansionDiagnostics(status="hit", candidate_count=-1)
    with pytest.raises(ValueError, match="elapsed_ms"):
        GraphExpansionDiagnostics(status="hit", elapsed_ms=math.inf)
