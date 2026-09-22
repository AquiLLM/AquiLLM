# ruff: noqa: F401
"""Related graph infrastructure regression scenarios."""

from lib.knowledge_graph.tests.test_gliner2_local import (
    ExtractionBackendError,
    Path,
    RelationCandidate,
    _backend,
    _clear_process_model_cache,
    _entity_result,
    _install_fake_provider,
    _ontology,
    _spanned_relation,
    gliner2_local,
    pytest,
    sys,
)


def test_disallowed_endpoint_type_is_diagnostic_not_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(
        monkeypatch,
        entity_results=[
            _entity_result(
                ("metric", "Qwen3", 0.95, 0, 5),
                ("dataset", "MMLU", 0.91, 11, 15),
            )
        ],
        relation_results=[
            _spanned_relation(
                "uses_dataset", ("Qwen3", 0.8, 0, 5), ("MMLU", 0.7, 11, 15)
            )
        ],
    )

    result = _backend().extract_batch(("Qwen3 uses MMLU.",), ontology=_ontology())[0]

    assert result.relations == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "disallowed_relation_endpoint"
    ]
    assert ("endpoint", "head") in result.diagnostics[0].details



def test_undirected_asymmetric_relation_accepts_either_provider_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ontology = _ontology()
    ontology.relations["uses_dataset"].direction = "undirected"
    text = "Qwen3 uses MMLU."
    entities = _entity_result(
        ("model", "Qwen3", 0.95, 0, 5),
        ("dataset", "MMLU", 0.91, 11, 15),
    )
    _install_fake_provider(
        monkeypatch,
        entity_results=[entities, entities],
        relation_results=[
            _spanned_relation(
                "uses_dataset", ("Qwen3", 0.8, 0, 5), ("MMLU", 0.7, 11, 15)
            ),
            _spanned_relation(
                "uses_dataset", ("MMLU", 0.7, 11, 15), ("Qwen3", 0.8, 0, 5)
            ),
        ],
    )

    forward, reverse = _backend().extract_batch((text, text), ontology=ontology)

    assert forward.relations == (
        RelationCandidate("uses_dataset", "Qwen3", "MMLU", 0, 5, 11, 15, 0.7),
    )
    assert reverse.relations == (
        RelationCandidate("uses_dataset", "MMLU", "Qwen3", 11, 15, 0, 5, 0.7),
    )
    assert forward.diagnostics == reverse.diagnostics == ()



def test_directed_asymmetric_relation_rejects_reverse_provider_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(
        monkeypatch,
        entity_results=[
            _entity_result(
                ("model", "Qwen3", 0.95, 0, 5),
                ("dataset", "MMLU", 0.91, 11, 15),
            )
        ],
        relation_results=[
            _spanned_relation(
                "uses_dataset", ("MMLU", 0.7, 11, 15), ("Qwen3", 0.8, 0, 5)
            )
        ],
    )

    result = _backend().extract_batch(("Qwen3 uses MMLU.",), ontology=_ontology())[0]

    assert result.relations == ()
    assert [item.code for item in result.diagnostics] == [
        "disallowed_relation_endpoint"
    ]



def test_undirected_reverse_orientation_preserves_grounding_and_ambiguity_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ontology = _ontology()
    ontology.relations["uses_dataset"].direction = "undirected"
    _install_fake_provider(
        monkeypatch,
        entity_results=[
            _entity_result(
                ("dataset", "MMLU", 0.91, 0, 4),
                ("model", "Qwen3", 0.95, 11, 16),
                ("paper", "Qwen3", 0.90, 11, 16),
            )
        ],
        relation_results=[
            {
                "uses_dataset": [
                    {
                        "head": {"text": "MMLU", "confidence": 0.7},
                        "tail": {"text": "Qwen3", "confidence": 0.8},
                    }
                ]
            }
        ],
    )

    result = _backend().extract_batch(("MMLU links Qwen3",), ontology=ontology)[0]

    assert result.relations == ()
    assert [item.code for item in result.diagnostics] == ["ambiguous_relation_endpoint"]



@pytest.mark.parametrize("ambiguous_forward", [False, True])
def test_undirected_rejects_distinct_typed_pairs_across_orientations(
    monkeypatch, ambiguous_forward
):
    ontology = _ontology()
    ontology.relations["uses_dataset"].direction = "undirected"
    entities = [
        ("model", "Qwen3", 0.95, 0, 5),
        ("dataset", "Qwen3", 0.95, 0, 5),
        ("dataset", "MMLU", 0.91, 11, 15),
        ("model", "MMLU", 0.91, 11, 15),
    ]
    if ambiguous_forward:
        entities.append(("paper", "Qwen3", 0.90, 0, 5))
    _install_fake_provider(
        monkeypatch,
        entity_results=[_entity_result(*entities)],
        relation_results=[
            _spanned_relation(
                "uses_dataset", ("Qwen3", 0.8, 0, 5), ("MMLU", 0.7, 11, 15)
            )
        ],
    )
    result = _backend().extract_batch(("Qwen3 uses MMLU.",), ontology=ontology)[0]
    assert result.relations == ()
    assert [item.code for item in result.diagnostics] == ["ambiguous_relation_endpoint"]



def test_undirected_overlapping_type_rules_count_identical_pair_once(monkeypatch):
    ontology = _ontology()
    definition = ontology.relations["uses_dataset"]
    definition.direction = "undirected"
    definition.allowed_head_types = definition.allowed_tail_types = ("model", "dataset")
    _install_fake_provider(
        monkeypatch,
        entity_results=[
            _entity_result(
                ("model", "Qwen3", 0.95, 0, 5), ("dataset", "MMLU", 0.91, 11, 15)
            )
        ],
        relation_results=[
            _spanned_relation(
                "uses_dataset", ("Qwen3", 0.8, 0, 5), ("MMLU", 0.7, 11, 15)
            )
        ],
    )
    result = _backend().extract_batch(("Qwen3 uses MMLU.",), ontology=ontology)[0]
    assert len(result.relations) == 1
    assert result.diagnostics == ()



def test_undirected_swapped_typed_pair_at_same_span_counts_once(monkeypatch):
    ontology = _ontology()
    ontology.relations["uses_dataset"].direction = "undirected"
    _install_fake_provider(
        monkeypatch,
        entity_results=[
            _entity_result(
                ("model", "Qwen3", 0.95, 0, 5), ("dataset", "Qwen3", 0.91, 0, 5)
            )
        ],
        relation_results=[
            _spanned_relation(
                "uses_dataset", ("Qwen3", 0.8, 0, 5), ("Qwen3", 0.7, 0, 5)
            )
        ],
    )
    result = _backend().extract_batch(("Qwen3",), ontology=ontology)[0]
    assert len(result.relations) == 1
    assert result.diagnostics == ()



@pytest.mark.parametrize("stage", ["load", "inference"])
def test_provider_failures_are_wrapped_without_django(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    provider_error = RuntimeError(f"{stage} failed")
    _install_fake_provider(
        monkeypatch,
        load_error=provider_error if stage == "load" else None,
        inference_error=provider_error if stage == "inference" else None,
    )

    with pytest.raises(ExtractionBackendError, match=stage) as captured:
        _backend().extract_batch(("A paper.",), ontology=_ontology())

    assert captured.value.__cause__ is provider_error



def test_rejects_a_different_gliner2_runtime_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    sys.modules["gliner2"].__version__ = "1.3.1"  # type: ignore[attr-defined]

    with pytest.raises(ExtractionBackendError, match="1.3.2"):
        _backend().extract_batch(("A paper.",), ontology=_ontology())



def test_backend_module_has_no_top_level_provider_or_framework_imports() -> None:
    source = Path(gliner2_local.__file__).read_text(encoding="utf-8")

    assert "from gliner2 import" not in source.partition("def _load_model")[0]
    assert "from huggingface_hub import" not in source.partition("def _load_model")[0]
    assert "django" not in source.lower()
