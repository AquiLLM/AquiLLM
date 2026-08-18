from __future__ import annotations

import importlib
import socket
from math import inf, nan
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml
from django.conf import settings
from django.core.exceptions import ValidationError

ONTOLOGY_PATH = Path(__file__).resolve().parents[1] / "ontologies" / "research-v1.yaml"


def _postgres_available() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _postgres_available(), reason="configured PostgreSQL database is not reachable"
)


def _document() -> dict:
    return {
        "version": "1.0.0",
        "entity_types": [
            {
                "name": "paper",
                "description": "A research paper.",
                "aliases": ["publication"],
                "default_retrieval_weight": 1.0,
                "default_suppression_policy": "never",
                "default_suppression_threshold": 0.0,
            },
            {
                "name": "author",
                "description": "A paper author.",
                "aliases": [],
                "default_retrieval_weight": 0.8,
                "default_suppression_policy": "below_confidence",
                "default_suppression_threshold": 0.4,
            },
        ],
        "relations": [
            {
                "name": "authored_by",
                "description": "Connects a paper to an author.",
                "direction": "directed",
                "allowed_head_types": ["paper"],
                "allowed_tail_types": ["author"],
            }
        ],
    }


def _write(tmp_path: Path, document: dict, name: str = "ontology.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_checked_in_research_ontology_loads_to_provider_neutral_immutable_value():
    from apps.knowledge_graph.services.ontology import load_ontology

    definition = load_ontology(ONTOLOGY_PATH)

    assert definition.version == "1.0.0"
    assert set(definition.entity_types) == {
        "paper",
        "author",
        "institution",
        "method",
        "model",
        "dataset",
        "metric",
        "task",
        "claim",
        "finding",
        "figure",
        "software",
    }
    assert set(definition.relations) == {
        "authored_by",
        "cites",
        "uses_method",
        "uses_model",
        "uses_dataset",
        "evaluates_on",
        "measures_with",
        "reports_metric",
        "supports",
        "contradicts",
        "compares_with",
        "shown_in_figure",
        "implemented_by",
        "affiliated_with",
    }
    assert isinstance(definition.entity_types, MappingProxyType)
    assert isinstance(definition.entity_types["paper"].aliases, tuple)
    assert isinstance(definition.relations["authored_by"].allowed_head_types, tuple)
    with pytest.raises(TypeError):
        definition.entity_types["new"] = object()
    with pytest.raises(AttributeError):
        definition.entity_types["paper"].name = "changed"


def test_checksum_is_canonical_semantic_content_not_yaml_formatting(tmp_path):
    from apps.knowledge_graph.services.ontology import load_ontology

    document = _document()
    compact = tmp_path / "compact.yaml"
    compact.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    reordered = tmp_path / "reordered.yaml"
    reordered.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")

    first = load_ontology(compact)
    second = load_ontology(reordered)

    assert first.checksum == second.checksum
    assert first.canonical_yaml == second.canonical_yaml
    assert first.raw_yaml != ""


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["entity_types"].append(doc["entity_types"][0].copy()),
        lambda doc: doc["entity_types"][1].update(aliases=["publication"]),
        lambda doc: doc["relations"][0].update(allowed_tail_types=["missing"]),
        lambda doc: doc["relations"][0].update(direction="sideways"),
        lambda doc: doc["relations"][0].update(description=""),
        lambda doc: doc["relations"][0].update(allowed_head_types=[]),
        lambda doc: doc["entity_types"][0].update(description=""),
        lambda doc: doc["entity_types"][0].update(default_retrieval_weight=True),
        lambda doc: doc["entity_types"][0].update(default_retrieval_weight=inf),
        lambda doc: doc["entity_types"][0].update(default_retrieval_weight=-0.1),
        lambda doc: doc["entity_types"][0].update(default_suppression_threshold=nan),
        lambda doc: doc["entity_types"][0].update(default_suppression_threshold=True),
        lambda doc: doc["entity_types"][0].update(default_suppression_threshold=1.1),
        lambda doc: doc.update(version="v1"),
    ],
)
def test_loader_rejects_invalid_ontology_semantics(tmp_path, mutate):
    from apps.knowledge_graph.services.ontology import (
        OntologyValidationError,
        load_ontology,
    )

    document = _document()
    mutate(document)

    with pytest.raises(OntologyValidationError):
        load_ontology(_write(tmp_path, document))


@pytest.mark.parametrize(
    "version",
    ["1.0.0-01", "1.0.0-alpha.01", "1.0.0-0.01+build.7"],
)
def test_loader_rejects_semver_prerelease_numeric_identifiers_with_leading_zero(
    tmp_path, version
):
    from apps.knowledge_graph.services.ontology import (
        OntologyValidationError,
        load_ontology,
    )

    document = _document()
    document["version"] = version

    with pytest.raises(OntologyValidationError, match="semantic version"):
        load_ontology(_write(tmp_path, document))


@pytest.mark.parametrize(
    "version",
    ["1١.0.0", "1.1١.0", "1.0.1١", "1.0.0-1١"],
)
def test_loader_rejects_semver_unicode_digits(tmp_path, version):
    from apps.knowledge_graph.services.ontology import (
        OntologyValidationError,
        load_ontology,
    )

    document = _document()
    document["version"] = version

    with pytest.raises(OntologyValidationError, match="semantic version"):
        load_ontology(_write(tmp_path, document))


@pytest.mark.parametrize(
    "version",
    ["1.0.0-0", "1.0.0-01a", "1.0.0-01a.0+build.01", "1.0.0+001"],
)
def test_loader_accepts_valid_semver_prerelease_and_build_identifiers(
    tmp_path, version
):
    from apps.knowledge_graph.services.ontology import load_ontology

    document = _document()
    document["version"] = version

    assert load_ontology(_write(tmp_path, document)).version == version


def test_loader_rejects_cycles_non_string_keys_and_unsupported_yaml(tmp_path):
    from apps.knowledge_graph.services.ontology import (
        OntologyValidationError,
        load_ontology,
    )

    cycle = tmp_path / "cycle.yaml"
    cycle.write_text("node: &node {self: *node}\n", encoding="utf-8")
    non_string_key = tmp_path / "non-string-key.yaml"
    non_string_key.write_text("1: value\n", encoding="utf-8")
    unsupported = tmp_path / "unsupported.yaml"
    unsupported.write_text("value: !!python/tuple [one, two]\n", encoding="utf-8")

    for path in (cycle, non_string_key, unsupported):
        with pytest.raises(OntologyValidationError):
            load_ontology(path)


def test_extension_merge_adds_alias_policy_and_new_relation_without_redefining_core(
    tmp_path,
):
    from apps.knowledge_graph.services.ontology import (
        load_ontology,
        load_ontology_extension,
        merge_ontology_extension,
    )

    extension = {
        "version": "1.1.0",
        "entity_types": [
            {
                "name": "paper",
                "aliases": ["article"],
                "default_retrieval_weight": 0.9,
                "default_suppression_policy": "below_confidence",
                "default_suppression_threshold": 0.2,
            },
            {
                "name": "venue",
                "description": "A publication venue.",
                "aliases": [],
                "default_retrieval_weight": 0.7,
                "default_suppression_policy": "never",
                "default_suppression_threshold": 0.0,
            },
        ],
        "relations": [
            {
                "name": "published_in",
                "description": "Connects a paper to a venue.",
                "direction": "directed",
                "allowed_head_types": ["paper"],
                "allowed_tail_types": ["venue"],
            }
        ],
    }

    merged = merge_ontology_extension(
        load_ontology(_write(tmp_path, _document(), "base.yaml")),
        load_ontology_extension(_write(tmp_path, extension, "extension.yaml")),
    )

    assert merged.version == "1.1.0"
    assert merged.entity_types["paper"].aliases == ("article", "publication")
    assert merged.entity_types["paper"].default_retrieval_weight == 0.9
    assert "venue" in merged.entity_types
    assert "published_in" in merged.relations


def test_extension_merge_rejects_core_redefinition(tmp_path):
    from apps.knowledge_graph.services.ontology import (
        OntologyValidationError,
        load_ontology,
        load_ontology_extension,
        merge_ontology_extension,
    )

    extension = {
        "version": "1.1.0",
        "entity_types": [{"name": "paper", "description": "A different meaning."}],
        "relations": [
            {
                "name": "authored_by",
                "direction": "undirected",
                "allowed_head_types": ["paper"],
                "allowed_tail_types": ["author"],
            }
        ],
    }

    with pytest.raises(OntologyValidationError, match="core"):
        merge_ontology_extension(
            load_ontology(_write(tmp_path, _document(), "base.yaml")),
            load_ontology_extension(_write(tmp_path, extension, "extension.yaml")),
        )


def test_pure_loader_does_not_import_django_model_module(tmp_path, monkeypatch):
    from apps.knowledge_graph.services.ontology import load_ontology

    real_import = importlib.import_module

    def guarded_import(name, package=None):
        if name == "apps.knowledge_graph.models" or name.startswith(
            "apps.knowledge_graph.models."
        ):
            raise AssertionError("loader must not import Django persistence models")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    definition = load_ontology(_write(tmp_path, _document()))

    assert definition.version == "1.0.0"


def test_graph_ontology_activation_uses_one_stable_transaction_lock():
    from apps.knowledge_graph.services.ontology import _lock_graph_ontology_activation

    calls = []

    class Cursor:
        def execute(self, query, parameters):
            calls.append((query, parameters))

    _lock_graph_ontology_activation(Cursor())

    assert calls == [("SELECT pg_advisory_xact_lock(%s)", [707_750_921])]


@pytest.mark.django_db(transaction=True)
@database_required
def test_activation_is_idempotent_conflict_safe_and_supersedes_previous_active(
    tmp_path,
):
    from apps.knowledge_graph.models import OntologyVersion
    from apps.knowledge_graph.services.ontology import (
        OntologyValidationError,
        activate_ontology,
        load_ontology,
    )

    first_definition = load_ontology(_write(tmp_path, _document(), "first.yaml"))
    first = activate_ontology(first_definition)
    assert first.kind == OntologyVersion.Kind.GRAPH
    assert first.version == "1.0.0"
    assert first.checksum == first_definition.checksum
    assert first.metadata["yaml"] == first_definition.raw_yaml
    assert first.status == OntologyVersion.Status.ACTIVE
    assert activate_ontology(first_definition).pk == first.pk

    changed = _document()
    changed["version"] = "1.1.0"
    changed["entity_types"][0]["description"] = "A peer-reviewed research paper."
    second = activate_ontology(load_ontology(_write(tmp_path, changed, "second.yaml")))
    first.refresh_from_db()
    assert first.status == OntologyVersion.Status.SUPERSEDED
    assert second.status == OntologyVersion.Status.ACTIVE

    conflict = _document()
    conflict["entity_types"][0]["description"] = "A changed research paper."
    with pytest.raises(OntologyValidationError, match="checksum"):
        activate_ontology(load_ontology(_write(tmp_path, conflict, "conflict.yaml")))


@pytest.mark.django_db(transaction=True)
@database_required
def test_activated_ontology_version_content_is_immutable(tmp_path):
    from apps.knowledge_graph.services.ontology import activate_ontology, load_ontology

    record = activate_ontology(load_ontology(_write(tmp_path, _document())))
    record.checksum = "f" * 64

    with pytest.raises(ValidationError, match="immutable"):
        record.save()
