"""Offline reproductions for the 2026-09-21 collection-schema audit.

Run from the repository root:
    rtk proxy python artifacts/audits/2026-09-21-knowledge-graph/schema_probes.py

Each probe runs in its own subprocess. Application functions are real; database
managers and transaction contexts are replaced with in-memory boundaries. No
database, broker, model download, inference service, or production secret is used.
These are reproductions of current behavior, not proposed regression assertions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "aquillm"))


def live_lease_redelivery_is_acknowledged():
    """A replacement delivery during the lost worker's lease schedules no retry."""
    import uuid
    from contextlib import nullcontext
    from datetime import timedelta
    from types import ModuleType, SimpleNamespace

    from django.conf import settings

    settings.configure(KG_EXTRACTION_QUEUE="kg-audit", USE_TZ=True)
    from apps.collections.tasks import schema_generation as tasks

    run = SimpleNamespace(
        status="running",
        lease_expires_at=tasks.timezone.now() + timedelta(minutes=9),
    )
    manager = SimpleNamespace()
    manager.select_for_update = lambda: manager
    manager.filter = lambda **kwargs: manager
    manager.first = lambda: run
    models = ModuleType("apps.collections.models")
    models.CollectionSchemaGenerationRun = SimpleNamespace(objects=manager)
    sys.modules[models.__name__] = models
    tasks.transaction.atomic = nullcontext
    retries = []
    tasks.generate_collection_schema_task.retry = lambda **kwargs: retries.append(kwargs)

    result = tasks.generate_collection_schema_task.run(str(uuid.uuid4()))
    assert result is None
    assert run.status == "running"
    assert retries == []
    print("REPRODUCED: live-lease redelivery returns normally; run remains running; zero retries")


def _ontology_document(entity_name="person", relation_name="related_to"):
    return {
        "version": "1.0.0",
        "entity_types": [{
            "name": entity_name,
            "description": "An audit entity.",
            "aliases": [],
            "default_retrieval_weight": 0.5,
            "default_suppression_policy": "none",
            "default_suppression_threshold": 0,
        }],
        "relations": [{
            "name": relation_name,
            "description": "An audit relation.",
            "direction": "directed",
            "allowed_head_types": [entity_name],
            "allowed_tail_types": [entity_name],
        }],
    }


def overlong_type_name_passes_publish_validator():
    """The same YAML validator used at publish accepts a 129-character type."""
    import ast
    import yaml

    from apps.knowledge_graph.services.ontology import load_ontology_yaml

    def model_field_limit(relative_path, model_class, field_name):
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == model_class)
        assignment = next(
            node for node in cls.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == field_name for target in node.targets)
        )
        return next(ast.literal_eval(keyword.value) for keyword in assignment.value.keywords if keyword.arg == "max_length")

    cases = (
        ("entity", "aquillm/apps/knowledge_graph/models/entities.py", "EntityMention", "entity_type"),
        ("relation", "aquillm/apps/knowledge_graph/models/relations.py", "RelationMention", "relation_type"),
    )
    for kind, source, cls, field in cases:
        limit = model_field_limit(source, cls, field)
        name = "x" * (limit + 1)
        document = _ontology_document(**{f"{kind}_name": name})
        ontology = load_ontology_yaml(yaml.safe_dump(document))
        names = ontology.entity_types if kind == "entity" else ontology.relations
        assert name in names
        print(f"REPRODUCED: {kind} name length {len(name)} validates; actual model field limit is {limit}")


def stale_revision_overwrites_replacement_draft():
    """A replacement UUID at revision 1 accepts the previous draft's revision."""
    import uuid
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace

    old_draft_id = uuid.uuid4()
    draft = SimpleNamespace(
        pk=uuid.uuid4(),
        revision=1,
        definitions={
            "entities": [{"key": "paper", "values": {"name": "paper", "description": "manager restored this"}}],
            "relations": [],
        },
        save=lambda **kwargs: None,
    )
    manager = SimpleNamespace()
    manager.select_for_update = lambda: manager
    manager.filter = lambda **kwargs: manager
    manager.first = lambda: draft
    collection_manager = SimpleNamespace()
    collection_manager.select_for_update = lambda: collection_manager
    collection_manager.get = lambda **kwargs: None
    models = ModuleType("apps.collections.models")
    models.Collection = SimpleNamespace(objects=collection_manager)
    models.CollectionSchemaDraft = SimpleNamespace(objects=manager)
    models.CollectionSchemaVersion = SimpleNamespace()
    sys.modules[models.__name__] = models
    from apps.collections.services import schema

    schema.transaction.atomic = nullcontext
    assert old_draft_id != draft.pk
    # The production mutation signature offers no draft UUID argument.
    result = schema.mutate_definition(
        SimpleNamespace(pk=1), object(), "entity", "paper", 1,
        {"description": "stale tab overwrite"},
    )
    assert result.pk == draft.pk
    assert result.revision == 2
    assert result.definitions["entities"][0]["values"]["description"] == "stale tab overwrite"
    print("REPRODUCED: stale revision 1 overwrites replacement draft UUID and advances it to revision 2")


def reserved_relation_is_silently_discarded():
    """The provider adapter silently discards the valid relation name entities."""
    import yaml

    from apps.knowledge_graph.services.ontology import load_ontology_yaml
    from lib.knowledge_graph.extractors.gliner2_local import _normalize_relations
    from lib.knowledge_graph.types import EntityCandidate

    ontology = load_ontology_yaml(yaml.safe_dump(_ontology_document(relation_name="entities")))
    entity = EntityCandidate(text="Alice", entity_type="person", confidence=0.9, start=0, end=5)
    result = _normalize_relations(
        {"entities": [{"head": "Alice", "tail": "Alice", "confidence": 0.9}]},
        text="Alice", input_index=0, ontology_relations=ontology.relations, entities=[entity],
    )
    assert "entities" in ontology.relations
    assert result == ([], [])
    print("REPRODUCED: accepted relation entities produces zero relations and zero diagnostics")


PROBES = (
    live_lease_redelivery_is_acknowledged,
    overlong_type_name_passes_publish_validator,
    stale_revision_overwrites_replacement_draft,
    reserved_relation_is_silently_discarded,
)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        selected = next(probe for probe in PROBES if probe.__name__ == sys.argv[1])
        selected()
    else:
        for probe in PROBES:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), probe.__name__], check=True)
        print("4/4 offline schema audit probes reproduced the reported behavior")
