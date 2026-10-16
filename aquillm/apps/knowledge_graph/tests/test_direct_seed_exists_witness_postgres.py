"""Adversarial single-witness parity against the independent old query."""

import os
from dataclasses import replace
from uuid import uuid4

import pytest
from django.db import connection

from apps.knowledge_graph.tests.test_direct_seed_exists_postgres import _options, _same

pytest_plugins = ["apps.knowledge_graph.tests.test_direct_seed_sql_postgres"]

pytestmark = [
    pytest.mark.django_db,
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="explicit PostgreSQL integration opt-in required",
    ),
]


def _set(model, pk, **fields):
    # Deliberate corruption fixtures bypass append-closed model validation, but
    # still obey database FK/check/unique constraints and roll back with pytest.
    table = connection.ops.quote_name(model._meta.db_table)
    assignments = ", ".join(
        f"{connection.ops.quote_name(name)} = %s" for name in fields
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE {table} SET {assignments} WHERE id = %s", [*fields.values(), pk]
        )


def test_duplicate_valid_links_do_not_duplicate_or_overflow(scoped_aliases):
    from apps.knowledge_graph.models import CollectionEntityDocumentLink
    from apps.knowledge_graph.retrieval import direct_seed_queries
    from apps.knowledge_graph.tests import (
        _direct_seed_queries_head318_oracle as old_queries,
    )

    _, _, entities, _ = scoped_aliases
    candidate = CollectionEntityDocumentLink.objects.get(collection_entity=entities[2])
    _set(
        CollectionEntityDocumentLink,
        candidate.pk,
        collection_entity_id=entities[0].pk,
        status="active",
        outcome="automatic",
        candidate_rank=None,
    )
    _same(_options(scoped_aliases), [entities[0].pk, entities[1].pk])
    _same(_options(scoped_aliases, limit=2), [entities[0].pk, entities[1].pk])
    for loader in (
        old_queries._load_candidate_rows,
        direct_seed_queries._load_candidate_rows,
    ):
        with pytest.raises(ValueError, match="hard cap"):
            loader(**_options(scoped_aliases, limit=1))


def test_alias_on_excluded_link_cannot_borrow_authorized_link(scoped_aliases):
    from apps.knowledge_graph.models import CollectionEntityDocumentLink, EntityMention

    _, _, entities, assignments = scoped_aliases
    candidate = CollectionEntityDocumentLink.objects.get(collection_entity=entities[2])
    _set(
        CollectionEntityDocumentLink, candidate.pk, collection_entity_id=entities[0].pk
    )
    _set(EntityMention, assignments[0].mention_id, normalized_text="other")
    _same(_options(scoped_aliases), [entities[1].pk])


def test_alias_cannot_borrow_from_other_document_with_wrong_manifest(scoped_aliases):
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntityDocumentLink,
        DocumentEntity,
        EntityMention,
    )

    scope, _, entities, assignments = scoped_aliases
    other_document = uuid4()
    candidate = CollectionEntityDocumentLink.objects.get(collection_entity=entities[2])
    _set(
        CollectionEntityDocumentLink,
        candidate.pk,
        collection_entity_id=entities[0].pk,
        status="active",
        outcome="automatic",
        candidate_rank=None,
    )
    _set(DocumentEntity, candidate.document_entity_id, document_id=other_document)
    _set(EntityMention, assignments[2].mention_id, document_id=other_document)
    _set(EntityMention, assignments[0].mention_id, normalized_text="other")
    expanded = replace(
        scope,
        selected_document_ids=tuple(
            sorted(
                {
                    *scope.selected_document_ids,
                    other_document,
                },
                key=str,
            )
        ),
    )
    manifest = candidate.manifest_input
    original_document = manifest.document_id
    _set(CollectionArtifactInput, manifest.pk, document_id=other_document)
    _same(_options(scoped_aliases, scope=expanded), [entities[0].pk])
    _set(CollectionArtifactInput, manifest.pk, document_id=original_document)
    _same(_options(scoped_aliases, scope=expanded), [entities[1].pk])


@pytest.mark.parametrize("bad_field", ["status", "type", "resolver"])
def test_split_assignments_cannot_combine_text_and_authority(scoped_aliases, bad_field):
    from apps.knowledge_graph.models import DocumentEntityMention, EntityMention

    _, _, entities, assignments = scoped_aliases
    _set(
        DocumentEntityMention,
        assignments[2].pk,
        document_entity_id=assignments[0].document_entity_id,
    )
    _set(EntityMention, assignments[2].mention_id, normalized_text="other")
    if bad_field == "status":
        _set(DocumentEntityMention, assignments[0].pk, status="superseded")
    elif bad_field == "type":
        _set(EntityMention, assignments[0].mention_id, entity_type="other")
    else:
        _set(DocumentEntityMention, assignments[0].pk, resolver_version="other")
    _same(_options(scoped_aliases), [entities[1].pk])


@pytest.mark.parametrize(
    "field", ["artifact_id", "collection_id", "document_id", "document_artifact_id"]
)
def test_manifest_pairing_is_required(scoped_aliases, field):
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntityDocumentLink,
    )

    scope, _, _, _ = scoped_aliases
    manifest = CollectionEntityDocumentLink.objects.first().manifest_input
    if field == "artifact_id":
        replacement = scope.selected_document_artifact_ids[0]
    elif field == "collection_id":
        replacement = Collection.objects.create(name="different collection").pk
    elif field == "document_id":
        replacement = uuid4()
    else:
        replacement = scope.selected_artifact_ids[0]
    _set(CollectionArtifactInput, manifest.pk, **{field: replacement})
    if field == "document_artifact_id":
        scope = replace(
            scope,
            selected_document_artifact_ids=tuple(
                sorted(
                    {
                        *scope.selected_document_artifact_ids,
                        replacement,
                    }
                )
            ),
        )
    _same(_options(scoped_aliases, scope=scope), [])


def test_document_artifact_superseded_and_independent_resolver_remain_eligible(
    scoped_aliases,
):
    from apps.knowledge_graph.models import GraphArtifact

    scope, _, entities, _ = scoped_aliases
    source = GraphArtifact.objects.get(pk=scope.selected_document_artifact_ids[0])
    assert source.resolver_version != scope.resolver_version
    _set(GraphArtifact, source.pk, status="superseded")
    _same(_options(scoped_aliases), [entities[0].pk, entities[1].pk])


def test_stale_assignment_document_resolver_rejects(scoped_aliases):
    from apps.knowledge_graph.models import DocumentEntityMention

    _, _, entities, assignments = scoped_aliases
    _set(
        DocumentEntityMention,
        assignments[0].pk,
        resolver_version="wrong-document-version",
    )
    _same(_options(scoped_aliases), [entities[1].pk])


def test_no_extra_link_artifact_equality(scoped_aliases):
    from apps.knowledge_graph.models import CollectionEntityDocumentLink

    scope, _, entities, _ = scoped_aliases
    link = CollectionEntityDocumentLink.objects.get(collection_entity=entities[0])
    other_artifact = scope.selected_document_artifact_ids[0]
    _set(CollectionEntityDocumentLink, link.pk, artifact_id=other_artifact)
    expanded = replace(
        scope,
        selected_artifact_ids=tuple(
            sorted({*scope.selected_artifact_ids, other_artifact})
        ),
        generation_keys_by_artifact=tuple(
            sorted((*scope.generation_keys_by_artifact, (other_artifact, "3" * 64)))
        ),
        generation_ids_by_artifact=tuple(
            sorted((*scope.generation_ids_by_artifact, (other_artifact, uuid4())))
        ),
    )
    _same(_options(scoped_aliases, scope=expanded), [entities[0].pk, entities[1].pk])


def test_no_extra_mention_document_or_artifact_equality(scoped_aliases):
    from apps.knowledge_graph.models import EntityMention

    scope, _, entities, assignments = scoped_aliases
    other_document = uuid4()
    other_artifact = scope.selected_artifact_ids[0]
    _set(
        EntityMention,
        assignments[0].mention_id,
        document_id=other_document,
        artifact_id=other_artifact,
    )
    expanded = replace(
        scope,
        selected_document_ids=tuple(
            sorted({*scope.selected_document_ids, other_document}, key=str)
        ),
        selected_document_artifact_ids=tuple(
            sorted({*scope.selected_document_artifact_ids, other_artifact})
        ),
    )
    _same(_options(scoped_aliases, scope=expanded), [entities[0].pk, entities[1].pk])
