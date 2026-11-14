"""Schema changes preserve old origins and refuse destructive provenance rollback."""
from uuid import uuid4

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

from apps.ingestion.models import IngestionOutput
from .current_adoption_support import adoption_candidate as adoption_candidate, upload as upload
from .current_adoption_support import insert_candidate

pytestmark = pytest.mark.django_db(transaction=True)

PREVIOUS = [("apps_ingestion", "0004_ingestion_output_ledger_v1")]


def previous_model(executor):
    return executor.loader.project_state(PREVIOUS).apps.get_model("apps_ingestion", "IngestionOutput")


def old_values(values):
    return {**{key: value for key, value in values.items() if key != "item"}, "item_id": values["item"].pk}


def test_existing_success_defaults_to_ingestion_without_manufacturing_evidence(adoption_candidate):
    _, _, values, _ = adoption_candidate
    executor = MigrationExecutor(connection)
    leaves = executor.loader.graph.leaf_nodes()
    try:
        executor.migrate(PREVIOUS)
        old = previous_model(executor).objects.create(**old_values(values))
    finally:
        MigrationExecutor(connection).migrate(leaves)
    output = IngestionOutput.objects.get(pk=old.pk)
    assert output.provenance_kind == "ingestion" and output.provenance_evidence == {}
    assert (output.document_id, output.payload, output.document_source_hash) == (
        values["document_id"], values["payload"], values["document_source_hash"])


def test_preexisting_duplicate_successes_stop_migration_without_rewriting(adoption_candidate):
    _, _, values, _ = adoption_candidate
    executor = MigrationExecutor(connection)
    leaves = executor.loader.graph.leaf_nodes()
    old = None
    try:
        executor.migrate(PREVIOUS)
        old = previous_model(executor)
        first = old.objects.create(**old_values(values))
        second = old.objects.create(**{**old_values(values), "output_key": uuid4().hex + uuid4().hex})
        with pytest.raises(RuntimeError, match="1 duplicate owner groups and 1 excess outputs"):
            MigrationExecutor(connection).migrate(leaves)
        assert list(old.objects.order_by("pk").values_list("pk", "document_id")) == [
            (first.pk, values["document_id"]), (second.pk, values["document_id"])]
    finally:
        if old is not None:
            old.objects.all().delete()
        MigrationExecutor(connection).migrate(leaves)


def test_reverse_refuses_to_remove_used_current_adoption_contract(adoption_candidate):
    _, _, values, evidence = adoption_candidate
    output = insert_candidate(values, evidence)
    executor = MigrationExecutor(connection)
    leaves = executor.loader.graph.leaf_nodes()
    try:
        with pytest.raises(RuntimeError, match="provenance must be retained"):
            executor.migrate(PREVIOUS)
    finally:
        MigrationExecutor(connection).migrate(leaves)
    output.refresh_from_db()
    assert output.provenance_kind == "current_adoption"
    assert output.provenance_evidence == evidence
