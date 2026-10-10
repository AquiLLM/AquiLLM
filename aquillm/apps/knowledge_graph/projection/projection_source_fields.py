"""Stable projection source field contract shared by the ORM loader."""

PURPOSE_STATES = {
    "build": ("building",),
    "audit": ("ready",),
    "prune": ("failed", "superseded"),
}
ARTIFACT_FIELDS = (
    "id scope_type scope_id collection_scope_id rebuild_request_id evaluation_only "
    "build_key build_generation orchestration_version source_hash ontology_version "
    "ontology_checksum extractor_version resolver_version resolution_config_checksum "
    "filter_policy_version filter_policy_checksum embedding_model_signature "
    "assembly_version assembly_config_checksum"
).split()


def artifact_source_row(row: dict) -> dict:
    return {
        ("collection_id" if key == "collection_scope_id" else key): value
        for key, value in row.items()
    }
