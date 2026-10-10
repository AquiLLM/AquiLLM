"""Compatibility facade for durable, independently checkpointed ingestion."""
from apps.ingestion.services.output_manifest import prepare_output_manifest
from apps.ingestion.services.output_ledger import claim_output, publish_output


def run_ingest_uploaded_file(item_id: int) -> None:
    for key in prepare_output_manifest(item_id, using='default'):
        lease = claim_output(key, using='default')
        if lease is not None:
            publish_output(lease, using='default')


__all__ = ['run_ingest_uploaded_file']
