"""JSON-only durable ingestion deliveries; leases carry no parser bytes."""
from uuid import UUID
from celery import shared_task


@shared_task(serializer='json', ignore_result=True)
def publish_ingestion_output(output_id, generation, attempt, *, using='default'):
    from apps.ingestion.services.output_ledger import OutputLeaseV1, publish_output
    return publish_output(OutputLeaseV1(output_id, UUID(generation), attempt), using=using).status


@shared_task(serializer='json', ignore_result=True)
def recover_ingestion_outputs(limit=25):
    from apps.ingestion.services.output_dispatch import recover_due_outputs
    return recover_due_outputs(limit=limit, using='default')
