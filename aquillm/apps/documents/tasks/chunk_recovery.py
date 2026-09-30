"""Bounded periodic recovery for lost or failed chunk task publication."""
from celery import shared_task


@shared_task(serializer='json', ignore_result=True)
def recover_chunk_publications(limit=25):
    from apps.documents.services.chunk_publication import recover_due_chunk_publications
    return recover_due_chunk_publications(limit=limit)
