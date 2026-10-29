"""Bounded periodic redelivery for embedding generations whose lease lapsed."""
from celery import shared_task


@shared_task(serializer='json', ignore_result=True)
def recover_embedding_generations(limit=3):
    from apps.documents.services.embedding_index_delivery import recover_due_embedding_generations
    return recover_due_embedding_generations(limit=limit)
