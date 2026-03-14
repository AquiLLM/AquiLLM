from celery import shared_task


@shared_task(serializer="json")
def create_conversation_memories_task(conversation_id: int) -> None:
    from .models import WSConversation
    from .memory import create_episodic_memories_for_conversation

    convo = WSConversation.objects.filter(id=conversation_id).first()
    if convo is None:
        return
    create_episodic_memories_for_conversation(convo)
