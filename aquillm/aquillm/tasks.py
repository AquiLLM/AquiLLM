from celery import shared_task


@shared_task(serializer="json")
def create_conversation_memories_task(conversation_id: int) -> None:
    from .models import WSConversation
    from .memory import create_episodic_memories_for_conversation

    convo = WSConversation.objects.filter(id=conversation_id).first()
    if convo is None:
        return
    create_episodic_memories_for_conversation(convo)


@shared_task(serializer="json", queue="memory-promotion")
def promote_profile_facts_task(
    user_id: int,
    user_content: str,
    assistant_content: str,
) -> None:
    from .memory import promote_profile_facts_for_turn

    promote_profile_facts_for_turn(
        user_id=user_id,
        user_content=user_content,
        assistant_content=assistant_content,
    )


@shared_task(serializer="json")
def ingest_uploaded_file_task(item_id: int) -> None:
    from .task_ingest_uploaded import run_ingest_uploaded_file

    run_ingest_uploaded_file(item_id)
