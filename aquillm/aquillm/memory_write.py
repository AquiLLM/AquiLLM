"""Source-bound Mem0 write receipts behind the existing memory facade."""
from lib.memory.mem0._receipts import capture_provider_receipt


def add_source_memory(
    api,
    user,
    user_content: str,
    assistant_content: str,
    conversation_id: int,
    assistant_message_uuid: str,
    *, source_message_uuid=None, source_content_digest=None,
) -> None:
    """Queue profile promotion separately and send the raw turn to Mem0 intelligent infer."""
    intent = api.parse_memory_intent(user_content)
    if intent.kind not in {'none', 'affirmative'}:
        api.logger.info('obs.memory.intelligent_write_excluded', intent=intent.kind)
        return
    from apps.memory.services.profile_facts import source_is_current, promote_source_payload
    if not source_is_current(user_id=user.id, conversation_id=conversation_id,
            message_uuid=source_message_uuid, digest=source_content_digest):
        return
    if api.UserMemoryFact.objects.filter(user=user, source_conversation_id=conversation_id,
            source_message_uuid=source_message_uuid, source_content_digest=source_content_digest).exclude(status='active').exists():
        return
    provider_ids, complete = [], False
    def receipt(result):
        nonlocal provider_ids, complete
        provider_ids, complete = capture_provider_receipt(result)
    wrote = api.add_mem0_memory_with_client(
        user_id=str(user.id),
        user_content=user_content,
        assistant_content=assistant_content,
        conversation_id=conversation_id,
        assistant_message_uuid=assistant_message_uuid,
        strict=True,
        source_message_uuid=source_message_uuid, source_content_digest=source_content_digest,
        result_observer=receipt,
    )
    provider_wrote = bool(provider_ids) or not complete
    if provider_wrote and intent.kind == 'affirmative':
        # A late receipt retains known IDs in an erased tombstone if the source
        # changed or was revoked while provider work was in flight.
        promote_source_payload(user=user, user_content=user_content,
            source_conversation_id=conversation_id, source_message_uuid=source_message_uuid,
            source_content_digest=source_content_digest, provider_memory_ids=provider_ids, provider_wrote=True, provider_receipt_complete=complete)
    # A provider failure raises before either promotion or a local dual-write
    # receipt can make the durable job appear complete. No-add inference is a
    # successful response and may still promote useful profile facts.
    api._enqueue_profile_fact_promotion(
        user_id=user.id,
        user_content=user_content,
        assistant_content=assistant_content,
        source_conversation_id=conversation_id, source_message_uuid=source_message_uuid,
        source_content_digest=source_content_digest, provider_memory_ids=provider_ids, provider_wrote=provider_wrote, provider_receipt_complete=complete,
    )
    if wrote:
        api.logger.info("obs.memory.intelligent_write_succeeded")
        return

    api.logger.warning("obs.memory.intelligent_write_no_adds")
