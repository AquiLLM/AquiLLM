"""Reconcile identified provider copies against their exact owner/source receipt."""


def reconcile_known_profile_copies(*, get_client, timeout_call, user_id: str, provider_memory_ids: list[str], source_conversation_id=None, source_message_uuid=None, source_content_digest=None) -> None:
    """Delete only identified copies whose provider receipt still verifies owner/source."""
    if not provider_memory_ids or len(provider_memory_ids) > 50:
        raise NotImplementedError('unidentified_provider_copies')
    client = get_client()
    if client is None or not callable(getattr(client, 'get', None)) or not callable(getattr(client, 'delete', None)):
        raise NotImplementedError('unsupported_provider_lifecycle')
    for identity in provider_memory_ids:
        if not isinstance(identity, str) or not identity or len(identity) > 256:
            raise ValueError('invalid_provider_identity')
        receipt = timeout_call(client.get, identity)
        if receipt is None:
            continue  # Already absent: idempotent retry.
        if not isinstance(receipt, dict) or receipt.get('id') != identity:
            raise ValueError('provider_identity_changed')
        metadata = receipt.get('metadata')
        if not isinstance(metadata, dict) or str(metadata.get('user_id')) != str(user_id):
            raise PermissionError('provider_owner_changed')
        if source_message_uuid is not None and (
            str(metadata.get('source_message_uuid')) != str(source_message_uuid)
            or metadata.get('source_content_digest') != source_content_digest
            or metadata.get('conversation_id') != source_conversation_id
        ):
            raise PermissionError('provider_source_changed')
        result = timeout_call(client.delete, identity)
        # Unknown/missing receipts are not proof of deletion.
        if not isinstance(result, dict) or result.get('message') not in {'deleted', 'Memory deleted successfully'}:
            raise NotImplementedError('unconfirmed_provider_deletion')
