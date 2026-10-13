"""Owner-scoped source-bound profile promotion and erased replay tombstones."""
from hashlib import sha256
import json
from uuid import UUID
from django.contrib.auth import get_user_model
from django.db import transaction
from django.utils import timezone
from apps.chat.models import Message, WSConversation
from apps.memory.models import UserMemoryFact, ProfileMemoryReconciliationJob
from lib.memory.extraction.stable_facts import INTENT_POLICY_VERSION, is_substantive_fact, parse_memory_intent
from lib.memory.types import RetrievedEpisodicMemory
from lib.memory.mem0._provenance import validated_source_provenance


class ProfileFactConflict(Exception):
    """The displayed fact revision is no longer current."""


def content_digest(text):
    return sha256(text.encode('utf-8')).hexdigest()


def fact_digest(text):
    return content_digest(' '.join(text.split()).casefold().rstrip('.!?'))


def _identity(user_id, convo_id, message_uuid, digest, start, end):
    payload = {'version': INTENT_POLICY_VERSION, 'user': user_id, 'conversation': convo_id,
               'message': str(UUID(str(message_uuid))), 'digest': digest, 'start': start, 'end': end}
    return sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _lock_user(user_id):
    return get_user_model().objects.select_for_update().get(pk=user_id)


def source_is_current(*, user_id, conversation_id, message_uuid, digest, lock=False):
    if type(conversation_id) is not int or conversation_id <= 0:
        return False
    try:
        message_uuid = UUID(str(message_uuid))
    except (ValueError, TypeError, AttributeError):
        return False
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    convos = WSConversation.objects
    messages = Message.objects
    if lock:
        convos, messages = convos.select_for_update(), messages.select_for_update()
    if not convos.filter(pk=conversation_id, owner_id=user_id).exists():
        return False
    rows = list(messages.filter(conversation_id=conversation_id, message_uuid=message_uuid, role='user')[:2])
    return len(rows) == 1 and content_digest(rows[0].content) == digest


def _queue_reconciliation(fact):
    if fact.reconciliation_status == 'not_needed' and not fact.provider_memory_ids and fact.origin != 'legacy_unknown':
        return
    fact.reconciliation_status = 'pending'
    fact.save(update_fields=['reconciliation_status'])
    ProfileMemoryReconciliationJob.objects.update_or_create(fact=fact, defaults={
        'state': 'pending', 'last_error': '', 'next_attempt_at': timezone.now()})
    from aquillm.tasks import reconcile_profile_memory_task
    transaction.on_commit(lambda: reconcile_profile_memory_task.delay(fact_id=fact.pk), robust=True)


def _erase(fact, status):
    fact.fact_digest = fact.fact_digest or fact_digest(fact.fact)
    fact.fact, fact.status = '', status
    fact.source_span_start = fact.source_span_end = None
    fact.revision += 1
    fact.save()
    _queue_reconciliation(fact)


def promote_source_payload(*, user, user_content, source_conversation_id, source_message_uuid, source_content_digest, provider_memory_ids=(), provider_wrote=False, provider_receipt_complete=False):
    """The exact accepted span is the durable fact; model wording cannot invent identity."""
    intent = parse_memory_intent(user_content)
    if type(source_conversation_id) is not int or source_conversation_id <= 0:
        return 0
    if intent.kind != 'affirmative' or content_digest(user_content) != source_content_digest:
        return 0
    with transaction.atomic():
        _lock_user(user.pk)
        current = source_is_current(user_id=user.pk, conversation_id=source_conversation_id,
                                    message_uuid=source_message_uuid, digest=source_content_digest, lock=True)
        related = UserMemoryFact.objects.filter(user=user, source_conversation_id=source_conversation_id,
                    source_message_uuid=source_message_uuid, source_content_digest=source_content_digest)
        blocked = related.exclude(status='active').exists()
        key = _identity(user.pk, source_conversation_id, source_message_uuid, source_content_digest, intent.start, intent.end)
        from aquillm.memory_categories import _categorize_profile_fact
        fact, created = UserMemoryFact.objects.get_or_create(promotion_key=key, defaults={
            'user': user, 'fact': intent.payload if current and not blocked else '',
            'status': 'active' if current and not blocked else 'revoked',
            'origin': 'explicit_request', 'source_conversation_id': source_conversation_id,
            'source_message_uuid': source_message_uuid, 'source_content_digest': source_content_digest,
            'source_span_start': intent.start if current and not blocked else None,
            'source_span_end': intent.end if current and not blocked else None,
            'intent_policy_version': INTENT_POLICY_VERSION, 'fact_digest': fact_digest(intent.payload),
            'category': _categorize_profile_fact(intent.payload),
            'provider_receipt_complete': not provider_wrote or (provider_receipt_complete is True and bool(provider_memory_ids)),
        }) if current or provider_wrote else (None, False)
        if fact is None:
            return 0
        if fact.status == 'active' and (not current or blocked):
            _erase(fact, 'revoked')
        ids = list(dict.fromkeys([*fact.provider_memory_ids, *provider_memory_ids]))
        valid_ids = [value for value in ids if isinstance(value, str) and value and len(value) <= 256][:50]
        fact.provider_memory_ids = valid_ids
        if provider_wrote:
            fact.provider_receipt_complete = (fact.provider_receipt_complete and provider_receipt_complete is True
                                             and bool(valid_ids) and valid_ids == ids)
            fact.reconciliation_status = 'linked' if fact.provider_receipt_complete else 'pending'
        fact.save(update_fields=['provider_memory_ids', 'provider_receipt_complete', 'reconciliation_status'])
        if fact.status != 'active':
            _queue_reconciliation(fact)
        return int(created and fact.status == 'active')


def _current_fact(*, user, fact_id, expected_revision):
    if type(expected_revision) is not int or expected_revision < 1:
        raise ValueError('expected_revision must be a positive integer')
    fact = UserMemoryFact.objects.select_for_update().get(pk=fact_id, user=user)
    if fact.status != 'active' or fact.revision != expected_revision:
        raise ProfileFactConflict()
    return fact


def correct_profile_fact(*, user, fact_id, replacement, expected_revision):
    if not is_substantive_fact(replacement) or any(c in replacement for c in ['\n', '\r']):
        raise ValueError('replacement must be a substantive statement')
    with transaction.atomic():
        _lock_user(user.pk)
        old = _current_fact(user=user, fact_id=fact_id, expected_revision=expected_revision)
        _erase(old, 'superseded')
        from aquillm.memory_categories import _categorize_profile_fact
        return UserMemoryFact.objects.create(user=user, fact=replacement.strip(), origin='manual',
            supersedes=old, fact_digest=fact_digest(replacement), category=_categorize_profile_fact(replacement), provider_receipt_complete=True)


def revoke_profile_fact(*, user, fact_id, expected_revision):
    with transaction.atomic():
        _lock_user(user.pk)
        fact = _current_fact(user=user, fact_id=fact_id, expected_revision=expected_revision)
        _erase(fact, 'revoked')


def get_active_profile_facts(user):
    """Recheck edited/deleted/foreign source revisions before prompt construction."""
    with transaction.atomic():
        _lock_user(user.pk)
        out = []
        for fact in UserMemoryFact.objects.select_for_update().filter(user=user, status='active').order_by('category', 'created_at'):
            if fact.source_message_uuid and not source_is_current(user_id=user.pk, conversation_id=fact.source_conversation_id,
                                                    message_uuid=fact.source_message_uuid, digest=fact.source_content_digest):
                _erase(fact, 'revoked')
            else:
                out.append(fact)
        return out


def invalidate_deleted_source(message):
    """Runs in the source deletion transaction; retains no source excerpts."""
    for fact in UserMemoryFact.objects.select_for_update().filter(source_conversation_id=message.conversation_id,
                                            source_message_uuid=message.message_uuid, status='active'):
        _erase(fact, 'revoked')


def filter_reconciled_memories(user, memories):
    """Known source/ID tombstones and erased fact digests apply before prompt injection."""
    get_active_profile_facts(user)  # Source edits use bulk_update, not post_save.
    tombstones = list(UserMemoryFact.objects.filter(user=user).exclude(status='active'))
    ids = {identity for fact in tombstones for identity in fact.provider_memory_ids}
    sources = {(str(fact.source_message_uuid), fact.source_content_digest) for fact in tombstones if fact.source_message_uuid}
    digests = {fact.fact_digest for fact in tombstones if fact.fact_digest}
    local_conversations = {fact.source_conversation_id for fact in tombstones if fact.source_message_uuid}
    unknown_copies_unresolved = any(fact.origin == 'legacy_unknown' or
        (fact.reconciliation_status in {'pending', 'failed'} and not fact.provider_receipt_complete) for fact in tombstones)
    source_uuids = {identity for identity, _ in sources}
    eligible = []
    for item in memories:
        source_uuid, digest = validated_source_provenance(
            getattr(item, 'source_message_uuid', None), getattr(item, 'source_content_digest', None))
        attributed = source_uuid is not None and digest is not None
        if (getattr(item, 'provider_memory_id', None) in ids
                or (source_uuid, digest) in sources or fact_digest(item.content) in digests):
            continue
        if source_uuid in source_uuids and (not attributed or not source_is_current(
                user_id=user.pk, conversation_id=getattr(item, 'conversation_id', None),
                message_uuid=source_uuid, digest=digest)):
            continue  # Same source needs proof of an independent live revision.
        # Old local episodes identify only the conversation/assistant, so omit
        # that conversation's copies rather than guess which user span survived.
        if hasattr(item, 'assistant_message_uuid') and getattr(item, 'conversation_id', None) in local_conversations:
            continue
        # Missing receipts/source attribution cannot prove independence from
        # an erased fact. Even an absent provider ID must not bypass erasure.
        if (tombstones and isinstance(item, RetrievedEpisodicMemory) and not attributed
                and (unknown_copies_unresolved or not item.provider_memory_id)):
            continue
        eligible.append(item)
    return eligible
