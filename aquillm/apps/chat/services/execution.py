"""Renewable turn ownership plus conservative, durable tool idempotency."""

import asyncio
from contextlib import asynccontextmanager, contextmanager, suppress
from datetime import timedelta
from hashlib import sha256
from json import dumps
from threading import Event, RLock
from uuid import uuid4

from channels.db import database_sync_to_async
from django.db import close_old_connections, transaction
from django.utils import timezone

from apps.chat.models import ToolExecution, WSConversation
from aquillm.message_adapters import ConversationConflictError
from lib.llm.execution_context import (
    UNCERTAIN_MESSAGE,
    UncertainToolExecution,
    bind_execution,
)

LEASE_SECONDS = 60


def claim_execution(conversation_id):
    with transaction.atomic():
        row = WSConversation.objects.select_for_update().get(pk=conversation_id)
        now = timezone.now()
        if (
            row.execution_token
            and row.execution_expires_at
            and row.execution_expires_at > now
        ):
            return None
        token = uuid4()
        WSConversation.objects.filter(pk=row.pk).update(
            execution_token=token,
            execution_expires_at=now + timedelta(seconds=LEASE_SECONDS),
        )
        return token


def release_execution(conversation_id, token):
    WSConversation.objects.filter(pk=conversation_id, execution_token=token).update(
        execution_token=None, execution_expires_at=None
    )


def verify_execution(row, token):
    if (
        row.execution_token != token
        or not row.execution_expires_at
        or row.execution_expires_at <= timezone.now()
    ):
        raise ConversationConflictError(
            "This chat's execution ownership changed. Refresh to load the current result."
        )


class Execution:
    def __init__(self, conversation_id, token):
        self.conversation_id, self.token = conversation_id, token
        self.cancelled = Event()
        self.publication_lock = RLock()

    def cancel(self):
        with self.publication_lock:
            self.cancelled.set()

    @contextmanager
    def publication(self):
        self.check_active()
        locked = False
        try:
            with transaction.atomic():
                yield
                self.publication_lock.acquire()
                locked = True
                self.check_active()
        finally:
            if locked:
                self.publication_lock.release()

    def check_active(self):
        if self.cancelled.is_set():
            raise asyncio.CancelledError()

    def renew(self):
        return WSConversation.objects.filter(
            pk=self.conversation_id, execution_token=self.token
        ).update(execution_expires_at=timezone.now() + timedelta(seconds=LEASE_SECONDS))

    def run_tool(self, message, function):
        self.check_active()
        fingerprint = sha256(
            dumps(
                [message.tool_call_name, message.tool_call_input],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        close_old_connections()
        try:
            with transaction.atomic():
                row = WSConversation.objects.select_for_update().get(
                    pk=self.conversation_id
                )
                verify_execution(row, self.token)
                receipt, created = ToolExecution.objects.get_or_create(
                    conversation_id=self.conversation_id,
                    call_id=message.tool_call_id,
                    defaults={"fingerprint": fingerprint},
                )
                if not created:
                    if receipt.fingerprint != fingerprint or receipt.result is None:
                        raise UncertainToolExecution(UNCERTAIN_MESSAGE)
                    return receipt.result
            # Never hold a database lock while executing arbitrary work. The durable
            # receipt is committed first; any crash/error leaves an uncertain result.
            self.check_active()
            try:
                result = function()
                ToolExecution.objects.filter(pk=receipt.pk).update(result=result)
            except Exception as exc:
                raise UncertainToolExecution(UNCERTAIN_MESSAGE) from exc
            return result
        finally:
            close_old_connections()


@asynccontextmanager
async def execution_turn(consumer, *, wait=False):
    token = await database_sync_to_async(claim_execution)(consumer.db_convo.pk)
    while token is None and wait:
        await asyncio.sleep(0.5)
        token = await database_sync_to_async(claim_execution)(consumer.db_convo.pk)
    if token is None:
        raise ConversationConflictError(
            "This chat is already generating a reply in another connection. Refresh to see its result."
        )
    execution = Execution(consumer.db_convo.pk, token)
    consumer._active_chat_execution = execution
    consumer.db_convo._execution_token = token
    owner = asyncio.current_task()

    async def renew():
        try:
            while True:
                await asyncio.sleep(LEASE_SECONDS / 3)
                if not await database_sync_to_async(execution.renew)():
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        execution.cancel()
        with suppress(Exception):
            await consumer.close(code=1011)
        owner.cancel()

    heartbeat = asyncio.create_task(renew())
    try:
        with bind_execution(execution):
            yield execution
    finally:
        execution.cancel()
        consumer._active_chat_execution = None
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        await database_sync_to_async(release_execution)(consumer.db_convo.pk, token)
        # Keep the token on the writer handle so any late save is rejected.
