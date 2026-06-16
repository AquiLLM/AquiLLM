"""Backfill ConversationChunk rows for existing conversations.

Usage:
    python manage.py index_conversations              # index all (hash-guarded)
    python manage.py index_conversations --user 42    # only this user's chats
    python manage.py index_conversations --force       # re-index even if unchanged
    python manage.py index_conversations --async       # enqueue Celery tasks instead
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from apps.chat.models import WSConversation
from apps.chat.services.conversation_indexing import index_conversation
from apps.chat.tasks import enqueue_index_conversation_task


class Command(BaseCommand):
    help = "Chunk and index existing conversations for semantic past-chat search."

    def add_arguments(self, parser):
        parser.add_argument("--user", type=int, default=None, help="Limit to one user id.")
        parser.add_argument(
            "--force", action="store_true", help="Re-index even if the transcript is unchanged."
        )
        parser.add_argument(
            "--async",
            action="store_true",
            dest="run_async",
            help="Enqueue Celery tasks instead of indexing inline.",
        )

    def handle(self, *args, **options):
        qs = WSConversation.objects.all().order_by("id")
        if options["user"] is not None:
            qs = qs.filter(owner_id=options["user"])

        total = qs.count()
        self.stdout.write(f"Indexing {total} conversation(s)...")

        processed = 0
        chunk_total = 0
        for convo in qs.iterator():
            if options["run_async"]:
                enqueue_index_conversation_task(
                    conversation_id=convo.id,
                    queued_updated_at=convo.updated_at.isoformat(),
                )
                processed += 1
                continue
            try:
                n = index_conversation(convo.id, force=options["force"])
                chunk_total += n
                processed += 1
                self.stdout.write(f"  convo {convo.id}: {n} chunk(s)")
            except Exception as exc:  # noqa: BLE001 - report and continue the backfill
                self.stderr.write(f"  convo {convo.id}: FAILED ({exc})")

        if options["run_async"]:
            self.stdout.write(self.style.SUCCESS(f"Enqueued {processed} indexing task(s)."))
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Done. Indexed {processed}/{total} conversation(s), {chunk_total} chunk(s) total."
                )
            )
