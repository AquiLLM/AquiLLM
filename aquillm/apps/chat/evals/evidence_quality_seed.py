"""Isolated synthetic DB fixture with real embeddings and immutable chunk boundaries."""

import json
import struct
from collections import defaultdict
from hashlib import sha256
from uuid import uuid4

from .evidence_quality_eval import digest, text_digest


def embedding_digest(vector):
    return sha256(struct.pack(f"!{len(vector)}f", *vector)).hexdigest()


def seed_cases(cases, path):
    from django.conf import settings
    from django.contrib.auth.models import User
    from django.db import transaction

    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument, TextChunk
    from aquillm.utils import get_embedding

    if path.exists():
        raise ValueError("manifest exists; refusing fixture replacement")
    # Embed before transaction, through the configured real embedding provider.
    vectors = {}
    for case in cases:
        for source in case["sources"]:
            vectors[text_digest(source["text"])] = get_embedding(
                source["text"], input_type="search_document"
            )
    manifest = {
        "kind": "evidence-quality-isolated-v1",
        "database": settings.DATABASES["default"]["NAME"],
        "source_digest": digest([c["sources"] for c in cases]),
        "cases": {},
    }
    with transaction.atomic():
        user = User(username="evidence-eval-" + uuid4().hex, is_active=False)
        user.set_unusable_password()
        user.save(force_insert=True)
        manifest["user_id"] = user.pk
        for case in cases:
            visible = Collection.objects.create(
                name=f"evidence-eval-{case['case_id']}-{uuid4().hex[:8]}"
            )
            hidden = Collection.objects.create(
                name=f"evidence-eval-hidden-{uuid4().hex}"
            )
            CollectionPermission.objects.create(
                user=user, collection=visible, permission="MANAGE"
            )
            selected = {}
            for source in case["sources"]:
                if (
                    source["source_id"] not in selected
                    or source["authorized_at_answer"]
                ):
                    selected[source["source_id"]] = source
            documents = defaultdict(list)
            for source in selected.values():
                documents[source["document_id"]].append(source)
            mappings = []
            for symbol, sources in documents.items():
                sources.sort(key=lambda s: s["chunk_number"])
                authorized = {s["authorized_at_answer"] for s in sources}
                if len(authorized) != 1:
                    raise ValueError("mixed authorization within document")
                text = "\n\n".join(s["text"] for s in sources)
                doc = RawTextDocument(
                    id=uuid4(),
                    title=symbol,
                    full_text=text,
                    full_text_hash=text_digest(text),
                    collection=visible if True in authorized else hidden,
                    ingested_by=user,
                    ingestion_complete=True,
                )
                RawTextDocument.objects.bulk_create([doc])
                start = 0
                for source in sources:
                    vector = vectors[text_digest(source["text"])]
                    if len(vector) != 1024:
                        raise ValueError("embedding dimensions changed")
                    chunk = TextChunk(
                        doc_id=doc.id,
                        content=source["text"],
                        start_position=start,
                        end_position=start + len(source["text"]),
                        chunk_number=source["chunk_number"],
                        embedding=vector,
                        metadata={
                            "evidence_eval_case": case["case_id"],
                            "source_id": source["source_id"],
                            "revision": source["revision"],
                        },
                    )
                    TextChunk.objects.bulk_create([chunk])
                    mappings.append(
                        {
                            "source_id": source["source_id"],
                            "revision": source["revision"],
                            "document_id": str(doc.id),
                            "chunk_id": chunk.pk,
                            "start": start,
                            "fingerprint": text_digest(source["text"]),
                            "embedding_digest": embedding_digest(vector),
                        }
                    )
                    start += len(source["text"]) + 2
            manifest["cases"][case["case_id"]] = {
                "collection_id": visible.pk,
                "hidden_collection_id": hidden.pk,
                "sources": mappings,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        # Creation is explicit and exclusive. Store no credentials in this file.
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
    return manifest


def validate_case(case, manifest):
    from django.conf import settings

    from apps.collections.models import CollectionPermission
    from apps.documents.models import TextChunk

    if (
        manifest["kind"] != "evidence-quality-isolated-v1"
        or manifest["database"] != settings.DATABASES["default"]["NAME"]
    ):
        raise ValueError("isolated fixture database mismatch")
    binding = manifest["cases"][case["case_id"]]
    expected = {(s["source_id"], s["revision"]): s for s in case["sources"]}
    sources = {}
    for row in binding["sources"]:
        source = expected[(row["source_id"], row["revision"])]
        chunk = TextChunk.objects.get(pk=row["chunk_id"], doc_id=row["document_id"])
        if (
            chunk.content != source["text"]
            or text_digest(chunk.content) != row["fingerprint"]
            or chunk.start_position != row["start"]
            or chunk.chunk_number != source["chunk_number"]
        ):
            raise ValueError("source/revision/coordinate drift")
        if (
            chunk.embedding is None
            or embedding_digest(chunk.embedding) != row["embedding_digest"]
        ):
            raise ValueError("embedding snapshot drift")
        permitted = CollectionPermission.objects.filter(
            user_id=manifest["user_id"],
            collection_id=chunk.document.collection_id,
            permission__in=("VIEW", "EDIT", "MANAGE"),
        ).exists()
        if permitted != source["authorized_at_answer"]:
            raise ValueError("authorization fixture drift")
        sources[(str(chunk.doc_id), chunk.pk)] = source
    return binding, sources


def create_conversation(case, manifest):
    from django.contrib.auth.models import User

    from apps.chat.models import WSConversation
    from aquillm.message_adapters import save_conversation_to_db
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

    binding, sources = validate_case(case, manifest)
    user = User.objects.get(pk=manifest["user_id"], is_active=False)
    messages = []
    all_sources = {s["source_id"]: s for s in case["sources"]}
    for turn in case["turns"]:
        ids = turn.get(
            "assistant_citations",
            [
                s["source_id"]
                for s in sorted(
                    turn.get("displayed_sources", []), key=lambda s: s["display_order"]
                )
            ],
        )
        rows = []
        for sid in ids:
            key = next(k for k, s in sources.items() if s["source_id"] == sid)
            historic = next(
                (
                    s
                    for s in case["sources"]
                    if s["source_id"] == sid
                    and s["revision"] == turn.get("source_revision")
                ),
                all_sources[sid],
            )
            rows.append(
                {
                    "doc_id": key[0],
                    "chunk_id": key[1],
                    "chunk": historic["chunk_number"],
                    "text": historic["text"],
                    "title": historic["document_id"],
                    "citation": f"[doc:{key[0]} chunk:{key[1]}]",
                }
            )
        messages.append(UserMessage(content=turn["user"]))
        if rows:
            messages.extend(
                [
                    AssistantMessage(
                        content="",
                        stop_reason="tool_use",
                        tool_call_id="history",
                        tool_call_name="vector_search",
                        tool_call_input={},
                    ),
                    ToolMessage(
                        content=json.dumps({"result": rows}),
                        result_dict={"result": rows},
                        tool_name="vector_search",
                        arguments={},
                        tool_call_id="history",
                        for_whom="assistant",
                    ),
                ]
            )
        messages.append(
            AssistantMessage(
                content=" ".join(r["citation"] for r in rows), stop_reason="end_turn"
            )
        )
    messages.append(UserMessage(content=case["question"]))
    conversation = Conversation(
        system=(
            "Answer using currently authorized sources, preserving "
            "qualifications and limitations."
        ),
        messages=messages,
    )
    db = WSConversation.objects.create(
        owner=user,
        name=case["case_id"],
        system_prompt=conversation.system,
        selected_collection_ids=[binding["collection_id"]],
    )
    save_conversation_to_db(conversation, db)
    return user, db.pk, sources
