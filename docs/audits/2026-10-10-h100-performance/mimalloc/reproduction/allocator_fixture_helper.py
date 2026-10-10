"""Coordinator-only development helper; uncommitted, never run by the worker.

Run inside web, cwd /app/aquillm, with --repo-root /app. provision credentials
are one stdin JSON object {username,password}; they are never written or printed.
Usernames must be allocator-replay-<run-id>. State and fixture contain only IDs.
prove reads retained runner rows and owned DB records. clear_conversations removes
only those chats and this disposable principal's live Mem0 namespace; fixture and
user remain. final_cleanup additionally removes its private collection and user.
Mem0's SDK deliberately retains SQLite deletion history; this helper does not
erase that history, reset stores, revoke global tasks, or change service config.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def quiet():
    prior = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(prior)


def setup(repo, replay_script=None):
    sys.path.insert(0, str(repo / "aquillm"))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aquillm.settings")
    import django

    with quiet():
        django.setup()
    spec = importlib.util.spec_from_file_location(
        "live_replay", replay_script or repo / "scripts/h100_performance/chat_replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owner(state):
    from django.contrib.auth import get_user_model

    user = get_user_model().objects.get(pk=state["user_id"])
    assert user.username == "allocator-replay-" + state["run_id"]
    assert user.email == "allocator-replay-" + state["run_id"] + "@example.invalid"
    assert not user.is_staff and not user.is_superuser
    return user


def synthetic_vtt(fact):
    from aquillm.vtt import coalesce_captions, parse, to_text

    payload = (
        "WEBVTT\n\n1\n00:00:00.000 --> 00:00:05.000\n" + "Calibration: " + fact + "\n"
    ).encode()
    captions = parse(io.BytesIO(payload))
    assert len(captions) == 1 and captions[0].text == fact
    assert captions[0].speaker == "Calibration"
    canonical = to_text(coalesce_captions(captions))
    assert canonical == "00:00:00 Calibration: " + fact + "\n\n"
    return payload, canonical


def profile_state(state):
    from apps.memory.models import UserMemoryFact

    facts = list(
        UserMemoryFact.objects.filter(user=owner(state))
        .order_by("pk")
        .values_list("pk", "category", "fact")
    )
    return dict(
        profile_fact_count=len(facts),
        profile_facts_sha256=digest(facts),
        profile_facts_empty=not facts,
        queued_profile_promotion_terminal_verified=False,
    )


def graph_status(state):
    from django.db.models import Q

    from apps.knowledge_graph.models import GraphBuildRun
    from lib.knowledge_graph.config import load_extraction_settings

    config = load_extraction_settings()
    runs = list(
        GraphBuildRun.objects.filter(
            Q(scope_type="document", scope_id=str(state["document_id"]))
            | Q(scope_type="collection", scope_id=str(state["collection_id"]))
        )
        .order_by("pk")
        .values("id", "scope_type", "scope_id", "status", "stage")
    )
    return dict(
        build_enabled=config.build_enabled,
        configured_extractor_provider=config.provider,
        configured_extractor_model=config.model_id,
        configured_extractor_device=config.device,
        owned_build_runs=runs,
        observed_nonterminal_runs=sum(
            row["status"] in {"pending", "running"} for row in runs
        ),
        queued_graph_tasks_quiescent_verified=False,
    )


def check_ready(state):
    from apps.chat.models import WSConversation

    profile = profile_state(state)
    assert profile["profile_facts_empty"], "profile_facts_not_empty"
    assert not WSConversation.objects.filter(owner=owner(state)).exists()
    fixture = fixture_proof(state)
    assert fixture["ready"]
    return dict(
        ready_for_arm=True,
        profile=profile,
        fixture=fixture,
        knowledge_graph=graph_status(state),
    )


def fixture_proof(state):
    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import TextChunk, VTTDocument
    from lib.embeddings.provenance import valid_provenance

    user = owner(state)
    collection = Collection.objects.get(
        pk=state["collection_id"], name="allocator-replay-" + state["run_id"]
    )
    assert collection.parent_id is None
    permissions = list(
        CollectionPermission.objects.filter(collection=collection).values_list(
            "user_id", "permission"
        )
    )
    assert permissions == [(user.pk, "MANAGE")]
    doc = VTTDocument.objects.get(
        id=state["document_id"], collection=collection, ingested_by=user
    )
    assert doc.full_text == state["expected_canonical_text"]
    assert doc.full_text_hash == hashlib.sha256(doc.full_text.encode()).hexdigest()
    chunks = list(TextChunk.objects.filter(doc_id=doc.id).order_by("id"))
    checks = []
    for chunk in chunks:
        receipt = valid_provenance(chunk.embedding, chunk.embedding_provenance)
        checks.append(
            dict(
                chunk_id=chunk.id,
                has_fact=state["synthetic_fact"] in " ".join(chunk.content.split()),
                vector_dimensions=len(chunk.embedding)
                if chunk.embedding is not None
                else None,
                valid_provenance=receipt is not None,
                provenance_sha256=digest(receipt) if receipt else None,
                content_sha256=hashlib.sha256(chunk.content.encode()).hexdigest(),
                provider=receipt["provider"] if receipt else None,
                route=receipt["route"] if receipt else None,
            )
        )
    ready = (
        doc.ingestion_complete
        and bool(checks)
        and all(
            row["has_fact"]
            and row["vector_dimensions"] == 1024
            and row["valid_provenance"]
            for row in checks
        )
    )
    return dict(
        ready=ready,
        document_id=str(doc.id),
        full_text_sha256=doc.full_text_hash,
        chunks=checks,
    )


def provision(args, replay):
    from allauth.account.models import EmailAddress
    from django.contrib.auth import get_user_model
    from django.db import transaction

    from apps.documents.models import VTTDocument

    assert args.run_id and re.fullmatch(r"[A-Za-z0-9_-]{1,48}", args.run_id)
    assert args.fixture and args.model and args.base_url
    assert not args.state.exists() and not args.fixture.exists()
    credentials = json.loads(sys.stdin.readline())
    assert set(credentials) == {"username", "password"}
    assert credentials["username"] == "allocator-replay-" + args.run_id
    assert (
        isinstance(credentials["password"], str) and len(credentials["password"]) >= 20
    )
    replay.checked_base(args.base_url)
    fact = f"For replay {args.run_id}, the amber calibration code is ORCHID-7391."
    vtt, canonical = synthetic_vtt(fact)
    with transaction.atomic():
        user = get_user_model().objects.create_user(
            username=credentials["username"],
            email="allocator-replay-" + args.run_id + "@example.invalid",
            password=credentials["password"],
            is_active=True,
            is_staff=False,
            is_superuser=False,
        )
        EmailAddress.objects.create(
            user=user, email=user.email, primary=True, verified=True
        )
    state = dict(
        schema_version=1,
        run_id=args.run_id,
        user_id=user.pk,
        synthetic_fact=fact,
        expected_canonical_text=canonical,
        fixture_path=str(args.fixture),
        collection_id=None,
        document_id=None,
        cleared_conversation_ids=[],
        expected_model=args.model,
    )
    save(
        args.state, state
    )  # Record own ID immediately, including partial setup failures.

    async def ingest():
        from http.cookies import SimpleCookie

        import httpx

        async with httpx.AsyncClient(timeout=30, trust_env=False) as session:
            client = replay.ReplayClient(session, args.base_url, {})
            await client.login(credentials["username"], credentials["password"])
            cookies = SimpleCookie()
            cookies.load(
                session.build_request("GET", args.base_url + "/").headers.get(
                    "Cookie", ""
                )
            )
            headers = {
                "X-CSRFToken": cookies["csrftoken"].value,
                "Referer": args.base_url + "/",
            }
            response = await session.post(
                args.base_url + "/api/collections/",
                json={"name": "allocator-replay-" + args.run_id},
                headers=headers,
                follow_redirects=False,
            )
            assert response.status_code == 200
            state["collection_id"] = response.json()["id"]
            save(args.state, state)
            response = await session.post(
                args.base_url + "/api/ingest_vtt/",
                headers=headers,
                data={
                    "collection": str(state["collection_id"]),
                    "title": "allocator-replay-" + args.run_id,
                },
                files={"vtt_file": ("synthetic.vtt", vtt, "text/vtt")},
                follow_redirects=False,
            )
            assert response.status_code == 200 and response.json() == {
                "status_message": "Success"
            }

    asyncio.run(ingest())
    documents = list(
        VTTDocument.objects.filter(
            collection_id=state["collection_id"], ingested_by_id=user.id
        )
    )
    assert len(documents) == 1
    state["document_id"] = str(documents[0].id)
    save(args.state, state)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        proof = fixture_proof(state)
        if proof["ready"] or time.monotonic() >= deadline:
            break
        time.sleep(1)
    assert proof["ready"]
    fixture = replay.validate_fixture(
        dict(
            collection_id=state["collection_id"],
            document_id=state["document_id"],
            chunk_ids=[row["chunk_id"] for row in proof["chunks"]],
            run_id=args.run_id,
            chat_code="CHAT-7391",
            rag_code="ORCHID-7391",
            expected_model=args.model,
        )
    )
    save(args.fixture, fixture)
    state["fixture_sha256"] = digest(fixture)
    save(args.state, state)
    return dict(
        provisioned=True,
        user_id=user.pk,
        fixture=fixture,
        ingestion_proof=proof,
        profile=profile_state(state),
        knowledge_graph=graph_status(state),
    )


def rows_from(args, state):
    rows = []
    for path in args.rows:
        assert path.stat().st_size <= 10 * 1024 * 1024
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    for row in rows:
        assert row.get("fixture_sha256") == state["fixture_sha256"]
        assert type(row.get("conversation_id")) is int and row["conversation_id"] > 0
    identities = [row["conversation_id"] for row in rows]
    assert len(set(identities)) == len(identities)
    return rows


def scoped_chats(rows, state):
    from apps.chat.models import WSConversation

    user = owner(state)
    ids = [row["conversation_id"] for row in rows]
    chats = list(WSConversation.objects.filter(pk__in=ids, owner=user).order_by("pk"))
    assert len(chats) == len(ids)
    # The principal is disposable; no unlisted chat may be hidden from cleanup.
    assert set(
        WSConversation.objects.filter(owner=user).values_list("id", flat=True)
    ) == set(ids)
    return chats


def background_status(chats):
    from apps.chat.models import ConversationChunk
    from apps.chat.services.conversation_indexing import conversation_transcript_hash
    from apps.memory.jobs import _execution_lock
    from apps.memory.models.jobs import ConversationMemoryJob

    observations = []
    for chat in chats:
        chat.refresh_from_db()
        job = ConversationMemoryJob.objects.filter(pk=chat.pk).first()
        with _execution_lock(chat.pk) as acquired:
            observations.append(
                dict(
                    conversation_id=chat.pk,
                    memory_writer_absent=bool(acquired),
                    memory_state=job.state if job else "missing",
                    memory_current=bool(
                        job
                        and job.state == "idle"
                        and job.desired_transcript_hash
                        and job.desired_transcript_hash == job.completed_transcript_hash
                    ),
                    index_current=bool(
                        chat.index_complete
                        and chat.indexed_transcript_hash
                        == conversation_transcript_hash(chat)
                        and not ConversationChunk.objects.filter(
                            conversation=chat, embedding__isnull=True
                        ).exists()
                    ),
                    foreground_idle=chat.execution_token is None,
                )
            )
    return dict(
        owned_chats=observations,
        current_memory_and_index=bool(observations)
        and all(
            row["memory_writer_absent"]
            and row["memory_current"]
            and row["index_current"]
            and row["foreground_idle"]
            for row in observations
        ),
        all_celery_tasks_quiescent_verified=False,
    )


def prove(args, state, replay):
    from django.apps import apps

    from aquillm.message_adapters import (
        django_message_to_pydantic,
        pydantic_message_to_frontend_dict,
    )

    rows = rows_from(args, state)
    chats = scoped_chats(rows, state)
    fixture = replay.validate_fixture(
        json.loads(Path(state["fixture_path"]).read_text())
    )
    assert digest(fixture) == state["fixture_sha256"]
    checks = []
    for row in rows:
        chat = next(chat for chat in chats if chat.pk == row["conversation_id"])
        assert row["complete"] and row["action_completion_acknowledged"]
        message = chat.db_messages.get(
            message_uuid=row["message_uuid"], role="assistant"
        )
        visible = pydantic_message_to_frontend_dict(
            django_message_to_pydantic(message)
        )["content"]
        checks.append(
            dict(
                conversation_id=chat.pk,
                label=row["label"],
                kind=row["kind"],
                model=message.model,
                stop_reason=message.stop_reason,
                persisted_output_matches=hashlib.sha256(visible.encode()).hexdigest()
                == row["output_sha256"],
                exact_answer=replay.exact_answer(row["kind"], fixture, visible),
                model_matches=message.model == state["expected_model"],
                successful_finish=message.stop_reason == "stop"
                and not message.tool_call_name,
            )
        )
    deadline = time.monotonic() + args.wait_seconds
    while True:
        background = background_status(chats)
        if background["current_memory_and_index"] or time.monotonic() >= deadline:
            break
        time.sleep(1)
    llm = apps.get_app_config("aquillm").llm_interface
    url = urlsplit(str(llm.client.base_url))
    assert not url.username and not url.password and not url.query and not url.fragment
    answer_proof = bool(checks) and all(
        row["persisted_output_matches"]
        and row["exact_answer"]
        and row["model_matches"]
        and row["successful_finish"]
        for row in checks
    )
    return dict(
        answer_proof_passed=answer_proof,
        conversations=checks,
        fixture=fixture_proof(state),
        background=background,
        profile=profile_state(state),
        knowledge_graph=graph_status(state),
        route_configuration_only=True,
        configured_interface=type(llm).__name__,
        configured_model=llm.base_args.get("model"),
        configured_base_url=urlunsplit((url.scheme, url.netloc, url.path, "", "")),
        actual_serving_process_allocator_and_dispatch_verified=False,
    )


def clear(args, state, final=False):
    from django.apps import apps

    from apps.chat.models import WSConversation
    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import Document, TextChunk, VTTDocument
    from apps.memory.jobs import _execution_lock
    from lib.memory.mem0.client import get_mem0_oss

    rows = rows_from(args, state) if args.rows else []
    # Journaled own IDs make a partial Mem0 failure safely retryable.
    rows = [
        row
        for row in rows
        if row["conversation_id"] not in state["cleared_conversation_ids"]
    ]
    chats = scoped_chats(rows, state)
    user = owner(state)
    assert all(chat.execution_token is None for chat in chats)
    profile_before = profile_state(state)
    if not final:
        assert profile_before["profile_facts_empty"], "profile_facts_not_empty"
    with quiet():
        mem0 = get_mem0_oss()
    assert mem0 is not None and mem0.enable_graph and mem0.graph is not None
    with ExitStack() as stack:
        for chat in chats:
            assert stack.enter_context(_execution_lock(chat.pk)), (
                "owned_memory_writer_active"
            )
        # Keep the exact writer locks through chat/job deletion and namespace clear.
        ids = [chat.pk for chat in chats]
        WSConversation.objects.filter(pk__in=ids, owner=user).delete()
        assert not WSConversation.objects.filter(pk__in=ids).exists()
        state["cleared_conversation_ids"].extend(ids)
        save(args.state, state)
        with quiet():
            mem0.delete_all(user_id=str(user.pk))
            remaining = mem0.vector_store.list(
                filters={"user_id": str(user.pk)}, limit=1
            )[0]
            count_rows = mem0.graph.graph.query(
                "MATCH (n:Entity {user_id: $user_id}) RETURN count(n) AS count",
                params={"user_id": str(user.pk)},
            )
        assert not remaining and len(count_rows) == 1 and count_rows[0]["count"] == 0
        if final:
            if state["collection_id"] is not None:
                collection = Collection.objects.get(
                    pk=state["collection_id"],
                    name="allocator-replay-" + state["run_id"],
                )
                assert list(
                    CollectionPermission.objects.filter(
                        collection=collection
                    ).values_list("user_id", "permission")
                ) == [(user.pk, "MANAGE")]
                assert collection.parent_id is None
                assert not Collection.objects.filter(parent=collection).exists()
                document_models = [
                    model
                    for model in apps.get_models()
                    if issubclass(model, Document) and not model._meta.abstract
                ]
                inventory = []
                for model in document_models:
                    inventory.extend(
                        (model, doc)
                        for doc in model.objects.filter(collection=collection)
                    )
                    assert (
                        not model.objects.filter(ingested_by=user)
                        .exclude(collection=collection)
                        .exists()
                    )
                if state["document_id"] is None:
                    assert not inventory
                else:
                    assert len(inventory) == 1
                    model, doc = inventory[0]
                    assert model is VTTDocument
                    assert (
                        str(doc.id) == state["document_id"]
                        and doc.ingested_by_id == user.pk
                    )
                    assert not doc.child_figures.exists()
                owned_docs = [doc.id for _, doc in inventory]
                collection.delete()
                assert not Collection.objects.filter(pk=state["collection_id"]).exists()
                assert not TextChunk.objects.filter(doc_id__in=owned_docs).exists()
            user.delete()
            state["principal_deleted"] = True
        else:
            assert profile_state(state)["profile_facts_empty"], (
                "profile_facts_not_empty"
            )
        save(args.state, state)
    return dict(
        cleared_conversation_ids=ids,
        live_mem0_vectors_empty=True,
        live_mem0_graph_nodes_empty=True,
        mem0_sqlite_history_erased=False,
        user_and_fixture_deleted=final,
        all_celery_tasks_quiescent_verified=False,
        profile_before=profile_before,
        profile_facts_empty_after=True,
        queued_profile_promotion_terminal_verified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "provision",
            "prove",
            "check_ready",
            "clear_conversations",
            "final_cleanup",
        ),
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--replay-script", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--proof-output", type=Path)
    parser.add_argument("--rows", type=Path, nargs="*", default=[])
    parser.add_argument("--base-url")
    parser.add_argument("--run-id")
    parser.add_argument("--model")
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    try:
        assert 0 <= args.wait_seconds <= 1800
        replay = setup(args.repo_root, args.replay_script)
        if args.mode == "provision":
            report = provision(args, replay)
        else:
            state = json.loads(args.state.read_text())
            assert state["schema_version"] == 1
            report = (
                prove(args, state, replay)
                if args.mode == "prove"
                else check_ready(state)
                if args.mode == "check_ready"
                else clear(args, state, args.mode == "final_cleanup")
            )
        if args.proof_output:
            save(args.proof_output, report)
        print(json.dumps(report))
        return 0
    except Exception as exc:
        # Never print exception messages, response bodies, credentials or tracebacks.
        locations = [
            dict(file=Path(frame.filename).name, function=frame.name, line=frame.lineno)
            for frame in traceback.extract_tb(exc.__traceback__)
        ]
        print(
            json.dumps(
                dict(
                    success=False,
                    phase=args.mode,
                    error_type=type(exc).__name__,
                    locations=locations,
                )
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
