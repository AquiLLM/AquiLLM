#!/usr/bin/env python
"""Replay explicit development questions without synthesis or chat persistence."""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aquillm"))
from apps.chat.evals.retrieval_replay import (  # noqa: E402
    EXPERIMENTS,
    fingerprint,
    load_manifest,
    replay_runtime,
    resolve_targets,
    run_turn,
)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def full_revision(value):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("revision must be a full 40-character hex SHA")
    return value.lower()


def revision_metadata(override):
    if override is not None:
        return {
            "git_revision": full_revision(override),
            "revision_provenance": "operator_supplied",
        }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "git_revision": full_revision(revision),
        "revision_provenance": "git_verified",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--principal-id", required=True, type=positive)
    parser.add_argument("--collection-ids", required=True, nargs="+", type=positive)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=positive, default=1)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="baseline")
    parser.add_argument("--revision", type=full_revision)
    args = parser.parse_args(argv)
    if (
        args.repetitions > 10
        or len(args.collection_ids) > 64
        or len(set(args.collection_ids)) != len(args.collection_ids)
    ):
        parser.error(
            "repetitions must be 1-10; collections must be unique and at most 64"
        )
    if args.output.resolve() == args.manifest.resolve():
        parser.error("output must differ from manifest")
    return args


def selected_sources(user, collections):
    from apps.chat.refs import CollectionsRef
    from apps.chat.services.tool_wiring.source_documents import (
        document_metadata,
        selected_document_metadata,
    )
    from apps.documents.models import TextChunk

    sources = []
    for metadata in selected_document_metadata(user, CollectionsRef(collections)):
        document = document_metadata(metadata.id)
        if document is None or not document.collection.user_can_view(user):
            raise ValueError("source_authorization_failed")
        sources.append(
            {
                "source": document.full_text,
                "chunks": [
                    {
                        "chunk_id": row.pk,
                        "text": row.content,
                        "start": row.start_position,
                        "end": row.end_position,
                    }
                    for row in TextChunk.objects.filter(doc_id=document.id)
                ],
            }
        )
    return sources


def effective_config(settings, config):
    from dataclasses import asdict

    from apps.chat.services import rag_config
    from apps.documents.services.chunk_rerank_config import (
        rerank_doc_char_limit,
        rerank_pair_token_limit,
        rerank_text_mode,
    )

    return {
        "vector_cap": config.vector_top_k,
        "trigram_cap": config.trigram_top_k,
        "candidate_multiplier": getattr(settings, "RAG_CANDIDATE_MULTIPLIER", 3.0),
        "cache_enabled": settings.RAG_CACHE_ENABLED,
        "graph_overlay": getattr(settings, "KG_OVERLAY_ENABLED", False),
        "graph_direct": getattr(settings, "KG_GRAPH_DIRECT_ENABLED", False),
        "graph_extended": getattr(settings, "KG_GRAPH_EXTENDED_ENABLED", False),
        "rerank_character_cap": rerank_doc_char_limit(),
        "rerank_pair_token_cap": rerank_pair_token_limit(),
        "rerank_text_mode": rerank_text_mode(),
        "preservation": asdict(rag_config.rag_preservation_config()),
        "selection": asdict(rag_config.evidence_selection_config()),
        "top_k": rag_config.direct_rag_top_k(),
        "candidate_top_k": rag_config.direct_rag_candidate_top_k(),
        "max_queries": rag_config.direct_rag_max_queries(),
        "token_budget": rag_config.evidence_token_budget(),
    }


def main(argv=None):
    args = parse_args(argv)
    report = {
        "schema_version": 1,
        "success": False,
        "experiment": args.experiment,
        "requested": {
            "question_count": None,
            "collection_count": len(args.collection_ids),
            "repetitions": args.repetitions,
        },
        "turns": [],
    }
    try:
        questions = load_manifest(args.manifest)
        if len(questions) * args.repetitions > 1000:
            raise ValueError("turn_limit")
        report["requested"]["question_count"] = len(questions)
        report["manifest_sha256"] = fingerprint(
            args.manifest.read_text(encoding="utf-8")
        )
        report.update(revision_metadata(args.revision))
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aquillm.settings")
        import django

        django.setup()
        from django.apps import apps
        from django.conf import settings
        from django.contrib.auth import get_user_model

        from apps.chat.services import rag_pipeline

        config = apps.get_app_config("aquillm")
        with replay_runtime(args.experiment, settings, config, rag_pipeline, None):
            user = get_user_model().objects.get(pk=args.principal_id)
            if not user.is_active:
                raise ValueError("principal_inactive")
            # Resolve against authorized selected sources, never against retrieved rows.
            sources = (
                selected_sources(user, args.collection_ids)
                if any(q.get("targets") for q in questions)
                else []
            )
            mapped = [resolve_targets(q.get("targets", []), sources) for q in questions]
            report["effective"] = effective_config(settings, config)

        async def execute():
            for repetition in range(1, args.repetitions + 1):
                for question, targets in zip(questions, mapped, strict=True):
                    row = await run_turn(
                        question,
                        targets,
                        user,
                        args.collection_ids,
                        args.experiment,
                        settings,
                        config,
                        rag_pipeline,
                    )
                    row["repetition"] = repetition
                    report["turns"].append(row)

        asyncio.run(execute())
        report["success"] = bool(report["turns"]) and all(
            row["success"] for row in report["turns"]
        )
    except Exception as exc:
        # Exception messages may contain source, prompts or identifiers.
        report["error_type"] = type(exc).__name__
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
