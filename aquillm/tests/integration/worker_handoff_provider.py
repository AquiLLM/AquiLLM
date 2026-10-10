"""Strict synthetic transport and owner-verified memory boundary for workers."""
import json
import os
from pathlib import Path
from types import SimpleNamespace


def verified_space():
    from lib.embeddings.tests.test_document_contracts import profile
    from lib.embeddings import document_provider as provider
    space = profile()
    provider.register_verified_profile(
        space, endpoint="http://fake-models:8000/v1",
        deployment_artifact_sha256="c" * 64,
        engine_manifest=provider.engine_manifest_for(space),
        query_template="{text}", document_template="{text}",
    )
    return space


def install_provider_boundaries():
    from lib.embeddings import document_provider as provider
    from lib.memory.mem0 import operations
    verified_space()

    def create(**kwargs):
        return SimpleNamespace(model="synthetic", data=[
            SimpleNamespace(index=i, embedding=[float(i + 1)] * 1024)
            for i in reversed(range(len(kwargs["input"])))
        ])

    provider._client = lambda: SimpleNamespace(embeddings=SimpleNamespace(create=create))
    provider._endpoint = lambda: "http://fake-models:8000/v1"
    operations.get_mem0_oss = lambda: SyntheticMemoryProvider()


class SyntheticMemoryProvider:
    def get(self, identity):
        from apps.memory.models import UserMemoryFact
        fact = UserMemoryFact.objects.get(provider_memory_ids__contains=[identity])
        metadata = {"user_id": str(fact.user_id)}
        if fact.source_conversation_id:
            metadata.update(conversation_id=fact.source_conversation_id,
                            source_message_uuid=str(fact.source_message_uuid),
                            source_content_digest=fact.source_content_digest)
        return {"id": identity, "metadata": metadata}

    def delete(self, identity):
        with Path(os.environ["HANDOFF_DELETIONS"]).open("a") as output:
            output.write(json.dumps({"deleted": identity}) + "\n")
        return {"message": "deleted"}
