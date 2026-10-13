"""Bounded reading and validation of operator pair capability attestations."""


def load_attested_attempt():
    """Consume an operator-verified, short-lived, locally mounted attestation.

    The worker has no Docker privilege. This file is deployment evidence, not
    inferred from /models, environment names or the successful warm probes.
    """
    from . import pair_worker_lifecycle as worker

    path = worker.os.getenv("RAG_PAIR_CAPABILITY_ATTESTATION")
    if not path:
        return None
    try:
        with worker.Path(path).open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            return None
        data = worker.json.loads(raw)
        provider = worker.canonical_provider(deadline=worker.monotonic() + 15)
        template = data["template"]
        identity = data["identity"]
        if not (
            provider
            and data["verified_by"]
            and data["verification_record"]
            and data["verified_at"] <= worker.time() < data["expires_at"]
            and 0 < data["expires_at"] - data["verified_at"] <= 300
            and data["endpoint"] == provider.endpoint
            and data["served_model"] == provider.model_name
            and data["pair_context"] == provider.pair_limit == 1024
            and data["tokenizer_name"] == worker.config.rerank_tokenizer()
            and identity["model_revision"] == worker.config.rerank_model_revision()
            and identity["tokenizer_revision"] == worker.config.rerank_tokenizer_revision()
            and identity["code_revision"] == worker.config.rerank_code_revision()
            and worker._valid_identity(identity, template)
        ):
            return None
        return dict(
            provider=provider,
            tokenizer_name=data["tokenizer_name"],
            template=template,
            identity=identity,
            attestation_expires=data["expires_at"],
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
