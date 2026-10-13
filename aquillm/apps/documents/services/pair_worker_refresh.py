"""Private same-identity warm preparation and guarded publication."""

import json
from dataclasses import replace
from time import monotonic, time

from .chunk_rerank_pair_capability import (
    _pair_snapshot,
    _publish_pair_candidate,
    _valid_identity,
)
from .chunk_rerank_results import fingerprint_text


def _header(args):
    if not args or not _valid_identity(args.get("identity"), args.get("template")):
        return None
    if args.get("attestation_expires", 0) <= time():
        return None
    return (
        args["provider"].scorer_fingerprint,
        json.dumps(args["identity"], sort_keys=True),
        args["template"],
        args["tokenizer_name"],
    )


def _matches(counter, args):
    identity = args["identity"]
    return counter is not None and (
        counter.identity == fingerprint_text(json.dumps(identity, sort_keys=True))
        and counter.tokenizer_identity == identity["tokenizer_revision"]
        and counter.template_identity == identity["template_sha256"]
        and counter.template == args["template"]
    )


def prepare_attempt(controller, budget, generation):
    """Runs wholly in the existing daemon slot; never publishes late work."""
    args = controller.load()
    header = _header(args)
    if header is None:
        return None
    # Freeze nested identity before any injected/native operation can change it.
    args = dict(args, identity=dict(args["identity"]))
    provider, expiry = args["provider"], args.pop("attestation_expires")
    with controller._lock:
        if controller.stopping or generation != controller.generation:
            return None
        expected = _pair_snapshot(provider)
        if not _matches(expected[0], args):
            controller._revoke()
            expected = _pair_snapshot(provider)
        if expected[0] is not None:
            controller._counter_revocation = expected[0].revoked
        controller.identity = dict(args["identity"])
    with budget._lock:
        budget._deadline = min(budget._deadline, monotonic() + max(0, expiry - time()))
    candidate = controller.initialize(**args, budget=budget)
    if candidate is None or not budget.can_publish():
        return None
    current = controller.load()
    if _header(current) != header or not _matches(candidate, args):
        return None
    expiry = min(expiry, current["attestation_expires"])
    candidate = replace(
        candidate, expires=min(candidate.expires, monotonic() + max(0, expiry - time()))
    )
    return _publish(
        controller, provider, candidate, expected, budget, generation, expiry
    )


def _publish(controller, provider, candidate, expected, budget, generation, expiry):
    from . import chunk_rerank_config as config
    from . import chunk_rerank_pair_capability as capability
    from . import rerank_transport_policy as protocol

    now = time()
    binding = protocol.RerankProtocolV1(
        provider.endpoint,
        provider.shape,
        provider.model_name,
        now,
        min(expiry, now + candidate.expires - monotonic()),
    )
    base = config.rerank_base_url()
    result = []

    def publish():
        # Pair attestation is worker-local. Publish its protocol only in memory;
        # cache IO could expose an unowned binding or finish after cancellation.
        with capability._lock, protocol._BINDING_LOCK:
            counter = _publish_pair_candidate(
                provider, candidate, expected=expected, budget=budget
            )
            if counter is not None:
                protocol._BINDINGS[(base.rstrip("/"), binding.model_key)] = binding
                controller._counter_revocation = counter.revoked
                controller.protocol_binding = (base, binding)
                result.append(counter)

    with controller._lock:
        if not controller.stopping and generation == controller.generation:
            budget.publish(publish)
    return result[0] if result else None


def revoke_local_protocol(base, binding):
    """Remove this worker's exact binding; never delete a newer discovery."""
    from . import rerank_transport_policy as protocol

    with protocol._BINDING_LOCK:
        key = (base.rstrip("/"), binding.model_key)
        if protocol._BINDINGS.get(key) is binding:
            protocol._BINDINGS.pop(key)


def initialize_pair_candidate(
    provider, *, tokenizer_name, template, identity, budget, verify
):
    """Pinned, local-only loading shared by explicit and renewing warm paths."""
    if not _valid_identity(identity, template):
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=identity["tokenizer_revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
    except (ImportError, OSError, ValueError):
        return None
    return verify(
        provider,
        tokenizer=tokenizer,
        template=template,
        identity=identity,
        budget=budget,
    )
