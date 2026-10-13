"""Record worker ownership inside the caller's original publication lock."""


def publish_owned_protocol(base, binding, on_registered):
    from .rerank_transport_policy import (
        register_rerank_protocol,
        revoke_rerank_protocol,
    )

    register_rerank_protocol(base, binding)
    try:
        if on_registered is not None:
            on_registered(binding)
    except Exception:
        revoke_rerank_protocol(base, binding)
        raise
