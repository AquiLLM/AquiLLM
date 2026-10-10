"""Redacted read-only embedding audit; no stored-vector provenance is inferred."""

import argparse
import json
from hashlib import sha256
from math import hypot
from os import getenv

from openai import OpenAI

from .config import (
    allow_embed_dimensions_override,
    get_local_embed_config,
    get_target_dims,
)
from .local import _response_vectors
from .utils import EmbeddingContractError, validate_embedding


def _digest(value: str | None) -> str | None:
    return sha256(value.encode()).hexdigest() if value else None


def build_report(*, probe: bool = False) -> dict:
    """At most one two-item synthetic request; no DB reads, retries or fallback.

    Configuration digests identify declarations only. Actual revision, precision,
    quantization and template need independent process/model evidence.
    """
    base_url, api_key, model = get_local_embed_config()
    target = get_target_dims()
    dimensions_override = allow_embed_dimensions_override()
    report = {
        "schema_version": 1,
        "declared": {
            "provider": "local-openai",
            "endpoint_digest": _digest(base_url),
            "model_digest": _digest(model),
            "revision_digest": _digest(getenv("APP_EMBED_MODEL_REVISION")),
            "target_dimensions": target,
            "request_dimensions_override": dimensions_override,
            "local_role_handling": "validated_only_raw_input_unchanged",
            "dimension_policy": "legacy_pad_or_truncate",
            "transport_failure_policy": "legacy_cohere_fallback",
            "fallback_space_compatibility": "unproven",
        },
        "observed": {"status": "not_probed"},
        "effective_precision_and_quantization": "unknown",
        "historical_identity": "unknown",
        "compatibility": "unproven",
    }
    if not probe:
        return report
    client = None
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=10.0, max_retries=0)
        response = client.embeddings.create(
            model=model,
            input=["Synthetic embedding contract audit sample."] * 2,
            **({"dimensions": target} if dimensions_override else {}),
        )
        vectors = _response_vectors(response, 2)
        for vector in vectors:
            # Verify fitting would preserve a usable vector without logging values.
            validate_embedding(vector[:target])
        response_model = getattr(response, "model", None)
        report["observed"] = {
            "status": "valid_synthetic_response",
            "raw_dimensions": [len(vector) for vector in vectors],
            "raw_norms": [hypot(*vector) for vector in vectors],
            "dimension_adaptation": [
                "unchanged"
                if len(vector) == target
                else "pad"
                if len(vector) < target
                else "truncate"
                for vector in vectors
            ],
            "repeated_input_vectors_equal": vectors[0] == vectors[1],
            "response_model_digest": _digest(response_model)
            if isinstance(response_model, str)
            else None,
            "response_model_matches_declared": response_model == model,
        }
    except EmbeddingContractError:
        report["observed"] = {"status": "invalid_vector_contract"}
    except Exception:
        # Upstream exceptions can contain request text, URLs or credentials.
        report["observed"] = {"status": "upstream_failure"}
    finally:
        if client is not None:
            client.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Send one synthetic two-item request (10s timeout, retries disabled)",
    )
    args = parser.parse_args()
    print(json.dumps(build_report(probe=args.probe), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
