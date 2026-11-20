"""Pure, versioned extraction behavior identity; no provider or Django imports.

Ontology semantics are bound separately by the artifact's ontology checksum.
Changing adapter, compiler or normalization behavior requires a version bump.
Operational cache, download and failure policy settings do not identify evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256

GLINER2_VERSION = "1.3.2"
ADAPTER_CONTRACT_VERSION = "gliner2-adapter-v1"
NORMALIZER_CONTRACT_VERSION = "exact-typed-spans-v1"
DEFAULT_SCHEMA_COMPILER_VERSION = "gliner2-relations-v1"


def _text(settings: object, name: str) -> str:
    value = getattr(settings, name, None)
    if type(value) is not str or not value.strip():
        raise ValueError(f"extraction {name} must be nonempty text")
    return value


def _positive_integer(settings: object, name: str) -> int:
    value = getattr(settings, name, None)
    if type(value) is not int or value < 1:
        raise ValueError(f"extraction {name} must be a positive integer")
    return value


def effective_inference_options(settings: object) -> dict[str, object]:
    """One source for hashed and supplied options, pinned to GLiNER2 1.3.2.

    The explicit threshold, worker count and length match the inspected runtime's
    batch_extract defaults; they do not lower confidence or truncate source text.
    """
    return {
        "batch_size": _positive_integer(settings, "batch_size"),
        "format_results": False,
        "include_confidence": True,
        "include_spans": True,
        "threshold": 0.5,
        "num_workers": 0,
        "max_len": None,
    }


@dataclass(frozen=True, slots=True)
class ExtractionContract:
    provider: str
    model_id: str
    model_revision: str
    provider_runtime_version: str | None
    adapter_contract_version: str
    schema_compiler_version: str
    normalizer_contract_version: str
    device: str
    max_batch_characters: int
    inference_options: tuple[tuple[str, object], ...]
    contract_version: str = "extract-v2"

    def to_dict(self) -> dict[str, object]:
        manifest = asdict(self)
        manifest["inference_options"] = dict(self.inference_options)
        return manifest


def extraction_contract(settings: object) -> ExtractionContract:
    """Capture whitelisted semantics of the supported extraction behavior."""
    provider = _text(settings, "provider")
    revision = _text(settings, "model_revision")
    if (
        provider == "gliner2_local"
        and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
    ):
        raise ValueError(
            "GLiNER2 extraction requires an immutable 40-character revision"
        )
    return ExtractionContract(
        provider=provider,
        model_id=_text(settings, "model_id"),
        model_revision=revision,
        provider_runtime_version=GLINER2_VERSION
        if provider == "gliner2_local"
        else None,
        adapter_contract_version=ADAPTER_CONTRACT_VERSION,
        schema_compiler_version=DEFAULT_SCHEMA_COMPILER_VERSION,
        normalizer_contract_version=NORMALIZER_CONTRACT_VERSION,
        device=_text(settings, "device"),
        max_batch_characters=_positive_integer(settings, "max_batch_characters"),
        inference_options=tuple(sorted(effective_inference_options(settings).items())),
    )


def extraction_identity(settings: object) -> str:
    contract = extraction_contract(settings)
    encoded = json.dumps(
        contract.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return f"{contract.contract_version}:{sha256(encoded).hexdigest()}"
