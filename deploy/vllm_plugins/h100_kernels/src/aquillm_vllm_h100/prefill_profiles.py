"""Bundled development-only crossover, experimental pending serving evidence.

The coordinator owns complete post-Genesis/P38 microbenchmark and serving
qualification for the actual page16 cache. An opt-in targets the bounded
long-prefix region only; selecting this experimental profile does not imply
production/default promotion or that serving qualification has passed.
"""
import hashlib
import json
import os

from .compatibility import CANDIDATE_PACKAGES, GENESIS_COMMIT, MODEL, PACKAGES
from .prefill import PrefillProfile, PrefillRegion

PROFILE_NAME = "h100-long-prefill-dev-v1"
MODEL_REVISION = "2d783431e303148fc6e16622fac5edac83a6b5c4"
_IDENTITY = {
    "packages": PACKAGES, "genesis": GENESIS_COMMIT, "model": MODEL,
    "model_revision": MODEL_REVISION, "gpu": "H100 80GB", "sm": [9, 0], "sm_count": 132,
    "dtype": "float16", "flash_attn_version": 2, "kv_dtype": "turboquant_k8v4", "hq": 24, "hkv": 4,
    "head_dim": 256, "page_size": 16, "key_bytes": 256, "value_bytes": 128,
}
RUNTIME_KEY = hashlib.sha256(json.dumps(_IDENTITY, sort_keys=True).encode()).hexdigest()
_PROFILE = PrefillProfile(
    RUNTIME_KEY, (PrefillRegion(32768, 65536, 1024, 4096),),
    "experimental_microbenchmark:docs/audits/2026-10-10-h100-performance/prefill-microbenchmark.json",
)
_CANDIDATE_IDENTITY = {**_IDENTITY, "packages": CANDIDATE_PACKAGES}
CANDIDATE_RUNTIME_KEY = hashlib.sha256(json.dumps(_CANDIDATE_IDENTITY, sort_keys=True).encode()).hexdigest()
_CANDIDATE_PROFILE = PrefillProfile(
    CANDIDATE_RUNTIME_KEY, _PROFILE.regions,
    "experimental_pending_serving:flashinfer-0.6.18",
)


def development_profile(name=None):
    """None selects the bundled profile only in the explicit prefill installer."""
    if name not in (None, PROFILE_NAME):
        raise ValueError(f"unknown H100 prefill profile: {name}")
    runtime_profile = os.environ.get("AQUILLM_H100_RUNTIME_PROFILE", "baseline")
    if runtime_profile in ("baseline", "native-gdn-baseline"):
        return _PROFILE
    if runtime_profile == "flashinfer-0.6.18":
        return _CANDIDATE_PROFILE
    raise ValueError(f"unknown H100 runtime profile: {runtime_profile}")


def matches_runtime(spec, dtype, properties):
    """Check actual worker metadata without CUDA reads or initialization here."""
    return (
        spec.dtype == "turboquant_k8v4" and spec.head_dim == 256
        and spec.num_q_heads == 24 and spec.num_kv_heads == 4
        and spec.block_size == 16 and spec.key_packed_size == 256 and spec.value_data_bytes == 128
        and str(dtype) == "torch.float16"
        and getattr(properties, "major", None) == 9 and getattr(properties, "minor", None) == 0
        and getattr(properties, "multi_processor_count", None) == 132
        and "H100" in getattr(properties, "name", "") and "80GB" in getattr(properties, "name", "")
    )
