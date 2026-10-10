#!/usr/bin/env python3
"""Opt-in K8V4 capacity planning; estimates never establish runtime readiness.

Additive fields: retained_full_attention_payload_bytes estimates C contexts;
tier_budget_requests preserves absent (null), auto, and explicit GiB requests.
tier_budgets contains integer byte caps or null (unresolved) for vram/ram/ssd.
The --plan CLI can serialize unsupported modes, but cannot emit launch argv.
"""
from __future__ import annotations

import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation, localcontext
from typing import Mapping, Sequence

MODEL = "hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16"
SETTINGS = (
    "KV_CACHE_TARGET_ACTIVE_SEQUENCES", "KV_CACHE_RETAINED_CONTEXTS",
    "KV_CACHE_VRAM_GIB", "KV_CACHE_RAM_GIB", "KV_CACHE_SSD_GIB",
    "KV_CACHE_STORAGE_MODE", "KV_CACHE_EXECUTION_MODE",
)
PROFILE_FLAGS = {"--max-num-seqs", "--max-model-len", "--kv-cache-dtype", "--kv-cache-memory-bytes"}
WEIGHT_FLAGS = {"--cpu-offload-gb", "--offload-group-size", "--offload-num-in-group", "--offload-prefetch-step"}
HYBRID_FLAGS = {"--disable-hybrid-kv-cache-manager", "--kv-transfer-config", "--offload-params"}
GUARDED_FLAGS = PROFILE_FLAGS | WEIGHT_FLAGS | HYBRID_FLAGS | {
    "--model", "--hf-overrides", "--config", "--kv-offloading",
}
MAX_BYTES = (1 << 63) - 1


class ConfigurationError(ValueError):
    """Requested profile is invalid or cannot safely be launched."""


def _count(value: str, name: str, limit: int = (1 << 31) - 1) -> int:
    if len(value) > len(str(limit)) or not re.fullmatch(r"[0-9]+", value):
        raise ConfigurationError(f"{name} must be a bounded positive integer")
    count = int(value)
    if not 1 <= count <= limit:
        raise ConfigurationError(f"{name} must be in [1, {limit}]")
    return count


def _decimal(value: str, name: str) -> Decimal:
    if len(value) > 128:
        raise ConfigurationError(f"{name} must be a finite nonnegative bounded number")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ConfigurationError(f"{name} must be a finite nonnegative number") from exc
    if not number.is_finite() or number < 0 or number > MAX_BYTES:
        raise ConfigurationError(f"{name} must be a finite nonnegative bounded number")
    return number


def _budget(value: str | None, name: str) -> int | None:
    if value is None or value == "auto":
        return None
    with localcontext() as context:
        context.prec = 160
        size = _decimal(value, name) * (1 << 30)
        if size > MAX_BYTES:
            raise ConfigurationError(f"{name} exceeds the signed 64-bit byte limit")
        if size != size.to_integral_value():
            raise ConfigurationError(f"{name} must resolve to whole bytes")
        return int(size)


def _option_parts(token: str) -> tuple[str, str, str, bool]:
    """Match vLLM's underscore normalization and dotted dictionary roots.

    Only the option root is normalized; dictionary keys and values are data.
    The same root interpretation is used for validation and argv removal.
    """
    name, equals, value = token.partition("=")
    root, dotted, _ = name.partition(".")
    if root.startswith("--"):
        root = root.replace("_", "-")
    return root, equals, value, bool(dotted)


def _options(args: Sequence[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    index = 0
    while index < len(args):
        token = args[index]
        name, equals, value, dotted = _option_parts(token)
        if name == "--config" or (name.startswith("-c") and not name.startswith("--")):
            raise ConfigurationError("--config/-c files are not allowed with a capacity profile; provide explicit full-name CLI options")
        if name.startswith("--") and name != "--" and name not in GUARDED_FLAGS:
            if any(flag.startswith(name) for flag in GUARDED_FLAGS):
                raise ConfigurationError(f"abbreviated guarded option {name} is not allowed; use the full option name")
        if name == "--hf-overrides":
            raise ConfigurationError("--hf-overrides can change the approved model layout; remove it for this capacity profile")
        if name.startswith("--kv-offloading") or name in HYBRID_FLAGS:
            raise ConfigurationError(f"{name} is incompatible with this hybrid profile; no adapter is validated")
        if name in PROFILE_FLAGS | WEIGHT_FLAGS | {"--model"}:
            if dotted:
                raise ConfigurationError(f"{name} is a scalar option; dotted overrides are not allowed")
            if name in options:
                raise ConfigurationError(f"duplicate {name} is not allowed in a capacity profile")
            if not equals:
                index += 1
                if index == len(args) or args[index].startswith("--"):
                    raise ConfigurationError(f"{name} requires a value")
                value = args[index]
            if not value:
                raise ConfigurationError(f"{name} requires a value")
            options[name] = value
            if name in WEIGHT_FLAGS and _decimal(value, name) != 0:
                raise ConfigurationError(f"{name} must be zero: all model weights must remain GPU resident")
        index += 1
    return options


def _launch_modes(profile: dict) -> None:
    if profile["storage_mode"] != "off":
        raise ConfigurationError(f"storage mode {profile['storage_mode']} has no validated hybrid K8V4 adapter; use off or --plan")
    if profile["execution_mode"] != "resident":
        raise ConfigurationError("paged execution has no validated adapter; use resident or --plan")


def _resolve(env: Mapping[str, str], extra_args: Sequence[str], *, planning: bool) -> dict | None:
    if not any(env.get(name, "") for name in SETTINGS):
        return None
    opts = _options(extra_args)
    if env.get("VLLM_MODEL", MODEL) != MODEL or opts.get("--model", MODEL) != MODEL:
        raise ConfigurationError(f"capacity layout is only defined for {MODEL}")
    n = _count(env.get("KV_CACHE_TARGET_ACTIVE_SEQUENCES") or "4", "KV_CACHE_TARGET_ACTIVE_SEQUENCES")
    c = _count(env.get("KV_CACHE_RETAINED_CONTEXTS") or str(n), "KV_CACHE_RETAINED_CONTEXTS")
    if c < n:
        raise ConfigurationError("KV_CACHE_RETAINED_CONTEXTS must be at least KV_CACHE_TARGET_ACTIVE_SEQUENCES")
    t = _count(env.get("VLLM_MAX_MODEL_LEN") or opts.get("--max-model-len", "262144"), "VLLM_MAX_MODEL_LEN", 262144)
    for flag, expected in (("--max-num-seqs", n), ("--max-model-len", t)):
        if flag in opts and _count(opts[flag], flag) != expected:
            raise ConfigurationError(f"{flag} conflicts with the requested capacity profile ({expected})")
    if opts.get("--kv-cache-dtype", "turboquant_k8v4") != "turboquant_k8v4":
        raise ConfigurationError("--kv-cache-dtype must be turboquant_k8v4 for this profile")
    payload = n * t * 16 * 4 * 388
    retained_payload = c * t * 16 * 4 * 388
    if max(payload, retained_payload) > MAX_BYTES:
        raise ConfigurationError("KV_CACHE_TARGET_ACTIVE_SEQUENCES / KV_CACHE_RETAINED_CONTEXTS payload overflows signed 64-bit bytes")
    requests = {tier: env.get(f"KV_CACHE_{tier.upper()}_GIB") or None for tier in ("vram", "ram", "ssd")}
    budgets = {tier: _budget(value, f"KV_CACHE_{tier.upper()}_GIB") for tier, value in requests.items()}
    if budgets["vram"] == 0:
        raise ConfigurationError("KV_CACHE_VRAM_GIB must be positive; zero would select automatic allocation rather than enforce a cap")
    if "--kv-cache-memory-bytes" in opts:
        size = _count(opts["--kv-cache-memory-bytes"], "--kv-cache-memory-bytes", MAX_BYTES)
        if budgets["vram"] is not None and size != budgets["vram"]:
            raise ConfigurationError("--kv-cache-memory-bytes conflicts with KV_CACHE_VRAM_GIB")
        budgets["vram"] = size
    storage = env.get("KV_CACHE_STORAGE_MODE") or "off"
    execution = env.get("KV_CACHE_EXECUTION_MODE") or "resident"
    if storage not in {"off", "local", "mooncake"}:
        raise ConfigurationError("KV_CACHE_STORAGE_MODE must be off, local, or mooncake")
    if execution not in {"resident", "paged"}:
        raise ConfigurationError("KV_CACHE_EXECUTION_MODE must be resident or paged")
    profile = dict(active_sequences=n, context_tokens=t, retained_contexts=c,
                   kv_dtype="turboquant_k8v4", full_attention_payload_bytes=payload,
                   retained_full_attention_payload_bytes=retained_payload,
                   storage_mode=storage, execution_mode=execution,
                   tier_budgets=budgets, tier_budget_requests=requests,
                   validation_status="unvalidated")
    if not planning:
        _launch_modes(profile)
    return profile


def resolve_profile(env: Mapping[str, str], extra_args: Sequence[str]) -> dict | None:
    return _resolve(env, extra_args, planning=False)


def profile_arguments(profile: dict) -> list[str]:
    _launch_modes(profile)
    args = ["--max-num-seqs", str(profile["active_sequences"]),
            "--max-model-len", str(profile["context_tokens"]),
            "--kv-cache-dtype", profile["kv_dtype"]]
    if profile["tier_budgets"]["vram"] is not None:
        args.extend(["--kv-cache-memory-bytes", str(profile["tier_budgets"]["vram"])])
    return args


def _without_profile_flags(args: Sequence[str]) -> list[str]:
    result = []
    index = 0
    while index < len(args):
        name, equals, _, _ = _option_parts(args[index])
        if name in PROFILE_FLAGS:
            index += 1 if equals else 2
        else:
            result.append(args[index])
            index += 1
    return result


def main() -> int:
    argv = sys.argv[1:]
    planning = bool(argv and argv[0] == "--plan")
    launch = bool(argv and argv[0] == "--launch-argv")
    if planning or launch:
        argv = argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    try:
        profile = _resolve(os.environ, argv, planning=planning)
        if launch:
            args = argv if profile is None else _without_profile_flags(argv) + profile_arguments(profile)
            if profile is not None:
                print("KV capacity plan (unvalidated; excludes hybrid/runtime overhead): " + json.dumps(profile, sort_keys=True), file=sys.stderr)
            for arg in args:
                sys.stdout.buffer.write(arg.encode("utf-8") + b"\0")
        else:
            print(json.dumps(profile, sort_keys=True))
    except ConfigurationError as exc:
        print(f"ERROR: KV capacity configuration: {exc}", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
