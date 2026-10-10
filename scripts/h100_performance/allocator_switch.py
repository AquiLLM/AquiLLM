"""Development-only, digest-protected allocator comparison on one pinned image.

Original H100 baseline state is read and proved, never modified. Credentials
remain in memory; state and overrides contain only digests or public choices.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import socket

import dev_switch as base

PREFILL_IMAGE = "sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7"
ENTRYPOINT = ["/genesis_entrypoint.sh"]
WRAPPER = ["/usr/local/bin/aquillm-allocator", "/genesis_entrypoint.sh"]
ALLOCATOR_KEYS = frozenset(("AQUILLM_ALLOCATOR", "PYTHONMALLOC"))
PREFILL_FLAGS = dict(AQUILLM_H100_MTP_KERNEL="baseline", AQUILLM_H100_SPLIT_POLICY="baseline",
                     AQUILLM_H100_PREFILL="1", AQUILLM_H100_GDN="baseline")


def exact_image(value):
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise SystemExit("An exact sha256 image ID is required; tags and abbreviated IDs are refused")
    return value


def reject_preloads(values):
    if any(values.get(name) for name in ("LD_PRELOAD", "LD_AUDIT")):
        raise SystemExit("External allocator preloads/audit libraries are forbidden")


def public_environment(values):
    return {name: values[name] for name in ALLOCATOR_KEYS if name in values and values[name] is not None}


def protected_environment(values):
    return base.protected_environment({name: value for name, value in values.items() if name not in ALLOCATOR_KEYS})


def validate_images(prefill, experiment):
    if prefill.get("Id") != PREFILL_IMAGE or exact_image(experiment.get("Id")) == PREFILL_IMAGE:
        raise SystemExit("Allocator image must extend the exact authorized prefill image")
    original, candidate = prefill["Config"], experiment["Config"]
    if original.get("Entrypoint") != ENTRYPOINT or candidate.get("Entrypoint") != WRAPPER:
        raise SystemExit("Image Entrypoint differs from the expected allocator wrapper and Genesis command")
    if {key: value for key, value in original.items() if key not in ("Env", "Entrypoint")} != {
            key: value for key, value in candidate.items() if key not in ("Env", "Entrypoint")}:
        raise SystemExit("Image runtime defaults changed beyond the allocator wrapper")
    before, after = base.environment(prefill), base.environment(experiment)
    reject_preloads(before)
    reject_preloads(after)
    if after.get("AQUILLM_ALLOCATOR") != "system" or after.get("PYTHONMALLOC", "default") != "default":
        raise SystemExit("Experiment image must default to the system allocator and default Python allocator")
    if {name: value for name, value in before.items() if name not in ALLOCATOR_KEYS} != {
            name: value for name, value in after.items() if name not in ALLOCATOR_KEYS}:
        raise SystemExit("Image protected environment defaults changed")
    for name in ALLOCATOR_KEYS:
        if name in before and before[name] != after.get(name):
            raise SystemExit("Existing image allocator defaults changed")
    for name in ("Architecture", "Os", "Variant"):
        if prefill.get(name) != experiment.get(name):
            raise SystemExit("Image platform changed")
    prior_layers = prefill.get("RootFS", {}).get("Layers")
    new_layers = experiment.get("RootFS", {}).get("Layers")
    if not prior_layers or not new_layers or new_layers[:len(prior_layers)] != prior_layers:
        raise SystemExit("Experiment image does not retain the exact prefill filesystem layers")


def validate_identity(original, current):
    labels = current["Config"].get("Labels", {})
    if (current.get("Name") != "/compose-vllm-1" or labels.get("com.docker.compose.service") != "vllm"
            or labels.get("com.docker.compose.project") != original["project"]
            or labels.get("com.docker.compose.project.working_dir") != original["working_dir"]):
        raise SystemExit("Main vLLM container/Compose project identity drift")


def prepare_state(original, current, resolved, compose_hash, prefill, experiment):
    validate_identity(original, current)
    if current["Image"] != PREFILL_IMAGE or prefill.get("Id") != PREFILL_IMAGE:
        raise SystemExit("Preparation requires the exact authorized prefill image, not an allocator candidate")
    if compose_hash != current["Config"]["Labels"].get("com.docker.compose.config-hash"):
        raise SystemExit("Running resolved Compose configuration is not verified")
    # This proves the original protected baseline without recapturing its state.
    base.validate_configuration(original, current, resolved)
    values = base.environment(current)
    reject_preloads(values)
    if {name: values[name] for name in base.FLAGS if name in values} != PREFILL_FLAGS:
        raise SystemExit("Preparation requires prefill=1 with other H100 controls at baseline")
    public = public_environment(values)
    if public.get("AQUILLM_ALLOCATOR", "system") != "system" or public.get("PYTHONMALLOC", "default") != "default":
        raise SystemExit("Prefill baseline has unsupported allocator environment")
    validate_images(prefill, experiment)
    return dict(schema_version=1, image=PREFILL_IMAGE, experiment_image=experiment["Id"],
        project=original["project"], working_dir=original["working_dir"], files=list(original["files"]),
        original_state_digest=base.digest(original), environment_digests=protected_environment(values),
        baseline_flags=PREFILL_FLAGS.copy(), baseline_allocator_environment=public,
        service_digest=base.service_digest(resolved), runtime_digest_format="mount-order-v1",
        runtime_digest=base.runtime_digest(current),
        prefill_image_digest=base.digest(prefill["Config"]), experiment_image_digest=base.digest(experiment["Config"]))


def normalize_environment(state, values, image):
    reject_preloads(values)
    if {name: values[name] for name in base.FLAGS if name in values} != PREFILL_FLAGS:
        raise SystemExit("H100 flag drift; both allocator arms require the captured prefill configuration")
    public = public_environment(values)
    if image == state["image"]:
        if public != state["baseline_allocator_environment"]:
            raise SystemExit("Prefill allocator environment key/value drift")
    elif image == state["experiment_image"]:
        if public.get("AQUILLM_ALLOCATOR") not in ("system", "mimalloc") or public.get("PYTHONMALLOC") != "default":
            raise SystemExit("Candidate allocator environment drift")
    else:
        raise SystemExit("Running/resolved image is not pinned in allocator experiment state")
    if protected_environment(values) != state["environment_digests"]:
        raise SystemExit("Protected environment key/value drift; switch and rollback refused")
    normalized = {name: value for name, value in values.items() if name not in ALLOCATOR_KEYS}
    normalized.update(state["baseline_allocator_environment"])
    return normalized


def validate_configuration(state, original, current, resolved=None):
    if (state.get("schema_version") != 1 or state.get("image") != PREFILL_IMAGE
            or state.get("original_state_digest") != base.digest(original)):
        raise SystemExit("Allocator/original baseline state mismatch; refusing switch and rollback")
    exact_image(state["experiment_image"])
    validate_identity(original, current)
    normalized = copy.deepcopy(current)
    values = normalize_environment(state, base.environment(current), current["Image"])
    expected_entrypoint = ENTRYPOINT if current["Image"] == state["image"] else WRAPPER
    if current["Config"].get("Entrypoint") != expected_entrypoint:
        raise SystemExit("Running Entrypoint drift")
    normalized["Config"]["Entrypoint"] = ENTRYPOINT.copy()
    normalized["Config"]["Env"] = [f"{name}={value}" for name, value in values.items()]
    if (base.protected_environment(values) != original.get("environment_digests")
            or not base.runtime_matches(original, normalized) or not base.runtime_matches(state, normalized)):
        raise SystemExit("Protected original environment/command/mount/runtime configuration drift")
    if resolved is not None:
        normalized_service = copy.deepcopy(resolved)
        normalized_service["environment"] = normalize_environment(state, resolved.get("environment", {}), resolved.get("image"))
        if base.service_digest(resolved) != state["service_digest"]:
            raise SystemExit("Resolved service command/mount configuration drift")
        base.validate_configuration(original, normalized, normalized_service)


def make_override(state, original, current, *, allocator="system", rollback=False):
    validate_configuration(state, original, current)
    if allocator not in ("system", "mimalloc"):
        raise SystemExit("Unsupported allocator choice")
    values = base.environment(current)
    inherited = {name: "${" + base.PREFIX + name + "}" for name in values
                 if name not in base.FLAGS and name not in ALLOCATOR_KEYS}
    inherited.update(state["baseline_flags"])
    public = state["baseline_allocator_environment"] if rollback else dict(AQUILLM_ALLOCATOR=allocator, PYTHONMALLOC="default")
    inherited.update({name: public.get(name) for name in ALLOCATOR_KEYS})
    process = {name: value for name, value in os.environ.items() if name not in base.FLAGS and name not in ALLOCATOR_KEYS}
    process.update({base.PREFIX + name: value for name, value in values.items()})
    image = state["image"] if rollback else state["experiment_image"]
    return {"services": {"vllm": {"image": image, "environment": inherited}}}, process


def verify_images(state):
    prefill = json.loads(base.run(["docker", "image", "inspect", state["image"]]))[0]
    experiment = json.loads(base.run(["docker", "image", "inspect", state["experiment_image"]]))[0]
    validate_images(prefill, experiment)
    if (base.digest(prefill["Config"]) != state["prefill_image_digest"]
            or base.digest(experiment["Config"]) != state["experiment_image_digest"]):
        raise SystemExit("Pinned image configuration digest mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "switch", "rollback"))
    parser.add_argument("--image", help="exact thin experiment image ID; preparation only")
    parser.add_argument("--allocator", choices=("system", "mimalloc"), default="system")
    args = parser.parse_args()
    if socket.gethostname() != "aquillm-dev2":
        raise SystemExit("This experiment is restricted to the authorized 254 development host")
    if args.image is not None and args.action != "prepare":
        raise SystemExit("--image is preparation-only; both arms use the state-pinned image")
    directory = Path.home() / ".config/aquillm/h100-performance"
    original_path, state_path = directory / "baseline.json", directory / "allocator.json"
    if not original_path.exists():
        raise SystemExit("Verified original H100 baseline state is required")
    original = json.loads(original_path.read_text())
    current = json.loads(base.run(["docker", "inspect", "compose-vllm-1"]))[0]
    validate_identity(original, current)
    if args.action == "prepare":
        if state_path.exists():
            raise SystemExit("Allocator experiment state already exists; candidate recapture is forbidden")
        image = exact_image(args.image)
        labels = current["Config"]["Labels"]
        command = base.compose_command(original, labels["com.docker.compose.project.config_files"].split(","))
        process = dict(os.environ, **{base.PREFIX + name: value for name, value in base.environment(current).items()})
        raw = base.run(command + ["config", "--format", "json"], env=process)
        resolved = base.parse_config(raw)["services"]["vllm"]
        prefill = json.loads(base.run(["docker", "image", "inspect", PREFILL_IMAGE]))[0]
        experiment = json.loads(base.run(["docker", "image", "inspect", image]))[0]
        if experiment.get("Id") != image:
            raise SystemExit("Experiment image inspection did not resolve the supplied exact ID")
        state = prepare_state(original, current, resolved, base.canonical_hash(original, raw, process), prefill, experiment)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        state_path.chmod(0o600)
        print(json.dumps(dict(action="prepare", image=image, verified=True)))
        return
    if not state_path.exists():
        raise SystemExit("Prepare separate allocator state before switching")
    state = json.loads(state_path.read_text())
    validate_configuration(state, original, current)
    verify_images(state)
    rollback = args.action == "rollback"
    override, process = make_override(state, original, current, allocator=args.allocator, rollback=rollback)
    override_path = directory / "allocator-next.json"
    override_path.write_text(json.dumps(override, indent=2) + "\n")
    override_path.chmod(0o600)
    command = base.compose_command(state) + ["-f", str(override_path)]
    raw = base.run(command + ["config", "--format", "json"], env=process)
    resolved = base.parse_config(raw)["services"]["vllm"]
    validate_configuration(state, original, current, resolved)
    expected_image = state["image"] if rollback else state["experiment_image"]
    expected_public = state["baseline_allocator_environment"] if rollback else dict(AQUILLM_ALLOCATOR=args.allocator, PYTHONMALLOC="default")
    if resolved.get("image") != expected_image or public_environment(resolved.get("environment", {})) != expected_public:
        raise SystemExit("Resolved allocator/image selection differs from requested operation")
    expected_hash = base.canonical_hash(state, raw, process)
    base.run(command + ["up", "-d", "--no-deps", "--no-build", "vllm"], env=process)
    restored = json.loads(base.run(["docker", "inspect", "compose-vllm-1"]))[0]
    validate_configuration(state, original, restored, resolved)
    if (restored["Image"] != expected_image or public_environment(base.environment(restored)) != expected_public
            or restored["Config"]["Labels"].get("com.docker.compose.config-hash") != expected_hash):
        raise SystemExit("Post-switch exact image/environment/config verification failed; inspect runtime")
    print(json.dumps(dict(action=args.action, image=expected_image, allocator=None if rollback else args.allocator, verified=True)))


if __name__ == "__main__":
    main()
