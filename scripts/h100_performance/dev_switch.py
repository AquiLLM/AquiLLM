"""Digest-only verified baseline. Drift blocks both switch and rollback.

Secrets stay in memory: a digest cannot reconstruct changed credentials/config.
Restore the original configuration before retrying any failed preflight.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import subprocess

FLAGS = frozenset(("AQUILLM_H100_MTP_KERNEL", "AQUILLM_H100_SPLIT_POLICY", "AQUILLM_H100_PREFILL", "AQUILLM_H100_GDN", "AQUILLM_H100_RUNTIME_PROFILE"))
PREFIX = "AQUILLM_H100_CAPTURED_ENV_"
FLAG_VALUES = {"AQUILLM_H100_MTP_KERNEL": {"baseline", "fused"},
               "AQUILLM_H100_SPLIT_POLICY": {"baseline", "adaptive"},
               "AQUILLM_H100_PREFILL": {"0", "1"},
               "AQUILLM_H100_GDN": {"baseline", "flashinfer"},
               "AQUILLM_H100_RUNTIME_PROFILE": {"baseline", "flashinfer-0.6.18"}}


def run(args, env=None, input=None):
    try:
        return subprocess.check_output(args, text=True, env=env, input=input, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        raise SystemExit("Docker/Compose failed; diagnostics suppressed to protect credentials") from None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_config(raw):
    # Compose config escapes every dollar for safe reuse as a Compose input.
    # Compare the actual values in memory, rather than their serialized escapes.
    return json.loads(raw.replace("$$", "$"))


def canonical_hash(state, raw_config, process_env):
    """Hash the resolved model, matching the model stamped by Compose up.

    Older Compose config --hash leaves env_file unresolved; up resolves it.
    Feed config's reusable JSON directly to stdin, never to disk or logs.
    """
    command = compose_command(state, []) + ["-f", "-", "config", "--hash", "vllm"]
    fields = run(command, env=process_env, input=raw_config).split()
    if len(fields) != 2 or fields[0] != "vllm":
        raise SystemExit("Unexpected resolved Compose hash output; refusing preparation")
    return fields[1]


def environment(current):
    return dict(item.split("=", 1) for item in current["Config"].get("Env") or [] if "=" in item)


def protected_environment(values):
    return {name: digest(str(value)) for name, value in values.items() if name not in FLAGS and value is not None}


def service_digest(service):
    return digest({key: value for key, value in service.items() if key not in ("image", "environment")})


def runtime_snapshot(current, *, canonical=True):
    config = {key: value for key, value in current["Config"].items() if key not in ("Image", "Env", "Labels", "Hostname")}
    host = dict(current["HostConfig"]) if isinstance(current.get("HostConfig"), dict) else current.get("HostConfig")
    mounts = current.get("Mounts")
    if canonical:
        order = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        if isinstance(mounts, list):
            mounts = sorted(mounts, key=order)
        if isinstance(host, dict):
            # Compose builds these two mount collections from a Go map as well.
            for name in ("Binds", "Mounts"):
                if isinstance(host.get(name), list):
                    host[name] = sorted(host[name], key=order)
    return dict(config=config, host=host, mounts=mounts)


def runtime_digest(current, *, canonical=True):
    return digest(runtime_snapshot(current, canonical=canonical))


def runtime_matches(state, current):
    format_name = state.get("runtime_digest_format")
    if format_name == "mount-order-v1":
        return runtime_digest(current) == state["runtime_digest"]
    if format_name is not None:
        return False
    snapshot = runtime_snapshot(current, canonical=False)
    if digest(snapshot) == state["runtime_digest"]:
        return True
    # Legacy states contain only a digest. Prove the exact historical digest
    # by reordering mount collections, retaining every protected record/value.
    # This admits no changed mounts, commands, logging, or other runtime fields.
    locations = []
    if isinstance(snapshot["mounts"], list):
        locations.append((snapshot, "mounts"))
    if isinstance(snapshot["host"], dict):
        for name in ("Binds", "Mounts"):
            if isinstance(snapshot["host"].get(name), list):
                locations.append((snapshot["host"], name))
    if math.prod(math.factorial(len(owner[name])) for owner, name in locations) > 4096:
        return False
    choices = [itertools.permutations(owner[name]) for owner, name in locations]
    for combination in itertools.product(*choices):
        for (owner, name), values in zip(locations, combination):
            owner[name] = list(values)
        if digest(snapshot) == state["runtime_digest"]:
            return True
    return False


def prepare_state(state, current, resolved, compose_hash, *, verified=False):
    if not verified:
        raise SystemExit("prepare requires --verify-current-baseline after verifying the restored baseline configuration")
    if current["Image"] != state["image"]:
        raise SystemExit("Preparation/migration requires the captured original image; refusing candidate recapture")
    if compose_hash != current["Config"]["Labels"].get("com.docker.compose.config-hash"):
        raise SystemExit("Running Compose configuration is unverified or changed; refusing preparation")
    values = environment(current)
    if any(values[name] not in FLAG_VALUES[name] for name in FLAGS if name in values):
        raise SystemExit("Baseline experiment flags contain unsupported values; refusing to persist them")
    for name, value in resolved.get("environment", {}).items():
        if value is not None and str(value) != values.get(name):
            raise SystemExit("Resolved environment differs from running baseline; refusing preparation")
    if state.get("schema_version") == 2:
        validate_configuration(state, current, resolved, check_resolved_environment=False)
        return dict(state, runtime_digest_format="mount-order-v1", runtime_digest=runtime_digest(current))
    return dict(state, schema_version=2, environment_digests=protected_environment(values),
                baseline_flags={name: value for name, value in values.items() if name in FLAGS},
                service_digest=service_digest(resolved), runtime_digest_format="mount-order-v1", runtime_digest=runtime_digest(current))


def validate_configuration(state, current, resolved, *, check_resolved_environment=True):
    if state.get("schema_version") != 2:
        raise SystemExit("Legacy baseline lacks protected configuration; run verified prepare on the original image")
    if protected_environment(environment(current)) != state["environment_digests"]:
        raise SystemExit("Running protected environment drift; restore original configuration before switch or rollback")
    if check_resolved_environment and protected_environment(resolved.get("environment", {})) != state["environment_digests"]:
        raise SystemExit("Compose protected environment key/value drift; refusing switch or rollback")
    if service_digest(resolved) != state["service_digest"] or not runtime_matches(state, current):
        raise SystemExit("Protected command/mount/runtime configuration drift; refusing switch or rollback")


def make_override(state, current, image, *, rollback=False, mtp="baseline", prefill="0", runtime_profile="baseline", gdn="baseline"):
    if state.get("schema_version") != 2 or protected_environment(environment(current)) != state.get("environment_digests"):
        raise SystemExit("Protected baseline environment unavailable; cannot safely reconstruct rollback")
    values = environment(current)
    inherited = {name: "${" + PREFIX + name + "}" for name in values if name not in FLAGS}
    flags = state["baseline_flags"] if rollback else dict(AQUILLM_H100_MTP_KERNEL=mtp,
        AQUILLM_H100_SPLIT_POLICY="baseline", AQUILLM_H100_PREFILL=prefill, AQUILLM_H100_GDN=gdn,
        AQUILLM_H100_RUNTIME_PROFILE=runtime_profile)
    if any(value not in FLAG_VALUES[name] for name, value in flags.items()):
        raise SystemExit("Unsupported experiment flag value; refusing operation")
    inherited.update({name: flags.get(name) for name in FLAGS})
    process_env = {name: value for name, value in os.environ.items() if name not in FLAGS}
    process_env.update({PREFIX + name: value for name, value in values.items()})
    return {"services": {"vllm": {"image": image, "environment": inherited}}}, process_env


def compose_command(state, files=None):
    command = ["docker", "compose", "--profile", "*", "--project-name", state["project"],
               "--project-directory", state["working_dir"], "--env-file", "/home/exouser/AquiLLM/.env"]
    for path in state["files"] if files is None else files:
        command += ["-f", path]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "switch", "rollback"))
    parser.add_argument("--image")
    parser.add_argument("--mtp", choices=("baseline", "fused"), default="baseline")
    parser.add_argument("--prefill", choices=("baseline", "0", "1"), default="baseline")
    parser.add_argument("--runtime-profile", choices=("baseline", "flashinfer-0.6.18"), default="baseline")
    parser.add_argument("--gdn", choices=("baseline", "flashinfer"), default="baseline")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--verify-current-baseline", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != "aquillm-dev2":
        raise SystemExit("This experiment is restricted to the authorized 254 development host")
    directory = args.state_dir or Path.home() / ".config/aquillm/h100-performance"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = directory / "baseline.json"
    current = json.loads(run(["docker", "inspect", "compose-vllm-1"]))[0]
    labels = current["Config"]["Labels"]
    if state_path.exists():
        state = json.loads(state_path.read_text())
    elif args.action == "prepare":
        state = dict(image=current["Image"], project=labels["com.docker.compose.project"],
                     working_dir=labels["com.docker.compose.project.working_dir"],
                     files=labels["com.docker.compose.project.config_files"].split(","))
    else:
        raise SystemExit("Capture the baseline with verified prepare before switching")
    if labels["com.docker.compose.project"] != state["project"] or labels["com.docker.compose.project.working_dir"] != state["working_dir"]:
        raise SystemExit("Compose project identity drift; refusing operation")
    if args.action == "prepare":
        command = compose_command(state, labels["com.docker.compose.project.config_files"].split(","))
        process_env = dict(os.environ, **{PREFIX + name: value for name, value in environment(current).items()})
        raw_config = run(command + ["config", "--format", "json"], env=process_env)
        resolved = parse_config(raw_config)["services"]["vllm"]
        state = prepare_state(state, current, resolved, canonical_hash(state, raw_config, process_env), verified=args.verify_current_baseline)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        state_path.chmod(0o600)
        print(json.dumps(dict(action="prepare", image=state["image"], schema_version=2, verified=True)))
        return
    if state.get("schema_version") != 2:
        raise SystemExit("Legacy baseline must be migrated with verified prepare on the restored original image")
    image = state["image"] if args.action == "rollback" else args.image
    if not image:
        raise SystemExit("--image is required for switch")
    image_info = json.loads(run(["docker", "image", "inspect", image]))[0]
    image_id = image_info["Id"]
    baseline_info = json.loads(run(["docker", "image", "inspect", state["image"]]))[0]
    if protected_environment(environment(image_info)) != protected_environment(environment(baseline_info)):
        raise SystemExit("Image protected environment key/value defaults changed; refusing before container replacement")
    keys = ("Cmd", "Entrypoint", "User", "WorkingDir", "Healthcheck", "ExposedPorts", "Volumes", "StopSignal", "Shell")
    if any(image_info["Config"].get(key) != baseline_info["Config"].get(key) for key in keys):
        raise SystemExit("Image runtime defaults changed; refusing an invalid image-only comparison")
    override_data, process_env = make_override(state, current, image_id, rollback=args.action == "rollback", mtp=args.mtp,
                                              prefill="0" if args.prefill == "baseline" else args.prefill,
                                              runtime_profile=args.runtime_profile, gdn=args.gdn)
    override = directory / "next.json"
    override.write_text(json.dumps(override_data, indent=2))
    override.chmod(0o600)
    command = compose_command(state) + ["-f", str(override)]
    resolved = parse_config(run(command + ["config", "--format", "json"], env=process_env))["services"]["vllm"]
    validate_configuration(state, current, resolved)
    run(command + ["up", "-d", "--no-deps", "--no-build", "vllm"], env=process_env)
    restored = json.loads(run(["docker", "inspect", "compose-vllm-1"]))[0]
    validate_configuration(state, restored, resolved)
    expected = state["baseline_flags"] if args.action == "rollback" else override_data["services"]["vllm"]["environment"]
    expected = {name: value for name, value in expected.items() if name in FLAGS and value is not None}
    actual = {name: value for name, value in environment(restored).items() if name in FLAGS}
    if restored["Image"] != image_id or actual != expected:
        raise SystemExit("Post-switch image/flag verification failed; runtime requires inspection")
    print(json.dumps(dict(action=args.action, image=image_id, verified=True)), flush=True)


if __name__ == "__main__":
    main()
