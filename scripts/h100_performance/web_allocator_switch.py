"""Digest-protected web-only allocator experiment on aquillm-dev2.

State contains digests and public allocator choices, never credentials.
The independent vLLM allocator helper and original H100 state are untouched.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import socket

import dev_switch as base
from allocator_switch import exact_image, public_environment, reject_preloads, ALLOCATOR_KEYS

WEB_IMAGE = "sha256:224d3155904e0fbb061c396e64808c2e4a0a4a7c56e58442359135041bb67bd8"
WRAPPER = ["/usr/local/bin/aquillm-allocator"]


def protected_environment(values):
    # Web has no mutable H100 controls: unlike dev_switch, protect those too.
    return {name: base.digest(str(value)) for name, value in values.items()
            if name not in ALLOCATOR_KEYS and value is not None}


def validate_identity(current, service, state=None):
    labels = current["Config"].get("Labels", {})
    if (current.get("Name") != "/compose-" + service + "-1"
            or labels.get("com.docker.compose.service") != service
            or labels.get("com.docker.compose.project") != "compose"
            or not labels.get("com.docker.compose.project.working_dir")
            or not labels.get("com.docker.compose.project.config_files")):
        raise SystemExit("Authorized container/Compose identity mismatch")
    if state is not None and (labels["com.docker.compose.project"] != state["project"]
            or labels["com.docker.compose.project.working_dir"] != state["working_dir"]):
        raise SystemExit("Compose project identity drift")


def model_digest(current):
    validate_identity(current, "vllm")
    # Capture the current arm for this operation only, not across web calls.
    return base.digest(dict(image=current["Image"], runtime=base.runtime_snapshot(current),
        environment={name: base.digest(value) for name, value in base.environment(current).items()},
        container_id=current.get("Id")))


def validate_images(original, experiment):
    if original.get("Id") != WEB_IMAGE or exact_image(experiment.get("Id")) == WEB_IMAGE:
        raise SystemExit("Experiment must extend the exact authorized web image")
    before, after = original["Config"], experiment["Config"]
    if before.get("Entrypoint") is not None or after.get("Entrypoint") != WRAPPER:
        raise SystemExit("Image Entrypoint differs from the expected web allocator wrapper")
    ignored = {"Env", "Entrypoint"}
    if original.get("Os") == experiment.get("Os") == "linux":
        ignored.add("ArgsEscaped")  # Deprecated Windows command-line metadata; retain in each pinned digest.
    if {key: value for key, value in before.items() if key not in ignored} != {
            key: value for key, value in after.items() if key not in ignored}:
        raise SystemExit("Image runtime defaults changed beyond the wrapper; preserve original Cmd")
    prior, new = base.environment(original), base.environment(experiment)
    reject_preloads(prior)
    reject_preloads(new)
    if new.get("AQUILLM_ALLOCATOR") != "system" or new.get("PYTHONMALLOC", "default") != "default":
        raise SystemExit("Image must default to system and default Python allocator")
    if {key: value for key, value in prior.items() if key not in ALLOCATOR_KEYS} != {
            key: value for key, value in new.items() if key not in ALLOCATOR_KEYS}:
        raise SystemExit("Image protected environment defaults changed")
    if any(prior[key] != new.get(key) for key in ALLOCATOR_KEYS if key in prior):
        raise SystemExit("Existing image allocator defaults changed")
    if any(original.get(key) != experiment.get(key) for key in ("Architecture", "Os", "Variant")):
        raise SystemExit("Image platform changed")
    layers = original.get("RootFS", {}).get("Layers")
    added = experiment.get("RootFS", {}).get("Layers")
    if not layers or not added or added[:len(layers)] != layers:
        raise SystemExit("Experiment image does not retain exact web filesystem layers")


def prepare_state(current, model, resolved, compose_hash, original, experiment):
    validate_identity(current, "web")
    labels = current["Config"]["Labels"]
    state = dict(project=labels["com.docker.compose.project"],
        working_dir=labels["com.docker.compose.project.working_dir"],
        files=labels["com.docker.compose.project.config_files"].split(","))
    validate_identity(model, "vllm", state)
    if current["Image"] != WEB_IMAGE or resolved.get("image") != WEB_IMAGE:
        raise SystemExit("Preparation requires the exact original web image")
    if current["Config"].get("Entrypoint") is not None:
        raise SystemExit("Original web Entrypoint is not null")
    if compose_hash != labels.get("com.docker.compose.config-hash"):
        raise SystemExit("Running resolved Compose configuration is not verified")
    values = base.environment(current)
    reject_preloads(values)
    public = public_environment(values)
    if public.get("AQUILLM_ALLOCATOR", "system") != "system" or public.get("PYTHONMALLOC", "default") != "default":
        raise SystemExit("Original web allocator environment is unsupported")
    if any(value is not None and str(value) != values.get(name)
           for name, value in resolved.get("environment", {}).items()):
        raise SystemExit("Resolved web environment differs from running baseline")
    validate_images(original, experiment)
    linux_argsescaped_ignored = original.get("Os") == experiment.get("Os") == "linux"
    runtime = copy.deepcopy(current)
    if linux_argsescaped_ignored:
        runtime["Config"].pop("ArgsEscaped", None)
    return dict(state, schema_version=1, image=WEB_IMAGE, experiment_image=experiment["Id"],
        baseline_allocator_environment=public, environment_digests=protected_environment(values),
        captured_compose_hash=compose_hash, service_digest=base.service_digest(resolved),
        runtime_digest_format="mount-order-v1", runtime_digest=base.runtime_digest(runtime),
        linux_argsescaped_ignored=linux_argsescaped_ignored,
        base_image_digest=base.digest(original["Config"]), experiment_image_digest=base.digest(experiment["Config"]))


def normalize_environment(state, values, image):
    reject_preloads(values)
    public = public_environment(values)
    if image == state["image"]:
        if public != state["baseline_allocator_environment"]:
            raise SystemExit("Original allocator environment key/value drift")
    elif image == state["experiment_image"]:
        if public.get("AQUILLM_ALLOCATOR") not in ("system", "mimalloc") or public.get("PYTHONMALLOC") != "default":
            raise SystemExit("Candidate allocator environment drift")
    else:
        raise SystemExit("Web image is not pinned in experiment state")
    if protected_environment(values) != state["environment_digests"]:
        raise SystemExit("Protected web environment key/value drift")


def validate_configuration(state, current, model, resolved=None):
    if state.get("schema_version") != 1 or state.get("image") != WEB_IMAGE:
        raise SystemExit("Invalid web allocator baseline state")
    exact_image(state["experiment_image"])
    validate_identity(current, "web", state)
    validate_identity(model, "vllm", state)
    normalize_environment(state, base.environment(current), current["Image"])
    expected_entrypoint = None if current["Image"] == WEB_IMAGE else WRAPPER
    if current["Config"].get("Entrypoint") != expected_entrypoint:
        raise SystemExit("Running web Entrypoint drift")
    normalized = copy.deepcopy(current)
    normalized["Config"]["Entrypoint"] = None
    if state.get("linux_argsescaped_ignored") is True:
        normalized["Config"].pop("ArgsEscaped", None)
    if not base.runtime_matches(state, normalized):
        raise SystemExit("Protected web command/mount/runtime drift")
    if resolved is not None:
        normalize_environment(state, resolved.get("environment", {}), resolved.get("image"))
        if base.service_digest(resolved) != state["service_digest"]:
            raise SystemExit("Resolved web command/mount configuration drift")


def make_override(state, current, model, *, allocator="system", rollback=False):
    validate_configuration(state, current, model)
    if allocator not in ("system", "mimalloc"):
        raise SystemExit("Unsupported allocator")
    values = base.environment(current)
    inherited = {name: "${" + base.PREFIX + name + "}" for name in values if name not in ALLOCATOR_KEYS}
    public = state["baseline_allocator_environment"] if rollback else dict(AQUILLM_ALLOCATOR=allocator, PYTHONMALLOC="default")
    inherited.update({name: public.get(name) for name in ALLOCATOR_KEYS})
    process = {name: value for name, value in os.environ.items() if name not in ALLOCATOR_KEYS}
    process.update({base.PREFIX + name: value for name, value in values.items()})
    return {"services": {"web": {"image": state["image"] if rollback else state["experiment_image"],
        "environment": inherited}}}, process


def canonical_hash(state, raw, process):
    fields = base.run(base.compose_command(state, []) + ["-f", "-", "config", "--hash", "web"],
        env=process, input=raw).split()
    if len(fields) != 2 or fields[0] != "web":
        raise SystemExit("Unexpected resolved web Compose hash")
    return fields[1]


def inspect(name):
    return json.loads(base.run(["docker", "inspect", "compose-" + name + "-1"]))[0]


def verify_images(state):
    original = json.loads(base.run(["docker", "image", "inspect", state["image"]]))[0]
    experiment = json.loads(base.run(["docker", "image", "inspect", state["experiment_image"]]))[0]
    validate_images(original, experiment)
    if state.get("linux_argsescaped_ignored", False) != (original.get("Os") == experiment.get("Os") == "linux"):
        raise SystemExit("Captured web platform metadata normalization mismatch")
    if (experiment.get("Id") != state["experiment_image"]
            or base.digest(original["Config"]) != state["base_image_digest"]
            or base.digest(experiment["Config"]) != state["experiment_image_digest"]):
        raise SystemExit("Pinned web image digest/config mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "switch", "rollback"))
    parser.add_argument("--image", help="exact thin web image ID; preparation only")
    parser.add_argument("--allocator", choices=("system", "mimalloc"), default="system")
    args = parser.parse_args()
    if socket.gethostname() != "aquillm-dev2":
        raise SystemExit("Restricted to the authorized 254 development host")
    if args.image is not None and args.action != "prepare":
        raise SystemExit("--image is preparation-only")
    directory = Path.home() / ".config/aquillm/h100-performance"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = directory / "web-allocator.json"
    if args.action == "prepare" and state_path.exists():
        raise SystemExit("Web allocator state already exists; recapture is forbidden")
    if args.action != "prepare" and not state_path.exists():
        raise SystemExit("Prepare separate web allocator state before switching")
    current, model = inspect("web"), inspect("vllm")
    validate_identity(current, "web")
    if args.action == "prepare":
        image = exact_image(args.image)
        labels = current["Config"]["Labels"]
        locator = dict(project=labels["com.docker.compose.project"], working_dir=labels["com.docker.compose.project.working_dir"],
            files=labels["com.docker.compose.project.config_files"].split(","))
        process = dict(os.environ, **{base.PREFIX + name: value for name, value in base.environment(current).items()})
        raw = base.run(base.compose_command(locator) + ["config", "--format", "json"], env=process)
        resolved = base.parse_config(raw)["services"]["web"]
        reference = resolved.get("image")
        if reference is None:
            implicit = locator["project"] + "-" + labels["com.docker.compose.service"]
            build = resolved.get("build")
            if (not isinstance(build, (dict, str)) or not build or implicit != "compose-web"
                    or current["Config"].get("Image") != implicit):
                raise SystemExit("Build-only web image identity is missing or ambiguous")
            reference = implicit
        if not isinstance(reference, str) or not reference:
            raise SystemExit("Configured web image reference is invalid")
        configured = json.loads(base.run(["docker", "image", "inspect", reference]))[0]
        if configured.get("Id") != WEB_IMAGE:
            raise SystemExit("Configured web image reference does not resolve to the exact authorized base")
        resolved["image"] = configured["Id"]  # Normalize only this in-memory service; raw hash input stays intact.
        original = json.loads(base.run(["docker", "image", "inspect", WEB_IMAGE]))[0]
        experiment = json.loads(base.run(["docker", "image", "inspect", image]))[0]
        if experiment.get("Id") != image:
            raise SystemExit("Supplied image did not resolve to the exact ID")
        state = prepare_state(current, model, resolved,
            canonical_hash(locator, raw, process), original, experiment)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        state_path.chmod(0o600)
        print(json.dumps(dict(action="prepare", image=image, verified=True)))
        return
    state = json.loads(state_path.read_text())
    validate_configuration(state, current, model)
    verify_images(state)
    rollback = args.action == "rollback"
    override, process = make_override(state, current, model, allocator=args.allocator, rollback=rollback)
    override_path = directory / "web-allocator-next.json"
    override_path.write_text(json.dumps(override, indent=2) + "\n")
    override_path.chmod(0o600)
    command = base.compose_command(state) + ["-f", str(override_path)]
    raw = base.run(command + ["config", "--format", "json"], env=process)
    resolved = base.parse_config(raw)["services"]["web"]
    validate_configuration(state, current, model, resolved)
    expected_image = state["image"] if rollback else state["experiment_image"]
    expected_public = state["baseline_allocator_environment"] if rollback else dict(AQUILLM_ALLOCATOR=args.allocator, PYTHONMALLOC="default")
    if resolved.get("image") != expected_image or public_environment(resolved.get("environment", {})) != expected_public:
        raise SystemExit("Resolved web allocator/image differs from requested operation")
    expected_hash = canonical_hash(state, raw, process)
    # vLLM may be switched by its own helper between operations. Guard this up only.
    before_model = inspect("vllm")
    validate_identity(before_model, "vllm", state)
    before_digest = model_digest(before_model)
    base.run(command + ["up", "-d", "--no-deps", "--no-build", "web"], env=process)
    restored, after_model = inspect("web"), inspect("vllm")
    validate_configuration(state, restored, after_model, resolved)
    if model_digest(after_model) != before_digest:
        raise SystemExit("vLLM changed during the web-only operation; inspect runtime")
    if (restored["Image"] != expected_image or public_environment(base.environment(restored)) != expected_public
            or restored["Config"]["Labels"].get("com.docker.compose.config-hash") != expected_hash):
        raise SystemExit("Post-switch exact web image/environment/config verification failed")
    print(json.dumps(dict(action=args.action, image=expected_image, allocator=None if rollback else args.allocator, verified=True)))


if __name__ == "__main__":
    main()
