"""Switch only development's main vLLM image, retaining an immutable rollback.

Runs on aquillm-dev2. Existing Compose files/env are preserved. Credentials are
compared in memory only; neither config output nor credential values are logged.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess


def run(args, env=None):
    return subprocess.check_output(args, text=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "switch", "rollback"))
    parser.add_argument("--image")
    parser.add_argument("--mtp", choices=("baseline", "fused"), default="baseline")
    args = parser.parse_args()
    if socket.gethostname() != "aquillm-dev2":
        raise SystemExit("This experiment is restricted to the authorized 254 development host")
    directory = Path.home() / ".config/aquillm/h100-performance"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = directory / "baseline.json"
    current = json.loads(run(["docker", "inspect", "compose-vllm-1"]))[0]
    labels = current["Config"]["Labels"]
    if not state_path.exists():
        if args.action != "prepare":
            raise SystemExit("Capture the baseline with prepare before switching")
        state = dict(image=current["Image"], project=labels["com.docker.compose.project"],
                     working_dir=labels["com.docker.compose.project.working_dir"],
                     files=labels["com.docker.compose.project.config_files"].split(","))
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        state_path.chmod(0o600)
    state = json.loads(state_path.read_text())
    if args.action == "prepare":
        print(json.dumps(state, indent=2))
        return
    image = state["image"] if args.action == "rollback" else args.image
    if not image:
        raise SystemExit("--image is required for switch")
    image_id = json.loads(run(["docker", "image", "inspect", image]))[0]["Id"]
    flags = dict(AQUILLM_H100_MTP_KERNEL=args.mtp if args.action == "switch" else "baseline",
                 AQUILLM_H100_SPLIT_POLICY="baseline", AQUILLM_H100_PREFILL="0",
                 AQUILLM_H100_GDN="baseline")
    previous_env = dict(item.split("=", 1) for item in current["Config"]["Env"] if "=" in item)
    # App settings may have changed in the shared .env since vLLM started.
    # Preserve this container's exact environment through subprocess-only
    # interpolation variables. The override contains names, never credentials;
    # prefixed names also avoid changing host HOME/PATH while Compose resolves.
    prefix = "AQUILLM_H100_CAPTURED_ENV_"
    inherited = {name: "${" + prefix + name + "}" for name in previous_env}
    process_env = dict(os.environ, **{prefix + name: value for name, value in previous_env.items()})
    override = directory / "current.json"
    override.write_text(json.dumps({"services": {"vllm": {"image": image_id, "environment": inherited | flags}}}, indent=2))
    compose = ["docker", "compose", "--profile", "*", "--project-name", state["project"],
               "--project-directory", state["working_dir"],
               "--env-file", "/home/exouser/AquiLLM/.env"]
    for path in state["files"]:
        compose += ["-f", path]
    compose += ["-f", str(override)]
    resolved = json.loads(run(compose + ["config", "--format", "json"], env=process_env))["services"]["vllm"]
    differences = [name for name, value in resolved.get("environment", {}).items()
                   if name in previous_env and not name.startswith("AQUILLM_H100_")
                   and str(value or "") != previous_env[name]]
    if differences:
        raise SystemExit("Compose environment drift; refusing an invalid A/B: " + ",".join(sorted(differences)))
    print(json.dumps(dict(action=args.action, image=image_id, flags=flags)), flush=True)
    subprocess.run(compose + ["up", "-d", "--no-deps", "--no-build", "vllm"], check=True, env=process_env)


if __name__ == "__main__":
    main()
