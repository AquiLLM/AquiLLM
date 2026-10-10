"""Capture allowlisted serving identity without emitting credentials or prompts."""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SAFE_ENV = {
    "VLLM_MODEL", "VLLM_SERVED_MODEL_NAME", "VLLM_DTYPE", "VLLM_KV_CACHE_DTYPE",
    "VLLM_MAX_MODEL_LEN", "VLLM_MAX_NUM_SEQS", "VLLM_MAX_NUM_BATCHED_TOKENS",
    "VLLM_GPU_MEMORY_UTILIZATION", "VLLM_TENSOR_PARALLEL_SIZE", "VLLM_SPECULATIVE_CONFIG",
    "VLLM_ENABLE_PREFIX_CACHING", "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ATTENTION_BACKEND",
    "CUDA_VISIBLE_DEVICES", "VLLM_REVISION", "VLLM_QUANTIZATION",
}


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def snapshot(container):
    record = json.loads(command("docker", "inspect", container))[0]
    config = record["Config"]
    env = dict(item.split("=", 1) for item in config["Env"] if "=" in item)
    safe = {k: v for k, v in env.items()
            if k in SAFE_ENV or k.startswith(("GENESIS_P", "GENESIS_ENABLE_", "AQUILLM_H100_"))
            or k in {"GENESIS_ENFORCE_VERSION_RANGE", "VLLM_GENESIS_BASE_IMAGE", "GENESIS_REF"}}
    probe = '''import importlib.metadata as m, json, subprocess, torch
packages = {}
for p in ["vllm", "torch", "triton", "flashinfer-python", "flash-attn"]:
 try: packages[p] = m.version(p)
 except m.PackageNotFoundError: packages[p] = None
try: genesis = subprocess.check_output(["git", "-C", "/opt/genesis", "rev-parse", "HEAD"], text=True).strip()
except Exception: genesis = None
print(json.dumps(dict(packages=packages, genesis_commit=genesis, cuda=torch.version.cuda,
 gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
 sm_count=torch.cuda.get_device_properties(0).multi_processor_count)))'''
    runtime = json.loads(command("docker", "exec", container, "python3", "-c", probe).splitlines()[-1])
    return dict(captured_at=datetime.now(timezone.utc).isoformat(), container=container,
                image_id=record["Image"], configured_image=config["Image"],
                compose_files=config.get("Labels", {}).get("com.docker.compose.project.config_files"),
                environment=safe, runtime=runtime,
                gpu=command("nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu", "--format=csv"),
                processes=command("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot(args.container)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
