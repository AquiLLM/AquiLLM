"""Install one ordered post-Genesis hook in the exact pinned plugin source."""
import argparse
import hashlib
from pathlib import Path

PLUGIN_SHA256 = "708fbba88ebe57fb6eff4b64c05fc1047abfe78d51495b67f43d37f10dfdcd1c"
ANCHOR = "stats = run(verbose=True, apply=apply_mode)"
MARKER = "# AquiLLM H100 post-Genesis registration v1"


def patch_plugin(path, *, expected_hash=PLUGIN_SHA256):
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        return
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Genesis plugin fingerprint differs from the pinned source")
    if source.count(ANCHOR) != 1:
        raise ValueError("Genesis registration anchor is not unique")
    line = next(line for line in source.splitlines() if ANCHOR in line)
    indent = line[:len(line) - len(line.lstrip())]
    hook = "\n".join(indent + text for text in (
        MARKER, "if apply_mode:",
        "    from aquillm_vllm_h100.bootstrap import install as _aquillm_h100_install",
        "    _aquillm_h100_install()",
    ))
    path.write_text(source.replace(line, line + "\n" + hook), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plugin", type=Path)
    patch_plugin(parser.parse_args().plugin)

