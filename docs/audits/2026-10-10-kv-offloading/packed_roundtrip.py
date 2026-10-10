import gc
import importlib.metadata
import importlib.util
import random
import subprocess
from pathlib import Path
import torch

PIN = "05a013b29da78cf2321b9b46ec5039dde2fb0bb0"
root = Path("/opt/LMCache")
assert subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip() == PIN
assert importlib.metadata.version("lmcache") == "0.5.5"
assert torch.cuda.is_available(), "CUDA is required; skips are not acceptance"
assert torch.cuda.device_count() == 1, "Expose only the selected local GPU"
assert "3090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
torch.cuda.set_device(0)
torch.manual_seed(42)

# Import these explicitly so missing extensions fail, not pytest-skip.
import lmcache.cuda_ops
import lmcache.lmcache_native as native

path = root / "tests/v1/test_mp_mem_kernels.py"
spec = importlib.util.spec_from_file_location("upstream_mp_kernel_fixture", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert m.torch_device_type == "cuda"

# Override only test geometry and unsupported uint8 random-data generation.
m.NB, m.BS = 16, 16
m.NUM_MEMORY_OBJECTS, m.TOKENS_PER_OBJECT = 2, 32
m.BLOCKS_PER_OBJECT = m.TOKENS_PER_OBJECT // m.BS
m.TOTAL_BLOCKS = m.NUM_MEMORY_OBJECTS * m.BLOCKS_PER_OBJECT

def random_bytes(shape, dtype, device):
    assert dtype == torch.uint8
    out = torch.randint(0, 256, shape, dtype=dtype, device=device)
    if shape[-1] == 400:
        out[..., 388:].fill_(0xA5)
    return out

m._create_random_tensor = random_bytes
saved = {}
make_source = m.create_vllm_tensors
make_objects = m.create_memory_objects

def capture_source(*args, **kwargs):
    saved["source"] = make_source(*args, **kwargs)
    return saved["source"]

def capture_objects(*args, **kwargs):
    saved["objects"] = make_objects(*args, **kwargs)
    return saved["objects"]

m.create_vllm_tensors = capture_source
m.create_memory_objects = capture_objects

for width in (388, 400):
    saved.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Real upstream D2H and H2D kernel launches + exact restored-block assertions.
    m.test_block_transfer_roundtrip(
        m.FMT_VLLM_CS_NHD, 2, 4, width, True, torch.uint8, "cpu"
    )

    # Independently check actual pinned CPU object bytes against the source.
    # These CPU reference copies are comparisons, not the tested transfer path.
    ids = random.Random(42).sample(range(m.NB), m.TOTAL_BLOCKS)
    for j, obj in enumerate(saved["objects"]):
        assert obj.device.type == "cpu" and obj.is_pinned()
        selected = ids[j*m.BLOCKS_PER_OBJECT:(j+1)*m.BLOCKS_PER_OBJECT]
        expected = torch.stack([
            layer[selected].reshape(m.TOKENS_PER_OBJECT, 4*width).cpu()
            for layer in saved["source"]
        ]).unsqueeze(0)
        assert torch.equal(obj, expected), ("CPU object mismatch", width, j)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    assert peak < 100 * 1024**2, peak
    print({"case": "MP kernel D2H/H2D", "dtype": "uint8",
           "format": "NL_X_NB_BS_NH_CS", "width": width,
           "layers": 2, "heads": 4, "block_tokens": 16,
           "chunk_tokens": 32, "objects": 2, "peak_tensor_bytes": peak,
           "result": "PASS"}, flush=True)

print("Both packed-width cases passed; MP IPC/SSD/hybrid/model remain untested.")
