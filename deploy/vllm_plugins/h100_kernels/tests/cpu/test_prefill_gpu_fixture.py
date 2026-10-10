"""The GPU caller fixture must configure the helper's actual FA selection."""
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("ignore_override", [False, True])
def test_gpu_fixture_holds_explicit_fa2_context_and_rejects_wrong_helper(monkeypatch, ignore_override):
    gpu_dir = Path(__file__).resolve().parents[1] / "gpu"
    monkeypatch.syspath_prepend(str(gpu_dir))
    spec = importlib.util.spec_from_file_location("prefill_gpu_fixture_contract", gpu_dir / "test_prefill_routing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "fa2_runtime"), "GPU fixture lacks current-vLLM FA2 context"

    active = [None]
    @contextmanager
    def configured(config):
        previous = active[0]
        active[0] = config
        try:
            yield
        finally:
            active[0] = previous
    def selected(*, head_size):
        assert head_size == 256
        return 3 if ignore_override or active[0] is None else active[0].attention_config.flash_attn_version
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(
        VllmConfig=lambda: SimpleNamespace(attention_config=SimpleNamespace(flash_attn_version=None)),
        set_current_vllm_config=configured))
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.fa_utils", SimpleNamespace(get_flash_attn_version=selected))
    fixture = module.fa2_runtime.__wrapped__()
    if ignore_override:
        with pytest.raises(AssertionError):
            next(fixture)
        assert active[0] is None
    else:
        next(fixture)
        assert selected(head_size=256) == 2
        with pytest.raises(StopIteration):
            next(fixture)
        assert active[0] is None
