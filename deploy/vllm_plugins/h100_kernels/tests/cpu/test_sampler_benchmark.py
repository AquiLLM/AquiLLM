"""Breaks caught: counting failed calls as execution or leaking instrumentation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "sampler.py"
    if not path.exists():
        pytest.fail("missing actual sampler dispatch tracing")
    spec = importlib.util.spec_from_file_location("sampler_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_counts_successful_execution_and_restores_both_functions():
    module = SimpleNamespace(flashinfer_sample=lambda value: value + 1)
    sampler = SimpleNamespace(forward_native=lambda value: value * 2)
    original_fi, original_native = module.flashinfer_sample, sampler.forward_native
    with load().DispatchTrace(module, sampler) as trace:
        assert module.flashinfer_sample(3) == 4
        assert sampler.forward_native(3) == 6
        assert trace.counts == {"flashinfer": 1, "native": 1}
    assert module.flashinfer_sample is original_fi
    assert sampler.forward_native is original_native


def test_trace_does_not_claim_flashinfer_execution_after_kernel_failure():
    def fail(value):
        raise RuntimeError("kernel failed")

    module = SimpleNamespace(flashinfer_sample=fail)
    sampler = SimpleNamespace(forward_native=lambda value: value)
    with load().DispatchTrace(module, sampler) as trace:
        with pytest.raises(RuntimeError, match="kernel failed"):
            module.flashinfer_sample(3)
        assert trace.counts == {"flashinfer": 0, "native": 0}
    assert module.flashinfer_sample is fail
