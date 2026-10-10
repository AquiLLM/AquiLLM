import importlib.util
from pathlib import Path


def test_benchmark_covers_required_crossover_grid_and_maximum_once():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "prefill.py"
    spec = importlib.util.spec_from_file_location("prefill_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = module.benchmark_cases(131072)
    assert len(cases) == 25
    assert (4096, 129) in cases and (16384, 512) in cases
    assert (32768, 2048) in cases and (65536, 8192) in cases
    assert (131072, 8192) in cases
    assert len(module.benchmark_cases(65536)) == 20
