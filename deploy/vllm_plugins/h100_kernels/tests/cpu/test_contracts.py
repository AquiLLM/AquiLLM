import pytest

from aquillm_vllm_h100.contracts import KVSpec, SplitPlan


def test_actual_k8v4_geometry():
    spec = KVSpec("turboquant_k8v4", 256, 24, 4, 32, 256, 128)
    assert spec.num_q_heads // spec.num_kv_heads == 6


@pytest.mark.parametrize("field,value", [
    ("head_dim", 0), ("num_q_heads", 23), ("num_kv_heads", 0),
    ("block_size", -1), ("dtype", "turboquant_k4_nc"),
    ("key_packed_size", 128), ("value_data_bytes", 64),
])
def test_reject_invalid_kv_spec(field, value):
    args = dict(dtype="turboquant_k8v4", head_dim=256, num_q_heads=24,
                num_kv_heads=4, block_size=32, key_packed_size=256,
                value_data_bytes=128)
    args[field] = value
    with pytest.raises(ValueError):
        KVSpec(**args)


def test_split_plan_accepts_fixed_and_measured_shapes():
    assert SplitPlan(15, ()).buckets == ()
    assert SplitPlan(31, ((2048, 7), (8192, 15), (131072, 31))).max_splits == 31


@pytest.mark.parametrize("maximum,buckets", [
    (0, ()), (15, ((1, 16),)), (15, ((1, 0),)),
    (15, ((4, 7), (4, 15))), (15, ((4, 7), (3, 15))),
    (15, ((-1, 7),)), (15, ((1.5, 7),)), (True, ()),
])
def test_invalid_split_plan(maximum, buckets):
    with pytest.raises(ValueError):
        SplitPlan(maximum, buckets)
