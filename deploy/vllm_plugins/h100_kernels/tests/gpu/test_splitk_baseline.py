"""Qualify the unchanged Genesis split-K launcher on independent packed fixtures."""
import pytest

torch = pytest.importorskip("torch")
H100 = torch.cuda.is_available() and "H100" in torch.cuda.get_device_name()
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not H100, reason="requires reserved pinned H100 runtime")]


@pytest.mark.parametrize("splits", [7, 15, 31, 47, 63])
@pytest.mark.parametrize("priors", [[0], [1], [7], [15], [16], [17], [31], [32], [33], [2048], [1, 33], [0, 7, 32, 2048]])
def test_fixed_split_launcher_matches_oracle(priors, splits, monkeypatch):
    from reference import assert_close, make_verify_batch, reference_verify
    from sndr.engines.vllm.kernels_legacy import p67_multi_query_kernel as baseline

    monkeypatch.setenv("GENESIS_P67_BLOCK_KV", "32")
    monkeypatch.setenv("GENESIS_P67_DOT_PRECISION", "tf32x3")
    batch = make_verify_batch(priors, length=5, dtype=torch.float16, strided=True)
    b, length, hq, dim = batch.q.shape
    hkv = batch.spec.num_kv_heads
    mid = torch.full((b, hkv, splits + 1, 8, 8, dim + 1), float("nan"), device="cuda", dtype=torch.float32)
    output = torch.empty_like(batch.q)
    pointers = (output.data_ptr(), mid.data_ptr())
    result = baseline.call_p67_splitk(
        batch.q, batch.kv_cache, batch.block_table, batch.seq_lens,
        batch.raw_k, batch.raw_v, batch.scale, batch.spec.block_size,
        batch.spec.key_packed_size, batch.spec.value_data_bytes,
        output=output, num_splits=splits, mid_o=mid,
    )
    assert result.data_ptr() == output.data_ptr()
    assert pointers == (output.data_ptr(), mid.data_ptr())
    assert_close(output, reference_verify(batch), batch.q.dtype)
    valid = mid[:, :, :, :length, :hq // hkv]
    assert torch.isfinite(valid[..., :dim]).all()
    # Prior shorter than split count must overwrite truly empty committed slots.
    for index, prior in enumerate(priors):
        step = (prior + splits - 1) // splits
        for sid in range(splits):
            if sid * step >= prior:
                assert (valid[index, :, sid, ..., :dim] == 0).all()
                assert torch.isneginf(valid[index, :, sid, ..., dim]).all()
