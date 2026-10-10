"""Installed store -> decode at actual TQ page16 and synthetic page2128 boundaries."""
import pytest
import torch

from aquillm_vllm_h100.contracts import KVSpec,VerifyBatch

pytestmark = [pytest.mark.gpu,pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA serving runtime required")]


@pytest.mark.parametrize("block_size", [pytest.param(16, id="actual-tq-page16"),
                                       pytest.param(2128, id="synthetic-page2128")])
def test_actual_k8v4_store_roundtrip_crosses_reordered_pages_and_preserves_other_slots(block_size):
    from vllm.config import VllmConfig,set_current_vllm_config
    from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl
    from reference import unpack_prefix
    with set_current_vllm_config(VllmConfig()):
        impl = TurboQuantAttentionImpl(num_heads=24,head_size=256,scale=0.0625,num_kv_heads=4,
                                      kv_cache_dtype="turboquant_k8v4")
    assert impl.tq_config.key_fp8 and impl.tq_config.effective_value_quant_bits == 4
    prior, length = block_size - 1, 3
    slot_bytes = impl.tq_config.slot_size_aligned
    cache = torch.full((3,block_size,4,slot_bytes),0xA5,dtype=torch.uint8,device="cuda")
    table = torch.tensor([[2,0,1]],dtype=torch.int32,device="cuda")
    # The three slots are physical page2/last, then physical page0/first two.
    slots = torch.tensor([3*block_size-1,0,1],dtype=torch.int64,device="cuda")
    keys = torch.linspace(-4,4,length*4*256,device="cuda",dtype=torch.float16).reshape(length,4,256)
    values = torch.linspace(-2,3,length*4*256,device="cuda",dtype=torch.float16).reshape(length,4,256)
    values[1,0] = 1.25  # constant-value scale path as well as nonconstant rows
    layer = torch.nn.Module()
    layer._tq_PiT = torch.eye(256,device="cuda")
    layer._tq_midpoints = torch.empty(0,device="cuda")
    impl._store_kv(keys,values,cache,slots,layer)
    # Oracle helper interprets seq_lens minus q length as committed length.
    spec = KVSpec("turboquant_k8v4",256,24,4,block_size,256,128)
    batch = VerifyBatch(torch.empty(1,1,24,256,device="cuda",dtype=torch.float16),cache,table,
                        torch.tensor([prior+length+1],dtype=torch.int32,device="cuda"),
                        keys[None,:1],values[None,:1],0.0625,spec)
    decoded_k,decoded_v = unpack_prefix(batch,0)
    torch.testing.assert_close(decoded_k[prior:prior+length],keys.to(torch.float8_e4m3fn).float(),atol=0,rtol=0)
    expected_range = values.float().amax(-1,keepdim=True)-values.float().amin(-1,keepdim=True)
    bound = expected_range/30 + 0.003
    assert ((decoded_v[prior:prior+length]-values.float()).abs() <= bound).all()
    # Guard bytes before/after each written slot prove correct physical mapping.
    assert (cache[2,block_size-2] == 0xA5).all()
    assert (cache[0,2] == 0xA5).all()
    assert (cache[1] == 0xA5).all()
