from types import SimpleNamespace

import pytest
import torch

from aquillm_vllm_h100.gdn.prefill_compat import prepare_install


def original(q, k, v, g, beta, initial_state, output_final_state,
             cu_seqlens=None, use_qk_l2norm_in_kernel=True):
    return locals()


@pytest.mark.parametrize('positional', [False, True])
def test_prefill_offsets_widen_without_changing_inputs(positional):
    module = SimpleNamespace(fi_chunk_gated_delta_rule=original)
    owner, call = prepare_install(module)
    assert owner is module and module.fi_chunk_gated_delta_rule is original
    offsets = torch.tensor([0, 9, 42], dtype=torch.int32)
    sentinel = object()
    if positional:
        result = call(sentinel,sentinel,sentinel,sentinel,sentinel,sentinel,True,offsets,False)
    else:
        result = call(q=sentinel,k=sentinel,v=sentinel,g=sentinel,beta=sentinel,
                      initial_state=sentinel,output_final_state=True,cu_seqlens=offsets,
                      use_qk_l2norm_in_kernel=False)
    assert result['cu_seqlens'].dtype == torch.int64
    assert result['cu_seqlens'].tolist() == [0,9,42]
    assert offsets.dtype == torch.int32
    assert result['q'] is sentinel and result['initial_state'] is sentinel
    assert result['use_qk_l2norm_in_kernel'] is False


@pytest.mark.parametrize('offsets', [None, torch.tensor([0,12],dtype=torch.int64)])
def test_prefill_existing_offset_type_is_preserved(offsets):
    module = SimpleNamespace(fi_chunk_gated_delta_rule=original)
    _, call = prepare_install(module)
    result = call(None,None,None,None,None,None,False,offsets)
    assert result['cu_seqlens'] is offsets
    module.fi_chunk_gated_delta_rule = call
    assert prepare_install(module)[1] is call


def test_prefill_signature_drift_is_rejected_before_mutation():
    altered = lambda *args, **kwargs: None
    module = SimpleNamespace(fi_chunk_gated_delta_rule=altered)
    with pytest.raises(RuntimeError,match='signature'):
        prepare_install(module)
    assert module.fi_chunk_gated_delta_rule is altered


def test_candidate_runtime_installs_compat_even_without_kernel_flags(monkeypatch):
    from aquillm_vllm_h100 import adapters, bootstrap, compatibility
    monkeypatch.setattr(bootstrap,'_installed',False)
    monkeypatch.setattr(compatibility,'verify_runtime',lambda:None)
    seen = []
    def install(config):
        seen.append(config)
        return {'status':'installed'}
    monkeypatch.setattr(adapters,'install_adapters',install)
    assert bootstrap.install({'AQUILLM_H100_RUNTIME_PROFILE':'flashinfer-0.6.18'})['status'] == 'installed'
    assert seen[0]['runtime_profile'] == 'flashinfer-0.6.18'
