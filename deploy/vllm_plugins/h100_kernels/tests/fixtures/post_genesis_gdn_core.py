# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Exact deployed vLLM 2dfaae752 + Genesis 34e269301, captured 2026-10-10.
# Fixture only: full _forward_core method; GPU dependencies supplied by tests.
from __future__ import annotations

def _forward_core(
    self,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
):
    """Core conv1d + recurrent attention (standard path).

    Args:
        mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
        b: beta gating vector                   (num_tokens, num_heads)
        a: alpha gating vector                  (num_tokens, num_heads)
        core_attn_out: Pre-allocated output buffer for attention results.
    """
    forward_context = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata

    if attn_metadata_raw is None:
        self._warmup_prefill_kernels(mixed_qkv, 0)
        return

    assert isinstance(attn_metadata_raw, dict)
    attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
    assert isinstance(attn_metadata, GDNAttentionMetadata)

    if (
        self.enable_packed_recurrent_decode
        and attn_metadata.spec_sequence_masks is None
        and attn_metadata.num_prefills == 0
        and attn_metadata.num_decodes > 0
    ):
        return self._forward_core_decode_non_spec(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            attn_metadata=attn_metadata,
        )

    has_initial_state = attn_metadata.has_initial_state
    spec_query_start_loc = attn_metadata.spec_query_start_loc
    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    spec_sequence_masks = attn_metadata.spec_sequence_masks
    spec_token_indx = attn_metadata.spec_token_indx
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
    self_kv_cache = self.kv_cache
    # conv_state must be (..., dim, width-1) for the conv kernels.
    # DS layout stores it that way directly; SD layout needs a transpose.
    conv_state = (
        self_kv_cache[0]
        if is_conv_state_dim_first()
        else self_kv_cache[0].transpose(-1, -2)
    )
    ssm_state = self_kv_cache[1]
    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens

    # [Genesis P60 vllm#40738] SSM state pre-copy from accepted block
    spec_decode_src_indices = getattr(
        attn_metadata, 'spec_decode_src_indices', None
    )
    if spec_decode_src_indices is not None:
        assert non_spec_state_indices_tensor is not None
        n_correct = spec_decode_src_indices.shape[0]
        dst_indices = non_spec_state_indices_tensor[:n_correct]
        ssm_state[dst_indices] = ssm_state[spec_decode_src_indices]

    mixed_qkv = mixed_qkv[:num_actual_tokens]
    b = b[:num_actual_tokens]
    a = a[:num_actual_tokens]

    # 1. Convolution sequence transformation
    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    if spec_sequence_masks is not None:
        if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            mixed_qkv_spec = mixed_qkv
            mixed_qkv_non_spec = None
        else:
            mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
            mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
    else:
        mixed_qkv_spec = None
        mixed_qkv_non_spec = mixed_qkv

    # 1.1: Process the multi-query part
    if spec_sequence_masks is not None:
        # spec_state_indices_tensor is always set when spec_sequence_masks is set
        assert spec_state_indices_tensor is not None
        mixed_qkv_spec = causal_conv1d_update(
            mixed_qkv_spec,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
            ],
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=spec_query_start_loc,
            max_query_len=spec_state_indices_tensor.size(-1),
            validate_data=False,
        )

    # 1.2: Process the remaining part
    if attn_metadata.num_prefills > 0:
        assert mixed_qkv_non_spec is not None
        mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
        # - "cache_indices" updates the conv_state cache in positions
        #   pointed to by "state_indices_tensor"
        # [Genesis P60b vllm#40738] gate num_accepted_tokens on spec activity
        _p60b_spec_src = getattr(
            attn_metadata, 'spec_decode_src_indices', None
        )
        _p60b_conv_num_accepted = (
            num_accepted_tokens if _p60b_spec_src is not None else None
        )
        mixed_qkv_non_spec = causal_conv1d_fn(
            mixed_qkv_non_spec_T,
            conv_weights,
            self.conv1d.bias,
            activation=self.activation,
            conv_states=conv_state,
            has_initial_state=has_initial_state,
            cache_indices=non_spec_state_indices_tensor,
            query_start_loc=non_spec_query_start_loc,
            num_accepted_tokens=_p60b_conv_num_accepted,
            metadata=attn_metadata,
        ).transpose(0, 1)
    elif attn_metadata.num_decodes > 0:
        assert mixed_qkv_non_spec is not None
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv_non_spec,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
            ],
            validate_data=True,
        )
    else:
        mixed_qkv_non_spec = None

    query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

    # Split mixed non-spec-decode+prefill to process independently
    split_non_spec = (
        spec_sequence_masks is None
        and attn_metadata.num_prefills > 0
        and attn_metadata.num_decodes > 0
    )
    num_decode_tokens = attn_metadata.num_decode_tokens

    if attn_metadata.num_prefills > 0:
        assert mixed_qkv_non_spec is not None, (
            "mixed_qkv_non_spec must be provided for prefill path"
        )
        if spec_sequence_masks is not None:
            a_non_spec = a.index_select(0, non_spec_token_indx)
            b_non_spec = b.index_select(0, non_spec_token_indx)
        else:
            a_non_spec = a
            b_non_spec = b

        if split_non_spec:
            conv_output_prefill = mixed_qkv_non_spec[num_decode_tokens:]
            a_prefill = a_non_spec[num_decode_tokens:]
            b_prefill = b_non_spec[num_decode_tokens:]
        else:
            conv_output_prefill = mixed_qkv_non_spec
            a_prefill = a_non_spec
            b_prefill = b_non_spec

        (
            query_non_spec,
            key_non_spec,
            value_non_spec,
            g_non_spec,
            beta_non_spec,
        ) = fused_post_conv_prep(
            conv_output=conv_output_prefill,
            a=a_prefill,
            b=b_prefill,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        query_non_spec = query_non_spec.unsqueeze(0)
        key_non_spec = key_non_spec.unsqueeze(0)
        value_non_spec = value_non_spec.unsqueeze(0)
        g_non_spec = g_non_spec.unsqueeze(0)
        beta_non_spec = beta_non_spec.unsqueeze(0)
    else:
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )
        g_non_spec = None
        beta_non_spec = None

    # 2. Recurrent attention

    # 2.1: Process the multi-query part
    if spec_sequence_masks is not None:
        core_attn_out_spec, last_recurrent_state = (
            fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a,
                b=b,
                dt_bias=self.dt_bias,
                q=query_spec,
                k=key_spec,
                v=value_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_spec_decodes
                    + 1  # type: ignore[attr-defined]
                ],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        )
    else:
        core_attn_out_spec, last_recurrent_state = None, None

    # 2.2: Process non-spec-decode part
    if split_non_spec:
        query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec[:num_decode_tokens]  # type: ignore[index]
        )
        core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            a=a[:num_decode_tokens],
            b=b[:num_decode_tokens],
            dt_bias=self.dt_bias,
            q=query_decode,
            k=key_decode,
            v=value_decode,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                : attn_metadata.num_decodes + 1
            ],
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out_decode = None

    # 2.3: Process the remaining part (prefill chunk, or non-spec decode-only)
    if attn_metadata.num_prefills > 0:
        # State indices, initial-state mask and cu_seqlens for the chunk
        # kernel are precomputed by the metadata builder (the prefill tail
        # when decodes are peeled off, else the full non-spec batch), so they
        # don't need to be re-derived per layer.
        prefill_state_indices = attn_metadata.prefill_state_indices
        prefill_has_initial_state = attn_metadata.prefill_has_initial_state
        assert prefill_state_indices is not None
        assert prefill_has_initial_state is not None
        initial_state = ssm_state[prefill_state_indices]
        initial_state[~prefill_has_initial_state, ...] = 0
        (
            core_attn_out_non_spec,
            last_recurrent_state,
        ) = self.chunk_gated_delta_rule(
            q=query_non_spec,
            k=key_non_spec,
            v=value_non_spec,
            g=g_non_spec,
            beta=beta_non_spec,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=attn_metadata.prefill_query_start_loc,
            chunk_indices=attn_metadata.chunk_indices,
            chunk_offsets=attn_metadata.chunk_offsets,
            use_qk_l2norm_in_kernel=False,
        )
        # Init cache
        ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)

        if split_non_spec:
            # Stitch the peeled decode outputs in front of the prefill
            # outputs (decode-first order).
            core_attn_out_non_spec = torch.cat(
                [core_attn_out_decode, core_attn_out_non_spec], dim=1
            )
    elif attn_metadata.num_decodes > 0:
        core_attn_out_non_spec, last_recurrent_state = (
            fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a,
                b=b,
                dt_bias=self.dt_bias,
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_decodes
                    + 1  # type: ignore[attr-defined]
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        )
    else:
        core_attn_out_non_spec, last_recurrent_state = None, None

    # 3. Merge core attention output
    if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
        merged_out = torch.empty(
            (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
            dtype=core_attn_out_non_spec.dtype,
            device=core_attn_out_non_spec.device,
        )
        merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
        merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
    elif spec_sequence_masks is not None:
        core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
    else:
        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
