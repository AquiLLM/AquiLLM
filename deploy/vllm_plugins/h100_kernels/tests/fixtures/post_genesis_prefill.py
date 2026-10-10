def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        # ════════════════════════════════════════════════════════════════
        # [Genesis PN401 — backport+improve of vllm#46461]
        # The flash_attn fast path below fires on
        # `max_query_len == max_seq_len` and passes
        # `cu_seqlens_k = query_start_loc` (the QUERY offsets). That is
        # only correct when EVERY request is a first-chunk prefill
        # (q_len == seq_len). A long first-chunk prefill can inflate
        # max_query_len to equal max_seq_len while the SAME batch carries
        # shorter continuation requests (q_len < seq_len — a non-first
        # chunked-prefill chunk or a prefix-cache hit). For those the fast
        # path would attend only to the current chunk's raw K/V and
        # silently DROP the cached prefix K/V -> hallucination. Detect any
        # continuation on the CPU-mirror tensors (no GPU sync) and skip the
        # fast path when present. Genesis hardening over the raw PR: if a
        # CPU mirror is None OR shape-inconsistent we conservatively treat
        # this as a continuation (skip the fast path) — a missing mirror
        # must never silently re-enable the buggy path (the fast path is
        # only a perf optimization; the per-request branch below is
        # always correct).
        _pn401_qsl_cpu = attn_metadata.query_start_loc_cpu
        _pn401_seq_lens_cpu = attn_metadata.seq_lens_cpu
        if _pn401_qsl_cpu is None or _pn401_seq_lens_cpu is None:
            _has_continuation = True
        else:
            _pn401_qsl = _pn401_qsl_cpu.tolist()
            _pn401_sl = _pn401_seq_lens_cpu.tolist()
            _pn401_n = len(_pn401_sl)
            if len(_pn401_qsl) < _pn401_n + 1:
                _has_continuation = True
            else:
                _has_continuation = any(
                    (_pn401_qsl[i + 1] - _pn401_qsl[i]) != _pn401_sl[i]
                    for i in range(_pn401_n)
                )
        # ════════════════════════════════════════════════════════════════
        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        # [Genesis PN401] gated with `not _has_continuation` (vllm#46461).
        if (
            _HAS_FLASH_ATTN
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
            and not _has_continuation
        ):
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        # [Genesis P26] Shared, profiler-visible prefill output pool.
        # First call reserves max_num_batched_tokens × Hq × D (picked up
        # by profile_run warmup); subsequent calls return a zeroed slice.
        from sndr.engines.vllm.kernels_legacy.dequant_buffer import (
            TurboQuantBufferManager as _GenesisTQBuf,
        )
        output = _GenesisTQBuf.acquire_prefill_output(
            num_tokens=N,
            num_q_heads=Hq,
            head_size=D,
            device=query.device,
            dtype=query.dtype,
            max_batched_tokens=getattr(self, '_max_num_batched_tokens', None),
        )

        # Prefer the CPU-resident copies from the metadata if populated —
        # otherwise `.tolist()` on GPU tensors forces a synchronizing copy.
        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        # Pre-allocate cu_seqlens for single-request flash_attn calls
        # to avoid per-request host→device tensor creation.
        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        # Cache arange on self (avoid per-call kernel launch).
        _max_seq = attn_metadata.max_seq_len
        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = _ac
        _arange_cache: torch.Tensor = _ac

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if _HAS_FLASH_ATTN:
                    # Assign to slice to avoid gpu/cpu sync.
                    self._cu_2[1:2] = q_len
                    cu = self._cu_2
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to TQ cache
                # by do_kv_cache_update. Use decode kernel directly to
                # avoid O(cached_len) full-dequant per continuation.
                # [Genesis P101 vllm#41123 selective backport]
                # Moderate continuations still use _continuation_prefill for
                # throughput, while long cached prefixes stay memory bounded.
                cached_len = seq_len - q_len
                use_decode_continuation = (
                    q_len <= _CONTINUATION_DECODE_THRESHOLD
                    or cached_len >= _CONTINUATION_DECODE_MAX_CACHED_LEN
                )
                if use_decode_continuation:
                    # Decode path: treat each query as a decode request
                    # with incremental seq_lens for causal masking. Keep
                    # large chunks sliced to bound decode scratch memory.
                    out = torch.empty_like(q_seq)
                    for q_offset in range(0, q_len, _CONTINUATION_DECODE_THRESHOLD):
                        q_next = min(q_offset + _CONTINUATION_DECODE_THRESHOLD, q_len)
                        q_part = q_seq[q_offset:q_next]
                        part_len = q_next - q_offset
                        output_part = out[q_offset:q_next]
                        # Slice from pre-built arange (no kernel launch)
                        synth_seq_lens = _arange_cache[
                            cached_len + q_offset + 1 : cached_len + q_next + 1
                        ]
                        synth_bt = attn_metadata.block_table[i : i + 1].expand(
                            part_len, -1
                        )
                        triton_turboquant_decode_attention(
                            query=q_part,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                            output_buf=output_part,
                            buf_holder=layer,
                            max_num_kv_splits=self.max_num_kv_splits,
                        )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    out = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output
