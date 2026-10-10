# FlashInfer auxiliary experiment on development H100

Captured runtime: vLLM 0.23.1rc1.dev748+g2dfaae752, FlashInfer 0.6.13,
PyTorch 2.11.0+cu130, Genesis 34e269301cc3df71ae4b0da00a0a159b16b4e5d8.
Model activation dtype is FP16; four MTP drafts require five verification positions.

The installed FlashInfer GDN MTP implementation stores intermediate query/key
registers and output through BF16 in both its warp and inline paths, even when
the public API accepts FP16. The existing recurrent kernel accumulates loaded
FP16 query/key values in FP32 and writes the requested output dtype. Replacing
it would introduce a precision change. This experiment therefore keeps GDN
decode and verification on the baseline; no dependency upgrade or reduced
speculation depth is included. Generic API support for T > 1 alone is insufficient.

The source probe and capability gate are in the H100 package's `gdn/` module
and `benchmarks/gdn.py`. The installed model has 16 linear key heads, 48 linear
value heads, and key/value dimension 128. State scatter/rollback and graph
qualification were not attempted after the precision gate failed.

FlashInfer sampling was already enabled on the captured deployment. The
installed sampler uses native paths for seeded per-request generators, absent
top-k/top-p, and certain processed-logit modes. The independent sampler tests
trace actual FlashInfer calls and test distributions rather than assuming the
flag establishes activation. GPU results will be added after serial execution;
there is no new sampling speedup claim or default change.
