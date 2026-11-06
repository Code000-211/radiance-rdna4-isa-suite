# v0.6 source basis and external references

## Adapter review against radiance HEAD

The pins were re-reviewed against the current Radiance tree (ABI 31) and are
recorded per file in `docs/ADAPTER_REVIEW_HEAD.md`: no catalog constant changed,
and six newly visible gaps are listed there as explicit `shortcut:` limits.

## Supplied source / primary evidence

The uploaded `radiance-main(1).zip` is the implementation basis. Its SHA-256 and
v0.5 input hashes are recorded in `V06_INPUTS.json`. The twelve exact file hashes
used by the new adapter are in `radplan/source_ops.py::PINNED` and generated catalogs.

- `arch/qwen4exp_fp8/qwen4exp_fp8.cpp`: expert/shared TP placement, model integration.
- `arch/common/rad_block_moe_fp8.h`: block-MoE implementation context.
- `arch/common/rad_qsa.h`, `docs/QSA.md`: indexer geometry, per-query selections,
  whole complete blocks, incomplete tail, dense bound and persistent-state boundary.
- `libr4d/r4d_attn_paged_h256_gqa12.hip`, `r4d_attn_paged_h256_gqa6_bs4.hip`,
  `r4d_attn_paged_gqa8.hip`, `r4d_attn_decode_h256_gqa6.hip`,
  `r4d_attn_prefill_h256_gqa6.hip`: family geometry and split/merge implementation.
- `libr4d/r4d_moe.hip`, `libr4d/r4d_rows.cpp`: routed W4A8 FP8-WMMA path and registry.
- `libr4d/r4d_ar_oneshot_2rank_exact.hip`: custom two-rank push/flags/reduction.
- `libref/ref_attn.cpp`, `libref/ref_registry.cpp`: actual native CPU attention row,
  paged layout, causal/windowed/ragged semantics used for host fixtures.

The existing v0.5 dense-GEMM source binding, ISA checks and replay diagnostics are
retained. This release is NOT a second instruction-by-instruction audit of the
707-page supplied ISA PDF. The native host source was rebuilt; HIP/GPU code was not.

## External primary references used for context, not calibration

- AMD HIP asynchronous execution documentation (FIFO streams, events, resource
  limitations and possible contention):
  https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html
- Dao et al., FlashAttention, arXiv:2205.14135 (IO-aware attention without the
  full attention matrix materialization): https://arxiv.org/abs/2205.14135
- Dao, FlashAttention-2, arXiv:2307.08691 (parallelism/work partitioning affects
  actual efficiency): https://arxiv.org/abs/2307.08691

These sources motivate why operation counts alone are insufficient. The fluid
scheduler, numerical rate assumptions, QSA selection-pass costs, candidate catalog
and sensitivity solver are implementation choices in this suite, not claims
proved by those references. No paper's reported GPU speed is imported as an R9700
measurement. The custom Radiance collective is not assumed to use RCCL.
