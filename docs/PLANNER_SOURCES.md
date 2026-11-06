# Source basis and external references

Review date: 2026-10-27. Public references were checked as implementation context;
none supplies a measured calibration of the user's GPUs. No proprietary model
weights or ISA PDF pages are copied into this package.

## Supplied project artifacts

- `radiance-main(1).zip`: actual source used for CPU builds and source adapter.
- `RADIANCE_RDNA4_ISA_TESTSUITE_v0.4.zip`: unchanged ISA/replay base and tests.
- Prior v0.4 review: existing CPU-only verification boundary, missing full-engine
  header and absent R9700/TP2 calibration remain applicable.
- Supplied AMD RDNA4 ISA reference, 2025-04-07, and the existing reviewed `radisa`
  resource implementation: physical VGPR granularity and resource limits. This
  planner release does not claim a fresh full 707-page ISA audit.

The three source-adapter file hashes are explicit in `radplan/radiance.py` and the
example catalog. Relevant supplied code is:

- `libr4d/r4d_gemm_w4a8_prefill.hip`: INT8 WMMA, tile template switch, LDS arrays,
  scalar scale-tail expression and guarded ASCVEC alternatives.
- `libr4d/r4d_gemm_args.h`: launcher argument contract.
- `libr4d/r4d_rows.cpp`: registered shape guards, group=128, prefill tuning table,
  and tile geometry. Registry N%256 is retained even for BN=128 variants.
- `libref/ref_gemm.cpp` and other libref translation units: genuine reference code.
- `replay/host-test/CMakeLists.txt`: real-source standalone build inherited from v0.4.

## Public primary sources

1. AMD Radeon AI PRO R9700 product specification:
   https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html
   Basis for 64 CUs, 640 GB/s and dense compute peak rates. INT8/FP8 383 TOPS/TFLOPS,
   FP16 matrix 191 TFLOPS, INT4 766 TOPS and FP32 vector 47.8 TFLOPS are peak-rate
   entries, not achieved rates. Sparse peak figures are not substituted.
2. AMD Matrix Instruction Calculator:
   https://github.com/ROCm/amd_matrix_instruction_calculator
   Primary reference for instruction-specific mapping and throughput capabilities.
   It is not silently bundled as a dependency and its whole instruction model is
   not implemented by this release.
3. AMD GPUOpen, Occupancy explained:
   https://gpuopen.com/learn/occupancy-explained/
   Supports the distinction between register/workgroup limits and useful achieved
   occupancy. The planner does not assume linear speedup with occupancy.

## New reasoning and engineering choices

The cost-feature model, illustrative efficiency factors, NNLS latency surrogate,
shape-group splitting, residual-envelope policy, measurement shortlist and serial
path solver are this implementation's design choices. They are not AMD-certified
formulas or quoted ISA requirements. Their limitations and actual CPU validation
are documented separately. No external source is used to replace missing GPU data.
