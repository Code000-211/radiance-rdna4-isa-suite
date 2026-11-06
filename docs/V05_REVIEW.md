# v0.5 review: offline GEMM planner and real CPU comparison

Date: 2026-10-27. Scope: supplied v0.4 overlay and Radiance source; proposed offline
cost model; actual authoring-host experiment. This is not a new full ISA audit.

## Implemented

- Separate `radplan` module and command line, no Python third-party dependency.
- Explicit workload/candidate/profile schemas; numerical format and layout contract
  matching; no arbitrary model dimension inference.
- Analytical executed-work/padding, logical/tile/scenario traffic, static LDS,
  group/grid, split-K and conditional register-resource calculations.
- Hash-pinned adapter for all 14 registered W4A8 prefill arms in the supplied source.
- Strict existing ISA-evidence import of actual compiler metadata and descriptor
  resources using a reviewed explicit symbol map.
- Per-candidate nonnegative latency surrogate, complete panels and whole-shape
  train/calibration/test separation. Candidate-simultaneous residual envelopes,
  visible held-out error/regret and measurement queue.
- Device/backend/profile/worker/mode/binary/parameter binding for actual portable
  single-operation replay measurements. CPU cannot calibrate gfx1201.
- Explicit serial path dynamic programming with layout-transition costs; no implicit
  overlap/communication model.
- Genuine Radiance CPU standalone replay experiment, independent fsum oracles,
  four CPU candidates and 160 checked executions on 40 shapes.

## Corrections during review

1. The actual source uses INT8 WMMA after INT4 unpacking. Earlier generic FP8
   roofline examples are not substituted for this kernel's arithmetic path.
2. Registry N alignment is N%256, even where a template's BN is 128. Both source
   geometry and the registered caller contract must hold.
3. The known scalar Ascale source-tail issue is excluded for affected M values;
   no source rewrite or assertion of an emitted fault was added.
4. Padded executed work is not penalized twice by another padding factor.
5. Split-K partial output stores and reduction reads/final writes are charged to
   separate phases, not double-counted in the main memory term.
6. Compiler VGPR requirement and descriptor encoding are separate. Missing
   resources remain unknown, and scratch size is not called measured spill traffic.
7. Warm-cache work cannot use cold byte counts as a universal DRAM lower bound.
8. Unknown calibration for even one rival prevents a mixed calibrated/scenario
   comparison. A changed binary/resource identity invalidates the previous fit.
9. Held-out shape groups are not repetitions of fitted shapes. The model never
   tunes its radius on the held-out set.
10. Noise, source-specific unmodeled instructions and lack of GPU calibration are
    retained. Ten successful CPU rankings are not a claim of universal optimality.
11. A report whose identity matches is still trusted local evidence, not remote
    attestation. Manual candidate-to-symbol association remains a review obligation.
12. Compiler/runtime descriptions and clock/power profiles are declarations unless
    observed in a supplied report. The suite does not change hardware settings.

## Executed validation

- 114 new planner tests: schema/math, illegal shape/format combinations, source
  adapter, split grouping, calibration identity, CPU/GPU separation, uncertain
  decisions, path transitions, checked-report import and CLI failure behavior.
- The retained 162 static/decoder tests and 142 replay-Python tests were rerun.
- A fresh genuine-source host component build passed all 43 native tests.
- A fresh instrumented clang host build passed all eight ASan tests.
- Total current unit/integration test groups: **469**, with zero test failures and
  zero skipped tests. See current evidence logs, not archived v0.4 counts.
- Original 160 CPU comparison executions: every one passed runtime correctness
  and contract diagnostics. All input/expected bytes and samples are retained.
- The latest fit/validation recomputes byte-equivalent calibration JSON from the
  retained measured data; it does not substitute new timings.
- Actual LLVM-generated smoke code object was collected and statically audited;
  the compiled-resource importer read its metadata from the actual ELF. That is
  an import test, not a compiled Radiance W4A8 kernel or an executed GPU test.
- Seven illustrative R9700 workloads were planned without GPU execution. Every
  result remains NEEDS_MEASUREMENT; an abstract serial path was also generated.

One initial fresh-build orchestration command hit the authoring tool's time limit
while compiling libref; the build was resumed incrementally and completed. Tests
were then executed as separate component commands. No timed-out test was marked
passed, and no unfinished build is described as a full engine build.

The full engine / monolithic rad-kbench executable was not built. The standalone
worker uses genuine registry/device/kBench-comparison code without the unrelated
missing `core/kvstream/bind.h` engine dependency. No stub is invented.

## CPU comparison, not GPU validation

The authoring CPU is an AMD EPYC 9V74, not the user's EPYC/R9700 server. Four CPU
candidates on 40 small finite-f32 GEMM shapes gave ten out of ten correct held-out
winner choices, 5.60% median and 25.51% p90 relative timing error. Only two held-out
comparisons had separated envelopes; the other eight request measurement. The
operations-only baseline had 8.10% median error and 81.78% worst selection regret.
This is a limited noisy-host experiment. Candidate choice, measured time and
conformal-style coverage must not be generalized beyond its evidence.

## Explicit exclusions

No AMD GPU execution, HIP compilation, R9700 calibration, TP2/P2P, opaque state,
raw AQL tape/device chain export, attention/QSA model, full MoE scheduling model,
all-model shape extraction, automatic kernel generation or production dispatch
mutation. Source-level scenarios do not prove numerical equivalence. The existing
v0.4 GPU/read-sanitizer and production-validation limitations remain unchanged.

A true new kernel implementation requires correctness checks and enough new
measurement evidence to qualify its performance model. Compilation alone supplies
resources, not its actual achieved throughput. Offline planning reduces repeated
shape/parameter measurement; it does not eliminate every future hardware test.
