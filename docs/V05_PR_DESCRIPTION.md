# feat(planner): add scoped offline GEMM ranking and checked CPU calibration experiment

## Summary

Add a standalone offline planner beside the unchanged v0.4 ISA/replay tooling.
Compute tile work/resources, import actual compiler metadata, fit checked latency
evidence, and request new measurements when candidate ordering is uncertain.
This is not GPU-on-CPU emulation, a universal ISA simulator, a full-model optimizer,
or a replacement for kernel correctness and target-device tests.

## Changes

- `radplan/`: strict workload/candidate/profile contracts, analytical GEMM features,
  per-exact-candidate NNLS surrogate and shape-disjoint fit/calibration/test sets.
- Candidate-simultaneous residual envelope, held-out ranking regret and explicit
  measurement shortlist. Changed binaries, profiles, modes, backend or out-of-range
  shapes do not inherit a silent performance approval.
- Hash-pinned source adapter for 14 supplied W4A8 prefill arms. Use actual INT8-WMMA
  compute path, preserve registry alignment and exclude known scalar-scale tails.
- Compiler-resource import cross-checking strict ISA report, plugin/code-object
  membership, real ELF metadata and descriptor identity. Explicit symbol mapping
  remains a reviewed input, not automatic launcher equivalence.
- Checked single-operation replay importer and observed-environment profile builder.
  No sanitized timings, CPU-to-GPU conversion or attribution of a whole chain's
  duration to a single kernel.
- Explicit serial-path solver with layout-conversion costs, no invented TP2/overlap.
- CPU-only comparison plugin, genuine-source Radiance replay build command and
  retained 160-run experiment with independent references.
- Documentation, unit/negative tests, examples and source/evidence hashes.

## Validation

Current test groups: 162 existing static + 142 existing replay-Python + 43 native
host + 8 host-ASan + 114 new planner = **469 passing tests**, zero test failures or
skips. Native/ASan host components were freshly rebuilt. An initial authoring build
command was interrupted by the tool time limit, then resumed successfully; the
component tests were subsequently run separately.

Additional evidence: 160 checked CPU runs, four candidates and 40 distinct shapes,
with 20/10/10 train/calibration/test groups. Held-out top-1 was 10/10, median timing
error 5.60%, p90 error 25.51%. Only 2/10 choices had separated uncertainty envelopes;
the remaining eight still request measurements. Shared authoring CPU noise was not
controlled. The baseline and every sample remain available; these are not R9700
performance numbers.

The optional compiler-metadata import was exercised on an actual LLVM-generated
small code object using the existing static gate. No invalid fixture was executed
on a GPU. The full engine/monolithic rad-kbench build was not performed; the
standalone path does not depend on the missing engine header.

## Not changed / not certified

No kernel arithmetic, memory ordering, quantization layout, source bug fix,
production dispatcher, watchdog, clock or power setting is changed. GPU calibration,
GPU read instrumentation, TP2/P2P, raw tape, attention/QSA/MoE scheduling and full
model speed prediction remain unimplemented or unvalidated. The supplied R9700
profile is explicitly unmeasured; all example GPU rankings request measurements.

## Merge checklist

- [ ] Review hash-pinned adapter against the intended source revision.
- [ ] Run `check.sh`, `check-planner.sh`, existing host/ASan checks, and the new CPU
      experiment command against trusted source/plugins.
- [ ] Review actual shape/layout/numerical contracts, not only chosen tile sizes.
- [ ] Build genuine gfx1201 candidates and reconcile source arms to actual symbols.
- [ ] Collect strict ISA evidence and checked single-operation GPU replay samples.
- [ ] Pin target identity, compiler, launch mode and unchanged power/clock profile.
- [ ] Validate on untouched shape groups; inspect errors/regret and uncertainty.
- [ ] Keep performance ranking independent from numerical/runtime approval.
- [ ] Do not label CPU tests as GPU calibration or scenario rankings as verified winners.
