# v0.6 review — operator and multi-device offline planning

Date: 2026-10-28. Basis: the supplied v0.5 archive and Radiance source archive,
existing reviews and source inspection. This review covers the added planning
code and host experiments, not all ISA instructions or production GPU execution.

## Changes actually implemented

1. Attention/QSA costs: explicit per-query visibility and selection, GQA reuse,
   tiled matrix work, softmax, split partials/merge, full-history indexer scoring,
   selection hypotheses and block-key updates.
2. MoE costs: exact supplied routing histogram and per-expert widths, padded grouped
   gate/up and down, router/scatter/activation/gather/shared phases and scale bytes.
3. Source-bound partitioning: whole 128-column expert slices, rotated uneven TP2/4
   widths and separate shared placement. Twelve exact input hashes fail closed.
4. Directional TP2 oneshot/ring cost hypotheses: bytes in both directions, startup,
   chunks, reduction dependencies, peer memory and shared root-path contention.
5. Explicit resource/DAG/queue scheduler, bounded candidate search, layout costs and
   declared memory lifetimes. CPU-only calculations; no TP2 backend is added.
6. Partial model-workload importer with a missing-component ledger, rather than
   inventing every model's tensor sizes or assigning zero time to GDN/hidden work.
7. Finite rate-assumption sensitivity and external identity-bound graph comparison.
   These are not confidence intervals or an automatic learned GPU cost model.
8. Genuine-source CPU paged-attention fixture execution with independent stable
   softmax/fsum outputs, plus mathematical chunked attention and TP expert oracles.

## Review decisions / corrections

- Equal sharing is an explicit model, not AMD hardware scheduler emulation.
- F16/FP8 matrix paths share a matrix-issue resource; concurrent paths cannot both
  receive an independent full engine. DRAM and root-link capacities are shared.
- Matching layouts do not authorize free cross-GPU data access; cross-device data
  edges without an explicit communication operation are rejected.
- Dense causal query tiles can share their visible union. Differing sparse-query
  lists are not silently collapsed into a batch-wide selection.
- Attention split workspace uses the actual declared partial precision. Effective
  nonempty split counts are a work hypothesis, not an exact source dispatch proof.
- Known source oneshot is not called a ring. Chunk scheduling does not reproduce
  its real flag/epoch/atomic/checksum operations or persistent residency.
- Collective launch cost is charged once per rank, not once for each reduced chunk.
- Dense-GEMM wrappers reserve every kernel slot because their shared resource
  decomposition is not reconstructed here; unknown overlap is not inferred.
- Profile scenarios retain assumed provenance. A stable ranking is not measured
  accuracy and winner frequency is not a probability.
- Model selections, histograms, recipe fields and missing work remain explicit.
  Source parameters justify geometry, NOT numerical equivalence of a candidate.
- The original ISA/replay/native integration code and tests are byte-identical to
  v0.5; the new source/hypothesis/catalog path does not mutate production dispatch.

## Executed validation

Final groups: **162 static ISA + 142 replay Python + 43 native host + 8 host ASan
+ 245 planner tests = 600**, zero final failures or skips. Of the 245 planner tests,
114 are retained v0.5 and 131 are new in this stage. They cover counts/contracts,
causal/tail cases, TP partitioning, resource conservation, memory budgets, DAG
cycles/layout/cross-device edges, beam/exhaustive scopes, observations and CLI
failure behavior. Protocol tests use explicit synthetic observations; those are
not entered into a measured GPU evidence corpus.

The genuine-source standalone host pipeline was freshly rebuilt and passed,
including 46 existing GEMM fixtures and independent declared-row coverage. It
still tests one of 210 libref registrations in that corpus; this is not all-row
coverage. Native host tests and a separate clang ASan build passed.

The new actual CPU experiment ran **20 ref_attn_paged fixtures**, each with one
warmup, five measured/checking repetitions and contract diagnostics. Cases cover
page boundaries, permuted physical blocks, causal/window masks, ragged sequences,
verify widths, and head_dim=256 with GQA 6/12. All checks passed. These are finite
FP32 host computations, not FP8 GPU numerical validation or GPU timing calibration.
No CPU latency is converted into R9700 latency; noise/clocks are uncontrolled.

The offline source-bound pipeline generated 37 cost hypotheses, seven operator
examples, a complete 96-assignment search for the example TP2 window, a three-rate
sensitivity study, and 144 partial model operator occurrences with 120 separately
unbound dense-shape requests. Every new GPU scenario remains NEEDS_MEASUREMENT;
strict calibration mode was tested to exit 2. The config/routing/selection data
are explicit examples, not a captured production workload.

See current `evidence/RELEASE_VALIDATION.json`, `evidence/v06/` and package validation
for the exact executed commands, logs and hashes. The retained v0.5 160-run CPU
GEMM experiment is historical; it was not silently relabeled as a new measurement.

## Remaining implementation/evidence boundary

No HIP build or AMD GPU/TP2 execution occurred. New cost hypotheses are not
instantiated native kernels, and their declared effective rates do not model
all wave placement, dependency latency, cache bandwidth, dequantization or spills.
The new graph observation interface compares trusted external records; it does
not replace the established checked runtime importer or fit all new operators.
Native new-operator launcher mappings, complete GPU calibration and held-out
runtime ranking validation remain necessary before trusting production choices.

No GPU-read sanitizer, TP2 runtime replay, raw AQL tape, device-chain execution,
GDN cost model, complete all-model shape extraction, auto-generated HIP kernels,
full-model latency/tokens-per-second forecast or production dispatcher is added.
The source-specific importer reports those omissions rather than hiding them.
The absent full-engine `core/kvstream/bind.h` remains absent; no stub was invented.
A host process timeout cannot certify or recover a wedged GPU.
