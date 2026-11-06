# feat(planner): add scoped attention/MoE/TP2 cost graphs and partial model workloads

## Summary

Extend v0.5 offline dense GEMM planning with explicit attention/QSA, histogram-based
MoE, directional two-rank communication costs, shared-resource DAG scheduling,
finite assignment search, memory lifetimes and rate sensitivity. Add a source-pinned
partial qwen4_exp shape importer and genuine-source CPU attention regressions.
No Radiance kernel, production dispatcher, runtime fence or hardware setting changes.

The new 37 candidates are **cost hypotheses**, not compiled GPU implementations.
All new operator/graph results require target measurement. This PR adds no TP2
runtime backend and makes no measured R9700 speedup or full-model TPS claim.

## Changes

- Strict per-query sparse KV/tail and separate historical QSA score accounting.
- Explicit MoE route histogram, padded gate/up/down, shared work and scale bytes.
- Reviewed 128-column TP2/TP4 partitions; width 640 is not blindly divided by TP.
- Per-direction push/pull/link/root constraints, custom oneshot versus ring hypotheses.
- Dependency/queue/resource graph, layout costs, memory highwater and bounded search.
- Finite profile stress test, NOT statistical confidence; identity-bound external
  graph comparison, NOT automatic learned calibration or remote attestation.
- Source-specific partial workload extraction; missing GDN/dense recipe/hidden work
  remains explicit. No arbitrary HF config or opaque state is guessed.
- Twenty actual libref paged-attention host fixtures with independent Python oracle.
- 131 new planner tests, documentation, exact hashes and reproducible offline command.

## Validation

600 final tests pass: 162 ISA, 142 replay-Python, 43 native host, 8 host ASan and
245 planner (114 old + 131 new). Host components were freshly built. Additionally,
46 existing GEMM fixtures and 20 new attention fixtures passed. No GPU kernel ran.
The source-bound example pipeline checks all 96 assignments and deliberately
expects strict calibration mode to exit 2, not PASS.

Current evidence is in `evidence/v06/` and `evidence/RELEASE_VALIDATION.json`.
Old v0.5 CPU timing/calibration artifacts remain historical. The full engine and
monolithic rad-kbench remain unbuilt because of the preexisting missing engine
header; standalone components do not need it and do not fake it.

## Suggested commits

1. Strict operator math + source/shape adapter + mathematical and negative tests.
2. Resource graph, search, memory ledger, sensitivity and comparison protocols.
3. Native CPU attention fixtures and offline orchestration scripts.
4. Documentation and current evidence; target-host GPU evidence separately.

## Merge checklist

- [ ] Review the pinned source geometry and cost-hypothesis boundaries.
- [ ] Run `check-planner-flow.sh SOURCE NEW_DIRECTORY` against the intended source.
- [ ] Run existing `check.sh` and standalone host/ASan checks.
- [ ] Run `attention-cpu-demo` with trusted actual libref/worker builds.
- [ ] Review explicit routing, KV selection, alias/memory and numerical contracts.
- [ ] Bind candidates to real compiled code and dispatches before GPU measurement.
- [ ] Validate actual GPU costs and overlap on held-out workloads under fixed settings.
- [ ] Keep new scenario rankings out of automatic production dispatch until qualified.
- [ ] Do not call the partial importer a complete model graph or the simulator TP2 replay.
- [ ] Keep GPU/TP2/wedge claims separate from CPU tests and numerical math oracles.
