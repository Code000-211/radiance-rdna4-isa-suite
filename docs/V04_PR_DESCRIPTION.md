# feat(test): harden replay with contract mutations, host ASan and scoped coverage

## Summary

Extend v0.3 without changing Radiance kernel arithmetic. The four demonstrated
host blind spots now have mandatory counterexamples and matching detectors:
output-only perturbation, explicit scratch initialization contracts,
operation-level write checks/checkpoints and instrumented host read checking.

Add an independently enumerated declared-row denominator, a finite-f32 GEMM
boundary corpus with independent oracles, versioned numeric policies, interleaved
ABBA comparisons, stop-on-first-nonpass corpus execution and strict scoped
qualification. An optional AQL diagnostic hook binds actual ordinary dispatches
to a prevalidated binary whitelist **before queue reservation**.

## Code changes

* Native worker: diagnostic passes separated from timing; readable aliases preserved;
  checkpoint tolerances honored; all normal iterations still checked; stable finite
  numeric comparison; explicit diagnostic/ASan labels.
* Host ASan: instrumented worker/plugin required, per-operation accessible ranges,
  guard/view poisoning, unsupported shadow boundaries refused, first error stops,
  no benchmark evidence from sanitized builds.
* Fixtures: optional `scratch_policy` and `checkpoints`, hash-pinned checkpoint blobs,
  exact validated manifest bytes, no silent scratch-policy inference for old snapshots.
* Corpus: stop-on-first-failure with all attempts retained, independent plugin row
  inventory, qualification that cannot “pass” unimplemented GPU read/TP2 coverage.
* Performance: equivalent workloads/frozen choices, ABBA interleaving and explicit
  block-level statistics; no tokens/s or production-equivalence claims.
* Optional AQL patch: exact object hash, descriptor, launch geometry, queue/node/pass
  observation; unknown identity/cap/background thread rejected before reservation.
* Standalone genuine-source host build and one-command validation without serving-engine
  dependencies; the missing `core/kvstream/bind.h` is not stubbed.

## Validation

See the **current** `evidence/RELEASE_VALIDATION.json` and logs. Counts refer to suite
and host tests, not that number of GPU kernels. The 46 synthetic f32 GEMM cases
exercise one `libref` row; the real supplied `libref` declares 210 rows. Old release
evidence is archived explicitly. The optional AQL patches were application-checked
and the observer was tested as host code; its HIP/HSA build and GPU run are absent.

## Intentional remaining exclusions

No certified GPU read sanitizer; no actual R9700 results; no TP2/P2P backend;
no raw AQL tape/device chains or opaque instances; no oracle for every operation;
no automatic production activation corpus; no engine-window calibration. Strict
profiles mark these **INCOMPLETE**, not fixed. See `docs/V04_GAP_CLOSURE.md`.

## Suggested commits

1. Replay schema/runner diagnostics plus host counterexample regressions.
2. ASan-specific build/supervision and read-violation tests.
3. Corpus, inventory, numeric policy, scoped acceptance and A/B tooling.
4. Optional AQL dispatch observation patch as a separate opt-in integration.
5. Documentation and reproducible host evidence; target-host evidence separately.

## Merge checklist

- [ ] Run `check-all-host.sh` against the intended complete source checkout.
- [ ] Review every scratch contract; never mark unknown scratch as zeroed just to pass.
- [ ] Approve numerical tolerance policy before inspecting candidate failures.
- [ ] Reconcile all required rows/shapes/variants; preserve NOT_TESTED cases.
- [ ] Review/apply optional AQL hooks separately and build full HIP/HSA on target.
- [ ] Strictly scan the actual plugin AND runtime-helper code objects.
- [ ] Confirm every supported diagnostic GPU kernel node is observed on target.
- [ ] Keep diagnostic serialization and ASan out of performance claims.
- [ ] Supply actual GPU, TP2 and production-calibration evidence before broad approval.
- [ ] Never treat process kill as GPU reset/recovery or disable watchdogs to pass.
