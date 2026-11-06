# Radiance RDNA4 ISA + Replay + Offline Planner — v0.5

**Offline candidate planning, not CPU emulation of a GPU.** v0.5 adds `radplan/`
to the byte-preserved v0.4 ISA and replay components. It does not edit Radiance
kernels or install a new production dispatcher. The implementation has no Python
third-party dependencies.

## Start here: create a plan without a GPU

From the Radiance repository root, after extracting the `tools/` overlay:

```bash
bash tools/rdna4-isa-suite/check-planner.sh
bash tools/rdna4-isa-suite/planner.sh plan \
  --profile tools/rdna4-isa-suite/planner/profiles/r9700-unmeasured.json \
  --catalog tools/rdna4-isa-suite/planner/examples/radiance-w4a8-catalog.json \
  --workloads tools/rdna4-isa-suite/planner/examples/w4a8-workloads.json \
  --out ./offline-plan-v05
```

The output directory must not exist. Results are `plan.json` and `plan.md`.
The included R9700 profile is **uncalibrated**. Its scenario rankings are useful
for inspecting resource tradeoffs, not a verified fastest-path recommendation.
All seven included GPU workloads request measurements; none receives deployment
approval. The sample dimensions are illustrative, not an inferred model config.

## Real CPU comparison, independent of the serving engine

```bash
bash tools/rdna4-isa-suite/check-planner-cpu.sh \
  "$PWD" "$PWD/planner-cpu-validation"
```

Requires Python 3.10+, CMake and a C++20 compiler. The second directory must be new.
This compiles genuine Radiance registry/device/kBench-comparison components and
`libref`, plus three small CPU comparison kernels, then executes 160 checked
standalone replay runs. It uses the existing `rad-kbench-replay` worker: **not a
completed build of the monolithic `rad-kbench` executable** and not a simulation
of HIP instructions. No model weights or server are required. Native source and
plugins must be trusted. No GPU code is launched by this command.

The supplied full source has a separate missing `core/kvstream/bind.h` dependency;
this standalone host target avoids the unrelated engine modules, without a stub.

See `docs/CPU_COMPARISON.md` for the actual retained experiment: 40 distinct shapes,
four candidates, 20 training / 10 calibration / 10 test shape groups. CPU results
cannot calibrate an R9700. The reference candidate is the actual supplied-source
`libref`; the other three are explicitly new CPU-only panel implementations.

## Scope of this version

| Component | Implemented | Not implied |
|---|---|---|
| Dense GEMM cost features | Tile padding, work, traffic scenario, LDS, grid, registers, split-K bookkeeping | Universal arbitrary-kernel simulation |
| Radiance adapter | 14 hash-pinned W4A8 prefill template arms | Every Radiance row/template or model |
| Compiled-resource import | Exact ELF/descriptor identity with existing strict ISA evidence | Automatic proof of launcher-to-symbol mapping |
| Calibration | Per-binary nonnegative latency model; shape-disjoint fitting and residual envelope | New binaries/shapes are automatically predictable |
| Ranking | Candidate validity filtering, conditional separation and measurement queue | Runtime/numerical safety or automatic dispatch |
| Path selection | Explicit serial stages and layout-transition costs | TP2, overlap, arbitrary fusion or full-engine tokens/s |

The adapter recognizes that this W4A8 prefill implementation computes using
**INT8 WMMA** after unpacking INT4 weights. It does not use FP8/INT4 peak throughput
just because a weight format has that name. Source-known scalar scale-tail cases
are excluded rather than silently benchmarked or labeled repaired.

## Calibration workflow

1. Review the actual workload and candidate contracts. For another source revision,
   review/update the narrow source adapter or supply an explicit catalog.
2. Compile candidates without running them. Use the existing ISA collector/auditor.
3. Bind compiler register/LDS/scratch metadata using an explicit reviewed
   candidate → plugin hash / code-object hash / descriptor mapping.
4. Obtain a small set of genuine, checked R9700 single-operation replay reports.
   Pin device, worker, mode, compiler and clock/power declarations in a new profile.
5. Import reports, fit on whole shape groups, inspect independent held-out error
   and selection regret, then plan new compatible workloads offline.
6. Measure when intervals overlap, an input is out of domain, a candidate changes,
   or a numerical/performance release check is needed. Do not transfer CPU fits.

Details and full command examples:
- `docs/OFFLINE_PLANNER.md`: formulas, assumptions, ranking and scope.
- `docs/PLANNER_FORMAT.md`: contracts, file formats, evidence import.
- `docs/V05_REVIEW.md`: changes, verification and known limitations.
- `docs/CPU_COMPARISON.md`: actual experiment, not a projected speedup.
- `docs/PLANNER_SOURCES.md`: primary references and supplied-source evidence.
- `PR_DESCRIPTION.md`: proposed Codeberg PR text.

## Status and exit codes

`plan` returning 0 means **a plan was written**, not that a kernel passed tests.
`--require-calibrated` returns 2 unless each workload has a scoped
`CALIBRATED_SEPARATION`. Ordinary uncertain results are `NEEDS_MEASUREMENT`;
unsupported shapes can be `NO_SUPPORTED_CANDIDATE`. Every plan has
`deployment_approved: false` and requires separate numerical/ISA validation.

`CALIBRATED_SEPARATION` means non-overlapping empirical envelopes within the
stated profile, shape range and candidate set. It is not a global optimum or a
production confidence guarantee. Changed code/resource identity invalidates the
candidate fit. Manual JSON evidence remains trusted local input, not attestation.

## Existing v0.4 workflows are preserved

`check.sh`, `check-all-host.sh`, `radisa/`, `radreplay/`, replay integration and all
existing tests are retained. Their interfaces and component version strings stay
unchanged. The planner has its own `check-planner.sh` and version 0.5.0.

For the previous suite instructions see `docs/V04_README.md`. Historical evidence
is explicitly under `evidence/v04/`; current results are listed in
`evidence/RELEASE_VALIDATION.json`. The archived tests do not certify a GPU run.

## Evidence portability

The retained CPU reports contain authoring-machine paths and original binary
hashes. They allow analysis of the measured results, not execution of absent
binaries. Re-run the CPU command to generate fixtures/reports bound to your own
build. No compiled host plugin or unsafe GPU fixture is installed automatically.
The real LLVM-produced small GPU fixture inherited from v0.4 is for CPU-side
binary analysis only, not automatically executed.
