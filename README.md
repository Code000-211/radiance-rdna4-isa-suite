# Radiance RDNA4 ISA + Replay + Offline Planner — v0.6

**Multi-operator offline cost planning, not GPU emulation or production certification.**
v0.6 retains v0.5 GEMM calibration and adds attention/QSA, routed MoE, directional
TP2 communication cost models, dependency/resource scheduling, memory lifetimes,
finite candidate search, profile-sensitivity analysis and a source-bound **partial**
`qwen4_exp` model importer. No Radiance kernel or production dispatcher is changed.

New operator candidates are **cost hypotheses**, not newly compiled or executed
GPU kernels. All new operator/graph scenarios remain `NEEDS_MEASUREMENT`.
`--require-calibrated` exits 2 even when the scenario calculation itself succeeds.
The existing dense-GEMM empirical calibration remains available independently.

## One command: offline tests and examples, no GPU

After extracting this `tools/` overlay into the reviewed Radiance source tree:

```bash
bash tools/rdna4-isa-suite/check-planner-flow.sh \
  "$PWD" "$PWD/offline-flow-v06"
```

The output directory must be new. This verifies source hashes, runs planner tests,
creates the 37-hypothesis catalog, plans seven example workloads, exhaustively
compares 96 TP2-window assignments, runs a three-profile sensitivity study and
expands a declared example configuration into partial model workloads. The final
strict-mode test deliberately requires exit 2: no calibration is fabricated.

The examples use **synthetic routing/selection and assumed rates**, not real
production snapshots. A changed source hash requires adapter review, not removal
of the hash check. The suite has no Python third-party dependencies.

## Individual planning commands

```bash
S=tools/rdna4-isa-suite
E="$S/planner/examples/v06"
bash "$S/planner.sh" plan-operators \
  --catalog "$E/operator-catalog.json" --workloads "$E/operator-workloads.json" \
  --system "$E/dual-r9700-unmeasured.json" --out ./operator-plan-v06
bash "$S/planner.sh" plan-graph \
  --catalog "$E/operator-catalog.json" --workloads "$E/operator-workloads.json" \
  --system "$E/dual-r9700-unmeasured.json" --graph "$E/tp2-moe-graph.json" \
  --out ./graph-plan-v06
bash "$S/planner.sh" sensitivity \
  --catalog "$E/operator-catalog.json" --workloads "$E/operator-workloads.json" \
  --systems "$E/rate-sensitivity.json" --out ./sensitivity-v06.json
```

Inspect JSON for counts, assumptions, rejections, service demands, phase timelines,
per-device memory highwater, searched candidates and explicit measurement requests.
The Markdown summaries do not replace those details. A calculated microsecond
value is a **conditional scenario**, not a measured R9700 latency.

## Run genuine Radiance CPU attention, without the serving engine

```bash
S=tools/rdna4-isa-suite
bash "$S/check-all-host.sh" "$PWD" "$PWD/host-v06"
bash "$S/planner.sh" attention-cpu-demo \
  --worker "$PWD/host-v06/normal/rad-kbench-replay" \
  --libref "$PWD/host-v06/normal/libref.so" \
  --out "$PWD/attention-cpu-v06" --execute-trusted-code
```

This uses the real registry, host backend, comparison code and `ref_attn_paged`.
Twenty fixtures cover causal/windowed attention, block boundaries, permuted pages,
ragged batches, verify widths, and head_dim=256 with GQA 6/12. Every warmup and timed
repetition is checked against independent Python fsum/stable-softmax outputs.
FP32 CPU fixtures are not FP8 GPU calibration. Native code must be trusted.
CMake, a C++20 compiler, LLVM and clang ASan are needed for the full host pipeline.
The separate missing `core/kvstream/bind.h` engine dependency is not stubbed.

## Lint one kernel: `radisa check`

```bash
S=tools/rdna4-isa-suite
bash "$S/check-kernel.sh" libr4d/r4d_attn_paged_h128_gqa4.hip \
  --compile-commands build/compile_commands.json     # HIP source -> gfx1201 machine code
bash "$S/check-kernel.sh" snippet.s                  # hand-written assembly
bash "$S/check-kernel.sh" build/.../r4d_gemm_fp8a8.hip.o --kernel gemm_8a8_prefill   # code object
```

Every instruction is mapped onto AMD's machine-readable RDNA4 operand table (XML, MIT), and the
memory counters are modelled over the whole CFG. Missing `s_wait_*` (typically after inline-asm
memory instructions, which the compiler does not track), WMMA and VOPD hazards, VGPR deallocation
errors, spills and scratch are reported against source lines with manual section and PDF page.
`radisa hook` runs the same check as a Claude Code PostToolUse hook. Details: `docs/CHECK.md`.

## Main documentation

- `docs/V06_REVIEW.md`: actual validation and remaining boundaries.
- `docs/OPERATOR_PLANNER.md`: formulas, source relationships and approximations.
- `docs/FLOW_FORMAT.md`: DAG, resources, allocation lifetimes, reproducibility.
- `docs/SOURCES_V06.md`: exact source files and primary external references.
- `docs/V05_README.md`: retained dense GEMM planning/calibration commands.
- `evidence/RELEASE_VALIDATION.json`: current run, not archived test counts.
- `evidence/v06/`: current example plans, logs and twenty real CPU fixture runs.
- `evidence/cpu-comparison/`, `evidence/v05*`: **historical v0.5** dense-CPU run.

## Scope is explicit

| Implemented in v0.6 | Not implied |
|---|---|
| QSA history scoring + selected attention + merge counts | Exact per-instruction GPU timing, selector accuracy or unseen inputs |
| Explicit expert histograms, whole-block TP widths, shared-expert costs | Router prediction, complete EP routing, actual MoE GPU execution |
| Push/pull directions, two-rank oneshot/ring cost hypotheses | Runtime TP2 support, atomic ordering or wedge testing |
| DAG/queue/resource contention and finite candidate search | Exact AMD scheduler, arbitrary native kernel generation, hardware-optimal schedule |
| Declared buffers and workspace highwater | Discovery of hidden allocator/KV/state buffers |
| Finite rate-sensitivity and external graph comparison protocol | Statistical confidence or automatic calibration of all new operators |
| Partial source-specific model shape extraction | Complete transformer/GDN/MTP graph or tokens/s prediction |

No clock, voltage, power, watchdog, memory scope, cache hint or kernel is modified.
Keep performance estimates separate from ISA, numerical and runtime acceptance.
