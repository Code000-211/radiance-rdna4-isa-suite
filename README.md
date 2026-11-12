# Radiance RDNA4 ISA Suite

Offline tools for Radiance's `gfx1201` (RDNA4, R9700) kernels. The suite ships as a
`tools/rdna4-isa-suite/` overlay for the Radiance source tree. Extracting it changes no
Radiance file, kernel or production dispatcher; the AQL integration patches are opt-in.
No clock, voltage, power, watchdog, memory scope, cache hint or kernel is modified.

| Part | Package | What it does |
|---|---|---|
| Kernel lint | `radisa check` | checks one HIP unit, assembly file or code object against AMD's RDNA4 ISA |
| Binary audit | `radisa collect/audit` | static gate over every `gfx1201` code object of a build |
| Replay | `radreplay` | replays declared kernel snapshots against real Radiance host code |
| Offline planner | `radplan` | multi-operator cost scenarios (v0.6), not GPU emulation |

Python 3.10+ with no third-party packages. An LLVM toolchain with `gfx1201` support is
needed for compiling and disassembling; CMake, a C++20 compiler and clang ASan for the host
pipeline. Keep performance estimates separate from ISA, numerical and runtime acceptance.

## Lint one kernel: `radisa check`

```bash
S=tools/rdna4-isa-suite
bash "$S/check-kernel.sh" libr4d/r4d_attn_paged_h128_gqa4.hip \
  --compile-commands build/compile_commands.json     # HIP source -> gfx1201 machine code
bash "$S/check-kernel.sh" snippet.s                  # hand-written assembly
bash "$S/check-kernel.sh" build/.../r4d_gemm_fp8a8.hip.o --kernel gemm_8a8_prefill   # code object
```

Every instruction is mapped onto AMD's machine-readable RDNA4 operand table (XML, MIT), and the
memory counters are modelled over the whole CFG. Findings point at source lines and cite the
manual section and PDF page:

- `MISSING_WAIT`: a register used while a load/LDS/SMEM/sample access may still be outstanding
  (typically after inline-asm memory instructions, which the compiler does not track), including
  D16 half-register loads;
- `MISSING_ALU_WAIT`: an SGPR read before its write is committed, without `s_wait_alu`;
- `LDS_STORE_BEFORE_BARRIER`: an LDS write still outstanding at a workgroup barrier;
- `CFG_FALLTHROUGH`: a path that never reaches `s_endpgm`;
- `VGPR_OUT_OF_RANGE`: a VGPR above the kernel descriptor's allocation (the hardware does not trap);
- WMMA and VOPD hazards, VGPR deallocation errors, spills, scratch and lexical HIP warnings.

Exit codes: 0 `PASS`, 1 `FAIL`, 3 `INCOMPLETE`. `--json FILE` writes the full report,
`--strict` fails on warnings too.

**Compiler.** `--compile-commands` reuses the CMake flags and compiler of that file. Both LLVM 20
(ROCm clang) and Radiance's own LLVM 24 compiler (`/opt/rocm-repeatable`) are supported; the
latter lacks `llc` and `llvm-extract`, so `opt` + `clang` are used instead
(reported as `incremental.codegen`).

**Incremental by default.** The unit is optimised once; only kernels whose IR fingerprint changed
are compiled and checked again (`--jobs`, cache in `--cache-dir` or `~/.cache/radisa/<unit>`).
This gives byte-identical code and descriptors to a whole-unit compile (4475 of 4475 kernels in
117 libr4d units with LLVM 24). `--no-incremental` restores the single compile.

On all 117 libr4d units: 115 `PASS`, 2 `INCOMPLETE` (`BYTE_COVERAGE`), no error findings.
`radisa hook` runs the same check as a Claude Code PostToolUse hook.
Details, rules and what is **not** checked: `docs/CHECK.md`.

## Audit a whole build: `radisa collect` + `audit`

```bash
bash tools/rdna4-isa-suite/run.sh BUILD_DIR SOURCE_ROOT NEW_RESULT_DIR
```

Extracts every `gfx1201` code object (ELF, `.hip_fatbin`, offload bundles, archives), checks
descriptors, byte coverage and decoded instructions, and writes JSON, Markdown and JUnit
reports. Kernel identity is code-object SHA-256 + descriptor symbol. This is a static
gate, not conformance certification. See `docs/ISA_V02.md`, `docs/RULE_COVERAGE.md`.

## Replay kernels: `radreplay`

```bash
S=tools/rdna4-isa-suite
bash "$S/replay.sh" validate MANIFEST.json            # no GPU, no native code
bash "$S/check-replay-host.sh" RADIANCE LIBREF.so NEW_BUILD_DIR
bash "$S/check-all-host.sh" "$PWD" "$PWD/host-out"   # full host pipeline incl. ASan
```

Manifests (`radiance-kernel-replay/1`) are declared fixtures validated before any native code
loads; results run through the real Radiance registry, host backend and comparison code.
`check-target-gpu.sh` traces one approved single-plugin corpus on a `gfx1201` GPU and requires
the AQL patches in `integration/`. Formats: `docs/REPLAY_FORMAT.md`, `integration/README.md`.

## Offline planner: `radplan` (v0.6)

**Multi-operator offline cost planning, not GPU emulation or production certification.**
v0.6 keeps the v0.5 dense-GEMM calibration and adds attention/QSA, routed MoE, directional TP2
communication cost models, dependency/resource scheduling, memory lifetimes, finite candidate
search, profile sensitivity and a source-bound **partial** `qwen4_exp` model importer.

```bash
bash tools/rdna4-isa-suite/check-planner-flow.sh "$PWD" "$PWD/offline-flow-v06"
```

The output directory must be new. This verifies source hashes, runs the planner tests, builds
the 37-hypothesis catalog, plans seven example workloads, compares 96 TP2-window assignments,
runs a three-profile sensitivity study and expands a declared model configuration. The final
strict-mode test deliberately requires exit 2: no calibration is fabricated.

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
bash "$S/planner.sh" attention-cpu-demo \
  --worker "$PWD/host-out/normal/rad-kbench-replay" \
  --libref "$PWD/host-out/normal/libref.so" \
  --out "$PWD/attention-cpu-v06" --execute-trusted-code
```

New operator candidates are **cost hypotheses**, not compiled or executed GPU kernels; all new
operator/graph scenarios stay `NEEDS_MEASUREMENT`, and `--require-calibrated` exits 2. Examples
use synthetic routing and assumed rates. A calculated microsecond value is a conditional
scenario, not a measured R9700 latency. The attention demo runs genuine Radiance CPU attention
(`ref_attn_paged`) on twenty fixtures checked against independent Python references; FP32 CPU
results are not FP8 GPU calibration. A changed source hash requires adapter review, not removal
of the hash check.

## Tests

```bash
bash tools/rdna4-isa-suite/check.sh            # tests/ (220) + replay_tests/ (142)
bash tools/rdna4-isa-suite/check-planner.sh    # planner_tests/ (257)
```

`check.sh` requires LLVM and a C++ compiler. Replay and planner tests need
`RADIANCE_SOURCE` pointing at the Radiance checkout. `SHA256SUMS` covers the shipped files.

## Documentation

- `docs/CHECK.md`: `radisa check` rules, counter model, compiler paths, hook, gaps.
- `docs/ISA_V02.md`, `docs/RULE_COVERAGE.md`, `docs/REVIEW.md`: binary audit gate.
- `docs/REPLAY_FORMAT.md`, `integration/README.md`: replay manifests and AQL patches.
- `docs/V06_REVIEW.md`, `docs/OPERATOR_PLANNER.md`, `docs/FLOW_FORMAT.md`, `docs/SOURCES_V06.md`: planner v0.6.
- `docs/V05_README.md`: retained dense-GEMM planning/calibration commands.
- `evidence/v06/`: current example plans, logs and CPU fixture runs;
  `evidence/cpu-comparison/`, `evidence/v05*`: **historical v0.5** run.
  `evidence/RELEASE_VALIDATION.json` still lists the v0.6 release test counts.
