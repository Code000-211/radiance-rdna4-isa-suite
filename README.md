# Radiance RDNA4 ISA Suite

Tools that check GPU kernels for the AMD Radeon R9700 (`gfx1201`, RDNA4) without running them.

The main tool is **`radisa check`**, a linter for kernel machine code. It finds bugs that the
compiler accepts and the GPU then runs wrong, for example a register that is read before the load
that fills it has finished.

The suite also contains three smaller tools: an audit of every kernel in a build, a replay of
recorded kernel inputs on the host, and an offline cost planner. They are described
[further down](#the-other-tools).

## Contents

- [What you need](#what-you-need)
- [A first check](#a-first-check)
- [What it finds](#what-it-finds)
- [Reading the result](#reading-the-result)
- [Silencing a finding](#silencing-a-finding)
- [Checking many files, CI and editors](#checking-many-files-ci-and-editors)
- [How fast it is](#how-fast-it-is)
- [What has been verified](#what-has-been-verified)
- [The other tools](#the-other-tools)
- [Tests](#tests)
- [Where to read more](#where-to-read-more)

## What you need

- **Python 3.10 or newer.** No third-party packages.
- **An LLVM toolchain that knows `gfx1201`**: `clang++` and `llvm-mc`. Use the compiler your
  kernels are built with. ROCm's clang 20 and AMD clang 24 are supported.
- Only for the replay and planner tools: CMake, a C++20 compiler and a Radiance checkout.

The suite is meant to sit in a Radiance checkout as `tools/rdna4-isa-suite/`. It does not change
any Radiance file. The commands below assume:

```bash
S=tools/rdna4-isa-suite
```

> **Use LLVM 20 or newer.** Older compilers do not insert `s_wait_alu`, so `radisa` reports
> `MISSING_ALU_WAIT` on correct compiler output.

## A first check

Save this as `snippet.s`. It loads a value and uses it straight away:

```asm
	global_load_b32 v1, v[2:3], off
	v_add_f32 v4, v1, v1
	s_endpgm
```

```bash
bash "$S/check-kernel.sh" snippet.s
```

```text
snippet.s:2:1: error: [MISSING_WAIT] v_add_f32_e32 reads v1 while global_load_b32 (snippet.s:1) may still be outstanding; insert s_wait_loadcnt 0 (or lower) before it
    v_add_f32_e32 v4, v1, v1
    see RDNA4 ISA §5.7.1 Memory Dependency Counters, PDF p.62
radisa check: 1 kernel(s), 3/3 instructions in the AMD operand model; 1 error(s), 0 warning(s), 0 note(s) -> FAIL
```

The finding names the line, the rule, the fix and the place in AMD's manual.

The same command checks the three kinds of input:

```bash
# A HIP source file. It is compiled for gfx1201 with the flags your build uses.
bash "$S/check-kernel.sh" libr4d/r4d_attn_paged_h128_gqa4.hip \
  --compile-commands build/compile_commands.json

# Hand-written assembly.
bash "$S/check-kernel.sh" snippet.s

# A compiled code object, here one kernel of it.
bash "$S/check-kernel.sh" build/.../r4d_gemm_fp8a8.hip.o --kernel gemm_8a8_prefill
```

## What it finds

`radisa check` reports three levels. Only errors make the check fail.

**Errors** are bugs the hardware manual documents, or that LLVM's own backend repairs in compiler
output. Hand-written assembly and inline `asm` do not get those repairs, which is where these
bugs come from.

| Rule | What is wrong |
|---|---|
| `MISSING_WAIT` | A register is used while the load that writes it may still be running. |
| `MISSING_ALU_WAIT` | A scalar register is read before its write is committed, with no `s_wait_alu`. |
| `LDS_STORE_BEFORE_BARRIER` | A write to shared memory may still be running when a barrier releases the other waves. |
| `WMMA_D_TO_AB` | A matrix instruction reads the result of the one before it too early. |
| `PERMLANE_AFTER_CMPX` | `v_permlane` follows `v_cmpx` with nothing in between that the hardware keeps. |
| `CFG_FALLTHROUGH` | A path through the kernel never reaches `s_endpgm`. |
| `VGPR_OUT_OF_RANGE` | A register above the number the kernel reserved. The hardware does not trap. |

**Warnings** are probably wrong or worth a look: a register read that nothing wrote
(`UNINITIALIZED_READ`), inline `asm` that changes a register without declaring it
(`ASM_UNDECLARED_CLOBBER`), a function that returns with a load still running
(`CALLEE_RETURN_OUTSTANDING`), register spills and scratch memory.

**Notes** are information only.

Kernels are checked along every path, across branches and loops. Functions that kernels call are
checked too.

To list every rule, or see the reasoning behind one:

```bash
PYTHONPATH="$S" python3 -m radisa rules
PYTHONPATH="$S" python3 -m radisa rules --explain MISSING_WAIT
```

## Reading the result

| Exit code | Status | Meaning |
|---|---|---|
| 0 | `PASS` | No errors. Warnings and notes may be present. |
| 1 | `FAIL` | At least one error. |
| 2 | | The command line or the configuration is wrong. Nothing was checked. |
| 3 | `INCOMPLETE` | Part of the code could not be analysed, so it is **not** known to be clean. |

Useful options:

| Option | Effect |
|---|---|
| `--json FILE` | Write the full report to a file. |
| `--format sarif\|github\|json` | Change what is printed. |
| `--strict` | Fail on warnings too. |
| `--kernel NAME` | Check only kernels whose name contains `NAME`. |

## Silencing a finding

Sometimes a finding is correct in general but does not apply to your code. There are three ways
to say so, from narrowest to widest.

**In the source, with a reason.** The reason is required.

```cpp
__builtin_amdgcn_s_barrier();  // radisa: ignore[LDS_STORE_BEFORE_BARRIER] no wave reads another's data here
```

**In a baseline**, to accept the findings a code base has today and see only new ones:

```bash
bash "$S/check-kernel.sh" k.hip --write-baseline radisa-baseline.json
bash "$S/check-kernel.sh" k.hip --baseline radisa-baseline.json
```

**In `.radisa.json`**, for a whole project. The nearest file above the checked file is used.

```json
{ "severity": { "REGISTER_SPILL": "note" }, "disable": ["HIP_NO_LAUNCH_BOUNDS"] }
```

A silenced error is never hidden completely: the summary line still names it.

## Checking many files, CI and editors

```bash
# Every HIP file of a build, 8 at a time, as a SARIF report.
bash "$S/check-tree.sh" --compile-commands build/compile_commands.json --jobs 8 \
  --format sarif --out radisa.sarif
```

| Where | How |
|---|---|
| Before a commit | `.pre-commit-hooks.yaml` |
| GitHub Actions | `.github/workflows/radisa.yml`, with annotations on the changed lines |
| Editors | Any SARIF viewer reads `--format sarif` |
| Coding agents | `radisa hook` checks a file right after an agent edits it and blocks on errors. Setup: [docs/CHECK.md](docs/CHECK.md), "As an edit hook". |

## How fast it is

A HIP file is optimised as a whole, but only the kernels whose code changed are compiled and
checked again. The rest come from a cache in `~/.cache/radisa/`. The result is the same as a full
compile: 4475 of 4475 kernels in 117 units were byte-identical.

Measured with AMD clang 24 on the R9700 host: a first check took 17 seconds per file on average
over 113 files. One very large file (`r4d_gemm_iqr`, 704 kernels) took 32 minutes. If a run is
stopped, the kernels it finished stay in the cache, so the next run continues from there.

`--no-incremental` turns the cache off.

## What has been verified

Measured on 2026-11-12 with AMD clang 24 (ROCm `d6f6cb69`) and one Radeon AI PRO R9700:

- **No false errors on real code.** 114 source files, 6448 kernels, 11.79 million instructions:
  112 `PASS`, 2 files without kernels, no `FAIL`, no `INCOMPLETE`.
- **Three rules confirmed on the GPU.** Small test kernels that break `MISSING_WAIT` and
  `WMMA_D_TO_AB` gave wrong results in 200 of 200 runs. The one for `LDS_STORE_BEFORE_BARRIER`
  did in 137 to 152 of 200. The corrected versions were always right. See
  [hwprobe/README.md](hwprobe/README.md).
- **Not confirmed on the GPU:** `PERMLANE_AFTER_CMPX`, `MISSING_ALU_WAIT` and `ALU_WAIT_TTMP`.
  Their test kernels showed no wrong result, which proves neither that the rules are needed nor
  that they are not.

A `PASS` means the implemented rules found nothing. It is not a proof that a kernel is correct.
What is not checked is listed in [docs/CHECK.md](docs/CHECK.md), "Not checked".

## The other tools

### Audit a whole build: `radisa collect` and `audit`

```bash
bash "$S/run.sh" BUILD_DIR SOURCE_ROOT NEW_RESULT_DIR
```

Finds every `gfx1201` code object of a build, checks its descriptors and that every byte decodes,
and writes JSON, Markdown and JUnit reports. It is a gate for a build, not a certification.
Details: [docs/ISA_V02.md](docs/ISA_V02.md), [docs/RULE_COVERAGE.md](docs/RULE_COVERAGE.md).

### Replay kernels on the host: `radreplay`

```bash
bash "$S/replay.sh" validate MANIFEST.json                 # no GPU, no native code
bash "$S/check-replay-host.sh" RADIANCE LIBREF.so NEW_BUILD_DIR
bash "$S/check-all-host.sh" "$PWD" "$PWD/host-out"         # the full host pipeline, with ASan
```

Runs recorded kernel inputs through Radiance's real host code and compares the results. The
recordings are validated before any native code is loaded. Recording on a GPU
(`check-target-gpu.sh`) needs the optional patches in `integration/`.
Details: [docs/REPLAY_FORMAT.md](docs/REPLAY_FORMAT.md), [integration/README.md](integration/README.md).

### Offline planner: `radplan`

```bash
bash "$S/check-planner-flow.sh" "$PWD" "$PWD/offline-flow-v06"   # the output directory must be new
```

Estimates the cost of workloads made of several operators (GEMM, attention, MoE, communication
between two GPUs). **These are estimates from assumed rates, not measurements and not GPU
emulation.** Every new scenario is marked `NEEDS_MEASUREMENT`.
Details: [docs/OPERATOR_PLANNER.md](docs/OPERATOR_PLANNER.md), [docs/V06_REVIEW.md](docs/V06_REVIEW.md).

## Tests

```bash
bash "$S/check.sh"            # linter tests (597) and replay tests (142)
bash "$S/check-planner.sh"    # planner tests (257)
```

- `check.sh` needs LLVM 20 or newer and a C++ compiler. With an older LLVM, six tests fail
  instead of being skipped. That is intended: the script is the full check for the target
  toolchain.
- The replay and planner tests need `RADIANCE_SOURCE` to point at a Radiance checkout.
- Without ROCm, the linter tests alone still run and skip what they cannot do:
  `PYTHONPATH=$PWD python3 -m unittest discover -s tests` gives 597 tests, OK, 7 skipped with
  Ubuntu's LLVM 18.
- `sha256sum -c SHA256SUMS` verifies the shipped files.

## Where to read more

| Topic | File |
|---|---|
| Every rule, the analysis model, options, the hook, what is not checked | [docs/CHECK.md](docs/CHECK.md) |
| Hardware test kernels and their results | [hwprobe/README.md](hwprobe/README.md) |
| Build audit | [docs/ISA_V02.md](docs/ISA_V02.md), [docs/RULE_COVERAGE.md](docs/RULE_COVERAGE.md), [docs/REVIEW.md](docs/REVIEW.md) |
| Replay formats and the optional patches | [docs/REPLAY_FORMAT.md](docs/REPLAY_FORMAT.md), [integration/README.md](integration/README.md) |
| Planner | [docs/OPERATOR_PLANNER.md](docs/OPERATOR_PLANNER.md), [docs/FLOW_FORMAT.md](docs/FLOW_FORMAT.md), [docs/V06_REVIEW.md](docs/V06_REVIEW.md), [docs/SOURCES_V06.md](docs/SOURCES_V06.md) |
| Older planner commands (v0.5) | [docs/V05_README.md](docs/V05_README.md) |
| Example plans and logs | `evidence/v06/` (current), `evidence/v05*` and `evidence/cpu-comparison/` (historical) |
| Working on this repository with a coding agent | [AGENTS.md](AGENTS.md) |
