# hwprobe: hardware probes for the `radisa check` hazard rules

> **UNTESTED ON HARDWARE.** This harness was written where no GPU and no ROCm exist. None of the
> probes has run on a gfx1201 (Radeon AI PRO R9700) or any other GPU. `host_runner.cpp` and
> `run_probes.sh` have never been built or run against a real HIP runtime. The list of what was
> and was not checked is at the end of this file.

No rule of `radisa check` has been confirmed on hardware (docs/CHECK.md: `ALU_WAIT_TTMP` "not
confirmed on hardware"; the fault-injection numbers there measure the linter, not the GPU). These
probes are meant to be run later on the R9700 to collect that evidence.

## What a probe is

For each hazard class, `cases/<case>.hip` holds two tiny kernels built around one inline-asm
statement:

* `<case>_hazard`: the wait, nop or VALU that the rule asks for is missing. `radisa check` reports
  the rule (`manifest.json`, `variants.hazard.expect`) on this kernel.
* `<case>_fixed`: the same code in the form LLVM's AMDGPU backend emits, or with the wait the rule
  asks for. `radisa check` reports no error on it.

Both kernels must produce the same result, which `generate.py` computes on the host
(`expected()`). Every kernel has the signature `(unsigned *out, const unsigned *in)` and runs
with workgroups of 256 threads (8 waves of 32). Before each launch, `out` is filled with `0xA5`
bytes.

| result | meaning |
|---|---|
| `_hazard` mismatches | hardware evidence that the rule describes a real hazard |
| `_hazard` has zero mismatches | **nothing**. Races depend on timing, so zero mismatches do NOT show that a rule is unnecessary |
| `_fixed` mismatches | a harness bug (wrong expected value, launch or layout), except for `d16_other_half`, see below |

To give a race many chances, each variant is launched `HWPROBE_ITERATIONS` times (default 200).
Each launch has `HWPROBE_GRID` workgroups (default 512, which is 4096 waves). The runner reports
the number of mismatching words and launches, and the first mismatch.

## Cases

| case | rule on `_hazard` | `_hazard` | `_fixed` | basis |
|---|---|---|---|---|
| `ds_load_wait` | `MISSING_WAIT` (error) | `v_add_nc_u32` reads a `ds_load_b32` destination with the load outstanding | `s_wait_dscnt 0x0` before the reader | RDNA4 ISA §5.7.1 (DScnt) |
| `global_load_wait` | `MISSING_WAIT` (error) | the same with `global_load_b32` | `s_wait_loadcnt 0x0` | RDNA4 ISA §5.7.1 (LOADcnt) |
| `lds_store_barrier` | `LDS_STORE_BEFORE_BARRIER` (error) | `ds_store_b32`, then `s_barrier_signal`/`s_barrier_wait` with no wait; each thread then reads the word stored by the next wave | `s_wait_dscnt 0x0` before `s_barrier_signal` | LLVM AMDGPUUsage, Memory Model GFX12 (LDS operations of different waves may be reordered) |
| `wmma_d_to_a` | `WMMA_D_TO_AB` (error) | the second `v_wmma_f32_16x16x16_f16` reads D of the first as A, directly | `v_nop` between them | RDNA4 ISA §7.12.1; LLVM `GCNHazardRecognizer::fixWMMAHazards` (inserts `V_NOP_e32`) |
| `cmpx_permlane` | `PERMLANE_AFTER_CMPX` (error) | `v_cmpx_gt_u32` (lanes 0-15 stay active), then `v_permlane16_b32` | `v_mov_b32 src0, src0` between them | RDNA4 ISA §7.2.8; LLVM `fixVcmpxPermlaneHazards` (`FeatureVcmpxPermlaneHazard` is in `FeatureISAVersion12`; inserts `V_MOV_B32`, because "V_NOP will be discarded by SQ") |
| `alu_wait_carry` | `MISSING_ALU_WAIT` (error) | `v_add_co_u32` writes the carry SGPR and `v_add_co_ci_u32` reads it next. The pair was already read by a VALU | `s_wait_alu 0xf1ff` (`depctr_va_sdst(0)`) before the reader | LLVM `AMDGPUWaitSGPRHazards.cpp` |
| `alu_wait_ttmp` | `ALU_WAIT_TTMP` (warning) | the same, with `s_cmp_eq_u32 ttmp0, 0` in between, after a VALU read of `s0` (the shape in `radisa/sgpr.py`). LLVM emits no wait here | `s_wait_alu 0xf1ff` | `AMDGPUWaitSGPRHazards.cpp` numbers every ttmp as SGPR 0. Not documented by AMD |
| `d16_other_half` | `MISSING_WAIT` (error) | hi-half D16 load, then lo-half D16 load, then `s_wait_loadcnt 0x1`, which only retires the hi load; `global_store_b16` then reads the lo half | lo-half load first; the store runs while the hi-half load is still outstanding | RDNA4 ISA §5.7.1 (in-order LOADcnt); the relaxation follows LLVM `SIInsertWaitcnts` |

The hardware consequence of each hazard (stale register value, stale EXEC, stale carry, stale LDS
word) is the probes' hypothesis. AMD does not document what goes wrong, and a hazard can show up
in other ways, or not at all.

Special cases:

* **`d16_other_half`**: `_fixed` is itself the claim under test. `radisa check` accepts a 16-bit
  store of one half while a D16 load of the other half is outstanding (570 such stores in the
  libr4d units, see docs/CHECK.md). A mismatch on `_fixed` would be evidence *against* that
  relaxation, not only a harness bug.
* **`alu_wait_ttmp`**: a mismatch on `_hazard` would mean the hardware needs the wait that LLVM
  leaves out, which is a compiler bug worth reporting upstream. Zero mismatches confirm nothing.
* **`wmma_d_to_a`**: the result does not depend on the lane layout of the WMMA operands. WMMA 1
  computes `D = 0 * 1 + C` with `C = 0x3c003c00`, so every f16 half of D is 1.0. WMMA 2 multiplies
  that by ones, so every element is 16.0, and the 8 D VGPRs sum to 128.0. A stale (zero) A gives 0.

Design notes. These keep `radisa check` exact on the generated code: only the probed rule fires.

* Global memory is addressed as a scalar per-workgroup base plus a 32-bit VGPR offset (SADDR
  form). With 64-bit per-lane pointers, the compiler's own VCC carry chains trip
  `MISSING_ALU_WAIT` under clang 18, whose output has no `s_wait_alu` at all.
* The SGPR probes (`cmpx_permlane`, `alu_wait_*`) start and end with 4 `ds_nop`. LLVM's
  `AMDGPUWaitSGPRHazards` (and `radisa/sgpr.py`) forget all tracked SGPR pairs after them. The
  pass does not look inside inline asm, so without them, compiler code before or after the asm
  could create or inherit an SGPR hazard.
* `cmpx_permlane` restores EXEC inside the asm and lists `"exec"` as clobbered. clang warns
  (`-Winline-asm`) that exec is a reserved register. That is harmless here, and the clobber keeps
  `ASM_UNDECLARED_CLOBBER` quiet.
* `wmma_d_to_a` uses the fixed registers v64-v95, all declared as clobbered.

## Running (on the R9700)

```bash
bash hwprobe/run_probes.sh [OUT_DIR]        # default hwprobe/build (git-ignored)
```

Environment variables:

* `ROCM_PATH` (default `/opt/rocm`).
* `HWPROBE_CLANG`: the device compiler, default `$ROCM_PATH/llvm/bin/clang++`.
* `HIPCC`: the host compiler, default `$ROCM_PATH/bin/hipcc`.
* `HWPROBE_ITERATIONS`, `HWPROBE_GRID`: positive integers (anything else is exit 2).
* `HWPROBE_CASE`: run only the cases whose name contains this substring. A substring that matches
  no case is exit 2, not an empty success.

There is no way to run on a device that is not gfx1201: the code objects are raw gfx1201 ELFs, and
the runner stops with exit 2 when `gcnArchName` does not start with `gfx1201`. (An earlier
`HWPROBE_ANY_GPU` option only skipped that name check; the HIP runtime's `hipModuleLoad` is
expected to refuse a code object of another ISA anyway. Not checked here: no ROCm.)

Steps:

1. `generate.py --check`: the checked-in `cases/` must match the generator (exit 2 if stale).
2. Each case is compiled to a raw gfx1201 code object, an ELF rather than an offload bundle:
   `clang++ -x hip --offload-arch=gfx1201 --offload-device-only --no-gpu-bundle-output -nogpuinc
   -nogpulib -O3`.
3. `radisa check` runs on those exact code objects and writes `OUT_DIR/lint/<case>.json`, so the
   linter's verdict refers to the bytes that ran. `generate.py --verify-lint OUT_DIR/lint` then
   checks every kernel against the manifest: `_hazard` reports exactly the expected rule and
   severity, `_fixed` reports no error and not the rule, and neither has an `incomplete` finding.
   The exit code of `radisa check` cannot show this, since `_hazard` alone makes it 1. On a
   mismatch (for example a newer compiler changed the code around the asm) nothing runs and the
   script exits 2, because the results would be filed under a rule the bytes do not have.
4. `generate.py --data OUT_DIR/data` writes the inputs (`<case>.in.bin`), the expected outputs
   (`<case>.expect.bin`) and `cases.tsv`.
5. `hipcc host_runner.cpp` builds the runner, which loads each code object with `hipModuleLoad`
   and launches it with `hipModuleLaunchKernel`. It needs only the HIP runtime.
6. Results go to `OUT_DIR/results.json`. Each entry has the case, variant, kernel, `expect_rule`,
   `launches`, `mismatched_words`, `mismatching_launches`, `first_mismatch` and a `verdict`:
   `hazard_observed` or `not_observed` for `_hazard`, `ok` or `FIXED_VARIANT_MISMATCH` for `_fixed`,
   `incomplete` for a kernel whose launches an error cut short. The file is rewritten after every
   kernel and on every error, so a fault or timeout in one probe keeps the counts collected before
   it. `complete` is `false` (and `error` says why, `fixed_variants_ok` is `null`) until the whole
   run has finished.

| exit | meaning |
|---|---|
| 0 | every `_fixed` variant matched in every launch (`_hazard` results are information only) |
| 1 | a `_fixed` variant mismatched |
| 2 | stale cases, build failure, missing toolchain, bad `HWPROBE_ITERATIONS`/`HWPROBE_GRID`, no case matching `HWPROBE_CASE`, a lint verdict that does not match the manifest, `radisa check` could not analyse a code object, wrong device, or HIP runtime error |

Regenerate after editing `generate.py`: `python3 hwprobe/generate.py`. `cases/*.hip` and
`cases/manifest.json` are generated and never edited by hand.

## What was verified here, and what was not

Verified (Ubuntu LLVM/clang 18.1.3, no GPU; `tests/test_hwprobe.py`):

* Generation is deterministic, and the checked-in `cases/` match the generator.
* Spot checks of the host-predicted results (neighbour wrap, permlane sentinel lanes, both carry
  outcomes present, D16 fill pattern, WMMA 128.0).
* clang 18 compiles every case for gfx1201 with `-nogpuinc -nogpulib`. `radisa check` reports
  exactly the expected rule (and severity) on each `_hazard` kernel and no error on each `_fixed`
  kernel. The test skips a case, with the reason, when the local clang cannot compile it (never
  with `RAD_ISA_REQUIRE_LLVM=1`).
* `host_runner.cpp` builds with `g++ -Wall -Wextra -Werror` against `mock/hip/hip_runtime.h`, a
  host-only stand-in that copies prepared words into the output buffer. Under the mock, its
  parsing, mismatch counting, JSON and exit codes 0/1/2 behave as described, including the partial
  `results.json` after a load error, and exit 2 for an empty selection or zero iterations.
* `generate.py --verify-lint` on real `radisa check` reports of the compiled cases, and on reports
  edited to drop the expected rule or add an error to `_fixed` (both caught).
* Once, by hand: `run_probes.sh` end to end, with clang 18 as `HWPROBE_CLANG` and a `g++ -I
  hwprobe/mock` wrapper as `HIPCC`. The code objects built, `lint/*.json` showed the expected rule
  per kernel, and the runner ran under the mock.

NOT compiled or run:

* `host_runner.cpp` with a real `hipcc` / HIP runtime (`hipModuleLoad`, `hipModuleLaunchKernel`,
  `hipGetDeviceProperties().gcnArchName`). Never built against ROCm headers.
* `run_probes.sh` with ROCm's clang and hipcc, on any GPU.
* Any probe kernel on hardware. No mismatch count, in either direction, exists yet.
* The probes with LLVM 20-24 / ROCm. The asm text is fixed, but the compiler code around it
  (where LLVM 20+ inserts `s_wait_alu` itself) has not been linted with those compilers.
* Whether reading `ttmp0` from user code returns without side effects on gfx1201. The compiler
  itself reads `ttmp9` (workgroup id) in every kernel here, which suggests it does.
* Whether `--no-gpu-bundle-output` is accepted by every ROCm clang. clang 18 accepts it; if a
  version does not, use `hipcc --genco --offload-arch=gfx1201`. Its output is an offload bundle,
  and whether `hipModuleLoad` accepts one was not checked here.
