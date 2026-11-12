# hwprobe: hardware probes for the `radisa check` hazard rules

> **Run on hardware once.** On 2026-11-12 every probe ran on one Radeon AI PRO R9700 (gfx1201,
> GPU 0) with AMD clang 24.0.0git (ROCm llvm-project d6f6cb69), the cases as of commit 1fe68fb,
> 200 launches per variant, grid 512; `host_runner.cpp` was built unmodified against the real HIP
> runtime. Results are under "Results on the R9700". One GPU, one run: what that does and does not
> show is at the end of this file.

The R9700 run is hardware evidence for `MISSING_WAIT` (LDS, global, and the D16 relaxation),
`WMMA_D_TO_AB` and `LDS_STORE_BEFORE_BARRIER`. For `PERMLANE_AFTER_CMPX`, `MISSING_ALU_WAIT` and
`ALU_WAIT_TTMP` no hazard was observed, which shows nothing either way. The fault-injection numbers
in docs/CHECK.md measure the linter, not the GPU.

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
| `lds_store_barrier` | `LDS_STORE_BEFORE_BARRIER` (error) | `ds_store_b32`, then `s_barrier_signal`/`s_barrier_wait` with no wait; each thread then reads the word stored by the next wave | `s_wait_dscnt 0x0` before `s_barrier_signal` | LLVM AMDGPUUsage, Memory Model GFX12. LLVM emits this wait only for a release fence that orders LDS together with global memory (`SIGfx12CacheControl::insertWait`, `DSCnt \|= IsCrossAddrSpaceOrdering`) and none at `S_BARRIER` with `FeatureBackOffBarrier` (`SIInsertWaitcnts`), so an LDS-only fence or a bare barrier gets none. The rule is an error because the R9700 needs the wait here: `_hazard` mismatched in 137 and 152 of 200 launches |
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

## Results on the R9700 (2026-11-12)

One Radeon AI PRO R9700 (GPU 0), AMD clang 24.0.0git (ROCm d6f6cb69), cases as of commit 1fe68fb,
`HWPROBE_ITERATIONS=200`, `HWPROBE_GRID=512` (131,072 words per launch for one word per thread).
`host_runner.cpp` built unmodified against the real HIP runtime. The kernels were launched, so
`--verify-lint` accepted every code object (`run_probes.sh` stops before launching otherwise).
Every `_fixed` variant: 0 mismatching launches.

| case | `_hazard`: mismatching launches of 200 | words | reading |
|---|---|---|---|
| `ds_load_wait` | 200 | all 26,214,400 | hazard observed |
| `global_load_wait` | 200 | all 26,214,400 | hazard observed |
| `wmma_d_to_a` | 200 | all 26,214,400 | hazard observed: the result is 120.0 instead of 128.0 |
| `lds_store_barrier` | 137 and 152 (two runs) | 8,576 and 8,960 | hazard observed |
| `d16_other_half` | 200 | not recorded here | hazard observed; `_fixed` clean, so no evidence against the D16 half-store relaxation |
| `cmpx_permlane` | 0 | 0 | not observed (proves nothing) |
| `alu_wait_carry` | 0 | 0 | not observed (proves nothing) |
| `alu_wait_ttmp` | 0 | 0 | not observed (proves nothing) |

`generate.HARDWARE_RUNS` (and `manifest.json` `hardware_runs`) records the same.

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
   The exit code of `radisa check` cannot show this, since `_hazard` alone makes it 1 (or 0 for
   `alu_wait_ttmp`, whose rule is a warning). On a
   mismatch (for example a newer compiler changed the code around the asm) nothing runs and the
   script exits 2, because the results would be filed under a rule the bytes do not have.
4. `generate.py --data OUT_DIR/data` writes the inputs (`<case>.in.bin`), the expected outputs
   (`<case>.expect.bin`) and `cases.tsv`.
5. `hipcc host_runner.cpp` builds the runner, which loads each code object with `hipModuleLoad`
   and launches it with `hipModuleLaunchKernel`. It needs only the HIP runtime.
6. Results go to `OUT_DIR/results.json` (`schema` `radisa-hwprobe-results-1`, `device`, `arch`,
   `iterations`, `complete`, `error`, `results`, `fixed_variants_ok`). Each entry of `results` has
   the case, variant, kernel, `expect_rule`, `launches`, `words_per_launch`, `mismatched_words`,
   `mismatching_launches`, `first_mismatch` (launch, index, got, want, or `null`) and a `verdict`:
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

## What was verified, and what was not

On the R9700 (2026-11-12, see "Results on the R9700"):

* `run_probes.sh` end to end with ROCm's clang and hipcc: case build, `radisa check` and
  `--verify-lint` on the code objects that ran, `host_runner.cpp` against the real HIP runtime
  (`hipModuleLoad`, `hipModuleLaunchKernel`, `gcnArchName`), 200 launches per variant.
* Every `_fixed` variant matched the host-predicted result in every launch, so the expected values,
  launch geometry and buffer layout are right.

Without a GPU (Ubuntu LLVM/clang 18.1.3; `tests/test_hwprobe.py`):

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

NOT run:

* A second GPU (another R9700 or another gfx1201 board), or the R9700 at other clocks or loads. The
  counts above are from one device.
* More than three runs. Commit 7448fd5 was re-run on the same device with the same outcome per
  case (`lds_store_barrier` wrong in 143 of 200 launches); the table above is of commit 1fe68fb.
* The probes with LLVM 20-23. The asm text is fixed, but the compiler code around it (where LLVM
  20+ inserts `s_wait_alu` itself) was linted only with clang 18 and AMD clang 24.
* A probe that makes `cmpx_permlane`, `alu_wait_carry` or `alu_wait_ttmp` visible. Their zero
  mismatches say nothing about whether the hazards are real; `alu_wait_ttmp` (a wait LLVM leaves
  out) stays unconfirmed in both directions.
* Whether reading `ttmp0` from user code returns without side effects on gfx1201. The compiler
  itself reads `ttmp9` (workgroup id) in every kernel here, which suggests it does.
* Whether `--no-gpu-bundle-output` is accepted by every ROCm clang (clang 18 and AMD clang 24
  accept it). If a version does not, use `hipcc --genco --offload-arch=gfx1201`; whether
  `hipModuleLoad` accepts the offload bundle that produces was not checked.
