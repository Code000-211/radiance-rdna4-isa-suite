# `radisa check`: lint one gfx1201 kernel

`radisa check` takes **one** HIP translation unit, assembly file or code object and checks every
gfx1201 kernel in it against rules derived from AMD's machine-readable RDNA4 ISA (XML) and the
RDNA4 ISA Reference Guide. Findings point at source lines and cite the manual section and PDF page.
Two rules the ISA guide does not state (`s_wait_alu`, the LDS write before a barrier) follow the
compiler that implements them, LLVM's AMDGPU backend, and cite it.

```bash
S=tools/rdna4-isa-suite
# HIP source: compiled for gfx1201 (device only), then checked as machine code
bash $S/check-kernel.sh libr4d/r4d_attn_paged_h128_gqa4.hip --compile-commands build/compile_commands.json
# Hand-written assembly; text without .amdhsa_kernel is checked as one wave32 sequence
bash $S/check-kernel.sh my_snippet.s
# Machine code: ELF code object, offload bundle, fat binary, .hip.o, archive, shared library
bash $S/check-kernel.sh build/libr4d/CMakeFiles/r4d.dir/r4d_gemm_fp8a8.hip.o --kernel gemm_8a8_prefill
```

Output is compiler-style, one finding per block:

```
libr4d/foo.hip:15:12: error: [MISSING_WAIT] v_add_f32_e32 reads v1 while ds_load_b32 (foo.hip:14)
may still be outstanding; insert s_wait_dscnt 0 (or lower) before it. The producer is inline asm:
the compiler does not insert waits for memory instructions inside asm statements
    v_add_f32_e32 v1, 1.0, v1
    see RDNA4 ISA §5.7.1 Memory Dependency Counters, PDF p.62
radisa check: 2 kernel(s), 27/27 instructions in the AMD operand model; 1 error(s), ... -> FAIL
```

`--json FILE` writes the full report (every kernel, resources, operand-model coverage, findings).

| exit | status | meaning |
|---|---|---|
| 0 | `PASS` / `NO_KERNELS` | no error-level finding (`--strict`: no warning either) |
| 1 | `FAIL` | at least one error, a compile error or an assembler error |
| 3 | `INCOMPLETE` | the tool could not analyse the input (decoder, source map, toolchain) |

## Compiling HIP

The compiler is, in order: `--cxx`, `$RADISA_HIPCC`, `$ROCM_PATH/llvm/bin/clang++`, `hipcc`,
`clang++`. With `--compile-commands`, the flags CMake used for that file are reused (output, depfile
and `-c` flags are dropped) and `-x hip --offload-arch=gfx1201 --offload-device-only -O3
-gline-tables-only -S` is added; project `-O`/`-m` flags still win. Extra flags: `--flag "-I inc -DX=1"`.
`-gline-tables-only` only adds line tables (`-g` crashes clang 20.1.2 on some libr4d units).

### Only the kernels an edit changed (default)

The optimiser needs the whole unit: whether a helper is inlined depends on how many kernels call it,
so optimising one kernel alone gives different code (seen on `r4d_ltx`). After it, every libr4d kernel
is a single self-contained function. `radisa check` therefore:

1. runs `clang -emit-llvm` with the flags above once for the unit (the optimised module);
2. fingerprints each kernel: its IR plus the globals, declarations, attributes and non-debug metadata
   it references, the `llc` options (the `-mllvm` options and `-O` level clang's own code generation
   gets, read from `clang -###`), the `llc` version and the ISA table; debug locations and the source
   offsets of inline-asm strings are left out;
3. runs `llvm-extract` + `llc` + `llvm-mc` and the checks only for kernels with a new fingerprint, in
   parallel (`--jobs`). ROCm's LLVM 24 packages, and the compiler Radiance builds from them
   (`/opt/rocm-repeatable`), ship neither `llvm-extract` nor `llc`; there the same two steps are
   `opt -passes=internalize,globaldce` (keeps the kernel and every global variable) and
   `clang -x ir -Xclang -disable-llvm-passes` (code generation only, same `-O` level and `-mllvm`
   options). The JSON report names the one used as `incremental.codegen` (`llc` or `clang`);
4. takes the other kernels' results from the cache (`--cache-dir`, default `$RADISA_CACHE` or
   `~/.cache/radisa/<unit>`), with their source lines moved along the edit by a diff of the old and new
   file; a kernel whose finding line was itself edited is compiled again.

Per-kernel `llc` gives the same machine code as compiling the unit: 241 of 241 kernels in seven
libr4d units, and identical findings from `--no-incremental` and the default mode on `attn_paged_h256_gqa12`,
`ltx` and `ar_oneshot_2rank_exact`. The `opt` + `clang` path was compared the same way with Radiance's
LLVM 24 compiler and the flags of its `compile_commands.json`: code bytes and kernel descriptor are
identical for 4475 of 4475 kernels in 117 libr4d units, and for 32 of 32 kernels sampled from the
2069 of `r4d_gemm_iqr`. When non-kernel functions survive optimisation, module-level code
generation (the LDS layout across kernels) could differ; then the whole module goes through `llc` and
nothing is cached. `--no-incremental` restores the single `clang -S` compile.

## What is checked

| rule | severity | basis |
|---|---|---|
| `MISSING_WAIT` | error | §5.7.1 counters, PDF pp.62-65: a register is read or overwritten while a load/LDS/SMEM/sample/BVH/export that writes it may still be outstanding on some CFG path |
| `LDS_STORE_BEFORE_BARRIER` | error | LLVM AMDGPU User Guide, memory model GFX12: an LDS write may still be outstanding (DScnt) when `s_barrier_signal` / `s_barrier_wait` is reached, so the waves the barrier releases can read LDS before the write has landed |
| `MISSING_ALU_WAIT` | error | LLVM `AMDGPUWaitSGPRHazards`: an SGPR that vector instructions read is read before its last write is committed and no `s_wait_alu` lies in between |
| `ALU_WAIT_TTMP` | warning | the same hazard at a place where the compiler itself emits no wait (see below) |
| `CFG_FALLTHROUGH` | error / note | §5.5: a path runs past the last instruction of the kernel, so it never executes `s_endpgm` (error); a note when that instruction is not reached by direct branches, and for a snippet, which is a fragment |
| `VGPR_OUT_OF_RANGE` | error | §3.3.2: an instruction names a VGPR above what the kernel descriptor allocates; the hardware does not trap, the instruction works on a register the kernel does not own. Not checked with dynamic VGPRs (`s_alloc_vgpr`). Replaces the VOPD-only `VOPD_REGISTER_RANGE` warning for the same kernel |
| `WMMA_D_TO_AB` | error | §7.12.1: WMMA D feeds the next WMMA's A/B without separation |
| `VOPD_*` | error / warning | §7.8, §16.11: raw VOPD legality (opcodes, banks, SRC2 parity, wave32, DPP) |
| `PERMLANE_AFTER_CMPX` | error | §7.2.8 |
| `USE_AFTER_DEALLOC`, `DEALLOC_*`, `DYNAMIC_*` | error / warning | §3.3.3 dynamic VGPRs, MSG_DEALLOC_VGPRS |
| `REACHABLE_CODE_END` | error | §2.4 shader padding |
| `SGPR_PAIR_GUARD` | warning | documented VALU-read/SALU-write SGPR-pair pattern |
| `WATERFALL_CANDIDATE` | note | §7.11: EXEC loop around relative VGPR indexing |
| `REGISTER_SPILL`, `SCRATCH_USE` | warning | code-object metadata |
| `HIP_INLINE_ASM_MEMORY` | warning | lexical: asm statement with a memory instruction and no wait in it |
| `HIP_WAVE64_WIDTH`, `HIP_LEGACY_WAITCNT` | warning | lexical: shuffle width 64; S_WAITCNT = S_WAIT_IDLE (§16.5) |
| `HIP_NO_LAUNCH_BOUNDS` | note | lexical |
| `COMPILE`, `ASSEMBLER` | error | the toolchain rejected the input |

### The counter model (`radisa/waits.py`)

* Every instruction is mapped onto its AMD XML operand list (`radisa/isadb.py`): explicit and
  implicit reads/writes, including EXEC, VCC, SCC and M0. EXEC is added for every VALU/VMEM/export
  instruction (the XML omits it). An instruction the table cannot align counts as *unknown*; its
  registers are still treated as both read and written, never as harmless. The summary line reports
  the coverage.
* Counters come from the encoding: SMEM → KMcnt; VDS → DScnt; VGLOBAL/VSCRATCH/VBUFFER/VIMAGE →
  LOADcnt (loads, returning atomics, `global_inv`) or STOREcnt; VFLAT → that plus DScnt; VSAMPLE →
  SAMPLEcnt; BVH → BVHcnt; export and LDS param/direct loads → EXPcnt.
* LOADcnt, SAMPLEcnt, BVHcnt and DScnt return in issue order, so `s_wait_<cnt> N` retires all but the
  N youngest. KMcnt and EXPcnt are out of order: only 0 retires anything. Combined waits decode
  SIMM16[13:8] and SIMM16[5:0] (§16.5).
* The analysis is a forward may-analysis over the direct-branch CFG (loops included). A later write
  to the same register by the same in-order counter needs no wait. A call (`s_swappc_b64`) is assumed
  to return with drained counters, because LLVM's callee prologue waits for every counter.
* An 8- or 16-bit store reads one half of its data register (`…_d16_hi_…` the high half, the
  others the low half). A D16 load of the other half may stay outstanding across it, as in LLVM
  `SIInsertWaitcnts`; every other reader, and a store whose data register is also its address,
  needs both halves.
* At a workgroup barrier only LDS writes are required to have landed. An LDS or global load may be
  outstanding there (the compiler leaves unused loads outstanding), and so may a global store: a
  barrier with fences scoped to LDS, like Radiance's `lds_barrier()`, orders local memory only.

### The SGPR read hazard (`radisa/sgpr.py`)

On GFX12 an SGPR that a vector instruction has read is committed late. A later read of that
register pair has to wait for the last write to it: `s_wait_alu depctr_sa_sdst(0)` after a scalar
write, `depctr_va_sdst(0)` after a vector write that a vector instruction reads,
`depctr_va_vcc(0)` for VCC. AMD's ISA guide documents the `s_wait_alu` fields but not when they are
needed; the rule is LLVM's `AMDGPUWaitSGPRHazards` pass (ROCm llvm-project `d6f6cb69`), run here over
the finished machine code: where the pass would insert a wait and none is there, it is missing.

* A scalar memory read and a buffer/image/sample instruction commit everything, as in the pass;
  `s_delay_alu` does not.
* The pass is not monotone (a pair that becomes tracked later in a loop changes what an earlier
  scalar read does), so its result at a join depends on the order it visits blocks in. The check
  first solves which pairs are tracked, then the hazards with those fixed. That solution is contained
  in the pass's result for any order, so it reports only waits the compiler inserts.
* The pass numbers every `ttmp` register as SGPR 0. A scalar read of a `ttmp` register between a
  vector write of an SGPR and its vector read makes it emit no wait. The check runs with both
  numberings: a wait both ask for is `MISSING_ALU_WAIT`; a wait only the real register numbers ask
  for is `ALU_WAIT_TTMP`, a warning, because it is in compiler output (2 places in the libr4d
  units, both in `r4d_gemm_mxfp4a8_gemv`) and whether the hardware needs the wait there is not confirmed.

### Measured

The libr4d units of Radiance, compiled with its own compiler (AMD clang 24, ROCm llvm-project
`d6f6cb69`, the flags of `compile_commands.json`), all but `r4d_gemm_iqr` (2069 kernels, left out
for its run time): 117 units, 4477 kernels, 8,438,289 instructions, 786,325 `s_wait_alu`.

| | before | now |
|---|---|---|
| `PASS` | 102 | 115 |
| `FAIL` | 13 (570 `MISSING_WAIT`, all false: D16 half loads) | 0 |
| `INCOMPLETE` | 2 (`BYTE_COVERAGE`) | 2 (the same) |
| error findings | 570 | 0 |

The 570 stores are now counted as `d16_other_half_stores` in the report. No error rule fires on
compiler output; that is evidence, not a proof. Warnings: 2 `ALU_WAIT_TTMP`, the rest as before.

Single faults put into the compiler's assembly of `r4d_attn_paged_h128_gqa4` (36 kernels, one fault
per run, the result is "at least one error"):

| fault | reported |
|---|---|
| a memory wait (`s_wait_loadcnt` …) deleted | 77 of 80 |
| its count raised by one | 76 of 80 |
| moved behind the next instruction | 55 of 60 |
| `s_wait_alu` deleted | 120 of 120 |
| `s_wait_dscnt 0` before a barrier deleted | 20 of 20 |
| WMMA reads the result of the previous WMMA | 30 of 30 |
| `s_endpgm` deleted | 5 of 6 |

The 13 that are not reported, by reading each one:

* 5: the wait stands in front of `global_atomic_*` or between two `global_inv`. It is there for the
  memory ordering of an atomic or a fence; no register depends on it.
* 3: the wait stands in front of `; wave barrier`, a barrier the compiler removed because the
  workgroup is one wave. Nothing waits there.
* 4: the wait was moved behind an `s_delay_alu` or an instruction that does not use the loaded
  value. The program is the same; these are not faults.
* 1: an `s_endpgm` in the middle of a kernel. Without it the code falls into the next block of the
  same kernel, which is a legal path as far as the machine code tells.

`tests/test_check.py` repeats the experiment on a checked-in kernel (`fixtures/check/kernel_reduce.s`):
every wait and the `s_endpgm` removed in turn, each mutant has to fail with exactly the expected rule.

## Not checked

* Numerical correctness, global memory safety, performance beyond the listed resource notes.
* Waits that only a memory ordering asks for: before atomics and fences, around `global_inv` /
  `global_wb`, cross-device release/acquire. Which ordering and scope the source asked for is not in
  the machine code. The one exception is the LDS write before a workgroup barrier.
* An `s_endpgm` missing in the middle of a kernel (see above).
* The waits LLVM puts in front of every non-atomic `scope:SCOPE_SYS` store on GFX12.0
  (`SIGfx12CacheControl::finalizeStore`). Checked against the libr4d units this would be 701
  errors in 8 of the 9 all-reduce units: the compiler emits `s_wait_loadcnt`/`s_wait_storecnt`
  in front of the stores but leaves `global_inv` and the store before it outstanding (it inserts
  the waits as "soft" waits, which the wait pass later merges and relaxes). What the hardware
  needs there is not documented by AMD; it stays unchecked until it is.
* SGPRs need no such rule: since GFX10 every wave gets the full SGPR file (the descriptor field
  is ignored), and the assembler rejects names beyond it.
* `s_delay_alu` (a performance hint, the hardware interlocks without it), callee bodies, indirect
  branches. A call is assumed to return with drained counters.
* `; wave barrier` (a barrier the compiler removed because the workgroup is one wave) is not an
  instruction: a missing wait there is harmless and not reported.

## As a Claude Code hook

`radisa hook` reads a PostToolUse event on stdin, checks the edited file when it matches
`$RADISA_HOOK_INCLUDE` (default `\.(hip|s)$`) and exits 2 with the report on stderr when the check
fails, so the model sees the errors and fixes them before it continues. Anything else never blocks.
In the Radiance checkout, `.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "RADISA_HOOK_INCLUDE='libr4d/.*\\.hip$' RADISA_COMPILE_COMMANDS=build/compile_commands.json PYTHONPATH=tools/rdna4-isa-suite python3 -m radisa hook",
            "timeout": 300
          }
        ]
      }
    ]
  }
}
```

Only `libr4d` targets RDNA4; `libhalo` (gfx1151) and `libr3d` (RDNA3) need their own ISA tables.
Configure CMake with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` so the hook compiles with the real flags.

## Rebuilding the instruction table

`radisa/data/rdna4_isa.json` is generated, never edited by hand:

```bash
python3 -m radisa isadb --xml amdgpu_isa_rdna4.xml --manual rdna4-instruction-set-architecture.json
```

The XML is AMD's machine-readable ISA specification (MIT licence; release 2026-02-20, schema 1.1.1,
SHA-256 recorded in the table). The optional manual JSON is a text dump of the RDNA4 ISA Reference
Guide (2025-04-07); only section titles and PDF page numbers are taken from it, no text. Neither
input file is stored in this repository.
