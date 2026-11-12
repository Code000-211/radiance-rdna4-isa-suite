# `radisa check`: lint gfx1201 kernels

`radisa check` takes **one** HIP translation unit, assembly file or code object and checks every
gfx1201 kernel in it, and every non-kernel function of the same code object, against rules derived
from AMD's machine-readable RDNA4 ISA (XML), the RDNA4 ISA Reference Guide and, where the guide is
silent, LLVM's AMDGPU backend. Findings point at source lines and cite the manual section and PDF
page or the LLVM function they follow. `radisa check-all` runs the same check over many files.

Severity policy: a rule is an `error` only where the hazard or wait is documented by the ISA guide
(the memory counters, VGPR range, VOPD encoding) or where LLVM's AMDGPU backend itself fixes or
enforces it for gfx1201 (the LLVM function is cited). One exception, stated as such:
`LDS_STORE_BEFORE_BARRIER` is an `error` because the hardware needs the wait, although LLVM does
not insert it for a one-address-space fence (see its row and "Measured on the target"). Everything
else is a `warning` (`review` in the JSON) or a `note` (`info`). Hardware probes on one R9700
showed the `MISSING_WAIT`, `WMMA_D_TO_AB` and `LDS_STORE_BEFORE_BARRIER` hazards; no other rule
has hardware evidence (see "Measured on the target" and "Not validated").

```bash
S=tools/rdna4-isa-suite
# HIP source: compiled for gfx1201 (device only), then checked as machine code
bash $S/check-kernel.sh libr4d/r4d_attn_paged_h128_gqa4.hip --compile-commands build/compile_commands.json
# Hand-written assembly; text without .amdhsa_kernel and without functions is one wave32 snippet
bash $S/check-kernel.sh my_snippet.s
# Machine code: ELF code object, offload bundle, fat binary, .hip.o, archive, shared library
bash $S/check-kernel.sh build/libr4d/CMakeFiles/r4d.dir/r4d_gemm_fp8a8.hip.o --kernel gemm_8a8_prefill
# Many files, one report (SARIF for code scanning)
bash $S/check-tree.sh --compile-commands build/compile_commands.json --jobs 8 --format sarif --out radisa.sarif
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

A finding without a source line is printed as `object:kernel+0xOFF`, the offset inside the kernel
or function (not the address in the object). `--json FILE` writes the full report (every kernel
and function, resources, operand-model coverage, findings, suppressed and baselined findings).
`--format text|json|sarif|github` chooses what goes to stdout. `radisa rules` lists every rule
with its severity and summary; `radisa rules --explain RULE` prints its basis.

## Status and exit codes

| exit | status | meaning |
|---|---|---|
| 0 | `PASS` | no error-level finding (`--strict`: no warning either) |
| 0 | `NO_KERNELS` | the input holds no gfx1201 kernel and no function |
| 1 | `FAIL` | at least one error (a compile error and an assembler error on hand-written assembly are errors) |
| 2 | | usage or configuration error: bad option, malformed `.radisa.json`, missing `--baseline` file, invalid `--liveness-entry`; nothing is checked |
| 3 | `INCOMPLETE` | something was not analysed (below) |

`FAIL` outranks `INCOMPLETE`: a file with an error and unanalysed code is `FAIL`.

`INCOMPLETE` means code exists that was not checked. It is set when there is no kernel or a
kernel without instructions, when bytes were not decoded (`DECODE`, `BYTE_COVERAGE`), or when one
of these is reported at severity `incomplete`: `CHECK_FAILED`, `ASSEMBLER` (llvm-mc rejected
compiler output), `SOURCE_MAP` (HIP or `.loc` input that cannot be mapped to lines),
`OBJECT_EVIDENCE` (a function without a symbol size, a descriptor without metadata), `CFG_TARGET`
and `CFG_ENCODING` (paths behind a branch whose target is not followed), `VOPD_IDENTITY` and
`WMMA_OPERANDS` (the VOPD or WMMA rules could not run on that instruction). Under `check-all`
also `CHECK_CRASHED`, `CHECK_TIMEOUT` and `SKIPPED_TOO_MANY_KERNELS`.

Not in that set, on purpose (`check.INCOMPLETE_RULES`): `KERNEL_EVIDENCE` (a kernel without bytes
already has no instructions; the remaining message is about prefetch padding), `CFG_FALLTHROUGH`
at `incomplete` (unreached code at the end, which was analysed), `DYNAMIC_VGPR_MODE` (every
instruction is analysed; only the VGPR range check is off; marking it would make every
dynamic-VGPR kernel incomplete with no way to fix it), and `CFG_INDIRECT` (calls; see the `CALLS`
note).

## Compiling HIP

The compiler is, in order: `--cxx`, `$RADISA_HIPCC`, `$ROCM_PATH/llvm/bin/clang++`, `hipcc`,
`clang++`. With `--compile-commands`, the flags CMake used for that file are reused (output, depfile
and `-c` flags are dropped) and `-x hip --offload-arch=gfx1201 --offload-device-only -O3
-gline-tables-only -S` is added; project `-O`/`-m` flags still win. Extra flags: `--flag "-I inc -DX=1"`.
`-gline-tables-only` only adds line tables (`-g` crashes clang 20.1.2 on some libr4d units).
Without ROCm device libraries (as in the test environment) pass `--flag="-nogpulib -nogpuinc"`.
With `--compile-commands` the compiler runs in that entry's `directory`, so `--keep DIR` is made
absolute first. Intermediate files go to a `radisa-check-*` directory under `$TMPDIR` unless
`--keep` is given; it is removed on every way out, including SIGTERM and SIGHUP (exit 128 + the
signal). `check-all` stops a worker (timeout, Ctrl-C, SIGTERM) with SIGTERM to its process group and
SIGKILL only 5 s later, and points each worker's `TMPDIR` into its own temporary directory, which it
removes even after a SIGKILL.

The compiler's `.s` is assembled again with llvm-mc. clang's `.s` names only the target, not the
wave size, and llvm-mc takes the wave size from its command line (the `.amdhsa_wavefront_size32 0`
directive is only checked against it, AMDGPUAsmParser `ParseDirectiveAMDHSAKernel`). So a unit
whose kernel descriptor or metadata says wave64 (`.amdhsa_wavefront_size32 0`, `.wavefront_size:
64`) is assembled with `-mattr=+wavefrontsize64`; a unit without a kernel uses the last
`-m[no-]wavefrontsize64` of the compile flags (`hipsrc.assembler_wave_args`). A module that mixes
wave32 and wave64 functions cannot be assembled in one run and is not supported.

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
   file; a kernel whose finding line was itself edited is compiled again. The cache key includes the
   enabled rules and the analysis options; the cache schema is `radisa-incremental-v4` (older
   entries are recompiled).

Per-kernel `llc` gives the same machine code as compiling the unit: 241 of 241 kernels in seven
libr4d units, and identical findings from `--no-incremental` and the default mode on `attn_paged_h256_gqa12`,
`ltx` and `ar_oneshot_2rank_exact`. The `opt` + `clang` path was compared the same way with Radiance's
LLVM 24 compiler and the flags of its `compile_commands.json`: code bytes and kernel descriptor are
identical for 4475 of 4475 kernels in 117 libr4d units, and for 32 of 32 kernels sampled from the
2069 of `r4d_gemm_iqr`. These comparisons predate the rules added since (they compare code, not
findings). When non-kernel functions survive optimisation, module-level code generation (the LDS
layout across kernels) could differ; then the whole module goes through `llc` and nothing is
cached. `--no-incremental` restores the single `clang -S` compile.

## Assembly input and line mapping

* **Compiler output with `.loc`**: findings map to the HIP lines named there (`hipsrc.parse_asm`).
* **Hand-written `.s`**: findings map to the line of the instruction in the `.s` file, column 1
  (`check.asm_line_map`). Comments (`;`, `//`, `/* */` across lines) are blanked, labels are
  skipped wherever they stand (also before an instruction on the same line), and a new function
  starts only at the label of a symbol declared `.type x,@function` or `.amdhsa_kernel x`, so
  `loop:` does not split a function. `.macro` bodies are skipped; macro invocations, `.rept` and
  `.irp` do not line up with the object, and `.include` files are not followed: then a
  `SOURCE_MAP` note (severity `info`, the analysis itself is complete) says that findings carry
  pcs only and line-scoped suppressions cannot match them.
* **Forms.** A `.s` with `.amdhsa_kernel`, or one that assembles to sized `STT_FUNC` symbols in an
  executable section (`functions.has_functions`), is checked as a code object: its kernels as
  kernels, its functions as called functions. Anything else is a snippet: one wave32 sequence
  (`--wave 64` for wave64) with no kernel descriptor, where `CFG_FALLTHROUGH` at the end is only a
  note.
* llvm-mc errors on hand-written assembly are `ASSEMBLER` errors (exit 1).

## Rules

Every rule `radisa rules` lists, except the `radisa audit` ones. Severity is the default; a
configuration can change it (see "Configuration"). `*` = off unless enabled.

### Memory counters and waits

| rule | severity | basis |
|---|---|---|
| `MISSING_WAIT` | error | RDNA4 ISA §5.7.1 counters, PDF pp.62-65: a register is read or overwritten while a load/LDS/SMEM/sample/BVH/export that writes it may still be outstanding on some CFG path |
| `LDS_STORE_BEFORE_BARRIER` | error | **Hardware-observed; the exception to the severity policy.** An LDS write may still be outstanding (DScnt) at `s_barrier_signal` / `s_barrier_wait`. Basis: hwprobe `lds_store_barrier` on a Radeon AI PRO R9700, 2026-11-12: `ds_store_b32`, then the barrier with no `s_wait_dscnt`, then each wave reads another wave's word gave wrong results in 137 and 152 of 200 launches; with `s_wait_dscnt 0x0` before the barrier, in none. LLVM does not enforce this wait: it emits it only for a release fence that orders LDS together with global memory (`SIMemoryLegalizer.cpp` `SIGfx12CacheControl::insertWait`, `DSCnt \|= IsCrossAddrSpaceOrdering` at workgroup scope, LLVM AMDGPU User Guide memory model GFX12), and `SIInsertWaitcnts` adds none at `S_BARRIER` (FeatureBackOffBarrier). So `__syncthreads()` gets it, but a one-address-space fence (`fence(release, "workgroup-one-as")`), an OpenCL local fence or a bare `__builtin_amdgcn_s_barrier()` does not, and that compiler output is reported (`fixtures/check/barrier_one_as.hip`, `barrier_no_fence.hip`). Code that really does not read another wave's LDS data after the barrier says so with an inline `radisa: ignore[LDS_STORE_BEFORE_BARRIER] reason` |
| `CALLEE_RETURN_OUTSTANDING` | warning | LLVM `SIInsertWaitcnts.cpp` (waits before a return); §5.7.1: a called function returns while memory operations other than stores may still be outstanding |
| `SYS_STORE_WAIT` | warning* | LLVM `SIMemoryLegalizer.cpp` `SIGfx12CacheControl::finalizeStore` / `insertWaitsBeforeSystemScopeStore`, FeatureWaitsBeforeSystemScopeStores: a `scope:SCOPE_SYS` store while loads/stores may be outstanding. Opt-in (`--enable SYS_STORE_WAIT`) and unconfirmed: compiler output does not meet the literal reading (below). A message lists at most 8 producers per counter and a count; `pending` lists all |
| `HIP_INLINE_ASM_MEMORY` | warning | lexical; §5.7.1: an asm statement (or a macro expanding to one) with a memory instruction and no wait in it. A note when only some definitions of a conditionally defined macro do so |
| `HIP_LEGACY_WAITCNT` | warning | lexical; §16.5: `__builtin_amdgcn_s_waitcnt` (S_WAITCNT = S_WAIT_IDLE on RDNA4) |

### SGPR commit hazard (`s_wait_alu`)

| rule | severity | basis |
|---|---|---|
| `MISSING_ALU_WAIT` | error | LLVM `AMDGPUWaitSGPRHazards.cpp`: an SGPR that vector instructions read is read before its last write is committed, with no `s_wait_alu` in between. A warning (`operand_model: assumed`) when it rests on the guessed operands of an instruction outside the operand table |
| `ALU_WAIT_TTMP` | warning | the same hazard at a place where the compiler itself emits no wait (ttmp read in between, see below) |
| `SGPR_PAIR_GUARD` | warning | §5.8: the narrow VALU-read / SALU-write / read SGPR-pair pattern without `s_wait_alu` (also in `radisa audit`) |

### ALU hazards LLVM fixes, whole CFG (`radisa/hazards.py`)

| rule | severity | basis |
|---|---|---|
| `WMMA_D_TO_AB` | error | `GCNHazardRecognizer.cpp` `fixWMMAHazards`; §7.12.1: a WMMA/SWMMAC whose A or B (on GFX12 also the SWMMAC index) overlaps the D of an earlier WMMA/SWMMAC on some path with no VALU in between (V_NOP counts; LLVM inserts `V_NOP_e32`). Overlap with C is allowed |
| `PERMLANE_AFTER_CMPX` | error | `GCNHazardRecognizer.cpp` `fixVcmpxPermlaneHazards` (FeatureVcmpxPermlaneHazard in FeatureISAVersion12); §7.2.8: a `v_permlane*` after a compare that writes EXEC, on some path with no VALU other than `v_nop` in between (`s_nop`, `v_nop` and SALU do not clear it). The message names the producers (first 6, then a count; `producers` lists all) and LLVM's fix, `v_mov_b32 vN, vN` with vN the permlane's src0 |
| `DEALLOC_NO_NOP` | warning | `SIInsertWaitcnts.cpp`, `GCNSubtarget.h` `requiresNopBeforeDeallocVGPRs` (true for gfx1201): `s_sendmsg MSG_DEALLOC_VGPRS` not directly preceded by `s_nop`. Neither file says what the hardware does without it |
| `CLAUSE_TOO_LONG` | warning | `SIInsertHardClauses.cpp`, FeatureMaxHardClauseLength32: an `s_clause` longer than the 32 instructions LLVM forms for gfx1201 |
| `CLAUSE_MIXED` | warning | `SIInsertHardClauses.cpp` `getHardClauseType`: an `s_clause` over memory instructions of different clause types, or over non-clause instructions |
| `CLAUSE_BRANCH` | warning | `SIInsertHardClauses.cpp` (clauses are formed per basic block): a branch targets the inside of an `s_clause` body |

### Control flow and calls

| rule | severity | basis |
|---|---|---|
| `CFG_FALLTHROUGH` | error / warning / note | §5.5: a path runs past the last instruction of a kernel without `s_endpgm` (error); past the end of a called function without a return (warning); a note when the last instruction is not reached by direct branches, and for a snippet |
| `NORETURN_CALL_RETURNS` | warning | AMDGPU calling convention (return address `s[30:31]`); §5.5: a kernel or function ends with a call, which is taken as a call to a noreturn function, but the callee is a function of the same object, checked in this run, with a reachable return. If it returns, execution runs past the end of the caller |
| `REACHABLE_CODE_END` | error | §2.4 shader padding: reachable `s_code_end` |
| `CFG_INDIRECT` | incomplete (not reported) | §5.5: a call or an unresolved indirect branch; folded into the `CALLS` note |
| `CFG_TARGET`, `CFG_ENCODING` | incomplete | §5.5: a branch target that is not an instruction boundary of the kernel; an unexpected branch size |
| `CALLS` | note | `SIInsertWaitcnts.cpp` callee prologue: the kernel calls functions or branches indirectly; says which functions of the object are checked on their own, which are not, and which final call is taken as noreturn |

### Registers and resources

| rule | severity | basis |
|---|---|---|
| `VGPR_OUT_OF_RANGE` | error | §3.3.2: an instruction names a VGPR above what the kernel descriptor allocates; the hardware does not trap, the instruction works on a register the kernel does not own. Not checked with dynamic VGPRs (`s_alloc_vgpr`) or for called functions. Replaces the VOPD-only `VOPD_REGISTER_RANGE` warning for the same kernel |
| `UNINITIALIZED_READ` | warning / note | LLVM AMDGPU User Guide "Initial Kernel Execution State"; `SIISelLowering.cpp`: a register is read before anything defines it. Warning when no path defines it, note when only some paths do (below) |
| `ASM_UNDECLARED_CLOBBER` | warning | LLVM LangRef inline assembler constraints; clang AMDGPU `GCCRegNames`: an inline asm statement writes a register that is neither an output nor a clobber (below) |
| `ASM_CLOBBER_SKIPPED` | note | the device IR for `ASM_UNDECLARED_CLOBBER` could not be produced (`--no-incremental`) |
| `OPERAND_MODEL_UNKNOWN` | warning | AMD RDNA4 ISA XML; `radisa/isadb.py`: instructions outside the operand table, assumed to read and write every register they name |
| `REGISTER_SPILL` | warning | code-object metadata: VGPR/SGPR spills to scratch |
| `SCRATCH_USE` | warning / note | code-object metadata: scratch without spills (a private array not kept in registers). A note in a kernel that makes calls: the private segment size includes the callees' stack (LLVM `AMDGPUMCResourceInfo.cpp` `gatherResourceInfo`, `AMDGPUResourceUsageAnalysis.cpp`), and the metadata does not split the two |
| `USE_AFTER_DEALLOC` | error | §3.3.3: a vector or memory operation reachable after `MSG_DEALLOC_VGPRS` |
| `DEALLOC_EPILOGUE`, `DEALLOC_DYNAMIC` | warning | §3.3.3: a nontrivial epilogue after the VGPR release; `s_alloc_vgpr` and full deallocation in one kernel |
| `DYNAMIC_WAVE64` | error | §3.3.3: `s_alloc_vgpr` in wave64 |
| `DYNAMIC_VGPR_MODE` | incomplete (does not change the status) | §3.3.3: dynamic VGPR mode; the VGPR range check is off for the kernel |
| `LAST_USE_LIFETIME` | warning | §4.1.1: a last-use (TH_LU) access needs a whole-buffer lifetime proof |
| `WATERFALL_CANDIDATE` | note | §7.11: an EXEC loop around relative VGPR indexing |

### VOPD (raw encoding, §7.8, §16.11)

| rule | severity | basis |
|---|---|---|
| `VOPD_OPCODE`, `VOPD_SOURCE`, `VOPD_BANK`, `VOPD_SRC2`, `VOPD_SCALAR_BUS`, `VOPD_DPP`, `VOPD_WAVE32` | error | reserved opcode or source encoding, SRC0/SRC1 bank conflict, SRC2 parity, more than two scalar/literal sources, DPP source, VOPD in wave64 |
| `VOPD_RAW_DPMACC` | error | §7.8; LLVM `SIInstrInfo::isVOPDAntidependencyAllowed`: X (DPMACC) writes a register Y reads |
| `VOPD_ANTIDEP` | warning | the same for non-DPMACC X, which LLVM emits |
| `VOPD_SOURCE_HI`, `VOPD_REGISTER_RANGE` | warning | VCC_HI/EXEC_HI sourced by a wave32 component; a VGPR outside the descriptor allocation (§3.3.2) |
| `VOPD_IDENTITY` | incomplete | LLVM's VOPD text does not match the raw format decoder |

`WMMA_OPERANDS` (incomplete): the WMMA operands could not be read, so its rules did not run.

### HIP source, toolchain, evidence, hygiene

| rule | severity | basis |
|---|---|---|
| `HIP_WAVE64_WIDTH` | warning | lexical: shuffle width 64 on a wave32 target |
| `HIP_NO_LAUNCH_BOUNDS` | note | lexical: kernel without `__launch_bounds__` |
| `COMPILE` | error | the gfx1201 device compile failed |
| `ASSEMBLER` | error / incomplete | llvm-mc rejected the input: error for hand-written assembly, incomplete for compiler output |
| `COMPILER_DIAGNOSTICS`, `CACHE`, `NO_KERNELS` | note | compiler warnings; the incremental cache was not written; no gfx1201 kernel |
| `DECODE`, `BYTE_COVERAGE`, `SOURCE_MAP`, `KERNEL_EVIDENCE`, `OBJECT_EVIDENCE`, `CHECK_FAILED` | incomplete | undecoded bytes, unmapped lines, descriptor/metadata or object problems, the input could not be read (`SOURCE_MAP` and `OBJECT_EVIDENCE` can also be notes, see above) |
| `CHECK_CRASHED`, `CHECK_TIMEOUT`, `SKIPPED_TOO_MANY_KERNELS` | incomplete | `check-all` only |
| `SUPPRESSION_NO_REASON`, `UNUSED_SUPPRESSION`, `CONFIG_UNKNOWN_RULE` | warning | suppression hygiene (below) |
| `BASELINE_STALE` | note | baseline entries of this file that no longer occur |

## Control-flow model (`rules.cfg`)

Successors of each instruction (RDNA4 ISA §5.5; where the ISA leaves a choice, what LLVM emits):

* `s_endpgm`, `s_code_end`: none. `s_trap 2` (llvm.trap): none; the AMDGPUUsage Trap Handler ABI
  says the wave is halted and its queue put into the error state. Any other `s_trap` falls
  through (`s_trap 3`, llvm.debugtrap, continues when no debugger is attached).
* `s_branch`, `s_cbranch_{scc,vcc,exec}*`: the SIMM16 target; the conditional ones also fall
  through.
* `s_setpc_b64`, `s_rfe_b64`: never fall through. LLVM emits `s_setpc_b64` for a return
  (`s[30:31]`), a tail call or a long branch, always at the end of its block. A **long branch** in
  the exact shape LLVM's BranchRelaxation emits (`SIInstrInfo::insertIndirectBranch`:
  `s_getpc_b64 s[N:N+1]`, `s_add_co_u32 sN, sN, lit`, `s_add_co_ci_u32 sN+1, sN+1, lit`,
  `s_setpc_b64 s[N:N+1]`, optionally with `s_wait_alu`, `s_waitcnt_depctr`, `s_delay_alu` or
  `s_nop` in between) has its target as its only successor (`rules.long_branch_target`); a target
  outside the code is `CFG_TARGET`. Any other `s_setpc_b64` has no successor (`CFG_INDIRECT`).
* `s_swappc_b64`, `s_call_b64`: the callee returns to the next instruction. A call that is the
  last instruction of the code, or followed only by `s_code_end` padding, has no successor and is
  taken as a call to a **noreturn** function: LLVM ends the block of such a call with the call
  itself, and a returning call is always followed by more code. Such calls are listed in the
  `CALLS` note (`noreturn_call_pcs`); `NORETURN_CALL_RETURNS` reports the ones whose callee is
  known to return. A noreturn call in the middle of a function keeps its fall-through edge.
* Anything else: the next instruction; past the last one, `CFG_FALLTHROUGH`.

The two literals of a long branch or a call are read with `isadb.immediate`, the one parser for an
immediate printed in disassembly. It takes LLVM 18-23's `0x20` / `32` / `-16` and LLVM 24's
`lit(0x20)` / `lit64(0x...)`: ROCm `d6f6cb69` prints a literal whose value would also fit an
inline constant that way (`AMDGPUDisassembler::decodeLiteralConstant` wraps it in
`AMDGPUMCExpr::createLit`, printed in hex). Every other immediate (waits, `s_clause`, branch
offsets, `s_sendmsg`, `s_trap`, VOPD) is read from the encoding, not the text. Baseline and SARIF
fingerprints fold `lit(X)` to the LLVM 18-23 spelling of an integer operand
(`isadb.canonical_literals`), so a baseline written with one disassembler matches the other; a
literal of a floating-point operand (LLVM 18-23 prints e.g. `1.0`) still differs.

How the passes treat a resolved long branch:

* Hazards, counters, liveness: a jump to its target, like `s_branch`. `UNINITIALIZED_READ` passes
  the register state on to the target (an unresolved `s_setpc_b64` still defines every register).
* SGPR hazard (`sgpr.py`): a scalar read of its address pair followed by a jump. LLVM builds the
  sequence after its hazard passes and flushes only the SGPR writes it adds itself with
  `s_wait_alu depctr_sa_sdst(0)`; a wait still pending from code before the branch is reported at
  the reader in the target block, not at the `s_setpc_b64`.

`NORETURN_CALL_RETURNS` finds the callee only from the `s_getpc_b64` / `s_add` / `s_swappc_b64`
sequence: through the relocation in a relocatable object, through the literals in a linked one
(`functions.call_target`). Calls through a loaded pointer, `s_call_b64`, and callees in another
object are never reported. It is a warning: LLVM enforces nothing here; compiler output has this
shape only for a callee the IR calls noreturn, and a reachable return there is undefined behaviour
at the source level that the machine CFG shows as a possible path.

## Cross-block hazard model (`radisa/hazards.py`)

LLVM's `GCNHazardRecognizer::getWaitStatesSince` walks backwards from the consumer through every
predecessor block until it finds the producer or an instruction that expires the hazard. The
check is the forward equivalent: a may-dataflow over the CFG above (union over predecessors,
producer sets killed by the expiring instructions), so `WMMA_D_TO_AB` and `PERMLANE_AFTER_CMPX`
are found across branches and loop back edges, not only between adjacent instructions.

* WMMA: expires at any VALU, `v_nop` included (`SIInstrInfo::isVALU(I, AllowLDSDMA=true)`).
* PERMLANE: expires at a VALU that is not `v_nop`.
* `fixWMMAHazards`, `fixVcmpxPermlaneHazards` and their helpers are identical in upstream main and
  ROCm `d6f6cb69`; the meta-instruction skip in the static `getWaitStatesSince` exists upstream
  only and changes nothing here (a meta instruction is never a WMMA, VOPC or VALU).
* A WMMA consumer reached by several overlapping producers gives one error per producer.
* Clauses: gfx1201 gets per-kind clause types in both trees (upstream gates the mixed load/store
  branch on FeatureMixedLoadStoreClauses, ROCm on `GFX10 || GFX1250`; gfx1201 is neither). Only
  `s_nop` may appear inside a clause; clauses never cross a basic block. The `s_clause` bits above
  [5:0] are not interpreted.

## The counter model (`radisa/waits.py`)

* Every instruction is mapped onto its AMD XML operand list (`radisa/isadb.py`): explicit and
  implicit reads/writes, including EXEC, VCC, SCC and M0. EXEC is added for every VALU/VMEM/export
  instruction (the XML omits it). An instruction the table cannot align counts as *unknown*; its
  registers are still treated as both read and written, never as harmless. The summary line reports
  the coverage. Operand details:
  * `v[7]`, `s[ 7 ]` and `ttmp[3]` are single registers, as llvm-mc reads them.
  * A buffer/image (MUBUF/MIMG) atomic has one VGPR operand for data and result; the XML lists it
    once, as VDST. It is always read (the data) and written only with `th:TH_ATOMIC_RETURN`.
    FLAT/GLOBAL/SCRATCH atomics have a separate data operand and no destination without a return.
  * A returning buffer `cmpswap` is modelled as writing its whole data tuple, though only the low
    half returns (conservative: it can give a wait on the high half that is not needed).
* Counters come from the encoding: SMEM → KMcnt; VDS → DScnt; VGLOBAL/VSCRATCH/VBUFFER/VIMAGE →
  LOADcnt (loads, returning atomics, `global_inv`) or STOREcnt; VFLAT → that plus DScnt; VSAMPLE →
  SAMPLEcnt; BVH → BVHcnt; export and LDS param/direct loads → EXPcnt.
* LOADcnt, SAMPLEcnt, BVHcnt and DScnt return in issue order, so `s_wait_<cnt> N` retires all but the
  N youngest. KMcnt and EXPcnt are out of order: only 0 retires anything. Combined waits decode
  SIMM16[13:8] and SIMM16[5:0] (§16.5).
* The analysis is a forward may-analysis over the direct-branch CFG (loops included). A later write
  to the same register by the same in-order counter needs no wait. An instruction with a single
  predecessor takes that predecessor's state without a join; the entry counts as one more
  predecessor of the first instruction, so a loop back to the entry still joins with the entry
  state. Cost is linear in the instruction count (about 7 s for 47,500 instructions with every
  pass, measured on synthetic kernels).
* A call (`s_swappc_b64` / `s_call_b64`) waits only for scalar loads that write its operands; after
  it every counter but STOREcnt is drained, because LLVM's callee waits at its entry and before it
  returns (`SIInsertWaitcnts`).
* An 8- or 16-bit store reads one half of its data register (`…_d16_hi_…` the high half, the
  others the low half). A D16 load of the other half may stay outstanding across it, as in LLVM
  `SIInsertWaitcnts`; every other reader, and a store whose data register is also its address,
  needs both halves.
* At a workgroup barrier only LDS writes are looked at (`LDS_STORE_BEFORE_BARRIER`, an error). An
  LDS or global load may be outstanding there (the compiler leaves unused loads outstanding), and so
  may a global store: a barrier with fences scoped to LDS, like Radiance's `lds_barrier()`, orders
  local memory only.

## Called functions (`radisa/functions.py`)

Every sized `STT_FUNC` symbol in an executable section of the code object is checked as a called
function, with the rules adapted to what LLVM does for a non-kernel function
(`SIInsertWaitcnts`, `AMDGPUWaitSGPRHazards`):

* At entry the caller's memory operations may be outstanding on every counter. They write
  registers the function cannot know: any VGPR it names on the vector counters (LOADcnt, DScnt,
  SAMPLEcnt, BVHcnt, EXPcnt), any SGPR up to VCC except `s[30:31]` on KMcnt. A missing entry wait
  gives one `MISSING_WAIT` per register file.
* Every SGPR pair counts as read by vector instructions at entry ("Callee must consider all SGPRs
  as tracked").
* It ends with `s_setpc_b64 s[30:31]`; a return must leave every counter but STOREcnt drained
  (`CALLEE_RETURN_OUTSTANDING`). A path past the end without a return is `CFG_FALLTHROUGH`
  (warning).
* It has no kernel descriptor: `VGPR_OUT_OF_RANGE` and `UNINITIALIZED_READ` do not apply.
* Its wave size is the one every kernel of the object declares. An object without kernels has no
  record of it: wave32, or `--wave` for a `.s` file.

An object that declares no kernel (`amdhsa.kernels: []`, e.g. a `-fgpu-rdc` unit after `llc` or a
device library) and a `.s` file with functions and no metadata are checked all the same: an
`OBJECT_EVIDENCE` note says the object declares no kernel, and its functions are checked as called
functions. Without functions such an input is `NO_KERNELS`. A function without a symbol size is not
checked: `OBJECT_EVIDENCE` (incomplete) and the `CALLS` note name it.

## The SGPR read hazard (`radisa/sgpr.py`)

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
  units, both in `r4d_gemm_mxfp4a8_gemv`) and is not confirmed on hardware (the `alu_wait_ttmp`
  probe showed no mismatch on the R9700, which shows nothing either way).
* A `MISSING_ALU_WAIT` that exists only because of the assumed operands of an instruction outside
  the operand table (as producer, consumer, or the earlier VALU read that makes the pair tracked)
  is re-graded to a warning: LLVM computes these waits from an instruction's uses only.
* Upstream LLVM before 20 has no `AMDGPUWaitSGPRHazards`; clang 18 output therefore gives
  `MISSING_ALU_WAIT` errors that LLVM 20+ fixes (`tests/test_compiler_clean.py` tolerates them for
  LLVM < 20 only).

## Uninitialised reads (`radisa/liveness.py`)

`UNINITIALIZED_READ` runs two forward problems over the CFG: registers defined on every path
(must) and on some path (may).

* **Warning** (`undefined_on: "every path"`): a read of a register no path from the entry writes.
* **Note** (`undefined_on: "some paths"`): a read of a register written on some paths but not all.
  Machine code cannot tell a bug from LLVM's phi with an undef input (a value set and read under
  the same condition, a loop-carried value, the lane-mask phis of `SILowerI1Copies.cpp`); clang 18
  -O2/-O3 even computes speculatively with such values. A write missing on one branch of a uniform
  if is therefore only a note.
* Defined at kernel entry (AMDGPU User Guide "Initial Kernel Execution State",
  `SIISelLowering.cpp`): the user SGPRs the descriptor enables, the enabled system SGPRs, TTMP0-15,
  `v0` only (packed work-item IDs, FeaturePackedTID), EXEC. Not defined: other SGPRs, VCC, M0,
  SCC, other VGPRs.
* Of the two successors of `s_cbranch_execz`/`execnz`, the one taken only when EXEC is zero carries
  no VGPR state, except when that edge leaves a loop (the layout range between a backward branch and
  its target).
* Exemptions: a multi-register data operand of which at least one register is defined (not for
  memory addresses and descriptors, nor F64 sources); `x^x`, `x-x` and self-moves (not with DPP,
  SDWA or op_sel); the implicit M0 of `s_sendmsg`. `s_cmov*` reads its destination (the kept value).
  A write under partial EXEC counts as a full definition: lane masks are not tracked.
* A call or unresolved indirect jump defines every register afterwards.
* One finding per register and severity, at the first read; `later_uses` lists up to 32 further
  reads and `later_use_count` counts them all.
* Not run for called functions, nor for snippets unless `--liveness-entry "s[0:1],v0"` lists the
  registers defined at the snippet's entry (each token must be a register or range, otherwise a
  usage error). `--no-liveness` turns it off.
* Known gaps: loops are taken from the layout (MachineBlockPlacement may move cold blocks out of a
  loop); the exemptions were derived from clang 18 output only.

## Inline-asm clobbers (`radisa/asmclobber.py`)

`ASM_UNDECLARED_CLOBBER` reads the optimised device IR (the module of the incremental path, or a
second clang run with `--no-incremental`), replaces each operand placeholder of an `asm` call by a
reserved register, and asks the operand table what every template line writes, implicit writes
included (VCC, SCC, EXEC of V_CMPX, M0). A written register that is neither an output nor a
clobber is reported (LLVM LangRef: only registers named as outputs or clobbers may change).

* One finding per source statement, asm line, registers and severity; further IR copies of the
  same statement (unrolled or inlined) only increment `copies`.
* An SGPR (EXEC, M0, VCC, sN) that ends the statement holding its entry value (`s_mov` copies,
  `s_*_saveexec` save slots) is not reported; SCC is.
* VOPD halves are modelled separately; numeric and `${:uid}` labels in templates are skipped.
* Clobber names: the single registers clang accepts (`GCCRegNames`) and ranges such as
  `~{v[4:7]}`, which LLVM's constraint parser accepts in IR.
* `HIP_INLINE_ASM_MEMORY` (lexical) ignores asm labels on declarations (`__asm__("name")`),
  directives, string literals and `#if 0` code; other `#if` conditions are not evaluated.

## Configuration and suppression (`radisa/suppress.py`)

All of this runs on the finished report and only changes which findings are shown and at which
severity; no analysis depends on it. Findings that say the analysis did not happen (severity
`incomplete`, and `DECODE`, `BYTE_COVERAGE`, `CHECK_FAILED`, `ASSEMBLER`, `SOURCE_MAP`,
`COMPILE`, `NO_KERNELS`) are never re-graded, suppressed or baselined.

**Configuration**: `--config FILE`, default the nearest `.radisa.json` in the input's directory or
above (`--no-config` skips it):

```json
{"severity": {"MISSING_WAIT": "error", "SCRATCH_USE": "off"}, "enable": ["SYS_STORE_WAIT"],
 "disable": ["HIP_NO_LAUNCH_BOUNDS"], "baseline": "radisa-baseline.json"}
```

Levels are `error`, `warning`, `note`, `off`. A relative baseline is relative to the config file.
`--severity RULE=LEVEL`, `--enable`, `--disable` and `--baseline` override the file. An unknown
rule name is `CONFIG_UNKNOWN_RULE`; a malformed file or a missing baseline file is exit 2. A
severity override applies to every finding of the rule (both levels of `UNINITIALIZED_READ`).

**Inline**, in a comment (`//`, `/* */` in HIP/C++; `;`, `//`, `/* */` in assembly):

```
// radisa: ignore[MISSING_WAIT,MISSING_ALU_WAIT] the asm block above waits for it
; radisa: ignore-file[HIP_NO_LAUNCH_BOUNDS] launch bounds come from the host wrapper
```

* A trailing `ignore[...]` (code before it on the line) covers that line only. On a comment-only
  line it covers its own line and the next line with code, over further comment-only lines but
  not over a blank line. A `/* */` directive may continue its reason on the next lines of the same
  block comment.
* `ignore-file[...]` covers the whole file, and in the checked input also findings without a
  source line (`REGISTER_SPILL`, ...).
* The reason after `]` is mandatory: without it nothing is suppressed and
  `SUPPRESSION_NO_REASON` is reported.
* A listed rule that suppressed nothing is `UNUSED_SUPPRESSION`, only for directives in the
  checked input file (a header directive may serve another translation unit) and not with
  `--kernel`. When the file has findings of that rule without a source line, the message says a
  line-scoped ignore cannot match them.
* Line-scoped suppression needs a source line: in `.hip` files from the line tables, in `.s` files
  from the line mapping above.

**Baseline**: `--write-baseline FILE` records the current findings (after configuration and inline
suppression); `--baseline FILE` hides the recorded ones and reports only new ones. The format is
`radisa-baseline-v1`: each entry has a fingerprint (sha256 over rule, kernel name, normalised
instruction and producer text, normalised source line text for lexical findings, occurrence index;
no pc, address or line number) and `input`, the input file relative to the baseline file's
directory. One file can hold any number of inputs:

* `--baseline` applies only the entries of the checked file, so one file's baseline never hides
  findings in another, also with a repository-wide `"baseline"` in `.radisa.json`. Entries of this
  file that no longer occur are `BASELINE_STALE` (not with `--kernel`). An older baseline without
  per-entry `input` belongs to its top-level `input`.
* `--write-baseline` replaces this input's entries (with `--kernel`, those of the checked kernels)
  and keeps every other file's. The file is rewritten atomically under a lock on its directory
  (POSIX `flock`; no lock on Windows). It refuses to overwrite a file that is not a baseline.

**Summary**: suppressed findings move to `report['suppressed']`, baselined ones to
`report['baselined']`. The summary line counts them ("3 suppressed inline", "2 in baseline",
"1 disabled by configuration", "4 re-graded by configuration") and names every error-severity
finding that is not shown as an error, whatever hid it: "(hidden errors: MISSING_WAIT x3)".

## Many files: `radisa check-all` (`check-tree.sh`)

```bash
bash $S/check-tree.sh --compile-commands build/compile_commands.json --changed-since origin/main --jobs 8
bash $S/check-tree.sh fixtures/check some/file.s --include '\.(hip|s)$' --format github
bash $S/check-tree.sh --each FILE...      # one check-kernel.sh run per file (the pre-commit hook)
```

Every selected file goes through exactly what `radisa check FILE` does, in its own process and
process group; every `check` option is accepted (its `--jobs` as `--check-jobs`, `--json` replaced
by `--out`). Files come from the positional paths (files always, directories filtered by
`--include`, default `\.hip$`) and from `--compile-commands`; `--exclude` drops files;
`--changed-since GITREF` keeps files changed relative to GITREF or untracked (any file name).
A file that crashes the checker, kills its process or exceeds `--timeout-per-file` is an
INCOMPLETE entry (`CHECK_CRASHED`, `CHECK_TIMEOUT`); `--max-kernels-per-file N` skips units with
more kernels (counting only `--kernel` matches) as `SKIPPED_TOO_MANY_KERNELS`.

| exit | meaning |
|---|---|
| 0 | no file FAILs or is INCOMPLETE |
| 1 | a file FAILs (`--fail-on warning`: warnings fail too; `--fail-on incomplete`: warnings and INCOMPLETE files too) |
| 2 | invalid option, configuration or missing `--baseline`, checked once before any file |
| 3 | no file FAILs, but one is INCOMPLETE |
| 130 / 143 | SIGINT / SIGTERM: every running check (its whole process group) is stopped |

Baselines: `--write-baseline B` writes one file for every checked file, each entry with its
`input` relative to B's directory; entries of files not checked in this run are kept.
`--baseline B` (or a configuration's baseline) gives each file only its own entries.

Text output puts a `==> INPUT: STATUS <==` header before each file when there is more than one.

### Formats (`radisa/formats.py`)

* `json`: the reports.
* `sarif`: SARIF 2.1.0. Each finding is a result with the rule, level, location (artifact URI
  relative to the current directory when the file lies under it, else a `file://` URI; region
  line/column) or a logical location `kernel+0xOFF`, and a
  `partialFingerprints` entry `radisaFinding/v2`. The fingerprint covers rule, base-relative input
  path, kernel, instruction text and the message without line numbers or pcs, so it survives line
  moves. (It was `radisaFinding/v1` without the path; existing code-scanning alerts get new
  fingerprints once.) Suppressed and baselined findings are results too, with a `suppressions`
  entry: kind `inSource` with the reason and the comment's location, or `external` for the
  baseline, status `accepted`.
* `github`: one `::error` / `::warning` / `::notice` workflow command per finding; `file=` is a
  path (relative to the base when the file is under it), never a URI. Per file with hidden
  findings, one `::notice title=radisa suppressions` repeats the summary suffix, so hidden errors
  stay visible in the job log.

The repository ships `.github/workflows/radisa.yml` and `.pre-commit-hooks.yaml` (hook id
`radisa-check`, which runs `check-tree.sh --each` on staged `.hip`/`.s` files; exit 1 and 3 block
the commit).

## As an edit hook

`radisa hook` reads a PostToolUse event (JSON) on stdin and checks the edited file when it matches
`$RADISA_HOOK_INCLUDE` (default `\.(hip|s)$`). `RADISA_COMPILE_COMMANDS`, `RADISA_FLAGS` and
`RADISA_HIPCC` are passed as `--compile-commands`, `--flag` and `--cxx`.

| `radisa check` result | hook |
|---|---|
| exit 1 (errors) | exit 2, report on stderr, so the editing agent fixes the errors before it continues |
| exit 2 (bad `.radisa.json`, missing baseline) | exit 2: a project configuration error blocks |
| exit 3 (INCOMPLETE) | exit 0, report on stderr |
| the environment values do not form a valid command line | exit 0, note on stderr, file not checked |
| PASS, or a path that does not match | exit 0 |

In the Radiance checkout, the hook settings file `.claude/settings.json`:

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

## LLVM source drift: `radisa drift`

Several rules restate decisions of LLVM's AMDGPU backend for gfx1201. `radisa/data/llvm_pins.json`
lists 21 source facts they depend on (feature sets, the gfx1201 processor definition, subtarget
predicates, `fixWMMAHazards`, `fixVcmpxPermlaneHazards` and their helpers `isPermlane` and
`isVCmpXWritesExec`, `AMDGPUWaitSGPRHazards.cpp`, parts of `SIInsertWaitcnts.cpp`,
`SIInsertHardClauses.cpp` and `SIMemoryLegalizer.cpp`), each with a file, an anchor, an extractor
(`file`, `cxx_function`, `td_def`, `statement`) and the sha256 of the normalised text.

```bash
python3 -m radisa drift --llvm-src /path/to/llvm-project          # compare
python3 -m radisa drift --llvm-src /path/to/llvm-project --update \
  --revision 'ROCm/llvm-project <commit>' --note 'what was reviewed'
```

* Exit 0: no fact changed (`PASS`); 1: a fact changed (`DRIFT`), with the rules to re-review;
  3: a file or anchor is missing, ambiguous or unbalanced.
* Normalisation drops comments and whitespace, keeps C++ multi-character operators and
  preprocessor directives as single tokens, so reformatting is not drift; any token change is. A
  `cxx_function` extract starts at its declaration (template header, specifiers, return type);
  a `statement` anchor is parsed as a full C++ statement (if/else chain, loops, `do ... while`).
* `alt_files`: further files searched when `file` lacks the anchor; it must be found in exactly
  one. The two feature facts use `AMDGPU.td` (ROCm `d6f6cb69`, where `FeatureISAVersion12` and
  `FeatureGFX12` are defined) with `GCNFeatures.td` (upstream main) as the alternative.
* `--update` rewrites the hashes and the whole provenance (revision, time, note: `--note` or a
  generated one listing the facts that changed). It refuses while an anchor is missing:
  re-anchor such a fact by editing its file, `alt_files` or anchor in the pin file.

**Provenance.** The pins are taken from ROCm/llvm-project `d6f6cb691863e9258034ab48c1dfce21689d5606`
(AMD clang 24.0.0git), the compiler Radiance targets, from single files of
`llvm/lib/Target/AMDGPU` at that commit: 21 of 21 facts unchanged there. Against upstream
llvm-project main (fetched 2026-11-12) 9 facts differ by design, each reviewed with no rule change
for gfx1201:

| fact | difference | effect on gfx1201 |
|---|---|---|
| `features.isa_version_12`, `features.gfx12_generation` | upstream-only features (LDS granularity, 32BitNumRecords, HalfAddressablePhysicalLocalMemory, GetDoorbellID, MaxWavesPerEU16, cond-sub atomics); file moved to GCNFeatures.td | none of them is read by a rule |
| `processor.gfx1201` | ROCm adds `let ArchFeatures` | none |
| `hazards.fixHazards` | upstream-only `fixVPermPk16Hazard` | gated by a feature only in FeatureISAVersion12_50_Common |
| `waitsgpr.whole_file`, `waitcnt.determineVGPR16Dependency` | `isVALU` AllowLDSDMA, option plumbing, casts | no gfx1201 instruction is LDS DMA (llvm-mc rejects both lds forms for gfx1201) |
| `waitcnt.dealloc_vgprs` | max waves per EU source | 16 in both for gfx10.3+ |
| `clauses.whole_file` | mixed-clause gate (`hasMixedLoadStoreClauses` vs `GFX10 \|\| GFX1250`), debug-instruction handling, cluster-size budget | gfx1201 gets per-kind clauses in both; the budget only shortens clauses |
| `memorder.gfx12_insertWaitsBeforeSystemScopeStore` | upstream guards the BVHCNT wait with `hasAnyBVHInsts()` | true for gfx1201 (FeatureBVHRayTracingInsts): same five waits |

When Radiance moves to another ROCm revision: run `radisa drift --llvm-src <that llvm-project>`;
for every CHANGED fact re-review the rules it names against the new source; then `--update
--revision '<revision>' --note '<what was reviewed>'` and commit the pin file. The static
`getWaitStatesSince` and `SIInstrInfo::isLDSDMA` are not pinned: both have two overloads, which the
`cxx_function` extractor rejects as ambiguous.

## Hardware probes

`hwprobe/` holds probe kernels for the hazard rules (`MISSING_WAIT`, `LDS_STORE_BEFORE_BARRIER`,
`WMMA_D_TO_AB`, `PERMLANE_AFTER_CMPX`, `MISSING_ALU_WAIT`, `ALU_WAIT_TTMP`, the D16 relaxation) and a
runner for the R9700. They ran once on one R9700 (2026-11-12); the results are under "Measured on
the target" and in `hwprobe/README.md`.

## Measured on the target (2026-11-12)

Run of commit `1fe68fb` (the per-rule counts were identical on `0b3712d`) with AMD clang
24.0.0git (ROCm llvm-project `d6f6cb69`) on one machine with one Radeon AI PRO R9700.

**Unit tests**, `RAD_ISA_REQUIRE_LLVM=1` (nothing skipped for a missing toolchain capability): 580
run, 7 failures and 1 error. The causes, all fixed in commit `7448fd5`, which was re-run on the target
(see "Re-run of `7448fd5`" below):

* LLVM 24 prints a literal whose value would fit an inline constant as `lit(0x20)` (the
  disassembler wraps it in `AMDGPUMCExpr` `lit(`/`lit64(`), and the long-branch and call-target
  resolution read it with `int(x, 0)`, which fails. 5 tests: `test_hazards`
  `FixtureTests.test_failing_fixtures` (`hazard_longbranch.s` gave `PASS` instead of the
  `PERMLANE_AFTER_CMPX` error), `FixtureTests.test_clean_fixtures`,
  `FixtureTests.test_long_branch_finding_is_at_the_target`, `CompiledTests.test_long_branches_resolved`,
  and `test_callee` `NoreturnCall.test_object_and_linked_object` (`NORETURN_CALL_RETURNS` missing on
  the linked object). Fixed by one shared parser, `isadb.immediate`.
* `test_check` `AssemblyCheckTests.test_vgpr_out_of_range`: its mutant reads v7, which nothing
  writes, so `UNINITIALIZED_READ` rightly fires; the test expected no finding at all. The test was
  wrong (it never ran with LLVM 18, which cannot assemble `kernel_reduce.s`).
* `test_compiler_clean` `CompilerClean.test_clean_fixtures_wave64`: `clean_compiled.hip` calls a
  wave32-only WMMA builtin that clang 24 rejects with `-mwavefrontsize64`. Such fixtures now carry a
  `radisa-test: wave32-only (reason)` comment and are left out of the wave64 run.

**Corpus**: the 114 libr4d units including `r4d_gemm_iqr`, with the flags of
`compile_commands.json`: 6448 kernels and 2 callee functions (`r4d_sam_attn_bf16`), 11.79 M
instructions, none outside the AMD operand model. 112 `PASS`, 2 `NO_KERNELS` (`ar_entry`,
`gemm_bf16_nt_m16`), 0 `FAIL`, 0 `INCOMPLETE`. 47.5 min wall time with 4 units in parallel;
`gemm_iqr` alone took 37 min at about 0.8 GB.

| findings | count |
|---|---|
| any rule at `error` | 0 for every rule. `LDS_STORE_BEFORE_BARRIER` had no finding either (a warning on `1fe68fb`, an error on `0b3712d`, as it is again now) |
| `VOPD_ANTIDEP` (warning) | 3065 |
| `REGISTER_SPILL` (warning) | 548 |
| `SCRATCH_USE` (warning) | 76 |
| `UNINITIALIZED_READ` (warning) | 46: 36 are `v_readfirstlane_b32 sN, v1` in `attn_paged`, the rest dead tuple copies in `gdn_kkt_solve` and `hc_bf16`; in none is the value used in arithmetic |
| `ALU_WAIT_TTMP` (warning) | 2 (`gemm_mxfp4a8_gemv`, the `ttmp9` pattern) |
| `VOPD_SOURCE_HI` (warning) | 2 |
| no finding at all | `WMMA_D_TO_AB`, `PERMLANE_AFTER_CMPX`, `DEALLOC_*`, `CLAUSE_*`, `ASM_UNDECLARED_CLOBBER`, `CALLEE_RETURN_OUTSTANDING`, `MISSING_WAIT`, `CFG_*`, `OPERAND_MODEL_UNKNOWN`, `BYTE_COVERAGE`, `DECODE`, `NORETURN_CALL_RETURNS` |

**Caveat: long branches were not resolved in this run** (the `lit(` bug above). Every long branch
in the corpus was taken as an indirect jump with no successor, so blocks reached only through a
long branch were not analysed at all; their findings, if any, are missing from these counts. The
zero for `CFG_*` does not cover them either.

**`SYS_STORE_WAIT`** (opt-in; run on `0b3712d`, 113 units without `gemm_iqr`): 1307 warnings in 10
units, largest `ar_twoshot_Nrank_exact` 548, `rmsnorm_quant_i8` 468, `fused_quant_fp8` 138,
`ar_ln_had_quant_i8` 76.

**Drift** against the ROCm `d6f6cb69` source: 21 of 21 pinned facts `PASS`.

**Hardware probes** (`hwprobe/`, 200 launches per variant, grid 512, GPU 0; `host_runner.cpp`
built unmodified against the real HIP runtime). Every `_fixed` variant: 0 mismatches.

| case | `_hazard` | reading |
|---|---|---|
| `ds_load_wait`, `global_load_wait` | 200 of 200 launches, all 26,214,400 words | `MISSING_WAIT` hazard observed |
| `wmma_d_to_a` | 200 of 200 launches, all 26,214,400 words (120.0 instead of 128.0) | `WMMA_D_TO_AB` hazard observed |
| `lds_store_barrier` | 137 and 152 of 200 launches (8,576 / 8,960 words) | `LDS_STORE_BEFORE_BARRIER` hazard observed: the basis of its `error` |
| `d16_other_half` | 200 of 200 launches; `_fixed` clean | hazard observed; no evidence against the D16 half-store relaxation |
| `cmpx_permlane`, `alu_wait_carry`, `alu_wait_ttmp` | not observed | proves nothing |


### Re-run of `7448fd5`

Same machine, compiler and GPU, same day, on commit `7448fd5` (the fixes for the causes above).

* **Unit tests**, `RAD_ISA_REQUIRE_LLVM=1`: 594 run, OK, 1 skipped (the pin test without
  `RADISA_LLVM_SRC`); with `RADISA_LLVM_SRC` set to the ROCm `d6f6cb69` files: 594 OK, none skipped.
  `RADIANCE_SOURCE=<radiance checkout> bash check.sh`: `tests` 594 OK and `replay_tests` 142 OK.
  Without `RADIANCE_SOURCE`, 124 of the 142 replay tests error with "cannot read RAD_ABI_VERSION".
* **Fixtures**: `hazard_longbranch.s` FAIL with `PERMLANE_AFTER_CMPX`, `hazard_longbranch_clean.s`
  PASS, `barrier_one_as.hip` and `barrier_no_fence.hip` FAIL with `LDS_STORE_BEFORE_BARRIER`,
  `clean_compiled.hip` PASS.
* **Corpus**: 114 units, 6448 kernels, 2 callee functions, 11,788,006 instructions, none outside
  the operand model; 112 PASS, 2 NO_KERNELS, 0 FAIL, 0 INCOMPLETE; 39.3 min wall with 4 parallel
  units (`r4d_gemm_iqr` alone 32 min). Every per-rule count is identical to the `1fe68fb` run: no
  error finding for any rule, including `LDS_STORE_BEFORE_BARRIER` at error severity.
* **Long branches**: the corpus contains none. The largest kernel function is 98,532 bytes
  (`r4d_hc_bf16`), below the 128 KiB branch range; the five units whose kernels could exceed it by
  instruction count were disassembled completely and contain no `s_setpc_b64`. The only two
  `s_setpc_b64` in the corpus are the callee returns in `r4d_sam_attn_bf16`. The long-branch caveat
  on the `1fe68fb` run therefore did not hide anything in this corpus, and the long-branch
  resolution is exercised only by fixtures and unit tests, not by corpus code.
* **Work directories**: 10 `/tmp/radisa-check-*` before the run and 10 after; none left behind.
* **hwprobe** (200 launches per variant, grid 512, `--verify-lint` passed, every `_fixed` variant 0
  mismatches): `ds_load_wait`, `global_load_wait`, `wmma_d_to_a` and `d16_other_half` wrong in
  200/200 launches (`d16_other_half`: 1,087,857 words); `lds_store_barrier` in 143/200 (8,960
  words); `cmpx_permlane`, `alu_wait_carry`, `alu_wait_ttmp` not observed.

Not run on the target: the `--objdump` decode backend on corpus code, `SYS_STORE_WAIT` on this
commit, the second GPU.

## Measured earlier (before the cross-block rules)

These numbers were taken before the cross-block hazards, callee analysis, `UNINITIALIZED_READ`,
`ASM_UNDECLARED_CLOBBER`, the clause and dealloc rules and `NORETURN_CALL_RETURNS`. The corpus run
was repeated with those rules on 2026-11-12 ("Measured on the target"); the fault injection below
was not.

The libr4d units of Radiance, compiled with its own compiler (AMD clang 24, ROCm llvm-project
`d6f6cb69`, the flags of `compile_commands.json`), all but `r4d_gemm_iqr` (2069 kernels, left out
for its run time): 117 units, 4477 kernels, 8,438,289 instructions, 786,325 `s_wait_alu`.

| | before | then |
|---|---|---|
| `PASS` | 102 | 115 |
| `FAIL` | 13 (570 `MISSING_WAIT`, all false: D16 half loads) | 0 |
| `INCOMPLETE` | 2 (`BYTE_COVERAGE`) | 2 (the same) |
| error findings | 570 | 0 |

The 570 stores are now counted as `d16_other_half_stores` in the report. No error rule fired on
compiler output; that is evidence, not a proof. Warnings: 2 `ALU_WAIT_TTMP`, the rest as before.

Single faults put into the compiler's assembly of `r4d_attn_paged_h128_gqa4` (36 kernels, one fault
per run, the result is "at least one error"):

| fault | reported |
|---|---|
| a memory wait (`s_wait_loadcnt` …) deleted | 77 of 80 |
| its count raised by one | 76 of 80 |
| moved behind the next instruction | 55 of 60 |
| `s_wait_alu` deleted | 120 of 120 |
| WMMA reads the result of the previous WMMA | 30 of 30 |
| `s_endpgm` deleted | 5 of 6 |
| `s_wait_dscnt 0` before a barrier deleted (`LDS_STORE_BEFORE_BARRIER`, an error then as now) | 20 of 20 |

The `LDS_STORE_BEFORE_BARRIER` row was measured while the rule was an error; with it at warning
(commit `1fe68fb` only) these mutants gave `PASS`. It is an error again, so the row applies.

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
every wait and the `s_endpgm` removed in turn; each mutant has to fail with exactly the expected
rule (the LDS one with `LDS_STORE_BEFORE_BARRIER`). The test needs a newer llvm-mc than LLVM 18 and
is skipped here.

## Not checked

* Numerical correctness, global memory safety, performance beyond the listed resource notes.
* Waits that only a memory ordering asks for: before atomics and fences, around `global_inv` /
  `global_wb`, cross-device release/acquire. Which ordering and scope the source asked for is not in
  the machine code. The LDS write before a workgroup barrier is the exception: it is an error on
  hardware evidence (`LDS_STORE_BEFORE_BARRIER`).
* The waits LLVM puts in front of every non-atomic `scope:SCOPE_SYS` store on GFX12.0
  (`SIGfx12CacheControl::finalizeStore`), unless `SYS_STORE_WAIT` is enabled. Checked against the
  libr4d units this would be 701 errors in 8 of the 9 all-reduce units: the compiler emits
  `s_wait_loadcnt`/`s_wait_storecnt` in front of the stores but leaves `global_inv` and the store
  before it outstanding (it inserts the waits as "soft" waits, which the wait pass later merges
  and relaxes). What the hardware needs there is not documented by AMD. Re-measured on 2026-11-12
  with the opt-in rule: 1307 warnings in 10 units ("Measured on the target").
* An `s_endpgm` missing in the middle of a kernel (see above).
* Callees outside the checked object, calls through a loaded pointer, and indirect branch targets
  other than LLVM's long branch. A call is assumed to return with every counter but STOREcnt
  drained.
* Whether a noreturn call in the middle of a function really does not return (its fall-through
  edge is kept).
* Per-lane register state: lane masks are not tracked (`UNINITIALIZED_READ` takes a write under
  partial EXEC as a full definition).
* `s_delay_alu` (a performance hint; the hardware interlocks without it).
* SGPR range: since GFX10 every wave gets the full SGPR file (the descriptor field is ignored), and
  the assembler rejects names beyond it.
* `; wave barrier` (a barrier the compiler removed because the workgroup is one wave) is not an
  instruction: a missing wait there is harmless and not reported.
* Wave64 modules that mix wave sizes per function.

## Not validated in this change

The rules were developed and tested with Ubuntu LLVM/clang 18.1.3, without a GPU and without
ROCm. Validated since, on the target (commit `1fe68fb`, "Measured on the target"): the unit tests
with AMD clang 24 and `RAD_ISA_REQUIRE_LLVM=1`, the 114-unit libr4d corpus, `radisa drift`
against ROCm `d6f6cb69`, and the hardware probes on one R9700, with `host_runner.cpp` built against
the real HIP runtime. Not validated:

* **Long branches in real code.** The corpus contains no long branch (see the re-run of
  `7448fd5`), so the long-branch resolution has only been exercised by fixtures and unit tests,
  on LLVM 18 and on AMD clang 24. Its behaviour on a compiler-generated kernel larger than the
  128 KiB branch range is not measured.
* **The `--objdump` decode backend** on corpus code, and `SYS_STORE_WAIT` after `0b3712d`.
* **Hardware**: one GPU, three runs on one day. Not repeated on a second gfx1201 board.
  `PERMLANE_AFTER_CMPX`, `MISSING_ALU_WAIT` and `ALU_WAIT_TTMP` were not made visible by their
  probes; nothing is known about them from hardware.
* **LLVM 20-23**: neither the corpus nor the tests were run with those compilers. The long-branch
  shape and its `s_wait_alu` fillers come from upstream main `SIInstrInfo::insertIndirectBranch`
  and clang 18 output; that AMD clang 24's long branches resolve is untested (on the target they
  failed on the `lit(` spelling before the shape was compared). `lit(` printing was checked in the ROCm `d6f6cb69` source only
  (`AMDGPUMCExpr::printImpl`, `AMDGPUDisassembler::decodeLiteralConstant`); in which LLVM release
  it starts was not checked, and both spellings are accepted. The call sequence that
  `NORETURN_CALL_RETURNS` resolves comes from clang 18 output.
* The `LDS_STORE_BEFORE_BARRIER` error rests on one probe on one GPU. The one-address-space fence
  behaviour of LLVM (no wait) was confirmed with clang 18 output and the `SIMemoryLegalizer.cpp`
  source of both trees, not with AMD clang 24 output of `barrier_one_as.hip`.
* Wave64 HIP units: whether llvm-mc 24 needs `+wavefrontsize64` and whether that compiler's `.s`
  still emits `.amdhsa_wavefront_size32 0` / `.wavefront_size: 64` were taken from the
  AMDGPUAsmParser source, not run.
* Called functions: the per-counter register files at entry come from the AMD XML operand table
  (the ISA PDF text is not in the repository); the `s_sendmsg_rtn` result is treated as an SGPR on
  KMcnt; a wave64 `-fgpu-rdc` object was not checked.
* `UNINITIALIZED_READ` exemptions were derived from clang 18 -O0..-O3 output. On the AMD clang 24
  corpus 46 warnings remain, classified above (the value is never used in arithmetic); no
  exemption was added for them.
* `ASM_UNDECLARED_CLOBBER`: the `s_*_saveexec`/`s_mov` restore semantics are ISA readings, not
  tested; the asm-label heuristic is lexical; LLVM 20-24 IR (e.g. the two-line `callbr` form) was
  not checked.
* Baseline locking was tested on Linux only; Windows has no lock, and `check-all` on Windows was
  only simulated.
* `radisa drift` re-review of the 9 upstream/ROCm differences was done from source; the optional
  pins for `getWMMAHazardInstInCategory` and `isCoexecutableVALUInst` (called from
  `fixWMMAHazards`, apparently for GFX1250) were not added.
* `SYS_STORE_WAIT`: the `pending` field lists every outstanding producer, so the JSON can grow
  quadratically with thousands of system-scope stores (the rule is opt-in).

## Rebuilding the instruction table

`radisa/data/rdna4_isa.json` is generated, never edited by hand:

```bash
python3 -m radisa isadb --xml amdgpu_isa_rdna4.xml --manual rdna4-instruction-set-architecture.json
```

The XML is AMD's machine-readable ISA specification (MIT licence; release 2026-02-20, schema 1.1.1,
SHA-256 recorded in the table). The optional manual JSON is a text dump of the RDNA4 ISA Reference
Guide (2025-04-07); only section titles and PDF page numbers are taken from it, no text. Neither
input file is stored in this repository.
