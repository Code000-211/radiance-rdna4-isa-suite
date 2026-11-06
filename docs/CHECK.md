# `radisa check`: lint one gfx1201 kernel

`radisa check` takes **one** HIP translation unit, assembly file or code object and checks every
gfx1201 kernel in it against rules derived from AMD's machine-readable RDNA4 ISA (XML) and the
RDNA4 ISA Reference Guide. Findings point at source lines and cite the manual section and PDF page.

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
The assembly is then assembled with the `llvm-mc` next to the compiler, so the checked bytes are
exactly the compiler's. `-gline-tables-only` only adds line tables (`-g` crashes clang 20.1.2 on some
libr4d units).

## What is checked

| rule | severity | basis |
|---|---|---|
| `MISSING_WAIT` | error | §5.7.1 counters, PDF pp.62-65: a register is read or overwritten while a load/LDS/SMEM/sample/BVH/export that writes it may still be outstanding on some CFG path |
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

Measured on this tree: the fixtures under `fixtures/check/`, plus libr4d units compiled with
clang 20 and ROCm 6.4 HIP headers (see the release notes of the change for counts). On
compiler-scheduled code `MISSING_WAIT` has had no false positive so far; that is evidence, not a proof.

## Not checked

Numerical correctness; inter-wave or cross-device memory ordering (P2P release/acquire, drain
modes); global memory safety; the timing of hazards the compiler resolves with `s_delay_alu` /
`s_wait_alu`; callee bodies; indirect branches; performance beyond the listed resource notes.

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
