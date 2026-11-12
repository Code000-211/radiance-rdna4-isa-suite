# Operating and interpreting the gate

## Safe order on the R9700 host

1. Run the CPU suite selftests. No GPU load is required.
2. Audit a copy of final linked artifacts with the same target/compiler flags as
   the intended run. Inspect inventory/byte coverage before interpreting findings.
3. Investigate `FAIL` findings in the exact object/PC, with disassembly and source.
   An ISA error is not automatically the cause of a previously observed wedge.
4. Investigate `INCOMPLETE` separately: missing evidence or unsupported analysis
   does not prove defective kernel code. Do not disable the check just to get green.
5. Only then apply/rebuild the optional capture integration and exercise a normal,
   controlled workload. Preserve logs, process exit codes and all rank captures.
6. Run the existing numerical and GPU stress tests as independent gates. This
   suite does not supply arbitrary GPU launch arguments for all Radiance kernels.

No script here resets a GPU, modifies firmware, changes clock/voltage limits,
disables watchdogs, or launches negative ISA fixtures. Do not turn off recovery
mechanisms to force a stress test to continue.

## Interpreting common findings

- `VOPD_BANK` / `VOPD_SRC2` / `VOPD_RAW_DPMACC`: inspect the independently
  decoded raw operands and exact object hash. Do not assume the source compiler
  emitted the same binary under a different toolchain or set of flags.
- `VOPD_SOURCE_HI` (`review`): a component sources `vcc_hi` or `exec_hi`. The Wave32
  VALU source table does not list the high halves, but gfx1201 reads and writes them:
  `tools/vcc_hi_probe.hip` returns 3+7 = 10 both from a plain `v_add_nc_u32` and from the
  Y slot of a real VOPD pair, and `s_mov_b32 exec_hi, 7` reads back as 7 with no effect on
  lane masking. Kept visible as a table deviation, not raised as a hardware failure.
- `VOPD_ANTIDEP` (`info`): `vdstX` is a source of `opY` in the encoded word,
  which ISA §7.8 forbids. LLVM's `gcn-create-vopd` produces this on purpose when
  a program-order WAR pair is swapped into the X/Y slots, and
  `SIInstrInfo::isVOPDAntidependencyAllowed` permits exactly that for every
  non-DPMACC opX. Program order is not visible in the object, so the finding
  cannot decide whether the pair was legal before the swap; it stays `review`.
  Only a DPMACC opX (`v_dot2acc_f32_f16`, `v_dot2acc_f32_bf16`) is an error,
  because that is the one case the compiler itself rejects.
- `LOCAL_MEMORY_HAZARD`: a local dependency lacks a proof within the supported
  scanner subset. Valid nonzero waits and optimized scheduling need manual
  analysis; this is intentionally `review`, not an automatic compiler-bug claim.
- `CFG_INDIRECT`: an indirect/call path needs callee/target evidence. Byte decode
  may be 100% while semantic control-flow coverage remains incomplete.
- `WATERFALL_CANDIDATE`: a structural loop pattern, not a hang or correctness bug.
  Measure the actual hot path before rewriting it.
- `OBJECT_SET_MISMATCH`: do not hand-delete evidence or reuse a directory. Collect
  into a fresh directory and reconcile the approved inventory.
- `LAST_USE_LIFETIME`: LU dirty-discard assumptions need application-level lifetime
  evidence. Do not insert or remove hints mechanically.

## Evidence to archive with a PR

Keep the collection manifest, checked object hashes, reports, disassembly,
compiler version/build flags, source revision, and the reviewed expected inventory.
For GPU validation, also keep the GPU/driver/firmware versions, actual workload
and shapes, numerical reference comparisons, process exit codes and host logs.
The source ZIP and this CPU suite alone do not provide those GPU results.

## Limits and input trust

The readers deliberately support ELF64/LE AMDHSA v3–v6 (ELF ABI bytes 1–4),
metadata major version 1, and the documented `gfx1201` target profile. Compressed
bundles, thin archives, opaque LTO/offloading containers and stripped/ambiguous
function extents stop complete coverage. Limits include 1 GiB per input container,
100,000 table entries, metadata nesting limits and a decoder timeout. Large valid
builds may need explicitly reviewed limit changes.

LLVM decoding is isolated in a timed process, but this is not a hardened sandbox
for hostile native decoder exploits. Use trusted compiler tools and a normal
unprivileged account. Capture requires a trusted, user-owned directory. Hashes
provide consistency, not attestation against someone able to rewrite the code,
manifest and baseline together.

The snapshot capture rejects fork after initialization. Multi-process workloads
need fresh worker processes and reconciliation of every PID log. There is no
log-completion seal and no claim that a successful resolution means execution.
