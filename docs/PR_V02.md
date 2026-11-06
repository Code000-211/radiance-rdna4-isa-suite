# feat(rdna4): add binary ISA gate, scoped kernel coverage and opt-in AQL capture

## Summary

Replace the v0.1 regex-oriented prototype with a CPU-only binary evidence
pipeline. Independently inventory AMDHSA kernels, decode all submitted function
bytes, validate documented raw GFX12 VOPD constraints, and report exact static
findings separately from unproved semantic/runtime domains.

Kernel identity is code-object SHA-256 plus descriptor symbol. Collection and
decode failures, empty scans, missing records, stale results, unsupported paths
and unresolved review findings do not pass silently.

## Changes

- Bounded ELF64/LE and AMDHSA MessagePack readers; kernel descriptor cross-checks.
- Extraction from raw GPU objects, host `.hip_fatbin`, uncompressed clang bundles
  and regular archives; explicit incomplete status for unsupported containers.
- LLVM library decoder in a timed child process, with an objdump alternative.
- Independent raw VOPD field/constraint checks including equal-source exceptions
  and FMAMK/MOV-specific port routing.
- Direct CFG and selected release/WMMA/PERMLANE checks; clearly bounded local
  memory and SGPR diagnostics; no blanket spill/waterfall/NT/deallocation rules.
- Correct Wave32/Wave64 physical VGPR profile and resource upper-bound reporting.
- Hash-pinned inventory comparison, JSON/Markdown/JUnit reports and negative tests.
- Optional AQL capture patch binding resolved symbols to capture-time object hashes.

## Review basis

See `docs/REVIEW.md`, `docs/RULE_COVERAGE.md`, `docs/SOURCES.md` and `INPUTS.json`.
The supplied engineering rules were not treated as authoritative when they
conflicted with or overstated the actual AMD ISA guide.

## Validation included in the archive

See `evidence/RELEASE_VALIDATION.json` and `evidence/unit-tests.txt` for the actual
run, tools and results. Tests include real LLVM-generated/linked gfx1201 ELF,
real LLVM CPU disassembly, a deliberately invalid VOPD mutation, host ELF fatbin
extraction, missing/corrupt artifact cases, coverage identity collisions and a
compiled C++ capture harness. No invalid fixture is executed on a GPU.

The optional patch was checked with `git apply --check` against the uploaded
Radiance source. The entire Radiance HIP/HSA executable was **not** compiled or
run in the authoring environment. Actual production kernels remain to be scanned.

## Deliberate exclusions

This PR does not fix the W4A8 source tail, rewrite QSA, tune occupancy, insert
manual waits, change cache hints, modify P2P synchronization or claim a GPU wedge
root cause. Those changes require separate evidence and review.

`STATIC_CHECKS_PASS` is not complete ISA, memory-safety, numerical or wedge
certification. Indirect/helper execution paths and unmodeled semantics are
explicitly documented. Default strict review blocks reported candidates.

## Merge checklist

- [ ] Review the standalone tooling and explicit coverage boundaries.
- [ ] Run `bash tools/rdna4-isa-suite/check.sh` with required LLVM/C++ tools.
- [ ] Record actual ROCm/LLVM/driver/build flags for the target build.
- [ ] Collect final linked artifacts; review the expected object/kernel inventory.
- [ ] Audit and resolve every error/incomplete/review finding before using the gate.
- [ ] Preserve existing numerical/selftests and performance baselines independently.
- [ ] Keep AQL capture in a separate opt-in commit until the full engine rebuild,
      disabled-path smoke test and enabled workload test pass on the target host.
- [ ] Reconcile every relevant rank/process and workload configuration explicitly.
- [ ] Do not describe a CPU-only test pass as a production GPU stability proof.
