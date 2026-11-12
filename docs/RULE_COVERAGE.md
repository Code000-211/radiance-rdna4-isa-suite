# Rule and evidence coverage

Normative ISA reference: supplied AMD RDNA4 guide, 7 April 2025. Page numbers below
are **printed pages**; physical PDF page = printed page + 10.

## Implemented coverage

| Area | Implemented check | Evidence/status boundary |
|---|---|---|
| Container discovery | Standalone AMDGPU ELF, host `.hip_fatbin`, uncompressed concatenated clang bundles, regular ar | Compression, thin archives, opaque LLVM-offloading/LTO and unknown target encodings are explicit incomplete evidence |
| Kernel inventory | AMDGPU type-32 MessagePack note; unique `.name`/`.symbol`; compare defined `.kd` objects | Missing/stripped or ambiguous function extents block exact byte coverage; other undocumented kernel entry mechanisms are not assumed absent |
| Descriptor | 64-byte descriptor; entry offset/address alignment; wave and LDS/scratch consistency; encoded VGPR capacity | Not every reserved descriptor bit or launch field is validated |
| Byte coverage | All bytes in every metadata-named function extent must decode; raw object/code hashes | Not proof of all possible source instantiations; indirect/helper call graphs are not certified |
| Padding, §2.4 p.12 | 256 subsequent bytes backed by an executable PT_LOAD | Conservative evidence requirement, not a demand that every padding DWORD is S_CODE_END |
| Resources, §3.3.2 p.17 | gfx1201 physical granularity/upper bound, distinct Wave32/Wave64 capacities | LDS, workgroup placement, barriers and actual achieved occupancy are not modeled |
| VOPD, §§7.8,15.3.7 pp.82–84,188–190 | Independent binary fields, opcode sets, literal length, Wave32, forbidden source/DPP encodings, source banks/port exceptions, SRC2 parity, X→Y read-after-write, scalar bus and descriptor register range | Other instruction encodings use LLVM decoding, not an independent full-chip legality checker |
| Direct CFG, §5.5 pp.47–48 | Signed PC-relative branches, decoded boundaries, reachable S_CODE_END, recognized termination. S_SETPC_B64/S_RFE_B64 never fall through; a long branch in LLVM's BranchRelaxation shape (`s_getpc_b64`, `s_add_co_u32`, `s_add_co_ci_u32`, `s_setpc_b64`) gets its target as successor; `s_trap 2` (llvm.trap) and a call that ends the code (taken as noreturn) have no successor | Calls, other indirect branches and returns (CFG_INDIRECT) and out-of-function transfers require more evidence and block the gate; whether a final call's callee really does not return is checked only by `radisa check` (NORETURN_CALL_RETURNS, same object only) |
| Deallocation, §§3.3.2,5.4 pp.17,47 | Reachable vector/memory use after release; nontrivial epilogues need review | Absence of release is allowed; dynamic-mode launch/forward progress unproved |
| WMMA, §7.12.1 pp.90–91 | LLVM `GCNHazardRecognizer::fixWMMAHazards` over the whole direct CFG: D of a WMMA/SWMMAC overlapping A/B (or the SWMMAC index) of a later one on some path with no VALU in between (any VALU, V_NOP included, expires it; C overlap is allowed) | Paths through calls and unresolved indirect branches are not followed; the hardware consequence is not documented by AMD. Observed on one R9700 (hwprobe `wmma_d_to_a`, 2026-11-12): without the V_NOP every launch gave 120.0 instead of 128.0 |
| PERMLANE, §7.2.8 p.75 | LLVM `fixVcmpxPermlaneHazards` over the whole direct CFG: V_PERMLANE* after an EXEC-writing compare on some path with no VALU other than V_NOP in between (S_NOP, V_NOP and SALU do not clear it) | Only the hazards LLVM fixes for gfx1201 are modelled, not a full scheduling model. Not observed on the R9700 (hwprobe `cmpx_permlane`, 0 of 200 launches), which shows nothing either way |
| Memory counters, §5.7 pp.51–56 | Audit gate: supported local result-use/WAW candidates (LOCAL_MEMORY_HAZARD); zero waits for LOAD/DS/KM; FLAT needs both LOAD and DS | The complete counter model (nonzero waits, every counter, CFG joins, loops, calls and callee entry state) is `radisa check` (`MISSING_WAIT`, docs/CHECK.md), not this gate; here diagnostic only, strict-review blocks reported candidates |
| LDS at a workgroup barrier | `radisa check` `LDS_STORE_BEFORE_BARRIER` (error): an LDS write may still be outstanding (DScnt) at S_BARRIER_SIGNAL/S_BARRIER_WAIT, over the whole CFG | The one error rule whose basis is a hardware observation rather than LLVM: LLVM emits the S_WAIT_DSCNT only for a cross-address-space release fence (`SIGfx12CacheControl::insertWait`), not for a one-as fence or a bare barrier, but hwprobe `lds_store_barrier` on one R9700 (2026-11-12) read stale LDS in 137 and 152 of 200 launches without it and in none with it. Code that does not read another wave's LDS data after the barrier needs an inline `radisa: ignore[LDS_STORE_BEFORE_BARRIER] reason` |
| SGPR pair, §5.8 | Audit gate: narrow documented VALU-read→SALU-write→read pattern (SGPR_PAIR_GUARD) | The full rule (LLVM `AMDGPUWaitSGPRHazards`, every SGPR pair and VCC over the CFG) is `radisa check` `MISSING_ALU_WAIT`, not this gate |
| VGPR release s_nop | LLVM `SIInsertWaitcnts` / `GCNSubtarget::requiresNopBeforeDeallocVGPRs`: S_SENDMSG MSG_DEALLOC_VGPRS not directly preceded by S_NOP (DEALLOC_NO_NOP, review) | What the hardware does without the S_NOP is not documented |
| Hard clauses | LLVM `SIInsertHardClauses`: S_CLAUSE longer than 32, over mixed clause types or non-clause instructions, or entered by a branch (CLAUSE_TOO_LONG/MIXED/BRANCH, review) | The S_CLAUSE bits above [5:0] are not interpreted |
| Waterfall, §7.11 pp.88–89 | Relative access + readfirstlane + EXEC manipulation in an actual EXEC backedge | Heuristic performance candidate only; no zero-waterfall correctness rule |
| Cache, §4.1.1 pp.39–42 | Instruction text retained; explicit LU spelling triggers lifetime review | No whole-lifetime proof and no universal NT/LU requirement |
| AQL runtime resolution | Optional captured object hash + resolved kernel name reconciled with inventory | Resolution is not dispatch, process completion, or exhaustive workload/configuration/JIT coverage |

Register-range findings are classified as review: the ISA defines fallback/
nullification behavior for out-of-range register references (§3.3.2.2). They are
not automatically illegal-instruction faults. The current explicit descriptor-
range operand check of this gate is implemented for raw VOPD, not all instruction
families. `radisa check` checks every instruction family (`VGPR_OUT_OF_RANGE`, an
error there; see docs/CHECK.md for its basis).

## Non-claims

`STATIC_CHECKS_PASS` means the **implemented** checks passed over the submitted
inventory, not that every row in a conceptual ISA rulebook has been proved.
`full_isa_certification` is always false. The report records explicit unproved
domains per kernel:

- Pointer bounds and object lifetimes, including W4A8 tail memory accesses.
- P2P coherence, release/acquire protocol correctness, inter-wave/workgroup races.
- Kernel numerical output, K permutation, quantization accuracy and sparse selectors.
- Barrier convergence, general termination, scheduler forward progress, hardware
  stability, driver/firmware errors and GPU reset/wedge behavior.

A memory diagnostic deliberately uses `review`, not `error`: optimized code can
use valid nonzero waits or other mechanisms outside this local subset. Do not
"fix" a review finding by adding waits until the actual dependency is understood.

## Conservative limitations that can stop real builds

A real Radiance build may currently return `INCOMPLETE` for legitimate indirect
chain calls, helper functions, relocatable intermediates or optimizations beyond
the modeled scheduling subset. That is preferable to an unsupported success
claim, but it is not proof that the kernel is incorrect. Provide the relevant
code object and dependency/call evidence, extend the checker with tests, or keep
that analysis outside this gate with an explicitly narrower scope. This release
does not offer an allowlist that turns missing evidence into certification.

Every discovered metadata kernel is attempted. Coverage of every theoretical
kernel specialization or every possible runtime kernel cannot be inferred from a
source ZIP or from one test workload.
