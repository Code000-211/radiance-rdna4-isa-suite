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
| Direct CFG, §5.5 pp.47–48 | Signed PC-relative branches, decoded boundaries, reachable S_CODE_END, recognized termination | Calls, indirect branches or out-of-function transfers require more evidence and block the gate |
| Deallocation, §§3.3.2,5.4 pp.17,47 | Reachable vector/memory use after release; nontrivial epilogues need review | Absence of release is allowed; dynamic-mode launch/forward progress unproved |
| WMMA, §7.12.1 pp.90–91 | Explicit D register overlap into next WMMA A/B/index within a basic block; V_NOP/known independent VALU separation | No whole-CFG scoreboard proof; unknown/dependent VALU separators are counted as unmodeled, not proved safe |
| PERMLANE, §7.2.8 p.75 | Immediate CMPX→PERMLANE pattern | No full scheduling/hazard recognizer |
| Memory counters, §5.7 pp.51–56 | Supported local result-use/WAW candidates; zero waits for LOAD/DS/KM; FLAT needs both LOAD and DS | Nonzero waits, atomics, wider semantics, CFG joins and initial state need a complete counter model; diagnostic only, strict-review blocks reported candidates |
| SGPR pair, §5.7 p.52 | Narrow documented VALU-read→SALU-write→read pattern | Not all VCC/SGPR dependencies; does not enforce blanket waits |
| Waterfall, §7.11 pp.88–89 | Relative access + readfirstlane + EXEC manipulation in an actual EXEC backedge | Heuristic performance candidate only; no zero-waterfall correctness rule |
| Cache, §4.1.1 pp.39–42 | Instruction text retained; explicit LU spelling triggers lifetime review | No whole-lifetime proof and no universal NT/LU requirement |
| AQL runtime resolution | Optional captured object hash + resolved kernel name reconciled with inventory | Resolution is not dispatch, process completion, or exhaustive workload/configuration/JIT coverage |

Register-range findings are classified as review: the ISA defines fallback/
nullification behavior for out-of-range register references (§3.3.2.2). They are
not automatically illegal-instruction faults. The current explicit descriptor-
range operand check is implemented for raw VOPD, not all instruction families.

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
