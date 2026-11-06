# Review before stage 2 implementation

Date: 2026-10-27. Basis: the supplied v0.1 ZIP, both PR reports, the engineering
rules, Radiance source ZIP, conversation claims, and the supplied AMD RDNA4 ISA
Reference Guide dated 7 April 2025. Exact input hashes are in `INPUTS.json`.

This was a review of the supplied artifacts and the relevant implementation and
ISA sections, not a claim to have manually verified every kernel specialization
or every instruction definition in the guide. External LLVM documentation was
used only for ELF/code-object, tool and target-specific implementation details.

## Findings in v0.1

| Finding | Consequence | v0.2 response |
|---|---|---|
| Kernels keyed by name across objects | Different binaries with the same symbol overwrite each other | Identity includes object SHA-256 and descriptor symbol |
| Text metadata parser assumes a particular `.name` ordering | Real metadata starting with `.args` can be missed | Decode the AMDHSA MessagePack note, independent of key order |
| Demangled template names contain `>` | Regex symbol boundaries can fail | Use ELF symbols and function extents, not label-name regex |
| Host `.hip.o` treated like a direct device object | Embedded code objects are not comprehensively inspected | Parse host ELF `.hip_fatbin`, raw bundles and regular archives |
| Extraction/disassembly errors can be suppressed | Empty/partial scans can appear successful | Collection errors, omitted entries and undecoded bytes block the gate |
| Same register capacity used for Wave32/Wave64 | Wave64 occupancy can be overstated | Explicit gfx1201 profile: 1536/24 and 768/12 respectively |
| Counts/hints are mistaken for semantic tests | Inventory alone cannot prove waits/layout/lifetime | Separate exact checks, bounded diagnostics and unproved domains |
| Only five unit tests | Missing negative and integration coverage | Binary, parser, rule, CLI, decoder and C++ capture tests |
| Packaged Python bytecode caches | Stale artifacts may pollute installations | Source-only overlay plus documented generated binary fixture |

The original five unit tests were actually run and passed during review. That
result did not validate the prototype's omitted extraction/coverage paths.

## Corrections to earlier audit framing

### W4A8 tail scale expression

`libr4d/r4d_gemm_w4a8_prefill.hip` contains the scale expression before the final
store guard. The minimal guard relocation remains a reasonable separate source
fix. The `inside` fast path and the launcher's N alignment must be preserved.

No optimized GPU load, memory fault, or GPU wedge was reproduced. This testsuite
must not label the source expression a demonstrated wedge root cause. It does
not automatically edit that kernel.

### Occupancy

The 24-VGPR physical allocation step is correct for the 1536-VGPR Wave32 device
configuration (§3.3.2.1). It is not a universal rule for every architecture or
wave mode. Descriptor **encoding** granularity is different from physical
allocation. v0.2 cross-checks descriptors and metadata and reports a register-
limited upper bound, not achieved occupancy or a mandatory tuning threshold.

### Waterfalls and scratch

The QSA local array accesses are candidates until compiled ISA is inspected.
A relative VGPR move is legal and does not itself establish a waterfall. A
waterfall candidate in v0.2 needs an actual EXEC backedge enclosing lane selection
and relative access. It remains a performance observation, not a hang diagnosis.

The source comment mentioning 20 bytes of GQA6 prefill spill is historical
context, not a measurement of the supplied build. Nonzero private-segment bytes
can also represent local arrays/stack, not exclusively register spills. The new
report calls this field scratch/private bytes, not measured spill traffic.

### VOPD

The source rules omitted the identical-register SRC1 exception and opcode-specific
port routing. The supplied guide explicitly includes the equal-register exception,
FMAMK SRC2 routing, and MOV/MOV Y-source routing (§7.8).

In raw VOPD encoding, Y's destination low bit is derived as the opposite of X's;
there are not two independent destination parity fields. v0.2 decodes that fact
rather than claiming to find an impossible same-parity raw encoding.

An illegal VOPD is a correctness issue. No illegal VOPD in the actual Radiance
build was demonstrated here because that build's device objects are absent.
The suite's deliberately illegal fixture is not a Radiance runtime finding.

### Cache hints, ALU guards, deallocation

NT is not a blanket bypass guarantee. LU's dirty-discard contract requires
whole-buffer lifetime reasoning. No automatic hint rewriting is performed.
The `s_wait_alu` conditions include the earlier VALU read of an SGPR pair; a
blanket SALU-to-VALU wait rule is wrong. The suite checks only a documented narrow
pattern and labels it for review, not as full scoreboard validation.

Missing optional `MSG_DEALLOC_VGPRS` is not a failure. Reachable vector/memory
use after that release is materially different and is checked. Dynamic VGPR
mode is not certified; it is explicitly reported as incomplete.

### WMMA

Correct matrix element pairing and floating-point bit parity are different
properties. A static opcode list cannot prove either one. The new implementation
adds the documented D-to-A/B/index scheduling check (§7.12.1), but it does not
claim to prove semantic K ordering or sparse selector mapping.

## Integration review: actual Radiance loading path

`core/device/aql.cpp` and `core/device/aql_code.cpp` show that this snapshot uses
its own HSA/AQL loading/resolution path, with raw clang bundles. The build disables
offload compression for this path. A generic HIP-JIT interception recipe alone
would not describe this implementation correctly.

The optional patch captures after successful AQL executable loading and before
publishing a newly resolved `KernelRec`. It does not modify kernel arithmetic,
queue packets, memory scopes, wait instructions or synchronization protocols.
Capture is disabled without `RAD_ISA_CAPTURE_DIR`. Failure while enabled returns
an explicit loader/resolver error instead of allowing an unrecorded resolution.

The patch was checked against the uploaded source with `git apply --check`.
Its standalone host header was compiled and tested. **The complete Radiance
HIP/HSA build and a real GPU workload were not run.** Keep this integration as a
separate opt-in commit until those tests are supplied.

## Review of the new implementation

Additional fixes during this stage included:

* Fresh output directories and explicit coverage counts, including detection of
  unlisted object files and manifest omissions.
* Relative-path confinement and duplicate JSON key rejection for evidence input.
* File symlinks are resolved; symlink directories produce an explicit collection
  error rather than silently hiding another artifact tree.
* Byte hashes are checked both after collection and against capture-time hashes.
* Capture uses exclusive creation, restrictive file permissions, serialized writes,
  and a pre-lock fork check to avoid an inherited locked mutex.
* The LLVM constant-bus cross-check caught the SGPR_NULL exception; it is
  excluded from the bus count and has its own regression test.
* Dynamic/indirect execution paths and opaque containers do not get a clean flag.
* Review findings are blocking by default. Optional exploratory mode is explicit.

These safeguards reduce particular error modes; they are not a security audit of
LLVM or a mathematical proof of the suite itself.

## Recommended PR split

1. Standalone Python binary inventory/decoder/rules and tests.
2. Documentation, rule coverage and reproducible CPU evidence.
3. Optional AQL capture hook, separately enabled and validated on the target host.
4. Any W4A8 source fix, hint tuning, occupancy tuning or P2P change in **separate**
   PRs with their own numerical/runtime evidence.

Do not bundle speculative kernel rewrites with the validator. A detected review
candidate is a request for evidence, not authorization to insert arbitrary waits,
change fragment layouts or disable watchdogs.
