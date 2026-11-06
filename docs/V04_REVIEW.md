# v0.4 implementation review

Date: 2026-10-27. Basis: the attached v0.3 overlay, its review and package evidence,
the supplied host blind-spot reproducer, and `radiance-main(1).zip`. This stage does
not re-audit all 707 ISA-guide pages. `radisa/` and its static tests are retained
byte-identically; the changed domain is replay diagnostics and acceptance.

## Implemented, not merely proposed

* Three operation-serialized diagnostic passes with two complementary pure-output
  patterns, readable-alias protection, explicit scratch initialization contracts,
  operation-level forbidden-write comparisons and intermediate expected outputs.
* A separately instrumented host worker/plugin mode with ASan user poisoning of
  allocation guards and per-operation inaccessible byte ranges. Lower boundaries
  that ASan shadow granularity cannot express are refused. Normal guard checking
  still cannot see an inconsequential read.
* A genuine-source native declared-row inventory, versioned numeric-policy checks,
  scoped corpus qualification and stop-on-first-nonpass batch execution.
* A deterministic 46-case finite-f32 GEMM generator and independent Python `fsum`
  oracle. Boundary axes include values adjacent to 16/32/64/96/128; the source
  `ref_gemm_nt` permits those host shapes. They are **not asserted valid for every
  GPU kernel**.
* ABBA paired benchmarking with hash-bound, otherwise identical fixtures, every
  attempt retained, environment/mode checks and block-level exploratory intervals.
* Optional ordinary-AQL-dispatch observation with an exact static whitelist checked
  before reserving a queue packet. The host collector and evidence binding are
  tested; the actual HIP/HSA integration remains target-host work.

## Review corrections made while implementing

1. A scratch contract must not be inferred from the runner's old zero fill. Legacy
   snapshots with scratch but no contract become incomplete in default diagnostics.
2. Output-only perturbations subtract the union of all readable aliases. In-place
   data cannot be destroyed just to test a missing store.
3. Copy/fill write permissions cover actual transferred bytes, not an entire larger
   destination view. Kernel permissions remain the explicitly declared view spans.
4. A final permitted write does not excuse an earlier operation modifying its own
   read-only inputs. Checkpoints enforce the current operation, not a union of the
   entire window's writes.
5. Explicit numeric checkpoints use their declared tolerance; byte-hashing the same
   checked ranges would incorrectly override acceptable numeric variation. Remaining
   writes use exact same-candidate baseline equality, not an independent reference.
6. Finite L2 comparison uses wide accumulation after explicit nonfinite checks and
   preserves kBench's zero-reference convention.
7. An instrumented worker cannot emit benchmark samples, even when a caller omits
   the memory-check flag. Timed runs and serial diagnostics are distinguished.
8. Missing/old worker diagnostic fields cannot produce a new scoped contract pass.
9. Inventory comes from the actual declared plugin table, not only successful fixture
   results. Binary hashes are checked around native loading/enumeration.
10. AQL observation occurs before queue reservation; refusal cannot leave an unfilled
    reserved packet. Unknown identities, trace capacity and unscoped background
    submission threads fail closed. Actual successful fixture completion is still
    required before a trace is called complete.
11. Validated manifest hashes now identify the exact parsed bytes; the supervisor
    rereads and verifies that same version before preparing its execution plan.
12. The standalone host build includes the real `libref` sources without inventing
    the absent `core/kvstream/bind.h` or pretending the whole engine was compiled.

## Executed evidence

The final standalone host verification command was executed end to end in a fresh
work directory: normal worker/capture/probe/inventory/libref build, static and
Python tests, native tests, a separate clang ASan build, eight ASan tests, 46 real
host GEMM fixture runs, scoped qualification and independent row coverage.

Current test groups: **162 static/decoder/tool tests + 142 replay/format/protocol
and collector tests + 43 native host tests + 8 host-ASan tests = 355 tests**, with
zero failures and zero skips in the recorded final run. Test count is not GPU
kernel count. See `evidence/RELEASE_VALIDATION.json` and the logs.

Additional retained executions:

* Eight host counterexample/control runs in `evidence/mutations/`; the four fault
  cases fail in the appropriate diagnostic modes. The uninstrumented overread
  deliberately remains a runtime pass with reads NOT_CHECKED, demonstrating the
  unresolved plain-run observation boundary.
* A functional A/A comparison: 3 ABBA blocks, 12 real host worker executions with
  the same binary/data. This validates the measurement workflow, not a speedup or
  stable production latency. Concurrent authoring/build activity was not controlled.
* Native inventory of the actual supplied-source `libref`: **210 declared rows**;
  the 46-case corpus tests **one** of them. Row coverage correctly stays INCOMPLETE.
* A 32-operation in-place host sequence with independent intermediate checkpoints
  is part of the native tests. Hidden engine state is not inferred from it.

The optional AQL patches were checked for sequential application to the supplied
source. The kBench patch is still separate. No complete HIP/HSA executable or real
R9700 workload was built/run here. No intentionally invalid GPU code was executed.

## Remaining gaps are not renamed as fixes

GPU read instrumentation for this gfx1201/AQL route is not implemented/certified.
TP2/P2P, opaque instances, device chains and raw-tape replay are not implemented.
There is no automatic production activation corpus, independent oracle for every
operation, or real engine-window performance calibration. Corresponding strict
qualification requirements remain missing. See `V04_GAP_CLOSURE.md`.

Host ASan cannot prove GPU accesses. Per-operation snapshots cannot detect every
transient write-and-restore or establish race freedom. Perturbation patterns are
counterexample detectors, not exhaustive proof of initialization or overwrite.
Native code is trusted; evidence JSONs are not cryptographic remote attestations.
A GPU timeout cannot be repaired merely by killing its host process.

## External implementation references

* LLVM AddressSanitizer documentation: compiler instrumentation and runtime,
  build/link flags, first-error behavior: https://clang.llvm.org/docs/AddressSanitizer.html
* LLVM sanitizer interface: poisoning/unpoisoning and representability limitations:
  https://github.com/llvm/llvm-project/blob/main/compiler-rt/include/sanitizer/asan_interface.h
* AMD ROCm GPU sanitizer documentation: device/toolchain requirements and limits:
  https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html

These external sources informed the host instrumentation implementation. They do
not supply a successful sanitizer run on the user's GPU.
