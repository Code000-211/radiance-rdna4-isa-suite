# v0.3 implementation review

Date: 2026-10-27. Basis: the supplied Radiance source ZIP, supplied v0.2 overlay and
review documents, and the user's production-like standalone-kernel testing goal.
The ISA rules/decoder are unchanged in this stage; this review does not reassert
a complete instruction-by-instruction audit of the AMD guide.

## Existing implementation, not assumptions

The supplied `tools/rad_kbench.cpp`, `kfixture.cpp`, `kbench_core.cpp`,
`kbench_timer.cpp`, `docs/TOOLS.md` and `core/device/aql.cpp` distinguish seed-based
fixtures, real model weight bytes, different correctness/timing inputs, cache
rotation and the engine's internal tape. The tape contains process-local state;
serializing its structs would not produce portable replay. The v0.2 capture hook
records code loads/resolutions, not full dispatch operand snapshots.

The implementation therefore uses existing Radiance plugin launchers and device
interfaces behind a portable declared-memory contract. It deliberately does NOT
claim that the internal raw tape has been exported or that arbitrary engine
subgraphs have been made portable.

## Important design decisions and fixes during implementation

1. Allocation IDs and byte offsets preserve aliases. Only explicit relocation
   slots are patched; raw 64-bit values are never scanned for possible pointers.
2. Every written view must be covered fully by checks. Checking just one byte of
   an allocation cannot certify all its outputs. Undeclared changed bytes inside
   mutable allocations, immutable inputs and external/scratch guards are checked.
3. Every warm-up and timed iteration uses the same initial bytes and is checked;
   there is no unchecked "more realistic timing" input distribution.
4. Schema parameters and row-local tuning axes are validated separately. Calling
   the op-schema validator directly on a launcher's appended tuning choices would
   reject valid rows. A native regression test covers this distinction.
5. Exact rows and frozen tunable choices are required. Collective/link protocols,
   lifecycle-managed instances and unknown parameters cannot silently fall back.
6. Numeric finiteness is checked explicitly before comparison. NaN is not treated
   as equal to infinity merely because both are nonfinite. Captured baseline and
   independently produced expected outputs retain different provenance labels.
7. A typed execution plan removes JSON parsing, row resolution and allocation
   from the measured submission loop. Gated and host-paced measurements have
   distinct labels; neither is called exact production tape replay.
8. An ISA report is bound to its manifest, object bytes and exact plugin inputs.
   Plugin membership alone is insufficient: its actual embedded gfx1201 objects
   must be included. This still does not prove launcher-to-dispatch mapping or
   automatically certify runtime helper kernels.
9. The supplied ABI exposes no current-device query. The optional automatic GPU
   capture refuses heterogeneous visible device architectures rather than
   guessing the device from pointer values; use a visible gfx1201-only set.
10. Capture is opt-in. Missing WTAB target inventories, case caps and memory caps
    are logged as NOT_CAPTURED, not coverage success. Reference-cache shortcuts
    are disabled during snapshot capture so a case cannot be "captured" without
    an actual invocation.
11. CPU supervision separates timeout/error evidence from comparison failures.
    It does not reset a GPU or claim recovery after SIGKILL. Native code remains
    explicitly trusted, not sandboxed.
12. Corpus approval is explicit and hash-pinned. Missing, duplicated, unexpected
    and stale results block it. It never infers all-template/all-kernel coverage
    from a smaller corpus.

## Native verification boundary

The host worker is compiled from the actual supplied registry, device dispatcher,
host backend, encoding and kBench comparison sources. The real `libref` target
was compiled from the same source. A true libref add → multiply chain, in-place
aliasing, restore on every iteration, exact captured libref pre/post bytes and
subsequent portable replay were exercised. Additional deliberately faulty **host**
probe rows exercise guards, immutable writes, scratch handling, NaN/infinity,
stateful-row refusal, missing/illegal tunables and timeout isolation.

The optional kBench patch was checked for application to the uploaded source;
its modified translation unit was syntax-checked with and without the option.
The standalone capture component was compiled and run. The entire modified
kBench binary was NOT built/run end-to-end: the supplied source lacks
`core/kvstream/bind.h`, and the original full `rad_core` build fails in
`core/runtime/ctx.cpp` at that include. No stub was invented to hide the problem.

No HIP compiler or AMD GPU is present here. AQL queue execution, R9700 timing,
real QSA/MoE/GEMM snapshots and TP2 stability are NOT validated by these host tests.
The full GPU build, disabled-path smoke test, capture enabled test and numerical
regression suite on the target host remain explicit merge requirements.

See `evidence/RELEASE_VALIDATION.json` for current executed test counts and logs;
`evidence/v02/` is archived prior-release evidence, not new testing.

## Remaining gaps relative to the production-replay goal

- No automatic inference-engine capture or dependency-closed raw-tape export.
- No TP2/P2P collectives, opaque instances, device chains or hidden state.
- Automatic kBench WTAB export is refused; manual explicit relocation is supported.
- No independently generated references for every Radiance operation.
- No real production fixture corpus was supplied or fabricated.
- No instruction-level dispatch equivalence, hardware profiling or engine speed
  prediction is proved.

The next target-host integration should start with one supported single-GPU row
and an explicit fixture, then short ordered windows, before adding special
protocols. Do not combine speculative kernel fixes or watchdog changes with this
replay infrastructure PR.
